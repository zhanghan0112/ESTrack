import torch
import copy
from torch import nn
import torch.nn.functional as F
from ltr.models.ops.modules.ms_deform_attn import MSDeformAttn
from ltr.models.tracking.utils import inverse_sigmoid
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
class FeatureFusion(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1, activation='relu', return_intermediate_dec=False,
                 num_feature_levels=2, dec_n_points=4, enc_n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_head = nhead
        encoder_layer = EncoderLayer(d_model, dim_feedforward, dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = Encoder(encoder_layer, num_encoder_layers)
        decoder_layer = DecoderLayer(d_model, dim_feedforward, dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = Decoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        self.level_embed=nn.Parameter(torch.Tensor(num_feature_levels,d_model))
        self.reference_points=nn.Linear(d_model,2)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    # 找出mask中没有填充的区域
    def get_valid_ratio(self, mask):
        _, h, w = mask.shape
        valid_h = torch.sum(~mask[:,:,0],1)
        valid_w = torch.sum(~mask[:,0,:],1)
        valid_ratio_h = valid_h.float() / h
        valid_ratio_w = valid_w.float() / w
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h],1) #(b,2)
        return valid_ratio

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []  # 把有效区域坐标归一化到0-1，其他 >1
        for lvl, (h, w) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # (b,h*w+h1*w1+h2*w2,2)
        return reference_points

    def forward(self, srcs, masks, pos_embeds, query_embeds): #list
        spatial_shapes = []
        lvl_pos_embed_flatten = []
        src_flatten = []
        mask_flatten = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h,w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1,2) #b,hxw,c
            mask = mask.flatten(1) #b,hxw
            pos_embed = pos_embed.flatten(2).transpose(1,2) #b,hxw,c
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1,1,-1)
            src_flatten.append(src)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
        src_flatten = torch.cat(src_flatten, 1) #(b,h_tem*w_tem+h_search*w_search,c)
        mask_flatten = torch.cat(mask_flatten, 1) #(b,h_tem*w_tem+h_search*w_search)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)  # b,level,2
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)  # b,h_tem*w_tem+h_search*w_search,C
        bs,_,c = memory.shape
        query_embed, tgt = torch.split(query_embeds,c,dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs,-1,-1)
        tgt = tgt.unsqueeze(0).expand(bs,-1,-1)
        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points #b,1,2
        hs, inter_references = self.decoder(tgt, reference_points, memory, spatial_shapes,
                                            level_start_index, valid_ratios, query_embed, mask_flatten) #b,n_q,c
        return hs, memory, init_reference_out, inter_references

class EncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation='relu', n_levels=2,
                 n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src = self.forward_ffn(src)
        return src

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = [] #把有效区域坐标归一化到0-1，其他 >1
        for lvl, (h,w) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, h-0.5, h, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, w-0.5, w, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:,None,lvl,1]*h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:,None,lvl,0]*w)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1) #(b,h*w+h1*w1+h2*w2,2)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src #b,len,c
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation='relu', n_levels=4, n_heads=8,n_points=4):
        super().__init__()
        self.d_model = d_model
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model,n_heads,dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4=nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                src_padding_mask=None):
        q=k=self.with_pos_embed(tgt,query_pos)
        tgt2 = self.self_attn(q.transpose(0,1),k.transpose(0,1),tgt.transpose(0,1))[0].transpose(0,1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt) #b,n,c

        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt = self.forward_ffn(tgt)

        return tgt

class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = _get_clones(MLP(decoder_layer.d_model, decoder_layer.d_model, 4, 3), self.num_layers)
        self.class_embed=None
        # nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        # nn.initit.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:,:,None] * torch.cat([src_valid_ratios,src_valid_ratios],-1)[:,None]
            else:
                assert reference_points.shape[-1]==2
                reference_points_input = reference_points[:,:,None] * src_valid_ratios[:,None]
            output = layer(output,query_pos,reference_points_input,src,src_spatial_shapes,src_level_start_index,
                           src_padding_mask)
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1]==4:
                    new_reference_points=tmp+inverse_sigmoid(reference_points)
                    new_reference_points=new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[...,:2]=tmp[...,:2]+inverse_sigmoid(reference_points)
                    new_reference_points=new_reference_points.sigmoid()
                reference_points = new_reference_points
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers-1) #[hidden_dim,hidden_dim,hidden_dim...]
        self.layers = nn.ModuleList(nn.Linear(n,k) for n,k in zip([input_dim]+h, h+[output_dim]))

    def forward(self,x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers-1 else layer(x)
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def build_feature_fusion(settings):
    return FeatureFusion(
        d_model=settings.hidden_dim,
        nhead=settings.nheads,
        num_encoder_layers=settings.enc_layers,
        num_decoder_layers=settings.dec_layers,
        dim_feedforward=settings.dim_feedforward,
        dropout=settings.dropout,
        activation='relu',
        return_intermediate_dec=False,
        num_feature_levels=2,
        dec_n_points = 4,
        enc_n_points = 4)

