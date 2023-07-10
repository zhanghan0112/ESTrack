import copy
from typing import Optional
import torch

import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_
import torch
from timm.models.layers import DropPath
import warnings


class FeatureFusionNetwork(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_featurefusion_layers=4,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        featurefusion_layer = FeatureFusionLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = Encoder(featurefusion_layer, num_featurefusion_layers)

        decoderCFA_layer = DecoderCFALayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoderCFA_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoderCFA_layer, decoderCFA_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_temp, mask_temp, src_search, mask_search, pos_temp, pos_search):
        b, h, w = src_search.shape[0], src_search.shape[2], src_search.shape[3]
        src_temp = src_temp.flatten(2).permute(2, 0, 1)
        pos_temp = pos_temp.flatten(2).permute(2, 0, 1)
        src_search = src_search.flatten(2).permute(2, 0, 1)
        pos_search = pos_search.flatten(2).permute(2, 0, 1)
        mask_temp = mask_temp.flatten(1)
        mask_search = mask_search.flatten(1)

        memory_temp, memory_search = self.encoder(src1=src_temp, src2=src_search,
                                                  src1_key_padding_mask=mask_temp,
                                                  src2_key_padding_mask=mask_search,
                                                  pos_src1=pos_temp,
                                                  pos_src2=pos_search)
        hs = self.decoder(memory_search, memory_temp,
                          tgt_key_padding_mask=mask_search,
                          memory_key_padding_mask=mask_temp,
                          pos_enc=pos_temp, pos_dec=pos_search)
        return hs.transpose(0, 1).transpose(1,2).reshape(b, self.d_model, h, w)


class Decoder(nn.Module):

    def __init__(self, decoderCFA_layer, norm=None):
        super().__init__()
        self.layers = _get_clones(decoderCFA_layer, 1)
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos_enc=pos_enc, pos_dec=pos_dec)

        if self.norm is not None:
            output = self.norm(output)

        return output

class Encoder(nn.Module):

    def __init__(self, featurefusion_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(featurefusion_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):
        output1 = src1
        output2 = src2

        for layer in self.layers:
            output1, output2 = layer(output1, output2, src1_mask=src1_mask,
                                     src2_mask=src2_mask,
                                     src1_key_padding_mask=src1_key_padding_mask,
                                     src2_key_padding_mask=src2_key_padding_mask,
                                     pos_src1=pos_src1, pos_src2=pos_src2)

        return output1, output2


class DecoderCFALayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos_enc: Optional[Tensor] = None,
                     pos_dec: Optional[Tensor] = None):

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos_dec),
                                   key=self.with_pos_embed(memory, pos_enc),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):

        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos_enc, pos_dec)

class FeatureFusionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model)

        self.linear21 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear22 = nn.Linear(dim_feedforward, d_model)

        self.norm11 = nn.LayerNorm(d_model)
        self.norm12 = nn.LayerNorm(d_model)
        self.norm13 = nn.LayerNorm(d_model)
        self.norm21 = nn.LayerNorm(d_model)
        self.norm22 = nn.LayerNorm(d_model)
        self.norm23 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)
        self.dropout23 = nn.Dropout(dropout)

        self.activation1 = _get_activation_fn(activation)
        self.activation2 = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1, src2,
                     src1_mask: Optional[Tensor] = None,
                     src2_mask: Optional[Tensor] = None,
                     src1_key_padding_mask: Optional[Tensor] = None,
                     src2_key_padding_mask: Optional[Tensor] = None,
                     pos_src1: Optional[Tensor] = None,
                     pos_src2: Optional[Tensor] = None):


        src12 = self.multihead_attn1(query=self.with_pos_embed(src1, pos_src1),
                                   key=self.with_pos_embed(src2, pos_src2),
                                   value=src2, attn_mask=src2_mask,
                                   key_padding_mask=src2_key_padding_mask)[0]

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = self.norm13(src1)

        src22 = self.multihead_attn2(query=self.with_pos_embed(src2, pos_src2),
                                     key=self.with_pos_embed(src1, pos_src1),
                                     value=src1, attn_mask=src1_mask,
                                     key_padding_mask=src1_key_padding_mask)[0]

        src2 = src2 + self.dropout22(src22)
        src2 = self.norm22(src2)
        src22 = self.linear22(self.dropout2(self.activation2(self.linear21(src2))))
        src2 = src2 + self.dropout23(src22)
        src2 = self.norm23(src2)

        return src1, src2

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):

        return self.forward_post(src1, src2, src1_mask, src2_mask,
                                 src1_key_padding_mask, src2_key_padding_mask, pos_src1, pos_src2)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_featurefusion_network(settings):
    return FeatureFusionNetwork(
        d_model=settings.hidden_dim,
        dropout=settings.dropout,
        nhead=settings.nheads,
        dim_feedforward=settings.dim_feedforward,
        num_featurefusion_layers=settings.featurefusion_layers
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, proj_drop=0, bias=True, kdim=None, vdim=None):
        super(Attention,self).__init__()
        self.dim = dim
        self.kdim = kdim if kdim is not None else dim
        self.vdim = vdim if vdim is not None else dim
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(dropout)
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}"
        self.q = nn.Linear(dim,dim,bias=bias)
        self.k = nn.Linear(dim,dim,bias=bias)
        self.v = nn.Linear(dim,dim,bias=bias)
        self.scale = float(self.head_dim) ** -0.5
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                constant_(m.bias, 0.)

    def forward(self,q:Tensor,k:Tensor,v:Tensor,key_padding_mask: Optional[Tensor] = None):
        tgt_len, bsz, embed_dim = q.size()
        q = q * self.scale
        q = self.q(q).contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k(k).contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v(v).contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        attn = torch.bmm(q,k.transpose(1,2)) #(bsxn_head,q_len,k_len)
        src_len = k.size(1)
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
            attn = attn.view(bsz, self.num_heads, tgt_len, src_len)
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )  #1的位置填充-inf,0的位置保持原状
            attn = attn.view(bsz * self.num_heads, tgt_len, src_len)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = torch.bmm(attn, v).transpose(0, 1).contiguous().view(tgt_len,bsz,embed_dim)
        attn = self.proj(attn)
        attn = self.proj_drop(attn)
        return attn

class TransformerEncoderensem1(nn.Module):
    def __init__(self,dim,num_heads,dropout=0.1,norm_layer=nn.LayerNorm,drop_path=0.1,mlp_ratio=4,
                 act_layer=nn.GELU):
        super().__init__()
        self.attn = Attention(dim,num_heads=num_heads,dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp_ratio = mlp_ratio
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=dropout)
    def forward(self, q: Tensor, k: Tensor, v: Tensor, key_padding_mask: Optional[Tensor] = None):
        b, c, h, w = q.shape[0],q.shape[1],q.shape[2],q.shape[3]
        q = q.flatten(2).permute(2,0,1)
        k = k.flatten(2).permute(2,0,1)
        v = v.flatten(2).permute(2,0,1)

        attn = self.attn(q,k,v,key_padding_mask)
        src = q + self.drop_path(attn)
        src = self.norm1(src)

        src2 = self.mlp(src)
        src = src + src2
        src = self.norm2(src)
        src = src.permute(1,2,0).reshape(b,c,h,w)
        return src

class TransformerEncoderensem2(nn.Module):
    def __init__(self,dim,num_heads,dropout=0.1,norm_layer=nn.LayerNorm,drop_path=0.1,mlp_ratio=4,
                 act_layer=nn.GELU):
        super().__init__()
        self.attn = Attention(dim,num_heads=num_heads,dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp_ratio = mlp_ratio
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=dropout)
    def forward(self, q: Tensor, k: Tensor, v: Tensor, key_padding_mask: Optional[Tensor] = None):
        b, c, h, w = q.shape[0],q.shape[1],q.shape[2],q.shape[3]
        q = q.flatten(2).permute(2,0,1)
        k = k.flatten(2).permute(2,0,1)
        v = v.flatten(2).permute(2,0,1)

        attn = self.attn(q,k,v,key_padding_mask)
        src = q + self.drop_path(attn)
        src = self.norm1(src)

        src2 = self.mlp(src)
        src = src + src2
        src = self.norm2(src)
        src = src.permute(1,2,0).reshape(b,c,h,w)
        return src