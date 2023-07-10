import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from ltr import model_constructor
from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.neck.featurefusion import build_feature_fusion
from util.misc import (NestedTensor, nested_tensor_from_tensor, box_xyxy_to_cxcywh, inverse_sigmoid)
from ltr.models.head.head import build_box_head


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransConv(nn.Module):
    def __init__(self, backbone, featurefusion, box_head, num_queries,
                 with_bbox_refine=False, head_type='corner', feat_sz=16):
        super().__init__()
        hidden_dim = featurefusion.d_model
        self.num_queries = num_queries
        self.backbone = backbone
        self.featurefusion = featurefusion
        self.box_head = box_head
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        #we use one feature
        self.input_proj_search = nn.ModuleList([nn.Sequential(
            nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim)
        )])
        self.input_proj_tem = nn.ModuleList([nn.Sequential(
            nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim)
        )])
        self.with_box_refine = with_bbox_refine

        if head_type == 'corner_predictor':
            self.feat_s = int(feat_sz)
            self.feat_len_s = int(feat_sz ** 2)

        for proj in self.input_proj_search:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        for proj in self.input_proj_tem:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        self.head_type = head_type

    def forward(self, search, template):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor(template)
        feat_search, pos_search = self.backbone(search) #list
        feat_tem, pos_tem = self.backbone(template) #list

        src_search, mask_search = feat_search[-1].decompose()
        src_tem, mask_tem = feat_tem[-1].decompose()
        tensor_list = [self.input_proj_tem[0](src_tem), self.input_proj_search[0](src_search)]
        mask_list = [mask_tem, mask_search]
        pos_list = [pos_tem[-1], pos_search[-1]]

        query_embeds = self.query_embed.weight
        hs, memory, init_reference, inter_reference = self.featurefusion(tensor_list, mask_list, pos_list, query_embeds)
        #hs(b,h_tem*w_tem+h_search*w_search,c) memory(b,1,c)
        if self.head_type =='corner_predictor':
            enc_opt = memory[:,-self.feat_len_s:,:] #b,hw,c
            dec_opt = hs.transpose(1,2) #b,c,n
            att = torch.matmul(enc_opt, dec_opt) #b,hw,n
            opt = (enc_opt.unsqueeze(-1)*att.unsqueeze(-2)).permute((0,3,2,1)).contiguous() #b,hw,c,n->b,n,c,hw
            bs, nq, c, hw = opt.size()
            opt_feat = opt.view(-1, c, self.feat_s, self.feat_s)
            coorxy = self.box_head(opt_feat)
            outputs_coord = box_xyxy_to_cxcywh(coorxy)
            outputs_coord_out = outputs_coord.view(bs,nq,4)
            out = {'pred_boxes':outputs_coord_out}
            return out, outputs_coord_out
        elif self.head_type == 'MLP':
            reference = inverse_sigmoid(inter_reference)
            tmp = self.box_head(hs)
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[...,:2] += reference
            outputs_coord_out = tmp.sigmoid()
            out = {'pred_boxes': outputs_coord_out}

            return out, outputs_coord_out

    def template(self, template):
        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor(template)
        feat_tem, self.pos_tem = self.backbone(template)  # list
        src_tem, self.mask_tem = feat_tem[-1].decompose()
        self.tensor_tem = self.input_proj_tem[0](src_tem)


    def track(self, search):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        feat_search, pos_search = self.backbone(search)  # list
        src_search, mask_search = feat_search[-1].decompose()
        tensor_list = [self.tensor_tem, self.input_proj_search[0](src_search)]
        mask_list = [self.mask_tem, mask_search]
        pos_list = [self.pos_tem[-1], pos_search[-1]]
        query_embeds = self.query_embed.weight
        hs, memory, init_reference, inter_reference = self.featurefusion(tensor_list, mask_list, pos_list, query_embeds)
        if self.head_type =='corner_predictor':
            enc_opt = memory[:,-self.feat_len_s:,:] #b,hw,c
            dec_opt = hs.transpose(1,2) #b,c,n
            att = torch.matmul(enc_opt, dec_opt) #b,hw,n
            opt = (enc_opt.unsqueeze(-1)*att.unsqueeze(-2)).permute((0,3,2,1)).contiguous() #b,hw,c,n->b,n,c,hw
            bs, nq, c, hw = opt.size()
            opt_feat = opt.view(-1, c, self.feat_s, self.feat_s)
            coorxy = self.box_head(opt_feat)
            outputs_coord = box_xyxy_to_cxcywh(coorxy)
            outputs_coord_out = outputs_coord.view(bs,nq,4)
            out = {'pred_boxes':outputs_coord_out}
            return out, outputs_coord_out
        elif self.head_type == 'MLP':
            reference = inverse_sigmoid(inter_reference)
            tmp = self.box_head(hs)
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[...,:2] += reference
            outputs_coord_out = tmp.sigmoid()
            out = {'pred_boxes': outputs_coord_out}
            return out, outputs_coord_out

@model_constructor
def transtrack_resnet50(settings):
    backbone_net = build_backbone(settings, backbone_pretrained=True)
    featurefusion = build_feature_fusion(settings)
    box_head = build_box_head(settings)
    model = TransConv(
        backbone_net,
        featurefusion,
        box_head,
        num_queries = settings.num_queris,
        with_bbox_refine = settings.with_bbox_refine,
        head_type = settings.head_type
    )
    return model