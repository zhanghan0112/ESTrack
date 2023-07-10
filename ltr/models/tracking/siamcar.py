import torch
import torch.nn as nn
import torch.nn.functional as F
from ltr import model_constructor
from ltr.models.backbone import resnet50
from ltr.models.neck.AdjustLayer import adjust_z_cls, adjust_z_loc, adjust_x_cls, adjust_x_loc, xcorr_depthwise
from ltr.models.head.siamhead import ESTrackHead

class SiamCAR(nn.Module):
    def __init__(self, backbone, neck_z_cls, neck_z_loc, neck_x_cls, neck_x_loc, head):
        super(SiamCAR, self).__init__()
        self.backbone = backbone
        self.neck_z_cls = neck_z_cls
        self.neck_z_loc = neck_z_loc
        self.neck_x_cls = neck_x_cls
        self.neck_x_loc = neck_x_loc
        self.head = head

    def log_softmax(self, cls):
        b, c, h, w = cls.size()
        cls = cls.view(b, 2, c//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, template, search):
        tem = self.backbone(template)['layer3']  #layer3 1024
        src = self.backbone(search)['layer3']
        tem_cls_adjust = self.neck_z_cls(tem)
        tem_loc_adjust = self.neck_z_loc(tem)
        src_cls_adjust = self.neck_x_cls(src)
        src_loc_adjust = self.neck_x_loc(src)
        dw_cls_feat = xcorr_depthwise(src_cls_adjust, tem_cls_adjust)
        dw_loc_feat = xcorr_depthwise(src_loc_adjust, tem_loc_adjust)
        cls, bbox, center = self.head(dw_cls_feat, dw_loc_feat)
        # cls = self.log_softmax(cls)
        return cls, bbox, center

    def template(self, template):
        tem = self.backbone(template)['layer3']
        tem_cls_adjust = self.neck_z_cls(tem)
        tem_loc_adjust = self.neck_z_loc(tem)
        self.tem_cls = tem_cls_adjust
        self.tem_loc = tem_loc_adjust

    def track(self, search):
        src = self.backbone(search)['layer3']
        src_cls_adjust = self.neck_x_cls(src)
        src_loc_adjust = self.neck_x_loc(src)
        dw_cls_feat = xcorr_depthwise(src_cls_adjust, self.tem_cls)
        dw_loc_feat = xcorr_depthwise(src_loc_adjust, self.tem_loc)
        cls, bbox, center, cls_feat = self.head(dw_cls_feat, dw_loc_feat)
        cls_prob_final = torch.sigmoid(cls)
        ctr_prob_final = torch.sigmoid(center)
        score_final = cls_prob_final * ctr_prob_final
        return {
            'cls': score_final,
            'loc': bbox,
            'cls_prob_final': cls_prob_final,
            'ctr_prob_final': ctr_prob_final
        }



@model_constructor
def siamcar_resnet50(backbone_output_layers=['layer3'], backbone_pretrained=False, neck_in_channels=1024,
                     neck_out_channels=256, head_name=None):
    backbone = resnet50(output_layers=backbone_output_layers, pretrained=backbone_pretrained)
    neck_z_cls = adjust_z_cls(in_channels=neck_in_channels, out_channels=neck_out_channels)
    neck_z_loc = adjust_z_loc(in_channels=neck_in_channels, out_channels=neck_out_channels)
    neck_x_cls = adjust_x_cls(in_channels=neck_in_channels, out_channels=neck_out_channels)
    neck_x_loc = adjust_x_loc(in_channels=neck_in_channels, out_channels=neck_out_channels)
    head = ESTrackHead(num_convs=4, in_channels=256, num_classes=1, prior_prob=0.01, head_name=head_name)
    net=SiamCAR(backbone=backbone,
                neck_z_cls=neck_z_cls,
                neck_z_loc=neck_z_loc,
                neck_x_cls=neck_x_cls,
                neck_x_loc=neck_x_loc,
                head=head)
    return net
