import torch
import torch.nn as nn
import torch.nn.functional as F

from ltr import model_constructor
from ltr.models.backbone.resnet import resnet50, resnet18
from ltr.models.backbone.repVGG import create_RepVGG_A0
from ltr.models.head.siamhead import ESTrackHead
from ltr.models.neck.AdjustLayer import adjust_z_cls, adjust_z_loc, adjust_x_cls, adjust_x_loc
from ltr.models.neck.AdjustLayer import xcorr_pixelwise


class SiamCAR(nn.Module):
    def __init__(self, backbone, neck_z_cls, neck_z_loc, neck_x_cls, neck_x_loc, adjust_weight, head):
        super(SiamCAR, self).__init__()
        self.backbone = backbone
        self.neck_z_cls = neck_z_cls
        self.neck_z_loc = neck_z_loc
        self.neck_x_cls = neck_x_cls
        self.neck_x_loc = neck_x_loc
        self.adjust_weight = adjust_weight
        self.head = head

    def forward(self, template, search):
        src_tem = self.backbone(template)['layer3']
        src_search = self.backbone(search)['layer3']
        tem_cls_adjust = self.neck_z_cls(src_tem)
        tem_loc_adjust = self.neck_z_loc(src_tem)
        src_cls_adjust = self.neck_x_cls(src_search)
        src_loc_adjust = self.neck_x_loc(src_search)
        dist_cls,h1,w1 = xcorr_pixelwise(src_cls_adjust, tem_cls_adjust)
        dist_loc,h2,w2 = xcorr_pixelwise(src_loc_adjust, tem_loc_adjust)
        dw_cls_feat = self.adjust_weight(dist_cls.reshape(*dist_cls.shape[:2],h1,w1))
        dw_loc_feat = self.adjust_weight(dist_loc.reshape(*dist_loc.shape[:2],h2,w2))
        cls, bbox, center = self.head(dw_cls_feat, dw_loc_feat)
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
        dist_cls, h1, w1 = xcorr_pixelwise(src_cls_adjust, self.tem_cls)
        dist_loc, h2, w2 = xcorr_pixelwise(src_loc_adjust, self.tem_loc)
        dw_cls_feat = self.adjust_weight(dist_cls.reshape(*dist_cls.shape[:2], h1, w1))
        dw_loc_feat = self.adjust_weight(dist_loc.reshape(*dist_loc.shape[:2], h2, w2))
        cls, bbox, center = self.head(dw_cls_feat, dw_loc_feat)
        cls_prob_final = torch.sigmoid(cls)
        ctr_prob_final = torch.sigmoid(center)
        score_final = cls_prob_final * ctr_prob_final
        return {
            'cls': score_final,
            'loc': bbox,
            'cls_prob_final': cls_prob_final,
            'ctr_prob_final': ctr_prob_final,
            'dist_cls': cls_prob_final
        }

class Adjust_weight(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Adjust_weight, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.downsample(x)
        return x

@model_constructor
def siamcar_resnet50_ori(backbone_pretrained=False, neck_in_channels=1024,
                     neck_out_channels=256, head_name=None):
    backbone = resnet50(output_layers=('layer3',), pretrained=backbone_pretrained)
    neck_z_cls = adjust_z_cls(in_channels=neck_in_channels, out_channels=neck_out_channels)
    neck_x_cls = adjust_x_cls(in_channels=neck_in_channels, out_channels=neck_out_channels)
    head = ESTrackHead(num_convs=4, in_channels=256, num_classes=1, prior_prob=0.01, head_name=head_name)
    net=SiamCAR(backbone=backbone,
                neck_z_cls=neck_z_cls,
                neck_x_cls=neck_x_cls,
                head=head)
    return net

@model_constructor
def ESTrack_resnet50(backbone_pretrained=False, neck_in_channels=1024,weight=64,
                     neck_out_channels=256, head_name=None):
    backbone = resnet50(output_layers=('layer3',), pretrained=backbone_pretrained)
    neck_z_cls = adjust_z_cls(in_channels=neck_in_channels, out_channels=neck_out_channels)
    neck_z_loc = adjust_z_loc(in_channels=neck_in_channels, out_channels=neck_out_channels)
    neck_x_cls = adjust_x_cls(in_channels=neck_in_channels, out_channels=neck_out_channels)
    neck_x_loc = adjust_x_loc(in_channels=neck_in_channels, out_channels=neck_out_channels)
    adjust_weight = Adjust_weight(in_channels=weight, out_channels=neck_out_channels)
    head = ESTrackHead(num_convs=4, in_channels=256, num_classes=1, prior_prob=0.01, head_name=head_name)
    net=SiamCAR(backbone=backbone,
                neck_z_cls=neck_z_cls,
                neck_z_loc=neck_z_loc,
                neck_x_cls=neck_x_cls,
                neck_x_loc=neck_x_loc,
                adjust_weight = adjust_weight,
                head=head)
    return net

@model_constructor
def ESTrack_resnet18(backbone_pretrained=False, neck_in_channels=256, weight=64,
                     neck_out_channels=256, head_name=None):
    backbone = resnet18(output_layers=('layer3',), pretrained=backbone_pretrained)
    neck_z_cls = adjust_z_cls(in_channels=neck_in_channels, out_channels=neck_out_channels)
    neck_z_loc = adjust_z_loc(in_channels=neck_in_channels, out_channels=neck_out_channels)
    neck_x_cls = adjust_x_cls(in_channels=neck_in_channels, out_channels=neck_out_channels)
    neck_x_loc = adjust_x_loc(in_channels=neck_in_channels, out_channels=neck_out_channels)
    adjust_weight = Adjust_weight(in_channels=weight, out_channels=neck_out_channels)
    head = ESTrackHead(num_convs=4, in_channels=256, num_classes=1, prior_prob=0.01, head_name=head_name)
    net=SiamCAR(backbone=backbone,
                neck_z_cls=neck_z_cls,
                neck_z_loc=neck_z_loc,
                neck_x_cls=neck_x_cls,
                neck_x_loc=neck_x_loc,
                adjust_weight = adjust_weight,
                head=head)
    return net

@model_constructor
def ESTrack_repvgg(neck_in_channels=192, weight=64,neck_out_channels=256, head_name=None):
    backbone = create_RepVGG_A0(deploy=True)
    neck_z_cls = adjust_z_cls(in_channels=neck_in_channels, out_channels=neck_out_channels)
    neck_z_loc = adjust_z_loc(in_channels=neck_in_channels, out_channels=neck_out_channels)
    neck_x_cls = adjust_x_cls(in_channels=neck_in_channels, out_channels=neck_out_channels)
    neck_x_loc = adjust_x_loc(in_channels=neck_in_channels, out_channels=neck_out_channels)
    adjust_weight = Adjust_weight(in_channels=weight, out_channels=neck_out_channels)
    head = ESTrackHead(num_convs=4, in_channels=256, num_classes=1, prior_prob=0.01, head_name=head_name)
    net = SiamCAR(backbone=backbone,
                  neck_z_cls=neck_z_cls,
                  neck_z_loc=neck_z_loc,
                  neck_x_cls=neck_x_cls,
                  neck_x_loc=neck_x_loc,
                  adjust_weight=adjust_weight,
                  head=head)
    return net
if __name__ == '__main__':
    # from ltr.models.backbone.repVGG import repvgg_model_convert
    # model = siamcar_repvgg()
    ckpt_ori = torch.load('/home/zxh/project/TransT/checkpoints/ltr/siamcar/important_ckpt/siamcar_new_ddp_onlycls/SiamCAR_ep0200.pth.tar')
    print(ckpt_ori['net'].keys())
    # model.load_state_dict(ckpt_ori['net'])
    # deploy_model = repvgg_model_convert(model,save_path='/home/zxh/project/TransT/checkpoints/ltr/siamcar/siamcar_new_ddp/SiamCAR_new_ep0200.pth.tar')
    # ckpt = torch.load('/home/zxh/project/TransT/checkpoints/ltr/siamcar/siamcar_new_ddp/SiamCAR_new_ep0200.pth.tar')
    # ckpt_ori['net'] = ckpt
    # print(ckpt_ori['net'].keys())
    # print(ckpt.keys())
    from pysot_toolkit.trackers.net_wrappers import NetWithBackbone
    # from thop import profile, clever_format
    # model = siamcar_resnet18()
    # deploy_ckpt = torch.load('/home/zxh/project/TransT/checkpoints/ltr/siamcar/siamcar_new_ddp_resnet18/SiamCAR_ep0200.pth.tar',map_location='cpu')['net']

    # # for k in ckpt.keys():
    # #     if 'featfusion' not in k:
    # #         ckpt.append(k)
    #
    # model.load_state_dict(deploy_ckpt,strict=False)
    # model.cuda()
    # model.eval()
    # input = torch.randn(1,3,127,127).cuda()
    # search = torch.randn(1,3,320,320).cuda()
    # macs, params = profile(model, inputs=(input,search))
    # macs, params = clever_format([macs, params], "%.3f")
    # print(macs)
    # print(params)


