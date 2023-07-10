import torch
import torch.nn as nn
import torch.nn.functional as F
from ltr import model_constructor
from ltr.models.neck.AdjustLayer import xcorr_pixelwise
from ltr.models.backbone.repVGG import create_RepVGG_A0
from ltr.models.head.yolohead import Yoloneck, RepConv, IDetect
from ltr.models.neck.AdjustLayer import adjust_z_cls, adjust_x_cls

class Siamyolo(nn.Module):
    def __init__(self, backbone, neck_z, neck_x, adjust_weight, head):
        super(Siamyolo, self).__init__()
        self.backbone = backbone
        self.neck_z = neck_z
        self.neck_x = neck_x
        self.adjust_weight=adjust_weight
        self.head = head

    def forward(self, template, search):
        tem = self.backbone(template)  #layer3 1024
        src = self.backbone(search)
        tem_neck=self.neck_z(tem['x3'])
        src_neck=self.neck_x(src['x3'])
        result=self.head(self.adjust_weight(xcorr_pixelwise(src_neck,tem_neck)[0]))
        return result

    def template(self, template):
        tem = self.backbone(template)
        tem_neck = self.neck_z(tem['x3'])
        self.tem_cls = tem_neck

    def track(self, search):
        src = self.backbone(search)
        src_neck = self.neck_x(src['x3'])
        response=xcorr_pixelwise(src_neck,self.tem_cls)[0]
        result=self.head(self.adjust_weight(response))
        return {'result': result}

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
def siamyolo_repvgg(neck_in_channels=1280,neck_mid_channel=512,neck_out_channels=256):
    backbone = create_RepVGG_A0(deploy=True)
    neck_z = Yoloneck(in_channel=neck_in_channels, mid_channel=neck_mid_channel,out_channel=neck_out_channels) #(b,256,8,8)
    neck_x = Yoloneck(in_channel=neck_in_channels, mid_channel=neck_mid_channel,out_channel=neck_out_channels) #(b,256,20,20)
    adjust_weight = RepConv(64, 256)
    head=IDetect(nc=1, channel=256)
    net = Siamyolo(backbone=backbone,
                  neck_z=neck_z,
                  neck_x=neck_x,
                  adjust_weight=adjust_weight,
                  head=head)
    return net

@model_constructor
def siamyolov1_repvgg(neck_in_channel=192,neck_out_channels=256):
    backbone = create_RepVGG_A0(deploy=True)
    neck_z = adjust_z_cls(in_channels=neck_in_channel, out_channels=neck_out_channels) #(b,256,8,8)
    neck_x = adjust_x_cls(in_channels=neck_in_channel, out_channels=neck_out_channels) #(b,256,20,20)
    adjust_weight = Adjust_weight(in_channels=64, out_channels=neck_out_channels)
    head=IDetect(nc=1, channel=256)
    net = Siamyolo(backbone=backbone,
                  neck_z=neck_z,
                  neck_x=neck_x,
                  adjust_weight=adjust_weight,
                  head=head)
    return net