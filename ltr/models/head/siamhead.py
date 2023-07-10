import torch
import torch.nn as nn
import math
import numpy as np

class ESTrackHead(nn.Module):
    def __init__(self, num_convs=4, in_channels=256, num_classes=1, prior_prob=0.01, stride=16, instance_size=320,
                 template_size=128,head_name=None):
        super(ESTrackHead, self).__init__()
        cls_tower = []
        bbox_tower = []
        self.total_stride = stride
        self.instance_size = instance_size
        self.template_size = template_size
        if head_name == "ESTrack":
            self.score_size = self.instance_size // self.total_stride
        if head_name == "siamCAR":
            self.score_size = (self.instance_size - self.template_size) // self.total_stride + 1
        self.offset = (self.instance_size - 1 - (self.score_size - 1) * self.total_stride) // 2
        for l in range(num_convs):
            cls_tower.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=1, stride=1, padding=0)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.bi = torch.nn.Parameter(torch.tensor(0.).type(torch.Tensor))
        self.si = torch.nn.Parameter(torch.tensor(1.).type(torch.Tensor))


        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits, self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.normal_(l.weight, std=0.01)
                    nn.init.constant_(l.bias, 0)
        prior_prob = prior_prob
        bias_value = -math.log((1-prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, cls, loc):
        cls_tower = self.cls_tower(cls)
        bbox_tower = self.bbox_tower(loc)

        cls_score = self.cls_logits(cls_tower)
        cls_score = cls_score.permute(0, 2, 3, 1)
        cls_score = cls_score.reshape(cls_score.shape[0], -1, 1)

        centerness = self.centerness(cls_tower)
        ctr_score = centerness.permute(0, 2, 3, 1)
        ctr_score = ctr_score.reshape(ctr_score.shape[0], -1, 1)

        offsets = self.bbox_pred(bbox_tower)
        offsets = torch.exp(self.si * offsets + self.bi) * self.total_stride

        fm_ctr = get_xy_ctr_np(self.score_size, self.offset, self.total_stride)

        bbox = get_bbox(fm_ctr.to(offsets.device), offsets).to(offsets.device)
        return cls_score, bbox, ctr_score

def get_bbox(xy_ctr, offsets):
    offsets = offsets.permute(0, 2, 3, 1) #(B, H, W, C), C=4
    offsets = offsets.reshape(offsets.shape[0], -1, 4)
    xy0 = (xy_ctr[:,:,:] - offsets[:,:,:2])
    xy1 = (xy_ctr[:,:,:] + offsets[:,:,2:])
    bboxes_pred = torch.cat([xy0, xy1], 2)
    return bboxes_pred

def get_xy_ctr_np(score_size, score_offset, total_stride):
    batch, fm_height, fm_width = 1, score_size, score_size
    y_list = np.linspace(0., fm_height-1., fm_height).reshape(1, fm_height, 1, 1)
    y_list = y_list.repeat(fm_width, axis=2)
    x_list = np.linspace(0., fm_width-1., fm_width).reshape(1, 1, fm_width, 1)
    x_list = x_list.repeat(fm_height, axis=1)
    xy_list = score_offset + np.concatenate((x_list, y_list), 3) * total_stride
    xy_ctr = np.repeat(xy_list, batch, axis=0).reshape(batch, -1, 2)
    xy_ctr = torch.from_numpy(xy_ctr.astype(np.float32))
    return xy_ctr

