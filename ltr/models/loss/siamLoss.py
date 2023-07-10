import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

eps = np.finfo(np.float32).tiny

def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)

def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5

def iou(pos_pred, pos_gt):
    pred_left = pos_pred[:, 0]
    pred_top = pos_pred[:, 1]
    pred_right = pos_pred[:, 2]
    pred_bottom = pos_pred[:, 3]

    target_left = pos_gt[:, 0]
    target_top = pos_gt[:, 1]
    target_right = pos_gt[:, 2]
    target_bottom = pos_gt[:, 3]

    pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
    target_area = (target_left + target_right) * (target_top + target_bottom)

    w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect
    return area_intersect/area_union

class SafeLog(nn.Module):
    default_hyper_params = dict()

    def __init__(self):
        super(SafeLog, self).__init__()
        self.register_buffer("t_eps", torch.tensor(eps, requires_grad=False))

    def forward(self, t):
        return torch.log(torch.max(self.t_eps, t))

class SigmoidCrossEntropyRetina(nn.Module):

    default_hyper_params = dict(
        name="focal_ce",
        background=0,
        ignore_label=-1,
        weight=1.0,
        alpha=0.5,
        gamma=0.0,
    )

    def __init__(self):
        super(SigmoidCrossEntropyRetina, self).__init__()
        self.safelog = SafeLog()
        self.register_buffer("t_one", torch.tensor(1., requires_grad=False))
        self.update_params()

    def update_params(self):
        self.background = self.default_hyper_params["background"]
        self.ignore_label = self.default_hyper_params["ignore_label"]
        self.weight = self.default_hyper_params["weight"]
        # self.gamma = torch.tensor(float(self.default_hyper_params["gamma"]),
        #                  requires_grad=False)
        # focal loss coefficients
        self.register_buffer(
            "alpha",
            torch.tensor(float(self.default_hyper_params["alpha"]),
                         requires_grad=False))
        self.register_buffer(
            "gamma",
            torch.tensor(float(self.default_hyper_params["gamma"]),
                         requires_grad=False))

    def forward(self, pred_data, target_data):
        r"""
        Focal loss
        :param pred: shape=(B, HW, C), classification logits (BEFORE Sigmoid)
        :param label: shape=(B, HW)
        """
        r"""
        Focal loss
        Arguments
        ---------
        pred: torch.Tensor
            classification logits (BEFORE Sigmoid)
            format: (B, HW)
        label: torch.Tensor
            training label
            format: (B, HW)

        Returns
        -------
        torch.Tensor
            scalar loss
            format: (,)
        """
        pred = pred_data
        label = target_data
        self.gamma=self.gamma.to(pred.device)
        self.alpha=self.alpha.to(pred.device)
        self.safelog=self.safelog.to(pred.device)
        self.t_one=self.t_one.to(pred.device)
        # self.weight.to(pred.device)
        mask = ~(label == self.ignore_label)
        mask = mask.type(torch.Tensor).to(label.device)
        vlabel = label * mask
        zero_mat = torch.zeros(pred.shape[0], pred.shape[1], pred.shape[2] + 1)

        one_mat = torch.ones(pred.shape[0], pred.shape[1], pred.shape[2] + 1)
        index_mat = vlabel.type(torch.LongTensor)         #[[[1],[0],[1],[0]]] (1,4,1)

        onehot_ = zero_mat.scatter(2, index_mat, one_mat)  #[[[0,1],[1,0],[0,1],[1,0]]] 把one_mat数组分配index_mat位置，无指定为0
        onehot = onehot_[:, :, 1:].type(torch.Tensor).to(pred.device)  #真正标签

        pred = torch.sigmoid(pred)
        # pos_part = (1 - pred) ** self.gamma
        pos_part = (1 - pred)**self.gamma * onehot * self.safelog(pred)
        neg_part = pred**self.gamma * (1 - onehot) * self.safelog(1 - pred)
        loss = -(self.alpha * pos_part +
                 (1 - self.alpha) * neg_part).sum(dim=2) * mask.squeeze(2)

        positive_mask = (label > 0).type(torch.Tensor).to(pred.device)

        loss = loss.sum() / torch.max(positive_mask.sum(),
                                      self.t_one) * self.weight

        return loss

class SigmoidCrossEntropyCenterness(nn.Module):
    default_hyper_params = dict(
        name="centerness",
        background=0,
        ignore_label=-1,
        weight=1.0,
    )

    def __init__(self):
        super(SigmoidCrossEntropyCenterness, self).__init__()
        self.register_buffer("t_one", torch.tensor(1., requires_grad=False))
        self.update_params()

    def update_params(self, ):
        self.background = self.default_hyper_params["background"]
        self.ignore_label = self.default_hyper_params["ignore_label"]
        self.weight = self.default_hyper_params["weight"]

    def forward(self, pred_data, target_data):
        r"""
        Center-ness loss
        Computation technique originated from this implementation:
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        P.S. previous implementation can be found at the commit 232141cdc5ac94602c28765c9cf173789da7415e

        Arguments
        ---------
        pred: torch.Tensor
            center-ness logits (BEFORE Sigmoid)
            format: (B, HW)
        label: torch.Tensor
            training label
            format: (B, HW)

        Returns
        -------
        torch.Tensor
            scalar loss
            format: (,)
        """
        self.t_one = self.t_one.to(pred_data.device)
        pred = pred_data
        label = target_data
        mask = (~(label == self.background)).type(torch.Tensor).to(pred.device)
        loss = F.binary_cross_entropy_with_logits(pred, label,
                                                  reduction="none") * mask
        # suppress loss residual (original vers.)
        loss_residual = F.binary_cross_entropy(label, label,
                                               reduction="none") * mask
        loss = loss - loss_residual.detach()

        loss = loss.sum() / torch.max(mask.sum(),
                                      self.t_one) * self.weight

        return loss

class IOULoss(nn.Module):

    default_hyper_params = dict(
        name="iou_loss",
        background=0,
        ignore_label=-1,
        weight=1.0,
    )

    def __init__(self):
        super().__init__()
        self.safelog = SafeLog()
        self.register_buffer("t_one", torch.tensor(1., requires_grad=False))
        self.register_buffer("t_zero", torch.tensor(0., requires_grad=False))
        self.update_params()

    def update_params(self):
        self.background = self.default_hyper_params["background"]
        self.ignore_label = self.default_hyper_params["ignore_label"]
        self.weight = self.default_hyper_params["weight"]

    def forward(self, pred_data, target_data, cls_data):
        self.t_one = self.t_one.to(pred_data.device)
        self.t_zero = self.t_zero.to(pred_data.device)
        self.safelog = self.safelog.to(pred_data.device)
        pred = pred_data
        gt = target_data
        cls_gt = cls_data
        mask = ((~(cls_gt == self.background)) *
                (~(cls_gt == self.ignore_label))).detach()
        mask = mask.type(torch.Tensor).squeeze(2).to(pred.device)

        aog = torch.abs(gt[:, :, 2]) * torch.abs(gt[:, :, 3])
        aop = torch.abs(pred[:, :, 2] - pred[:, :, 0] +
                        1) * torch.abs(pred[:, :, 3] - pred[:, :, 1] + 1)

        iw = torch.min(pred[:, :, 2], gt[:, :, 2]+gt[:,:,0]) - torch.max(
            pred[:, :, 0], gt[:, :, 0]) + 1
        ih = torch.min(pred[:, :, 3], gt[:, :, 3]+gt[:,:,1]) - torch.max(
            pred[:, :, 1], gt[:, :, 1]) + 1
        inter = torch.max(iw, self.t_zero) * torch.max(ih, self.t_zero)

        union = aog + aop - inter
        iou = torch.max(inter / union, self.t_zero)
        loss = -self.safelog(iou)

        loss = (loss * mask).sum() / torch.max(
            mask.sum(), self.t_one) * self.weight
        iou = iou.detach()
        iou = (iou * mask).sum() / torch.max(mask.sum(), self.t_one)
        return loss, iou

class MSELoss(nn.Module):
    def __init__(self,reduction='mean'):
        super(MSELoss,self).__init__()
        self.reduction=reduction
    def forward(self,inputs,targets):
        pos_id=(targets==1.0).float()
        neg_id=(targets==0.0).float()
        pos_loss = pos_id * (inputs - targets) ** 2
        neg_loss = neg_id * (inputs) ** 2
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1))
            return pos_loss, neg_loss
        else:
            return pos_loss, neg_loss



# class IOULoss(nn.Module):
#     def forward(self, pred, target, weight=None): #ltrb
#         pred_left = pred[:, 0]
#         pred_top = pred[:, 1]
#         pred_right = pred[:, 2]
#         pred_bottom = pred[:, 3]
#
#         target_left = target[:, 0]
#         target_top = target[:, 1]
#         target_right = target[:, 2]
#         target_bottom = target[:, 3]
#
#         pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
#         target_area = (target_left + target_right) * (target_top + target_bottom)
#
#         w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
#         h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
#
#         area_intersect = w_intersect * h_intersect
#         area_union = target_area + pred_area - area_intersect
#         # print(area_intersect/area_union)
#         # print((area_intersect/area_union).sum())
#         # print((area_intersect/area_union).sum().mean())
#         # exit()
#         losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
#         if weight is not None and weight.sum() > 0:
#             return (losses * weight).sum() / weight.sum(), (area_intersect/area_union).mean()
#         else:
#             assert losses.numel() != 0
#             return losses.mean(), (area_intersect/area_union).sum().mean()

class siamcarLoss(object):
    def __init__(self):
        self.box_reg_loss = IOULoss()
        self.centerness_loss = nn.BCEWithLogitsLoss()

    def prepare_targets(self, points, labels, gt_bbox):
        labels, reg_targets = self.targets_for_locations(points, labels, gt_bbox)
        return labels, reg_targets

    def targets_for_locations(self, locations, labels, gt_bbox, output_size=13):
        xs, ys = locations[:, 0], locations[:, 1]
        bboxes = gt_bbox
        bboxes[:,2] = bboxes[:,0] + bboxes[:,2]
        bboxes[:,3] = bboxes[:,1] + bboxes[:,3]
        labels = labels.view(output_size**2, -1)
        l = xs[:, None] - bboxes[:, 0][None].float() #(169,b)
        t = ys[:, None] - bboxes[:, 1][None].float()
        r = bboxes[:, 2][None].float() - xs[:, None]
        b = bboxes[:, 3][None].float() - ys[:, None]
        reg_targets_per_img = torch.stack([l, t, r, b], dim=2)
        s1 = reg_targets_per_img[:,:,0] > 0.6*((bboxes[:,2]-bboxes[:,0])/2).float()
        s2 = reg_targets_per_img[:,:,2] > 0.6*((bboxes[:,2]-bboxes[:,0])/2).float()
        s3 = reg_targets_per_img[:,:,1] > 0.6*((bboxes[:,3]-bboxes[:,1])/2).float()
        s4 = reg_targets_per_img[:,:,3] > 0.6*((bboxes[:,3]-bboxes[:,1])/2).float()
        is_in_boxes = s1*s2*s3*s4
        pos = np.where(is_in_boxes.cpu() == 1)
        labels[pos] = 1
        return labels.permute(1,0).contiguous(), reg_targets_per_img.permute(1,0,2).contiguous()
        #(b,169) (b,169,4)

    def centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0,2]]
        top_bottom = reg_targets[:, [1,3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])*\
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, labels, reg_targets):
        num_images = box_cls.size(0)
        # proposal_list =
        label_cls, reg_targets = self.prepare_targets(locations, labels, reg_targets)
        box_regression_flatten = (box_regression.permute(0,2,3,1).contiguous().view(-1,4))
        labels_flatten = (label_cls.view(-1))
        reg_targets_flatten = (reg_targets.view(-1,4))
        centerness_flatten = (centerness.view(-1))

        pos_inds = torch.nonzero(labels_flatten>0, as_tuple=False).squeeze(1)
        # print(pos_inds)
        # exit()
        # pos_inds = (labels_flatten > 0).nonzero().squeeze(1)
        # pos_inds = labels_flatten.data.eq(1).nonzero().squeeze()
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        cls_loss = select_cross_entropy_loss(box_cls, labels_flatten)

        if pos_inds.numel() > 0:
            centerness_targets = self.centerness_targets(reg_targets_flatten)
            reg_loss, mean_iou = self.box_reg_loss(box_regression_flatten, reg_targets_flatten, centerness_targets)
            centerness_loss = self.centerness_loss(centerness_flatten, centerness_targets)
            # mean_iou = iou(box_regression_flatten, reg_targets_flatten).sum().mean()
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()
            mean_iou = 0
        return cls_loss, reg_loss, centerness_loss, mean_iou

def compute_loss():
    loss_evaluator = siamcarLoss()
    return loss_evaluator