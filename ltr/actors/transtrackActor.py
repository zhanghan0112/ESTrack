from .base_actor import BaseActor
import torch
from util.misc import box_cxcywh_to_xyxy
from util.misc import box_xywh_to_xyxy

class TranstrackActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.settings = settings
        self.loss_weight = loss_weight

    def __call__(self, data):
        outdict, _ = self.net(data['search_images'], data['template_images'])
        gt_bboxes = data['search_anno'].to(self.settings.device)#(b,4) x1,y1,w,h
        pred_bboxes = outdict['pred_boxes'].to(self.settings.device)
        loss, status = self.compute_losses(pred_bboxes, gt_bboxes)
        return loss, status

    def compute_losses(self, pred_boxes, gt_boxes, return_status=True):
        # pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError('Network outputs is NAN! Stop training.')
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1,4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_boxes)[:,None,:].repeat((1,num_queries,1)).view(-1,4).clamp(min=0.0, max=1.0)
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss
        if return_status:
            mean_iou = iou.detach().mean()
            status = {'Loss/total': loss.item(),
                      'Loss/giou': giou_loss.item(),
                      'Loss/l1': l1_loss.item(),
                      'iou': mean_iou.item()}
            return loss, status
        else:
            return  loss



