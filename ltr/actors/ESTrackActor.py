from .base_actor import BaseActor
import torch
import torch.nn as nn

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

class ESTrackActor(BaseActor):
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'loc_weight': 1.0, 'cls_weight': 1.0}
        self.loss_weight = loss_weight
        self.clsloss = objective['clsloss']
        self.iouloss = objective['iouloss']
        self.centerloss = objective['centerloss']

    def __call__(self, data):
        cls_score, bbox, center_score = self.net(data['template_images'], data['search_images'])
        clsloss = self.clsloss(cls_score, data['cls_label'])
        iouloss, iou = self.iouloss(bbox, data['gt_label'], data['cls_label'])
        centerloss = self.centerloss(center_score, data['center_label'])
        total_loss = self.loss_weight['cls_weight'] * clsloss + self.loss_weight['loc_weight'] * \
                     iouloss + self.loss_weight['cen_weight'] * centerloss

        status = {'Loss/total': total_loss.item(),
                  'Loss/cls_loss': clsloss.item(),
                  'Loss/loc_loss': iouloss.item(),
                  'Loss/cen_loss': centerloss.item(),
                  'mean_iou': iou.item()}
        return total_loss, status

class siamyolov1Actor(BaseActor):
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'loc_weight': 1.0, 'cls_weight': 1.0}
        self.cls_loss=objective['clsloss']
        self.txty_loss=objective['txtyloss']
        self.twth_loss=objective['twth_loss']

    def __call__(self,data):
        result=self.net(data['template_images'], data['search_images'])
        target_tensor=data['gt_yolov1_label']
        pred_tensor=result.view(result.size()[0],-1,5)
        pred_cls=pred_tensor[:,:,0]
        txtytwth_pred=pred_tensor[:,:,1:]

        pred_txty=txtytwth_pred[:,:,:2]
        pred_twth=txtytwth_pred[:,:,2:]

        gt_cls=target_tensor[:,:,0]
        gt_txtytwth=target_tensor[:,:,1:]
        #iou
        pos_id=(gt_cls==1.0).float()
        pos_index=torch.nonzero(pos_id)

        ious=torch.randn(len(pos_index)).cuda()
        ws, hs = 20, 20
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(hs * ws, 2)

        grid_xy += torch.tensor([7.5, 7.5])
        grid_xy.cuda()
        for i in range(pos_index.size(0)):
            pic=pos_index[i][0]
            txty_pred_pos=torch.index_select(pred_txty[pic],0,pos_index[i][1])
            twth_pred_pos = torch.index_select(pred_twth[pic], 0, pos_index[i][1])
            txtytwth_gt_pos = torch.index_select(gt_txtytwth[pic], 0, pos_index[i][1])
            pred_cx=grid_xy[pos_index[i][1]][0]-txty_pred_pos[0][0]
            pred_cy=grid_xy[pos_index[i][1]][1]-txty_pred_pos[0][1]
            pred_w=torch.exp(twth_pred_pos[0][0])*320
            pred_h=torch.exp(twth_pred_pos[0][1])*320
            gt_cx=grid_xy[pos_index[i][1]][0]-txtytwth_gt_pos[0][0]
            gt_cy=grid_xy[pos_index[i][1]][1]-txtytwth_gt_pos[0][1]
            gt_w=torch.exp(txtytwth_gt_pos[0][2])*320
            gt_h=torch.exp(txtytwth_gt_pos[0][3])*320
            xmin=max(pred_cx-pred_w/2,gt_cx-gt_w/2)
            ymin=max(pred_cy-pred_h/2,gt_cy-gt_h/2)
            xmax=min(pred_cx+pred_w/2,gt_cx+gt_w/2)
            ymax=min(pred_cy+pred_h/2,gt_cy+gt_h/2)
            insert=max(xmax-xmin,0)*max(ymax-ymin,0)
            pred_s=pred_h*pred_w
            gt_s=gt_h*gt_w
            iou=max(insert/(pred_s+gt_s-insert),0)
            ious[i]=iou
        iou_avg=torch.mean(ious)
        cls_loss=self.cls_loss(pred_cls.unsqueeze(-1), gt_cls.unsqueeze(-1))
        #box loss
        txty_loss=self.txty_loss(pred_txty, gt_txtytwth[:, :, :2])
        twth_loss=self.twth_loss(pred_twth,gt_txtytwth[:,:,2:])
        txtytwth_loss=txty_loss+twth_loss
        total_loss=cls_loss+txtytwth_loss
        status = {'Loss/total': total_loss.item(),
                  'Loss/cls_loss': cls_loss.item(),
                  'Loss/txty_loss': txty_loss.item(),
                  'Loss/twth_loss': twth_loss.item(),
                  'Loss/txtytwth_loss': txtytwth_loss.item(),
                  'iou': iou_avg.item()}
        return total_loss, status



