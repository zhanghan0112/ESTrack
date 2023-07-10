import numpy as np
import math
import torchvision.transforms.functional as tvisf
import cv2
import torch
import torch.nn.functional as F
import time

def pltshow(pred_map, name):
    import matplotlib.pyplot as plt
    plt.figure(2)
    pred_frame = plt.gca()
    plt.imshow(pred_map, 'jet')
    pred_frame.axes.get_yaxis().set_visible(False)
    pred_frame.axes.get_xaxis().set_visible(False)
    pred_frame.spines['top'].set_visible(False)
    pred_frame.spines['bottom'].set_visible(False)
    pred_frame.spines['left'].set_visible(False)
    pred_frame.spines['right'].set_visible(False)
    pred_name = '/home/zxh/picture/' + name + '.png'
    plt.savefig(pred_name, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(2)

def draw_grid(img,line_color=(255,255,0),thickness=1):
    h,w,c=img.shape
    xstep=int(w/320*16)
    ystep=int(h/320*16)
    x=xstep
    y=ystep
    while x<img.shape[1]:
        cv2.line(img,(x,0),(x,w-1),color=line_color,thickness=thickness)
        x+=xstep
    while y<img.shape[0]:
        cv2.line(img,(0,y),(h-1,y),color=line_color,thickness=thickness)
        y+=ystep

class siamTracker(object):
    def __init__(self, name, net, exemplar_size=128, instance_size=320, window_influence=0.44, stride=16, score_size=20):
        self.name = name
        self.net = net
        self.exemplar_size = exemplar_size
        self.instance_size = instance_size
        self.score_size = score_size
        self.stride = stride

        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        self.window_influence = window_influence

        self.initialize_features()
        self.i = 0

    def sz(self, w, h):
        pad = (w+h) * 0.5
        return np.sqrt((w+pad) * (h+pad))

    def change(self, r):
        return np.maximum(r, 1. / r)

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: rgb based image
            pos: center position
            model_sz: exemplar size
            original_sz: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        # im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        im_patch = im_patch.cuda()
        return im_patch,[context_xmin,context_ymin,context_xmax,context_ymax]

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.net.initialize()
        self.features_initialized = True

    def tensor_to_numpy(self, t):
        arr = t.detach().cpu().numpy()
        return arr

    def xyxy2cxywh(self, bbox):
        bbox = np.array(bbox, dtype=np.float32)
        return np.concatenate([(bbox[..., [0]] + bbox[..., [2]]) / 2,
                               (bbox[..., [1]] + bbox[..., [3]]) / 2,
                               bbox[..., [2]] - bbox[..., [0]] + 1,
                               bbox[..., [3]] - bbox[..., [1]] + 1],
                              axis=-1)

    def _restrict_box(self, target_pos, target_sz, img_h, img_w):
        r"""
        Restrict target position & size
        :param target_pos: (2, ), target position
        :param target_sz: (2, ), target size
        :return:
            target_pos, target_sz
        """
        target_pos[0] = max(0, min(img_w, target_pos[0]))
        target_pos[1] = max(0, min(img_h, target_pos[1]))
        target_sz[0] = max(10, min(img_h, target_sz[0]))
        target_sz[1] = max(10, min(img_w, target_sz[1]))
        return target_pos, target_sz

    def create_grid(self,input_size):
        w,h=input_size[1],input_size[0]
        ws,hs=w//self.stride, h//self.stride
        grid_y, grid_x=torch.meshgrid([torch.arange(hs),torch.arange(ws)])
        grid_xy=torch.stack([grid_x,grid_y],dim=-1).float()
        grid_xy=grid_xy.view(1,hs*ws,2).cuda()
        return grid_xy

    def decode_box(self,pred):
        grid=self.create_grid([320,320])*self.stride
        # draw_grid(image)
        pred[:,:,:2]= grid[:,:,:]+torch.tensor([7.5,7.5]).cuda()-pred[:,:,:2]
        # pred[:,:,:2]=pred[:,:,:2]+grid
        pred[:,:,2:]=torch.exp(pred[:,:,2:])*320
        return pred

    def initialize(self, image, info: dict) -> dict:
        tic = time.time()
        bbox = info['init_bbox']
        self.center_pos = np.array([bbox[0] + bbox[2] / 2,
                                    bbox[1] + bbox[3] / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_z = self.size[1] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_z = math.ceil(math.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(image, axis=(0, 1))

        # get crop
        z_crop,zr_list = self.get_subwindow(image, self.center_pos,
                                    self.exemplar_size,
                                    s_z, self.channel_average)

        # normalize
        z_crop = z_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.inplace = False
        z_crop = tvisf.normalize(z_crop, self.mean, self.std, self.inplace)

        # initialize template feature
        with torch.no_grad():
            self.net.template(z_crop.unsqueeze(0))
        out = {'time': time.time() - tic}
        return out

    def track(self, image,idx, info: dict = None) -> dict:
        # calculate x crop size
        img_h, img_w = image.shape[:2]
        w_x = self.size[0] + (5 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_x = self.size[1] + (5 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_x = math.ceil(math.sqrt(w_x * h_x))  #crop的边长
        self.scale_x = self.instance_size / s_x
        # get crop
        x_crop,sr_list = self.get_subwindow(image, self.center_pos,
                                    self.instance_size,
                                    round(s_x), self.channel_average)

        # normalize
        x_crop = x_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        x_crop = tvisf.normalize(x_crop, self.mean, self.std, self.inplace)

        # track
        with torch.no_grad():
            outputs = self.net.track(x_crop.unsqueeze(0))
        result=outputs['result'].view(1,-1,5)
        txtytwth_pred = result[:,:,1:]
        cls=torch.sigmoid(result[:,:,0])[0]
        box=torch.clamp(self.decode_box(txtytwth_pred)[0],0,self.instance_size)
        conf=cls.detach().cpu().numpy()
        box_wh=box.detach().cpu().numpy() #cxcywh
        # box = self.tensor_to_numpy(outputs['loc'][0])
        # score = self.tensor_to_numpy(outputs['cls'][0])[:, 0]
        # feat = self.tensor_to_numpy(outputs['dist_cls'][0])[:,0]
        # box_wh = self.xyxy2cxywh(box)
        #max score
        # pscore = conf * (1 - self.window_influence) + self.window * self.window_influence
        best_pscore_id = np.argmax(conf)
        # for i in range(200,210):
        #     pred_in_crop = box_wh[i, :] / np.float32(self.scale_x)
        #     res_x1 = pred_in_crop[0] + self.center_pos[0] - (self.instance_size // 2) / self.scale_x
        #     res_y1 = pred_in_crop[1] + self.center_pos[1] - (self.instance_size // 2) / self.scale_x
        #     res_w = pred_in_crop[2]
        #     res_h = pred_in_crop[3]
        #     x1 = res_x1 - res_w / 2
        #     y1 = res_y1 - res_h / 2
        #     x2 = res_x1 + res_w / 2
        #     y2 = res_y1 + res_h / 2
        #     pred_bbox = list(map(int, [x1,y1,x2,y2]))
        #     cv2.rectangle(image, (pred_bbox[0], pred_bbox[1]),
        #                   (pred_bbox[2], pred_bbox[3]), (0, 255, 255), 1)
        # cv2.imwrite('/home/zxh/pic_{}.jpg'.format(idx), image)
        #the predicted bbox on the patch
        pred_in_crop = box_wh[best_pscore_id, :] / np.float32(self.scale_x)  # crop坐标
        res_x = pred_in_crop[0] + self.center_pos[0] - (self.instance_size // 2) / self.scale_x
        res_y = pred_in_crop[1] + self.center_pos[1] - (self.instance_size // 2) / self.scale_x
        res_w = pred_in_crop[2]
        res_h = pred_in_crop[3]
        new_target_pos = np.array([res_x, res_y])
        new_target_sz = np.array([res_w, res_h])
        new_target_pos, new_target_sz = self._restrict_box(
            new_target_pos, new_target_sz, img_h, img_w)
        self.center_pos = new_target_pos
        self.size = new_target_sz
        bbox = [self.center_pos[0] - self.size[0] / 2,
                self.center_pos[1] - self.size[1] / 2,
                self.size[0],
                self.size[1]]
        out = {'bbox': bbox,
               'best_score': conf[best_pscore_id],
               'best_score_id':best_pscore_id,
               'search_region':sr_list}
        # print(best_pscore_id)
        #
        # self.i += 1
        # pltshow(score.reshape(20,20), 'cls'+str(self.i))
        # pltshow(feat.reshape(20, 20), 'feat' + str(self.i))
        # exit()
        return out

    # pltshow(box[:,0].reshape(20,20),'box_l'+str(self.i))
    # pltshow(box[:, 1].reshape(20, 20), 'box_t' + str(self.i))
    # pltshow(box[:, 2].reshape(20, 20), 'box_r' + str(self.i))
    # pltshow(box[:, 3].reshape(20, 20), 'box_b' + str(self.i))
    # pltshow(score_cls.reshape(20,20), 'cls'+str(self.i))
    # pltshow(score_ctr.reshape(20, 20), 'ctr' + str(self.i))


