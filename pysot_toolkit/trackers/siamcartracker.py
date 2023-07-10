import numpy as np
import math
import torchvision.transforms.functional as tvisf
import cv2
import torch
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

class siamTracker(object):

    def __init__(self, name, net, exemplar_size=128, instance_size=256, penalty_k=0.04, window_influence=0.44,
                 lr=0.33, stride=16, score_size=20):
        self.name = name
        self.net = net
        self.exemplar_size = exemplar_size
        self.instance_size = instance_size
        self.score_size = score_size
        self.stride = stride
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()

        self.penalty_k = penalty_k
        self.window_influence = window_influence
        self.lr = lr

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
        cv2.imwrite('/home/zxh/picture/search.jpg',cv2.cvtColor(im_patch, cv2.COLOR_BGR2RGB))
        # return
        im_patch = im_patch.transpose(2, 0, 1)
        # im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        im_patch = im_patch.cuda()
        return im_patch

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

    def _postprocess_score(self, score, box_wh, scale_x):
        r"""
        Perform SiameseRPN-based tracker's post-processing of score
        :param score: (HW, ), score prediction
        :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
        :param target_sz: previous state (w & h)
        :param scale_x:
        :return:
            best_pscore_id: index of chosen candidate along axis HW
            pscore: (HW, ), penalized score
            penalty: (HW, ), penalty due to scale/ratio change
        """

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # size penalty
        target_sz_in_crop = self.size * scale_x
        s_c = change(
            sz(box_wh[:, 2], box_wh[:, 3]) /
            (sz_wh(target_sz_in_crop)))  # scale penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (box_wh[:, 2] / box_wh[:, 3]))  # ratio penalty
        penalty = np.exp(-(r_c * s_c - 1) * self.penalty_k)
        # pscore = penalty * score
        pscore = score

        pscore = pscore * (
                1 - self.window_influence) + self.window * self.window_influence
        best_pscore_id = np.argmax(pscore)

        return best_pscore_id, pscore, penalty

    def _postprocess_box(self, best_pscore_id, score, box_wh, target_pos,
                         target_sz, scale_x, x_size, penalty):
        r"""
        Perform SiameseRPN-based tracker's post-processing of box
        :param score: (HW, ), score prediction
        :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
        :param target_pos: (2, ) previous position (x & y)
        :param target_sz: (2, ) previous state (w & h)
        :param scale_x: scale of cropped patch of current frame
        :param x_size: size of instance size #(320,320)
        :param penalty: scale/ratio change penalty calculated during score post-processing
        :return:
            new_target_pos: (2, ), new target position
            new_target_sz: (2, ), new target size
        """
        pred_in_crop = box_wh[best_pscore_id, :] / np.float32(scale_x) #crop坐标
        # lr = penalty[best_pscore_id] * score[best_pscore_id] * self.lr
        res_x = pred_in_crop[0] + target_pos[0] - (x_size // 2) / scale_x
        res_y = pred_in_crop[1] + target_pos[1] - (x_size // 2) / scale_x
        res_w = pred_in_crop[2]
        res_h = pred_in_crop[3]
        # res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
        # res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

        new_target_pos = np.array([res_x, res_y])
        new_target_sz = np.array([res_w, res_h])

        return new_target_pos, new_target_sz

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
        z_crop = self.get_subwindow(image, self.center_pos,
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

    def track(self, image, info: dict = None) -> dict:
        # calculate x crop size
        img_h, img_w = image.shape[:2]
        w_x = self.size[0] + (5 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_x = self.size[1] + (5 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_x = math.ceil(math.sqrt(w_x * h_x))  #crop的边长
        self.scale_x = self.instance_size / s_x
        # get crop
        x_crop = self.get_subwindow(image, self.center_pos,
                                    self.instance_size,
                                    round(s_x), self.channel_average)

        # normalize
        x_crop = x_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        x_crop = tvisf.normalize(x_crop, self.mean, self.std, self.inplace)

        # track
        with torch.no_grad():
            outputs = self.net.track(x_crop.unsqueeze(0))
        box = self.tensor_to_numpy(outputs['loc'][0])
        score = self.tensor_to_numpy(outputs['cls'][0])[:, 0]

        # score = self.tensor_to_numpy(outputs['ctr_prob_final'][0])[:, 0]
        # score = self.tensor_to_numpy(outputs['cls_prob_final'][0])[:, 0]

        box_wh = self.xyxy2cxywh(box)

        best_pscore_id, pscore, penalty = self._postprocess_score(
            score, box_wh, self.scale_x)
        new_target_pos, new_target_sz = self._postprocess_box(
            best_pscore_id, score, box_wh, self.center_pos, self.size, self.scale_x,
            self.instance_size, penalty)

        new_target_pos, new_target_sz = self._restrict_box(
            new_target_pos, new_target_sz, img_h, img_w)

        self.center_pos = new_target_pos
        self.size = new_target_sz

        bbox = [self.center_pos[0] - self.size[0] / 2,
                self.center_pos[1] - self.size[1] / 2,
                self.size[0],
                self.size[1]]
        out = {'bbox': bbox,
               'best_score': pscore[best_pscore_id]}
        # self.i+=1
        # pltshow(cen.reshape(20,20), 'cen'+str(self.i))
        # pltshow(score.reshape(20,20),'score'+str(self.i))
        # pltshow(cls_pre.reshape(20,20), 'cls_pre'+str(self.i))
        return out





