import torch
import torchvision.transforms as transforms
from pytracking import TensorDict
import ltr.data.processing_utils as prutils
import numpy as np
import math

def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x

class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), search_transform=None, template_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if search_transform or
                                template_transform is None.
            search_transform - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            template_transform  - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the search and template images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'search': transform if search_transform is None else search_transform,
                          'template':  transform if template_transform is None else template_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError

class TransTProcessing(BaseProcessing):
    """ The processing class used for training TransT. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument search_sz.

    """

    def __init__(self, search_area_factor, template_area_factor, search_sz, temp_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region relative to the target size.
            template_area_factor - The size of the template region relative to the template target size.
            search_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            temp_sz - An integer, denoting the size to which the template region is resized. The search region is always
                      square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.template_area_factor = template_area_factor
        self.search_sz = search_sz
        self.temp_sz = temp_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'search' or 'template' indicating search or template data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.sum() * 0.5 * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'search_images', template_images', 'search_anno', 'template_anno'
        returns:
            TensorDict - output data block with following fields:
                'search_images', 'template_images', 'search_anno', 'template_anno'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['search_images'], data['search_anno'] = self.transform['joint'](image=data['search_images'], bbox=data['search_anno'])
            data['template_images'], data['template_anno'] = self.transform['joint'](image=data['template_images'], bbox=data['template_anno'], new_roll=False)

        for s in ['search', 'template']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num search/template frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            if s == 'search':
                crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                           self.search_area_factor, self.search_sz)
            elif s == 'template':
                crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                           self.template_area_factor, self.temp_sz)
            else:
                raise NotImplementedError

            # Apply transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)
        data['template_images'] = data['template_images'].squeeze()
        data['search_images'] = data['search_images'].squeeze()
        data['template_anno'] = data['template_anno'].squeeze()
        data['search_anno'] = data
        return data

class siamyolov1Processing(BaseProcessing):
    def __init__(self, search_area_factor, template_area_factor, search_sz, temp_sz, center_jitter_factor,
                 scale_jitter_factor,mode='pair', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.template_area_factor = template_area_factor
        self.search_sz = search_sz
        self.temp_sz = temp_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode

    def _get_jittered_box(self, box, mode):
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.sum() * 0.5 * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def set_grid(self,hs,ws):
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()*16
        return grid_xy

    def getyolov1label(self,img,bbox,stride):
        bs,_,hi,wi=img.shape
        hs,ws=hi//stride, wi//stride
        # grid_xy = grid_xy.view(hs * ws, 2)
        gt_tensor=torch.zeros((bs,hs,ws,1+4))
        for i in range(bs):
            grid_xy=self.set_grid(hs,ws)
            x1,y1,w,h=bbox[i]
            x2,y2=x1+w,y1+h
            c_x = (x1 + x2) / 2
            c_y = (y1 + y2) / 2
            if w<1. or h<1.:
                continue
            #左上角
            row_x1=int(x1//stride)
            col_y1=int(y1//stride)
            row_x2=int(x2//stride+1)
            col_y2=int(y2//stride+1)
            gt_tensor[i,col_y1:col_y2,row_x1:row_x2,0]=1.0 #(h,w)
            grid_xy[col_y1:col_y2,row_x1:row_x2,:]=grid_xy[col_y1:col_y2,row_x1:row_x2,:]+torch.tensor([7.5,7.5])-torch.tensor([c_x,c_y])
            gt_tensor[i,col_y1:col_y2,row_x1:row_x2,1:3]=grid_xy[col_y1:col_y2,row_x1:row_x2,:]
            tw = torch.log(w/320)
            th=torch.log(h/320)
            gt_tensor[i,col_y1:col_y2,row_x1:row_x2,3:]+=torch.tensor([tw,th])
        return gt_tensor.reshape(bs,-1,1+4)


    def __call__(self, data: TensorDict):
        if self.transform['joint'] is not None:
            data['search_images'], data['search_anno'] = self.transform['joint'](image=data['search_images'], bbox=data['search_anno'])
            data['template_images'], data['template_anno'] = self.transform['joint'](image=data['template_images'], bbox=data['template_anno'], new_roll=False)

        for s in ['search', 'template']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num search/template frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            if s == 'search':
                crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                           self.search_area_factor, self.search_sz)
            elif s == 'template':
                crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                           self.template_area_factor, self.temp_sz)
            else:
                raise NotImplementedError

            # Apply transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)
        # Prepare output

        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)
        data['gt_yolov1_label'] = self.getyolov1label(data['search_images'], data['search_anno'], stride=16).squeeze()
        data['template_images'] = data['template_images'].squeeze()
        data['search_images'] = data['search_images'].squeeze()
        data['template_anno'] = data['template_anno'].squeeze()
        data['search_anno'] = data['search_anno'].squeeze()
        return data

class siamProcessing(BaseProcessing):
    """ The processing class used for training TransT. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument search_sz.

    """

    def __init__(self, search_area_factor, template_area_factor, search_sz, temp_sz, center_jitter_factor, scale_jitter_factor,sigma,score_size,
                 mode='pair', cls_label=True, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region relative to the target size.
            template_area_factor - The size of the template region relative to the template target size.
            search_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            temp_sz - An integer, denoting the size to which the template region is resized. The search region is always
                      square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.template_area_factor = template_area_factor
        self.search_sz = search_sz
        self.temp_sz = temp_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.cls_label = cls_label
        self.score_size = score_size
        self.total_stride = 16
        self.offset = (self.search_sz - 1 - (self.score_size - 1) * self.total_stride) // 2
        self.sigma = sigma

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'search' or 'template' indicating search or template data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.sum() * 0.5 * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def gauss_1d(self, sz, sigma, center, end_pad=0, density=False):
        k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad).reshape(1, -1)
        gauss = torch.exp(-1.0 / (2 * sigma ** 2) * (k - center.reshape(-1, 1)) ** 2)
        if density:
            gauss /= math.sqrt(2 * math.pi) * sigma
        return gauss

    def gauss_2d(self, sz, sigma, center, end_pad=(0, 0), density=False):
        if isinstance(sigma, (float, int)):
            sigma = (sigma, sigma)
        return self.gauss_1d(sz[0].item(), sigma[0], center[0], end_pad[0], density).reshape(1, -1) * \
               self.gauss_1d(sz[1].item(), sigma[1], center[1], end_pad[1], density).reshape(-1, 1)

    def gaussian_label_function(self, target_bb, sigma_factor, kernel_sz, feat_sz, image_sz, end_pad_if_even=False,
                                density=False, uni_bias=0):
        """Construct Gaussian label function."""

        if isinstance(kernel_sz, (float, int)):
            kernel_sz = (kernel_sz, kernel_sz)
        if isinstance(feat_sz, (float, int)):
            feat_sz = (feat_sz, feat_sz)
        if isinstance(image_sz, (float, int)):
            image_sz = (image_sz, image_sz)

        image_sz = torch.Tensor(image_sz)
        feat_sz = torch.Tensor(feat_sz)

        target_center = target_bb[0:2] + 0.5 * target_bb[2:4]
        target_center_norm = (target_center - image_sz / 2) / image_sz

        center = feat_sz * target_center_norm + 0.5 * \
                 torch.Tensor([(kernel_sz[0] + 1) % 2, (kernel_sz[1] + 1) % 2])

        sigma = sigma_factor * feat_sz.prod().sqrt().item()

        if end_pad_if_even:
            end_pad = (int(kernel_sz[0] % 2 == 0), int(kernel_sz[1] % 2 == 0))
        else:
            end_pad = (0, 0)

        gauss_label = self.gauss_2d(feat_sz, sigma, center, end_pad, density=density)
        if density:
            sz = (feat_sz + torch.Tensor(end_pad)).prod()
            label = (1.0 - uni_bias) * gauss_label + uni_bias / sz
        else:
            label = gauss_label + uni_bias
        return label

    def get_cls_label(self, gt_boxes):
        eps = 1e-5
        gt_boxes = torch.cat([torch.zeros(1, 5, dtype=torch.float32), gt_boxes], dim=0)
        gt_boxes_area = (torch.abs(gt_boxes[:, 2]* gt_boxes[:, 3]))
        gt_boxes = gt_boxes[torch.argsort(gt_boxes_area)]
        '''
        torch.argsort 返回按照给定维度升序排列的元素索引
        tensor: [[0.06, 1.52, -0.85],
                 [0.15, 0.07, -0.07],
                 [1.22, 1.07, -0.70]]
        torch.argsort(a,dim=1)
        tensor: [[2,0,1],
                 [2,1,0],
                 [2,1,0]]
        a[2][b[2]]=[-0.70,1.07,1.22]
        '''
        boxes_cnt = len(gt_boxes)
        x_coords = torch.arange(0, self.search_sz, dtype=torch.int64)  # (W, )
        y_coords = torch.arange(0, self.search_sz, dtype=torch.int64)  # (H, )
        y_coords, x_coords = torch.meshgrid(x_coords, y_coords)
        #y行x列  4行3列
        '''
        y_coords:                 x_coords:
        [[0, 0, 0],               [[0, 1, 2],  
         [1, 1, 1],                [0, 1, 2],
         [2, 2, 2],                [0, 1, 2],
         [3, 3, 3]]                [0, 1, 2]]
        '''

        off_l = (x_coords[:, :, np.newaxis, np.newaxis].type(torch.float32) -
                 gt_boxes[np.newaxis, np.newaxis, :, 0, np.newaxis])
        off_t = (y_coords[:, :, np.newaxis, np.newaxis].type(torch.float32) -
                 gt_boxes[np.newaxis, np.newaxis, :, 1, np.newaxis])
        off_r = -(x_coords[:, :, np.newaxis, np.newaxis].type(torch.float32) -
                  (gt_boxes[np.newaxis, np.newaxis, :, 2, np.newaxis]+gt_boxes[np.newaxis, np.newaxis, :, 0, np.newaxis]))
        off_b = -(y_coords[:, :, np.newaxis, np.newaxis].type(torch.float32) -
                  (gt_boxes[np.newaxis, np.newaxis, :, 3, np.newaxis]+gt_boxes[np.newaxis, np.newaxis, :, 1, np.newaxis]))

        center = ((torch.min(off_l, off_r) * torch.min(off_t, off_b)) /
                  (torch.max(off_l, off_r) * torch.max(off_t, off_b) + eps))
        center = torch.squeeze(torch.sqrt(torch.abs(center)), dim=3)
        center[:, :, 0] = 0

        offset = torch.cat([off_l, off_t, off_r, off_b], dim=3)
        cls = gt_boxes[:, 4]
        fm_height, fm_width = self.score_size, self.score_size  # h, w
        fm_offset = self.offset
        stride = self.total_stride
        x_coords_on_fm = torch.arange(0, fm_width, dtype=torch.int64)  # (w, )
        y_coords_on_fm = torch.arange(0, fm_height, dtype=torch.int64)  # (h, )
        y_coords_on_fm, x_coords_on_fm = torch.meshgrid(x_coords_on_fm, y_coords_on_fm)
        y_coords_on_fm = y_coords_on_fm.reshape(-1)  # (hxw, ), flattened
        x_coords_on_fm = x_coords_on_fm.reshape(-1)

        offset_on_fm = offset[fm_offset + y_coords_on_fm * stride, fm_offset + x_coords_on_fm * stride]
        is_in_boxes = (offset_on_fm > 0).all(dim=2).type(torch.uint8)
        offset_valid = torch.zeros((fm_height, fm_width, boxes_cnt), dtype=torch.uint8)
        offset_valid[y_coords_on_fm, x_coords_on_fm, :] = is_in_boxes
        offset_valid[:, :, 0] = 0
        hit_gt_ind = np.argmax(offset_valid, axis=2)
        gt_boxes_res = torch.zeros((fm_height, fm_width, 4))
        gt_boxes_res[y_coords_on_fm, x_coords_on_fm] = gt_boxes[hit_gt_ind[y_coords_on_fm, x_coords_on_fm],:4]  # gt_boxes: (#boxes, 5)
        gt_boxes_res = gt_boxes_res.reshape(-1, 4)
        cls_res = torch.zeros((fm_height, fm_width))
        cls_res[y_coords_on_fm, x_coords_on_fm] = cls[hit_gt_ind[y_coords_on_fm, x_coords_on_fm]]
        cls_res = cls_res.reshape(-1, 1)
        ##
        gauss_label = self.gaussian_label_function(gt_boxes[1,:4], self.sigma, kernel_sz=4, feat_sz=20, image_sz=self.search_sz)
        center_res = gauss_label.reshape(-1,1)

        return cls_res, center_res, gt_boxes_res

    def get_cls_simplify(self, gt_boxes):
        gt_boxes = gt_boxes[:, :4]

        # 320行320列元素
        x_coords = torch.arange(0, self.search_sz, dtype=torch.int64)
        y_coords = torch.arange(0, self.search_sz, dtype=torch.int64)
        y_coords, x_coords = torch.meshgrid(x_coords, y_coords)

        #计算每个像素点与gt的l,t,r,b偏移量
        off_l = (x_coords[:, :, np.newaxis, np.newaxis].type(torch.float32)-
                gt_boxes[np.newaxis, np.newaxis, :, 0])
        off_t = (y_coords[:, :, np.newaxis, np.newaxis].type(torch.float32)-
                gt_boxes[np.newaxis, np.newaxis, :, 1])
        off_r = -(x_coords[:, :, np.newaxis, np.newaxis].type(torch.float32)-
                  (gt_boxes[np.newaxis, np.newaxis, :, 2]+gt_boxes[np.newaxis, np.newaxis, :, 0]))
        off_b = -(y_coords[:, :, np.newaxis, np.newaxis].type(torch.float32)-
                  (gt_boxes[np.newaxis, np.newaxis, :, 3]+gt_boxes[np.newaxis, np.newaxis, :, 1]))

        offset = torch.cat([off_l, off_t, off_r, off_b], dim=2)

        fm_height, fm_width = self.score_size, self.score_size
        fm_offset = self.offset
        stride = self.total_stride
        x_coords_on_fm = torch.arange(0, fm_width, dtype=torch.int64)
        y_coords_on_fm = torch.arange(0, fm_height, dtype=torch.int64)
        y_coords_on_fm, x_coords_on_fm = torch.meshgrid(x_coords_on_fm, y_coords_on_fm)
        y_coords_on_fm = y_coords_on_fm.reshape(-1)
        x_coords_on_fm = x_coords_on_fm.reshape(-1)
        offset_on_fm = offset[fm_offset+y_coords_on_fm*stride, fm_offset+x_coords_on_fm*stride] #(400,4)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'search_images', template_images', 'search_anno', 'template_anno'
        returns:
            TensorDict - output data block with following fields:
                'search_images', 'template_images', 'search_anno', 'template_anno'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['search_images'], data['search_anno'] = self.transform['joint'](image=data['search_images'], bbox=data['search_anno'])
            data['template_images'], data['template_anno'] = self.transform['joint'](image=data['template_images'], bbox=data['template_anno'], new_roll=False)

        for s in ['search', 'template']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num search/template frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            if s == 'search':
                crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                           self.search_area_factor, self.search_sz)
            elif s == 'template':
                crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                           self.template_area_factor, self.temp_sz)
            else:
                raise NotImplementedError

            # Apply transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Prepare output
        data['gt_label'][:,:4] = data['search_anno'][0][:]
        cls_label, center_label, box_label = self.get_cls_label(data['gt_label'])
        data['gt_label'] = box_label
        data['cls_label'] = cls_label
        data['center_label'] = center_label
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)
        data['template_images'] = data['template_images'].squeeze()
        data['search_images'] = data['search_images'].squeeze()
        data['template_anno'] = data['template_anno'].squeeze()
        data['search_anno'] = data['search_anno'].squeeze()
        data['gt_label'] = data['gt_label']
        data['cls_label'] = data['cls_label']
        data['center_label'] = data['center_label']
        return data

class siamCARProcessing(BaseProcessing):
    """ The processing class used for training TransT. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument search_sz.

    """

    def __init__(self, search_area_factor, template_area_factor, search_sz, temp_sz, center_jitter_factor, scale_jitter_factor,sigma,score_size,
                 mode='pair', cls_label=True, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region relative to the target size.
            template_area_factor - The size of the template region relative to the template target size.
            search_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            temp_sz - An integer, denoting the size to which the template region is resized. The search region is always
                      square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.template_area_factor = template_area_factor
        self.search_sz = search_sz
        self.temp_sz = temp_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.cls_label = cls_label
        self.score_size = score_size
        self.total_stride = 16
        self.offset = (self.search_sz - 1 - (self.score_size - 1) * self.total_stride) // 2
        self.sigma = sigma

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'search' or 'template' indicating search or template data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.sum() * 0.5 * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def gauss_1d(self, sz, sigma, center, end_pad=0, density=False):
        k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad).reshape(1, -1)
        gauss = torch.exp(-1.0 / (2 * sigma ** 2) * (k - center.reshape(-1, 1)) ** 2)
        if density:
            gauss /= math.sqrt(2 * math.pi) * sigma
        return gauss

    def gauss_2d(self, sz, sigma, center, end_pad=(0, 0), density=False):
        if isinstance(sigma, (float, int)):
            sigma = (sigma, sigma)
        return self.gauss_1d(sz[0].item(), sigma[0], center[0], end_pad[0], density).reshape(1, -1) * \
               self.gauss_1d(sz[1].item(), sigma[1], center[1], end_pad[1], density).reshape(-1, 1)

    def gaussian_label_function(self, target_bb, sigma_factor, kernel_sz, feat_sz, image_sz, end_pad_if_even=False,
                                density=False, uni_bias=0):
        """Construct Gaussian label function."""

        if isinstance(kernel_sz, (float, int)):
            kernel_sz = (kernel_sz, kernel_sz)
        if isinstance(feat_sz, (float, int)):
            feat_sz = (feat_sz, feat_sz)
        if isinstance(image_sz, (float, int)):
            image_sz = (image_sz, image_sz)

        image_sz = torch.Tensor(image_sz)
        feat_sz = torch.Tensor(feat_sz)

        target_center = target_bb[0:2] + 0.5 * target_bb[2:4]
        target_center_norm = (target_center - image_sz / 2) / image_sz

        center = feat_sz * target_center_norm + 0.5 * \
                 torch.Tensor([(kernel_sz[0] + 1) % 2, (kernel_sz[1] + 1) % 2])

        sigma = sigma_factor * feat_sz.prod().sqrt().item()

        if end_pad_if_even:
            end_pad = (int(kernel_sz[0] % 2 == 0), int(kernel_sz[1] % 2 == 0))
        else:
            end_pad = (0, 0)

        gauss_label = self.gauss_2d(feat_sz, sigma, center, end_pad, density=density)
        if density:
            sz = (feat_sz + torch.Tensor(end_pad)).prod()
            label = (1.0 - uni_bias) * gauss_label + uni_bias / sz
        else:
            label = gauss_label + uni_bias
        return label

    def get_cls_label(self, gt_boxes):
        eps = 1e-5
        gt_boxes = torch.cat([torch.zeros(1, 5, dtype=torch.float32), gt_boxes], dim=0)
        gt_boxes_area = (torch.abs(gt_boxes[:, 2]* gt_boxes[:, 3]))
        gt_boxes = gt_boxes[torch.argsort(gt_boxes_area)]
        '''
        torch.argsort 返回按照给定维度升序排列的元素索引
        tensor: [[0.06, 1.52, -0.85],
                 [0.15, 0.07, -0.07],
                 [1.22, 1.07, -0.70]]
        torch.argsort(a,dim=1)
        tensor: [[2,0,1],
                 [2,1,0],
                 [2,1,0]]
        a[2][b[2]]=[-0.70,1.07,1.22]
        '''
        boxes_cnt = len(gt_boxes)
        x_coords = torch.arange(0, self.search_sz, dtype=torch.int64)  # (W, )
        y_coords = torch.arange(0, self.search_sz, dtype=torch.int64)  # (H, )
        y_coords, x_coords = torch.meshgrid(x_coords, y_coords)
        #y行x列  4行3列
        '''
        y_coords:                 x_coords:
        [[0, 0, 0],               [[0, 1, 2],  
         [1, 1, 1],                [0, 1, 2],
         [2, 2, 2],                [0, 1, 2],
         [3, 3, 3]]                [0, 1, 2]]
        '''

        off_l = (x_coords[:, :, np.newaxis, np.newaxis].type(torch.float32) -
                 gt_boxes[np.newaxis, np.newaxis, :, 0, np.newaxis])
        off_t = (y_coords[:, :, np.newaxis, np.newaxis].type(torch.float32) -
                 gt_boxes[np.newaxis, np.newaxis, :, 1, np.newaxis])
        off_r = -(x_coords[:, :, np.newaxis, np.newaxis].type(torch.float32) -
                  (gt_boxes[np.newaxis, np.newaxis, :, 2, np.newaxis]+gt_boxes[np.newaxis, np.newaxis, :, 0, np.newaxis]))
        off_b = -(y_coords[:, :, np.newaxis, np.newaxis].type(torch.float32) -
                  (gt_boxes[np.newaxis, np.newaxis, :, 3, np.newaxis]+gt_boxes[np.newaxis, np.newaxis, :, 1, np.newaxis]))

        center = ((torch.min(off_l, off_r) * torch.min(off_t, off_b)) /
                  (torch.max(off_l, off_r) * torch.max(off_t, off_b) + eps))
        center = torch.squeeze(torch.sqrt(torch.abs(center)), dim=3)
        center[:, :, 0] = 0

        offset = torch.cat([off_l, off_t, off_r, off_b], dim=3)
        cls = gt_boxes[:, 4]
        fm_height, fm_width = self.score_size, self.score_size  # h, w
        fm_offset = self.offset
        stride = self.total_stride
        x_coords_on_fm = torch.arange(0, fm_width, dtype=torch.int64)  # (w, )
        y_coords_on_fm = torch.arange(0, fm_height, dtype=torch.int64)  # (h, )
        y_coords_on_fm, x_coords_on_fm = torch.meshgrid(x_coords_on_fm, y_coords_on_fm)
        y_coords_on_fm = y_coords_on_fm.reshape(-1)  # (hxw, ), flattened
        x_coords_on_fm = x_coords_on_fm.reshape(-1)

        offset_on_fm = offset[fm_offset + y_coords_on_fm * stride, fm_offset + x_coords_on_fm * stride]
        is_in_boxes = (offset_on_fm > 0).all(dim=2).type(torch.uint8)
        offset_valid = torch.zeros((fm_height, fm_width, boxes_cnt), dtype=torch.uint8)
        offset_valid[y_coords_on_fm, x_coords_on_fm, :] = is_in_boxes
        offset_valid[:, :, 0] = 0
        hit_gt_ind = np.argmax(offset_valid, axis=2)
        gt_boxes_res = torch.zeros((fm_height, fm_width, 4))
        gt_boxes_res[y_coords_on_fm, x_coords_on_fm] = gt_boxes[hit_gt_ind[y_coords_on_fm, x_coords_on_fm],:4]  # gt_boxes: (#boxes, 5)
        gt_boxes_res = gt_boxes_res.reshape(-1, 4)
        cls_res = torch.zeros((fm_height, fm_width))
        cls_res[y_coords_on_fm, x_coords_on_fm] = cls[hit_gt_ind[y_coords_on_fm, x_coords_on_fm]]
        cls_res = cls_res.reshape(-1, 1)
        ##
        gauss_label = self.gaussian_label_function(gt_boxes[1,:4], self.sigma, kernel_sz=4, feat_sz=13, image_sz=self.search_sz)
        center_res = gauss_label.reshape(-1,1)
        return cls_res, center_res, gt_boxes_res

    def get_cls_simplify(self, gt_boxes):
        eps = 1e-5
        gt_boxes = gt_boxes[:, :4]
        gt_boxes_area = torch.abs(gt_boxes[:,2]*gt_boxes[:,3])
        # 320行320列元素
        x_coords = torch.arange(0, self.search_sz, dtype=torch.int64)
        y_coords = torch.arange(0, self.search_sz, dtype=torch.int64)
        y_coords, x_coords = torch.meshgrid(x_coords, y_coords)

        #计算每个像素点与gt的l,t,r,b偏移量
        off_l = (x_coords[:, :, np.newaxis, np.newaxis].type(torch.float32)-
                gt_boxes[np.newaxis, np.newaxis, :, 0])
        off_t = (y_coords[:, :, np.newaxis, np.newaxis].type(torch.float32)-
                gt_boxes[np.newaxis, np.newaxis, :, 1])
        off_r = -(x_coords[:, :, np.newaxis, np.newaxis].type(torch.float32)-
                  (gt_boxes[np.newaxis, np.newaxis, :, 2]+gt_boxes[np.newaxis, np.newaxis, :, 0]))
        off_b = -(y_coords[:, :, np.newaxis, np.newaxis].type(torch.float32)-
                  (gt_boxes[np.newaxis, np.newaxis, :, 3]+gt_boxes[np.newaxis, np.newaxis, :, 1]))
        center = ((torch.min(off_l, off_r) * torch.min(off_t, off_b)) /
                  (torch.max(off_l, off_r) * torch.max(off_t, off_b)+eps))
        center = torch.sqrt(torch.abs(center)) #(320,320)

        offset = torch.cat([off_l, off_t, off_r, off_b], dim=2)

        fm_height, fm_width = self.score_size, self.score_size
        fm_offset = self.offset
        stride = self.total_stride
        x_coords_on_fm = torch.arange(0, fm_width, dtype=torch.int64)
        y_coords_on_fm = torch.arange(0, fm_height, dtype=torch.int64)
        y_coords_on_fm, x_coords_on_fm = torch.meshgrid(x_coords_on_fm, y_coords_on_fm)
        y_coords_on_fm = y_coords_on_fm.reshape(-1)
        x_coords_on_fm = x_coords_on_fm.reshape(-1)
        offset_on_fm = offset[fm_offset+y_coords_on_fm*stride, fm_offset+x_coords_on_fm*stride] #(400,4)
        is_in_boxes = (offset_on_fm>0).all(dim=1).type(torch.uint8)
        in_box_index = torch.nonzero(is_in_boxes) #z*n
        gt_boxes_res = torch.zeros((fm_height, fm_width, 4))

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'search_images', template_images', 'search_anno', 'template_anno'
        returns:
            TensorDict - output data block with following fields:
                'search_images', 'template_images', 'search_anno', 'template_anno'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['search_images'], data['search_anno'] = self.transform['joint'](image=data['search_images'], bbox=data['search_anno'])
            data['template_images'], data['template_anno'] = self.transform['joint'](image=data['template_images'], bbox=data['template_anno'], new_roll=False)

        for s in ['search', 'template']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num search/template frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            if s == 'search':
                crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                           self.search_area_factor, self.search_sz)
            elif s == 'template':
                crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                           self.template_area_factor, self.temp_sz)
            else:
                raise NotImplementedError

            # Apply transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Prepare output
        data['gt_label'][:,:4] = data['search_anno'][0][:]
        cls_label, center_label, box_label = self.get_cls_label(data['gt_label'])
        data['gt_label'] = box_label
        data['cls_label'] = cls_label
        data['center_label'] = center_label
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)
        data['template_images'] = data['template_images'].squeeze()
        data['search_images'] = data['search_images'].squeeze()
        data['template_anno'] = data['template_anno'].squeeze()
        data['search_anno'] = data['search_anno'].squeeze()
        data['gt_label'] = data['gt_label']
        data['cls_label'] = data['cls_label']
        data['center_label'] = data['center_label']
        return data



if __name__ == '__main__':
    pass
    # def getyolov1gtlabel(bbox, img): #bbox没有归一化
    #     b,c,h,w=img.shape  #(b,c,h,w)
    #     grid_num=4
    #     target=torch.zeros((grid_num, grid_num, 11)) #(h,w,c) 12为两个锚框 5+5+
    #     wh=bbox[:,2:]
    #     cell_size=h/grid_num
    #     cxcy=(bbox[:,:2]+wh)/2
    #     for i in range(cxcy.size()[0]):
    #         sample_cen=cxcy[i]  #
    #         sample_cell=(sample_cen/grid_num).ceil()-1 #
    #         x_cell, y_cell=int(sample_cell[1]),int(sample_cell[0])
    #         target[x_cell,y_cell,4]=1
    #         target[x_cell,y_cell,9]=1
    #         target[x_cell,y_cell,10]=1
    #         x0y0=torch.tensor([x_cell*cell_size,y_cell*cell_size])
    #         print(x0y0)
    #         delta_xy=(sample_cen-x0y0)/cell_size #匹配到的网格左上角相对坐标
    #         target[x_cell,y_cell,2:4]=torch.tensor([wh[i][0]/h,wh[i][1]/w])
    #         target[x_cell,y_cell,:2]=delta_xy
    #         target[x_cell, y_cell, 7:9] = torch.tensor([wh[i][0] / h, wh[i][1] / w])
    #         target[x_cell, y_cell, 5:7] = delta_xy
    #     return target
    # boxes=torch.tensor([[25.0,67.0,40.0,80.0],[31.0,42.0,40.0,80.0]])
    # img=torch.randn(2,3,96,96)
    # f=getyolov1label(img,boxes,16)
    # print(boxes)
    # print(f)
