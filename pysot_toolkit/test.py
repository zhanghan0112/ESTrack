# Copyright (c) SenseTime. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '..')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
import argparse
import os

import cv2
import torch
import numpy as np

from pysot_toolkit.bbox import get_axis_aligned_bbox
from pysot_toolkit.toolkit.datasets import DatasetFactory
from pysot_toolkit.toolkit.utils.region import vot_overlap, vot_float2str
from pysot_toolkit.trackers.siamyolotracker import siamTracker,draw_grid
# from pysot_toolkit.trackers.siamtracker_one_param import siamTracker
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone

parser = argparse.ArgumentParser(description='transt tracking')
parser.add_argument('--dataset', default='UAV123', type=str,
        help='datasets')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', default=True, action='store_true',
        help='whether visualzie result')
parser.add_argument('--name', default='siamyolo', type=str,
        help='name of results')
parser.add_argument('--net_path', default='/home/zxh/project/TransT/checkpoints/ltr/siamyolo/siamyolo_ddp/Siamyolo_ep0200.pth.tar', type=str, help='checkpoint of the net')
parser.add_argument('--gpu', default='0', type=str, help='the gpu to test')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#
torch.set_num_threads(1)

def main():
    # load config

    dataset_root = '/media/zxh/30CC86E5CC86A526/SOT_eval/VTUAV' #Absolute path of the dataset
    # dataset_root = '/media/zxh/E404D23504D20B06/UAV_dataset/DTB70'
    # net_path = '/home/zxh/project/TransT/checkpoints/ltr/siamcar/siamcar_new_ddp/SiamCAR_ep0200.pth.tar' #Absolute path of the model
    net = NetWithBackbone(net_path=args.net_path, use_gpu=True)
    # tracker = siamTracker(name='siamcar_179', net=net, exemplar_size=128, instance_size=320,
    #                   penalty_k=0.60638817644704091, window_influence=0.41316447361459629,
    #                   lr=0.57887777503228255, stride=16, score_size=13)
    tracker = siamTracker(name='', net=net, exemplar_size=128, instance_size=320,
                          window_influence=0.41316447361459629,stride=16, score_size=20)

    # create dataset
    model_name = args.name
    total_lost = 0
    # dataset = os.listdir(dataset_root)
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
    for v_idx, video in enumerate(dataset):
        if args.video != '':
            # test one special video
            if video != args.video:
                continue
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        # for idx, img in enumerate(sorted(os.listdir(os.path.join(dataset_root, video, 'img')))):
        for idx, (img, gt_bbox) in enumerate(video):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # print('({}) video:{} img:{}'.format(v_idx, video, img))
            # img = cv2.imread(os.path.join(dataset_root, video, 'img', img))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # gt_path = os.path.join(dataset_root, '../DTB70_GT',video+'.txt')
            # gt_bbox = np.loadtxt(gt_path, delimiter=',')[0, :]
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-w/2, cy-h/2, w, h]
                init_info = {'init_bbox':gt_bbox_}
                tracker.initialize(img, init_info)
                pred_bbox = gt_bbox_
                scores.append(None)
                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes.append([1])
                else:
                    pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img,idx)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
                sr_list=outputs['search_region']
                best_score_id=outputs['best_score_id']
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                gt_bbox = list(map(int, gt_bbox))
                pred_bbox = list(map(int, pred_bbox))
                search_region=list(map(int,sr_list))
                x_id=best_score_id//20
                y_id=best_score_id-x_id*20
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                cv2.rectangle(img,(search_region[0],search_region[1]),(search_region[2],search_region[3]),
                              (255,0,0),2)
                draw_grid(img[search_region[1]:search_region[3]+1,search_region[0]:search_region[2]+1,:])
                h,w=search_region[3]-search_region[1]+1, search_region[2]-search_region[0]+1
                cv2.rectangle(img,(search_region[0]+x_id*int(w/320*16),search_region[1]+y_id*int(h/320*16)),
                              (search_region[0]+(x_id+1)*int(w/320*16),search_region[1]+(y_id+1)*int(h/320*16)),(0,0,255),2)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imwrite('/home/zxh/pic/{}.jpg'.format(idx),img)
                cv2.imshow(video.name, img)
                cv2.waitKey(2)
        toc /= cv2.getTickFrequency()
            # save results
        model_path = os.path.join('/home/zxh/project/TransT/results', args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(v_idx+1, video.name, toc, idx / toc))



if __name__ == '__main__':
    main()
