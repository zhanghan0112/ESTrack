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
from pysot_toolkit.trackers.siamtracker_one_param import siamTracker
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone
from pysot_toolkit.toolkit.utils.statistics import overlap_ratio

parser = argparse.ArgumentParser(description='transt tracking')
parser.add_argument('--dataset', default='VTUAV', type=str,
        help='datasets')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', default=False, action='store_true',
        help='whether visualzie result')
parser.add_argument('--name', default='now_vtuav_lt', type=str,
        help='name of results')
parser.add_argument('--wi', default=0.41316447361459629, type=float,help='wi')
parser.add_argument('--net_path', default='', type=str, help='checkpoint of the net')
parser.add_argument('--gpu', default='0', type=str, help='the gpu to test')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

torch.set_num_threads(1)

def main():
    # load config

    dataset_root = ''
    # create model
    #dtb70 resnet18 wi:0.879436016407285
    #uavdt resnet18 wi:0.7537833826806632

    #repvgg
    #dtb70 wi: 0.7752969124274975
    #uavdt wi: 0.52581553484087395
    #visdrone wi: 0.6176413563670935


    #uav123 wi:0.41316447361459629  auc:0.685
    #uav10fps:0.41316447361459629
    #uav20L:wi:0.05721393985387185
    #visdrone:wi:0.84957896336090577  auc:0.656
    #0-0.5
    net = NetWithBackbone(net_path=args.net_path, use_gpu=True)
    tracker = siamTracker(name='siamcar_179', net=net, exemplar_size=128, instance_size=320,
                          window_influence=args.wi, stride=16, score_size=20)

    # create dataset
    model_name = args.name
    dataset = os.listdir(dataset_root)
    # dataset = DatasetFactory.create_dataset(name=args.dataset,
    #                                         dataset_root=dataset_root,
    #                                         load_img=False)
    for v_idxes, video_all in enumerate(dataset):
        alldata=os.listdir(os.path.join(dataset_root,video_all))
        for v_idx, video in enumerate(alldata):
            predict_path='/home/zxh/project/TransT/results/VTUAV/now_vtuav_lt'
            if video+'.txt' in os.listdir(predict_path):
                continue
            if args.video != '':
                # test one special video
                if video != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, img in enumerate(sorted(os.listdir(os.path.join(dataset_root, video_all, video, 'rgb')))):
            # for idx, gt in enumerate(sorted(os.listdir(os.path.join(dataset_root,'../anno/LT')))):
            # for idx, (img, gt_bbox) in enumerate(video):
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     if img.split('.')[1] != 'jpg':
            #         continue
            #     print(sorted(os.listdir(os.path.join(dataset_root, video_all, video, 'rgb'))))
                print('({}) video:{} img:{}'.format(v_idx, video, img))
                img = cv2.imread(os.path.join(dataset_root, video_all, video, 'rgb', img))
                # print(os.path.join(dataset_root, video_all, video, 'rgb', img))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gt_path = os.path.join(dataset_root, video_all, video, 'rgb.txt')
                gt_bbox = np.loadtxt(gt_path, delimiter=' ')[0, :]
                tic = cv2.getTickCount()
                if idx==0:
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
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                    (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                    (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
                # save results
            model_path = os.path.join('', args.dataset, model_name)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path, '{}.txt'.format(video))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video, toc, idx / toc))

if __name__ == '__main__':
    main()
