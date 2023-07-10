import os
import cv2
import sys
import random
import torch
import joblib
import optuna
import logging
import argparse

import numpy as np
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pysot_toolkit.bbox import get_axis_aligned_bbox
from pysot_toolkit.toolkit.datasets import DatasetFactory
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone
from pysot_toolkit.toolkit.datasets import OTBDataset, UAVDataset, LaSOTDataset, VOTDataset, NFSDataset, VOTLTDataset
from pysot_toolkit.toolkit.evaluation import OPEBenchmark, EAOBenchmark, F1Benchmark
from pysot_toolkit.trackers.siamcartracker import siamTracker
from pysot_toolkit.toolkit.utils.statistics import success_overlap

class uav123_10fps(object):
    def __init__(self, name, dataset_root, pred_root, gt_root):
        self.name = name
        self.dataset_root = dataset_root
        self.pred_root = pred_root
        self.gt_root = gt_root
        self.videos = None

    def read_videos(self):
        videos = os.listdir(self.dataset_root)
        self.videos = {}
        for video in videos:
            if self.name == 'UAVDT':
                gt_np = np.loadtxt(os.path.join(self.gt_root,video+'_gt.txt'),dtype=np.float32,delimiter=',')
            else:
                gt_np = np.loadtxt(os.path.join(self.gt_root, video + '.txt'), dtype=np.float32, delimiter=',')
            self.videos[video] = gt_np
        return self.videos

    def cal_auc(self):
        success_ret_ = {}
        self.videos = self.read_videos()
        for video in self.videos:
            gt_np = self.videos[video]
            pred_np = np.loadtxt(os.path.join(self.pred_root, video+'.txt'), dtype=np.float32, delimiter=',')
            n_frame = len(gt_np)
            success_ret_[video] = success_overlap(gt_np, pred_np, n_frame)
        auc = np.mean(list(success_ret_.values()))
        return auc


    def set_tracker(self, path, tracker_names):
        """
        Args:
            path: path to tracker results,
            tracker_names: list of tracker name
        """
        self.tracker_path = path
        self.tracker_names = tracker_names

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def eval(dataset, tracker_name):

    tracker_dir = "./"
    trackers = [tracker_name]
    if 'OTB' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'LaSOT' == args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'UAV123' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'NFS' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    if args.dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = EAOBenchmark(dataset)
        eval_eao = benchmark.eval(tracker_name)
        eao = eval_eao[tracker_name]['all']
        return eao
    elif 'VOT2018-LT' == args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = F1Benchmark(dataset)
        eval_f1 = benchmark.eval(tracker_name)
        f1 = eval_f1[tracker_name]['f1']
        return f1
    if args.dataset in ['UAV20L', 'UAV123_10fps']:
        if args.dataset == 'UAV20L':
            data_root = ''
            pred_root = tracker_name
            gt_root = ''
            dataset_ = uav123_10fps(args.dataset, data_root, pred_root, gt_root)
            return dataset_.cal_auc()
        if args.dataset == 'UAV123_10fps':
            data_root = ''
            pred_root = tracker_name
            gt_root = ''
            dataset = uav123_10fps(args.dataset, data_root, pred_root, gt_root)
            return dataset.cal_auc()
    if args.dataset == 'DTB70':
        data_root = ''
        pred_root = tracker_name
        gt_root = ''
        dataset = uav123_10fps(args.dataset, data_root, pred_root, gt_root)
        return dataset.cal_auc()
    if args.dataset == 'UAVDT':
        data_root = ''
        pred_root = tracker_name
        gt_root = ''
        dataset = uav123_10fps(args.dataset, data_root, pred_root, gt_root)
        return dataset.cal_auc()
    if args.dataset == 'visdrone':
        data_root = ''
        pred_root = tracker_name
        gt_root = ''
        dataset = uav123_10fps(args.dataset, data_root, pred_root, gt_root)
        return dataset.cal_auc()

# fitness function
def objective(trial):
    # different params
    # WINDOW_INFLUENCE = trial.suggest_uniform('window_influence', 0.250, 0.450)
    # PENALTY_K = trial.suggest_uniform('penalty_k', 0.000, 0.200)
    # LR = trial.suggest_uniform('scale_lr', 0.500, 0.850)
    WINDOW_INFLUENCE = trial.suggest_uniform('window_influence', args.min_wi, args.max_wi)
    tracker_name = os.path.join(tune_result, model_name + \
                                '_wi-{:.3f}'.format(WINDOW_INFLUENCE))
    # PENALTY_K = trial.suggest_uniform('penalty_k', args.min_k, args.max_k)
    # LR = trial.suggest_uniform('scale_lr', args.min_lr, args.max_lr)
    # tracker_name = os.path.join(tune_result, model_name + \
    #                             '_wi-{:.3f}'.format(WINDOW_INFLUENCE) + \
    #                             '_pk-{:.3f}'.format(PENALTY_K) + \
    #                             '_lr-{:.3f}'.format(LR))
    tracker = siamTracker(name='', net=net, exemplar_size=128, instance_size=320,
                      window_influence=WINDOW_INFLUENCE, stride=16, score_size=20)
    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            seed_torch(SEED)
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    init_info = {'init_bbox': gt_bbox_}
                    tracker.initialize(img, init_info)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5  # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                    # no vis function
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join(tracker_name, 'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                v_idx + 1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
        eao = eval(dataset=dataset_eval, tracker_name=tracker_name)
        info = "{:s} window_influence: {:1.17f}, penalty_k: {:1.17f}, scale_lr: {:1.17f}, EAO: {:1.3f}".format(
            model_name, WINDOW_INFLUENCE, PENALTY_K, LR, eao)
        logging.getLogger().info(info)
        print(info)
        return eao
    elif args.dataset == 'GOT10kVal':
        """需要run出结果文件，并且进行evaluation得到AO"""
        from got10k.trackers import Tracker
        from got10k.experiments import ExperimentGOT10k
        class GOT10kTrackerWrapper(Tracker):
            def __init__(self, name, dtracker):
                super(GOT10kTrackerWrapper, self).__init__(name=name, is_deterministic=True)
                self.dtracker = dtracker

            def init(self, image, box):
                image = np.array(image)
                self.gt_bbox_ = box
                init_info = {'init_bbox': self.gt_bbox_}
                self.dtracker.initialize(image, init_info)

            def update(self, image):
                image = np.array(image)
                outputs = self.dtracker.track(image)
                pred_bbox = outputs['bbox']
                return pred_bbox
        GOT10kTracker = GOT10kTrackerWrapper('/'.join(tracker_name.split('/')[1:]), tracker)
        experiment = ExperimentGOT10k(dataset, subset='val', result_dir='tune_results', report_dir='tune_results')
        experiment.run(GOT10kTracker, visualize=False)
        # report performance
        performance = experiment.report([GOT10kTracker.name])
        AO = performance[GOT10kTracker.name]['overall']['ao']
        info = "{:s} window_influence: {:1.17f}, penalty_k: {:1.17f}, scale_lr: {:1.17f}, AO: {:1.3f}".format(
            model_name, WINDOW_INFLUENCE, PENALTY_K, LR, AO)
        # logging.getLogger().info(info)
        print(info)
        return AO
    elif args.dataset in ['UAV20L', 'UAV123_10fps']:
        if args.dataset == 'UAV20L':
            dataset_root = '/media/zxh/E404D23504D20B06/UAV_dataset/uav20L'
        if args.dataset == 'UAV123_10fps':
            dataset_root = '/media/zxh/E404D23504D20B06/UAV_dataset/uav123_10fps'
        for v_idx, video in enumerate(dataset):
            seed_torch(SEED)
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, img in enumerate(sorted(os.listdir(os.path.join(dataset_root, video)))):
                # print('({}) video:{} img:{}'.format(v_idx, video, img))
                img = cv2.imread(os.path.join(dataset_root, video, img))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gt_path = os.path.join(dataset_root, '../UAV123/anno/UAV20L', video + '.txt')
                gt_bbox = np.loadtxt(gt_path, delimiter=',')[0, :]
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - w / 2, cy - h / 2, w, h]
                    init_info = {'init_bbox': gt_bbox_}
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
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            toc /= cv2.getTickFrequency()
            # save results
            if not os.path.isdir(tracker_name):
                os.makedirs(tracker_name)
            result_path = os.path.join(tracker_name, '{}.txt'.format(video))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx + 1, video, toc, idx / toc))

        auc = eval(dataset=dataset_eval, tracker_name=tracker_name)
        info = "{:s} window_influence: {:1.17f}, penalty_k: {:1.17f}, scale_lr: {:1.17f}, AUC: {:1.3f}".format(
            model_name, WINDOW_INFLUENCE, PENALTY_K, LR, auc)
        logging.getLogger().info(info)
        print(info)
        return auc
    elif args.dataset == 'DTB70':
        dataset_root = '/media/zxh/E404D23504D20B06/UAV_dataset/DTB70'
        for v_idx, video in enumerate(dataset):
            seed_torch(SEED)
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, img in enumerate(sorted(os.listdir(os.path.join(dataset_root, video, 'img')))):
                # print('({}) video:{} img:{}'.format(v_idx, video, img))
                img = cv2.imread(os.path.join(dataset_root, video, 'img', img))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gt_path = os.path.join(dataset_root, video, video + '.txt')
                gt_bbox = np.loadtxt(gt_path, delimiter=',')[0, :]
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - w / 2, cy - h / 2, w, h]
                    init_info = {'init_bbox': gt_bbox_}
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
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            toc /= cv2.getTickFrequency()
            # save results
            if not os.path.isdir(tracker_name):
                os.makedirs(tracker_name)
            result_path = os.path.join(tracker_name, '{}.txt'.format(video))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx + 1, video, toc, idx / toc))

        auc = eval(dataset=dataset_eval, tracker_name=tracker_name)
        info = "{:s} window_influence: {:1.17f}, AUC: {:1.3f}".format(
            model_name, WINDOW_INFLUENCE, auc)
        logging.getLogger().info(info)
        print(info)
        return auc
    elif args.dataset == 'UAVDT':
        dataset_root = '/media/zxh/E404D23504D20B06/UAV_dataset/UAVDT'
        for v_idx, video in enumerate(dataset):
            seed_torch(SEED)
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, img in enumerate(sorted(os.listdir(os.path.join(dataset_root, video)))):
                # print('({}) video:{} img:{}'.format(v_idx, video, img))
                img = cv2.imread(os.path.join(dataset_root, video, img))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gt_path = os.path.join(dataset_root, '../../UAVDT_toolkit/anno', video+'_gt.txt')
                gt_bbox = np.loadtxt(gt_path, delimiter=',')[0, :]
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - w / 2, cy - h / 2, w, h]
                    init_info = {'init_bbox': gt_bbox_}
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
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            toc /= cv2.getTickFrequency()
            # save results
            if not os.path.isdir(tracker_name):
                os.makedirs(tracker_name)
            result_path = os.path.join(tracker_name, '{}.txt'.format(video))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx + 1, video, toc, idx / toc))

        auc = eval(dataset=dataset_eval, tracker_name=tracker_name)
        info = "{:s} window_influence: {:1.17f}, AUC: {:1.3f}".format(
            model_name, WINDOW_INFLUENCE, auc)
        logging.getLogger().info(info)
        print(info)
        return auc
    elif args.dataset == 'visdrone':
        dataset_root = '/media/zxh/E404D23504D20B06/UAV_dataset/VisDrone2018-SOT-test-dev'
        for v_idx, video in enumerate(dataset):
            seed_torch(SEED)
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, img in enumerate(sorted(os.listdir(os.path.join(dataset_root, 'sequences', video)))):
                # print('({}) video:{} img:{}'.format(v_idx, video, img))
                img = cv2.imread(os.path.join(dataset_root, 'sequences', video, img))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gt_path = os.path.join(dataset_root, 'annotations', video + '.txt')
                gt_bbox = np.loadtxt(gt_path, delimiter=',')[0, :]
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - w / 2, cy - h / 2, w, h]
                    init_info = {'init_bbox': gt_bbox_}
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
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            toc /= cv2.getTickFrequency()
            # save results
            if not os.path.isdir(tracker_name):
                os.makedirs(tracker_name)
            result_path = os.path.join(tracker_name, '{}.txt'.format(video))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx + 1, video, toc, idx / toc))

        auc = eval(dataset=dataset_eval, tracker_name=tracker_name)
        info = "{:s} window_influence: {:1.17f}, AUC: {:1.3f}".format(
            model_name, WINDOW_INFLUENCE, auc)
        logging.getLogger().info(info)
        print(info)
        return auc
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            seed_torch(SEED)
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    init_info = {'init_bbox': gt_bbox_}
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
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join(tracker_name, 'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                                           '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
                result_path = os.path.join(video_path,
                                           '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT10k' == args.dataset:
                video_path = os.path.join(tracker_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                if not os.path.isdir(tracker_name):
                    os.makedirs(tracker_name)
                result_path = os.path.join(tracker_name, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
            print('({:3d}) Video: {:17s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx + 1, video.name, toc, idx / toc))
        auc = eval(dataset=dataset_eval, tracker_name=tracker_name)
        info = "{:s} window_influence: {:1.17f},AUC: {:1.3f}".format(
            model_name, WINDOW_INFLUENCE, auc)
        logging.getLogger().info(info)
        print(info)
        return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tuning for siam')
    parser.add_argument('--dataset', default='visdrone', type=str, help='datasets,like [GOT10kVal,LaSOT,OTB,UAV123, NFS30]')
    parser.add_argument('--name', default='visdrone', type=str, help='name of results')
    parser.add_argument('--epoch', default='200', type=str, help='the epoch of model to tune')
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu id")
    parser.add_argument('--net_path', default='', type=str, help='net path')
    parser.add_argument('--min_wi', default=0.00, type=float, help='min window influence')
    parser.add_argument('--max_wi', default=0.90, type=float, help='max window influence')
    # parser.add_argument('--min_k', default=0.00, type=float, help='min penalty k')
    # parser.add_argument('--max_k', default=0.90, type=float, help='max penalty k')
    # parser.add_argument('--min_lr', default=0.00, type=float, help='min scale lr')
    # parser.add_argument('--max_lr', default=0.90, type=float, help='max scale lr')
    args = parser.parse_args()
    torch.set_num_threads(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = 12345
    dir = '/home/zxh/project/TransT'

    # Absolute path of the dataset
    # Absolute path of the model
    if args.dataset == 'GOT10kVal':
        dataset_root = '/media/zxh/30CC86E5CC86A526/SOT_train/GOT10k'
    elif 'OTB' in args.dataset:
        dataset_root = '/media/zxh/30CC86E5CC86A526/SOT_eval/OTB2015'
    elif 'UAV123' in args.dataset:
        dataset_root = '/media/zxh/30CC86E5CC86A526/SOT_eval/UAV123'
    elif 'NFS' in args.dataset:
        dataset_root = '/media/zxh/30CC86E5CC86A526/SOT_eval/Nfs'
    elif 'UAV20L' in args.dataset:
        dataset_root = '/media/zxh/E404D23504D20B06/UAV_dataset/uav20L'
    elif 'UAV123_10fps' in args.dataset:
        dataset_root = '/media/zxh/E404D23504D20B06/UAV_dataset/uav123_10fps'
    elif 'DTB70' == args.dataset:
        dataset_root = '/media/zxh/E404D23504D20B06/UAV_dataset/DTB70'
    elif 'visdrone' == args.dataset:
        dataset_root = '/media/zxh/E404D23504D20B06/UAV_dataset/VisDrone2018-SOT-test-dev'
    elif 'UAVDT' == args.dataset:
        dataset_root = '/media/zxh/E404D23504D20B06/UAV_dataset/UAVDT'

    # create model
    net = NetWithBackbone(net_path=args.net_path, use_gpu=True, backbone_pretrained=False)

    if args.dataset == 'GOT10kVal':
        dataset = dataset_root
    elif args.dataset in ['UAV20L', 'UAV123_10fps', 'DTB70', 'UAVDT']:
        dataset = os.listdir(dataset_root)
    elif args.dataset in ['visdrone']:
        dataset = os.listdir(os.path.join(dataset_root, 'sequences'))
    else:
        # create dataset
        dataset = DatasetFactory.create_dataset(name=args.dataset,
                                                dataset_root=dataset_root,
                                                load_img=False)

    model_name = args.name
    root = dataset_root
    if 'OTB' in args.dataset:
        dataset_eval = OTBDataset(args.dataset, root)
    elif 'LaSOT' == args.dataset:
        dataset_eval = LaSOTDataset(args.dataset, root)
    elif 'UAV123' in args.dataset:
        dataset_eval = UAVDataset(args.dataset, root)
    elif 'NFS' in args.dataset:
        dataset_eval = NFSDataset(args.dataset, root)
    if args.dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
        dataset_eval = VOTDataset(args.dataset, root)
    elif 'VOT2018-LT' == args.dataset:
        dataset_eval = VOTLTDataset(args.dataset, root)
    if 'UAV20L' == args.dataset:
        dataset_eval = 'UAV20L'
    if 'UAV123_10fps' == args.dataset:
        dataset_eval = 'UAV123_10fps'
    if 'DTB70' == args.dataset:
        dataset_eval = 'DTB70'
    if 'visdrone' == args.dataset:
        dataset_eval = 'visdrone'
    if 'UAVDT' == args.dataset:
        dataset_eval = 'UAVDT'

    # tune_result = os.path.join(dir, 'tune_results', args.dataset, args.epoch, 'wi_'+str(args.min_wi)+'-'+str(args.max_wi)+'k_'+str(args.min_k)+'-'+str(args.max_k)+'lr_'+str(args.min_lr)+'-'+str(args.max_lr))
    tune_result = os.path.join(dir, 'tune_results', args.dataset, args.epoch, 'wi_'+str(args.min_wi)+'-'+str(args.max_wi))
    if not os.path.isdir(tune_result):
        os.makedirs(tune_result)
    log_path = os.path.join(tune_result, model_name + '-'+ args.epoch + '.log')
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.FileHandler(log_path))
    optuna.logging.enable_propagation()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200)
    print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))
    joblib.dump(study, os.path.join(tune_result, model_name+'-'+args.epoch + '.pkl'))