import os
import sys
import argparse
import importlib
import multiprocessing
import cv2 as cv
import torch.backends.cudnn
import torch.distributed as dist


env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
import ltr.admin.settings as ws_settings


def run_training(train_module, train_name, cudnn_benchmark=True, local_rank=-1):
    """Run a train scripts in train_settings.
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """

    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    print('Training:  {}  {}'.format(train_module, train_name))

    settings = ws_settings.Settings()
    settings.module_name = train_module
    settings.script_name = train_name
    settings.local_rank = local_rank
    settings.project_path = 'ltr/{}/{}'.format(train_module, train_name)

    expr_module = importlib.import_module('ltr.train_settings.{}.{}'.format(train_module, train_name))
    expr_func = getattr(expr_module, 'run')

    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('train_module', type=str, help='Name of module in the "train_settings/" folder.')
    parser.add_argument('train_name', type=str, help='Name of the train settings file.')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training.')
    args = parser.parse_args()

    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(0)

    run_training(args.train_module, args.train_name, args.cudnn_benchmark, args.local_rank)


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn', force=True)
    main()
