import os
import sys
prj_apth = os.path.join(os.path.dirname((__file__)), '../..')
if prj_apth not in sys.path:
    sys.path.append(prj_apth)
import argparse
from pytracking.analysis.plot_results import print_results
from pytracking.evaluation import trackerlist, get_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='parse args for analysis')
    parser.add_argument('--script', default='transconv', type=str, help='training script name')
    parser.add_argument('--config', default='transconv', type=str, help='yaml configure file name')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    trackers = []
    trackers.extend(trackerlist(args.script, args.config, None, None))
    dataset = get_dataset('lasot')
    print_results(trackers, dataset, 'lasot', merge_results=False, plot_types=('success', 'prec', 'norm_prec'))