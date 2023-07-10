from got10k_toolkit.toolkit.experiments import ExperimentUAV123
from got10k_toolkit.toolkit.trackers.identity_tracker import IdentityTracker
from got10k_toolkit.toolkit.trackers.net_wrappers import NetWithBackbone

#Specify the path
net_path = '' #Absolute path of the model
dataset_root= '/media/zxh/E404D23504D20B06/UAV123' #Absolute path of the datasets

#TransT
net = NetWithBackbone(net_path=net_path, use_gpu=True)
tracker = IdentityTracker(name='transt', net=net, window_penalty=0.49, exemplar_size=128, instance_size=256)

#Test
experiment = ExperimentUAV123(
    root_dir=dataset_root,  # UAV123's root directory
    result_dir='/home/zxh/project/TransT/results',  # where to store tracking results
    report_dir='reports'  # where to store evaluation reports
)
experiment.report(['siamcar_wi-0.413_pk-0.606_lr-0.579_200_pix_685'])
