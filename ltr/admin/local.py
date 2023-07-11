class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/media/zxh/30CC86E5CC86A526/LaSOT'
        self.got10k_dir = '/media/zxh/30CC86E5CC86A526/SOT_train/GOT-10k/train'
        self.trackingnet_dir = '/media/zxh/30CC86E5CC86A526/SOT_train/TrackingNet'
        self.coco_dir = '/media/zxh/30CC86E5CC86A526/SOT_train/coco2017'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = '/media/zxh/30CC86E5CC86A526/VOS/DAVIS'
        self.youtubevos_dir = '/media/zxh/30CC86E5CC86A526/YoutubeVOS2019'
