import torch
from ltr.dataset import Lasot
from ltr.data import processing, sampler, LTRLoader
import ltr.models.tracking.siamcar as siam
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from ltr.models.loss.siamLoss import SigmoidCrossEntropyRetina, SigmoidCrossEntropyCenterness, IOULoss

def run(settings):
    # Most common settings are assigned in the settings struct
    # settings.device = 'cuda'
    settings.description = 'siamcar with default settings.'
    settings.batch_size = 8
    settings.num_workers = 1
    settings.multi_gpu = False
    settings.DDP = False
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 5.0
    settings.template_area_factor = 2.0
    settings.search_feature_sz = 20
    settings.template_feature_sz = 8
    settings.search_sz = settings.search_feature_sz * 16
    settings.temp_sz = settings.template_feature_sz * 16
    settings.center_jitter_factor = {'search': 2.5, 'template': 0}
    settings.scale_jitter_factor = {'search': 0.25, 'template': 0}
    settings.scheduler_type = 'step'
    settings.output_sigma_factor = 1 / 4
    settings.score_size = 13

    #hyper-parameter
    settings.data_max_gap = 200
    settings.data_sample_epoch = 60000
    settings.data_num_template = 1
    settings.data_num_search = 1
    settings.frame_sample_mode='causal'
    settings.response_map = 13
    settings.neg_sample = 0.2

    # Train datasets
    lasot_train = Lasot(settings.env.lasot_dir, split='train')

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),)

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip(probability=0.5),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # Data processing to do on the training pairs
    data_processing_train = processing.siamCARProcessing(search_area_factor=settings.search_area_factor,
                                                      template_area_factor = settings.template_area_factor,
                                                      search_sz=settings.search_sz,
                                                      temp_sz=settings.temp_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      sigma=settings.output_sigma_factor / settings.search_area_factor,
                                                      score_size=settings.score_size,
                                                      mode='sequence',
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)

    # The sampler for training
    dataset_train = sampler.ESTrackSampler([lasot_train], [1],
                                        samples_per_epoch=settings.data_sample_epoch, max_gap=settings.data_max_gap,
                                        num_template_frames=settings.data_num_template, num_search_frames=settings.data_num_search,processing=data_processing_train,
                                        frame_sample_mode=settings.frame_sample_mode,response_map=settings.response_map,
                                        neg=settings.neg_sample)

    # The loader for training
    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size,
                             num_workers=settings.num_workers, shuffle=shuffle, drop_last=True, stack_dim=0, sampler=train_sampler)

    if settings.local_rank != -1:
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    # Create network and actor
    model = siam.siamcar_resnet50(backbone_pretrained=True, head_name="siamCAR")
    model = model.to(settings.device)
    print(model)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        model = MultiGPU(model, dim=0)
    if settings.local_rank != -1:
        model = DDP(model, device_ids=[settings.local_rank], output_device=settings.local_rank, find_unused_parameters=True)

    objective = {'clsloss':SigmoidCrossEntropyRetina(), 'centerloss':SigmoidCrossEntropyCenterness(), 'iouloss':IOULoss()}
    loss_weight = {'loc_weight':1.0, 'cls_weight':1.0, 'cen_weight':1.0}
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    actor = actors.ESTrackActor(net=model, objective=objective, loss_weight=loss_weight)

    # Optimizer
    param_dicts = [
        {"params": filter(lambda x: x.requires_grad, model.backbone.parameters()), "lr":1e-4},
        {"params": model.neck_z_cls.parameters()},
        {"params": model.neck_z_loc.parameters()},
        {"params": model.neck_x_cls.parameters()},
        {"params": model.neck_x_loc.parameters()},
        {"params": model.head.parameters()}]

    optimizer = torch.optim.AdamW(param_dicts, lr=1e-3,
                                  weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [160], gamma=0.1)

    # Create trainer
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(400, load_latest=True, fail_safe=True)
