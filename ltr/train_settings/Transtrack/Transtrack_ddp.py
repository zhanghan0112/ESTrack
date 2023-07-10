import torch
from ltr.dataset import Lasot, MSCOCOSeq, Got10k, TrackingNet
from ltr.data import processing, sampler, LTRLoader
import ltr.models.tracking.transtrack as transtrack
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from torch.nn.parallel import DistributedDataParallel as DDP
from util.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.utils.data.distributed import DistributedSampler


def run(settings):
    # Most common settings are assigned in the settings struct
    settings.description = 'TransT with default settings.'
    settings.batch_size = 64
    settings.num_workers = 4
    settings.DDP = True
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 4.0
    settings.template_area_factor = 2.0
    settings.search_feature_sz = 16
    settings.template_feature_sz = 8
    settings.search_sz = settings.search_feature_sz * 16
    settings.temp_sz = settings.template_feature_sz * 16
    settings.center_jitter_factor = {'search': 3, 'template': 0}
    settings.scale_jitter_factor = {'search': 0.25, 'template': 0}

    # Transformer
    settings.position_embedding = 'sine'
    settings.hidden_dim = 128
    settings.dropout = 0.1
    settings.nheads = 8
    settings.dim_feedforward = 1024
    settings.enc_layers = 4
    settings.dec_layers = 4
    settings.head_type = 'corner_predictor'
    settings.num_queries = 1
    settings.num_feature_levels = 1
    settings.with_bbox_refine = False
    settings.giou_weight = 2.0
    settings.l1_weight = 5.0
    settings.num_queris = 1
    settings.scheduler_type = 'step'

    # Train datasets
    lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
    trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))
    coco_train = MSCOCOSeq(settings.env.coco_dir)
    got10k_val = Got10k(settings.env.got10k_dir, split='votval')

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # Data processing to do on the training pairs
    data_processing_train = processing.TransTProcessing(search_area_factor=settings.search_area_factor,
                                                      template_area_factor = settings.template_area_factor,
                                                      search_sz=settings.search_sz,
                                                      temp_sz=settings.temp_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)

    data_processing_val = processing.TransTProcessing(search_area_factor=settings.search_area_factor,
                                                      template_area_factor=settings.template_area_factor,
                                                      search_sz=settings.search_sz,
                                                      temp_sz=settings.temp_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      transform=transform_val,
                                                      joint_transform=transform_joint
                                                      )

    # The sampler for training
    dataset_train = sampler.TransTSampler([lasot_train, got10k_train, coco_train, trackingnet_train], [1,1,1,1],
                                samples_per_epoch=60000, max_gap=100, processing=data_processing_train)

    dataset_val = sampler.TransTSampler([got10k_val], [1], samples_per_epoch=20000, max_gap=100,
                                        processing=data_processing_val)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True
    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=shuffle, drop_last=True, stack_dim=0, sampler=train_sampler)

    loder_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size,
                          num_workers=settings.num_workers,
                          shuffle=shuffle, drop_last=True, stack_dim=0, sampler=val_sampler, epoch_interval=50)

    # Create network and actor
    if settings.local_rank != -1:
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    model = transtrack.transtrack_resnet50(settings)
    model = model.to(settings.device)
    print(model)

    # Wrap the network for multi GPU training
    if settings.local_rank != -1:
        model = DDP(model, device_ids=[settings.local_rank], output_device=settings.local_rank, find_unused_parameters=True)

    objective = {'giou': giou_loss, 'l1':l1_loss}
    loss_weight = {'giou': settings.giou_weight, 'l1': settings.l1_weight}
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    actor = actors.TranstrackActor(net=model, objective=objective, loss_weight=loss_weight, settings=settings)

    # Optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-5,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=1e-4,
                                  weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 400)

    # Create trainer
    trainer = LTRTrainer(actor, [loader_train, loder_val], optimizer, settings, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(500, load_latest=True, fail_safe=True)
