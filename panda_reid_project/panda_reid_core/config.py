# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import torch
import yaml
from yacs.config import CfgNode as CN

# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Optional non-Swin backbone selector (used when MODEL.TYPE in {timm, convnext, resnet, efficientnet, osnet})
_C.MODEL.BACKBONE_NAME = 'convnextv2_base.fcmae_ft_in22k_in1k'
_C.MODEL.BACKBONE_PRETRAINED = True
# Optional local checkpoint path for timm backbone (e.g. modelori/pytorch_model.bin).
# When provided and exists, it is loaded first and online download is skipped.
_C.MODEL.BACKBONE_WEIGHTS = ''
# Pooling strategy for timm backbones: avg / avgmax / gem
_C.MODEL.POOL_TYPE = 'avgmax'
# Enable horizontal part pooling for better open-world clustering
_C.MODEL.PART_POOL_ENABLE = False
_C.MODEL.PART_NUM_PARTS = 3

# External comparison backbones
_C.MODEL.TRANSREID_TYPE = 'vit_base_patch16_224_TransReID'
_C.MODEL.TRANSREID_STRIDE_SIZE = [12, 12]
_C.MODEL.MEGE_GRAPH_VERTICES = 64
_C.MODEL.MEGE_GRAPH_SIGMA = 2.0
_C.MODEL.MEGE_BLEND = 0.35

# Optional: attention dropout (if not set in YAML, code can fall back to DROP_RATE)
_C.MODEL.ATTN_DROP_RATE = 0.0

# Optional: feature projection head (bottleneck) for better generalization
# - 0 means disabled (use backbone feature_dim)
_C.MODEL.PROJ_DIM = 0
_C.MODEL.PROJ_DROP = 0.0

# Optional: decouple auxiliary task gradients from ReID embedding
_C.MODEL.AUX_DETACH = False
# Auxiliary-task gradient ratio to shared embedding:
# 1.0 = full gradient, 0.0 = fully detached (same effect as AUX_DETACH=True)
_C.MODEL.AUX_GRAD_RATIO = 1.0

# Optional: train prototype_net (quality/confidence) with an auxiliary loss
# - Only affects prototype_net's internal heads; ReID backbone features are detached for this loss.
_C.MODEL.PROTO_AUX_WEIGHT = 0.05
_C.MODEL.PROTO_CONF_WEIGHT = 1.0
_C.MODEL.PROTO_QUAL_WEIGHT = 0.5
_C.MODEL.PROTO_LR_MULT = 10.0

# Metric-learning auxiliary contrastive term
_C.MODEL.SUPCON_WEIGHT = 0.0
_C.MODEL.SUPCON_TEMP = 0.07

# Training-time multi-prototype metric memory
_C.MODEL.MULTI_PROTO_TRAIN_WEIGHT = 0.0
_C.MODEL.MULTI_PROTO_TEMP = 0.07
_C.MODEL.MULTI_PROTO_HARD_NEG_K = 32
_C.MODEL.MULTI_PROTO_MAX_SLOTS = 4
_C.MODEL.MULTI_PROTO_MOMENTUM = 0.9
_C.MODEL.MULTI_PROTO_SPAWN_THRESHOLD = 0.55
_C.MODEL.MULTI_PROTO_UPDATE_THRESHOLD = 0.45
_C.MODEL.MULTI_PROTO_BOOTSTRAP_BATCHES = 8

# Training-time continual metric consistency loss against historical prototypes
_C.MODEL.CONTINUAL_METRIC_WEIGHT = 0.0
_C.MODEL.CONTINUAL_METRIC_MARGIN = 0.15
_C.MODEL.CONTINUAL_METRIC_HARD_NEG_K = 16
_C.MODEL.CONTINUAL_METRIC_STABILITY_WEIGHT = 0.5

# Dynamic metric topology modeling
_C.MODEL.DYNAMIC_TOPO_WEIGHT = 0.0
_C.MODEL.DYNAMIC_TOPO_TEMP = 0.10
_C.MODEL.DYNAMIC_TOPO_NEG_MARGIN = 0.15
_C.MODEL.DYNAMIC_TOPO_PULL_WEIGHT = 0.5

# Uncertainty-aware robust topology preservation and feature purification
_C.MODEL.UNCERTAINTY_TOPO_WEIGHT = 0.0
_C.MODEL.UNCERTAINTY_PURIFY_BLEND = 0.35
_C.MODEL.UNCERTAINTY_MARGIN = 0.10

# Few-shot meta-metric fast adaptation
_C.MODEL.META_TOPO_WEIGHT = 0.0
_C.MODEL.META_TOPO_SUPPORT_SHOTS = 1
_C.MODEL.META_TOPO_QUERY_MAX = 2
_C.MODEL.META_TOPO_ADAPT_BLEND = 0.35
_C.MODEL.META_TOPO_TEMP = 0.07

# Dynamic topology incremental learning stability
_C.MODEL.INCREMENTAL_TOPO_WEIGHT = 0.0
_C.MODEL.INCREMENTAL_TOPO_TEMP = 0.10
_C.MODEL.INCREMENTAL_TOPO_SLOT_RADIUS = 0.18
_C.MODEL.INCREMENTAL_TOPO_ENTROPY_WEIGHT = 0.5
_C.MODEL.INCREMENTAL_TOPO_CENTROID_WEIGHT = 0.5
_C.MODEL.INCREMENTAL_TOPO_SLOT_WEIGHT = 0.25

# Memory-bank update gates for robust prototype topology
_C.MODEL.MULTI_PROTO_SPAWN_MIN_QUALITY = 0.0
_C.MODEL.MULTI_PROTO_CLASS_CENTROID_GUARD = 0.0
_C.MODEL.MULTI_PROTO_UPDATE_MIN_QUALITY = 0.0

# Open-world gate tuning (prototype assignment/new-ID decision)
_C.MODEL.OW_BASE_THRESHOLD_NO_STATS = 0.50
_C.MODEL.OW_BASE_THRESHOLD_WITH_STATS = 0.45
_C.MODEL.OW_QUALITY_ADJUST_SCALE = 0.08
_C.MODEL.OW_DEVIATION_ADJUST_SCALE = 0.05
_C.MODEL.OW_DEVIATION_ADJUST_CAP = 0.12
_C.MODEL.OW_CONFIDENCE_THRESHOLD = 0.55
_C.MODEL.OW_QUALITY_THRESHOLD = 0.35
_C.MODEL.OW_AMBIGUOUS_MARGIN = 0.03
_C.MODEL.OW_AMBIGUOUS_OFFSET = 0.02

# Optional post-processing on predicted clusters during evaluation
_C.MODEL.EVAL_SMALL_CLUSTER_MAX = 0
_C.MODEL.EVAL_MERGE_SIM_THR = 0.999

# Validation protocol (aligned with evaluate_openworld_with_specialists.py)
_C.MODEL.EVAL_MAX_IMAGES = 0                 # 0 means evaluate all test images
_C.MODEL.EVAL_CLUSTER_THRESHOLD = 0.24       # used when threshold sweep is empty
_C.MODEL.EVAL_THRESHOLD_SWEEP = []           # e.g. [0.22, 0.23, 0.24, 0.25]
_C.MODEL.EVAL_PROTOTYPE_MOMENTUM = 0.9

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True

# Swin Transformer V2 parameters
_C.MODEL.SWINV2 = CN()
_C.MODEL.SWINV2.PATCH_SIZE = 4
_C.MODEL.SWINV2.IN_CHANS = 3
_C.MODEL.SWINV2.EMBED_DIM = 96
_C.MODEL.SWINV2.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWINV2.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWINV2.WINDOW_SIZE = 7
_C.MODEL.SWINV2.MLP_RATIO = 4.
_C.MODEL.SWINV2.QKV_BIAS = True
_C.MODEL.SWINV2.APE = False
_C.MODEL.SWINV2.PATCH_NORM = True
_C.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]

# ReID specific parameters
_C.MODEL.NECK_FEAT = 'after'  # 'before' or 'after'
_C.MODEL.LOSS_TYPE = 'combined'  # 'arcface', 'triplet', 'combined'
_C.MODEL.ARCFACE_SCALE = 30.0
_C.MODEL.ARCFACE_MARGIN = 0.5
_C.MODEL.ARCFACE_WEIGHT = 1.0
_C.MODEL.TRIPLET_MARGIN = 0.3
_C.MODEL.TRIPLET_WEIGHT = 1.0
_C.MODEL.HARD_MINING = True

# Center Loss parameters
_C.MODEL.CENTER_WEIGHT = 0.0

# Variance Loss parameters
_C.MODEL.VARIANCE_WEIGHT = 0.0

# Auxiliary task parameters
_C.MODEL.GENDER_WEIGHT = 0.3
_C.MODEL.GENDER_LABEL_SMOOTHING = 0.0
_C.MODEL.AGE_WEIGHT = 0.1
_C.MODEL.AGE_LOSS_BETA = 1.0

# Auxiliary task imbalance handling (rare ages / gender imbalance)
# - Gender: optional class-weighted CE
_C.MODEL.GENDER_USE_CLASS_WEIGHTS = False
_C.MODEL.GENDER_CLASS_WEIGHT_POWER = 0.5   # 0.5=sqrt inverse freq; 1.0=inverse freq
_C.MODEL.GENDER_CLASS_WEIGHT_CLAMP = 5.0   # cap to avoid exploding loss on extreme imbalance
# - Age: optional bin-weighted SmoothL1 (Huber)
_C.MODEL.AGE_USE_BIN_WEIGHTS = False
_C.MODEL.AGE_BIN_SIZE = 2.0               # years per bin (2 is a good default for sparse ages)
_C.MODEL.AGE_BIN_WEIGHT_POWER = 0.5       # 0.5=sqrt inverse freq; 1.0=inverse freq
_C.MODEL.AGE_BIN_WEIGHT_CLAMP = 5.0       # cap to avoid exploding loss on extreme sparsity

# Swin Transformer MoE parameters
_C.MODEL.SWIN_MOE = CN()
_C.MODEL.SWIN_MOE.PATCH_SIZE = 4
_C.MODEL.SWIN_MOE.IN_CHANS = 3
_C.MODEL.SWIN_MOE.EMBED_DIM = 96
_C.MODEL.SWIN_MOE.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MOE.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MOE.WINDOW_SIZE = 7
_C.MODEL.SWIN_MOE.MLP_RATIO = 4.
_C.MODEL.SWIN_MOE.QKV_BIAS = True
_C.MODEL.SWIN_MOE.QK_SCALE = None
_C.MODEL.SWIN_MOE.APE = False
_C.MODEL.SWIN_MOE.PATCH_NORM = True
_C.MODEL.SWIN_MOE.MLP_FC2_BIAS = True
_C.MODEL.SWIN_MOE.INIT_STD = 0.02
_C.MODEL.SWIN_MOE.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
_C.MODEL.SWIN_MOE.MOE_BLOCKS = [[-1], [-1], [-1], [-1]]
_C.MODEL.SWIN_MOE.NUM_LOCAL_EXPERTS = 1
_C.MODEL.SWIN_MOE.TOP_VALUE = 1
_C.MODEL.SWIN_MOE.CAPACITY_FACTOR = 1.25
_C.MODEL.SWIN_MOE.COSINE_ROUTER = False
_C.MODEL.SWIN_MOE.NORMALIZE_GATE = False
_C.MODEL.SWIN_MOE.USE_BPR = True
_C.MODEL.SWIN_MOE.IS_GSHARD_LOSS = False
_C.MODEL.SWIN_MOE.GATE_NOISE = 1.0
_C.MODEL.SWIN_MOE.COSINE_ROUTER_DIM = 256
_C.MODEL.SWIN_MOE.COSINE_ROUTER_INIT_T = 0.5
_C.MODEL.SWIN_MOE.MOE_DROP = 0.0
_C.MODEL.SWIN_MOE.AUX_LOSS_WEIGHT = 0.01

# Swin MLP parameters
_C.MODEL.SWIN_MLP = CN()
_C.MODEL.SWIN_MLP.PATCH_SIZE = 4
_C.MODEL.SWIN_MLP.IN_CHANS = 3
_C.MODEL.SWIN_MLP.EMBED_DIM = 96
_C.MODEL.SWIN_MLP.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MLP.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MLP.WINDOW_SIZE = 7
_C.MODEL.SWIN_MLP.MLP_RATIO = 4.
_C.MODEL.SWIN_MLP.APE = False
_C.MODEL.SWIN_MLP.PATCH_NORM = True

# [SimMIM] Norm target during training
_C.MODEL.SIMMIM = CN()
_C.MODEL.SIMMIM.NORM_TARGET = CN()
_C.MODEL.SIMMIM.NORM_TARGET.ENABLE = False
_C.MODEL.SIMMIM.NORM_TARGET.PATCH_SIZE = 47

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# Resume behavior controls
_C.TRAIN.RESET_EPOCH_ON_RESUME = False
_C.TRAIN.RESET_OPTIMIZER_ON_RESUME = False
_C.TRAIN.RESET_LR_SCHEDULER_ON_RESUME = False
_C.TRAIN.RESET_BEST_ON_RESUME = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []
# Restart epochs for cosine annealing
_C.TRAIN.LR_SCHEDULER.RESTART_EPOCHS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0
# RandomIdentitySampler controls
_C.TRAIN.SAMPLER_USE_CEIL = False
_C.TRAIN.SAMPLER_MAX_GROUPS_PER_ID = 0
_C.TRAIN.SAMPLER_MIN_GROUPS_PER_ID = 1

# MoE
_C.TRAIN.MOE = CN()
# Only save model on master device
_C.TRAIN.MOE.SAVE_MASTER = False
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'
# Use random resized crop instead of plain resize for train transform
_C.AUG.RRC_ENABLE = False
# Min scale for random resized crop (max scale is always 1.0)
_C.AUG.RRC_SCALE_MIN = 0.6
# Random grayscale probability
_C.AUG.GRAY_PROB = 0.0
# Random gaussian blur probability
_C.AUG.BLUR_PROB = 0.0

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Inference settings
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.SIMILARITY_THRESHOLD = 0.75
_C.INFERENCE.MAX_FEATURES_PER_ID = 50
_C.INFERENCE.DATABASE_PATH = 'panda_id_database.pkl'

# Similarity matching network configuration
_C.INFERENCE.SIMILARITY_NET = CN()
_C.INFERENCE.SIMILARITY_NET.ENABLE = False
_C.INFERENCE.SIMILARITY_NET.HIDDEN_DIM = 512
_C.INFERENCE.SIMILARITY_NET.PRETRAINED = ''

# -----------------------------------------------------------------------------
# ROI processing settings
# -----------------------------------------------------------------------------
_C.ROI = CN()
_C.ROI.EXPAND_RATIO = 0.1
_C.ROI.MIN_SIZE = 32

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# [SimMIM] Whether to enable pytorch amp, overwritten by command line argument
_C.ENABLE_AMP = False

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# for acceleration
_C.FUSED_WINDOW_PROCESS = False
_C.FUSED_LAYERNORM = False

# Distributed training settings
_C.WORLD_SIZE = 1
_C.RANK = 0
_C.DIST_URL = 'tcp://127.0.0.1:23456'
_C.DIST_BACKEND = 'nccl'


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r', encoding='utf-8') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    #
    try:
        config.merge_from_file(cfg_file)
    except UnicodeDecodeError:
        #
        with open(cfg_file, 'r', encoding='utf-8') as f:
            yaml_content = yaml.load(f, Loader=yaml.FullLoader)
        config.merge_from_other_cfg(CN(yaml_content))
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_opt_level'):
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == 'O0':
            config.AMP_ENABLE = False
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True

    # [SimMIM]
    if _check_args('enable_amp'):
        config.ENABLE_AMP = args.enable_amp

    # for acceleration
    if _check_args('fused_window_process'):
        config.FUSED_WINDOW_PROCESS = True
    if _check_args('fused_layernorm'):
        config.FUSED_LAYERNORM = True
    ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    # set local rank for distributed training (robust to missing args / torch version)
    lr = None
    if hasattr(args, 'local_rank'):
        lr = getattr(args, 'local_rank')
    else:
        lr = os.environ.get('LOCAL_RANK', None)
    try:
        config.LOCAL_RANK = int(lr) if lr is not None else 0
    except Exception:
        config.LOCAL_RANK = 0  # Default to 0 for single GPU training

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
