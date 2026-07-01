#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" """

import os
import sys
import time
import argparse
import datetime
import heapq
import random
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
import torch.amp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter
import seaborn as sns
import re
import cv2
from PIL import Image
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using fallback for optimal matching")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from panda_reid_core.config import get_config
from panda_reid_core.data.panda_dataset import PandaDataset, build_panda_transform
from panda_reid_core.models.panda_reid_model import build_panda_reid_model
from panda_reid_core.data.roiimg_dataset import PandaRoiImgDataset
from panda_reid_core.models.prototype_reid_network import PrototypeReIDNetwork, PrototypeLoss
from panda_reid_core.models.losses import CombinedLoss
from panda_reid_core.models.open_world_metrics import compute_open_world_cluster_metrics
from panda_reid_core.models.training_multi_prototype import (
    TrainingMultiPrototypeMemory,
    compute_multi_prototype_metric_loss,
    compute_continual_metric_consistency_loss,
    compute_dynamic_topology_loss,
    compute_uncertainty_topology_purification_loss,
    compute_meta_fewshot_topology_loss,
    compute_incremental_topology_stability_loss,
)
from panda_reid_core.optimizer import build_optimizer
from panda_reid_core.lr_scheduler import build_scheduler
from panda_reid_core.logger import create_logger
from panda_reid_core.utils import NativeScalerWithGradNormCount
import json


def _path_for_write(path: str) -> str:
    """ """
    if os.name != "nt":
        return path

    abs_path = os.path.normpath(os.path.abspath(path))
    #
    if len(abs_path) < 250:
        return abs_path

    if abs_path.startswith("\\\\?\\"):
        return abs_path
    if abs_path.startswith("\\\\"):
        # UNC: \\server\share\path -> \\?\UNC\server\share\path
        return "\\\\?\\UNC\\" + abs_path.lstrip("\\")
    return "\\\\?\\" + abs_path


def parse_option():
    """ """
    parser = argparse.ArgumentParser('Prototype ReID training script')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument('--image-root', type=str, required=False, help='train image root')
    parser.add_argument('--roi-root', type=str, required=False, help='train ROI root')
    parser.add_argument('--roiimg-root', type=str, required=False, help='train ROI image root')
    parser.add_argument('--test-roiimg-root', type=str, required=False, help='test ROI image root')
    parser.add_argument('--test-image-root', type=str, help='test image root')
    parser.add_argument('--test-roi-root', type=str, help='test ROI root')
    parser.add_argument('--output', type=str, default='output_prototype', help='output directory')
    parser.add_argument('--resume', type=str, help='resume checkpoint path')

    parser.add_argument(
        '--roi-format',
        type=str,
        default='mask',
        choices=['yolo', 'mask', 'auto'],
        help='ROI data format',
    )
    parser.add_argument('--mask-root', type=str, help='mask root directory')

    parser.add_argument('--eval-interval', type=int, default=5, help='evaluate every N epochs')
    parser.add_argument('--save-interval', type=int, default=5, help='save checkpoint every N epochs')

    parser.add_argument(
        '--num-instances',
        type=int,
        default=4,
        help='K in PK sampling (images per identity in one batch)',
    )

    parser.add_argument(
        '--opts',
        help="Modify config options by adding 'KEY VALUE' pairs",
        default=[],
        nargs='+',
    )

    args, _ = parser.parse_known_args()
    config = get_config(args)
    return args, config


class RandomIdentitySampler(Sampler):
    """PK sampler with optional per-identity group cap for long-tail datasets."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_instances: int,
        max_groups_per_pid: int = 0,
        min_groups_per_pid: int = 1,
        use_ceil_groups: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_instances <= 0:
            raise ValueError(f"num_instances must be positive, got {num_instances}")
        if batch_size % num_instances != 0:
            raise ValueError(
                f"batch_size({batch_size}) must be divisible by num_instances({num_instances})"
            )

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_instances = int(num_instances)
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.max_groups_per_pid = max(0, int(max_groups_per_pid or 0))
        self.min_groups_per_pid = max(1, int(min_groups_per_pid or 1))
        self.use_ceil_groups = bool(use_ceil_groups)

        self.index_dic = defaultdict(list)
        labels = self._extract_labels(dataset)
        for index, pid in enumerate(labels):
            self.index_dic[int(pid)].append(index)

        self.pids = list(self.index_dic.keys())
        if len(self.pids) == 0:
            raise ValueError('RandomIdentitySampler: dataset has 0 identities')
        if len(self.pids) < self.num_pids_per_batch:
            raise ValueError(
                f"RandomIdentitySampler: dataset identities({len(self.pids)}) < "
                f"num_pids_per_batch({self.num_pids_per_batch}). "
                f"Please reduce batch_size or num_instances."
            )

        self.group_counts = {}
        for pid in self.pids:
            num = len(self.index_dic[pid])
            if self.use_ceil_groups:
                group_count = max(1, int(math.ceil(float(num) / float(self.num_instances))))
            else:
                group_count = 1 if num < self.num_instances else num // self.num_instances
            if self.max_groups_per_pid > 0:
                group_count = min(group_count, self.max_groups_per_pid)
            group_count = max(self.min_groups_per_pid, group_count)
            self.group_counts[pid] = int(group_count)

        heap = [(-cnt, int(pid)) for pid, cnt in self.group_counts.items() if cnt > 0]
        heapq.heapify(heap)
        num_batches = 0
        while len(heap) >= self.num_pids_per_batch:
            picked = []
            for _ in range(self.num_pids_per_batch):
                cnt, pid = heapq.heappop(heap)
                picked.append((cnt, pid))
            for cnt, pid in picked:
                remaining = -cnt - 1
                if remaining > 0:
                    heapq.heappush(heap, (-remaining, pid))
            num_batches += 1

        self.length = int(num_batches * self.batch_size)

    @staticmethod
    def _extract_labels(dataset):
        if not hasattr(dataset, 'samples') or not hasattr(dataset, 'id_to_label'):
            raise ValueError(
                'RandomIdentitySampler requires dataset.samples and dataset.id_to_label'
            )

        labels = []
        for s in dataset.samples:
            if not isinstance(s, dict):
                raise ValueError('dataset.samples must be a list of dict')
            if 'id_name' in s:
                pid = dataset.id_to_label.get(s['id_name'])
            elif 'id' in s:
                pid = dataset.id_to_label.get(s['id'])
            else:
                raise ValueError("Unsupported sample dict: missing 'id_name' or 'id'")
            if pid is None:
                raise ValueError('Failed to map sample id to label')
            labels.append(int(pid))
        return labels

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            pool = list(self.index_dic[pid])
            target_groups = int(self.group_counts.get(pid, 0))
            if target_groups <= 0:
                continue
            need = target_groups * self.num_instances
            random.shuffle(pool)
            if len(pool) >= need:
                sampled = pool[:need]
            else:
                sampled = pool[:]
                sampled.extend(random.choices(pool, k=need - len(pool)))
            random.shuffle(sampled)
            batch_idxs_dict[pid] = [
                sampled[i: i + self.num_instances]
                for i in range(0, need, self.num_instances)
            ]

        heap = [
            (-len(batch_idxs_dict[pid]), random.random(), int(pid))
            for pid in self.pids
            if len(batch_idxs_dict[pid]) > 0
        ]
        heapq.heapify(heap)
        final_idxs = []
        while len(heap) >= self.num_pids_per_batch:
            picked = []
            for _ in range(self.num_pids_per_batch):
                cnt, tie, pid = heapq.heappop(heap)
                picked.append((cnt, tie, pid))

            for cnt, _tie, pid in picked:
                batch = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch)
                remaining = -cnt - 1
                if remaining > 0:
                    heapq.heappush(heap, (-remaining, random.random(), pid))

        if self.length > 0:
            final_idxs = final_idxs[: self.length]
        return iter(final_idxs)

    def __len__(self):
        return self.length


def compute_prototype_aux_loss(
    prototype_net: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    conf_weight: float = 1.0,
    qual_weight: float = 0.5,
):
    """ """
    device = features.device
    labels = labels.long()

    if features.dim() != 2:
        raise ValueError(f"features must be 2D [B, D], got {tuple(features.shape)}")
    if labels.dim() != 1:
        labels = labels.view(-1)

    bsz, feat_dim = features.shape
    if bsz < 2:
        zero = torch.tensor(0.0, device=device)
        return zero, {'conf_loss': zero, 'qual_loss': zero}

    uniq, inv = torch.unique(labels, sorted=True, return_inverse=True)  # inv: [B] in [0, P)
    num_pids = int(uniq.numel())
    if num_pids < 2:
        zero = torch.tensor(0.0, device=device)
        return zero, {'conf_loss': zero, 'qual_loss': zero}

    #
    sums = torch.zeros((num_pids, feat_dim), device=device, dtype=features.dtype)
    sums.index_add_(0, inv, features)
    counts = torch.zeros((num_pids,), device=device, dtype=features.dtype)
    counts.index_add_(0, inv, torch.ones((bsz,), device=device, dtype=features.dtype))
    proto_means = sums / counts.unsqueeze(1).clamp(min=1.0)
    proto_means = F.normalize(proto_means, p=2, dim=1)

    #
    sum_per_sample = sums[inv]  # [B, D]
    count_per_sample = counts[inv].unsqueeze(1)  # [B, 1]
    pos_proto = torch.empty_like(features)
    multi_mask = (count_per_sample.squeeze(1) > 1.5)
    pos_proto[multi_mask] = (sum_per_sample[multi_mask] - features[multi_mask]) / (
        count_per_sample[multi_mask] - 1.0
    )
    pos_proto[~multi_mask] = features[~multi_mask]
    pos_proto = F.normalize(pos_proto, p=2, dim=1)

    #
    rand = torch.randint(low=0, high=num_pids - 1, size=(bsz,), device=device)
    neg_group = rand + (rand >= inv).long()
    neg_proto = proto_means[neg_group]

    #
    conf_pos_logits = prototype_net.confidence_net(torch.cat([features, pos_proto], dim=1)).squeeze(1).float()
    conf_neg_logits = prototype_net.confidence_net(torch.cat([features, neg_proto], dim=1)).squeeze(1).float()
    conf_loss = 0.5 * (
        F.binary_cross_entropy_with_logits(conf_pos_logits, torch.ones_like(conf_pos_logits))
        + F.binary_cross_entropy_with_logits(conf_neg_logits, torch.zeros_like(conf_neg_logits))
    )

    #
    qual_pred = prototype_net.quality_net(features).squeeze(1).float()
    cos_sim = (features * pos_proto).sum(dim=1).float()  # [-1,1] for normalized
    qual_tgt = ((cos_sim + 1.0) / 2.0).clamp(0.0, 1.0).detach()
    qual_loss = F.smooth_l1_loss(qual_pred, qual_tgt)

    total = float(conf_weight) * conf_loss + float(qual_weight) * qual_loss
    return total, {'conf_loss': conf_loss, 'qual_loss': qual_loss}


def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    Supervised contrastive loss on a single view per sample.
    features: [B, D], labels: [B]
    """
    if features.dim() != 2:
        raise ValueError(f"features must be 2D [B, D], got {tuple(features.shape)}")
    labels = labels.view(-1).long()
    bsz = int(features.size(0))
    if bsz <= 1:
        return torch.tensor(0.0, device=features.device)

    feat = F.normalize(features, p=2, dim=1)
    logits = torch.matmul(feat, feat.t()) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

    self_mask = torch.eye(bsz, device=features.device, dtype=torch.bool)
    logits_mask = ~self_mask

    same = labels.unsqueeze(1).eq(labels.unsqueeze(0))
    pos_mask = same & logits_mask

    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(dim=1).float()
    valid = pos_count > 0
    if not valid.any():
        return torch.tensor(0.0, device=features.device)

    mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / pos_count.clamp(min=1.0)
    loss = -mean_log_prob_pos[valid].mean()
    return loss


def forward_reid_embeddings(model, images):
    """ """
    # Timm-based backbones may return either a pooled tensor or
    # a tuple like (pooled_feature, spatial_feature_map).
    if hasattr(model, "_forward_backbone") and hasattr(model, "neck"):
        backbone_out = model._forward_backbone(images)
        if isinstance(backbone_out, (tuple, list)):
            embed_feat = backbone_out[0]
        else:
            embed_feat = backbone_out
            proj = getattr(model, "proj", None)
            if proj is not None and isinstance(embed_feat, torch.Tensor) and embed_feat.dim() == 2:
                embed_feat = proj(embed_feat)
        feat_before_bn, feat_after_bn, _ = model.neck(embed_feat)
        return feat_after_bn, feat_before_bn

    if hasattr(model, "backbone") and hasattr(model, "neck"):
        backbone_feat = model.backbone.forward_features(images)
        if hasattr(model, "_pool_backbone_features"):
            backbone_feat = model._pool_backbone_features(backbone_feat)
        proj = getattr(model, "proj", None)
        embed_feat = proj(backbone_feat) if proj is not None else backbone_feat
        feat_before_bn, feat_after_bn, _ = model.neck(embed_feat)
        return feat_after_bn, feat_before_bn

    if hasattr(model, "forward_multitask"):
        feat_after_bn, feat_before_bn, *_ = model.forward_multitask(images)
        return feat_after_bn, feat_before_bn

    raise RuntimeError("Model does not expose a supported ReID forward path")


def train_one_epoch(config, model, prototype_net, criterion, data_loader,
                   optimizer, epoch, lr_scheduler, loss_scaler, logger, prototype_criterion=None,
                   training_memory_bank=None):
    """ """

    model.train()
    prototype_net.train()

    clip_parameters = list(model.parameters())
    try:
        clip_parameters.extend([p for p in criterion.parameters() if p.requires_grad])
    except Exception:
        pass
    try:
        clip_parameters.extend([p for p in prototype_net.parameters() if p.requires_grad])
    except Exception:
        pass

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    arc_meter = AverageMeter()
    triplet_meter = AverageMeter()
    center_meter = AverageMeter()
    variance_meter = AverageMeter()
    proto_meter = AverageMeter()
    supcon_meter = AverageMeter()
    mp_metric_meter = AverageMeter()
    continual_metric_meter = AverageMeter()
    dynamic_topology_meter = AverageMeter()
    uncertainty_topology_meter = AverageMeter()
    meta_topology_meter = AverageMeter()
    incremental_topology_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, batch_data in enumerate(data_loader):
        images = batch_data['image'].cuda(non_blocking=True)
        labels = batch_data['id_label'].cuda(non_blocking=True)

        with torch.amp.autocast('cuda', enabled=config.AMP_ENABLE):
            feat_after_bn, feat_before_bn = forward_reid_embeddings(model, images)
            feat_after_bn = F.normalize(feat_after_bn, p=2, dim=1)
            feat_before_bn = F.normalize(feat_before_bn, p=2, dim=1)
            quality_scores_live = prototype_net.quality_net(feat_after_bn).squeeze(1).float()

            loss_dict = criterion(
                feat_before_bn,
                None,
                labels,
                arcface_features=feat_after_bn,
            )
            main_loss = loss_dict['total_loss']

            # Update prototypes from ReID embeddings only.
            for feat, label in zip(feat_after_bn, labels):
                label_name = f'class_{label.item():03d}'
                quality_val = float(prototype_net.quality_net(feat.detach().unsqueeze(0)).item())
                prototype_net.update_prototype(
                    label_name,
                    feat.detach(),
                    quality_score=quality_val,
                )

            proto_aux_loss = torch.tensor(0.0, device=images.device)
            proto_conf_loss = torch.tensor(0.0, device=images.device)
            proto_qual_loss = torch.tensor(0.0, device=images.device)
            supcon_loss = torch.tensor(0.0, device=images.device)
            multi_proto_loss = torch.tensor(0.0, device=images.device)
            continual_metric_loss = torch.tensor(0.0, device=images.device)
            dynamic_topology_loss = torch.tensor(0.0, device=images.device)
            uncertainty_topology_loss = torch.tensor(0.0, device=images.device)
            meta_topology_loss = torch.tensor(0.0, device=images.device)
            incremental_topology_loss = torch.tensor(0.0, device=images.device)
            proto_aux_weight = float(getattr(config.MODEL, 'PROTO_AUX_WEIGHT', 0.05))
            if proto_aux_weight > 0:
                proto_conf_w = float(getattr(config.MODEL, 'PROTO_CONF_WEIGHT', 1.0))
                proto_qual_w = float(getattr(config.MODEL, 'PROTO_QUAL_WEIGHT', 0.5))
                proto_aux_loss, _stats = compute_prototype_aux_loss(
                    prototype_net,
                    feat_after_bn,
                    labels,
                    conf_weight=proto_conf_w,
                    qual_weight=proto_qual_w,
                )
                proto_conf_loss = _stats.get('conf_loss', proto_conf_loss)
                proto_qual_loss = _stats.get('qual_loss', proto_qual_loss)

            supcon_weight = float(getattr(config.MODEL, 'SUPCON_WEIGHT', 0.0))
            if supcon_weight > 0:
                supcon_temp = float(getattr(config.MODEL, 'SUPCON_TEMP', 0.07))
                supcon_loss = supervised_contrastive_loss(
                    feat_after_bn,
                    labels,
                    temperature=supcon_temp,
                )

            multi_proto_weight = float(getattr(config.MODEL, 'MULTI_PROTO_TRAIN_WEIGHT', 0.0))
            if training_memory_bank is not None and multi_proto_weight > 0:
                multi_proto_loss, _mp_stats = compute_multi_prototype_metric_loss(
                    training_memory_bank,
                    feat_after_bn,
                    labels,
                    temperature=float(getattr(config.MODEL, 'MULTI_PROTO_TEMP', 0.07)),
                    hard_neg_k=int(getattr(config.MODEL, 'MULTI_PROTO_HARD_NEG_K', 32)),
                )

            continual_metric_weight = float(getattr(config.MODEL, 'CONTINUAL_METRIC_WEIGHT', 0.0))
            if training_memory_bank is not None and continual_metric_weight > 0:
                continual_metric_loss, _cm_stats = compute_continual_metric_consistency_loss(
                    training_memory_bank,
                    feat_after_bn,
                    labels,
                    margin=float(getattr(config.MODEL, 'CONTINUAL_METRIC_MARGIN', 0.15)),
                    hard_neg_k=int(getattr(config.MODEL, 'CONTINUAL_METRIC_HARD_NEG_K', 16)),
                    stability_weight=float(getattr(config.MODEL, 'CONTINUAL_METRIC_STABILITY_WEIGHT', 0.5)),
                )

            dynamic_topology_weight = float(getattr(config.MODEL, 'DYNAMIC_TOPO_WEIGHT', 0.0))
            if training_memory_bank is not None and dynamic_topology_weight > 0:
                dynamic_topology_loss, _dt_stats = compute_dynamic_topology_loss(
                    training_memory_bank,
                    feat_after_bn,
                    labels,
                    temperature=float(getattr(config.MODEL, 'DYNAMIC_TOPO_TEMP', 0.10)),
                    negative_margin=float(getattr(config.MODEL, 'DYNAMIC_TOPO_NEG_MARGIN', 0.15)),
                    pull_weight=float(getattr(config.MODEL, 'DYNAMIC_TOPO_PULL_WEIGHT', 0.5)),
                )

            uncertainty_topology_weight = float(getattr(config.MODEL, 'UNCERTAINTY_TOPO_WEIGHT', 0.0))
            if training_memory_bank is not None and uncertainty_topology_weight > 0:
                uncertainty_topology_loss, _ut_stats = compute_uncertainty_topology_purification_loss(
                    training_memory_bank,
                    feat_after_bn,
                    labels,
                    quality_scores_live,
                    purify_blend=float(getattr(config.MODEL, 'UNCERTAINTY_PURIFY_BLEND', 0.35)),
                    margin=float(getattr(config.MODEL, 'UNCERTAINTY_MARGIN', 0.10)),
                )

            meta_topology_weight = float(getattr(config.MODEL, 'META_TOPO_WEIGHT', 0.0))
            if training_memory_bank is not None and meta_topology_weight > 0:
                meta_topology_loss, _mt_stats = compute_meta_fewshot_topology_loss(
                    training_memory_bank,
                    feat_after_bn,
                    labels,
                    support_shots=int(getattr(config.MODEL, 'META_TOPO_SUPPORT_SHOTS', 1)),
                    query_max=int(getattr(config.MODEL, 'META_TOPO_QUERY_MAX', 2)),
                    adapt_blend=float(getattr(config.MODEL, 'META_TOPO_ADAPT_BLEND', 0.35)),
                    temperature=float(getattr(config.MODEL, 'META_TOPO_TEMP', 0.07)),
                )

            incremental_topology_weight = float(getattr(config.MODEL, 'INCREMENTAL_TOPO_WEIGHT', 0.0))
            if training_memory_bank is not None and incremental_topology_weight > 0:
                incremental_topology_loss, _it_stats = compute_incremental_topology_stability_loss(
                    training_memory_bank,
                    feat_after_bn,
                    labels,
                    temperature=float(getattr(config.MODEL, 'INCREMENTAL_TOPO_TEMP', 0.10)),
                    slot_radius=float(getattr(config.MODEL, 'INCREMENTAL_TOPO_SLOT_RADIUS', 0.18)),
                    entropy_weight=float(getattr(config.MODEL, 'INCREMENTAL_TOPO_ENTROPY_WEIGHT', 0.5)),
                    centroid_weight=float(getattr(config.MODEL, 'INCREMENTAL_TOPO_CENTROID_WEIGHT', 0.5)),
                    slot_weight=float(getattr(config.MODEL, 'INCREMENTAL_TOPO_SLOT_WEIGHT', 0.25)),
                )

            loss = (
                main_loss
                + proto_aux_weight * proto_aux_loss
                + supcon_weight * supcon_loss
                + multi_proto_weight * multi_proto_loss
                + continual_metric_weight * continual_metric_loss
                + dynamic_topology_weight * dynamic_topology_loss
                + uncertainty_topology_weight * uncertainty_topology_loss
                + meta_topology_weight * meta_topology_loss
                + incremental_topology_weight * incremental_topology_loss
            )

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Skip invalid loss: {loss.item()}")
            continue

        optimizer.zero_grad()

        if config.AMP_ENABLE:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=config.TRAIN.CLIP_GRAD,
                parameters=clip_parameters,
                create_graph=is_second_order,
                update_grad=True,
            )
            if grad_norm is None:
                grad_norm = 0.0
        else:
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(clip_parameters, config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(clip_parameters)
            optimizer.step()

        if training_memory_bank is not None:
            with torch.no_grad():
                training_memory_bank.update_batch(
                    feat_after_bn.detach(),
                    labels.detach(),
                    quality_scores=quality_scores_live.detach(),
                )

        lr_scheduler.step_update(epoch * num_steps + idx)
        torch.cuda.synchronize()

        loss_meter.update(loss.item(), images.size(0))
        arcface_loss_val = loss_dict.get('arcface_loss', torch.tensor(0.0)).item()
        triplet_loss_val = loss_dict.get('triplet_loss', torch.tensor(0.0)).item()
        center_loss_val = loss_dict.get('center_loss', torch.tensor(0.0)).item()
        variance_loss_val = loss_dict.get('variance_loss', torch.tensor(0.0)).item()
        proto_aux_loss_val = float(proto_aux_loss.item()) if isinstance(proto_aux_loss, torch.Tensor) else float(proto_aux_loss)
        supcon_loss_val = float(supcon_loss.item()) if isinstance(supcon_loss, torch.Tensor) else float(supcon_loss)
        multi_proto_loss_val = float(multi_proto_loss.item()) if isinstance(multi_proto_loss, torch.Tensor) else float(multi_proto_loss)
        continual_metric_loss_val = float(continual_metric_loss.item()) if isinstance(continual_metric_loss, torch.Tensor) else float(continual_metric_loss)
        dynamic_topology_loss_val = float(dynamic_topology_loss.item()) if isinstance(dynamic_topology_loss, torch.Tensor) else float(dynamic_topology_loss)
        uncertainty_topology_loss_val = float(uncertainty_topology_loss.item()) if isinstance(uncertainty_topology_loss, torch.Tensor) else float(uncertainty_topology_loss)
        meta_topology_loss_val = float(meta_topology_loss.item()) if isinstance(meta_topology_loss, torch.Tensor) else float(meta_topology_loss)
        incremental_topology_loss_val = float(incremental_topology_loss.item()) if isinstance(incremental_topology_loss, torch.Tensor) else float(incremental_topology_loss)

        arc_meter.update(arcface_loss_val, images.size(0))
        triplet_meter.update(triplet_loss_val, images.size(0))
        center_meter.update(center_loss_val, images.size(0))
        variance_meter.update(variance_loss_val, images.size(0))
        proto_meter.update(proto_aux_loss_val, images.size(0))
        supcon_meter.update(supcon_loss_val, images.size(0))
        mp_metric_meter.update(multi_proto_loss_val, images.size(0))
        continual_metric_meter.update(continual_metric_loss_val, images.size(0))
        dynamic_topology_meter.update(dynamic_topology_loss_val, images.size(0))
        uncertainty_topology_meter.update(uncertainty_topology_loss_val, images.size(0))
        meta_topology_meter.update(meta_topology_loss_val, images.size(0))
        incremental_topology_meter.update(incremental_topology_loss_val, images.size(0))

        if grad_norm is not None:
            norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)

            proto_conf_loss_val = float(proto_conf_loss.item()) if isinstance(proto_conf_loss, torch.Tensor) else float(proto_conf_loss)
            proto_qual_loss_val = float(proto_qual_loss.item()) if isinstance(proto_qual_loss, torch.Tensor) else float(proto_qual_loss)

            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'arc {arcface_loss_val:.4f}\t tri {triplet_loss_val:.4f}\t'
                f'cen {center_loss_val:.4f}\t var {variance_loss_val:.4f}\t'
                f'proto {proto_aux_loss_val:.4f}\t pc {proto_conf_loss_val:.4f}\t pq {proto_qual_loss_val:.4f}\t'
                f'supcon {supcon_loss_val:.4f}\t mpm {multi_proto_loss_val:.4f}\t cml {continual_metric_loss_val:.4f}\t'
                f'dto {dynamic_topology_loss_val:.4f}\t uto {uncertainty_topology_loss_val:.4f}\t '
                f'mto {meta_topology_loss_val:.4f}\t ito {incremental_topology_loss_val:.4f}\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB'
            )

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return {
        "loss": float(loss_meter.avg),
        "arcface_loss": float(arc_meter.avg),
        "triplet_loss": float(triplet_meter.avg),
        "center_loss": float(center_meter.avg),
        "variance_loss": float(variance_meter.avg),
        "proto_loss": float(proto_meter.avg),
        "supcon_loss": float(supcon_meter.avg),
        "multi_proto_loss": float(mp_metric_meter.avg),
        "continual_metric_loss": float(continual_metric_meter.avg),
        "dynamic_topology_loss": float(dynamic_topology_meter.avg),
        "uncertainty_topology_loss": float(uncertainty_topology_meter.avg),
        "meta_topology_loss": float(meta_topology_meter.avg),
        "incremental_topology_loss": float(incremental_topology_meter.avg),
    }


def smart_sample_test_data(test_image_root, target_samples=500, min_samples_per_id=3, max_samples_per_id=15, min_ids=10):
    """ """
    import random
    from collections import defaultdict

    #
    id_samples = defaultdict(list)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    for root, dirs, files in os.walk(test_image_root):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(root, file)

                #
                true_id = parse_true_id_from_path(image_path, test_image_root)

                id_samples[true_id].append({
                    'image_path': image_path,
                    'true_id': true_id
                })

    #
    valid_ids = {id_name: samples for id_name, samples in id_samples.items()
                 if len(samples) >= min_samples_per_id}

    #
    if len(valid_ids) < min_ids:
        #
        min_samples_per_id = max(1, min_samples_per_id - 1)
        valid_ids = {id_name: samples for id_name, samples in id_samples.items()
                     if len(samples) >= min_samples_per_id}

    if not valid_ids:
        return []

    #
    total_valid_samples = sum(len(samples) for samples in valid_ids.values())
    num_valid_ids = len(valid_ids)

    #
    base_samples_per_id = max(min_samples_per_id, target_samples // num_valid_ids)
    base_samples_per_id = min(base_samples_per_id, max_samples_per_id)

    #
    sampled_data = []
    remaining_quota = target_samples

    #
    sorted_ids = sorted(valid_ids.items(), key=lambda x: len(x[1]), reverse=True)

    for id_name, samples in sorted_ids:
        if remaining_quota <= 0:
            break

        #
        available_samples = len(samples)
        desired_samples = min(base_samples_per_id, available_samples, remaining_quota)

        #
        remaining_ids = len([x for x in sorted_ids if x[0] not in [s['true_id'] for s in sampled_data]])
        if remaining_ids <= 3 and remaining_quota > desired_samples:
            desired_samples = min(available_samples, remaining_quota // max(1, remaining_ids))

        #
        selected_samples = random.sample(samples, desired_samples)
        sampled_data.extend(selected_samples)
        remaining_quota -= desired_samples

    #
    if remaining_quota > 0:
        for id_name, samples in sorted_ids:
            if remaining_quota <= 0:
                break

            current_count = len([s for s in sampled_data if s['true_id'] == id_name])
            if current_count < max_samples_per_id and current_count < len(samples):
                additional_needed = min(
                    remaining_quota,
                    max_samples_per_id - current_count,
                    len(samples) - current_count,
                )

                #
                used_paths = {s['image_path'] for s in sampled_data if s['true_id'] == id_name}
                unused_samples = [s for s in samples if s['image_path'] not in used_paths]

                if unused_samples and additional_needed > 0:
                    additional_samples = random.sample(
                        unused_samples,
                        min(additional_needed, len(unused_samples)),
                    )
                    sampled_data.extend(additional_samples)
                    remaining_quota -= len(additional_samples)

    return sampled_data


def merge_small_clusters_by_centroid_similarity(
    predictions,
    logger,
    small_cluster_max=3,
    merge_similarity_thr=0.90,
):
    if not predictions:
        return predictions

    feats_by_pid = defaultdict(list)
    for p in predictions:
        pid = str(p.get("predicted_id", ""))
        feat = p.get("feature", None)
        if pid and feat is not None:
            feats_by_pid[pid].append(np.asarray(feat, dtype=np.float32))

    if len(feats_by_pid) <= 1:
        for p in predictions:
            p.pop("feature", None)
        return predictions

    counts = {pid: len(v) for pid, v in feats_by_pid.items()}
    centroids = {}
    for pid, vecs in feats_by_pid.items():
        c = np.mean(np.stack(vecs, axis=0), axis=0)
        n = float(np.linalg.norm(c) + 1e-12)
        centroids[pid] = c / n

    small_ids = [pid for pid, c in counts.items() if c <= int(small_cluster_max)]
    anchor_ids = [pid for pid, c in counts.items() if c > int(small_cluster_max)]
    if not anchor_ids:
        anchor_ids = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)[: max(1, len(counts) // 3)]

    mapping = {}
    merged_clusters = 0
    for sid in sorted(small_ids, key=lambda k: counts[k]):
        svec = centroids.get(sid, None)
        if svec is None:
            continue
        best_id = None
        best_sim = -1.0
        for aid in anchor_ids:
            if aid == sid:
                continue
            avec = centroids.get(aid, None)
            if avec is None:
                continue
            sim = float(np.dot(svec, avec))
            if sim > best_sim:
                best_sim = sim
                best_id = aid
        if best_id is not None and best_sim >= float(merge_similarity_thr):
            mapping[sid] = best_id
            merged_clusters += 1

    if mapping:
        for p in predictions:
            pid = str(p.get("predicted_id", ""))
            p["predicted_id"] = mapping.get(pid, pid)

        after_ids = len(set(str(p.get("predicted_id", "")) for p in predictions))
        before_ids = len(feats_by_pid)
        logger.info(
            f"Small-cluster merge applied: merged_clusters={merged_clusters}, "
            f"pred_ids {before_ids} -> {after_ids}, "
            f"small_max={small_cluster_max}, sim_thr={merge_similarity_thr:.3f}"
        )

    for p in predictions:
        p.pop("feature", None)

    return predictions


def imread_unicode(path: str, flags=cv2.IMREAD_COLOR):
    """Read image robustly on Windows Unicode paths."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data is None or data.size == 0:
            raise RuntimeError('empty file')
        img = cv2.imdecode(data, flags)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        return cv2.imread(path, flags)
    except Exception:
        return None


def is_readable_file(path: str) -> bool:
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data is None or data.size == 0:
            return False
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img is not None
    except Exception:
        return False


def collect_eval_image_paths(root: str, max_images: int = 0):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
    paths = []
    limit = int(max_images)
    if limit > 0:
        for dp, _dirs, fnames in os.walk(root):
            for name in sorted(fnames):
                if os.path.splitext(name)[1] not in image_exts:
                    continue
                path = os.path.join(dp, name)
                if not is_readable_file(path):
                    continue
                paths.append(path)
                if len(paths) >= limit:
                    return paths
        return paths

    for dp, _dirs, fnames in os.walk(root):
        for name in fnames:
            if os.path.splitext(name)[1] not in image_exts:
                continue
            path = os.path.join(dp, name)
            if not is_readable_file(path):
                continue
            paths.append(path)
    paths.sort()
    return paths


def parse_true_id_from_path(path: str, base_root: str) -> str:
    rel = os.path.relpath(path, base_root)
    parts = [p for p in os.path.normpath(rel).split(os.sep) if p]
    if not parts:
        return "unknown"
    first = str(parts[0])

    legacy_parts = first.split("_")
    if len(legacy_parts) == 3 and legacy_parts[1].isdigit() and len(legacy_parts[1]) == 4:
        return legacy_parts[0] if legacy_parts[0] else "unknown"

    # Generic species layout may be evaluated from either:
    # - full root:   species/individual/file  -> use species__individual
    # - species root: individual/file         -> use individual
    if len(parts) >= 3:
        second = str(parts[1]).strip()
        if first and second:
            return f"{first}__{second}"
    if len(parts) == 2 and first:
        return first

    fallback = legacy_parts[0] if legacy_parts and legacy_parts[0] else first
    return fallback or "unknown"


class EvalImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, base_root: str, transform):
        self.image_paths = list(image_paths)
        self.base_root = str(base_root)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        try:
            img = imread_unicode(path)
            if img is None:
                raise RuntimeError('imread returned None')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        pil = Image.fromarray(img)
        x = self.transform(pil)
        true_id = parse_true_id_from_path(path, self.base_root)
        return {
            "image": x,
            "true_id": true_id,
            "image_path": path,
        }


def sequential_cluster_assign(features: np.ndarray, threshold: float, momentum: float = 0.9):
    # Features are expected to be L2-normalized.
    n, d = features.shape
    if n == 0:
        return np.zeros((0,), dtype=np.int32), 0

    capacity = 128
    protos = np.zeros((capacity, d), dtype=np.float32)
    pred = np.zeros((n,), dtype=np.int32)
    k = 0

    for i in range(n):
        f = features[i]
        if k == 0:
            protos[0] = f
            pred[i] = 0
            k = 1
            continue

        sims = protos[:k] @ f
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= float(threshold):
            pred[i] = best_idx
            upd = float(momentum) * protos[best_idx] + (1.0 - float(momentum)) * f
            norm = float(np.linalg.norm(upd)) + 1e-12
            protos[best_idx] = upd / norm
        else:
            if k >= capacity:
                new_capacity = capacity * 2
                new_protos = np.zeros((new_capacity, d), dtype=np.float32)
                new_protos[:capacity] = protos
                protos = new_protos
                capacity = new_capacity
            protos[k] = f
            pred[i] = k
            k += 1

    return pred, k


def resolve_eval_thresholds(config):
    thresholds = []
    raw = getattr(config.MODEL, "EVAL_THRESHOLD_SWEEP", [])
    if isinstance(raw, (list, tuple)):
        for x in raw:
            try:
                thresholds.append(float(x))
            except Exception:
                pass
    else:
        txt = str(raw).strip()
        if txt:
            for x in txt.split(","):
                x = x.strip()
                if not x:
                    continue
                try:
                    thresholds.append(float(x))
                except Exception:
                    pass

    if len(thresholds) == 0:
        thresholds = [float(getattr(config.MODEL, "EVAL_CLUSTER_THRESHOLD", 0.24))]

    uniq = sorted(set(float(x) for x in thresholds))
    return uniq


def evaluate_prototype_system(model, prototype_net, test_image_root, test_roi_root,
                            config, device, logger):
    """
    Validation in the same metric protocol as evaluate_openworld_with_specialists.py:
    1) Full/partial ordered feature extraction on test root.
    2) Sequential open-world clustering on embeddings.
    3) Threshold sweep and best-threshold selection by mean(assign, purity, id_count_acc).
    """
    logger.info("Starting prototype system evaluation (offline clustering protocol)...")

    model.eval()
    prototype_net.eval()

    thresholds = resolve_eval_thresholds(config)
    max_images = int(getattr(config.MODEL, "EVAL_MAX_IMAGES", 0))
    proto_momentum = float(getattr(config.MODEL, "EVAL_PROTOTYPE_MOMENTUM", 0.9))
    logger.info(
        f"Eval settings | max_images={max_images if max_images > 0 else 'ALL'} "
        f"thresholds={thresholds} momentum={proto_momentum:.3f}"
    )

    image_paths = collect_eval_image_paths(test_image_root, max_images=max_images)
    if len(image_paths) == 0:
        logger.warning("No test images found")
        return {'accuracy': 0.0, 'predicted_ids': 0, 'true_ids': 0, 'id_error': 0, 'cluster_metrics': {}}

    transform = build_panda_transform(is_train=False, img_size=config.DATA.IMG_SIZE)
    eval_dataset = EvalImagePathDataset(
        image_paths=image_paths,
        base_root=test_image_root,
        transform=transform,
    )
    eval_loader_kwargs = {}
    eval_num_workers = max(0, int(getattr(config.DATA, "NUM_WORKERS", 8)))
    if eval_num_workers > 0:
        eval_loader_kwargs["persistent_workers"] = True
        eval_loader_kwargs["prefetch_factor"] = 4
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=max(1, int(getattr(config.DATA, "BATCH_SIZE", 32))),
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=True,
        drop_last=False,
        **eval_loader_kwargs,
    )

    logger.info(f"Extracting test embeddings from {len(eval_dataset)} images...")
    features = []
    true_ids = []
    amp_enabled = bool(getattr(config, "AMP_ENABLE", True))
    with torch.no_grad():
        for bi, batch in enumerate(eval_loader):
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                feat_after_bn, _feat_before_bn = forward_reid_embeddings(model, images)
                feat_after_bn = F.normalize(feat_after_bn, p=2, dim=1)
            features.append(feat_after_bn.cpu().numpy().astype(np.float32, copy=False))
            true_ids.extend([str(x) for x in batch["true_id"]])
            if (bi + 1) % 50 == 0:
                logger.info(f"Feature extraction progress: {bi + 1}/{len(eval_loader)}")

    if len(features) == 0:
        logger.warning("No test features extracted")
        return {'accuracy': 0.0, 'predicted_ids': 0, 'true_ids': 0, 'id_error': 0, 'cluster_metrics': {}}

    features_np = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    logger.info(f"Extracted features shape: {tuple(features_np.shape)}")

    threshold_results = []
    best_idx = 0
    best_score = -1e9
    for i, th in enumerate(thresholds):
        pred_cluster, k = sequential_cluster_assign(
            features=features_np,
            threshold=float(th),
            momentum=float(proto_momentum),
        )
        pred_ids = [f"wild_{int(x) + 1:05d}" for x in pred_cluster.tolist()]
        preds = [{"true_id": t, "predicted_id": p} for t, p in zip(true_ids, pred_ids)]
        cluster_metrics = compute_open_world_cluster_metrics(preds)

        assignment = float(cluster_metrics.get("assignment_accuracy", 0.0))
        purity = float(cluster_metrics.get("cluster_purity", 0.0))
        id_acc = float(cluster_metrics.get("id_count_accuracy", 0.0))
        score = (assignment + purity + id_acc) / 3.0

        row = {
            "threshold": float(th),
            "num_predicted_clusters": int(k),
            "score": float(score),
            "clustering": {
                "cluster_purity": purity,
                "assignment_accuracy": assignment,
                "cluster_contamination": float(cluster_metrics.get("cluster_contamination", 1.0)),
                "id_count_accuracy": id_acc,
                "predicted_id_count": int(cluster_metrics.get("predicted_id_count", 0)),
                "true_id_count": int(cluster_metrics.get("true_id_count", 0)),
                "id_count_abs_error": int(cluster_metrics.get("id_count_abs_error", 0)),
                "id_count_rel_error": float(cluster_metrics.get("id_count_rel_error", 0.0)),
            },
        }
        threshold_results.append(row)

        logger.info(
            f"[TH={th:.3f}] assignment={assignment:.4f}, purity={purity:.4f}, "
            f"id_acc={id_acc:.4f}, pred_ids={row['clustering']['predicted_id_count']}, "
            f"true_ids={row['clustering']['true_id_count']}"
        )
        if score > best_score:
            best_score = score
            best_idx = i

    best = threshold_results[best_idx]
    best_clustering = dict(best["clustering"])
    accuracy = float(best_clustering.get("assignment_accuracy", 0.0))
    purity = float(best_clustering.get("cluster_purity", 0.0))
    predicted_ids = int(best_clustering.get("predicted_id_count", 0))
    true_ids_cnt = int(best_clustering.get("true_id_count", 0))
    id_error = int(abs(predicted_ids - true_ids_cnt))

    logger.info("Evaluation results (offline clustering):")
    logger.info(f"  Best Threshold: {float(best['threshold']):.4f}")
    logger.info(f"  Assignment Accuracy: {accuracy:.4f}")
    logger.info(f"  Cluster Purity: {purity:.4f}")
    logger.info(f"  Cluster Contamination: {float(best_clustering.get('cluster_contamination', 1.0)):.4f}")
    logger.info(f"  ID Count Accuracy: {float(best_clustering.get('id_count_accuracy', 0.0)):.4f}")
    logger.info(f"  Predicted IDs: {predicted_ids}")
    logger.info(f"  True IDs: {true_ids_cnt}")
    logger.info(f"  ID count error: {id_error}")

    cluster_metrics = dict(best_clustering)
    cluster_metrics["best_threshold"] = float(best["threshold"])
    cluster_metrics["threshold_results"] = threshold_results
    cluster_metrics["eval_num_images"] = int(features_np.shape[0])

    return {
        'accuracy': accuracy,
        'assignment_accuracy': accuracy,
        'cluster_purity': purity,
        'predicted_ids': predicted_ids,
        'true_ids': true_ids_cnt,
        'id_error': id_error,
        'cluster_metrics': cluster_metrics,
    }


def compute_traditional_accuracy(predictions, logger):
    if len(predictions) == 0:
        return 0.0
    id_mapping = {}
    correct = 0
    total = 0
    for pred in predictions:
        true_id = pred["true_id"]
        predicted_id = pred["predicted_id"]
        if true_id not in id_mapping:
            id_mapping[true_id] = predicted_id
        if id_mapping[true_id] == predicted_id:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def compute_optimal_matching_accuracy(predictions, logger):
    if len(predictions) == 0:
        return 0.0
    if not SCIPY_AVAILABLE:
        logger.info("Scipy not available, using greedy matching")
        return compute_greedy_matching_accuracy(predictions, logger)

    try:
        true_ids = list(set(pred["true_id"] for pred in predictions))
        pred_ids = list(set(pred["predicted_id"] for pred in predictions))

        confusion_matrix = defaultdict(lambda: defaultdict(int))
        for pred in predictions:
            confusion_matrix[pred["true_id"]][pred["predicted_id"]] += 1

        matrix = np.zeros((len(true_ids), len(pred_ids)), dtype=np.float32)
        for i, true_id in enumerate(true_ids):
            for j, pred_id in enumerate(pred_ids):
                matrix[i, j] = confusion_matrix[true_id][pred_id]

        row_indices, col_indices = linear_sum_assignment(-matrix)
        optimal_matches = sum(matrix[row_indices[i], col_indices[i]] for i in range(len(row_indices)))
        total_predictions = len(predictions)
        return float(optimal_matches / total_predictions) if total_predictions > 0 else 0.0
    except Exception as e:
        logger.warning(f"Optimal matching failed: {e}, falling back to greedy matching")
        return compute_greedy_matching_accuracy(predictions, logger)


def compute_greedy_matching_accuracy(predictions, logger):
    if len(predictions) == 0:
        return 0.0
    try:
        true_ids = list(set(pred["true_id"] for pred in predictions))
        pred_ids = list(set(pred["predicted_id"] for pred in predictions))

        confusion_matrix = defaultdict(lambda: defaultdict(int))
        for pred in predictions:
            confusion_matrix[pred["true_id"]][pred["predicted_id"]] += 1

        used_pred_ids = set()
        total_matches = 0
        matches = []

        for true_id in true_ids:
            for pred_id in pred_ids:
                if pred_id not in used_pred_ids:
                    count = confusion_matrix[true_id][pred_id]
                    if count > 0:
                        matches.append((count, true_id, pred_id))

        matches.sort(reverse=True)
        for count, true_id, pred_id in matches:
            if pred_id not in used_pred_ids:
                total_matches += count
                used_pred_ids.add(pred_id)

        return total_matches / len(predictions) if len(predictions) > 0 else 0.0
    except Exception as e:
        logger.warning(f"Greedy matching failed: {e}, falling back to traditional method")
        return compute_traditional_accuracy(predictions, logger)


def compute_confidence_weighted_accuracy(predictions, logger):
    high_confidence_preds = [p for p in predictions if p.get("confidence", 0.0) > 0.1]
    if len(high_confidence_preds) == 0:
        return 0.0
    return compute_optimal_matching_accuracy(high_confidence_preds, logger)


def compute_improved_accuracy(predictions, logger):
    try:
        true_ids = list(set(pred["true_id"] for pred in predictions))
        pred_ids = list(set(pred["predicted_id"] for pred in predictions))

        confusion_matrix = defaultdict(lambda: defaultdict(int))
        for pred in predictions:
            confusion_matrix[pred["true_id"]][pred["predicted_id"]] += 1

        matches = []
        for true_id in true_ids:
            for pred_id in pred_ids:
                count = confusion_matrix[true_id][pred_id]
                if count > 0:
                    matches.append((count, true_id, pred_id))

        matches.sort(reverse=True)
        used_pred_ids = set()
        used_true_ids = set()
        total_correct = 0
        final_mapping = {}

        for count, true_id, pred_id in matches:
            if true_id not in used_true_ids and pred_id not in used_pred_ids:
                final_mapping[true_id] = pred_id
                used_true_ids.add(true_id)
                used_pred_ids.add(pred_id)
                total_correct += count

        accuracy = total_correct / len(predictions) if len(predictions) > 0 else 0.0
        logger.info(f"Improved accuracy (one-to-one mapping): {accuracy:.4f}")
        logger.info(f"  Total predictions: {len(predictions)}")
        logger.info(f"  Correct predictions: {total_correct}")
        logger.info(f"  Mapped pairs: {len(final_mapping)}")
        logger.info(f"  True IDs: {len(true_ids)}, Predicted IDs: {len(pred_ids)}")
        return accuracy
    except Exception as e:
        logger.warning(f"Improved accuracy calculation failed: {e}")
        return compute_traditional_accuracy(predictions, logger)


def calculate_comprehensive_score(
    accuracy,
    predicted_ids,
    true_ids,
    clustering_metrics=None,
):
    assignment_score = float(accuracy)

    purity_score = float(
        clustering_metrics.get("cluster_purity", 0.0)
        if isinstance(clustering_metrics, dict)
        else 0.0
    )

    if isinstance(clustering_metrics, dict) and "id_count_accuracy" in clustering_metrics:
        id_score = float(clustering_metrics.get("id_count_accuracy", 0.0))
    else:
        id_ratio = min(predicted_ids, true_ids) / max(predicted_ids, true_ids, 1)
        id_penalty = abs(predicted_ids - true_ids) / max(true_ids, 1)
        id_score = id_ratio * (1 - min(id_penalty, 0.5))

    if isinstance(clustering_metrics, dict) and "separation_ratio" in clustering_metrics:
        separation_ratio = float(clustering_metrics["separation_ratio"])
        feature_quality_score = min(max((separation_ratio - 1.0) / 1.0, 0.0), 1.0)
    else:
        feature_quality_score = 0.5

    comprehensive_score = (
        0.50 * assignment_score
        + 0.25 * purity_score
        + 0.20 * id_score
        + 0.05 * feature_quality_score
    )

    return comprehensive_score, {
        "assignment_score": assignment_score,
        "purity_score": purity_score,
        "id_score": id_score,
        "feature_quality_score": feature_quality_score,
    }


def targeted_adaptive_tuning(config, criterion, prototype_net, eval_results, logger):
    if not isinstance(eval_results, dict):
        return False

    changed = False
    changes = []

    assignment_acc = float(eval_results.get("assignment_accuracy", 0.0) or 0.0)
    purity = float(eval_results.get("cluster_purity", 0.0) or 0.0)

    if assignment_acc < 0.90 or purity < 0.90:
        old_tw = float(getattr(criterion, "triplet_weight", 1.0))
        old_cw = float(getattr(criterion, "center_weight", 0.01))
        criterion.triplet_weight = min(2.5, old_tw * 1.15)
        criterion.center_weight = min(0.08, max(old_cw, 0.005) * 1.20)

        if hasattr(prototype_net, "sim_cosine_w") and hasattr(prototype_net, "sim_euclid_w"):
            prototype_net.sim_cosine_w = min(0.90, float(prototype_net.sim_cosine_w) + 0.05)
            prototype_net.sim_euclid_w = max(0.10, 1.0 - float(prototype_net.sim_cosine_w))

        config.defrost()
        config.MODEL.PROTO_AUX_WEIGHT = min(0.20, float(getattr(config.MODEL, "PROTO_AUX_WEIGHT", 0.05)) * 1.15)
        config.freeze()

        changes.append(
            f"open-world->triplet_weight:{old_tw:.3f}->{criterion.triplet_weight:.3f}, "
            f"center_weight:{old_cw:.4f}->{criterion.center_weight:.4f}, "
            f"proto_aux_weight->{float(getattr(config.MODEL, 'PROTO_AUX_WEIGHT', 0.05)):.4f}"
        )
        changed = True

    if changed:
        logger.info("Targeted adaptive tuning applied:")
        for c in changes:
            logger.info(f"  - {c}")

    return changed


def create_test_dataloader_for_analysis(test_image_root, test_roi_root, config, test_roiimg_root=None):
    try:
        transform = build_panda_transform(is_train=False, img_size=config.DATA.IMG_SIZE)

        if test_roiimg_root:
            test_dataset = PandaRoiImgDataset(
                roiimg_root=test_roiimg_root,
                transform=transform,
                is_train=False,
            )
        else:
            test_dataset = PandaDataset(
                image_root=test_image_root,
                roi_root=test_roi_root,
                transform=transform,
                is_train=False,
                roi_format='mask',
                mask_root=test_roi_root,
            )

        if len(test_dataset) == 0:
            print("[ERROR] Test dataset is empty!")
            return None

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
        )
        return test_loader
    except Exception as e:
        print(f"[ERROR] Failed to create test dataloader: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None


def analyze_feature_clustering(model, data_loader, device, logger, epoch, output_dir, max_samples=500, dataset_name="train"):
    model.eval()

    features_list = []
    labels_list = []
    id_names_list = []

    logger.info(f"Extracting features for ID separation analysis ({dataset_name} set)...")

    with torch.no_grad():
        for batch_data in data_loader:
            images = batch_data['image'].to(device)
            labels = batch_data['id_label'].to(device)
            id_names = batch_data['id_name']

            feat_after_bn, _feat_before_bn = forward_reid_embeddings(model, images)
            feat_after_bn = F.normalize(feat_after_bn, p=2, dim=1)

            for i in range(feat_after_bn.size(0)):
                features_list.append(feat_after_bn[i].cpu())
                labels_list.append(labels[i].cpu())
                id_names_list.append(id_names[i])
                if len(features_list) >= max_samples:
                    break
            if len(features_list) >= max_samples:
                break

    if len(features_list) == 0:
        logger.warning("No features extracted for clustering analysis")
        return {}

    all_features_np = torch.stack(features_list).numpy()
    all_labels_np = torch.stack(labels_list).numpy()

    unique_id_names = sorted(list(set(id_names_list)))
    id_to_label = {id_name: i for i, id_name in enumerate(unique_id_names)}
    mapped_labels_np = np.array([id_to_label[x] for x in id_names_list], dtype=np.int64)

    logger.info(f"Analyzing {len(all_features_np)} features from {len(unique_id_names)} IDs")

    metrics = {}

    try:
        if len(np.unique(mapped_labels_np)) > 1:
            metrics['silhouette_score'] = float(silhouette_score(all_features_np, mapped_labels_np))
        else:
            metrics['silhouette_score'] = 0.0

        intra_class_distances = []
        inter_class_distances = []
        unique_labels = np.unique(mapped_labels_np)

        for label in unique_labels:
            class_features = all_features_np[mapped_labels_np == label]
            if len(class_features) > 1:
                class_center = class_features.mean(axis=0)
                intra_distances = np.linalg.norm(class_features - class_center, axis=1)
                intra_class_distances.extend(intra_distances)

                for other_label in unique_labels:
                    if other_label != label:
                        other_features = all_features_np[mapped_labels_np == other_label]
                        other_center = other_features.mean(axis=0)
                        inter_distance = np.linalg.norm(class_center - other_center)
                        inter_class_distances.append(inter_distance)

        avg_intra_distance = float(np.mean(intra_class_distances)) if intra_class_distances else 0.0
        avg_inter_distance = float(np.mean(inter_class_distances)) if inter_class_distances else 0.0
        metrics['avg_intra_distance'] = avg_intra_distance
        metrics['avg_inter_distance'] = avg_inter_distance
        metrics['separation_ratio'] = float(avg_inter_distance / (avg_intra_distance + 1e-8))

        logger.info(f"Feature metrics ({dataset_name}): silhouette={metrics['silhouette_score']:.4f}, "
                    f"intra={avg_intra_distance:.4f}, inter={avg_inter_distance:.4f}, "
                    f"ratio={metrics['separation_ratio']:.4f}")
    except Exception as e:
        logger.error(f"Error in feature quality analysis: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {}

    try:
        if len(all_features_np) > 50 and len(np.unique(mapped_labels_np)) > 1:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(all_features_np)//4)))
            features_2d = tsne.fit_transform(all_features_np)

            plt.figure(figsize=(12, 8))
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
            for i, label in enumerate(np.unique(mapped_labels_np)):
                mask = mapped_labels_np == label
                id_name = unique_id_names[label] if label < len(unique_id_names) else f'ID_{label}'
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], c=[colors[i]], label=id_name, alpha=0.7, s=30)

            plt.title(f'Feature Distribution ({dataset_name.upper()} - Epoch {epoch})')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            if len(unique_labels) <= 10:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            os.makedirs(output_dir, exist_ok=True)
            viz_path = os.path.join(output_dir, f'feature_clustering_{dataset_name}_epoch_{epoch}.png')
            plt.savefig(_path_for_write(viz_path), dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"t-SNE saved: {viz_path}")
    except Exception as e:
        logger.warning(f"Error in t-SNE visualization: {e}")

    try:
        if len(all_features_np) > 50 and len(np.unique(mapped_labels_np)) > 1:
            os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
            from umap import UMAP
            reducer = UMAP(
                n_components=2,
                n_neighbors=min(30, max(5, len(all_features_np) // 20)),
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
            features_2d = reducer.fit_transform(all_features_np.astype(np.float32, copy=False))

            plt.figure(figsize=(12, 8))
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
            for i, label in enumerate(np.unique(mapped_labels_np)):
                mask = mapped_labels_np == label
                id_name = unique_id_names[label] if label < len(unique_id_names) else f'ID_{label}'
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], c=[colors[i]], label=id_name, alpha=0.7, s=30)

            plt.title(f'Feature UMAP ({dataset_name.upper()} - Epoch {epoch})')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            if len(unique_labels) <= 10:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            os.makedirs(output_dir, exist_ok=True)
            umap_path = os.path.join(output_dir, f'feature_umap_{dataset_name}_epoch_{epoch}.png')
            plt.savefig(_path_for_write(umap_path), dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"UMAP saved: {umap_path}")
    except Exception as e:
        logger.warning(f"Error in UMAP visualization: {e}")

    return metrics


def save_training_history_artifacts(train_history, output_dir, logger):
    """ """
    try:
        os.makedirs(output_dir, exist_ok=True)
        hist_path = os.path.join(output_dir, "training_history.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(train_history, f, ensure_ascii=False, indent=2)

        epochs = train_history.get("epochs", [])
        if len(epochs) == 0:
            logger.warning("Training history is empty, skip plotting curves")
            return

        def _to_plot_array(values):
            arr = []
            for v in values:
                if v is None:
                    arr.append(np.nan)
                else:
                    arr.append(float(v))
            return np.array(arr, dtype=np.float32)

        loss_arr = _to_plot_array(train_history.get("train_loss", []))
        purity_arr = _to_plot_array(train_history.get("eval_purity", []))
        assign_arr = _to_plot_array(train_history.get("eval_assignment_acc", []))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, loss_arr, "b-o", linewidth=2, markersize=4)
        ax1.set_title("Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True)

        ax2.plot(epochs, purity_arr, color="teal", marker="o", linewidth=2, markersize=4, label="Purity")
        ax2.plot(epochs, assign_arr, color="navy", marker="s", linewidth=2, markersize=4, label="Assignment")
        ax2.axhline(0.90, color="r", linestyle="--", linewidth=1, label="Target=0.90")
        ax2.set_title("Open-World ID Metrics")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score")
        ax2.legend()
        ax2.grid(True)

        plt.suptitle("Training and ReID Evaluation Curves", fontsize=16)
        plt.tight_layout()

        curve_path = os.path.join(output_dir, "training_eval_curves.png")
        plt.savefig(_path_for_write(curve_path), dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved training history json: {hist_path}")
        logger.info(f"Saved training/eval curves: {curve_path}")
    except Exception as e:
        logger.warning(f"Failed to save training history artifacts: {e}")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p is not None and p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    norm_type = float(norm_type)
    total_norm = 0.0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += float(param_norm.item()) ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return float(total_norm)


def main():
    args, config = parse_option()
    torch.backends.cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    logger.info(f"Creating prototype ReID model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    logger.info("ReID-only training mode (age/gender disabled)")

    if args.roiimg_root:
        dataset_train = PandaRoiImgDataset(
            roiimg_root=args.roiimg_root,
            img_size=config.DATA.IMG_SIZE,
            transform=build_panda_transform(is_train=True, img_size=config.DATA.IMG_SIZE, config=config),
            is_train=True,
        )
    else:
        dataset_train = PandaDataset(
            image_root=args.image_root,
            roi_root=args.roi_root,
            img_size=config.DATA.IMG_SIZE,
            transform=build_panda_transform(is_train=True, img_size=config.DATA.IMG_SIZE, config=config),
            is_train=True,
            roi_format=args.roi_format,
            mask_root=args.mask_root,
        )

    logger.info(f"Train samples: {len(dataset_train)}")
    logger.info(f"Train IDs: {dataset_train.get_num_classes()}")

    num_instances = int(getattr(args, "num_instances", 4))
    use_identity_sampler = num_instances > 1
    if use_identity_sampler:
        logger.info(
            f"Using RandomIdentitySampler(PK): batch_size={config.DATA.BATCH_SIZE}, "
            f"K={num_instances}, P={config.DATA.BATCH_SIZE // max(1, num_instances)}"
        )
        sampler_max_groups = int(getattr(config.TRAIN, 'SAMPLER_MAX_GROUPS_PER_ID', 0) or 0)
        sampler_min_groups = int(getattr(config.TRAIN, 'SAMPLER_MIN_GROUPS_PER_ID', 1) or 1)
        sampler_use_ceil = bool(getattr(config.TRAIN, 'SAMPLER_USE_CEIL', False))
        logger.info(
            f"Sampler balancing | use_ceil={sampler_use_ceil} max_groups_per_id={sampler_max_groups} "
            f"min_groups_per_id={sampler_min_groups}"
        )
        sampler = RandomIdentitySampler(
            dataset_train,
            batch_size=int(config.DATA.BATCH_SIZE),
            num_instances=num_instances,
            max_groups_per_pid=sampler_max_groups,
            min_groups_per_pid=sampler_min_groups,
            use_ceil_groups=sampler_use_ceil,
        )
    else:
        sampler = None

    loader_kwargs = {}
    if int(config.DATA.NUM_WORKERS) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        **loader_kwargs,
    )

    model = build_panda_reid_model(config, dataset_train.get_num_classes())
    model.cuda()

    prototype_net = PrototypeReIDNetwork(
        feature_dim=model.feature_dim,
        temperature=0.05,
        momentum=0.9,
        min_samples=1,
    )
    try:
        if hasattr(config.MODEL, "OW_BASE_THRESHOLD_NO_STATS"):
            prototype_net.base_threshold_no_stats = float(config.MODEL.OW_BASE_THRESHOLD_NO_STATS)
        if hasattr(config.MODEL, "OW_BASE_THRESHOLD_WITH_STATS"):
            prototype_net.base_threshold_with_stats = float(config.MODEL.OW_BASE_THRESHOLD_WITH_STATS)
        if hasattr(config.MODEL, "OW_QUALITY_ADJUST_SCALE"):
            prototype_net.quality_adjust_scale = float(config.MODEL.OW_QUALITY_ADJUST_SCALE)
        if hasattr(config.MODEL, "OW_DEVIATION_ADJUST_SCALE"):
            prototype_net.deviation_adjust_scale = float(config.MODEL.OW_DEVIATION_ADJUST_SCALE)
        if hasattr(config.MODEL, "OW_DEVIATION_ADJUST_CAP"):
            prototype_net.deviation_adjust_cap = float(config.MODEL.OW_DEVIATION_ADJUST_CAP)
        if hasattr(config.MODEL, "OW_CONFIDENCE_THRESHOLD"):
            prototype_net.confidence_threshold = float(config.MODEL.OW_CONFIDENCE_THRESHOLD)
        if hasattr(config.MODEL, "OW_QUALITY_THRESHOLD"):
            prototype_net.quality_threshold = float(config.MODEL.OW_QUALITY_THRESHOLD)
        if hasattr(config.MODEL, "OW_AMBIGUOUS_MARGIN"):
            prototype_net.ambiguous_margin = float(config.MODEL.OW_AMBIGUOUS_MARGIN)
        if hasattr(config.MODEL, "OW_AMBIGUOUS_OFFSET"):
            prototype_net.ambiguous_offset = float(config.MODEL.OW_AMBIGUOUS_OFFSET)
    except Exception as e:
        logger.warning(f"Open-world gate override failed: {e}")
    prototype_net.cuda()

    logger.info("Prototype network focus: open-world ID recognition")

    multi_proto_weight = float(getattr(config.MODEL, 'MULTI_PROTO_TRAIN_WEIGHT', 0.0))
    continual_metric_weight = float(getattr(config.MODEL, 'CONTINUAL_METRIC_WEIGHT', 0.0))
    dynamic_topology_weight = float(getattr(config.MODEL, 'DYNAMIC_TOPO_WEIGHT', 0.0))
    uncertainty_topology_weight = float(getattr(config.MODEL, 'UNCERTAINTY_TOPO_WEIGHT', 0.0))
    meta_topology_weight = float(getattr(config.MODEL, 'META_TOPO_WEIGHT', 0.0))
    incremental_topology_weight = float(getattr(config.MODEL, 'INCREMENTAL_TOPO_WEIGHT', 0.0))
    use_training_memory_bank = any([
        multi_proto_weight > 0.0,
        continual_metric_weight > 0.0,
        dynamic_topology_weight > 0.0,
        uncertainty_topology_weight > 0.0,
        meta_topology_weight > 0.0,
        incremental_topology_weight > 0.0,
    ])
    training_memory_bank = None
    if use_training_memory_bank:
        training_memory_bank = TrainingMultiPrototypeMemory(
            feature_dim=model.feature_dim,
            max_slots=int(getattr(config.MODEL, 'MULTI_PROTO_MAX_SLOTS', 4)),
            momentum=float(getattr(config.MODEL, 'MULTI_PROTO_MOMENTUM', 0.9)),
            spawn_threshold=float(getattr(config.MODEL, 'MULTI_PROTO_SPAWN_THRESHOLD', 0.55)),
            update_threshold=float(getattr(config.MODEL, 'MULTI_PROTO_UPDATE_THRESHOLD', 0.45)),
            spawn_min_quality=float(getattr(config.MODEL, 'MULTI_PROTO_SPAWN_MIN_QUALITY', 0.0)),
            class_centroid_guard=float(getattr(config.MODEL, 'MULTI_PROTO_CLASS_CENTROID_GUARD', 0.0)),
            update_min_quality=float(getattr(config.MODEL, 'MULTI_PROTO_UPDATE_MIN_QUALITY', 0.0)),
        ).to(torch.device('cuda'))
        logger.info(
            "Training multi-prototype memory enabled | "
            f"multi_proto_w={multi_proto_weight:.4f} continual_metric_w={continual_metric_weight:.4f} "
            f"dyn_topo_w={dynamic_topology_weight:.4f} unc_topo_w={uncertainty_topology_weight:.4f} "
            f"meta_topo_w={meta_topology_weight:.4f} inc_topo_w={incremental_topology_weight:.4f} "
            f"max_slots={training_memory_bank.max_slots}"
        )

    reset_epoch_on_resume = bool(getattr(config.TRAIN, "RESET_EPOCH_ON_RESUME", False))
    reset_opt_on_resume = bool(getattr(config.TRAIN, "RESET_OPTIMIZER_ON_RESUME", False))
    reset_sched_on_resume = bool(getattr(config.TRAIN, "RESET_LR_SCHEDULER_ON_RESUME", False))
    reset_best_on_resume = bool(getattr(config.TRAIN, "RESET_BEST_ON_RESUME", False))

    start_epoch = 0
    transfer_resume = False
    if config.MODEL.RESUME and os.path.exists(config.MODEL.RESUME):
        logger.info(f"Loading pretrained model from: {config.MODEL.RESUME}")
        try:
            checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

            ckpt_model = None
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                ckpt_model = checkpoint['model']
            elif isinstance(checkpoint, dict):
                ckpt_model = {k: v for k, v in checkpoint.items() if hasattr(v, 'shape')}

            if ckpt_model is not None:
                current_sd = model.state_dict()
                filtered_sd = {}
                skipped = []
                for k, v in ckpt_model.items():
                    if k in current_sd and getattr(v, 'shape', None) == getattr(current_sd[k], 'shape', None):
                        filtered_sd[k] = v
                    else:
                        skipped.append(k)
                model.load_state_dict(filtered_sd, strict=False)
                logger.info(f"Model weights loaded: {len(filtered_sd)} tensors")
                if skipped:
                    transfer_resume = True
                    logger.warning(f"Skipped {len(skipped)} model keys due to shape mismatch")
            else:
                logger.warning("Checkpoint has no recognizable model state_dict; skip model load")

            if isinstance(checkpoint, dict) and 'prototype_net' in checkpoint:
                try:
                    prototype_net.load_state_dict(checkpoint['prototype_net'], strict=False)
                    logger.info("Prototype network weights loaded")
                except Exception as e:
                    logger.warning(f"Failed to load prototype_net state: {e}")

            if training_memory_bank is not None and isinstance(checkpoint, dict) and 'training_memory_bank' in checkpoint:
                try:
                    training_memory_bank.load_state_dict(checkpoint['training_memory_bank'])
                    training_memory_bank.to(torch.device('cuda'))
                    logger.info(
                        f"Training memory bank loaded | classes={training_memory_bank.num_classes()} "
                        f"slots={training_memory_bank.num_slots()}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load training memory bank: {e}")

            if isinstance(checkpoint, dict) and 'epoch' in checkpoint and not transfer_resume:
                if reset_epoch_on_resume:
                    start_epoch = 0
                    logger.info("RESET_EPOCH_ON_RESUME=True -> start from epoch 0")
                else:
                    start_epoch = checkpoint['epoch'] + 1
                    logger.info(f"Resuming from epoch {start_epoch}")
            elif transfer_resume:
                start_epoch = 0
                logger.info("Detected shape mismatch, entering transfer finetune mode from epoch 0")

        except Exception as e:
            logger.warning(f"Failed to load pretrained model: {e}")
            logger.info("Starting training from scratch")
    else:
        logger.info("No pretrained model specified, initializing prototype network")

    def initialize_prototypes_for_openset_training():
        model.eval()
        prototype_net.train()
        class_features = {}
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader_train):
                if batch_idx >= 5:
                    break
                images = batch_data['image'].cuda()
                labels = batch_data['id_label'].cuda()
                feat_after_bn, _feat_before_bn = forward_reid_embeddings(model, images)
                feat_after_bn = F.normalize(feat_after_bn, p=2, dim=1)
                for feat, label in zip(feat_after_bn, labels):
                    lid = int(label.item())
                    class_features.setdefault(lid, [])
                    if len(class_features[lid]) < 3:
                        class_features[lid].append(feat)

        for class_id, feat_list in class_features.items():
            if len(feat_list) > 0:
                init_feat = feat_list[0]
                init_q = float(prototype_net.quality_net(init_feat.unsqueeze(0)).item())
                prototype_net.update_prototype(f'class_{class_id:03d}', init_feat, init_q)

        model.train()
        logger.info(f"Initialized {len(class_features)} training prototypes for discrimination warm-up")
        return len(class_features)

    def initialize_training_memory_bank():
        if training_memory_bank is None:
            return 0
        model.eval()
        bootstrap_batches = int(getattr(config.MODEL, 'MULTI_PROTO_BOOTSTRAP_BATCHES', 8) or 8)
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader_train):
                if batch_idx >= bootstrap_batches:
                    break
                images = batch_data['image'].cuda()
                labels = batch_data['id_label'].cuda()
                feat_after_bn, _feat_before_bn = forward_reid_embeddings(model, images)
                feat_after_bn = F.normalize(feat_after_bn, p=2, dim=1)
                quality_scores = prototype_net.quality_net(feat_after_bn).squeeze(1).float()
                training_memory_bank.update_batch(
                    feat_after_bn,
                    labels,
                    quality_scores=quality_scores,
                )
        model.train()
        logger.info(
            f"Initialized training multi-prototype memory | "
            f"classes={training_memory_bank.num_classes()} slots={training_memory_bank.num_slots()}"
        )
        return training_memory_bank.num_slots()

    if len(prototype_net.prototypes) == 0:
        initialize_prototypes_for_openset_training()
    else:
        logger.info(f"Using loaded prototypes: {len(prototype_net.prototypes)}")
    if training_memory_bank is not None and training_memory_bank.num_slots() == 0:
        initialize_training_memory_bank()

    criterion = CombinedLoss(
        in_features=model.feature_dim,
        num_classes=dataset_train.get_num_classes(),
        arcface_scale=getattr(config.MODEL, 'ARCFACE_SCALE', 16.0),
        arcface_margin=getattr(config.MODEL, 'ARCFACE_MARGIN', 0.25),
        triplet_margin=getattr(config.MODEL, 'TRIPLET_MARGIN', 0.4),
        arcface_weight=getattr(config.MODEL, 'ARCFACE_WEIGHT', 0.3),
        triplet_weight=getattr(config.MODEL, 'TRIPLET_WEIGHT', 1.0),
        center_weight=getattr(config.MODEL, 'CENTER_WEIGHT', 0.01),
        variance_weight=getattr(config.MODEL, 'VARIANCE_WEIGHT', 0.001),
        use_hard_mining=getattr(config.MODEL, 'HARD_MINING', True),
    )
    criterion.cuda()

    prototype_criterion = PrototypeLoss(temperature=0.05, margin=0.3)
    prototype_criterion.cuda()

    optimizer = build_optimizer(config, model)

    try:
        criterion_params = [p for p in criterion.parameters() if p.requires_grad]
        if len(criterion_params) > 0:
            optimizer.add_param_group({'params': criterion_params})
            logger.info(f"Added criterion params to optimizer: {sum(p.numel() for p in criterion_params):,}")
    except Exception as e:
        logger.warning(f"Failed to add criterion params: {e}")

    try:
        proto_params = [p for p in prototype_net.parameters() if p.requires_grad]
        if len(proto_params) > 0:
            proto_lr_mult = float(getattr(config.MODEL, 'PROTO_LR_MULT', 10.0))
            proto_lr = float(config.TRAIN.BASE_LR) * proto_lr_mult
            optimizer.add_param_group({'params': proto_params, 'lr': proto_lr})
            logger.info(
                f"Added prototype_net params to optimizer: {sum(p.numel() for p in proto_params):,}, "
                f"lr={proto_lr:.6g}"
            )
    except Exception as e:
        logger.warning(f"Failed to add prototype params: {e}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    loss_scaler = NativeScalerWithGradNormCount() if config.AMP_ENABLE else None

    best_accuracy = 0.0
    best_epoch = 0
    patience = 15
    no_improve_count = 0

    best_comprehensive_score = -1.0
    best_comprehensive_epoch = 0

    if config.MODEL.RESUME and os.path.exists(config.MODEL.RESUME) and not transfer_resume:
        try:
            checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
            if isinstance(checkpoint, dict) and 'criterion' in checkpoint:
                try:
                    ckpt_crit = checkpoint['criterion']
                    if isinstance(ckpt_crit, dict):
                        cur_sd = criterion.state_dict()
                        filtered = {
                            k: v for k, v in ckpt_crit.items()
                            if k in cur_sd and getattr(v, 'shape', None) == getattr(cur_sd[k], 'shape', None)
                        }
                        criterion.load_state_dict(filtered, strict=False)
                        logger.info(f"Criterion state loaded: {len(filtered)} tensors")
                except Exception as e:
                    logger.warning(f"Failed to load criterion state: {e}")

            if isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
                if reset_opt_on_resume:
                    logger.info("RESET_OPTIMIZER_ON_RESUME=True -> skip optimizer state load")
                else:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    logger.info("Optimizer state loaded")

            if isinstance(checkpoint, dict) and 'lr_scheduler' in checkpoint:
                if reset_sched_on_resume:
                    logger.info("RESET_LR_SCHEDULER_ON_RESUME=True -> skip LR scheduler state load")
                else:
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                    logger.info("LR scheduler state loaded")

            if isinstance(checkpoint, dict) and 'best_accuracy' in checkpoint:
                best_accuracy = float(checkpoint['best_accuracy'])

            if isinstance(checkpoint, dict) and 'best_comprehensive_score' in checkpoint:
                best_comprehensive_score = float(checkpoint['best_comprehensive_score'])

            if isinstance(checkpoint, dict) and 'best_epoch' in checkpoint:
                best_epoch = int(checkpoint['best_epoch'])
                best_comprehensive_epoch = int(checkpoint.get('epoch', best_epoch))

            if reset_best_on_resume:
                best_accuracy = 0.0
                best_epoch = 0
                best_comprehensive_score = -1.0
                best_comprehensive_epoch = 0
                no_improve_count = 0
                logger.info("RESET_BEST_ON_RESUME=True -> reset best-score tracking")

        except Exception as e:
            logger.warning(f"Failed to load optimizer/scheduler state: {e}")

    clustering_history = {
        'epochs': [],
        'silhouette_scores': [],
        'separation_ratios': [],
        'intra_distances': [],
        'inter_distances': [],
    }

    logger.info(f"Training loop: epochs {start_epoch} to {config.TRAIN.EPOCHS-1}")
    train_history = {
        "epochs": [],
        "train_loss": [],
        "eval_purity": [],
        "eval_assignment_acc": [],
    }

    start_time = time.time()

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_stats = train_one_epoch(
            config,
            model,
            prototype_net,
            criterion,
            data_loader_train,
            optimizer,
            epoch,
            lr_scheduler,
            loss_scaler,
            logger,
            prototype_criterion,
            training_memory_bank,
        )

        train_history["epochs"].append(int(epoch + 1))
        train_history["train_loss"].append(float(train_stats.get("loss", 0.0)))

        current_accuracy = 0.0
        predicted_ids = 0
        true_ids = 0
        clustering_metrics = {}
        openworld_metrics = {}
        eval_purity = np.nan
        eval_assignment_acc = np.nan
        stop_training = False

        eval_freq = getattr(args, 'eval_interval', 5)
        test_root = args.test_roiimg_root if args.test_roiimg_root else args.test_image_root
        test_roi_root = args.test_roi_root

        if (epoch + 1) % eval_freq == 0 and test_root:
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}: Prototype system evaluation (every {eval_freq} epochs)")
            logger.info(f"{'='*60}")

            eval_results = evaluate_prototype_system(
                model,
                prototype_net,
                test_root,
                test_roi_root,
                config,
                torch.cuda.current_device(),
                logger,
            )

            current_accuracy = float(eval_results.get('accuracy', 0.0))
            predicted_ids = int(eval_results.get('predicted_ids', 0))
            true_ids = int(eval_results.get('true_ids', 0))
            openworld_metrics = eval_results.get('cluster_metrics', {}) if isinstance(eval_results, dict) else {}
            eval_purity = float(eval_results.get('cluster_purity', np.nan))
            eval_assignment_acc = float(eval_results.get('assignment_accuracy', np.nan))

            target_purity_ok = (not np.isnan(eval_purity)) and (eval_purity >= 0.90)
            target_assign_ok = (not np.isnan(eval_assignment_acc)) and (eval_assignment_acc >= 0.90)
            logger.info(
                f"Target status | purity>=0.90:{target_purity_ok} assignment>=0.90:{target_assign_ok}"
            )

            logger.info(f"\n{'='*60}")
            logger.info(f"Feature Quality Analysis - Epoch {epoch + 1}")
            logger.info(f"{'='*60}")

            logger.info("Analyzing TRAINING set features...")
            train_clustering_metrics = analyze_feature_clustering(
                model,
                data_loader_train,
                torch.cuda.current_device(),
                logger,
                epoch + 1,
                config.OUTPUT,
                max_samples=300,
                dataset_name="train",
            )

            test_clustering_metrics = {}
            if test_root:
                logger.info("\nAnalyzing TEST set features...")
                test_data_loader = create_test_dataloader_for_analysis(
                    test_root,
                    test_roi_root,
                    config,
                    args.test_roiimg_root,
                )
                if test_data_loader:
                    test_clustering_metrics = analyze_feature_clustering(
                        model,
                        test_data_loader,
                        torch.cuda.current_device(),
                        logger,
                        epoch + 1,
                        config.OUTPUT,
                        max_samples=300,
                        dataset_name="test",
                    )

            clustering_metrics = test_clustering_metrics if test_clustering_metrics else train_clustering_metrics

            if clustering_metrics:
                clustering_history['epochs'].append(epoch + 1)
                clustering_history['silhouette_scores'].append(clustering_metrics.get('silhouette_score', 0))
                clustering_history['separation_ratios'].append(clustering_metrics.get('separation_ratio', 0))
                clustering_history['intra_distances'].append(clustering_metrics.get('avg_intra_distance', 0))
                clustering_history['inter_distances'].append(clustering_metrics.get('avg_inter_distance', 0))

            score_metrics = {}
            if isinstance(openworld_metrics, dict):
                score_metrics.update(openworld_metrics)
            if isinstance(clustering_metrics, dict):
                score_metrics.update(clustering_metrics)

            current_score, score_details = calculate_comprehensive_score(
                current_accuracy,
                predicted_ids,
                true_ids,
                clustering_metrics=score_metrics,
            )

            logger.info("Comprehensive Evaluation (open-world ReID):")
            logger.info(f"  Assignment Score: {score_details['assignment_score']:.4f}")
            logger.info(f"  Purity Score: {score_details['purity_score']:.4f}")
            logger.info(f"  ID Count Score: {score_details['id_score']:.4f}")
            logger.info(f"  Feature Quality Score: {score_details['feature_quality_score']:.4f}")
            logger.info(f"  Overall Score: {current_score:.4f}")
            logger.info(f"  Previous Best Score: {best_comprehensive_score:.4f} (Epoch {best_comprehensive_epoch})")

            if targeted_adaptive_tuning(config, criterion, prototype_net, eval_results, logger):
                logger.info("Adaptive tuning updated training weights for next epoch")

            if current_score > best_comprehensive_score:
                prev_best = best_comprehensive_score
                best_comprehensive_score = current_score
                best_comprehensive_epoch = epoch + 1
                best_accuracy = current_accuracy
                best_epoch = epoch + 1
                no_improve_count = 0

                save_dict = {
                    'model': model.state_dict(),
                    'prototype_net': prototype_net.state_dict(),
                    'training_memory_bank': training_memory_bank.state_dict() if training_memory_bank is not None else {},
                    'criterion': criterion.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'num_classes': getattr(model, 'num_classes', dataset_train.get_num_classes()),
                    'config': config,
                    'best_accuracy': best_accuracy,
                    'best_comprehensive_score': current_score,
                    'score_details': score_details,
                    'openworld_metrics': openworld_metrics if isinstance(openworld_metrics, dict) else {},
                    'feature_metrics': clustering_metrics if isinstance(clustering_metrics, dict) else {},
                }

                best_path = os.path.join(config.OUTPUT, 'prototype_best_model.pth')
                torch.save(save_dict, best_path)
                logger.info(f"New best model saved: {best_path}")
                logger.info(f"  New best comprehensive score: {current_score:.4f} (Epoch {epoch + 1})")
                logger.info(f"  Improvement: {current_score - (prev_best if prev_best > 0 else 0):.4f}")
            else:
                no_improve_count += 1

            logger.info(f"Current accuracy: {current_accuracy:.4f}")
            logger.info(f"Best accuracy: {best_accuracy:.4f} (Epoch {best_epoch})")
            logger.info(f"Best comprehensive score: {best_comprehensive_score:.4f} (Epoch {best_comprehensive_epoch})")
            logger.info(f"No improvement count: {no_improve_count}/{patience}")

            if no_improve_count >= patience:
                logger.info("Early stopping triggered")
                stop_training = True

        train_history["eval_purity"].append(float(eval_purity) if not np.isnan(eval_purity) else None)
        train_history["eval_assignment_acc"].append(
            float(eval_assignment_acc) if not np.isnan(eval_assignment_acc) else None
        )

        if stop_training:
            break

        save_every = getattr(args, 'save_interval', 5)
        if save_every and (epoch + 1) % save_every == 0:
            save_dict_epoch = {
                'model': model.state_dict(),
                'prototype_net': prototype_net.state_dict(),
                'training_memory_bank': training_memory_bank.state_dict() if training_memory_bank is not None else {},
                'criterion': criterion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'num_classes': getattr(model, 'num_classes', dataset_train.get_num_classes()),
                'config': config,
                'best_accuracy': best_accuracy,
                'best_comprehensive_score': best_comprehensive_score,
                'best_epoch': best_epoch,
                'openworld_metrics': openworld_metrics if isinstance(openworld_metrics, dict) else {},
                'feature_metrics': clustering_metrics if isinstance(clustering_metrics, dict) else {},
            }
            epoch_path = os.path.join(config.OUTPUT, f'prototype_epoch_{epoch + 1:03d}.pth')
            try:
                torch.save(save_dict_epoch, epoch_path)
                logger.info(f"Saved periodic checkpoint: {epoch_path}")
            except Exception as e:
                logger.warning(f"Failed to save periodic checkpoint at epoch {epoch + 1}: {e}")

    total_time = time.time() - start_time
    save_training_history_artifacts(train_history, config.OUTPUT, logger)
    logger.info(f"Training completed, total time: {datetime.timedelta(seconds=int(total_time))}")
    logger.info(f"Best accuracy: {best_accuracy:.4f} (Epoch {best_epoch})")


if __name__ == '__main__':
    main()
