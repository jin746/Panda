#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a dedicated age/gender specialist model on ROI images.

"""

import argparse
import datetime
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms

import matplotlib.pyplot as plt

from data.roiimg_dataset import PandaRoiImgDataset

try:
    import timm
except Exception as ex:  # pragma: no cover
    timm = None
    _TIMM_IMPORT_ERROR = ex
else:
    _TIMM_IMPORT_ERROR = None


def parse_args():
    parser = argparse.ArgumentParser("Age/Gender specialist training")
    parser.add_argument("--train-roiimg-root", type=str, required=True)
    parser.add_argument("--val-roiimg-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output_age_gender_specialist")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--backbone-name", type=str, default="convnextv2_base.fcmae_ft_in22k_in1k")
    parser.add_argument("--backbone-weights", type=str, default="modelori/pytorch_model.bin")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--gender-loss-weight", type=float, default=1.0)
    parser.add_argument("--age-ce-weight", type=float, default=1.0)
    parser.add_argument("--age-reg-weight", type=float, default=2.0)
    parser.add_argument("--age-huber-beta", type=float, default=1.0)
    parser.add_argument("--max-age-bin", type=int, default=40)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--disable-channels-last", action="store_true")
    parser.add_argument("--embedding-max-points", type=int, default=2000)
    parser.add_argument("--balanced-gender-sampler", action="store_true")
    parser.add_argument(
        "--sampler-mode",
        type=str,
        default="gender_age",
        choices=["none", "gender", "age", "gender_age"],
        help="training sampler policy",
    )
    parser.add_argument("--age-sampler-bin-size", type=int, default=2)
    parser.add_argument("--age-sampler-power", type=float, default=0.5)
    parser.add_argument("--age-ldl-weight", type=float, default=0.6)
    parser.add_argument("--age-ldl-sigma", type=float, default=1.5)
    parser.add_argument("--age-mean-align-weight", type=float, default=0.15)
    parser.add_argument("--age-loss-ramp-epochs", type=int, default=2)
    parser.add_argument(
        "--age-expected-mix",
        type=float,
        default=0.6,
        help="final age prediction = mix*E[age_logits] + (1-mix)*age_reg_head",
    )
    parser.add_argument("--disable-age-calibration", action="store_true")
    parser.add_argument("--calibration-min-count", type=int, default=20)
    parser.add_argument("--age-bin-report-width", type=int, default=5)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def build_train_transform(img_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize((img_size + 24, img_size + 24)),
            transforms.RandomResizedCrop(
                (img_size, img_size), scale=(0.65, 1.0), ratio=(0.8, 1.25)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.15, value=0),
        ]
    )


def build_eval_transform(img_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def maybe_subsample_dataset(dataset: PandaRoiImgDataset, max_samples: int, seed: int):
    if max_samples is None or int(max_samples) <= 0 or len(dataset.samples) <= int(max_samples):
        return
    rng = np.random.default_rng(seed)
    idx = np.arange(len(dataset.samples))
    rng.shuffle(idx)
    keep = idx[: int(max_samples)]
    keep_set = set(int(x) for x in keep.tolist())
    dataset.samples = [s for i, s in enumerate(dataset.samples) if i in keep_set]
    dataset.id_to_label = dataset._build_id_mapping()


def compute_gender_class_weights(dataset: PandaRoiImgDataset) -> torch.Tensor:
    counts = np.zeros(2, dtype=np.int64)
    for s in dataset.samples:
        g = int(s.get("gender_label", 0))
        if 0 <= g < 2:
            counts[g] += 1
    counts = np.maximum(counts, 1)
    total = float(counts.sum())
    w = np.array([total / (2.0 * counts[0]), total / (2.0 * counts[1])], dtype=np.float32)
    w = np.sqrt(w)
    w = w / max(float(w.mean()), 1e-8)
    w = np.clip(w, 0.25, 5.0)
    return torch.tensor(w, dtype=torch.float32)


def build_gender_balanced_sampler(dataset: PandaRoiImgDataset):
    labels = []
    for s in dataset.samples:
        g = int(s.get("gender_label", 0))
        labels.append(1 if g == 1 else 0)
    labels = np.array(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    class_w = 1.0 / counts
    sample_w = class_w[labels]
    sample_w = sample_w / max(sample_w.mean(), 1e-12)
    sample_w = torch.tensor(sample_w, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=sample_w,
        num_samples=len(labels),
        replacement=True,
    )
    return sampler


def build_balanced_sampler(
    dataset: PandaRoiImgDataset,
    mode: str,
    max_age_bin: int,
    age_bin_size: int = 2,
    power: float = 0.5,
):
    mode = str(mode or "none").lower().strip()
    if mode == "none":
        return None

    bin_size = max(1, int(age_bin_size))
    p = float(max(power, 0.0))
    genders = []
    age_bins = []
    keys = []
    for s in dataset.samples:
        g = 1 if int(s.get("gender_label", 0)) == 1 else 0
        age = s.get("age_years", None)
        if age is None:
            age_bin = -1
        else:
            a = int(np.clip(round(float(age)), 0, int(max_age_bin)))
            age_bin = int(a // bin_size)
        genders.append(g)
        age_bins.append(age_bin)
        if mode == "gender":
            key = ("g", g)
        elif mode == "age":
            key = ("a", age_bin)
        else:
            key = ("ga", g, age_bin)
        keys.append(key)

    if mode == "gender_age":
        g_arr = np.asarray(genders, dtype=np.int64)
        a_arr = np.asarray(age_bins, dtype=np.int64)
        g_cnt = np.bincount(g_arr, minlength=2).astype(np.float64)
        g_cnt = np.maximum(g_cnt, 1.0)
        g_w = 1.0 / g_cnt[g_arr]
        age_unique, age_counts = np.unique(a_arr, return_counts=True)
        age_freq = {int(k): float(v) for k, v in zip(age_unique.tolist(), age_counts.tolist())}
        a_w = np.array([1.0 / max(age_freq[int(v)], 1.0) for v in a_arr], dtype=np.float64)
        weights = np.power(np.sqrt(g_w * a_w), p)
    else:
        freq: Dict = {}
        for k in keys:
            freq[k] = int(freq.get(k, 0)) + 1
        weights = np.array(
            [np.power(1.0 / max(float(freq[k]), 1.0), p) for k in keys],
            dtype=np.float64,
        )
    weights = weights / max(float(weights.mean()), 1e-12)
    weights = np.clip(weights, 0.2, 10.0)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_prob).sum(dim=1).mean()


def build_age_soft_targets(
    age_values: torch.Tensor,
    age_targets: torch.Tensor,
    sigma: float,
    class_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    sig = max(float(sigma), 1e-4)
    dist2 = (age_values.unsqueeze(0) - age_targets.unsqueeze(1)) ** 2
    probs = torch.exp(-0.5 * dist2 / (sig * sig))
    if class_weight is not None:
        probs = probs * class_weight.unsqueeze(0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return probs


def compute_age_class_weights(dataset: PandaRoiImgDataset, max_age_bin: int) -> torch.Tensor:
    counts = np.zeros(max_age_bin + 1, dtype=np.int64)
    for s in dataset.samples:
        age = s.get("age_years", None)
        if age is None:
            continue
        a = int(np.clip(round(float(age)), 0, max_age_bin))
        counts[a] += 1
    counts = np.maximum(counts, 1)
    total = float(counts.sum())
    num_cls = len(counts)
    w = np.array([total / (num_cls * c) for c in counts], dtype=np.float32)
    w = np.sqrt(w)
    w = w / max(float(w.mean()), 1e-8)
    w = np.clip(w, 0.2, 8.0)
    return torch.tensor(w, dtype=torch.float32)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
):
    def lr_lambda(cur_epoch: int):
        if cur_epoch < max(warmup_epochs, 1):
            return float(cur_epoch + 1) / float(max(warmup_epochs, 1))
        progress = (cur_epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        floor = min_lr / max(optimizer.param_groups[0]["lr"], 1e-12)
        return max(float(cosine), float(floor))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)



class AgeGenderSpecialist(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_age_bins: int,
        dropout: float = 0.2,
        age_expected_mix: float = 0.6,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError(f"timm import failed: {_TIMM_IMPORT_ERROR}")
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0, global_pool="avg"
        )
        self.feature_dim = int(getattr(self.backbone, "num_features", 0))
        if self.feature_dim <= 0:
            raise RuntimeError(f"Cannot infer num_features from backbone: {backbone_name}")

        self.neck = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(p=float(dropout)),
        )
        self.gender_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(256, 2),
        )
        self.age_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(512, num_age_bins),
        )
        self.age_reg_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(256, 1),
        )
        self.num_age_bins = int(num_age_bins)
        self.age_expected_mix = float(np.clip(float(age_expected_mix), 0.0, 1.0))
        self.register_buffer("age_values", torch.arange(self.num_age_bins, dtype=torch.float32))
        self.register_buffer("age_calib_slope", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("age_calib_bias", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer(
            "age_calib_bin_residual",
            torch.zeros(self.num_age_bins, dtype=torch.float32),
        )
        self.register_buffer("age_calib_enabled", torch.tensor(0, dtype=torch.uint8))
        self._init_heads()

    def _init_heads(self):
        for m in (
            list(self.gender_head.modules())
            + list(self.age_head.modules())
            + list(self.age_reg_head.modules())
        ):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _forward_backbone(self, x: torch.Tensor):
        feat = self.backbone(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[-1]
        if feat.dim() == 4:
            feat = F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)
        return feat

    def set_age_calibration(self, calibration: Optional[Dict]):
        if calibration is None:
            self.age_calib_slope.fill_(1.0)
            self.age_calib_bias.fill_(0.0)
            self.age_calib_bin_residual.zero_()
            self.age_calib_enabled.fill_(0)
            return

        slope = float(calibration.get("slope", 1.0))
        bias = float(calibration.get("bias", 0.0))
        residual = calibration.get("bin_residual", None)
        self.age_calib_slope.fill_(slope)
        self.age_calib_bias.fill_(bias)
        self.age_calib_bin_residual.zero_()
        if isinstance(residual, (list, tuple, np.ndarray)):
            arr = np.asarray(residual, dtype=np.float32).reshape(-1)
            n = min(int(arr.shape[0]), int(self.age_calib_bin_residual.shape[0]))
            if n > 0:
                self.age_calib_bin_residual[:n] = torch.from_numpy(arr[:n]).to(
                    self.age_calib_bin_residual.device
                )
        self.age_calib_enabled.fill_(1)

    def apply_age_calibration(self, age_pred: torch.Tensor) -> torch.Tensor:
        if int(self.age_calib_enabled.item()) <= 0:
            return age_pred
        slope = self.age_calib_slope.to(device=age_pred.device, dtype=age_pred.dtype)
        bias = self.age_calib_bias.to(device=age_pred.device, dtype=age_pred.dtype)
        y = age_pred * slope + bias
        if self.age_calib_bin_residual.numel() > 0:
            residual = self.age_calib_bin_residual.to(device=age_pred.device, dtype=age_pred.dtype)
            idx = torch.clamp(torch.round(y).long(), 0, residual.shape[0] - 1)
            y = y + residual[idx]
        return torch.clamp(y, 0.0, float(self.num_age_bins - 1))

    def forward(self, x: torch.Tensor):
        feat = self._forward_backbone(x)
        feat = self.neck(feat)
        gender_logits = self.gender_head(feat)
        age_logits = self.age_head(feat)
        age_reg = self.age_reg_head(feat).squeeze(1)
        age_prob = torch.softmax(age_logits, dim=1)
        age_expect = (age_prob * self.age_values.unsqueeze(0)).sum(dim=1)
        age_pred_raw = self.age_expected_mix * age_expect + (1.0 - self.age_expected_mix) * age_reg
        age_pred_raw = torch.clamp(age_pred_raw, 0.0, float(self.num_age_bins - 1))
        age_pred = self.apply_age_calibration(age_pred_raw)
        return feat, gender_logits, age_logits, age_pred


def load_local_weights(model: AgeGenderSpecialist, weight_path: str):
    if not weight_path or (not os.path.exists(weight_path)):
        return
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        elif "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
    if not isinstance(state, dict):
        return
    cleaned = {}
    for k, v in state.items():
        nk = str(k)
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        cleaned[nk] = v
    missing, unexpected = model.backbone.load_state_dict(cleaned, strict=False)
    print(
        f"[INFO] Loaded local backbone weights: {weight_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )


@dataclass
class EpochStats:
    train_loss: float
    train_gender_loss: float
    train_age_ce_loss: float
    train_age_reg_loss: float
    train_age_ldl_loss: float
    train_age_mean_align_loss: float
    val_gender_acc: float
    val_age_mae_raw: float
    val_age_rmse_raw: float
    val_age_mae: float
    val_age_rmse: float
    val_age_bias: float
    val_age_within_1: float
    val_age_within_2: float
    val_age_within_3: float


def compute_age_metrics_from_arrays(age_true: np.ndarray, age_pred: np.ndarray) -> Dict[str, float]:
    if age_true.size <= 0 or age_pred.size <= 0:
        return {
            "age_mae": 999.0,
            "age_rmse": 999.0,
            "age_bias": 0.0,
            "age_within_1": 0.0,
            "age_within_2": 0.0,
            "age_within_3": 0.0,
        }
    diff = (age_pred - age_true).astype(np.float32, copy=False)
    ad = np.abs(diff)
    return {
        "age_mae": float(np.mean(ad)),
        "age_rmse": float(np.sqrt(np.mean(diff * diff))),
        "age_bias": float(np.mean(diff)),
        "age_within_1": float(np.mean(ad <= 1.0)),
        "age_within_2": float(np.mean(ad <= 2.0)),
        "age_within_3": float(np.mean(ad <= 3.0)),
    }


def _smooth_residual_1d(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32).copy()
    if x.size <= 1:
        return x
    y = x.copy()
    if x.size == 2:
        y[0] = 0.75 * x[0] + 0.25 * x[1]
        y[1] = 0.25 * x[0] + 0.75 * x[1]
        return y
    y[0] = 0.75 * x[0] + 0.25 * x[1]
    y[-1] = 0.75 * x[-1] + 0.25 * x[-2]
    y[1:-1] = 0.25 * x[:-2] + 0.5 * x[1:-1] + 0.25 * x[2:]
    return y


def fit_age_calibration(
    age_pred_raw: np.ndarray,
    age_true: np.ndarray,
    max_age_bin: int,
    min_count: int = 20,
) -> Dict[str, object]:
    x = np.asarray(age_pred_raw, dtype=np.float32).reshape(-1)
    y = np.asarray(age_true, dtype=np.float32).reshape(-1)
    max_bin = int(max_age_bin)
    identity = {
        "slope": 1.0,
        "bias": 0.0,
        "bin_residual": [0.0 for _ in range(max_bin + 1)],
    }
    if x.size < 8 or y.size < 8:
        return identity

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    var_x = float(np.mean((x - x_mean) ** 2))
    if var_x < 1e-8:
        slope = 1.0
    else:
        cov_xy = float(np.mean((x - x_mean) * (y - y_mean)))
        slope = cov_xy / max(var_x, 1e-8)
    bias = y_mean - slope * x_mean

    slope = float(np.clip(slope, 0.5, 1.8))
    bias = float(np.clip(bias, -8.0, 8.0))
    pred_lin = slope * x + bias
    pred_bin = np.clip(np.rint(pred_lin), 0, max_bin).astype(np.int64)
    residual = np.zeros(max_bin + 1, dtype=np.float32)
    for b in range(max_bin + 1):
        m = pred_bin == b
        if int(m.sum()) >= int(min_count):
            residual[b] = float(np.mean(y[m] - pred_lin[m]))
    residual = _smooth_residual_1d(_smooth_residual_1d(residual))
    residual = np.clip(residual, -4.0, 4.0)
    return {
        "slope": float(slope),
        "bias": float(bias),
        "bin_residual": residual.tolist(),
    }


def apply_age_calibration_numpy(
    age_pred_raw: np.ndarray,
    calibration: Optional[Dict[str, object]],
    max_age_bin: int,
) -> np.ndarray:
    y = np.asarray(age_pred_raw, dtype=np.float32).reshape(-1)
    max_bin = int(max_age_bin)
    if calibration is None:
        return np.clip(y, 0.0, float(max_bin))
    slope = float(calibration.get("slope", 1.0))
    bias = float(calibration.get("bias", 0.0))
    residual = np.asarray(calibration.get("bin_residual", []), dtype=np.float32).reshape(-1)
    out = slope * y + bias
    if residual.size >= (max_bin + 1):
        idx = np.clip(np.rint(out), 0, max_bin).astype(np.int64)
        out = out + residual[idx]
    return np.clip(out, 0.0, float(max_bin))


def build_age_bin_report(
    age_true: np.ndarray,
    age_pred: np.ndarray,
    max_age_bin: int,
    width: int = 5,
) -> List[Dict[str, object]]:
    w = max(1, int(width))
    rows = []
    if age_true.size <= 0:
        return rows
    for lo in range(0, int(max_age_bin) + 1, w):
        hi = min(int(max_age_bin), lo + w - 1)
        m = (age_true >= float(lo)) & (age_true <= float(hi))
        cnt = int(np.sum(m))
        if cnt <= 0:
            continue
        diff = (age_pred[m] - age_true[m]).astype(np.float32, copy=False)
        rows.append(
            {
                "age_range": f"{lo}-{hi}",
                "count": cnt,
                "mae": float(np.mean(np.abs(diff))),
                "bias": float(np.mean(diff)),
            }
        )
    return rows


def save_age_diagnostics(
    output_dir: str,
    prefix: str,
    age_true: np.ndarray,
    age_pred_raw: np.ndarray,
    age_pred_cal: np.ndarray,
    max_age_bin: int,
    report_width: int = 5,
):
    if age_true.size <= 0:
        return
    max_bin = int(max_age_bin)
    true_hist = np.bincount(np.clip(np.rint(age_true), 0, max_bin).astype(np.int64), minlength=max_bin + 1)
    raw_hist = np.bincount(np.clip(np.rint(age_pred_raw), 0, max_bin).astype(np.int64), minlength=max_bin + 1)
    cal_hist = np.bincount(np.clip(np.rint(age_pred_cal), 0, max_bin).astype(np.int64), minlength=max_bin + 1)
    xs = np.arange(max_bin + 1, dtype=np.int64)

    plt.figure(figsize=(11, 5))
    plt.plot(xs, true_hist, label="GT", linewidth=2)
    plt.plot(xs, raw_hist, label="Pred(raw)", linewidth=1.8)
    plt.plot(xs, cal_hist, label="Pred(calibrated)", linewidth=1.8)
    plt.xlabel("Age bin")
    plt.ylabel("Count")
    plt.title(f"Age Distribution Comparison ({prefix})")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"age_distribution_{prefix}.png"), dpi=160, bbox_inches="tight")
    plt.close()

    payload = {
        "prefix": prefix,
        "summary_raw": compute_age_metrics_from_arrays(age_true, age_pred_raw),
        "summary_calibrated": compute_age_metrics_from_arrays(age_true, age_pred_cal),
        "bin_report_raw": build_age_bin_report(
            age_true=age_true,
            age_pred=age_pred_raw,
            max_age_bin=max_age_bin,
            width=report_width,
        ),
        "bin_report_calibrated": build_age_bin_report(
            age_true=age_true,
            age_pred=age_pred_cal,
            max_age_bin=max_age_bin,
            width=report_width,
        ),
    }
    with open(
        os.path.join(output_dir, f"age_diagnostics_{prefix}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_eval(
    model,
    loader,
    device,
    max_age_bin: int,
    age_calibration: Optional[Dict[str, object]] = None,
    return_arrays: bool = False,
) -> Dict[str, object]:
    model.eval()
    g_correct = 0
    g_total = 0
    age_true = []
    age_pred_raw = []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            gender_t = batch["gender_label"].to(device, non_blocking=True).long()
            age_t = batch["age_years"].to(device, non_blocking=True).float()
            age_valid = batch["age_valid"].to(device, non_blocking=True).float() > 0.5

            _feat, gender_logits, _age_logits, age_pred = model(x)
            gender_p = torch.argmax(gender_logits, dim=1)
            g_correct += int((gender_p == gender_t).sum().item())
            g_total += int(gender_t.numel())

            if age_valid.any():
                age_true.extend(age_t[age_valid].float().cpu().numpy().tolist())
                age_pred_raw.extend(age_pred[age_valid].float().cpu().numpy().tolist())

    gender_acc = float(g_correct / max(g_total, 1))
    if age_true:
        age_true_np = np.asarray(age_true, dtype=np.float32)
        age_pred_raw_np = np.asarray(age_pred_raw, dtype=np.float32)
        age_pred_np = apply_age_calibration_numpy(
            age_pred_raw=age_pred_raw_np,
            calibration=age_calibration,
            max_age_bin=max_age_bin,
        )
        raw_metrics = compute_age_metrics_from_arrays(age_true_np, age_pred_raw_np)
        cal_metrics = compute_age_metrics_from_arrays(age_true_np, age_pred_np)
    else:
        age_true_np = np.zeros((0,), dtype=np.float32)
        age_pred_raw_np = np.zeros((0,), dtype=np.float32)
        age_pred_np = np.zeros((0,), dtype=np.float32)
        raw_metrics = compute_age_metrics_from_arrays(age_true_np, age_pred_raw_np)
        cal_metrics = raw_metrics

    stats: Dict[str, object] = {
        "gender_acc": gender_acc,
        "age_mae_raw": float(raw_metrics["age_mae"]),
        "age_rmse_raw": float(raw_metrics["age_rmse"]),
        "age_bias_raw": float(raw_metrics["age_bias"]),
        "age_mae": float(cal_metrics["age_mae"]),
        "age_rmse": float(cal_metrics["age_rmse"]),
        "age_bias": float(cal_metrics["age_bias"]),
        "age_within_1": float(cal_metrics["age_within_1"]),
        "age_within_2": float(cal_metrics["age_within_2"]),
        "age_within_3": float(cal_metrics["age_within_3"]),
        "age_count": int(age_true_np.shape[0]),
    }
    if return_arrays:
        stats["age_true_np"] = age_true_np
        stats["age_pred_raw_np"] = age_pred_raw_np
        stats["age_pred_np"] = age_pred_np
    return stats


def plot_curves(history: List[Dict], out_path: str):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_gender = [h["val_gender_acc"] for h in history]
    val_age_mae = [h["val_age_mae"] for h in history]
    val_age_mae_raw = [h.get("val_age_mae_raw", h["val_age_mae"]) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, train_loss, "b-o")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    axes[1].plot(epochs, val_gender, "g-o")
    axes[1].axhline(0.90, color="r", linestyle="--", linewidth=1)
    axes[1].set_title("Val Gender Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True)

    axes[2].plot(epochs, val_age_mae_raw, "c--o", label="raw")
    axes[2].plot(epochs, val_age_mae, "m-o", label="calibrated")
    axes[2].axhline(3.0, color="r", linestyle="--", linewidth=1)
    axes[2].set_title("Val Age MAE")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MAE (years)")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_embedding_visualizations(
    model: AgeGenderSpecialist,
    loader: DataLoader,
    device: torch.device,
    output_dir: str,
    max_points: int = 2000,
):
    features = []
    genders = []
    ages = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            feat, _g, _al, _ap = model(x)
            features.append(feat.cpu().numpy())
            genders.append(batch["gender_label"].numpy())
            ages.append(batch["age_years"].numpy())
            if sum(arr.shape[0] for arr in features) >= int(max_points):
                break

    if not features:
        return
    X = np.concatenate(features, axis=0)
    G = np.concatenate(genders, axis=0)
    A = np.concatenate(ages, axis=0)
    if X.shape[0] > max_points:
        idx = np.random.default_rng(42).choice(X.shape[0], size=max_points, replace=False)
        X, G, A = X[idx], G[idx], A[idx]

    # t-SNE
    try:
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, X.shape[0] // 15)))
        z = tsne.fit_transform(X.astype(np.float32, copy=False))
        plt.figure(figsize=(9, 7))
        plt.scatter(z[:, 0], z[:, 1], c=G, cmap="coolwarm", s=8, alpha=0.7)
        plt.title("Embedding t-SNE (color=gender)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "embedding_tsne_gender.png"), dpi=160)
        plt.close()
    except Exception as ex:
        print(f"[WARN] t-SNE visualization failed: {ex}")

    # UMAP
    try:
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        from umap import UMAP

        umap = UMAP(
            n_components=2,
            n_neighbors=min(30, max(5, X.shape[0] // 25)),
            min_dist=0.15,
            metric="cosine",
            random_state=42,
        )
        z = umap.fit_transform(X.astype(np.float32, copy=False))
        plt.figure(figsize=(9, 7))
        sc = plt.scatter(z[:, 0], z[:, 1], c=A, cmap="viridis", s=8, alpha=0.7)
        plt.colorbar(sc, label="Age")
        plt.title("Embedding UMAP (color=age)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "embedding_umap_age.png"), dpi=160)
        plt.close()
    except Exception as ex:
        print(f"[WARN] UMAP visualization failed: {ex}")


def main():
    args = parse_args()
    set_seed(int(args.seed))
    safe_mkdir(args.output_dir)

    if timm is None:
        raise RuntimeError(f"timm is required but import failed: {_TIMM_IMPORT_ERROR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (not args.disable_amp) and torch.cuda.is_available()
    channels_last = (not args.disable_channels_last)

    train_tf = build_train_transform(args.img_size)
    val_tf = build_eval_transform(args.img_size)

    train_dataset = PandaRoiImgDataset(
        roiimg_root=args.train_roiimg_root,
        img_size=args.img_size,
        transform=train_tf,
        is_train=True,
    )
    val_dataset = PandaRoiImgDataset(
        roiimg_root=args.val_roiimg_root,
        img_size=args.img_size,
        transform=val_tf,
        is_train=False,
    )

    maybe_subsample_dataset(train_dataset, args.max_train_samples, args.seed)
    maybe_subsample_dataset(val_dataset, args.max_val_samples, args.seed + 1)

    print(f"[INFO] train samples={len(train_dataset)}")
    print(f"[INFO] val samples={len(val_dataset)}")

    gender_w = compute_gender_class_weights(train_dataset)
    age_w = compute_age_class_weights(train_dataset, args.max_age_bin)
    print(f"[INFO] gender class weights: {gender_w.tolist()}")
    print(f"[INFO] age class weights shape: {tuple(age_w.shape)}")

    loader_kwargs = {}
    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    sampler_mode = str(args.sampler_mode or "none").lower()
    if args.balanced_gender_sampler and sampler_mode == "none":
        sampler_mode = "gender"
    train_sampler = build_balanced_sampler(
        dataset=train_dataset,
        mode=sampler_mode,
        max_age_bin=int(args.max_age_bin),
        age_bin_size=int(args.age_sampler_bin_size),
        power=float(args.age_sampler_power),
    )
    print(
        f"[INFO] sampler_mode={sampler_mode}, "
        f"age_sampler_bin_size={int(args.age_sampler_bin_size)}, "
        f"age_sampler_power={float(args.age_sampler_power):.3f}"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        drop_last=False,
        **({} if int(args.num_workers) == 0 else {"persistent_workers": True, "prefetch_factor": 2}),
    )

    model = AgeGenderSpecialist(
        backbone_name=args.backbone_name,
        num_age_bins=args.max_age_bin + 1,
        dropout=float(args.dropout),
        age_expected_mix=float(args.age_expected_mix),
    )
    load_local_weights(model, args.backbone_weights)
    model = model.to(device)
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.999),
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        epochs=int(args.epochs),
        warmup_epochs=int(args.warmup_epochs),
        min_lr=float(args.min_lr),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    gender_w = gender_w.to(device)
    age_w = age_w.to(device)

    history = []
    best_score = -1e9
    best_epoch = 0
    best_path = os.path.join(args.output_dir, "age_gender_specialist_best.pth")
    best_age_calibration = None
    use_age_calibration = not bool(args.disable_age_calibration)

    print(f"[INFO] device={device}, amp={amp_enabled}, channels_last={channels_last}")
    print(f"[INFO] start training: epochs={args.epochs}, batch={args.batch_size}")
    start_all = datetime.datetime.now()

    accumulation = max(1, int(args.accumulation_steps))
    for epoch in range(1, int(args.epochs) + 1):
        model.set_age_calibration(None)
        model.train()
        total_loss_meter = 0.0
        gender_loss_meter = 0.0
        age_ce_meter = 0.0
        age_reg_meter = 0.0
        age_ldl_meter = 0.0
        age_mean_align_meter = 0.0
        sample_meter = 0
        age_loss_scale = min(1.0, float(epoch) / float(max(1, int(args.age_loss_ramp_epochs))))

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            x = batch["image"].to(device, non_blocking=True)
            if channels_last and device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)

            gender_t = batch["gender_label"].to(device, non_blocking=True).long()
            age_t = batch["age_years"].to(device, non_blocking=True).float()
            age_valid = batch["age_valid"].to(device, non_blocking=True).float() > 0.5

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                _feat, gender_logits, age_logits, age_pred = model(x)

                gender_loss = F.cross_entropy(
                    gender_logits,
                    gender_t,
                    weight=gender_w,
                    label_smoothing=float(args.label_smoothing),
                )

                if age_valid.any():
                    age_logits_v = age_logits[age_valid]
                    age_t_v = age_t[age_valid]
                    age_pred_v = age_pred[age_valid]
                    age_target_cls = torch.clamp(
                        torch.round(age_t_v).long(), min=0, max=args.max_age_bin
                    )
                    age_ce_loss = F.cross_entropy(
                        age_logits_v,
                        age_target_cls,
                        weight=age_w,
                        label_smoothing=float(args.label_smoothing),
                    )
                    if float(args.age_ldl_weight) > 0:
                        age_soft_t = build_age_soft_targets(
                            age_values=model.age_values.to(age_logits_v.device, dtype=age_logits_v.dtype),
                            age_targets=age_t_v,
                            sigma=float(args.age_ldl_sigma),
                            class_weight=age_w.to(age_logits_v.device, dtype=age_logits_v.dtype),
                        )
                        age_ldl_loss = soft_cross_entropy(age_logits_v, age_soft_t)
                    else:
                        age_ldl_loss = torch.zeros([], device=device)
                    age_reg_loss = F.smooth_l1_loss(
                        age_pred_v,
                        age_t_v,
                        beta=float(args.age_huber_beta),
                    )
                    age_mean_align_loss = torch.abs(age_pred_v.mean() - age_t_v.mean())
                else:
                    age_ce_loss = torch.zeros([], device=device)
                    age_reg_loss = torch.zeros([], device=device)
                    age_ldl_loss = torch.zeros([], device=device)
                    age_mean_align_loss = torch.zeros([], device=device)

                loss = (
                    float(args.gender_loss_weight) * gender_loss
                    + age_loss_scale
                    * float(args.age_ce_weight)
                    * (age_ce_loss + float(args.age_ldl_weight) * age_ldl_loss)
                    + age_loss_scale * float(args.age_reg_weight) * age_reg_loss
                    + age_loss_scale * float(args.age_mean_align_weight) * age_mean_align_loss
                ) / accumulation

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if step % accumulation == 0:
                if float(args.grad_clip) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bsz = int(x.size(0))
            sample_meter += bsz
            total_loss_meter += float(loss.item()) * accumulation * bsz
            gender_loss_meter += float(gender_loss.item()) * bsz
            age_ce_meter += float(age_ce_loss.item()) * bsz
            age_reg_meter += float(age_reg_loss.item()) * bsz
            age_ldl_meter += float(age_ldl_loss.item()) * bsz
            age_mean_align_meter += float(age_mean_align_loss.item()) * bsz

            if step % max(1, int(args.print_freq)) == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[epoch {epoch:02d}] step {step:04d}/{len(train_loader):04d} "
                    f"lr={lr:.6g} "
                    f"loss={total_loss_meter/max(sample_meter,1):.4f} "
                    f"g={gender_loss_meter/max(sample_meter,1):.4f} "
                    f"age_scale={age_loss_scale:.2f} "
                    f"age_ce={age_ce_meter/max(sample_meter,1):.4f} "
                    f"age_ldl={age_ldl_meter/max(sample_meter,1):.4f} "
                    f"age_reg={age_reg_meter/max(sample_meter,1):.4f} "
                    f"age_align={age_mean_align_meter/max(sample_meter,1):.4f}"
                )

        scheduler.step()

        eval_stats = run_eval(
            model=model,
            loader=val_loader,
            device=device,
            max_age_bin=int(args.max_age_bin),
            age_calibration=None,
            return_arrays=True,
        )
        age_true_np = np.asarray(eval_stats.pop("age_true_np"), dtype=np.float32)
        age_pred_raw_np = np.asarray(eval_stats.pop("age_pred_raw_np"), dtype=np.float32)
        age_pred_np = np.asarray(eval_stats.pop("age_pred_np"), dtype=np.float32)

        age_calibration = None
        if use_age_calibration and age_true_np.size > 0:
            age_calibration = fit_age_calibration(
                age_pred_raw=age_pred_raw_np,
                age_true=age_true_np,
                max_age_bin=int(args.max_age_bin),
                min_count=int(args.calibration_min_count),
            )
            age_pred_np = apply_age_calibration_numpy(
                age_pred_raw=age_pred_raw_np,
                calibration=age_calibration,
                max_age_bin=int(args.max_age_bin),
            )
            cal_metrics = compute_age_metrics_from_arrays(age_true_np, age_pred_np)
            eval_stats["age_mae"] = float(cal_metrics["age_mae"])
            eval_stats["age_rmse"] = float(cal_metrics["age_rmse"])
            eval_stats["age_bias"] = float(cal_metrics["age_bias"])
            eval_stats["age_within_1"] = float(cal_metrics["age_within_1"])
            eval_stats["age_within_2"] = float(cal_metrics["age_within_2"])
            eval_stats["age_within_3"] = float(cal_metrics["age_within_3"])

        save_age_diagnostics(
            output_dir=args.output_dir,
            prefix=f"epoch_{epoch:03d}",
            age_true=age_true_np,
            age_pred_raw=age_pred_raw_np,
            age_pred_cal=age_pred_np,
            max_age_bin=int(args.max_age_bin),
            report_width=int(args.age_bin_report_width),
        )

        epoch_stats = EpochStats(
            train_loss=float(total_loss_meter / max(sample_meter, 1)),
            train_gender_loss=float(gender_loss_meter / max(sample_meter, 1)),
            train_age_ce_loss=float(age_ce_meter / max(sample_meter, 1)),
            train_age_reg_loss=float(age_reg_meter / max(sample_meter, 1)),
            train_age_ldl_loss=float(age_ldl_meter / max(sample_meter, 1)),
            train_age_mean_align_loss=float(age_mean_align_meter / max(sample_meter, 1)),
            val_gender_acc=float(eval_stats["gender_acc"]),
            val_age_mae_raw=float(eval_stats["age_mae_raw"]),
            val_age_rmse_raw=float(eval_stats["age_rmse_raw"]),
            val_age_mae=float(eval_stats["age_mae"]),
            val_age_rmse=float(eval_stats["age_rmse"]),
            val_age_bias=float(eval_stats["age_bias"]),
            val_age_within_1=float(eval_stats["age_within_1"]),
            val_age_within_2=float(eval_stats["age_within_2"]),
            val_age_within_3=float(eval_stats["age_within_3"]),
        )
        row = {
            "epoch": int(epoch),
            **epoch_stats.__dict__,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "age_count": int(eval_stats.get("age_count", 0)),
            "age_bias_raw": float(eval_stats.get("age_bias_raw", 0.0)),
            "age_calibration": age_calibration,
        }
        history.append(row)

        # Prioritize target: gender>=0.90 and age_mae<3
        # score: higher better
        score = float(epoch_stats.val_gender_acc) - float(epoch_stats.val_age_mae) / 10.0
        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch
            best_age_calibration = age_calibration
            model.set_age_calibration(age_calibration)
            save_obj = {
                "model": model.state_dict(),
                "epoch": int(epoch),
                "best_score": float(best_score),
                "backbone_name": args.backbone_name,
                "img_size": int(args.img_size),
                "max_age_bin": int(args.max_age_bin),
                "age_expected_mix": float(args.age_expected_mix),
                "age_calibration": age_calibration,
                "metrics": row,
                "args": vars(args),
            }
            torch.save(save_obj, best_path)

        print(
            f"[epoch {epoch:02d}] "
            f"val_gender_acc={epoch_stats.val_gender_acc:.4f} "
            f"val_age_mae(raw->cal)=({epoch_stats.val_age_mae_raw:.4f}->{epoch_stats.val_age_mae:.4f}) "
            f"val_age_rmse={epoch_stats.val_age_rmse:.4f} "
            f"val_age_bias={epoch_stats.val_age_bias:.4f} "
            f"within(1/2/3)=({epoch_stats.val_age_within_1:.3f}/"
            f"{epoch_stats.val_age_within_2:.3f}/{epoch_stats.val_age_within_3:.3f}) "
            f"age_count={int(eval_stats.get('age_count', 0))} "
            f"{'[BEST]' if is_best else ''}"
        )

        # Save rolling checkpoint each epoch
        model.set_age_calibration(age_calibration)
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": int(epoch),
                "backbone_name": args.backbone_name,
                "img_size": int(args.img_size),
                "max_age_bin": int(args.max_age_bin),
                "age_expected_mix": float(args.age_expected_mix),
                "age_calibration": age_calibration,
                "metrics": row,
                "args": vars(args),
            },
            os.path.join(args.output_dir, f"age_gender_specialist_epoch_{epoch:03d}.pth"),
        )

        with open(os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        plot_curves(history, os.path.join(args.output_dir, "training_curves.png"))

    elapsed = datetime.datetime.now() - start_all
    print(f"[INFO] training done in {elapsed}, best_epoch={best_epoch}, best_score={best_score:.4f}")
    print(f"[INFO] best checkpoint: {best_path}")
    if best_age_calibration is not None:
        calib_path = os.path.join(args.output_dir, "age_calibration_best.json")
        with open(calib_path, "w", encoding="utf-8") as f:
            json.dump(best_age_calibration, f, ensure_ascii=False, indent=2)
        print(f"[INFO] best age calibration saved: {calib_path}")

    # Save embedding visualizations from best model
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        model.to(device)
        if channels_last and device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        save_embedding_visualizations(
            model=model,
            loader=val_loader,
            device=device,
            output_dir=args.output_dir,
            max_points=int(args.embedding_max_points),
        )
        print("[INFO] embedding visualizations saved")


if __name__ == "__main__":
    main()
