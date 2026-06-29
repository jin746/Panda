#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeReIDNetwork(nn.Module):
    """Prototype-based open-world ReID head with online update support."""

    def __init__(
        self,
        feature_dim: int,
        temperature: float = 0.1,
        momentum: float = 0.9,
        min_samples: int = 3,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.min_samples = int(min_samples)

        self.quality_net = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Output is logit; sigmoid is applied at use site.
        self.confidence_net = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, 1),
        )

        self.prototypes: Dict[str, torch.Tensor] = {}
        self.prototype_counts: Dict[str, int] = {}
        self.prototype_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.prototype_meta: Dict[str, Dict] = {}

        self.global_mean: Optional[torch.Tensor] = None
        self.global_std: Optional[torch.Tensor] = None

        # Tunables for fused similarity and optional aux reweighting.
        self.sim_cosine_w: float = 0.7
        self.sim_euclid_w: float = 0.3
        self.aux_gender_penalty: float = 0.15
        self.aux_age_reweight: float = 0.15
        self.aux_min_age_sigma2: float = 4.0

        # Open-world gate hyper-parameters (runtime configurable)
        self.base_threshold_no_stats: float = 0.50
        self.base_threshold_with_stats: float = 0.45
        self.quality_adjust_scale: float = 0.08
        self.deviation_adjust_scale: float = 0.05
        self.deviation_adjust_cap: float = 0.12
        self.confidence_threshold: float = 0.55
        self.quality_threshold: float = 0.35
        self.ambiguous_margin: float = 0.03
        self.ambiguous_offset: float = 0.02

    def to(self, device):
        super().to(device)
        for key in list(self.prototypes.keys()):
            self.prototypes[key] = self.prototypes[key].to(device)
        for key in list(self.prototype_stats.keys()):
            stat = self.prototype_stats[key]
            if "mean" in stat:
                stat["mean"] = stat["mean"].to(device)
            if "std" in stat:
                stat["std"] = stat["std"].to(device)
        if self.global_mean is not None:
            self.global_mean = self.global_mean.to(device)
        if self.global_std is not None:
            self.global_std = self.global_std.to(device)
        return self

    def update_global_stats(self, features: torch.Tensor):
        """Update running global feature stats for adaptive thresholding."""
        with torch.no_grad():
            if features.dim() == 1:
                features = features.unsqueeze(0)
            device = features.device

            batch_mean = features.mean(dim=0)
            if features.size(0) >= 2:
                batch_std = features.std(dim=0, unbiased=False)
            else:
                if self.global_mean is None:
                    batch_std = torch.full_like(batch_mean, 0.1)
                else:
                    batch_std = torch.clamp(torch.abs(features[0] - self.global_mean.to(device)), min=1e-4)

            if self.global_mean is None:
                self.global_mean = batch_mean.to(device)
                self.global_std = torch.clamp(batch_std, min=1e-4).to(device)
            else:
                self.global_mean = self.global_mean.to(device)
                self.global_std = self.global_std.to(device)
                self.global_mean = self.momentum * self.global_mean + (1.0 - self.momentum) * batch_mean
                self.global_std = self.momentum * self.global_std + (1.0 - self.momentum) * torch.clamp(
                    batch_std, min=1e-4
                )

    def compute_adaptive_threshold(self, query_feature: torch.Tensor) -> float:
        with torch.no_grad():
            quality_score = float(self.quality_net(query_feature.unsqueeze(0)).item())
            if self.global_mean is None or self.global_std is None:
                base = float(self.base_threshold_no_stats)
                return float(max(0.25, min(0.75, base - (quality_score - 0.5) * float(self.quality_adjust_scale))))

            device = query_feature.device
            global_mean = self.global_mean.to(device)
            global_std = self.global_std.to(device)
            deviation = float(torch.norm(query_feature - global_mean).item())
            normalized_deviation = deviation / (float(torch.norm(global_std).item()) + 1e-8)

            base_threshold = float(self.base_threshold_with_stats)
            quality_adjustment = (quality_score - 0.5) * float(self.quality_adjust_scale)
            deviation_adjustment = min(normalized_deviation * float(self.deviation_adjust_scale), float(self.deviation_adjust_cap))
            adaptive_threshold = base_threshold - quality_adjustment + deviation_adjustment
            return float(max(0.25, min(0.75, adaptive_threshold)))

    def update_prototype(
        self,
        id_name: str,
        new_feature: torch.Tensor,
        quality_score: float,
        gender_prob: Optional[float] = None,
        age_pred: Optional[float] = None,
    ):
        with torch.no_grad():
            device = new_feature.device
            feat = F.normalize(new_feature.float(), p=2, dim=0)
            is_new = id_name not in self.prototypes

            # Gate low-quality updates for existing prototypes only.
            if (not is_new) and (quality_score is not None) and float(quality_score) < 0.15:
                return

            meta = self.prototype_meta.get(
                id_name,
                {
                    "gender_prob": None,
                    "age_mean": None,
                    "age_m2": 0.0,
                    "age_count": 0,
                    "stable": False,
                },
            )

            if is_new:
                self.prototypes[id_name] = feat.clone().to(device)
                self.prototype_counts[id_name] = 1
                self.prototype_stats[id_name] = {
                    "mean": feat.clone().to(device),
                    "std": torch.full_like(feat, 1e-4).to(device),
                }
            else:
                old_proto = self.prototypes[id_name].to(device)
                count = int(self.prototype_counts.get(id_name, 1))

                q = float(max(0.0, min(1.0, quality_score)))
                effective_momentum = self.momentum * (1.0 - q) + q * 0.3
                if count < 5:
                    effective_momentum = max(effective_momentum, 0.90)

                updated = effective_momentum * old_proto + (1.0 - effective_momentum) * feat
                self.prototypes[id_name] = F.normalize(updated, p=2, dim=0)
                self.prototype_counts[id_name] = count + 1

                old_mean = self.prototype_stats[id_name]["mean"].to(device)
                new_mean = (old_mean * count + feat) / float(count + 1)
                self.prototype_stats[id_name]["mean"] = new_mean

                if count > 1:
                    old_std = self.prototype_stats[id_name]["std"].to(device)
                    diff_old = (old_mean - new_mean) ** 2
                    diff_new = (feat - new_mean) ** 2
                    new_var = (old_std ** 2 * (count - 1) + diff_old * count + diff_new) / max(count, 1)
                    self.prototype_stats[id_name]["std"] = torch.sqrt(torch.clamp(new_var, min=1e-8))
                else:
                    self.prototype_stats[id_name]["std"] = torch.full_like(feat, 1e-4)

            if gender_prob is not None:
                gp = float(max(0.0, min(1.0, gender_prob)))
                if meta["gender_prob"] is None:
                    meta["gender_prob"] = gp
                else:
                    alpha = 0.2 if self.prototype_counts[id_name] < 20 else 0.1
                    meta["gender_prob"] = (1.0 - alpha) * float(meta["gender_prob"]) + alpha * gp

            if age_pred is not None:
                try:
                    x = float(age_pred)
                    if meta["age_mean"] is None:
                        meta["age_mean"] = x
                        meta["age_m2"] = 0.0
                        meta["age_count"] = 1
                    else:
                        meta["age_count"] += 1
                        delta = x - float(meta["age_mean"])
                        meta["age_mean"] = float(meta["age_mean"]) + delta / float(meta["age_count"])
                        delta2 = x - float(meta["age_mean"])
                        meta["age_m2"] = float(meta["age_m2"]) + delta * delta2
                except Exception:
                    pass

            if int(self.prototype_counts.get(id_name, 0)) >= max(3, self.min_samples):
                meta["stable"] = True
            self.prototype_meta[id_name] = meta

    def compute_similarity_with_confidence(
        self,
        query_feature: torch.Tensor,
        prototype: torch.Tensor,
        id_name: Optional[str] = None,
        aux: Optional[Dict] = None,
    ) -> Tuple[float, float]:
        with torch.no_grad():
            device = query_feature.device
            q = F.normalize(query_feature.float(), p=2, dim=0)
            p = F.normalize(prototype.to(device).float(), p=2, dim=0)

            cosine_sim = float(F.cosine_similarity(q.unsqueeze(0), p.unsqueeze(0)).item())
            euclidean_dist = float(torch.norm(q - p).item())
            euclidean_sim = 1.0 / (1.0 + euclidean_dist)

            denom = max(1e-6, float(self.sim_cosine_w + self.sim_euclid_w))
            w_cos = float(self.sim_cosine_w) / denom
            w_euc = float(self.sim_euclid_w) / denom
            similarity = w_cos * cosine_sim + w_euc * euclidean_sim

            if aux is not None and id_name is not None and id_name in self.prototype_meta:
                meta = self.prototype_meta[id_name]
                gp_q = aux.get("gender_prob", None)
                gp_p = meta.get("gender_prob", None)
                if gp_q is not None and gp_p is not None:
                    mismatch = abs(float(gp_q) - float(gp_p))
                    penalty = float(self.aux_gender_penalty)
                    gender_factor = 1.0 - penalty * mismatch
                    gender_factor = max(1.0 - penalty, min(1.0 + penalty, gender_factor))
                    similarity *= gender_factor

                ap_q = aux.get("age_pred", None)
                mu = meta.get("age_mean", None)
                if ap_q is not None and mu is not None:
                    var = 0.0
                    if meta.get("age_count", 0) > 1:
                        var = float(meta.get("age_m2", 0.0)) / max(1.0, float(meta.get("age_count", 1) - 1))
                    sigma2 = max(float(self.aux_min_age_sigma2), var + 1e-6)
                    age_weight = math.exp(-0.5 * (float(ap_q) - float(mu)) ** 2 / sigma2)
                    reweight = (1.0 - float(self.aux_age_reweight)) + float(self.aux_age_reweight) * age_weight
                    similarity *= reweight

            conf_input = torch.cat([q, p], dim=0).unsqueeze(0)
            confidence_logit = self.confidence_net(conf_input).squeeze(1)
            confidence = float(torch.sigmoid(confidence_logit).item())
            return float(similarity), confidence

    def forward(
        self,
        query_feature: torch.Tensor,
        known_ids: Optional[List[str]] = None,
        aux: Optional[Dict] = None,
    ) -> Dict:
        if query_feature.dim() != 1:
            query_feature = query_feature.flatten()
        query_feature = F.normalize(query_feature.float(), p=2, dim=0)

        self.update_global_stats(query_feature.unsqueeze(0))
        adaptive_threshold = self.compute_adaptive_threshold(query_feature)
        quality_score = float(self.quality_net(query_feature.unsqueeze(0)).item())

        if len(self.prototypes) == 0:
            return {
                "predicted_id": "new_id_001",
                "similarity": 1.0,
                "confidence": quality_score,
                "is_new_id": True,
                "adaptive_threshold": adaptive_threshold,
                "all_similarities": {},
                "quality_score": quality_score,
                "debug_info": {
                    "best_similarity": 1.0,
                    "adaptive_threshold": adaptive_threshold,
                    "best_confidence": quality_score,
                    "confidence_threshold": 0.55,
                    "quality_score": quality_score,
                    "quality_threshold": 0.35,
                    "num_prototypes": 0,
                },
            }

        search_ids = known_ids if known_ids else list(self.prototypes.keys())
        similarities: Dict[str, float] = {}
        confidences: Dict[str, float] = {}

        for id_name in search_ids:
            if id_name in self.prototypes:
                sim, conf = self.compute_similarity_with_confidence(
                    query_feature=query_feature,
                    prototype=self.prototypes[id_name],
                    id_name=id_name,
                    aux=aux,
                )
                similarities[id_name] = sim
                confidences[id_name] = conf

        if not similarities:
            return {
                "predicted_id": "new_id_001",
                "similarity": 1.0,
                "confidence": quality_score,
                "is_new_id": True,
                "adaptive_threshold": adaptive_threshold,
                "all_similarities": {},
                "quality_score": quality_score,
                "debug_info": {
                    "best_similarity": 1.0,
                    "adaptive_threshold": adaptive_threshold,
                    "best_confidence": quality_score,
                    "confidence_threshold": 0.55,
                    "quality_score": quality_score,
                    "quality_threshold": 0.35,
                    "num_prototypes": len(self.prototypes),
                },
            }

        best_id = max(similarities.keys(), key=lambda x: similarities[x])
        best_similarity = float(similarities[best_id])
        best_confidence = float(confidences[best_id])

        sorted_sims = sorted(similarities.values(), reverse=True)
        second_best = float(sorted_sims[1]) if len(sorted_sims) > 1 else -1.0
        sim_margin = best_similarity - second_best if second_best >= -0.5 else 1.0

        confidence_threshold = float(self.confidence_threshold)
        quality_threshold = float(self.quality_threshold)
        similarity_gate = best_similarity < adaptive_threshold
        confidence_gate = best_confidence < confidence_threshold
        quality_gate = quality_score < quality_threshold
        ambiguous_gate = (
            sim_margin < float(self.ambiguous_margin)
            and best_similarity < adaptive_threshold + float(self.ambiguous_offset)
        )
        is_new_id = bool(similarity_gate and (confidence_gate or quality_gate or ambiguous_gate))

        if is_new_id:
            new_id_num = len([k for k in self.prototypes.keys() if k.startswith("new_id_")]) + 1
            predicted_id = f"new_id_{new_id_num:03d}"
        else:
            predicted_id = best_id

        return {
            "predicted_id": predicted_id,
            "similarity": best_similarity,
            "confidence": best_confidence,
            "is_new_id": is_new_id,
            "adaptive_threshold": adaptive_threshold,
            "all_similarities": similarities,
            "quality_score": quality_score,
            "debug_info": {
                "best_similarity": best_similarity,
                "adaptive_threshold": adaptive_threshold,
                "best_confidence": best_confidence,
                "confidence_threshold": confidence_threshold,
                "quality_score": quality_score,
                "quality_threshold": quality_threshold,
                "sim_margin": sim_margin,
                "num_prototypes": len(self.prototypes),
            },
        }


class PrototypeLoss(nn.Module):
    """Prototype contrastive classification loss."""

    def __init__(self, temperature: float = 0.1, margin: float = 0.5):
        super().__init__()
        self.temperature = float(temperature)
        self.margin = float(margin)

    def forward(self, features: torch.Tensor, prototypes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if prototypes.dim() != 2:
            raise ValueError(f"Prototypes should be 2D [N, D], got {tuple(prototypes.shape)}")
        if features.dim() != 2:
            raise ValueError(f"Features should be 2D [B, D], got {tuple(features.shape)}")
        if prototypes.size(0) == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        sims = torch.mm(features, prototypes.t()) / max(self.temperature, 1e-6)
        valid_labels = torch.clamp(labels, 0, prototypes.size(0) - 1)
        return F.cross_entropy(sims, valid_labels)
