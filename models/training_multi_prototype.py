import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _zero_loss(device: torch.device) -> torch.Tensor:
    return torch.tensor(0.0, device=device)


def _safe_normalize(vec: torch.Tensor, dim: int) -> torch.Tensor:
    return F.normalize(vec.float(), p=2, dim=dim)


def _safe_mean(losses, device: torch.device) -> torch.Tensor:
    if not losses:
        return _zero_loss(device)
    return torch.stack(losses).mean()


def _weighted_centroid(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.float().view(-1, 1)
    centroid = (features * weights).sum(dim=0)
    return _safe_normalize(centroid, dim=0)


class TrainingMultiPrototypeMemory:
    def __init__(
        self,
        feature_dim: int,
        max_slots: int = 4,
        momentum: float = 0.9,
        spawn_threshold: float = 0.55,
        update_threshold: float = 0.45,
        spawn_min_quality: float = 0.0,
        class_centroid_guard: float = 0.0,
        update_min_quality: float = 0.0,
    ):
        self.feature_dim = int(feature_dim)
        self.max_slots = max(1, int(max_slots))
        self.momentum = float(momentum)
        self.spawn_threshold = float(spawn_threshold)
        self.update_threshold = float(update_threshold)
        self.spawn_min_quality = float(spawn_min_quality)
        self.class_centroid_guard = float(class_centroid_guard)
        self.update_min_quality = float(update_min_quality)
        self.device = torch.device("cpu")
        self.memory: Dict[int, List[Dict[str, torch.Tensor]]] = {}

    def to(self, device: torch.device):
        self.device = torch.device(device)
        for slots in self.memory.values():
            for slot in slots:
                slot["feature"] = slot["feature"].to(self.device)
        return self

    def clear(self):
        self.memory.clear()

    def num_classes(self) -> int:
        return int(len(self.memory))

    def num_slots(self) -> int:
        return int(sum(len(slots) for slots in self.memory.values()))

    def state_dict(self) -> Dict:
        out = {
            "feature_dim": int(self.feature_dim),
            "max_slots": int(self.max_slots),
            "momentum": float(self.momentum),
            "spawn_threshold": float(self.spawn_threshold),
            "update_threshold": float(self.update_threshold),
            "spawn_min_quality": float(self.spawn_min_quality),
            "class_centroid_guard": float(self.class_centroid_guard),
            "update_min_quality": float(self.update_min_quality),
            "memory": {},
        }
        for label, slots in self.memory.items():
            out["memory"][int(label)] = [
                {
                    "feature": slot["feature"].detach().cpu(),
                    "count": int(slot.get("count", 1) or 1),
                    "quality": float(slot.get("quality", 1.0) or 1.0),
                }
                for slot in slots
            ]
        return out

    def load_state_dict(self, state: Optional[Dict]):
        self.clear()
        if not isinstance(state, dict):
            return
        self.feature_dim = int(state.get("feature_dim", self.feature_dim))
        self.max_slots = max(1, int(state.get("max_slots", self.max_slots)))
        self.momentum = float(state.get("momentum", self.momentum))
        self.spawn_threshold = float(state.get("spawn_threshold", self.spawn_threshold))
        self.update_threshold = float(state.get("update_threshold", self.update_threshold))
        self.spawn_min_quality = float(state.get("spawn_min_quality", self.spawn_min_quality))
        self.class_centroid_guard = float(state.get("class_centroid_guard", self.class_centroid_guard))
        self.update_min_quality = float(state.get("update_min_quality", self.update_min_quality))
        raw_memory = state.get("memory", {}) or {}
        for label, slots in raw_memory.items():
            lid = int(label)
            self.memory[lid] = []
            for slot in slots or []:
                feat = slot.get("feature", None)
                if feat is None:
                    continue
                feat = _safe_normalize(feat.to(self.device).float().flatten(), dim=0)
                self.memory[lid].append(
                    {
                        "feature": feat,
                        "count": int(slot.get("count", 1) or 1),
                        "quality": float(slot.get("quality", 1.0) or 1.0),
                    }
                )

    def get_flat_snapshot(self, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        proto_features, proto_labels, proto_counts, _proto_qualities = self.get_flat_snapshot_ex(device)
        return proto_features, proto_labels, proto_counts

    def get_flat_snapshot_ex(
        self,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        proto_features = []
        proto_labels = []
        proto_counts = []
        proto_qualities = []
        for label, slots in self.memory.items():
            for slot in slots:
                feat = slot.get("feature", None)
                if feat is None:
                    continue
                proto_features.append(_safe_normalize(feat.to(device).float().flatten(), dim=0).detach())
                proto_labels.append(int(label))
                proto_counts.append(int(slot.get("count", 1) or 1))
                proto_qualities.append(float(slot.get("quality", 1.0) or 1.0))
        if not proto_features:
            return None, None, None, None
        return (
            torch.stack(proto_features, dim=0),
            torch.tensor(proto_labels, device=device, dtype=torch.long),
            torch.tensor(proto_counts, device=device, dtype=torch.float32),
            torch.tensor(proto_qualities, device=device, dtype=torch.float32),
        )

    def get_class_centroid(self, label: int, device: torch.device) -> Optional[torch.Tensor]:
        slots = self.memory.get(int(label), [])
        if not slots:
            return None
        feats = []
        weights = []
        for slot in slots:
            feat = slot.get("feature", None)
            if feat is None:
                continue
            feats.append(_safe_normalize(feat.to(device).float().flatten(), dim=0))
            count = float(slot.get("count", 1) or 1)
            quality = float(slot.get("quality", 1.0) or 1.0)
            weights.append(count * (0.5 + 0.5 * quality))
        if not feats:
            return None
        return _weighted_centroid(torch.stack(feats, dim=0), torch.tensor(weights, device=device))

    def update_batch(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None,
    ):
        if features.dim() != 2:
            raise ValueError(f"features must be [B, D], got {tuple(features.shape)}")
        labels = labels.view(-1).long()
        feats = _safe_normalize(features.detach().float(), dim=1)
        if quality_scores is not None:
            quality_scores = quality_scores.detach().float().view(-1)
        for idx in range(int(feats.size(0))):
            self.update_one(
                int(labels[idx].item()),
                feats[idx],
                None if quality_scores is None else float(quality_scores[idx].item()),
            )

    def update_one(self, label: int, feature: torch.Tensor, quality_score: Optional[float] = None):
        lid = int(label)
        feat = _safe_normalize(feature.detach().to(self.device).float().flatten(), dim=0)
        q = 1.0 if quality_score is None else float(max(0.0, min(1.0, quality_score)))
        slots = self.memory.setdefault(lid, [])
        if len(slots) == 0:
            slots.append({"feature": feat, "count": 1, "quality": q})
            return

        slot_features = [
            _safe_normalize(slot["feature"].to(self.device).float().flatten(), dim=0)
            for slot in slots
        ]
        sims = [float(torch.dot(feat, slot_feat).item()) for slot_feat in slot_features]
        best_idx = max(range(len(sims)), key=lambda i: sims[i])
        best_sim = float(sims[best_idx])

        class_centroid = self.get_class_centroid(lid, self.device)
        class_sim = float(torch.dot(feat, class_centroid).item()) if class_centroid is not None else -1.0

        can_spawn = len(slots) < self.max_slots and best_sim < self.spawn_threshold
        if can_spawn and q < self.spawn_min_quality:
            can_spawn = False
        if can_spawn and self.class_centroid_guard > 0.0 and class_sim >= self.class_centroid_guard:
            can_spawn = False
        if can_spawn:
            slots.append({"feature": feat, "count": 1, "quality": q})
            return

        if self.update_min_quality > 0.0 and q < self.update_min_quality and best_sim < self.update_threshold:
            return

        slot = slots[best_idx]
        count = int(slot.get("count", 1) or 1)
        effective_momentum = self.momentum * (1.0 - q) + 0.30 * q
        if count < 5:
            effective_momentum = max(effective_momentum, 0.90)
        if best_sim < self.update_threshold:
            effective_momentum = min(effective_momentum, 0.75)
        if self.update_min_quality > 0.0 and q < self.update_min_quality:
            effective_momentum = max(effective_momentum, 0.97)
        updated = effective_momentum * slot["feature"].to(self.device) + (1.0 - effective_momentum) * feat
        slot["feature"] = _safe_normalize(updated, dim=0)
        slot["count"] = count + 1
        old_q = float(slot.get("quality", 1.0) or 1.0)
        slot["quality"] = 0.9 * old_q + 0.1 * q


def _get_snapshot(memory_bank: TrainingMultiPrototypeMemory, device: torch.device):
    proto_features, proto_labels, proto_counts, proto_qualities = memory_bank.get_flat_snapshot_ex(device)
    if proto_features is None or proto_labels is None or proto_counts is None or proto_qualities is None:
        return None
    return proto_features, proto_labels, proto_counts, proto_qualities


def _get_positive_assignment(
    sims_row: torch.Tensor,
    proto_labels: torch.Tensor,
    label: torch.Tensor,
    temperature: float,
    proto_counts: torch.Tensor,
    proto_qualities: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    pos_mask = proto_labels == label
    if not bool(pos_mask.any()):
        return None, None, None
    sims_row = sims_row.float()
    pos_logits = sims_row[pos_mask] / max(float(temperature), 1e-6)
    pos_logits = pos_logits + 0.10 * torch.log1p(proto_counts[pos_mask])
    pos_logits = pos_logits + 0.10 * proto_qualities[pos_mask]
    pos_probs = torch.softmax(pos_logits, dim=0)
    assignment = torch.zeros_like(sims_row, dtype=pos_probs.dtype)
    assignment[pos_mask] = pos_probs
    return assignment, pos_mask, pos_probs


def compute_multi_prototype_metric_loss(
    memory_bank: TrainingMultiPrototypeMemory,
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    hard_neg_k: int = 32,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = features.device
    features = _safe_normalize(features.float(), dim=1)
    labels = labels.view(-1).long()
    snapshot = _get_snapshot(memory_bank, device)
    if snapshot is None:
        zero = _zero_loss(device)
        return zero, {"valid": 0.0, "avg_pos": 0.0, "avg_neg": 0.0}

    proto_features, proto_labels, _proto_counts, _proto_qualities = snapshot
    sims = torch.matmul(features, proto_features.t())
    losses = []
    pos_vals = []
    neg_vals = []
    hard_neg_k = max(1, int(hard_neg_k))
    temp = max(float(temperature), 1e-6)

    for i in range(int(features.size(0))):
        pos_mask = proto_labels == labels[i]
        neg_mask = ~pos_mask
        if not bool(pos_mask.any()) or not bool(neg_mask.any()):
            continue

        pos_sims = sims[i][pos_mask]
        neg_sims = sims[i][neg_mask]
        pos_vals.append(float(pos_sims.max().item()))

        neg_topk = min(hard_neg_k, int(neg_sims.numel()))
        neg_sims = torch.topk(neg_sims, k=neg_topk, largest=True).values
        neg_vals.append(float(neg_sims.max().item()))

        pos_logit = torch.logsumexp(pos_sims / temp, dim=0).view(1)
        logits = torch.cat([pos_logit, neg_sims / temp], dim=0).unsqueeze(0)
        target = torch.zeros((1,), dtype=torch.long, device=device)
        losses.append(F.cross_entropy(logits, target))

    if not losses:
        zero = _zero_loss(device)
        return zero, {"valid": 0.0, "avg_pos": 0.0, "avg_neg": 0.0}

    loss = torch.stack(losses).mean()
    stats = {
        "valid": float(len(losses)),
        "avg_pos": float(sum(pos_vals) / max(1, len(pos_vals))),
        "avg_neg": float(sum(neg_vals) / max(1, len(neg_vals))),
    }
    return loss, stats


def compute_continual_metric_consistency_loss(
    memory_bank: TrainingMultiPrototypeMemory,
    features: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.15,
    hard_neg_k: int = 16,
    stability_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = features.device
    features = _safe_normalize(features.float(), dim=1)
    labels = labels.view(-1).long()
    snapshot = _get_snapshot(memory_bank, device)
    if snapshot is None:
        zero = _zero_loss(device)
        return zero, {"valid": 0.0, "avg_pos": 0.0, "avg_neg": 0.0}

    proto_features, proto_labels, _proto_counts, _proto_qualities = snapshot
    sims = torch.matmul(features, proto_features.t())
    losses = []
    pos_vals = []
    neg_vals = []
    hard_neg_k = max(1, int(hard_neg_k))
    margin = float(margin)
    stability_weight = float(stability_weight)

    for i in range(int(features.size(0))):
        pos_mask = proto_labels == labels[i]
        neg_mask = ~pos_mask
        if not bool(pos_mask.any()):
            continue

        pos_sims = sims[i][pos_mask]
        best_pos = pos_sims.max()
        pos_vals.append(float(best_pos.item()))
        stability_loss = 1.0 - best_pos

        if bool(neg_mask.any()):
            neg_sims = sims[i][neg_mask]
            neg_topk = min(hard_neg_k, int(neg_sims.numel()))
            hard_neg = torch.topk(neg_sims, k=neg_topk, largest=True).values.max()
            neg_vals.append(float(hard_neg.item()))
            margin_loss = F.relu(torch.tensor(margin, device=device) - best_pos + hard_neg)
        else:
            margin_loss = _zero_loss(device)

        losses.append(stability_weight * stability_loss + margin_loss)

    if not losses:
        zero = _zero_loss(device)
        return zero, {"valid": 0.0, "avg_pos": 0.0, "avg_neg": 0.0}

    loss = torch.stack(losses).mean()
    stats = {
        "valid": float(len(losses)),
        "avg_pos": float(sum(pos_vals) / max(1, len(pos_vals))),
        "avg_neg": float(sum(neg_vals) / max(1, len(neg_vals))) if neg_vals else 0.0,
    }
    return loss, stats


def compute_dynamic_topology_loss(
    memory_bank: TrainingMultiPrototypeMemory,
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.10,
    negative_margin: float = 0.15,
    pull_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = features.device
    features = _safe_normalize(features.float(), dim=1)
    labels = labels.view(-1).long()
    snapshot = _get_snapshot(memory_bank, device)
    if snapshot is None:
        zero = _zero_loss(device)
        return zero, {"valid": 0.0, "pos_pairs": 0.0, "avg_pull": 0.0}

    proto_features, proto_labels, proto_counts, proto_qualities = snapshot
    sims = torch.matmul(features, proto_features.t())
    assignments = []
    valid_indices = []
    pull_losses = []

    for i in range(int(features.size(0))):
        assignment, pos_mask, pos_probs = _get_positive_assignment(
            sims[i], proto_labels, labels[i], temperature, proto_counts, proto_qualities
        )
        if assignment is None or pos_mask is None or pos_probs is None:
            continue
        centroid = _weighted_centroid(proto_features[pos_mask], pos_probs)
        pull_losses.append(1.0 - torch.dot(features[i], centroid))
        assignments.append(assignment)
        valid_indices.append(i)

    if len(assignments) < 2:
        loss = _safe_mean(pull_losses, device)
        return loss, {"valid": float(len(assignments)), "pos_pairs": 0.0, "avg_pull": float(loss.item())}

    assign_mat = torch.stack(assignments, dim=0)
    valid_features = features[valid_indices]
    valid_labels = labels[valid_indices]

    topo_target = torch.matmul(assign_mat, assign_mat.t()).clamp(0.0, 1.0)
    pair_sim = torch.matmul(valid_features, valid_features.t()).clamp(-1.0, 1.0)
    pair_sim = (pair_sim + 1.0) * 0.5

    same_mask = valid_labels.unsqueeze(1).eq(valid_labels.unsqueeze(0)).float()
    eye = torch.eye(len(valid_indices), device=device)
    same_mask = same_mask * (1.0 - eye)
    diff_mask = (1.0 - same_mask - eye).clamp(min=0.0)

    pos_loss = ((pair_sim - topo_target) ** 2 * same_mask).sum() / same_mask.sum().clamp(min=1.0)
    neg_loss = (F.relu(pair_sim - float(negative_margin)) ** 2 * diff_mask).sum() / diff_mask.sum().clamp(min=1.0)
    pull_loss = _safe_mean(pull_losses, device)
    loss = pos_loss + 0.5 * neg_loss + float(pull_weight) * pull_loss
    stats = {
        "valid": float(len(valid_indices)),
        "pos_pairs": float(same_mask.sum().item()),
        "avg_pull": float(pull_loss.item()),
    }
    return loss, stats


def compute_uncertainty_topology_purification_loss(
    memory_bank: TrainingMultiPrototypeMemory,
    features: torch.Tensor,
    labels: torch.Tensor,
    quality_scores: torch.Tensor,
    purify_blend: float = 0.35,
    margin: float = 0.10,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = features.device
    features = _safe_normalize(features.float(), dim=1)
    labels = labels.view(-1).long()
    quality_scores = quality_scores.view(-1).float().clamp(0.0, 1.0)
    snapshot = _get_snapshot(memory_bank, device)
    if snapshot is None:
        zero = _zero_loss(device)
        return zero, {"valid": 0.0, "avg_uncertainty": 0.0}

    proto_features, proto_labels, _proto_counts, _proto_qualities = snapshot
    sims = torch.matmul(features, proto_features.t())
    losses = []
    unc_vals = []

    for i in range(int(features.size(0))):
        pos_mask = proto_labels == labels[i]
        neg_mask = ~pos_mask
        if not bool(pos_mask.any()):
            continue

        class_centroid = memory_bank.get_class_centroid(int(labels[i].item()), device)
        if class_centroid is None:
            class_centroid = proto_features[pos_mask][sims[i][pos_mask].argmax()]

        pos_sim = torch.dot(features[i], class_centroid)
        if bool(neg_mask.any()):
            hard_neg = sims[i][neg_mask].max()
        else:
            hard_neg = torch.tensor(-1.0, device=device)

        unc = 1.0 - quality_scores[i]
        unc_vals.append(float(unc.item()))
        alpha = float(purify_blend) * float(unc.item())
        purified = _safe_normalize((1.0 - alpha) * features[i] + alpha * class_centroid, dim=0)
        consistency_loss = 1.0 - torch.dot(features[i], purified)
        align_loss = 1.0 - torch.dot(purified, class_centroid)
        margin_loss = F.relu(torch.tensor(float(margin), device=device) - pos_sim + hard_neg)
        losses.append(unc * (0.5 * consistency_loss + align_loss + 0.5 * margin_loss))

    if not losses:
        zero = _zero_loss(device)
        return zero, {"valid": 0.0, "avg_uncertainty": 0.0}

    loss = torch.stack(losses).mean()
    stats = {
        "valid": float(len(losses)),
        "avg_uncertainty": float(sum(unc_vals) / max(1, len(unc_vals))),
    }
    return loss, stats


def compute_meta_fewshot_topology_loss(
    memory_bank: TrainingMultiPrototypeMemory,
    features: torch.Tensor,
    labels: torch.Tensor,
    support_shots: int = 1,
    query_max: int = 2,
    adapt_blend: float = 0.35,
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = features.device
    features = _safe_normalize(features.float(), dim=1)
    labels = labels.view(-1).long()
    support_shots = max(1, int(support_shots))
    query_max = max(1, int(query_max))
    temp = max(float(temperature), 1e-6)

    unique_labels = torch.unique(labels)
    support_protos = []
    class_index = {}
    query_feats = []
    query_targets = []

    for label in unique_labels:
        idxs = torch.nonzero(labels == label, as_tuple=False).view(-1)
        if int(idxs.numel()) <= support_shots:
            continue
        support_idx = idxs[:support_shots]
        query_idx = idxs[support_shots:support_shots + query_max]
        if int(query_idx.numel()) == 0:
            continue

        support_proto = _safe_normalize(features[support_idx].mean(dim=0), dim=0)
        mem_centroid = memory_bank.get_class_centroid(int(label.item()), device)
        if mem_centroid is not None:
            support_proto = _safe_normalize(
                (1.0 - float(adapt_blend)) * support_proto + float(adapt_blend) * mem_centroid,
                dim=0,
            )

        class_index[int(label.item())] = len(support_protos)
        support_protos.append(support_proto)
        for qidx in query_idx:
            query_feats.append(features[int(qidx.item())])
            query_targets.append(class_index[int(label.item())])

    if len(support_protos) < 2 or len(query_feats) == 0:
        zero = _zero_loss(device)
        return zero, {"n_way": 0.0, "queries": 0.0}

    proto_mat = torch.stack(support_protos, dim=0)
    query_mat = torch.stack(query_feats, dim=0)
    logits = torch.matmul(query_mat, proto_mat.t()) / temp
    targets = torch.tensor(query_targets, device=device, dtype=torch.long)
    loss = F.cross_entropy(logits, targets)
    stats = {
        "n_way": float(len(support_protos)),
        "queries": float(len(query_feats)),
    }
    return loss, stats


def compute_incremental_topology_stability_loss(
    memory_bank: TrainingMultiPrototypeMemory,
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.10,
    slot_radius: float = 0.18,
    entropy_weight: float = 0.5,
    centroid_weight: float = 0.5,
    slot_weight: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = features.device
    features = _safe_normalize(features.float(), dim=1)
    labels = labels.view(-1).long()
    snapshot = _get_snapshot(memory_bank, device)
    if snapshot is None:
        zero = _zero_loss(device)
        return zero, {"valid": 0.0, "avg_entropy": 0.0, "slot_dispersion": 0.0}

    proto_features, proto_labels, proto_counts, proto_qualities = snapshot
    sims = torch.matmul(features, proto_features.t())
    temp = max(float(temperature), 1e-6)
    sample_losses = []
    ent_vals = []

    for i in range(int(features.size(0))):
        assignment, pos_mask, pos_probs = _get_positive_assignment(
            sims[i], proto_labels, labels[i], temp, proto_counts, proto_qualities
        )
        if assignment is None or pos_mask is None or pos_probs is None:
            continue

        centroid = _weighted_centroid(proto_features[pos_mask], pos_probs)
        pull_loss = 1.0 - torch.dot(features[i], centroid)
        if int(pos_probs.numel()) > 1:
            entropy = -(pos_probs * torch.log(pos_probs.clamp(min=1e-8))).sum() / math.log(float(pos_probs.numel()))
        else:
            entropy = torch.tensor(0.0, device=device)
        ent_vals.append(float(entropy.item()))
        sample_losses.append(float(entropy_weight) * entropy + float(centroid_weight) * pull_loss)

    slot_losses = []
    unique_proto_labels = torch.unique(proto_labels)
    for label in unique_proto_labels:
        mask = proto_labels == label
        if int(mask.sum().item()) <= 1:
            continue
        feats = proto_features[mask]
        counts = proto_counts[mask]
        centroid = _weighted_centroid(feats, counts)
        dissimilarity = 1.0 - torch.matmul(feats, centroid.unsqueeze(1)).squeeze(1)
        slot_losses.append(F.relu(dissimilarity - float(slot_radius)).mean())

    sample_loss = _safe_mean(sample_losses, device)
    slot_loss = _safe_mean(slot_losses, device)
    total_loss = sample_loss + float(slot_weight) * slot_loss
    stats = {
        "valid": float(len(sample_losses)),
        "avg_entropy": float(sum(ent_vals) / max(1, len(ent_vals))) if ent_vals else 0.0,
        "slot_dispersion": float(slot_loss.item()),
    }
    return total_loss, stats
