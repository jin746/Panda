import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _normalize_torch_feature(feature: torch.Tensor) -> torch.Tensor:
    if feature.dim() != 1:
        feature = feature.flatten()
    return F.normalize(feature.float(), p=2, dim=0)


def _normalize_np_features(features: np.ndarray) -> np.ndarray:
    if features is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return arr / norms


def _fused_similarity_np(
    feat_a: np.ndarray,
    feat_b: np.ndarray,
    *,
    sim_cosine_w: float = 0.7,
    sim_euclid_w: float = 0.3,
    spherical: bool = False,
) -> float:
    a = np.asarray(feat_a, dtype=np.float32).reshape(-1)
    b = np.asarray(feat_b, dtype=np.float32).reshape(-1)
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an <= 1e-12 or bn <= 1e-12:
        return -1.0
    a = a / an
    b = b / bn
    cosine_sim = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if spherical:
        return cosine_sim
    euclidean_dist = float(np.linalg.norm(a - b))
    euclidean_sim = 1.0 / (1.0 + euclidean_dist)
    denom = max(1e-6, float(sim_cosine_w + sim_euclid_w))
    w_cos = float(sim_cosine_w) / denom
    w_euc = float(sim_euclid_w) / denom
    return float(w_cos * cosine_sim + w_euc * euclidean_sim)


class IdentityPrototypeBank:
    def __init__(
        self,
        *,
        strategy: str = "single",
        momentum: float = 0.9,
        max_slots: int = 4,
        spawn_similarity: float = 0.68,
        update_similarity: float = 0.58,
        topk: int = 2,
        aggregate_weight: float = 0.75,
        sim_cosine_w: float = 0.7,
        sim_euclid_w: float = 0.3,
        aux_gender_penalty: float = 0.15,
        aux_age_reweight: float = 0.10,
        aux_min_age_sigma: float = 2.0,
    ):
        self.strategy = str(strategy or "single").lower()
        self.momentum = float(momentum)
        self.max_slots = max(1, int(max_slots))
        self.spawn_similarity = float(spawn_similarity)
        self.update_similarity = float(update_similarity)
        self.topk = max(1, int(topk))
        self.aggregate_weight = float(max(0.0, min(1.0, aggregate_weight)))
        self.sim_cosine_w = float(sim_cosine_w)
        self.sim_euclid_w = float(sim_euclid_w)
        self.aux_gender_penalty = float(aux_gender_penalty)
        self.aux_age_reweight = float(aux_age_reweight)
        self.aux_min_age_sigma2 = float(max(0.01, aux_min_age_sigma) ** 2)
        self.identities: Dict[str, Dict] = {}

    @property
    def is_multi(self) -> bool:
        return self.strategy in {"multi", "spherical_multi"}

    @property
    def is_spherical(self) -> bool:
        return self.strategy in {"spherical", "spherical_multi"}

    def clear(self) -> None:
        self.identities.clear()

    def ids(self) -> List[str]:
        return list(self.identities.keys())

    def has_id(self, gid: str) -> bool:
        return str(gid) in self.identities

    def count(self, gid: str) -> int:
        info = self.identities.get(str(gid))
        if not info:
            return 0
        return int(sum(int(slot.get("count", 0) or 0) for slot in info.get("slots", [])))

    def get_feature(self, gid: str) -> Optional[torch.Tensor]:
        info = self.identities.get(str(gid))
        if not info:
            return None
        slots = info.get("slots", [])
        if not slots:
            return None
        feats = []
        weights = []
        for slot in slots:
            feat = slot.get("feature")
            if feat is None:
                continue
            feats.append(_normalize_torch_feature(feat))
            weights.append(max(1, int(slot.get("count", 1) or 1)))
        if not feats:
            return None
        if len(feats) == 1:
            return feats[0].clone()
        stacked = torch.stack(feats, dim=0)
        w = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device).view(-1, 1)
        merged = (stacked * w).sum(dim=0) / torch.clamp(w.sum(), min=1.0)
        return _normalize_torch_feature(merged)

    def get_all_slot_features(self, gid: str) -> List[torch.Tensor]:
        info = self.identities.get(str(gid))
        if not info:
            return []
        out = []
        for slot in info.get("slots", []):
            feat = slot.get("feature")
            if feat is not None:
                out.append(_normalize_torch_feature(feat))
        return out

    def _slot_similarity(
        self,
        query_feature: torch.Tensor,
        slot: Dict,
        *,
        aux: Optional[Dict] = None,
    ) -> float:
        q = _normalize_torch_feature(query_feature)
        p = _normalize_torch_feature(slot["feature"].to(q.device))
        cosine_sim = float(F.cosine_similarity(q.unsqueeze(0), p.unsqueeze(0)).item())
        if self.is_spherical:
            similarity = cosine_sim
        else:
            euclidean_dist = float(torch.norm(q - p).item())
            euclidean_sim = 1.0 / (1.0 + euclidean_dist)
            denom = max(1e-6, float(self.sim_cosine_w + self.sim_euclid_w))
            w_cos = float(self.sim_cosine_w) / denom
            w_euc = float(self.sim_euclid_w) / denom
            similarity = w_cos * cosine_sim + w_euc * euclidean_sim

        if aux is not None:
            gp_q = aux.get("gender_prob", None)
            gp_p = slot.get("gender_prob", None)
            if gp_q is not None and gp_p is not None:
                mismatch = abs(float(gp_q) - float(gp_p))
                penalty = float(self.aux_gender_penalty)
                gender_factor = 1.0 - penalty * mismatch
                gender_factor = max(1.0 - penalty, min(1.0 + penalty, gender_factor))
                similarity *= gender_factor

            ap_q = aux.get("age_pred", None)
            mu = slot.get("age_mean", None)
            if ap_q is not None and mu is not None:
                sigma2 = float(max(self.aux_min_age_sigma2, slot.get("age_var", 0.0) + 1e-6))
                age_weight = math.exp(-0.5 * (float(ap_q) - float(mu)) ** 2 / sigma2)
                reweight = (1.0 - float(self.aux_age_reweight)) + float(self.aux_age_reweight) * age_weight
                similarity *= reweight
        return float(similarity)

    def score_identity(
        self,
        gid: str,
        query_feature: torch.Tensor,
        *,
        aux: Optional[Dict] = None,
    ) -> Optional[Dict]:
        info = self.identities.get(str(gid))
        if not info:
            return None
        slots = info.get("slots", [])
        if not slots:
            return None

        slot_scores = []
        for idx, slot in enumerate(slots):
            score = self._slot_similarity(query_feature, slot, aux=aux)
            slot_scores.append((score, idx, slot))
        slot_scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_idx, best_slot = slot_scores[0]
        second_best = float(slot_scores[1][0]) if len(slot_scores) > 1 else -1.0
        if len(slot_scores) == 1 or not self.is_multi:
            aggregate = float(best_score)
        else:
            top = slot_scores[: max(1, min(self.topk, len(slot_scores)))]
            top_mean = float(sum(item[0] for item in top) / max(1, len(top)))
            aggregate = float(self.aggregate_weight * best_score + (1.0 - self.aggregate_weight) * top_mean)
        return {
            "gid": str(gid),
            "score": float(aggregate),
            "best_similarity": float(best_score),
            "second_similarity": float(second_best),
            "best_slot_index": int(best_idx),
            "best_feature": _normalize_torch_feature(best_slot["feature"]),
            "all_slot_scores": [float(item[0]) for item in slot_scores],
        }

    def match(
        self,
        query_feature: torch.Tensor,
        *,
        candidate_ids: Optional[Sequence[str]] = None,
        reserved_ids: Optional[Iterable[str]] = None,
        aux: Optional[Dict] = None,
    ) -> Optional[Dict]:
        reserved = set()
        reserved_checker = reserved_ids
        if reserved_ids is not None:
            try:
                reserved = {str(x) for x in reserved_ids if x is not None}
            except TypeError:
                reserved = set()
        candidates = [str(x) for x in candidate_ids] if candidate_ids is not None else self.ids()
        scored = []
        for gid in candidates:
            if gid in reserved:
                continue
            if not reserved and reserved_checker is not None:
                try:
                    if gid in reserved_checker:
                        continue
                except Exception:
                    pass
            result = self.score_identity(gid, query_feature, aux=aux)
            if result is not None:
                scored.append(result)
        if not scored:
            return None
        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0]
        second = float(scored[1]["score"]) if len(scored) > 1 else -1.0
        best["second_identity_score"] = float(second)
        best["all_similarities"] = {item["gid"]: float(item["score"]) for item in scored}
        return best

    def update(
        self,
        gid: str,
        feature: torch.Tensor,
        *,
        quality_score: Optional[float] = None,
        gender_prob: Optional[float] = None,
        age_pred: Optional[float] = None,
        id_conf: Optional[float] = None,
    ) -> None:
        gid = str(gid)
        feat = _normalize_torch_feature(feature)
        quality_val = 1.0 if quality_score is None else float(max(0.0, min(1.0, quality_score)))
        id_conf_val = 1.0 if id_conf is None else float(max(0.0, min(1.0, id_conf)))

        if gid not in self.identities:
            self.identities[gid] = {"slots": [self._build_slot(feat, gender_prob, age_pred)]}
            return

        slots = self.identities[gid].get("slots", [])
        if not slots:
            self.identities[gid]["slots"] = [self._build_slot(feat, gender_prob, age_pred)]
            return

        scored = []
        for idx, slot in enumerate(slots):
            sim = self._slot_similarity(feat, slot, aux=None)
            scored.append((sim, idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_sim, best_idx = scored[0]

        should_spawn = (
            self.is_multi
            and len(slots) < self.max_slots
            and float(best_sim) < float(self.spawn_similarity)
            and quality_val >= 0.20
            and id_conf_val >= 0.65
        )
        if should_spawn:
            slots.append(self._build_slot(feat, gender_prob, age_pred))
            return

        slot = slots[best_idx]
        momentum = float(self.momentum)
        momentum = momentum * (1.0 - quality_val) + quality_val * 0.30
        if int(slot.get("count", 1) or 1) < 5:
            momentum = max(momentum, 0.90)
        if float(best_sim) < float(self.update_similarity):
            momentum = min(momentum, 0.75)
        updated = momentum * _normalize_torch_feature(slot["feature"].to(feat.device)) + (1.0 - momentum) * feat
        slot["feature"] = _normalize_torch_feature(updated)
        slot["count"] = int(slot.get("count", 1) or 1) + 1
        self._update_slot_meta(slot, gender_prob=gender_prob, age_pred=age_pred)

    def merge(self, root_gid: str, child_gid: str) -> None:
        root_gid = str(root_gid)
        child_gid = str(child_gid)
        if root_gid == child_gid:
            return
        child = self.identities.get(child_gid)
        if child is None:
            return
        if root_gid not in self.identities:
            self.identities[root_gid] = child
            self.identities.pop(child_gid, None)
            return
        root_slots = self.identities[root_gid].setdefault("slots", [])
        root_slots.extend(child.get("slots", []))
        self.identities.pop(child_gid, None)
        self._compress_slots(root_gid)

    def _compress_slots(self, gid: str) -> None:
        info = self.identities.get(str(gid))
        if not info:
            return
        slots = info.get("slots", [])
        while len(slots) > self.max_slots:
            best_pair = None
            for i in range(len(slots)):
                for j in range(i + 1, len(slots)):
                    sim = self._slot_similarity(slots[i]["feature"], slots[j], aux=None)
                    cand = (sim, i, j)
                    if best_pair is None or cand[0] > best_pair[0]:
                        best_pair = cand
            if best_pair is None:
                slots.pop()
                continue
            _, i, j = best_pair
            a = slots[i]
            b = slots[j]
            count_a = max(1, int(a.get("count", 1) or 1))
            count_b = max(1, int(b.get("count", 1) or 1))
            merged = (count_a * _normalize_torch_feature(a["feature"]) + count_b * _normalize_torch_feature(b["feature"])) / float(count_a + count_b)
            a["feature"] = _normalize_torch_feature(merged)
            a["count"] = count_a + count_b
            self._merge_slot_meta(a, b)
            slots.pop(j)

    def _build_slot(
        self,
        feature: torch.Tensor,
        gender_prob: Optional[float],
        age_pred: Optional[float],
    ) -> Dict:
        slot = {
            "feature": _normalize_torch_feature(feature),
            "count": 1,
            "gender_prob": None,
            "age_mean": None,
            "age_m2": 0.0,
            "age_var": 0.0,
            "age_count": 0,
        }
        self._update_slot_meta(slot, gender_prob=gender_prob, age_pred=age_pred)
        return slot

    def _merge_slot_meta(self, dst: Dict, src: Dict) -> None:
        gp_a = dst.get("gender_prob", None)
        gp_b = src.get("gender_prob", None)
        count_a = max(1, int(dst.get("count", 1) or 1))
        count_b = max(1, int(src.get("count", 1) or 1))
        if gp_a is None:
            dst["gender_prob"] = gp_b
        elif gp_b is not None:
            dst["gender_prob"] = (count_a * float(gp_a) + count_b * float(gp_b)) / float(count_a + count_b)

        mu_a = dst.get("age_mean", None)
        mu_b = src.get("age_mean", None)
        n_a = int(dst.get("age_count", 0) or 0)
        n_b = int(src.get("age_count", 0) or 0)
        if mu_a is None:
            dst["age_mean"] = mu_b
            dst["age_m2"] = float(src.get("age_m2", 0.0) or 0.0)
            dst["age_var"] = float(src.get("age_var", 0.0) or 0.0)
            dst["age_count"] = n_b
            return
        if mu_b is None or n_b <= 0:
            return
        total = n_a + n_b
        delta = float(mu_b) - float(mu_a)
        new_mean = float(mu_a) + delta * float(n_b) / float(max(1, total))
        new_m2 = float(dst.get("age_m2", 0.0) or 0.0) + float(src.get("age_m2", 0.0) or 0.0) + delta * delta * float(n_a * n_b) / float(max(1, total))
        dst["age_mean"] = new_mean
        dst["age_m2"] = new_m2
        dst["age_count"] = total
        dst["age_var"] = new_m2 / float(max(1, total - 1)) if total > 1 else 0.0

    def _update_slot_meta(
        self,
        slot: Dict,
        *,
        gender_prob: Optional[float],
        age_pred: Optional[float],
    ) -> None:
        if gender_prob is not None:
            gp = float(max(0.0, min(1.0, gender_prob)))
            if slot.get("gender_prob", None) is None:
                slot["gender_prob"] = gp
            else:
                alpha = 0.2 if int(slot.get("count", 1) or 1) < 20 else 0.1
                slot["gender_prob"] = (1.0 - alpha) * float(slot["gender_prob"]) + alpha * gp
        if age_pred is not None:
            x = float(age_pred)
            if slot.get("age_mean", None) is None:
                slot["age_mean"] = x
                slot["age_m2"] = 0.0
                slot["age_var"] = 0.0
                slot["age_count"] = 1
            else:
                count = int(slot.get("age_count", 0) or 0) + 1
                mean = float(slot["age_mean"])
                delta = x - mean
                mean = mean + delta / float(count)
                delta2 = x - mean
                m2 = float(slot.get("age_m2", 0.0) or 0.0) + delta * delta2
                slot["age_mean"] = mean
                slot["age_m2"] = m2
                slot["age_count"] = count
                slot["age_var"] = m2 / float(max(1, count - 1)) if count > 1 else 0.0


def cluster_embeddings_multiproto(
    features: np.ndarray,
    *,
    threshold: float,
    momentum: float = 0.9,
    max_slots: int = 4,
    spawn_similarity: float = 0.68,
    topk: int = 2,
    aggregate_weight: float = 0.75,
    sim_cosine_w: float = 0.7,
    sim_euclid_w: float = 0.3,
    spherical: bool = False,
    seed: int = 42,
) -> np.ndarray:
    feats = _normalize_np_features(features)
    n = int(feats.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.int32)
    order = np.arange(n, dtype=np.int32)
    if n > 1:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(order)

    clusters: List[Dict] = []
    labels = np.full((n,), -1, dtype=np.int32)
    topk = max(1, int(topk))
    aggregate_weight = float(max(0.0, min(1.0, aggregate_weight)))
    max_slots = max(1, int(max_slots))

    for idx in order.tolist():
        feat = feats[idx]
        best_cluster = None
        best_score = -10.0
        best_slot_idx = -1
        for cid, cluster in enumerate(clusters):
            slot_scores = []
            for slot_idx, slot in enumerate(cluster["slots"]):
                sim = _fused_similarity_np(
                    feat,
                    slot["feature"],
                    sim_cosine_w=sim_cosine_w,
                    sim_euclid_w=sim_euclid_w,
                    spherical=spherical,
                )
                slot_scores.append((sim, slot_idx))
            if not slot_scores:
                continue
            slot_scores.sort(key=lambda x: x[0], reverse=True)
            best_slot_score = float(slot_scores[0][0])
            if len(slot_scores) == 1:
                cluster_score = best_slot_score
            else:
                top = slot_scores[: max(1, min(topk, len(slot_scores)))]
                top_mean = float(sum(x[0] for x in top) / max(1, len(top)))
                cluster_score = float(aggregate_weight * best_slot_score + (1.0 - aggregate_weight) * top_mean)
            if cluster_score > best_score:
                best_cluster = cid
                best_score = float(cluster_score)
                best_slot_idx = int(slot_scores[0][1])

        if best_cluster is None or best_score < float(threshold):
            clusters.append({"slots": [{"feature": feat.copy(), "count": 1}]})
            labels[idx] = int(len(clusters) - 1)
            continue

        labels[idx] = int(best_cluster)
        slots = clusters[best_cluster]["slots"]
        best_slot = slots[best_slot_idx]
        raw_best = _fused_similarity_np(
            feat,
            best_slot["feature"],
            sim_cosine_w=sim_cosine_w,
            sim_euclid_w=sim_euclid_w,
            spherical=spherical,
        )
        should_spawn = len(slots) < max_slots and float(raw_best) < float(spawn_similarity)
        if should_spawn:
            slots.append({"feature": feat.copy(), "count": 1})
            continue

        count = max(1, int(best_slot.get("count", 1) or 1))
        eff_momentum = float(momentum)
        if count < 5:
            eff_momentum = max(eff_momentum, 0.90)
        updated = eff_momentum * best_slot["feature"] + (1.0 - eff_momentum) * feat
        updated = _normalize_np_features(updated.reshape(1, -1))[0]
        best_slot["feature"] = updated.astype(np.float32, copy=False)
        best_slot["count"] = count + 1

    uniq = np.unique(labels)
    remap = {int(lb): idx for idx, lb in enumerate(uniq.tolist())}
    return np.asarray([remap[int(lb)] for lb in labels], dtype=np.int32)
