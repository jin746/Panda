#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open-world identification clustering metrics.

These metrics are computed from per-image predictions:
- true_id: ground-truth identity string
- predicted_id: predicted cluster/identity string

Main outputs:
- cluster_purity
- assignment_accuracy (Hungarian one-to-one matching)
- cluster_contamination (1 - purity)
- predicted/true identity count accuracy
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def _build_confusion_matrix(
    true_ids: List[str], pred_ids: List[str]
) -> Tuple[np.ndarray, List[str], List[str]]:
    uniq_true = sorted(set(true_ids))
    uniq_pred = sorted(set(pred_ids))

    true_to_idx = {k: i for i, k in enumerate(uniq_true)}
    pred_to_idx = {k: i for i, k in enumerate(uniq_pred)}

    mat = np.zeros((len(uniq_true), len(uniq_pred)), dtype=np.int64)
    for t, p in zip(true_ids, pred_ids):
        mat[true_to_idx[t], pred_to_idx[p]] += 1

    return mat, uniq_true, uniq_pred


def _greedy_assignment_accuracy(mat: np.ndarray) -> float:
    if mat.size == 0:
        return 0.0
    rows, cols = mat.shape
    used_r = set()
    used_c = set()
    total = int(mat.sum())
    matched = 0

    candidates = []
    for r in range(rows):
        for c in range(cols):
            cnt = int(mat[r, c])
            if cnt > 0:
                candidates.append((cnt, r, c))
    candidates.sort(reverse=True)

    for cnt, r, c in candidates:
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        matched += cnt

    return float(matched / max(total, 1))


def _assignment_accuracy(mat: np.ndarray) -> float:
    if mat.size == 0:
        return 0.0
    total = int(mat.sum())
    if total <= 0:
        return 0.0

    if SCIPY_AVAILABLE:
        try:
            r_idx, c_idx = linear_sum_assignment(-mat)
            matched = int(sum(int(mat[r, c]) for r, c in zip(r_idx, c_idx)))
            return float(matched / total)
        except Exception:
            pass

    return _greedy_assignment_accuracy(mat)


def compute_open_world_cluster_metrics(
    predictions: Iterable[Dict],
) -> Dict[str, float]:
    """
    Compute open-world clustering quality from prediction records.

    Required keys in each prediction:
    - true_id
    - predicted_id
    """
    true_ids = []
    pred_ids = []
    for r in predictions:
        t = r.get("true_id", None)
        p = r.get("predicted_id", None)
        if t is None or p is None:
            continue
        true_ids.append(str(t))
        pred_ids.append(str(p))

    n = len(true_ids)
    if n == 0:
        return {
            "num_samples": 0.0,
            "cluster_purity": 0.0,
            "assignment_accuracy": 0.0,
            "cluster_contamination": 1.0,
            "predicted_id_count": 0.0,
            "true_id_count": 0.0,
            "id_count_abs_error": 0.0,
            "id_count_rel_error": 0.0,
            "id_count_accuracy": 0.0,
        }

    mat, uniq_true, uniq_pred = _build_confusion_matrix(true_ids, pred_ids)

    # Cluster purity: for each predicted cluster, keep dominant true class.
    col_max = mat.max(axis=0) if mat.shape[1] > 0 else np.array([0])
    cluster_purity = float(np.sum(col_max) / max(int(mat.sum()), 1))

    assignment_acc = _assignment_accuracy(mat)
    contamination = float(1.0 - cluster_purity)

    pred_count = int(len(uniq_pred))
    true_count = int(len(uniq_true))
    id_count_abs_error = int(abs(pred_count - true_count))
    id_count_rel_error = float(id_count_abs_error / max(true_count, 1))
    id_count_accuracy = float(1.0 - min(1.0, id_count_rel_error))

    return {
        "num_samples": float(n),
        "cluster_purity": cluster_purity,
        "assignment_accuracy": assignment_acc,
        "cluster_contamination": contamination,
        "predicted_id_count": float(pred_count),
        "true_id_count": float(true_count),
        "id_count_abs_error": float(id_count_abs_error),
        "id_count_rel_error": id_count_rel_error,
        "id_count_accuracy": id_count_accuracy,
    }

