import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from models.open_world_metrics import compute_open_world_cluster_metrics


def first_folder_of_rel_path(rel_path: Optional[str]) -> Optional[str]:
    if not rel_path:
        return None
    s = str(rel_path).replace('/', os.sep).replace('\\', os.sep)
    parts = [p for p in s.split(os.sep) if p]
    return parts[0] if parts else None


def _iter_prediction_rows(
    detection_records: Sequence[Tuple[str, Sequence[Tuple[str, str, Any]]]],
):
    for rel_path, id_records in detection_records:
        true_id = first_folder_of_rel_path(rel_path) or "unknown"
        for item in id_records:
            if isinstance(item, dict):
                disp_id = str(item.get("display_id") or item.get("predicted_id") or "")
                gender = item.get("gender")
                age = item.get("age")
            else:
                try:
                    disp_id, gender, age = item
                except Exception:
                    continue
                disp_id = str(disp_id)
            if not disp_id:
                continue
            yield {
                "rel_path": str(rel_path),
                "true_id": true_id,
                "predicted_id": disp_id,
                "gender": gender,
                "age": age,
            }


def build_folder_level_metrics(
    detection_records: Sequence[Tuple[str, Sequence[Tuple[str, str, Any]]]],
) -> Dict[str, Any]:
    rows = list(_iter_prediction_rows(detection_records))
    preds = [{"true_id": r["true_id"], "predicted_id": r["predicted_id"]} for r in rows]
    clustering = compute_open_world_cluster_metrics(preds) if preds else {
        "assignment_accuracy": 0.0,
        "cluster_purity": 0.0,
        "cluster_contamination": 1.0,
        "id_count_accuracy": 0.0,
        "predicted_id_count": 0,
        "true_id_count": 0,
        "id_count_abs_error": 0,
        "id_count_rel_error": 0.0,
    }

    pred_to_true = defaultdict(Counter)
    true_to_pred = defaultdict(Counter)
    image_to_pred_count = {}
    multi_det_images = 0
    multi_id_images = 0

    per_image = defaultdict(list)
    for r in rows:
        pred_to_true[r["predicted_id"]][r["true_id"]] += 1
        true_to_pred[r["true_id"]][r["predicted_id"]] += 1
        per_image[r["rel_path"]].append(r["predicted_id"])

    for rel_path, pred_ids in per_image.items():
        image_to_pred_count[rel_path] = len(pred_ids)
        if len(pred_ids) > 1:
            multi_det_images += 1
        if len(set(pred_ids)) > 1:
            multi_id_images += 1

    cluster_details = []
    for pred_id, counter in sorted(pred_to_true.items(), key=lambda kv: str(kv[0])):
        total = int(sum(counter.values()))
        major_true, major_count = counter.most_common(1)[0]
        cluster_details.append({
            "predicted_id": str(pred_id),
            "total": total,
            "major_true_id": str(major_true),
            "major_count": int(major_count),
            "purity": float(major_count / max(1, total)),
            "true_id_breakdown": dict(sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))),
        })

    true_details = []
    for true_id, counter in sorted(true_to_pred.items(), key=lambda kv: str(kv[0])):
        total = int(sum(counter.values()))
        major_pred, major_count = counter.most_common(1)[0]
        true_details.append({
            "true_id": str(true_id),
            "total": total,
            "major_predicted_id": str(major_pred),
            "major_count": int(major_count),
            "assignment_ratio": float(major_count / max(1, total)),
            "predicted_id_breakdown": dict(sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))),
        })

    return {
        "num_prediction_rows": len(rows),
        "num_images_with_predictions": len(per_image),
        "num_multi_detection_images": int(multi_det_images),
        "num_multi_predicted_id_images": int(multi_id_images),
        "clustering": clustering,
        "cluster_details": cluster_details,
        "true_id_details": true_details,
    }


def save_folder_level_metrics(
    output_dir: str,
    detection_records: Sequence[Tuple[str, Sequence[Tuple[str, str, Any]]]],
    detection_details: Optional[List[Dict[str, Any]]] = None,
    extra_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics = build_folder_level_metrics(detection_records)
    payload: Dict[str, Any] = {
        "folder_level_metrics": metrics,
    }
    if detection_details is not None:
        payload["detection_details"] = detection_details
    if extra_summary is not None:
        payload["summary"] = extra_summary

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "detection_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload
