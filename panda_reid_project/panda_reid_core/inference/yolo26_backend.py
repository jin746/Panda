import os
import sys
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class YOLO26RuntimeEnv:
    project_root: str
    vendor_root: str
    ultralytics_file: str
    ultralytics_version: str
    yolo_class: Any
    default_weight: str
    default_tracker: str


def bootstrap_local_yolo26_runtime(project_root: str) -> YOLO26RuntimeEnv:
    """Resolve YOLO runtime without requiring vendored code in git."""
    proj = os.path.abspath(str(project_root))
    vendor_root = os.path.join(proj, "third_party", "ultralytics_yolo26")
    pkg_root = os.path.join(vendor_root, "ultralytics")

    if os.path.isdir(pkg_root) and vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)

    try:
        from ultralytics import YOLO
        import ultralytics as ultralytics_pkg
    except Exception:
        YOLO = None
        ultralytics_file = ""
        ultralytics_version = "unavailable"
    else:
        ultralytics_file = str(getattr(ultralytics_pkg, "__file__", "") or "")
        ultralytics_version = str(getattr(ultralytics_pkg, "__version__", "unknown") or "unknown")

    default_weight = os.path.abspath(
        os.path.join(proj, "weights", "panda_yolo26s_aug_best.pt")
    )
    default_tracker = os.path.join(vendor_root, "ultralytics", "cfg", "trackers", "bytetrack.yaml")
    if not os.path.isfile(default_tracker):
        default_tracker = ""

    return YOLO26RuntimeEnv(
        project_root=proj,
        vendor_root=os.path.abspath(vendor_root),
        ultralytics_file=ultralytics_file,
        ultralytics_version=ultralytics_version,
        yolo_class=YOLO,
        default_weight=default_weight,
        default_tracker=default_tracker,
    )


def resolve_existing_path(path_value: Optional[str], bases=None) -> Optional[str]:
    if path_value is None:
        return None
    p = str(path_value).strip().strip('"').strip("'")
    if p == "":
        return p
    if os.path.exists(p):
        return os.path.abspath(p)
    if os.path.isabs(p):
        return p
    for b in (bases or []):
        if not b:
            continue
        c = os.path.join(str(b), p)
        if os.path.exists(c):
            return os.path.abspath(c)
    return p


def resolve_detector_runtime_paths(args, env: YOLO26RuntimeEnv):
    """Resolve detector/tracker paths without any external repo dependency."""
    repo_root = getattr(args, "yolo_repo_root", None) or env.vendor_root
    repo_root = os.path.abspath(str(repo_root))
    setattr(args, "yolo_repo_root", repo_root)

    bases = [repo_root, env.project_root, env.vendor_root]
    args.det_model = resolve_existing_path(getattr(args, "det_model", None), bases=bases)
    args.tracker = resolve_existing_path(getattr(args, "tracker", None), bases=bases)

    if (not args.det_model) or (not os.path.exists(str(args.det_model))):
        args.det_model = resolve_existing_path(env.default_weight, bases=bases)
    if (not args.tracker) or (not os.path.exists(str(args.tracker))):
        args.tracker = resolve_existing_path(env.default_tracker, bases=bases)
    return args.det_model, args.tracker, repo_root


def build_yolo_detector(args, env: YOLO26RuntimeEnv, verbose: bool = True):
    det_model, tracker_cfg, repo_root = resolve_detector_runtime_paths(args, env)
    if not det_model:
        raise ValueError("detector model path is empty")

    yolo_class = env.yolo_class
    if yolo_class is None:
        try:
            from ultralytics import YOLO as yolo_class
        except Exception as exc:
            raise ImportError(
                "Ultralytics is required for detector inference. Install it or provide "
                "a local third_party/ultralytics_yolo26 runtime."
            ) from exc

    ext = os.path.splitext(str(det_model))[1].lower()
    if ext in {".pt", ".onnx", ".engine", ".tflite", ".xml", ".yaml", ".yml"}:
        has_sep = (os.path.sep in str(det_model)) or ("/" in str(det_model)) or ("\\" in str(det_model))
        if (os.path.isabs(str(det_model)) or has_sep) and not os.path.exists(str(det_model)):
            raise FileNotFoundError(f"Detector weights/model not found: {det_model}")

    yolo = yolo_class(str(det_model))
    if verbose:
        src = str(env.ultralytics_file).replace("/", "\\")
        print(f"[INFO] Ultralytics backend: version={env.ultralytics_version}, module={src}")
        print(f"[INFO] YOLO repo root: {repo_root}")
        print(f"[INFO] Detector model: {det_model}")
        if tracker_cfg:
            print(f"[INFO] Tracker cfg: {tracker_cfg}")
    return yolo
