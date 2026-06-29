# Panda ReID

This repository contains only the core code needed for panda ReID model
training, testing, and inference.

## Contents

- `configs/`: training and inference configuration files.
- `data/`: dataset loading and transform utilities.
- `models/`: Swin/Panda ReID model definitions, losses, and metrics.
- `inference_modules/`: inference helpers and open-world ReID utilities.
- `segment_anything/`: SAM wrapper code used by the runtime pipeline.
- `kernels/`: optional Swin window-process CUDA extension.
- Top-level `train_*.py` and `video_open_world_reid_yewai*.py` entry points.

## Excluded

Datasets, model weights, generated outputs, videos, figures, paper/report files,
analysis scripts, vendored third-party packages, and local IDE/cache files are
not tracked. Restore those assets locally before running workflows that depend
on them.
