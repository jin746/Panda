# Panda ReID Code Package

This folder contains the source code needed for Panda ReID training, testing, and runtime inference.

## Layout

- `configs/`: selected training and inference configuration files.
- `panda_reid_core/`: reusable data, model, loss, optimizer, inference, SAM, and CUDA-extension code.
- `scripts/`: command-line entry points for training, fine-tuning, prototype inference, and open-world video inference.

## Excluded Assets

Datasets, model weights, checkpoints, generated outputs, videos, figures, reports, IDE files, cache files, and vendored third-party packages are intentionally excluded. Place required runtime assets such as model weights under a local `weights/` directory before running inference.
