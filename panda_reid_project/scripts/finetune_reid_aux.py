#!/usr/bin/env python3

import argparse
from contextlib import nullcontext
import datetime
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from panda_reid_core.config import get_config
from panda_reid_core.data.roiimg_dataset import PandaRoiImgDataset
from panda_reid_core.models.panda_reid_model import build_panda_reid_model
from scripts.train_age_gender import (
    build_balanced_sampler,
    build_eval_transform,
    build_scheduler,
    build_train_transform,
    compute_age_class_weights,
    compute_gender_class_weights,
    maybe_subsample_dataset,
    plot_curves,
    safe_mkdir,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser("Age/Gender aux-only finetune from ReID multitask checkpoint")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--resume-ckpt", type=str, required=True)
    parser.add_argument("--train-roiimg-root", type=str, required=True)
    parser.add_argument("--val-roiimg-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--aux-detach", action="store_true")
    parser.add_argument("--aux-grad-ratio", type=float, default=1.0)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-neck", action="store_true")
    parser.add_argument("--gender-loss-weight", type=float, default=1.0)
    parser.add_argument("--age-loss-weight", type=float, default=1.0)
    parser.add_argument("--age-huber-beta", type=float, default=1.0)
    parser.add_argument("--max-age-bin", type=int, default=40)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--disable-channels-last", action="store_true")
    parser.add_argument(
        "--sampler-mode",
        type=str,
        default="gender_age",
        choices=["none", "gender", "age", "gender_age"],
    )
    parser.add_argument("--age-sampler-bin-size", type=int, default=2)
    parser.add_argument("--age-sampler-power", type=float, default=0.5)
    return parser.parse_args()


def load_cfg(cfg_path: str):
    class _Tmp:
        pass

    tmp = _Tmp()
    tmp.cfg = cfg_path
    tmp.opts = []
    return get_config(tmp)


def load_checkpoint_filtered(model, ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_sd = model.state_dict()
    filtered = {}
    for key, value in state.items():
        if key in model_sd and getattr(value, "shape", None) == getattr(model_sd[key], "shape", None):
            filtered[key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return checkpoint, len(filtered), len(missing), len(unexpected)


def set_module_trainable(module, requires_grad: bool):
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = bool(requires_grad)


def forward_aux_outputs(model, images, freeze_backbone: bool = False, freeze_neck: bool = False):
    if not freeze_backbone and not freeze_neck:
        return model.forward_multitask(images)

    backbone_ctx = torch.no_grad() if freeze_backbone else nullcontext()
    with backbone_ctx:
        if hasattr(model, "_forward_backbone"):
            backbone_feat = model._forward_backbone(images)
        else:
            backbone_feat = model.backbone.forward_features(images)
        embed_feat = model.proj(backbone_feat) if getattr(model, "proj", None) is not None else backbone_feat

    neck_ctx = torch.no_grad() if (freeze_backbone or freeze_neck) else nullcontext()
    with neck_ctx:
        feat_before_bn, feat_after_bn, _cls_score = model.neck(embed_feat)

    aux_feat = feat_after_bn.detach() if (freeze_backbone or freeze_neck or getattr(model, "aux_detach", False)) else feat_after_bn
    gender_logits = model.gender_head(aux_feat)
    age_pred = model.age_head(aux_feat).squeeze(1)
    return feat_after_bn, feat_before_bn, gender_logits, age_pred


@dataclass
class EpochStats:
    train_loss: float
    train_gender_loss: float
    train_age_loss: float
    val_gender_acc: float
    val_age_mae: float
    val_age_rmse: float
    val_age_within_1: float
    val_age_within_2: float
    val_age_within_3: float


def run_eval(model, loader, device) -> Dict[str, float]:
    model.eval()
    gender_correct = 0
    gender_total = 0
    age_abs_err = []
    age_sq_err = []
    age_within_1 = []
    age_within_2 = []
    age_within_3 = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            gender_t = batch["gender_label"].to(device, non_blocking=True).long()
            age_t = batch["age_years"].to(device, non_blocking=True).float()
            age_valid = batch["age_valid"].to(device, non_blocking=True).float() > 0.5

            _fa, _fb, gender_logits, age_pred = model.forward_multitask(images)
            gender_p = torch.argmax(gender_logits, dim=1)
            gender_correct += int((gender_p == gender_t).sum().item())
            gender_total += int(gender_t.numel())

            if age_valid.any():
                diff = (age_pred[age_valid] - age_t[age_valid]).float()
                abs_diff = torch.abs(diff)
                age_abs_err.extend(abs_diff.cpu().numpy().tolist())
                age_sq_err.extend((diff * diff).cpu().numpy().tolist())
                age_within_1.extend((abs_diff <= 1.0).cpu().numpy().tolist())
                age_within_2.extend((abs_diff <= 2.0).cpu().numpy().tolist())
                age_within_3.extend((abs_diff <= 3.0).cpu().numpy().tolist())

    gender_acc = float(gender_correct / max(gender_total, 1))
    if age_abs_err:
        age_mae = float(np.mean(age_abs_err))
        age_rmse = float(np.sqrt(np.mean(age_sq_err)))
        within_1 = float(np.mean(age_within_1))
        within_2 = float(np.mean(age_within_2))
        within_3 = float(np.mean(age_within_3))
    else:
        age_mae = 999.0
        age_rmse = 999.0
        within_1 = 0.0
        within_2 = 0.0
        within_3 = 0.0

    return {
        "gender_acc": gender_acc,
        "age_mae": age_mae,
        "age_rmse": age_rmse,
        "age_within_1": within_1,
        "age_within_2": within_2,
        "age_within_3": within_3,
    }


def main():
    args = parse_args()
    set_seed(int(args.seed))
    safe_mkdir(args.output_dir)

    config = load_cfg(args.cfg)
    config.defrost()
    config.MODEL.AUX_DETACH = bool(args.aux_detach)
    config.MODEL.AUX_GRAD_RATIO = float(args.aux_grad_ratio)
    config.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (not args.disable_amp) and torch.cuda.is_available()
    channels_last = (not args.disable_channels_last)

    train_tf = build_train_transform(int(args.img_size))
    val_tf = build_eval_transform(int(args.img_size))

    train_dataset = PandaRoiImgDataset(
        roiimg_root=args.train_roiimg_root,
        img_size=int(args.img_size),
        transform=train_tf,
        is_train=True,
    )
    val_dataset = PandaRoiImgDataset(
        roiimg_root=args.val_roiimg_root,
        img_size=int(args.img_size),
        transform=val_tf,
        is_train=False,
    )

    maybe_subsample_dataset(train_dataset, int(args.max_train_samples), int(args.seed))
    maybe_subsample_dataset(val_dataset, int(args.max_val_samples), int(args.seed) + 1)

    gender_w = compute_gender_class_weights(train_dataset).to(device)
    age_w = compute_age_class_weights(train_dataset, int(args.max_age_bin)).to(device)
    train_sampler = build_balanced_sampler(
        dataset=train_dataset,
        mode=args.sampler_mode,
        max_age_bin=int(args.max_age_bin),
        age_bin_size=int(args.age_sampler_bin_size),
        power=float(args.age_sampler_power),
    )

    loader_kwargs = {}
    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, int(args.batch_size) // 2),
        shuffle=False,
        num_workers=max(1, int(args.num_workers) // 2),
        pin_memory=True,
        drop_last=False,
        **({} if int(args.num_workers) == 0 else {"persistent_workers": True, "prefetch_factor": 2}),
    )

    model = build_panda_reid_model(config, num_classes=train_dataset.get_num_classes())
    checkpoint, loaded_count, missing_count, unexpected_count = load_checkpoint_filtered(model, args.resume_ckpt)
    print(
        f"[INFO] loaded resume checkpoint: {args.resume_ckpt}\n"
        f"       tensors={loaded_count}, missing={missing_count}, unexpected={unexpected_count}"
    )
    model = model.to(device)
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    set_module_trainable(model.backbone, not args.freeze_backbone)
    set_module_trainable(getattr(model, "proj", None), not args.freeze_backbone)
    set_module_trainable(model.neck, not args.freeze_neck)
    set_module_trainable(model.gender_head, True)
    set_module_trainable(model.age_head, True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"[INFO] freeze_backbone={bool(args.freeze_backbone)}, "
        f"freeze_neck={bool(args.freeze_neck)}, trainable_params={sum(p.numel() for p in trainable_params):,}"
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
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

    history: List[Dict] = []
    best_score = -1e9
    best_epoch = 0
    best_path = os.path.join(args.output_dir, "age_gender_aux_best.pth")

    print(f"[INFO] train samples={len(train_dataset)}, val samples={len(val_dataset)}")
    print(f"[INFO] sampler_mode={args.sampler_mode}, batch={args.batch_size}, epochs={args.epochs}")
    start_all = datetime.datetime.now()
    accumulation = max(1, int(args.accumulation_steps))

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        if args.freeze_backbone:
            model.backbone.eval()
            if getattr(model, "proj", None) is not None:
                model.proj.eval()
        if args.freeze_neck:
            model.neck.eval()
        total_loss_meter = 0.0
        gender_loss_meter = 0.0
        age_loss_meter = 0.0
        sample_meter = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            if channels_last and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            gender_t = batch["gender_label"].to(device, non_blocking=True).long()
            age_t = batch["age_years"].to(device, non_blocking=True).float()
            age_valid = batch["age_valid"].to(device, non_blocking=True).float() > 0.5

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                _fa, _fb, gender_logits, age_pred = forward_aux_outputs(
                    model,
                    images,
                    freeze_backbone=bool(args.freeze_backbone),
                    freeze_neck=bool(args.freeze_neck),
                )
                gender_loss = F.cross_entropy(
                    gender_logits,
                    gender_t,
                    weight=gender_w,
                    label_smoothing=float(args.label_smoothing),
                )
                if age_valid.any():
                    age_bins = torch.clamp(
                        torch.round(age_t[age_valid]).long(),
                        min=0,
                        max=int(args.max_age_bin),
                    )
                    age_sample_w = age_w[age_bins]
                    age_loss_raw = F.smooth_l1_loss(
                        age_pred[age_valid],
                        age_t[age_valid],
                        beta=float(args.age_huber_beta),
                        reduction="none",
                    )
                    age_loss = (age_loss_raw * age_sample_w).mean()
                else:
                    age_loss = torch.zeros([], device=device)

                loss = (
                    float(args.gender_loss_weight) * gender_loss
                    + float(args.age_loss_weight) * age_loss
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

            batch_size = int(images.size(0))
            sample_meter += batch_size
            total_loss_meter += float(loss.item()) * accumulation * batch_size
            gender_loss_meter += float(gender_loss.item()) * batch_size
            age_loss_meter += float(age_loss.item()) * batch_size

            if step % max(1, int(args.print_freq)) == 0:
                print(
                    f"[epoch {epoch:02d}] step {step:04d}/{len(train_loader):04d} "
                    f"lr={optimizer.param_groups[0]['lr']:.6g} "
                    f"loss={total_loss_meter / max(sample_meter, 1):.4f} "
                    f"gender={gender_loss_meter / max(sample_meter, 1):.4f} "
                    f"age={age_loss_meter / max(sample_meter, 1):.4f}"
                )

        scheduler.step()
        eval_stats = run_eval(model, val_loader, device)
        epoch_stats = EpochStats(
            train_loss=float(total_loss_meter / max(sample_meter, 1)),
            train_gender_loss=float(gender_loss_meter / max(sample_meter, 1)),
            train_age_loss=float(age_loss_meter / max(sample_meter, 1)),
            val_gender_acc=float(eval_stats["gender_acc"]),
            val_age_mae=float(eval_stats["age_mae"]),
            val_age_rmse=float(eval_stats["age_rmse"]),
            val_age_within_1=float(eval_stats["age_within_1"]),
            val_age_within_2=float(eval_stats["age_within_2"]),
            val_age_within_3=float(eval_stats["age_within_3"]),
        )
        row = {"epoch": int(epoch), **epoch_stats.__dict__, "lr": float(optimizer.param_groups[0]["lr"])}
        history.append(row)

        score = float(epoch_stats.val_gender_acc) - float(epoch_stats.val_age_mae) / 10.0
        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": int(epoch),
                    "config": config,
                    "metrics": row,
                    "resume_ckpt": args.resume_ckpt,
                    "args": vars(args),
                },
                best_path,
            )

        torch.save(
            {
                "model": model.state_dict(),
                "epoch": int(epoch),
                "config": config,
                "metrics": row,
                "resume_ckpt": args.resume_ckpt,
                "args": vars(args),
            },
            os.path.join(args.output_dir, f"age_gender_aux_epoch_{epoch:03d}.pth"),
        )

        with open(os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        plot_curves(history, os.path.join(args.output_dir, "training_curves.png"))

        print(
            f"[epoch {epoch:02d}] val_gender_acc={epoch_stats.val_gender_acc:.4f} "
            f"val_age_mae={epoch_stats.val_age_mae:.4f} "
            f"val_age_rmse={epoch_stats.val_age_rmse:.4f} "
            f"within(1/2/3)=({epoch_stats.val_age_within_1:.3f}/"
            f"{epoch_stats.val_age_within_2:.3f}/{epoch_stats.val_age_within_3:.3f}) "
            f"{'[BEST]' if is_best else ''}"
        )

    elapsed = datetime.datetime.now() - start_all
    print(f"[INFO] training done in {elapsed}, best_epoch={best_epoch}, best_score={best_score:.4f}")
    print(f"[INFO] best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
