"""
Training script for UNet variants using PyTorch 2.x.

Supports: UNet, R2UNet, AttentionUNet, AttentionR2UNet.
Includes mixed precision, cosine LR schedule, gradient clipping, and checkpointing.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from helpers.coco_dataset import CocoPaths, CocoSegmentationDataset, make_transforms
from unet import (
    AttentionR2UNet,
    AttentionUNet,
    R2UNet,
    UNet,
    init_weights,
)

# Optional: import your dataset and evaluation utilities here.
# from your_dataset import SegmentationDataset
# from evaluation import get_accuracy, get_sensitivity, get_specificity, get_precision, get_F1, get_JS, get_DC


ModelName = Literal["unet", "r2unet", "attunet", "attr2unet"]


@dataclass
class TrainConfig:
    data_root: Path
    save_dir: Path
    model: ModelName = "attr2unet"
    t: int = 2
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100
    num_epochs_decay: int = 20
    log_every: int = 25
    amp: bool = True
    compile: bool = False
    resume: Path | None = None
    image_size: int = 256
    coco_json: Path | None = None
    masks_dir: Path | None = None
    images_dir: Path | None = None
    sample_every: int = 0  # steps between sample dumps (0 to disable)
    sample_dir: Path | None = None
    sample_count: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    img_ch: int = 3
    output_ch: int = 1


def build_model(cfg: TrainConfig) -> nn.Module:
    if cfg.model == "unet":
        model = UNet(img_ch=cfg.img_ch, output_ch=cfg.output_ch)
    elif cfg.model == "r2unet":
        model = R2UNet(img_ch=cfg.img_ch, output_ch=cfg.output_ch, t=cfg.t)
    elif cfg.model == "attunet":
        model = AttentionUNet(img_ch=cfg.img_ch, output_ch=cfg.output_ch)
    elif cfg.model == "attr2unet":
        model = AttentionR2UNet(img_ch=cfg.img_ch, output_ch=cfg.output_ch, t=cfg.t)
    else:
        raise ValueError(f"Unknown model type: {cfg.model}")
    init_weights(model, init_type="kaiming")
    if cfg.compile:
        model = torch.compile(model)
    return model


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    denominator = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1 - dice.mean()


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
        },
        path,
    )


def save_debug_samples(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    out_dir: Path,
    tag: str,
    max_samples: int = 3,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    images = images[:max_samples].cpu()
    masks = masks[:max_samples].cpu()
    preds = preds[:max_samples].cpu()

    # Make sure spatial dims match (padding can introduce small differences)
    if preds.shape[-2:] != images.shape[-2:]:
        preds = torch.nn.functional.interpolate(preds, size=images.shape[-2:], mode="nearest")
    if masks.shape[-2:] != images.shape[-2:]:
        masks = torch.nn.functional.interpolate(masks, size=images.shape[-2:], mode="nearest")

    preds_bin = (preds > 0.5).float()

    vutils.save_image(images, out_dir / f"{tag}_images.png", nrow=max_samples, normalize=True)
    vutils.save_image(masks, out_dir / f"{tag}_masks.png", nrow=max_samples)
    vutils.save_image(preds_bin, out_dir / f"{tag}_preds.png", nrow=max_samples)

    # Overlay: image + red prediction
    overlay = images.clone()
    if overlay.shape[1] == 3:
        overlay[:, 0, :, :] = torch.clamp(overlay[:, 0, :, :] + preds_bin.squeeze(1), 0, 1)
    vutils.save_image(overlay, out_dir / f"{tag}_overlay.png", nrow=max_samples, normalize=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    log_every: int = 25,
    sample_every: int = 0,
    sample_dir: Path | None = None,
    sample_count: int = 3,
    global_step_offset: int = 0,
) -> float:
    model.train()
    total_loss = 0.0
    for step, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):
            logits = model(images)
            bce = F.binary_cross_entropy_with_logits(logits, masks)
            probs = torch.sigmoid(logits)
            dloss = dice_loss(probs, masks)
            loss = bce + dloss

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        if (step + 1) % log_every == 0:
            print(f"Step {step+1}/{len(loader)} loss={loss.item():.4f}")

        if sample_every > 0 and sample_dir and ((global_step_offset + step) % sample_every == 0):
            with torch.no_grad():
                preds = torch.sigmoid(logits.detach())
                save_debug_samples(
                    images.detach(),
                    masks.detach(),
                    preds.detach(),
                    out_dir=sample_dir,
                    tag=f"train_step{global_step_offset + step}",
                    max_samples=sample_count,
                )
                print(f"[samples] saved train_step{global_step_offset + step} to {sample_dir}")
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(logits, masks) + dice_loss(probs, masks)
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def get_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    coco_json = cfg.coco_json or (cfg.data_root / "result.json")
    images_dir = cfg.images_dir or (cfg.data_root / "images")
    masks_dir = cfg.masks_dir or (cfg.data_root / "mask")
    masks_dir = masks_dir if masks_dir.exists() else None

    paths = CocoPaths(coco_json=coco_json, images_dir=images_dir, masks_dir=masks_dir)

    base_dataset = CocoSegmentationDataset(paths, transform=None)
    total = len(base_dataset)
    if total < 3:
        raise ValueError("Dataset too small to split into train/val/test.")

    # 70% train, 10% val, 20% test
    train_size = max(1, int(total * 0.7))
    val_size = max(1, int(total * 0.1))
    remaining = total - train_size - val_size
    test_size = max(1, remaining)
    # Adjust if rounding caused overflow
    if train_size + val_size + test_size > total:
        test_size = total - train_size - val_size

    indices = list(range(total))
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices, test_indices = torch.utils.data.random_split(
        indices, [train_size, val_size, test_size], generator=generator
    )

    train_ds = CocoSegmentationDataset(
        paths,
        transform=make_transforms(train=True, size=cfg.image_size),
        indices=list(train_indices),
    )
    val_ds = CocoSegmentationDataset(
        paths,
        transform=make_transforms(train=False, size=cfg.image_size),
        indices=list(val_indices),
    )
    test_ds = CocoSegmentationDataset(
        paths,
        transform=make_transforms(train=False, size=cfg.image_size),
        indices=list(test_indices),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, val_loader, test_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UNet variants with PyTorch 2.x")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, default=Path("runs"))
    parser.add_argument("--model", type=str, default="attr2unet", choices=["unet", "r2unet", "attunet", "attr2unet"])
    parser.add_argument("--t", type=int, default=2, help="recurrent steps for R2 variants")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--num-epochs-decay", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--image-size", type=int, default=600)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--img-ch", type=int, default=3)
    parser.add_argument("--output-ch", type=int, default=1)
    parser.add_argument("--coco-json", type=Path, default=None, help="Path to result.json (default: data_root/result.json)")
    parser.add_argument("--images-dir", type=Path, default=None, help="Directory with COCO images (default: data_root/images)")
    parser.add_argument("--masks-dir", type=Path, default=None, help="Optional precomputed masks directory")
    parser.add_argument("--sample-every", type=int, default=0, help="Steps between saving train samples (0 disables)")
    parser.add_argument("--sample-count", type=int, default=3, help="Number of images per sample dump")
    parser.add_argument("--sample-dir", type=Path, default=None, help="Directory to save training samples")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        save_dir=args.save_dir,
        model=args.model.lower(),  # type: ignore[arg-type]
        t=args.t,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        num_epochs_decay=args.num_epochs_decay,
        log_every=args.log_every,
        amp=args.amp,
        compile=args.compile,
        resume=args.resume,
        image_size=args.image_size,
        coco_json=args.coco_json,
        masks_dir=args.masks_dir,
        images_dir=args.images_dir,
        sample_every=args.sample_every,
        sample_dir=args.sample_dir,
        sample_count=args.sample_count,
        img_ch=args.img_ch,
        output_ch=args.output_ch,
    )

    device = torch.device(cfg.device)
    model = build_model(cfg).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=cfg.amp and device.type == "cuda"
    )

    # Dataset/Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    start_epoch = 0
    if cfg.resume and cfg.resume.is_file():
        checkpoint = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {cfg.resume}, epoch {start_epoch}")

    best_val = float("inf")
    ckpt_dir = cfg.save_dir / cfg.model
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    for epoch in range(start_epoch, cfg.num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            log_every=cfg.log_every,
            sample_every=cfg.sample_every,
            sample_dir=cfg.sample_dir or (cfg.save_dir / "samples"),
            sample_count=cfg.sample_count,
            global_step_offset=epoch * len(train_loader),
        )
        val_loss = evaluate(model, val_loader, device)
        lr_scheduler.step()
        print(f"Epoch {epoch+1}/{cfg.num_epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, ckpt_dir / "best.pt")
            print(f"Saved new best checkpoint at epoch {epoch+1}")

    save_checkpoint(model, optimizer, cfg.num_epochs - 1, ckpt_dir / "last.pt")

    # Final test evaluation
    test_loss = evaluate(model, test_loader, device)
    print(f"[Test] loss={test_loss:.4f}")


if __name__ == "__main__":
    main()
