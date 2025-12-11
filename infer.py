"""
Inference demo for Attention R2UNet.

Loads a trained checkpoint, runs inference on a few random samples from the COCO-derived
segmentation dataset, and saves predictions (and optional overlays) to disk.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from helpers.coco_dataset import CocoPaths, CocoSegmentationDataset, make_transforms
from unet import AttentionR2UNet


def save_prediction(
    image_t: torch.Tensor,
    mask_pred: torch.Tensor,
    out_dir: Path,
    stem: str,
    mask_gt: torch.Tensor | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    image = TF.to_pil_image(image_t)
    pred_mask = TF.to_pil_image(mask_pred)
    image.save(out_dir / f"{stem}_image.png")
    pred_mask.save(out_dir / f"{stem}_pred.png")

    if mask_gt is not None:
        gt_mask = TF.to_pil_image(mask_gt)
        gt_mask.save(out_dir / f"{stem}_gt.png")

        # Ensure all masks match the image size
        if pred_mask.size != image.size:
            pred_mask = pred_mask.resize(image.size, resample=Image.NEAREST)
        if gt_mask.size != image.size:
            gt_mask = gt_mask.resize(image.size, resample=Image.NEAREST)

        # Simple overlay: red prediction, green ground truth
        image_rgba = image.convert("RGBA")
        red = Image.new("RGBA", image.size, (255, 0, 0, 0))
        red.putalpha(pred_mask)
        green = Image.new("RGBA", image.size, (0, 255, 0, 0))
        green.putalpha(gt_mask)
        composite = Image.alpha_composite(image_rgba, red)
        composite = Image.alpha_composite(composite, green)
        composite.save(out_dir / f"{stem}_overlay.png")


def run_inference(
    checkpoint: Path,
    coco_json: Path,
    images_dir: Path,
    masks_dir: Path | None,
    num_samples: int = 3,
    image_size: int = 256,
    t: int = 2,
    threshold: float = 0.5,
    output_dir: Path = Path("inference_outputs"),
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    device = torch.device(device)

    model = AttentionR2UNet(img_ch=3, output_ch=1, t=t)
    state = torch.load(checkpoint, map_location=device)
    if "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    paths = CocoPaths(coco_json=coco_json, images_dir=images_dir, masks_dir=masks_dir)
    dataset = CocoSegmentationDataset(paths, transform=make_transforms(train=False, size=image_size))
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; check paths.")

    indices = random.sample(range(len(dataset)), k=min(num_samples, len(dataset)))

    with torch.no_grad():
        for idx in indices:
            image_t, mask_gt = dataset[idx]
            image_t = image_t.to(device).unsqueeze(0)
            logits = model(image_t)
            probs = torch.sigmoid(logits)
            pred = (probs > threshold).float()

            # Move to CPU for saving
            image_cpu = image_t.squeeze(0).cpu()
            pred_cpu = pred.squeeze(0).cpu()
            gt_cpu = mask_gt.cpu() if mask_gt is not None else None

            save_prediction(
                image_cpu,
                pred_cpu,
                out_dir=output_dir,
                stem=f"sample_{idx}",
                mask_gt=gt_cpu,
            )
            print(f"[ok] saved sample_{idx} to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attention R2UNet inference demo")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root directory")
    parser.add_argument("--coco-json", type=Path, default=None, help="Path to result.json (default: data_root/result.json)")
    parser.add_argument("--images-dir", type=Path, default=None, help="Directory with images (default: data_root/images)")
    parser.add_argument("--masks-dir", type=Path, default=None, help="Optional masks directory")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of random samples to run")
    parser.add_argument("--image-size", type=int, default=256, help="Resize side before padding")
    parser.add_argument("--t", type=int, default=2, help="Recurrent steps for AttentionR2UNet")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for mask binarization")
    parser.add_argument("--output-dir", type=Path, default=Path("inference_outputs"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    coco_json = args.coco_json or (args.data_root / "result.json")
    images_dir = args.images_dir or (args.data_root / "images")
    masks_dir = args.masks_dir or (args.data_root / "mask")
    masks_dir = masks_dir if masks_dir.exists() else None

    run_inference(
        checkpoint=args.checkpoint,
        coco_json=coco_json,
        images_dir=images_dir,
        masks_dir=masks_dir,
        num_samples=args.num_samples,
        image_size=args.image_size,
        t=args.t,
        threshold=args.threshold,
        output_dir=args.output_dir,
        device=args.device,
    )
