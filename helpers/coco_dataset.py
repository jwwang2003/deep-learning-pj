"""COCO segmentation dataset + augmentations"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .coco import COCO, COCOAnnotation, COCOImage
from .coco_mask_utils import add_annotation_to_mask
from .data_augmentations import PairTransform, make_transforms


@dataclass
class CocoPaths:
    coco_json: Path
    images_dir: Path
    masks_dir: Optional[Path] = None  # optional precomputed masks


def _load_or_build_mask(
    img: COCOImage,
    anns: Iterable[COCOAnnotation],
    images_dir: Path,
    masks_dir: Optional[Path],
) -> Image.Image:
    """Load a precomputed mask if available, otherwise rasterize polygons/boxes."""
    src = images_dir / Path(img.file_name).name
    if masks_dir:
        candidate = masks_dir / f"{src.stem}_mask.png"
        if candidate.exists():
            return Image.open(candidate).convert("L")

    mask = Image.new("L", (int(img.width), int(img.height)), 0)
    draw = ImageDraw.Draw(mask)
    for ann in anns:
        if add_annotation_to_mask(mask, ann):
            continue
        if ann.bbox:
            x, y, w, h = ann.bbox
            draw.rectangle([x, y, x + w, y + h], fill=255)
    return mask


class CocoSegmentationDataset(Dataset):
    """Lightweight COCO-based dataset that produces image/mask pairs."""

    def __init__(
        self,
        paths: CocoPaths,
        transform: Optional[Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]] = None,
        limit: Optional[int] = None,
        indices: Optional[List[int]] = None,
    ) -> None:
        self.coco = COCO.from_file(paths.coco_json)
        base_images: List[COCOImage] = self.coco.images
        if indices is not None:
            self.images = [base_images[i] for i in indices]
        else:
            self.images = base_images[:limit] if limit is not None else base_images
        self.images_dir = paths.images_dir
        self.masks_dir = paths.masks_dir
        self.transform = transform or make_transforms(train=False)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_meta = self.images[idx]
        img_path = self.images_dir / Path(img_meta.file_name).name
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        anns = self.coco.get_annotations(img_meta.id)
        mask = _load_or_build_mask(img_meta, anns, self.images_dir, self.masks_dir)

        image_t, mask_t = self.transform(image, mask)
        return image_t, mask_t


# ----------------------------
# Demo utilities
# ----------------------------

def tensor_to_pil_rgb(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(0, 1)
    return TF.to_pil_image(x)


def tensor_to_pil_mask(x: torch.Tensor) -> Image.Image:
    # x: [1,H,W] float {0,1}
    m = (x.detach().cpu().squeeze(0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(m, mode="L")


def save_overlay(image_rgb: Image.Image, mask_l: Image.Image, out_path: Path, alpha: float = 0.45) -> None:
    base = image_rgb.convert("RGBA")
    mask = mask_l.convert("L")
    # convert mask to alpha channel
    alpha_mask = mask.point(lambda p: int(p * max(0.0, min(alpha, 1.0))))
    red = Image.new("RGBA", base.size, (255, 0, 0, 0))
    red.putalpha(alpha_mask)
    comp = Image.alpha_composite(base, red)
    comp.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser("augmentation demo writer")
    ap.add_argument("coco_json", type=str, help="Path to COCO json file (e.g., result.json)")
    ap.add_argument("--images-dir", type=str, default=None, help="Images directory (default: coco_json.parent/images)")
    ap.add_argument("--masks-dir", type=str, default=None, help="Precomputed masks directory (optional)")
    ap.add_argument("--out-dir", type=str, default=None, help="Output dir (default: coco_json.parent/aug_demo)")
    ap.add_argument("--num", type=int, default=8, help="How many samples to export per mode")
    ap.add_argument("--size", type=int, default=600, help="Transform size")
    ap.add_argument("--jitter", type=float, default=0.4, help="Strong-mode jitter strength")
    ap.add_argument("--seed", type=int, default=0, help="Seed for reproducible samples")
    ap.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha")
    ap.add_argument("--pick", choices=["first", "random"], default="random", help="How to select samples")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    coco_path = Path(args.coco_json)
    root = coco_path.parent
    images_dir = Path(args.images_dir) if args.images_dir else (root / "images")
    masks_dir = Path(args.masks_dir) if args.masks_dir else (root / "mask" if (root / "mask").exists() else None)
    out_root = Path(args.out_dir) if args.out_dir else (root / "aug_demo")

    coco = COCO.from_file(coco_path)
    all_imgs: List[COCOImage] = list(coco.images)

    if len(all_imgs) == 0:
        raise RuntimeError(f"No images found in {coco_path}")

    if args.pick == "first":
        chosen = all_imgs[: args.num]
    else:
        k = min(args.num, len(all_imgs))
        chosen = random.sample(all_imgs, k=k)

    modes = ["none", "light", "strong"]

    print("=== Augmentation demo parameters ===")
    print(f"coco_json     : {coco_path}")
    print(f"images_dir    : {images_dir}")
    print(f"masks_dir     : {masks_dir}")
    print(f"out_dir       : {out_root}")
    print(f"num           : {args.num}")
    print(f"size          : {args.size}")
    print(f"jitter        : {args.jitter}")
    print(f"seed          : {args.seed}")
    print(f"overlay alpha : {args.alpha}")
    print(f"pick          : {args.pick}")
    print("===================================")

    for mode in modes:
        # train=True to actually show augmentations for light/strong
        tfm = PairTransform(train=True, size=args.size, aug_mode=mode, jitter_strength=args.jitter)

        mode_dir = out_root / mode
        img_out = mode_dir / "images"
        msk_out = mode_dir / "masks"
        ovl_out = mode_dir / "overlays"
        img_out.mkdir(parents=True, exist_ok=True)
        msk_out.mkdir(parents=True, exist_ok=True)
        ovl_out.mkdir(parents=True, exist_ok=True)

        for i, meta in enumerate(chosen):
            src_path = images_dir / Path(meta.file_name).name
            if not src_path.exists():
                print(f"[skip] missing image: {src_path}")
                continue

            image = Image.open(src_path).convert("RGB")
            anns = coco.get_annotations(meta.id)
            mask = _load_or_build_mask(meta, anns, images_dir, masks_dir)

            # apply transform
            x, y = tfm(image, mask)

            # save
            base_name = f"{i:03d}_{src_path.stem}"
            out_img = img_out / f"{base_name}.png"
            out_msk = msk_out / f"{base_name}_mask.png"
            out_ovl = ovl_out / f"{base_name}_overlay.png"

            pil_img = tensor_to_pil_rgb(x)
            pil_msk = tensor_to_pil_mask(y)

            pil_img.save(out_img)
            pil_msk.save(out_msk)
            save_overlay(pil_img, pil_msk, out_ovl, alpha=args.alpha)

            print(f"[{mode}] wrote: {out_img.name}, {out_msk.name}, {out_ovl.name}")

    print(f"\nDone. See outputs in: {out_root}")


if __name__ == "__main__":
    main()
