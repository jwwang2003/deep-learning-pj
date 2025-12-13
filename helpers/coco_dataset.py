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
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import RandomResizedCrop

from .coco import COCO, COCOAnnotation, COCOImage


@dataclass
class CocoPaths:
    coco_json: Path
    images_dir: Path
    masks_dir: Optional[Path] = None  # optional precomputed masks


class PairTransform:
    """
    Pickle-safe paired transform for image/mask.

    aug_mode:
      - "none"   : resize + pad only, no augmentation
      - "light"  : ORIGINAL augmentations (to reproduce overfitting runs)
      - "strong" : stronger augmentations (more geometry + color jitter + noise)
    """

    def __init__(
        self,
        train: bool,
        size: int = 600,
        aug_mode: str = "light",
        jitter_strength: float = 0.3,
    ) -> None:
        self.train = train
        self.size = size
        self.aug_mode = aug_mode
        self.jitter_strength = jitter_strength

        if self.aug_mode not in {"none", "light", "strong"}:
            raise ValueError(f"Invalid aug_mode={aug_mode!r}, expected 'none' | 'light' | 'strong'.")

    def _pad_to_multiple_of_16(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Pad to next multiple of 16 in width/height."""
        pad_w = (16 - (image.width % 16)) % 16
        pad_h = (16 - (image.height % 16)) % 16
        if pad_w or pad_h:
            padding = (0, 0, pad_w, pad_h)  # left, top, right, bottom
            image = TF.pad(image, padding, fill=0)
            mask = TF.pad(mask, padding, fill=0)
        return image, mask

    def _apply_light_aug(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Your ORIGINAL augmentations:

        - hflip p=0.5
        - vflip p=0.2
        - rotation [-10, 10] deg, p=0.3
        - brightness & contrast in [0.8, 1.2], p=0.4

        Note: in the original code, padding happened BEFORE augmentation.
        We keep that behavior in 'light' mode to reproduce earlier runs.
        """
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.2:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() < 0.3:
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        if random.random() < 0.4:
            image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.8, 1.2))

        return image, mask

    def _apply_strong_aug(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Stronger augmentations:

        - hflip p=0.5
        - vflip p=0.2
        - random affine (rot/translate/scale/shear) p=0.7
        - color jitter (brightness/contrast/sat/hue) p=0.8 on image only

        Padding is applied AFTER geometry in 'strong' mode.
        """
        # Geometric: shared between image & mask
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.2:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() < 0.7:
            angle = random.uniform(-15.0, 15.0)
            translate_frac = 0.05
            max_dx = translate_frac * image.width
            max_dy = translate_frac * image.height
            translate = (
                random.uniform(-max_dx, max_dx),
                random.uniform(-max_dy, max_dy),
            )
            scale = random.uniform(0.9, 1.1)
            shear = random.uniform(-5.0, 5.0)

            image = TF.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

        # Photometric: image only
        if random.random() < 0.9:
            s = self.jitter_strength  # e.g. 0.4 for these images
            # brightness/contrast/saturation
            image = TF.adjust_brightness(image, random.uniform(1.0 - s, 1.0 + s))
            image = TF.adjust_contrast(image,   random.uniform(1.0 - s, 1.0 + s))
            image = TF.adjust_saturation(image, random.uniform(1.0 - s, 1.0 + s))

            # gamma handles "washed out" vs "too dark" better than linear brightness
            if random.random() < 0.7:
                gamma = random.uniform(0.6, 1.6)
                image = TF.adjust_gamma(image, gamma=gamma, gain=1.0)

            # hue small, but helpful for warm/cool tone shifts
            if random.random() < 0.5:
                image = TF.adjust_hue(image, random.uniform(-0.06, 0.06))

            # sometimes images are effectively monochrome
            if random.random() < 0.15:
                image = TF.rgb_to_grayscale(image, num_output_channels=3)

            # mimic camera processing differences
            if random.random() < 0.25:
                image = TF.autocontrast(image)
            if random.random() < 0.15:
                image = TF.equalize(image)

            # mild blur to reduce overfitting to sharp edges / halos
            if random.random() < 0.20:
                sigma = random.uniform(0.1, 1.2)
                image = TF.gaussian_blur(image, kernel_size=[3, 3], sigma=sigma)

        return image, mask
    
    def _resize_letterbox(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Resize longest side to self.size while preserving aspect ratio, then pad to (size,size).
        This avoids geometric distortion vs TF.resize((size,size)).
        """
        w, h = image.size
        scale = self.size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        image = TF.resize(image, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  (new_h, new_w), interpolation=InterpolationMode.NEAREST)

        pad_w = self.size - new_w
        pad_h = self.size - new_h
        # pad right/bottom to keep it simple and deterministic
        image = TF.pad(image, (0, 0, pad_w, pad_h), fill=0)
        mask  = TF.pad(mask,  (0, 0, pad_w, pad_h), fill=0)
        return image, mask
    
    def _random_resized_crop_pair(
        self, image: Image.Image, mask: Image.Image,
        scale: Tuple[float, float] = (0.7, 1.0),
        ratio: Tuple[float, float] = (0.9, 1.1),
        p: float = 0.6,
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() > p:
            return image, mask
        
        # use torchvision's get_params:
        crop_i, crop_j, crop_h, crop_w = RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
        image = TF.resized_crop(image, crop_i, crop_j, crop_h, crop_w, (self.size, self.size),
                                interpolation=InterpolationMode.BILINEAR)
        mask  = TF.resized_crop(mask,  crop_i, crop_j, crop_h, crop_w, (self.size, self.size),
                                interpolation=InterpolationMode.NEAREST)
        return image, mask

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- 1) Resize ---
        # image = TF.resize(image, (self.size, self.size), interpolation=InterpolationMode.BILINEAR)
        # mask = TF.resize(mask, (self.size, self.size), interpolation=InterpolationMode.NEAREST)
        image, mask = self._resize_letterbox(image, mask)
        
        # --- 2) Padding / augmentation order depends on mode ---
        if not self.train:
            # Val / test: always pad, no aug
            image, mask = self._pad_to_multiple_of_16(image, mask)

        elif self.aug_mode == "none":
            # Train but no augmentation: same as val/test, only resize + pad
            image, mask = self._pad_to_multiple_of_16(image, mask)

        elif self.aug_mode == "light":
            # ORIGINAL behavior: pad first, then light aug
            image, mask = self._apply_light_aug(image, mask)
            image, mask = self._pad_to_multiple_of_16(image, mask)

        elif self.aug_mode == "strong":
            # Strong aug: geometry first, then pad
            image, mask = self._apply_strong_aug(image, mask)
            image, mask = self._random_resized_crop_pair(image, mask, scale=(0.65, 1.0), ratio=(0.85, 1.15), p=0.7)
            image, mask = self._pad_to_multiple_of_16(image, mask)

        # --- 3) To tensor + optional noise (strong mode only) ---
        image_tensor = TF.to_tensor(image)  # [0,1]

        if self.train and self.aug_mode == "strong" and random.random() < 0.5:
            # Small Gaussian noise
            noise_std = random.uniform(0.01, 0.04)
            noise = torch.randn_like(image_tensor) * noise_std
            image_tensor = (image_tensor + noise).clamp(0.0, 1.0)

        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor


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
        if ann.segmentation:
            for seg in ann.segmentation:
                if isinstance(seg, list) and len(seg) >= 6:
                    draw.polygon(seg, fill=255)
        elif ann.bbox:
            x, y, w, h = ann.bbox
            draw.rectangle([x, y, x + w, y + h], fill=255)
    return mask


def make_transforms(
    train: bool,
    size: int = 600,
    aug_mode: str = "light",
    jitter_strength: float = 0.3,
) -> Callable:
    """
    Return a pickle-safe callable that applies paired transforms to image/mask.

    aug_mode:
      - "none"   : resize + pad only
      - "light"  : original augmentations (hflip/vflip/rot + light brightness/contrast)
      - "strong" : stronger geometric + photometric + optional noise
    """
    return PairTransform(
        train=train,
        size=size,
        aug_mode=aug_mode,
        jitter_strength=jitter_strength,
    )


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