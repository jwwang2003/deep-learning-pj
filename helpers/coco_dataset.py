"""COCO segmentation dataset + augmentations"""

from __future__ import annotations

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
        if random.random() < 0.8:
            s = self.jitter_strength  # e.g. 0.3
            b = random.uniform(1.0 - s, 1.0 + s)
            c = random.uniform(1.0 - s, 1.0 + s)
            t = random.uniform(1.0 - s, 1.0 + s)
            h = random.uniform(-0.02, 0.02)

            image = TF.adjust_brightness(image, b)
            image = TF.adjust_contrast(image, c)
            image = TF.adjust_saturation(image, t)
            image = TF.adjust_hue(image, h)

        return image, mask

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- 1) Resize ---
        image = TF.resize(image, (self.size, self.size), interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.size, self.size), interpolation=InterpolationMode.NEAREST)

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
            image, mask = self._pad_to_multiple_of_16(image, mask)

        # --- 3) To tensor + optional noise (strong mode only) ---
        image_tensor = TF.to_tensor(image)  # [0,1]

        if self.train and self.aug_mode == "strong" and random.random() < 0.3:
            # Small Gaussian noise
            noise_std = 0.02
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