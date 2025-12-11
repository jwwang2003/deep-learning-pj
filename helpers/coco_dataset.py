"""COCO segmentation dataset + augmentations for 600x600 training."""

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
    """Pickle-safe paired transform for image/mask."""

    def __init__(self, train: bool, size: int = 600) -> None:
        self.train = train
        self.size = size

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resize to the requested size
        image_resized = TF.resize(
            image, (self.size, self.size), interpolation=InterpolationMode.BILINEAR
        )
        mask_resized = TF.resize(mask, (self.size, self.size), interpolation=InterpolationMode.NEAREST)

        # Pad up to the next multiple of 16 so UNet skip connections align after pooling/upsampling.
        pad_w = (16 - (image_resized.width % 16)) % 16
        pad_h = (16 - (image_resized.height % 16)) % 16
        if pad_w or pad_h:
            padding = (0, 0, pad_w, pad_h)  # left, top, right, bottom
            image_resized = TF.pad(image_resized, padding, fill=0)
            mask_resized = TF.pad(mask_resized, padding, fill=0)

        if self.train:
            if random.random() < 0.5:
                image_resized = TF.hflip(image_resized)
                mask_resized = TF.hflip(mask_resized)
            if random.random() < 0.2:
                image_resized = TF.vflip(image_resized)
                mask_resized = TF.vflip(mask_resized)
            if random.random() < 0.3:
                angle = random.uniform(-10, 10)
                image_resized = TF.rotate(
                    image_resized, angle, interpolation=InterpolationMode.BILINEAR
                )
                mask_resized = TF.rotate(mask_resized, angle, interpolation=InterpolationMode.NEAREST)
            if random.random() < 0.4:
                image_resized = TF.adjust_brightness(
                    image_resized, brightness_factor=random.uniform(0.8, 1.2)
                )
                image_resized = TF.adjust_contrast(
                    image_resized, contrast_factor=random.uniform(0.8, 1.2)
                )

        image_tensor = TF.to_tensor(image_resized)
        mask_tensor = torch.from_numpy(np.array(mask_resized, dtype=np.float32) / 255.0).unsqueeze(0)
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


def make_transforms(train: bool, size: int = 600) -> Callable:
    """Return a pickle-safe callable that applies paired transforms to image/mask."""
    return PairTransform(train=train, size=size)


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
