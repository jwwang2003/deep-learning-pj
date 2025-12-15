"""Reusable paired image/mask augmentations shared across training pipelines."""

from __future__ import annotations

import random
from typing import Callable, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


class PairTransform:
    """
    Pickle-safe paired transform for image/mask augmentations.

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
        Legacy augmentations used for the early UNet experiments.
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
        Stronger augmentations: shared across UNet, YOLO notebooks, etc.
        """
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

        if random.random() < 0.9:
            s = self.jitter_strength
            image = TF.adjust_brightness(image, random.uniform(1.0 - s, 1.0 + s))
            image = TF.adjust_contrast(image, random.uniform(1.0 - s, 1.0 + s))
            image = TF.adjust_saturation(image, random.uniform(1.0 - s, 1.0 + s))

            if random.random() < 0.7:
                gamma = random.uniform(0.6, 1.6)
                image = TF.adjust_gamma(image, gamma=gamma, gain=1.0)

            if random.random() < 0.5:
                image = TF.adjust_hue(image, random.uniform(-0.06, 0.06))

            if random.random() < 0.15:
                image = TF.rgb_to_grayscale(image, num_output_channels=3)

            if random.random() < 0.25:
                image = TF.autocontrast(image)
            if random.random() < 0.15:
                image = TF.equalize(image)

            if random.random() < 0.20:
                sigma = random.uniform(0.1, 1.2)
                image = TF.gaussian_blur(image, kernel_size=[3, 3], sigma=sigma)

        return image, mask

    def _resize_letterbox(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Resize longest side to self.size while preserving aspect ratio, then pad to (size,size).
        """
        w, h = image.size
        scale = self.size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        image = TF.resize(image, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (new_h, new_w), interpolation=InterpolationMode.NEAREST)

        pad_w = self.size - new_w
        pad_h = self.size - new_h
        image = TF.pad(image, (0, 0, pad_w, pad_h), fill=0)
        mask = TF.pad(mask, (0, 0, pad_w, pad_h), fill=0)
        return image, mask

    def _random_resized_crop_pair(
        self,
        image: Image.Image,
        mask: Image.Image,
        scale: Tuple[float, float] = (0.7, 1.0),
        ratio: Tuple[float, float] = (0.9, 1.1),
        p: float = 0.6,
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() > p:
            return image, mask
        crop_i, crop_j, crop_h, crop_w = RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
        image = TF.resized_crop(
            image,
            crop_i,
            crop_j,
            crop_h,
            crop_w,
            (self.size, self.size),
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = TF.resized_crop(
            mask,
            crop_i,
            crop_j,
            crop_h,
            crop_w,
            (self.size, self.size),
            interpolation=InterpolationMode.NEAREST,
        )
        return image, mask

    def transform_pil(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        image, mask = self._resize_letterbox(image, mask)

        if not self.train:
            image, mask = self._pad_to_multiple_of_16(image, mask)
        elif self.aug_mode == "none":
            image, mask = self._pad_to_multiple_of_16(image, mask)
        elif self.aug_mode == "light":
            image, mask = self._apply_light_aug(image, mask)
            image, mask = self._pad_to_multiple_of_16(image, mask)
        elif self.aug_mode == "strong":
            image, mask = self._apply_strong_aug(image, mask)
            image, mask = self._random_resized_crop_pair(
                image, mask, scale=(0.65, 1.0), ratio=(0.85, 1.15), p=0.7
            )
            image, mask = self._pad_to_multiple_of_16(image, mask)

        return image, mask

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.transform_pil(image, mask)
        image_tensor = TF.to_tensor(image)

        if self.train and self.aug_mode == "strong" and random.random() < 0.5:
            noise_std = random.uniform(0.01, 0.04)
            noise = torch.randn_like(image_tensor) * noise_std
            image_tensor = (image_tensor + noise).clamp(0.0, 1.0)

        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor


def make_transforms(
    train: bool,
    size: int = 600,
    aug_mode: str = "light",
    jitter_strength: float = 0.3,
) -> Callable:
    """
    Return a pickle-safe callable that applies paired transforms to image/mask.
    """
    return PairTransform(
        train=train,
        size=size,
        aug_mode=aug_mode,
        jitter_strength=jitter_strength,
    )


__all__ = ["PairTransform", "make_transforms"]
