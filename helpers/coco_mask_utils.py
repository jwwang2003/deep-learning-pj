from __future__ import annotations

from typing import Iterable, List, Sequence, TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

try:
    from pycocotools import mask as coco_mask  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    coco_mask = None

if TYPE_CHECKING:  # pragma: no cover
    from .coco import COCOAnnotation


def add_annotation_to_mask(mask: Image.Image, ann: "COCOAnnotation", *, fill: int = 255) -> bool:
    """
    Rasterize a COCO annotation's segmentation into the provided mask image.
    Returns True if the segmentation was drawn, False if nothing was added.
    """
    segmentation = ann.segmentation
    if not segmentation:
        return False

    if isinstance(segmentation, list):
        draw = ImageDraw.Draw(mask)
        drawn = False
        for seg in segmentation:
            coords = _flatten_polygon(seg)
            if coords:
                draw.polygon(coords, fill=fill)
                drawn = True
        return drawn

    if isinstance(segmentation, dict):
        width, height = mask.size
        seg_mask = _decode_rle_mask(segmentation, height=int(height), width=int(width), fill=fill)
        if seg_mask is None:
            return False
        mask.paste(fill, mask=seg_mask)
        return True

    # Some exporters wrap a single polygon as a tuple instead of list
    coords = _flatten_polygon(segmentation)
    if coords:
        draw = ImageDraw.Draw(mask)
        draw.polygon(coords, fill=fill)
        return True

    return False


def _flatten_polygon(poly: Sequence[float] | Iterable[float]) -> List[float] | None:
    if isinstance(poly, (list, tuple)):
        pts = list(poly)
        if len(pts) >= 6:
            return [float(v) for v in pts]
    return None


def _decode_rle_mask(
    rle: dict,
    *,
    height: int,
    width: int,
    fill: int = 255,
) -> Image.Image | None:
    counts = rle.get("counts")
    if counts is None:
        return None

    if isinstance(counts, list):
        arr = _decode_uncompressed_counts(counts, height=height, width=width, fill=fill)
        return Image.fromarray(arr, mode="L")

    if isinstance(counts, str):
        if coco_mask is None:
            raise RuntimeError(
                "Compressed RLE requires pycocotools. Install it with 'pip install pycocotools' to decode segmentations."
            )
        decoded = coco_mask.decode(rle)
        if decoded.ndim == 3:  # pycocotools returns (H,W,1)
            decoded = decoded[:, :, 0]
        return Image.fromarray((decoded.astype(np.uint8)) * fill, mode="L")

    return None


def _decode_uncompressed_counts(
    counts: Sequence[int],
    *,
    height: int,
    width: int,
    fill: int,
) -> np.ndarray:
    total = height * width
    flat = np.zeros(total, dtype=np.uint8)
    idx = 0
    value = 0
    for run in counts:
        r = int(run)
        if r < 0:
            raise ValueError("RLE run-length entries must be non-negative.")
        if idx + r > total:
            r = total - idx
        if value == 1:
            flat[idx : idx + r] = fill
        idx += r
        value ^= 1
        if idx >= total:
            break
    return flat.reshape((height, width), order="F")
