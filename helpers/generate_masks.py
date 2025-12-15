from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw

from .coco import COCO, COCOAnnotation, COCOImage
from .coco_mask_utils import add_annotation_to_mask


def build_masks(
    coco_json: Path,
    images_dir: Path | None = None,
    masks_dir: Path | None = None,
    limit: int | None = None,
    overlay: bool = False,
    overlay_dir: Path | None = None,
    overlay_alpha: float = 0.5,
) -> None:
    """
    Generate binary masks (mode 'L') for images referenced in a COCO json.
    Prioritizes polygon segmentations; falls back to bounding boxes.
    Optionally generate translucent overlays for alignment checks.
    """
    coco = COCO.from_file(coco_json)
    images_dir = images_dir or coco_json.parent / "images"
    masks_dir = masks_dir or coco_json.parent / "mask"
    masks_dir.mkdir(parents=True, exist_ok=True)
    if overlay:
        overlay_dir = overlay_dir or coco_json.parent / "mask_overlay"
        overlay_dir.mkdir(parents=True, exist_ok=True)

    images: Iterable[COCOImage] = coco.images
    if limit is not None:
        images = list(coco.images)[:limit]

    for img in images:
        src = images_dir / Path(img.file_name).name
        if not src.exists():
            print(f"[skip] source image missing: {src}")
            continue

        anns: List[COCOAnnotation] = coco.get_annotations(img.id)
        mask = Image.new("L", (int(img.width), int(img.height)), 0)
        draw = ImageDraw.Draw(mask)

        for ann in anns:
            if add_annotation_to_mask(mask, ann):
                continue
            if ann.bbox:
                x, y, w, h = ann.bbox
                draw.rectangle([x, y, x + w, y + h], fill=255)

        dest = masks_dir / f"{src.stem}_mask.png"
        mask.save(dest)
        print(f"[ok] {dest}")

        if overlay:
            overlay_out = overlay_dir / f"{src.stem}_overlay.png"
            base = Image.open(src).convert("RGBA")
            alpha_mask = mask.point(
                lambda p: int(p * max(0.0, min(overlay_alpha, 1.0)))
            )
            red_overlay = Image.new("RGBA", mask.size, (255, 0, 0, 0))
            red_overlay.putalpha(alpha_mask)
            composite = Image.alpha_composite(base, red_overlay)
            composite.save(overlay_out)
            print(f"[ok] overlay {overlay_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate masks from a COCO json and corresponding images."
    )
    parser.add_argument("coco_json", type=str, help="Path to COCO json file.")
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Directory containing the images (default: alongside COCO json / images).",
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        default=None,
        help="Directory to write masks (default: alongside COCO json / mask).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Number of images to process (default: 4 for demo).",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Also save a translucent overlay (mask on top of the image) for alignment checks.",
    )
    parser.add_argument(
        "--overlay-dir",
        type=str,
        default=None,
        help="Directory to write overlays (default: alongside COCO json / mask_overlay).",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.5,
        help="Opacity factor for overlay (0.0-1.0, default 0.5).",
    )
    args = parser.parse_args()

    coco_path = Path(args.coco_json)
    images_dir = Path(args.images_dir) if args.images_dir else None
    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None

    build_masks(
        coco_json=coco_path,
        images_dir=images_dir,
        masks_dir=masks_dir,
        limit=args.limit,
        overlay=args.overlay,
        overlay_dir=overlay_dir,
        overlay_alpha=args.overlay_alpha,
    )


if __name__ == "__main__":
    main()
