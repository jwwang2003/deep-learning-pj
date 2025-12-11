from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import sys


@dataclass
class COCOImage:
    id: int
    file_name: str
    width: float
    height: float
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "COCOImage":
        known = {
            "id": data["id"],
            "file_name": data.get("file_name", ""),
            "width": data.get("width", 0),
            "height": data.get("height", 0),
        }
        extras = {k: v for k, v in data.items() if k not in known}
        return cls(**known, extras=extras)


@dataclass
class COCOCategory:
    id: int
    name: str
    supercategory: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "COCOCategory":
        known = {
            "id": data["id"],
            "name": data.get("name", ""),
            "supercategory": data.get("supercategory"),
        }
        extras = {k: v for k, v in data.items() if k not in known}
        return cls(**known, extras=extras)


@dataclass
class COCOAnnotation:
    id: int
    image_id: int
    category_id: int
    bbox: List[float]
    segmentation: List[Any] = field(default_factory=list)
    area: Optional[float] = None
    iscrowd: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "COCOAnnotation":
        known = {
            "id": data["id"],
            "image_id": data["image_id"],
            "category_id": data.get("category_id", -1),
            "bbox": data.get("bbox", []),
            "segmentation": data.get("segmentation", []),
            "area": data.get("area"),
            "iscrowd": data.get("iscrowd", 0),
        }
        extras = {k: v for k, v in data.items() if k not in known}
        return cls(**known, extras=extras)


class COCO:
    def __init__(
        self,
        images: Iterable[COCOImage],
        categories: Iterable[COCOCategory],
        annotations: Iterable[COCOAnnotation],
        info: Optional[Dict[str, Any]] = None,
        licenses: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.images: List[COCOImage] = list(images)
        self.categories: List[COCOCategory] = list(categories)
        self.annotations: List[COCOAnnotation] = list(annotations)
        self.info = info or {}
        self.licenses = licenses or []

        self._image_index: Dict[int, COCOImage] = {img.id: img for img in self.images}
        self._category_index: Dict[int, COCOCategory] = {
            cat.id: cat for cat in self.categories
        }
        self._annotations_by_image: Dict[int, List[COCOAnnotation]] = {}
        for ann in self.annotations:
            self._annotations_by_image.setdefault(ann.image_id, []).append(ann)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "COCO":
        images = [COCOImage.from_dict(img) for img in data.get("images", [])]
        categories = [COCOCategory.from_dict(cat) for cat in data.get("categories", [])]
        annotations = [
            COCOAnnotation.from_dict(ann) for ann in data.get("annotations", [])
        ]
        return cls(
            images=images,
            categories=categories,
            annotations=annotations,
            info=data.get("info"),
            licenses=data.get("licenses"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "COCO":
        with Path(path).open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        return cls.from_dict(data)

    def get_image(self, image_id: int) -> Optional[COCOImage]:
        return self._image_index.get(image_id)

    def get_category(self, category_id: int) -> Optional[COCOCategory]:
        return self._category_index.get(category_id)

    def get_annotations(self, image_id: int) -> List[COCOAnnotation]:
        return self._annotations_by_image.get(image_id, [])

    def summary(self) -> Dict[str, Any]:
        return {
            "images": len(self.images),
            "annotations": len(self.annotations),
            "categories": len(self.categories),
            "info": self.info.get("description", ""),
        }


def _demo(path: str | Path) -> None:
    _configure_stdio_utf8()
    coco = COCO.from_file(path)
    _safe_print(f"COCO summary: {coco.summary()}")
    sample = coco.images[:3]
    for img in sample:
        anns = coco.get_annotations(img.id)
        _safe_print(f"- image_id={img.id} file={img.file_name} anns={len(anns)}")


def _configure_stdio_utf8() -> None:
    # Try to make stdout/stderr UTF-8 so Chinese characters display correctly on Windows consoles.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _safe_print(text: Any) -> None:
    message = str(text)
    try:
        print(message)
    except UnicodeEncodeError:
        # Last-resort: write bytes directly so Unicode is not dropped.
        sys.stdout.buffer.write((message + "\n").encode("utf-8", errors="replace"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse a COCO JSON file.")
    parser.add_argument("path", type=str, help="Path to a COCO annotation json file.")
    args = parser.parse_args()
    _demo(args.path)
