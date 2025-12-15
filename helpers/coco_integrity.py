from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

from PIL import Image


@dataclass
class CocoIntegrityReport:
    """Summary of the consistency checks between COCO json entries and image files on disk."""

    coco_path: Path
    images_dir: Path
    total_entries: int
    files_on_disk: int
    missing_files: List[str]
    orphan_files: List[str]
    duplicate_filenames: List[str]
    updated_dimensions: List[str]
    missing_dimensions: List[str]
    images_without_annotations: List[str]

    @property
    def ok(self) -> bool:
        return (
            not self.missing_files
            and not self.duplicate_filenames
            and not self.missing_dimensions
            and not self.images_without_annotations
        )


def check_coco_integrity(
    coco_json: Path | str,
    images_dir: Optional[Path | str] = None,
    *,
    recursive: bool = True,
) -> CocoIntegrityReport:
    """
    Compares COCO images[*].file_name entries (S3 URIs or regular paths) with the files present in images_dir.
    Only the basename (e.g., foo.jpg) has to match between the manifest and the directory contents.
    """
    coco_path = Path(coco_json)
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO json not found: {coco_path}")

    data = _load_coco_json(coco_path)
    images = _validate_images_list(data, coco_path)
    annotations = _validate_annotations_list(data, coco_path)
    img_dir = Path(images_dir) if images_dir else coco_path.parent / "images"
    if not img_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")
    if not img_dir.is_dir():
        raise NotADirectoryError(f"Images directory is not a directory: {img_dir}")

    updated_dimensions, missing_dimensions = _ensure_dimensions(images, img_dir)
    if updated_dimensions:
        _write_coco_json(coco_path, data)

    json_names: List[str] = []
    for entry in images:
        file_name = entry.get("file_name")
        if isinstance(file_name, str) and file_name:
            json_names.append(_basename_from_any_path(file_name))
    duplicate_filenames = _find_duplicates(json_names)

    disk_files = _collect_file_basenames(img_dir, recursive=recursive)

    json_set = set(json_names)
    missing = sorted(name for name in json_set if name not in disk_files)
    orphan_files = sorted(name for name in disk_files if name not in json_set)
    images_without_annotations = _find_images_without_annotations(images, annotations)

    return CocoIntegrityReport(
        coco_path=coco_path,
        images_dir=img_dir,
        total_entries=len(json_names),
        files_on_disk=len(disk_files),
        missing_files=missing,
        orphan_files=orphan_files,
        duplicate_filenames=duplicate_filenames,
        updated_dimensions=updated_dimensions,
        missing_dimensions=missing_dimensions,
        images_without_annotations=images_without_annotations,
    )


def print_report(report: CocoIntegrityReport, *, limit: int = 10) -> None:
    """Pretty-print the result of check_coco_integrity."""
    status = "PASS" if report.ok else "FAIL"
    print(f"[{status}] COCO: {report.coco_path}")
    print(f"  images dir  : {report.images_dir}")
    print(f"  manifest img: {report.total_entries}")
    print(f"  files on disk: {report.files_on_disk}")

    _print_section("missing files", report.missing_files, limit=limit)
    _print_section("duplicate entries", report.duplicate_filenames, limit=limit)
    _print_section("unreferenced files", report.orphan_files, limit=limit)
    _print_section("dimensions updated", report.updated_dimensions, limit=limit)
    _print_section("dimensions missing", report.missing_dimensions, limit=limit)
    _print_section("images without annotations", report.images_without_annotations, limit=limit)


def _collect_file_basenames(root: Path, *, recursive: bool) -> Set[str]:
    if recursive:
        iterator: Iterable[Path] = root.rglob("*")
    else:
        iterator = root.iterdir()

    names: Set[str] = set()
    for path in iterator:
        if path.is_file():
            names.add(path.name)
    return names


def _basename_from_any_path(value: str) -> str:
    """
    Accepts windows paths, unix paths, and s3://bucket/key entries.
    Only the filename is relevant for integrity checking because the on-disk download
    stores images without the S3 folder structure.
    """
    normalized = value.replace("\\", "/")
    return Path(normalized).name


def _find_duplicates(names: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    dups: Set[str] = set()
    for name in names:
        if name in seen:
            dups.add(name)
        else:
            seen.add(name)
    return sorted(dups)


def _print_section(label: str, items: Sequence[str], *, limit: Optional[int]) -> None:
    if not items:
        print(f"  {label}: none")
        return

    print(f"  {label} ({len(items)}):")
    if limit is None or limit <= 0:
        subset = list(items)
    else:
        subset = list(items)[:limit]
    for entry in subset:
        print(f"    - {entry}")
    if limit is not None and limit > 0:
        remaining = len(items) - limit
        if remaining > 0:
            print(f"    ... {remaining} more")


def _load_coco_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_images_list(data: dict, path: Path) -> List[dict]:
    images = data.get("images", [])
    if not isinstance(images, list):
        raise ValueError(f"Invalid COCO json: 'images' is not a list in {path}")
    return images


def _validate_annotations_list(data: dict, path: Path) -> List[dict]:
    annotations = data.get("annotations", [])
    if not isinstance(annotations, list):
        raise ValueError(f"Invalid COCO json: 'annotations' is not a list in {path}")
    return annotations


def _has_valid_dimensions(entry: dict) -> bool:
    width = entry.get("width")
    height = entry.get("height")
    try:
        w_ok = isinstance(width, (int, float)) and width > 0
        h_ok = isinstance(height, (int, float)) and height > 0
    except TypeError:
        return False
    return w_ok and h_ok


def _ensure_dimensions(images: List[dict], images_dir: Path) -> tuple[List[str], List[str]]:
    updated: List[str] = []
    missing: List[str] = []
    for entry in images:
        if not isinstance(entry, dict):
            continue
        file_name = entry.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            continue
        if _has_valid_dimensions(entry):
            continue

        basename = _basename_from_any_path(file_name)
        src = images_dir / basename
        if not src.exists():
            missing.append(basename)
            continue

        try:
            with Image.open(src) as img:
                width, height = img.size
        except Exception:
            missing.append(basename)
            continue

        entry["width"] = width
        entry["height"] = height
        updated.append(basename)
    return sorted(updated), sorted(set(missing))


def _find_images_without_annotations(images: List[dict], annotations: List[dict]) -> List[str]:
    counts = {}
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        image_id = ann.get("image_id")
        if image_id is None:
            continue
        counts[image_id] = counts.get(image_id, 0) + 1

    missing: List[str] = []
    for entry in images:
        if not isinstance(entry, dict):
            continue
        image_id = entry.get("id")
        if image_id in counts and counts[image_id] > 0:
            continue
        file_name = entry.get("file_name")
        basename = _basename_from_any_path(file_name) if isinstance(file_name, str) else "unknown-file"
        missing.append(f"{image_id}::{basename}")
    return sorted(missing)


def _write_coco_json(path: Path, data: dict) -> None:
    text = json.dumps(data, ensure_ascii=True, indent=2)
    text = text.replace("/", "\\/")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that COCO result.json entries line up with downloaded images.")
    parser.add_argument("coco_json", type=str, help="Path to result.json (COCO annotations).")
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Directory that contains the downloaded images (defaults to <coco_json>/../images).",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only inspect files directly inside images-dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="How many entries to print per issue category (0 = no limit).",
    )
    args = parser.parse_args()

    report = check_coco_integrity(args.coco_json, args.images_dir, recursive=not args.non_recursive)
    print_report(report, limit=args.limit)

    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
