#!/usr/bin/env python3
"""
Template entry point for fine-tuning YOLOv5 on the project1 dataset.

How to use this template
------------------------
1. Make sure your dataset yaml only lists the four classes you want to detect and that
   every label file only contains IDs in [0, 3] following the order below:
       0 -> 可接受
       1 -> 墨点
       2 -> 崩边
       3 -> 沾污
   (Regenerate labels or filter out unwanted categories before training.)

2. Pick the model config/weights you want to fine-tune (yolov5s/yolov5m/…).

3. Adjust `TRAINING_PARAMS` to match your hardware (batch size, epochs, etc.) and run:
       python yolo_aoi_fine_tune.py
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, Any, Iterable, List

from yolov5.detect import run as yolov5_detect
from yolov5.train import run as yolov5_train
from yolov5.utils.general import yaml_load

PROJECT_ROOT = Path(__file__).resolve().parent
Y5_ROOT = PROJECT_ROOT / "yolov5"
DATA_CONFIG = PROJECT_ROOT / "data" / "aoi.yaml"
MODEL_CFG = Y5_ROOT / "models" / "yolov5s.yaml"
PRETRAINED_WEIGHTS = Y5_ROOT / "yolov5s.pt"  # swap with custom checkpoint if desired
HYPERPARAMS = Y5_ROOT / "data" / "hyps" / "hyp.scratch-low.yaml"

TARGET_CLASS_NAMES = [
    "Other defect",
    "Acceptable",
    "Ink spot",
    "Edge chip",
    "Stain",
]


DEMO_SAMPLE_COUNT = 4
DEMO_SPLIT = "val"
DEMO_CONF_THRES = 0.35
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Central place to tweak common training settings.
TRAINING_PARAMS: Dict[str, Any] = {
    "epochs": 150,
    "batch_size": 16,
    "imgsz": 640,
    "optimizer": "SGD",
    "project": str(PROJECT_ROOT / "runs_aoi_project"),
    "name": "yolov5s-aoi-fourcls",
    "device": "",  # e.g. "0" or "0,1" or "cpu"
    "workers": 8,
    "cos_lr": False,
    "cache": "ram",  # switch to "disk" or None depending on memory
    "freeze": [0],  # freeze first layer group; extend to freeze more of the backbone
}


def _verify_data_config() -> None:
    """Optional guard that reminds the user when the yaml does not match the requested classes."""
    data = yaml_load(DATA_CONFIG)
    yaml_names = data.get("names")
    if isinstance(yaml_names, dict):
        yaml_names = [name for _, name in sorted(yaml_names.items())]

    missing = [name for name in TARGET_CLASS_NAMES if name not in (yaml_names or [])]
    if missing:
        print(
            "[WARN] The dataset yaml does not list the following target classes: "
            + ", ".join(missing)
        )
        print("       Make sure you updated your yaml/labels before launching training.\n")


def _ensure_absolute(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()


def _load_yaml() -> Dict[str, Any]:
    return yaml_load(DATA_CONFIG)


def _dataset_root(data: Dict[str, Any]) -> Path:
    data_path = data.get("path")
    if data_path:
        return _ensure_absolute(Path(data_path), DATA_CONFIG.parent)
    return DATA_CONFIG.parent


def _paths_from_txt(txt_file: Path) -> List[Path]:
    images: List[Path] = []
    for line in txt_file.read_text().splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = (txt_file.parent / path).resolve()
        if path.exists() and path.suffix.lower() in IMG_EXTENSIONS:
            images.append(path)
    return images


def _expand_split_entry(entry: str | Path, dataset_root: Path) -> List[Path]:
    entry_path = Path(entry)
    if not entry_path.is_absolute():
        entry_path = (dataset_root / entry_path).resolve()

    if entry_path.is_dir():
        return [p for p in entry_path.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS]
    if entry_path.suffix.lower() == ".txt" and entry_path.exists():
        return _paths_from_txt(entry_path)
    if entry_path.exists() and entry_path.suffix.lower() in IMG_EXTENSIONS:
        return [entry_path]
    # Allow glob-like patterns
    parent = entry_path.parent
    if parent.exists():
        return [
            p
            for p in parent.glob(entry_path.name)
            if p.suffix.lower() in IMG_EXTENSIONS and p.exists()
        ]
    return []


def _collect_split_images(split: str) -> List[Path]:
    data = _load_yaml()
    split_value = data.get(split)
    if split_value is None:
        print(f"[WARN] Dataset yaml does not define a '{split}' split – skipping demos.")
        return []

    dataset_root = _dataset_root(data)
    entries: Iterable[str | Path]
    if isinstance(split_value, (list, tuple)):
        entries = split_value
    else:
        entries = [split_value]

    images: List[Path] = []
    for entry in entries:
        images.extend(_expand_split_entry(entry, dataset_root))
    unique_images = []
    seen = set()
    for img in images:
        resolved = img.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_images.append(resolved)
    if not unique_images:
        print(
            f"[WARN] Could not find any images for split '{split}'. "
            "Double-check your dataset paths."
        )
    return unique_images


def _sample_demo_images() -> List[Path]:
    candidates = _collect_split_images(DEMO_SPLIT)
    if not candidates:
        return []
    random.shuffle(candidates)
    sample_count = min(DEMO_SAMPLE_COUNT, len(candidates))
    selected = candidates[:sample_count]
    print(
        f"[INFO] Selected {len(selected)} demo image(s) from the '{DEMO_SPLIT}' split "
        f"(out of {len(candidates)} candidates)."
    )
    return selected


def _prepare_demo_source_dir(save_dir: Path, samples: List[Path]) -> Path:
    demo_source = save_dir / "demo_samples"
    if demo_source.exists():
        shutil.rmtree(demo_source)
    demo_source.mkdir(parents=True, exist_ok=True)
    for idx, img_path in enumerate(samples, start=1):
        target_name = f"{idx:02d}_{img_path.name}"
        shutil.copy(img_path, demo_source / target_name)
    return demo_source


def _run_demo_inference(weights_path: Path, samples: List[Path], save_dir: Path) -> None:
    if not samples:
        print("[INFO] Skipping demo inference because no sample images were selected.")
        return
    if not weights_path.exists():
        print(
            f"[WARN] Could not find trained weights at {weights_path}. "
            "Skipping demo inference."
        )
        return

    demo_source = _prepare_demo_source_dir(save_dir, samples)
    demo_output_name = "demo"
    yolov5_detect(
        weights=str(weights_path),
        source=str(demo_source),
        data=str(DATA_CONFIG),
        imgsz=(TRAINING_PARAMS["imgsz"], TRAINING_PARAMS["imgsz"]),
        conf_thres=DEMO_CONF_THRES,
        device=TRAINING_PARAMS.get("device", ""),
        project=str(save_dir),
        name=demo_output_name,
        exist_ok=True,
    )
    print(
        "[INFO] Demo predictions saved to "
        f"{save_dir / demo_output_name} (images) and {demo_source} (inputs)."
    )


def main() -> None:
    _verify_data_config()
    run_args = {
        "data": str(DATA_CONFIG),
        "cfg": str(MODEL_CFG),
        "weights": str(PRETRAINED_WEIGHTS),
        "hyp": str(HYPERPARAMS),
        "single_cls": False,
        "rect": False,
        "label_smoothing": 0.0,
        **TRAINING_PARAMS,
    }
    train_opt = yolov5_train(**run_args)

    if not train_opt or not getattr(train_opt, "save_dir", None):
        print("[WARN] Could not determine training save directory; skipping demos.")
        return

    save_dir = Path(train_opt.save_dir)
    weights_path = save_dir / "weights" / "best.pt"
    samples = _sample_demo_images()
    _run_demo_inference(weights_path, samples, save_dir)


if __name__ == "__main__":
    main()
