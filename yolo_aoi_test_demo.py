#!/usr/bin/env python3
"""
Run inference on the AOI test split using the most recent fine-tuned YOLOv5 weights and
report full metrics using YOLOv5's built-in validation pipeline.

This script scans `runs_aoi_project/*/weights/best.pt`, picks the newest checkpoint, and
launches YOLOv5's `val.py` for statistics plus `detect.py` for qualitative outputs
against every location declared under the `test` key in `data/aoi.yaml`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib

from yolov5.detect import run as yolov5_detect
from yolov5.val import run as yolov5_val
from yolov5.utils.general import yaml_load

# Prefer fonts that support Chinese glyphs so saved plots render labels legibly.
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "PingFang SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_ROOT / "runs_aoi_project"
DATA_CONFIG = PROJECT_ROOT / "data" / "aoi.yaml"
IMG_SIZE = 640
CONF_THRES = 0.25
VAL_BATCH_SIZE = 16


def _ensure_absolute(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()


def _dataset_root(data: dict) -> Path:
    data_path = data.get("path")
    if data_path:
        return _ensure_absolute(Path(data_path), DATA_CONFIG.parent)
    return DATA_CONFIG.parent


def _collect_test_sources() -> List[Path]:
    data = yaml_load(DATA_CONFIG)
    entries: Iterable[str | Path] | None = data.get("test")
    if entries is None:
        print("[WARN] Dataset yaml does not define a 'test' split; nothing to evaluate.")
        return []
    if not isinstance(entries, (list, tuple)):
        entries = [entries]

    dataset_root = _dataset_root(data)
    resolved: List[Path] = []
    for entry in entries:
        if not entry:
            continue
        entry_path = Path(entry)
        if not entry_path.is_absolute():
            entry_path = (dataset_root / entry_path).resolve()
        if entry_path.exists():
            resolved.append(entry_path)
        else:
            print(f"[WARN] Skipping missing test source: {entry_path}")
    if not resolved:
        print("[WARN] Could not resolve any valid test sources.")
    return resolved


def _latest_best_weights() -> Path:
    candidates: List[tuple[float, Path]] = []
    if not RUNS_ROOT.exists():
        raise FileNotFoundError(
            f"Could not find any training runs under {RUNS_ROOT}. Run training first."
        )
    for run_dir in RUNS_ROOT.iterdir():
        if not run_dir.is_dir():
            continue
        weights_path = run_dir / "weights" / "best.pt"
        if weights_path.exists():
            candidates.append((weights_path.stat().st_mtime, weights_path))
    if not candidates:
        raise FileNotFoundError(
            f"No best.pt checkpoints were found in {RUNS_ROOT}. Launch fine-tuning first."
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _run_test_metrics(weights: Path, run_label: str) -> None:
    metrics_project = RUNS_ROOT / "test_metrics"
    metrics_project.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Evaluating test metrics for '{run_label}' using {weights}")
    results, maps, _ = yolov5_val(
        data=str(DATA_CONFIG),
        weights=str(weights),
        task="test",
        imgsz=IMG_SIZE,
        batch_size=VAL_BATCH_SIZE,
        project=str(metrics_project),
        name=run_label,
        exist_ok=True,
        device="",
        verbose=True,
        plots=True,
    )
    if results:
        mp, mr, map50, map = results[:4]
        print(
            "[INFO] Test metrics -> "
            f"Precision: {mp:.4f}, Recall: {mr:.4f}, mAP50: {map50:.4f}, mAP50-95: {map:.4f}"
        )


def main() -> None:
    try:
        weights = _latest_best_weights()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return

    run_label = weights.parents[1].name  # e.g. yolov5s-aoi-fourcls2
    _run_test_metrics(weights, run_label)

    sources = _collect_test_sources()
    if not sources:
        return

    save_project = RUNS_ROOT / "test_demos"
    save_project.mkdir(parents=True, exist_ok=True)

    for source in sources:
        split_name = source.parent.name if source.parent != source else source.name
        save_name = f"{run_label}-{split_name}"
        print(
            f"[INFO] Running detection on '{source}' "
            f"using weights '{weights}' (results -> {save_name})."
        )
        yolov5_detect(
            weights=str(weights),
            source=str(source),
            data=str(DATA_CONFIG),
            imgsz=(IMG_SIZE, IMG_SIZE),
            conf_thres=CONF_THRES,
            device="",
            project=str(save_project),
            name=save_name,
            exist_ok=True,
        )
    print("[INFO] Test demos completed.")


if __name__ == "__main__":
    main()
