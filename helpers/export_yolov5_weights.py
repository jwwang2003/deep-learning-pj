"""
Utility to strip YOLOv5 checkpoints down to lightweight CPU/GPU inference weights.

Given a `best.pt` (or any YOLOv5 training checkpoint), this helper emits two
artifacts:

1. CPU-friendly FP32 weights (`*_cpu_fp32.pt`) that keep the model object but
   drop optimizer/EMA state for lean inference on machines without CUDA.
2. GPU-oriented FP16 weights (`*_gpu_fp16.pt`) that mirror the same structure
   but store the parameters in half precision for faster transfers and a smaller
   footprint.

Example:
    python -m helpers.export_yolov5_weights \
        --checkpoint runs_aoi_project/yolov5s-aoi-fourcls/weights/best.pt \
        --output-dir exports/yolov5s-aoi-fourcls
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Any, Dict

import torch


def _default_run_name(checkpoint: Path) -> str:
    try:
        return checkpoint.parents[1].name
    except IndexError:
        return checkpoint.stem


def _clone_model(model: torch.nn.Module) -> torch.nn.Module:
    clone = copy.deepcopy(model).cpu().eval()
    for param in clone.parameters():
        param.requires_grad_(False)
    return clone


def _save_variant(
    model: torch.nn.Module,
    out_path: Path,
    precision: str,
    metadata: Dict[str, Any],
) -> None:
    variant = _clone_model(model)
    if precision == "fp16":
        variant = variant.half()
    else:
        variant = variant.float()

    payload = {
        "model": variant,
        "optimizer": None,
        "ema": None,
        "updates": None,
        "best_fitness": None,
        "epoch": -1,
        "meta": {**metadata, "precision": precision},
    }
    torch.save(payload, out_path)
    print(f"[ok] wrote {precision} weights -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv5 checkpoints into CPU/GPU inference weights.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to YOLOv5 training checkpoint (best.pt)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination directory for exported weights")
    parser.add_argument("--run-name", type=str, default=None, help="Optional prefix for exported filenames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    # Allow checkpoints pickled on Windows (Path objects) to load on POSIX.
    import os

    if os.name != "nt":
        try:
            import pathlib

            class _WindowsPathAsPosix(pathlib.PurePosixPath):
                def __new__(cls, *args, **kwargs):
                    return pathlib.PosixPath(*args, **kwargs)

            pathlib.WindowsPath = _WindowsPathAsPosix  # type: ignore[attr-defined]
        except (ImportError, AttributeError):
            pass

    y5_root = Path(__file__).resolve().parents[1] / "yolov5"
    if y5_root.is_dir():
        path_str = str(y5_root)
        if path_str not in sys.path:
            sys.path.append(path_str)

    # YOLOv5 checkpoints pickle DetectionModel objects, so disable the safety-only
    # loading path (PyTorch 2.6+ defaults to weights_only=True).
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in raw:
        raise ValueError(f"Invalid YOLOv5 checkpoint: missing 'model' key in {args.checkpoint}")

    base_model = (raw.get("ema") or raw["model"]).cpu().float().eval()
    run_name = args.run_name or _default_run_name(args.checkpoint)
    out_dir = args.output_dir or (Path("exports") / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source": str(args.checkpoint),
        "run_name": run_name,
        "names": list(base_model.names) if isinstance(base_model.names, (list, tuple)) else base_model.names,
        "nc": getattr(base_model, "nc", None),
        "stride": int(max(getattr(base_model, "stride", [32]))),
    }

    cpu_path = out_dir / f"{run_name}_cpu_fp32.pt"
    gpu_path = out_dir / f"{run_name}_gpu_fp16.pt"

    _save_variant(base_model, cpu_path, "fp32", metadata)
    _save_variant(base_model, gpu_path, "fp16", metadata)


if __name__ == "__main__":
    main()
