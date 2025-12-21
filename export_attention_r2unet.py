"""
Export final Attention R2UNet inference artifacts (weights + TorchScript).

The script expects a PyTorch checkpoint (ideally the final/best training
checkpoint) and produces:

1. FP32 and FP16 "weights-only" checkpoints that strip optimizer/training state.
2. A CPU TorchScript bundle (fp32).
3. An INT8-quantized CPU TorchScript bundle (experimental, uses FX quantization).
4. A GPU-friendly TorchScript bundle traced in FP32 (if CUDA is present).
"""
from __future__ import annotations


import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from unet import AttentionR2UNet

import sys
print("[debug] python =", sys.executable)
print("[debug] torch  =", torch.__version__)


def _read_config(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    with path.open("r") as handle:
        return json.load(handle)


def _resolve(value, *fallbacks):
    for candidate in (value, *fallbacks):
        if candidate is None:
            continue
        return candidate
    return None


def _save_weights(state_dict: Mapping[str, torch.Tensor], precision: str, out_path: Path, metadata: dict) -> None:
    def convert(tensor: torch.Tensor):
        if not isinstance(tensor, torch.Tensor):
            return tensor  # type: ignore[return-value]
        if not tensor.is_floating_point():
            return tensor
        if precision == "fp16":
            return tensor.half()
        return tensor

    converted = {k: convert(v) for k, v in state_dict.items()}
    payload = {"model_state": converted, "meta": {**metadata, "precision": precision}}
    torch.save(payload, out_path)
    print(f"[ok] saved weights ({precision}) -> {out_path}")


def _export_script(model: nn.Module, example: torch.Tensor, dest: Path, *, use_half: bool, device: torch.device | str):
    model = model.to(device)
    example = example.to(device)
    if use_half:
        model = model.half()
        example = example.half()
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
        try:
            optimized = torch.jit.optimize_for_inference(traced)
            scripted = torch.jit.freeze(optimized)
        except AttributeError:
            scripted = traced
    scripted.save(str(dest))
    dtype = "fp16" if use_half else "fp32"
    print(f"[ok] TorchScript ({dtype}, device={torch.device(device).type}) -> {dest}")


def _export_int8(model: nn.Module, example: torch.Tensor, dest: Path) -> None:
    try:
        from torch.ao.quantization import QConfigMapping, get_default_qconfig
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    except ImportError as exc:
        raise RuntimeError(f"INT8 export import failed: {exc}") from exc

    model = model.cpu().eval()
    example = example.cpu()

    # CPU INT8 backend
    torch.backends.quantized.engine = "fbgemm"

    # Stable across torch 2.x
    from torch.ao.quantization import get_default_qconfig_mapping
    qconfig_map = get_default_qconfig_mapping("fbgemm")

    prepared = prepare_fx(model, qconfig_map, example_inputs=(example,))

    # "Calibration" pass (collect activation stats)
    with torch.no_grad():
        prepared(example)

    quantized = convert_fx(prepared)

    # Export TorchScript
    with torch.no_grad():
        traced = torch.jit.trace(quantized, example)
        try:
            optimized = torch.jit.optimize_for_inference(traced)
            scripted = torch.jit.freeze(optimized)
        except AttributeError:
            scripted = traced

    scripted.save(str(dest))
    print(f"[ok] TorchScript (int8 cpu) -> {dest}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Attention R2UNet inference artifacts.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to training checkpoint (.pt)")
    parser.add_argument("--config", type=Path, default=None, help="Optional config.json to auto-fill params")
    parser.add_argument("--img-ch", type=int, default=None, help="Input channels (default from config or 3)")
    parser.add_argument("--output-ch", type=int, default=None, help="Output channels (default from config or 1)")
    parser.add_argument("--t", type=int, default=None, help="Recurrent steps (default from config or 2)")
    parser.add_argument("--image-size", type=int, default=None, help="Training image size (required if no config)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for exported artifacts")
    parser.add_argument("--gpu-device", type=str, default="cuda:0", help="CUDA device to trace GPU TorchScript")
    parser.add_argument(
        "--skip-cpu-int8",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip generating INT8 TorchScript for CPU.",
    )
    parser.add_argument(
        "--allow-gpu-trace-on-cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Trace the GPU TorchScript variant on CPU when CUDA is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint)

    cfg = _read_config(args.config or (args.checkpoint.parent / "config.json"))
    t = _resolve(args.t, cfg.get("t"), 2)
    img_ch = _resolve(args.img_ch, cfg.get("img_ch"), 3)
    output_ch = _resolve(args.output_ch, cfg.get("output_ch"), 1)
    image_size = _resolve(args.image_size, cfg.get("image_size"))
    if image_size is None:
        raise ValueError("--image-size not provided and missing in config.json")
    padded_size = int(math.ceil(image_size / 16) * 16)
    if padded_size != image_size:
        print(f"[info] requested image_size={image_size} -> padded to {padded_size} to satisfy stride requirements")

    meta = {
        "model": "attention_r2unet",
        "img_ch": img_ch,
        "output_ch": output_ch,
        "t": t,
        "image_size": image_size,
        "padded_size": padded_size,
        "source": str(args.checkpoint),
    }

    example = torch.randn(1, img_ch, padded_size, padded_size)
    model = AttentionR2UNet(img_ch=img_ch, output_ch=output_ch, t=t)
    model.load_state_dict(state)

    out_dir = args.output_dir or (Path("exports") / args.checkpoint.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_weights(state, "fp32", out_dir / "attention_r2unet_inference_fp32.pt", meta)
    _save_weights(state, "fp16", out_dir / "attention_r2unet_inference_fp16.pt", meta)

    _export_script(model, example.clone(), out_dir / "attention_r2unet_cpu_fp32.ts", use_half=False, device="cpu")

    if not args.skip_cpu_int8:
        try:
            _export_int8(model, example.clone(), out_dir / "attention_r2unet_cpu_int8.ts")
        except Exception as exc:
            print(f"[warn] CPU INT8 export failed: {exc}")

    if torch.cuda.is_available():
        _export_script(model, example.clone(), out_dir / "attention_r2unet_gpu_fp32.ts", use_half=False, device=args.gpu_device)
    elif args.allow_gpu_trace_on_cpu:
        try:
            _export_script(model, example.clone(), out_dir / "attention_r2unet_gpu_fp32.ts", use_half=False, device="cpu")
        except RuntimeError as exc:
            print(f"[warn] GPU TorchScript tracing on CPU failed: {exc}")
    else:
        print("[warn] CUDA unavailable; skipping GPU TorchScript export")


if __name__ == "__main__":
    main()
