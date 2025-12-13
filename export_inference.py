"""
Utilities to export inference-only weights and TorchScript artifacts.

This script produces two kinds of outputs:

1. Lightweight checkpoints that only contain the model weights (no optimizer state)
   stored in FP32 and FP16 to keep GPU deployments lean.
2. TorchScript bundles that have `torch.jit.optimize_for_inference` applied.  A
   CPU-friendly FP32 script is always emitted, and we attempt to create a GPU-
   optimized FP16 script (requires CUDA at export time).  Optionally, an
   experimental int8 quantized script for CPU is generated via FX quantization.

Example:
    python export_inference.py \
        --checkpoint runs/attr2unet/best.pt \
        --config runs/attr2unet/config.json \
        --output-dir exports/attr2unet \
        --image-size 600 \
        --targets weights gpu-script cpu-script cpu-int8
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import nn
import torch.backends.quantized

from unet import AttentionR2UNet, AttentionUNet, R2UNet, UNet

ModelName = str


def _read_training_config(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    with path.open("r") as f:
        data = json.load(f)
    return data


def build_export_model(model_name: ModelName, img_ch: int, output_ch: int, t: int) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "unet":
        return UNet(img_ch=img_ch, output_ch=output_ch)
    if model_name == "r2unet":
        return R2UNet(img_ch=img_ch, output_ch=output_ch, t=t)
    if model_name == "attunet":
        return AttentionUNet(img_ch=img_ch, output_ch=output_ch)
    if model_name in {"attr2unet", "attention_r2unet", "att_r2unet"}:
        return AttentionR2UNet(img_ch=img_ch, output_ch=output_ch, t=t)
    raise ValueError(f"Unsupported model={model_name!r}")


def save_weights_only(
    state_dict: Mapping[str, torch.Tensor],
    out_path: Path,
    precision: str,
    metadata: dict,
) -> None:
    def _convert(tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            return tensor  # type: ignore[return-value]
        if not tensor.is_floating_point():
            return tensor
        return tensor.half() if precision == "fp16" else tensor

    converted = {k: _convert(v) for k, v in state_dict.items()}
    payload = {"model_state": converted, "meta": {**metadata, "precision": precision}}
    torch.save(payload, out_path)
    print(f"[ok] wrote weights ({precision}) -> {out_path}")


def export_torchscript(
    model: nn.Module,
    example: torch.Tensor,
    out_path: Path,
    use_half: bool = False,
    device: torch.device | str = "cpu",
) -> None:
    dev = torch.device(device)
    model = model.to(dev)
    example = example.to(dev)

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
            # Some PyTorch builds drop the `.training` attribute from traced modules.
            # Fall back to the traced object so export can proceed.
            scripted = traced
    scripted.save(str(out_path))
    dtype = "fp16" if use_half else "fp32"
    print(f"[ok] TorchScript ({dtype}, device={dev.type}) -> {out_path}")


def export_quantized_cpu(model: nn.Module, example: torch.Tensor, out_path: Path) -> None:
    try:
        from torch.ao.quantization import QConfig, QConfigMapping
        from torch.ao.quantization.fake_quantize import default_minmax_observer
        from torch.ao.quantization.observer import default_per_tensor_affine_observer
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch.ao.quantization is unavailable in this environment") from exc

    model = model.cpu().eval()
    example = example.cpu()

    torch.backends.quantized.engine = "fbgemm"
    per_tensor_qconfig = QConfig(
        activation=default_minmax_observer,
        weight=default_per_tensor_affine_observer,
    )
    qconfig_mapping = QConfigMapping().set_global(per_tensor_qconfig)
    prepared = prepare_fx(model, qconfig_mapping, example_inputs=(example,))
    with torch.no_grad():
        prepared(example)
    quantized = convert_fx(prepared)
    with torch.no_grad():
        traced = torch.jit.trace(quantized, example)
        try:
            optimized = torch.jit.optimize_for_inference(traced)
            scripted = torch.jit.freeze(optimized)
        except AttributeError:
            scripted = traced
    scripted.save(str(out_path))
    print(f"[ok] TorchScript (int8 cpu) -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export inference-ready weights / TorchScript bundles")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to training checkpoint (best.pt)")
    parser.add_argument("--config", type=Path, default=None, help="Optional config.json to auto-detect model params")
    parser.add_argument("--model", type=str, default=None, help="Model name override (unet|r2unet|attunet|attr2unet)")
    parser.add_argument("--t", type=int, default=None, help="Recurrent steps for R2 variants")
    parser.add_argument("--img-ch", type=int, default=None, help="Input channels (default from config or 3)")
    parser.add_argument("--output-ch", type=int, default=None, help="Output channels (default from config or 1)")
    parser.add_argument("--image-size", type=int, default=None, help="Input spatial size used during training")
    parser.add_argument("--targets", nargs="+", default=["weights", "gpu-script", "cpu-script"], choices=["weights", "gpu-script", "cpu-script", "cpu-int8", "vanilla"], help="Artifacts to export")
    parser.add_argument("--output-dir", type=Path, default=Path("exports"), help="Directory for exported files")
    parser.add_argument("--gpu-device", type=str, default="cuda:0", help="GPU device to use when tracing fp16 script")
    parser.add_argument("--skip-fp16-script-on-cpu", action=argparse.BooleanOptionalAction, default=True, help="Skip fp16 TorchScript export if CUDA is unavailable.")
    return parser.parse_args()


def resolve_param(value, *fallbacks):
    for item in (value, *fallbacks):
        if item is None:
            continue
        return item
    return None


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    metadata = {"source": str(args.checkpoint)}

    cfg_json = _read_training_config(args.config or (args.checkpoint.parent / "config.json"))
    model_name = resolve_param(args.model, cfg_json.get("model"), "attr2unet")
    t = resolve_param(args.t, cfg_json.get("t"), 2)
    img_ch = resolve_param(args.img_ch, cfg_json.get("img_ch"), 3)
    output_ch = resolve_param(args.output_ch, cfg_json.get("output_ch"), 1)
    requested_size = resolve_param(args.image_size, cfg_json.get("image_size"))
    if requested_size is None:
        raise ValueError("--image-size was not provided and could not be inferred from config.json")
    padded_size = int(math.ceil(requested_size / 16) * 16)
    if padded_size != requested_size:
        print(f"[info] requested image_size={requested_size} -> padding to {padded_size} (multiple of 16).")

    metadata.update(
        {
            "model": model_name,
            "t": t,
            "img_ch": img_ch,
            "output_ch": output_ch,
            "image_size": requested_size,
            "padded_size": padded_size,
        }
    )

    example = torch.randn(1, img_ch, padded_size, padded_size)

    model = build_export_model(model_name, img_ch=img_ch, output_ch=output_ch, t=t)
    model.load_state_dict(state)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if "weights" in args.targets:
        save_weights_only(state, out_dir / "inference_fp32.pt", "fp32", metadata)
        save_weights_only(state, out_dir / "inference_fp16.pt", "fp16", metadata)

    if "vanilla" in args.targets:
        dest = out_dir / "checkpoint_vanilla.pt"
        shutil.copy2(args.checkpoint, dest)
        print(f"[ok] copied original checkpoint -> {dest}")

    if "cpu-script" in args.targets:
        export_torchscript(model, example.clone(), out_dir / "model_cpu_fp32.ts", use_half=False, device="cpu")

    if "cpu-int8" in args.targets:
        try:
            export_quantized_cpu(model, example.clone(), out_dir / "model_cpu_int8.ts")
        except Exception as exc:
            print(f"[warn] CPU quantization failed: {exc}")

    if "gpu-script" in args.targets:
        if torch.cuda.is_available():
            export_torchscript(
                model,
                example.clone(),
                out_dir / "model_gpu_fp16.ts",
                use_half=True,
                device=args.gpu_device,
            )
        else:
            msg = "CUDA unavailable; skipping GPU TorchScript export."
            if not args.skip_fp16_script_on_cpu:
                try:
                    export_torchscript(
                        model,
                        example.clone(),
                        out_dir / "model_gpu_fp16.ts",
                        use_half=True,
                        device="cpu",
                    )
                except RuntimeError as exc:
                    msg += f" (tried CPU fp16 and failed: {exc})"
            print(f"[warn] {msg}")


if __name__ == "__main__":
    main()
