"""
Batch inference helper for exported UNet checkpoints.

Given a directory that contains artifacts produced by `export_inference.py`
and a list of image paths, this script runs the segmentation network and
writes prediction masks (and optional overlays) to disk.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import time
import torch
import torch.backends.quantized
from PIL import Image
from torchvision.transforms import functional as TF

from unet import AttentionR2UNet, AttentionUNet, R2UNet, UNet


def build_model(name: str, img_ch: int, output_ch: int, t: int) -> torch.nn.Module:
    name = name.lower()
    if name == "unet":
        return UNet(img_ch=img_ch, output_ch=output_ch)
    if name == "r2unet":
        return R2UNet(img_ch=img_ch, output_ch=output_ch, t=t)
    if name == "attunet":
        return AttentionUNet(img_ch=img_ch, output_ch=output_ch)
    if name in {"attr2unet", "attention_r2unet", "att_r2unet"}:
        return AttentionR2UNet(img_ch=img_ch, output_ch=output_ch, t=t)
    raise ValueError(f"Unsupported model name: {name}")


def resolve_value(*candidates):
    for value in candidates:
        if value is None:
            continue
        return value
    return None


def _letterbox_and_pad(image: Image.Image, target_size: int) -> Tuple[Image.Image, Tuple[int, int]]:
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = image.resize((new_w, new_h), Image.BILINEAR)
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    padded = TF.pad(resized, (0, 0, pad_w, pad_h), fill=0)

    extra_w = (16 - (padded.width % 16)) % 16
    extra_h = (16 - (padded.height % 16)) % 16
    if extra_w or extra_h:
        padded = TF.pad(padded, (0, 0, extra_w, extra_h), fill=0)

    return padded, (new_w, new_h)


def preprocess_image(path: Path, target_size: int) -> Tuple[torch.Tensor, dict, Image.Image]:
    image = Image.open(path).convert("RGB")
    padded, resized_shape = _letterbox_and_pad(image, target_size)
    tensor = TF.to_tensor(padded).unsqueeze(0)
    meta = {
        "orig_size": image.size,
        "resized_size": resized_shape,
        "target_size": padded.size,
    }
    return tensor, meta, image


def postprocess_mask(mask: torch.Tensor, meta: dict, threshold: float) -> Image.Image:
    mask = torch.sigmoid(mask)
    mask = (mask > threshold).float()
    pil_mask = TF.to_pil_image(mask.squeeze(0))
    resized_w, resized_h = meta["resized_size"]
    if pil_mask.size != meta["target_size"]:
        pil_mask = pil_mask.resize(meta["target_size"], Image.NEAREST)
    pil_mask = pil_mask.crop((0, 0, resized_w, resized_h))
    pil_mask = pil_mask.resize(meta["orig_size"], Image.NEAREST)
    return pil_mask


def save_overlay(image: Image.Image, mask: Image.Image, out_path: Path, alpha: float = 0.45) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    overlay.putalpha(mask)
    blended = Image.blend(base, overlay, alpha=alpha)
    blended.save(out_path)


def load_model(weights_path: Path, device: torch.device, model_name: str, t: int, img_ch: int, output_ch: int):
    if weights_path.suffix == ".ts":
        model = torch.jit.load(str(weights_path), map_location=device)
        model.eval()
        return model, True, {}

    payload = torch.load(weights_path, map_location=device)
    state = payload.get("model_state", payload)
    meta = payload.get("meta", {})

    model_name = resolve_value(model_name, meta.get("model"), "attr2unet")
    t = resolve_value(t, meta.get("t"), 2)
    img_ch = resolve_value(img_ch, meta.get("img_ch"), 3)
    output_ch = resolve_value(output_ch, meta.get("output_ch"), 1)

    model = build_model(model_name, img_ch=img_ch, output_ch=output_ch, t=t)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, False, meta


def infer_on_images(
    model,
    scripted: bool,
    image_paths: Iterable[Path],
    target_size: int,
    threshold: float,
    output_dir: Path,
    device: torch.device,
    save_overlay_flag: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    total_time = 0.0
    count = 0
    for path in image_paths:
        tensor, meta, original = preprocess_image(path, target_size)
        tensor = tensor.to(device)
        with torch.no_grad():
            start = time.perf_counter()
            logits = model(tensor) if scripted else model(tensor)
            torch.cuda.synchronize() if device.type == "cuda" else None
            total_time += time.perf_counter() - start
            count += 1
        mask = postprocess_mask(logits.cpu(), meta, threshold)

        stem = path.stem
        mask_path = output_dir / f"{stem}_mask.png"
        mask.save(mask_path)

        if save_overlay_flag:
            overlay_path = output_dir / f"{stem}_overlay.png"
            save_overlay(original, mask, overlay_path)

        print(f"[ok] {path} -> {mask_path}")
    if count:
        avg = total_time / count
        print(f"[perf] processed {count} images in {total_time:.3f}s (avg {avg:.4f}s per image)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exported UNet weights on a list of images.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to inference weights (.pt or TorchScript .ts).")
    parser.add_argument("--images", type=Path, nargs="+", required=True, help="List of image files.")
    parser.add_argument("--output-dir", type=Path, default=Path("inference_results"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", type=str, default=None, help="Model name override if not stored in weights metadata.")
    parser.add_argument("--t", type=int, default=None, help="Recurrent steps override for R2 variants.")
    parser.add_argument("--img-ch", type=int, default=None)
    parser.add_argument("--output-ch", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=600, help="Base resize dimension before padding.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--quant-engine", type=str, default="fbgemm", help="Quantized backend (use fbgemm or qnnpack).")
    parser.add_argument("--save-overlay", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.quant_engine:
        torch.backends.quantized.engine = args.quant_engine
    padded_size = int(math.ceil(args.image_size / 16) * 16)

    model, scripted, meta = load_model(
        weights_path=args.weights,
        device=device,
        model_name=args.model,
        t=args.t,
        img_ch=args.img_ch,
        output_ch=args.output_ch,
    )

    target_size = resolve_value(meta.get("padded_size"), padded_size)
    infer_on_images(
        model=model,
        scripted=scripted,
        image_paths=args.images,
        target_size=target_size,
        threshold=args.threshold,
        output_dir=args.output_dir,
        device=device,
        save_overlay_flag=args.save_overlay,
    )


if __name__ == "__main__":
    main()
