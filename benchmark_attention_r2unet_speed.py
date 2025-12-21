"""
Benchmark exported Attention R2UNet weights on the held-out test split.

The script mirrors the dataset splitting logic from `train.py`, loads the
artifacts produced by `export_attention_r2unet.py`, and measures the average
inference latency for every CPU artifact (on CPU) and every GPU artifact on
each available CUDA device.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Set

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from PIL import Image

from helpers.coco_dataset import CocoPaths, CocoSegmentationDataset, make_transforms
from unet import AttentionR2UNet


@dataclass
class BenchmarkArtifact:
    name: str
    path: Path
    device_type: str  # "cpu" or "cuda"

    @property
    def is_script(self) -> bool:
        return self.path.suffix == ".ts"


@dataclass
class DemoConfig:
    out_dir: Path
    threshold: float
    saved_keys: Set[str] = field(default_factory=set)

    def key(self, artifact: BenchmarkArtifact, device: torch.device) -> str:
        suffix = device.type if device.index is None else f"{device.type}{device.index}"
        return f"{artifact.name}_{suffix}"

    def has(self, artifact: BenchmarkArtifact, device: torch.device) -> bool:
        return self.key(artifact, device) in self.saved_keys

    def mark(self, artifact: BenchmarkArtifact, device: torch.device) -> None:
        self.saved_keys.add(self.key(artifact, device))


class MetricTracker:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.tn = 0.0
        self.fn = 0.0

    def update(self, outputs, masks: torch.Tensor) -> None:
        logits = ensure_tensor_output(outputs)
        if logits.dim() < 4:
            return
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        labels = (masks.to(logits.device) > 0.5).float()
        inv_preds = 1.0 - preds
        inv_labels = 1.0 - labels
        self.tp += (preds * labels).sum().item()
        self.fp += (preds * inv_labels).sum().item()
        self.fn += (inv_preds * labels).sum().item()
        self.tn += (inv_preds * inv_labels).sum().item()

    def summary(self) -> dict[str, float]:
        eps = 1e-8
        total = self.tp + self.fp + self.fn + self.tn
        accuracy = (self.tp + self.tn) / (total + eps) if total > 0 else 0.0
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        dice = (2 * self.tp) / (2 * self.tp + self.fp + self.fn + eps)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "dice": dice,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure inference speed of exported Attention R2UNet weights.")
    parser.add_argument("--export-dir", type=Path, required=True, help="Directory with export_attention_r2unet outputs.")
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root containing result.json/images/mask.")
    parser.add_argument("--coco-json", type=Path, default=None, help="Optional override for COCO annotations.")
    parser.add_argument("--images-dir", type=Path, default=None, help="Optional override for COCO images directory.")
    parser.add_argument("--masks-dir", type=Path, default=None, help="Optional override for mask directory.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=None, help="Base image size (defaults to export metadata).")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed used for the train/val/test split.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional limit for debugging.")
    parser.add_argument("--warmup-batches", type=int, default=2, help="How many batches to run before timing.")
    parser.add_argument("--cpu-artifacts", type=Path, nargs="*", default=None, help="Explicit list of CPU artifacts.")
    parser.add_argument("--gpu-artifacts", type=Path, nargs="*", default=None, help="Explicit list of GPU artifacts.")
    parser.add_argument("--img-ch", type=int, default=None, help="Override image channels when loading .pt weights.")
    parser.add_argument("--output-ch", type=int, default=None, help="Override output channels for .pt weights.")
    parser.add_argument("--t", type=int, default=None, help="Override recurrent steps for .pt weights.")
    parser.add_argument("--quant-engine", type=str, default="fbgemm", help="Quantized backend for INT8 models.")
    parser.add_argument("--demo-dir", type=Path, default=None, help="If set, export demo images per artifact/device.")
    parser.add_argument("--demo-threshold", type=float, default=None, help="Override threshold for demo mask binarization.")
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binarizing predictions when computing metrics.",
    )
    parser.add_argument(
        "--device-mode",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Run only CPU artifacts, GPU artifacts, or both.",
    )
    return parser.parse_args()


def discover_artifacts(
    export_dir: Path,
    overrides: Sequence[Path] | None,
    target_device: str,
) -> List[BenchmarkArtifact]:
    if overrides:
        return [BenchmarkArtifact(name=p.stem, path=p, device_type=target_device) for p in overrides]

    pattern = "*cpu*.ts" if target_device == "cpu" else "*gpu*.ts"
    discovered = sorted(export_dir.glob(pattern))
    return [BenchmarkArtifact(name=path.stem, path=path, device_type=target_device) for path in discovered]


def infer_precision_hint(path: Path) -> str | None:
    stem = path.stem.lower()
    if "int8" in stem:
        return "int8"
    if "fp16" in stem:
        return "fp16"
    if "fp32" in stem:
        return "fp32"
    return None


def load_meta_from_export(export_dir: Path) -> dict:
    """Grab metadata from the first available *.pt inference weight."""
    for candidate in sorted(export_dir.glob("*inference*.pt")):
        payload = torch.load(candidate, map_location="cpu")
        meta = payload.get("meta")
        if isinstance(meta, dict):
            return meta
    return {}


def build_test_loader(
    args: argparse.Namespace,
    image_size: int,
) -> tuple[DataLoader, int]:
    coco_json = args.coco_json or (args.data_root / "result.json")
    images_dir = args.images_dir or (args.data_root / "images")
    masks_dir = args.masks_dir if args.masks_dir else (args.data_root / "mask")
    if masks_dir and not masks_dir.exists():
        masks_dir = None

    paths = CocoPaths(coco_json=coco_json, images_dir=images_dir, masks_dir=masks_dir)
    base_dataset = CocoSegmentationDataset(paths, transform=None)
    total = len(base_dataset)
    if total < 3:
        raise RuntimeError(f"Dataset too small for a split (found {total} samples).")

    train_size = max(1, int(total * 0.7))
    val_size = max(1, int(total * 0.1))
    test_size = max(1, total - train_size - val_size)
    if train_size + val_size + test_size > total:
        test_size = total - train_size - val_size

    indices = list(range(total))
    generator = torch.Generator().manual_seed(args.split_seed)
    _, _, test_subset = torch.utils.data.random_split(indices, [train_size, val_size, test_size], generator=generator)
    test_indices = list(test_subset)

    test_dataset = CocoSegmentationDataset(
        paths,
        transform=make_transforms(train=False, size=image_size),
        indices=test_indices,
    )
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    return loader, len(test_dataset)


def build_attention_r2unet(
    meta: dict,
    overrides: argparse.Namespace,
) -> AttentionR2UNet:
    img_ch = overrides.img_ch or meta.get("img_ch") or 3
    output_ch = overrides.output_ch or meta.get("output_ch") or 1
    t = overrides.t or meta.get("t") or 2
    return AttentionR2UNet(img_ch=img_ch, output_ch=output_ch, t=t)


def load_model_from_artifact(
    artifact: BenchmarkArtifact,
    device: torch.device,
    overrides: argparse.Namespace,
) -> tuple[torch.nn.Module, dict, str | None]:
    if artifact.is_script:
        model = torch.jit.load(str(artifact.path), map_location=device)
        model.eval()
        return model, {}, infer_precision_hint(artifact.path)

    payload = torch.load(artifact.path, map_location="cpu")
    state = payload.get("model_state", payload)
    meta = payload.get("meta", {})
    precision = (meta.get("precision") or infer_precision_hint(artifact.path)) or "fp32"
    if precision == "fp16" and device.type != "cuda":
        state = {
            k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
            for k, v in state.items()
        }
    model = build_attention_r2unet(meta, overrides)
    model.load_state_dict(state)
    model = model.to(device)
    if precision == "fp16" and device.type == "cuda":
        model = model.half()
    model.eval()
    return model, meta, precision


def prepare_batch(images: torch.Tensor, device: torch.device, precision: str | None) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if precision == "fp16" and device.type == "cuda":
        images = images.half()
    return images


def ensure_tensor_output(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Unsupported model output type: {type(output)}")


def build_overlay(image: Image.Image, mask: Image.Image, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGBA")
    mask_l = mask.convert("L")
    red = Image.new("RGBA", base.size, (255, 0, 0, 0))
    red.putalpha(mask_l)
    return Image.blend(base, red, alpha=alpha)


def maybe_save_demo(
    demo_config: DemoConfig | None,
    artifact: BenchmarkArtifact,
    device: torch.device,
    images: torch.Tensor,
    outputs,
):
    if demo_config is None or demo_config.has(artifact, device):
        return
    logits = ensure_tensor_output(outputs)
    if logits.dim() < 4:
        return
    image = images[0].detach().float().cpu().clamp(0, 1)
    mask = torch.sigmoid(logits[0].detach().cpu())
    mask = (mask > demo_config.threshold).float()
    pil_image = TF.to_pil_image(image)
    pil_mask = TF.to_pil_image(mask)
    overlay = build_overlay(pil_image, pil_mask)
    base_name = demo_config.key(artifact, device)
    demo_config.out_dir.mkdir(parents=True, exist_ok=True)
    pil_image.save(demo_config.out_dir / f"{base_name}_image.png")
    pil_mask.save(demo_config.out_dir / f"{base_name}_mask.png")
    overlay.save(demo_config.out_dir / f"{base_name}_overlay.png")
    demo_config.mark(artifact, device)
    print(f"[demo] saved demos for {artifact.name} on {device} -> {demo_config.out_dir}")


def warmup_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    precision: str | None,
):
    if max_batches <= 0:
        return
    with torch.inference_mode():
        it = iter(loader)
        for _ in range(max_batches):
            try:
                images, _ = next(it)
            except StopIteration:
                break
            images = prepare_batch(images, device, precision)
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)


def benchmark_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
    precision: str | None,
    artifact: BenchmarkArtifact,
    demo_config: DemoConfig | None,
    threshold: float,
) -> tuple[float, int, dict[str, float]]:
    total_time = 0.0
    total_samples = 0
    processed_batches = 0
    metrics = MetricTracker(threshold)
    with torch.inference_mode():
        for images, masks in loader:
            images = prepare_batch(images, device, precision)
            masks = masks.to(device, non_blocking=True)
            start = time.perf_counter()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            total_time += elapsed
            total_samples += images.size(0)
            metrics.update(outputs, masks)
            maybe_save_demo(demo_config, artifact, device, images, outputs)
            processed_batches += 1
            if max_batches and processed_batches >= max_batches:
                break
    return total_time, total_samples, metrics.summary()


def summarize_result(
    artifact: BenchmarkArtifact,
    device: torch.device,
    meta: dict,
    precision: str | None,
    total_time: float,
    total_samples: int,
    metrics: dict[str, float],
):
    if total_samples == 0:
        print(f"[skip] {artifact.name} on {device}: no samples processed")
        return
    avg = total_time / total_samples
    throughput = total_samples / total_time if total_time > 0 else 0.0
    precision_hint = precision or meta.get("precision") or "unknown"
    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    print(
        f"[perf] {artifact.name} @ {device} "
        f"(precision={precision_hint}) -> "
        f"{total_samples} samples in {total_time:.3f}s | "
        f"avg {avg*1000:.3f} ms/sample | {throughput:.2f} samples/s | {metrics_str}"
    )


def ensure_exists(artifacts: Iterable[BenchmarkArtifact]) -> List[BenchmarkArtifact]:
    present = []
    for artifact in artifacts:
        if artifact.path.exists():
            present.append(artifact)
        else:
            print(f"[warn] missing artifact: {artifact.path}")
    return present


def main() -> None:
    args = parse_args()
    if args.quant_engine:
        torch.backends.quantized.engine = args.quant_engine

    cpu_artifacts = ensure_exists(discover_artifacts(args.export_dir, args.cpu_artifacts, "cpu"))
    gpu_artifacts = ensure_exists(discover_artifacts(args.export_dir, args.gpu_artifacts, "cuda"))

    cpu_paths = {art.path.resolve() for art in cpu_artifacts}
    gpu_paths = {art.path.resolve() for art in gpu_artifacts}
    for path in sorted(args.export_dir.glob("*inference*.pt")):
        if path.suffix != ".pt":
            continue
        target = "cuda" if infer_precision_hint(path) == "fp16" else "cpu"
        artifact = BenchmarkArtifact(name=path.stem, path=path, device_type=target)
        if target == "cpu":
            if path.resolve() not in cpu_paths:
                cpu_artifacts.append(artifact)
                cpu_paths.add(path.resolve())
        else:
            if path.resolve() not in gpu_paths:
                gpu_artifacts.append(artifact)
                gpu_paths.add(path.resolve())

    if args.device_mode == "cpu":
        gpu_artifacts = []
    elif args.device_mode == "gpu":
        cpu_artifacts = []

    if not cpu_artifacts and not gpu_artifacts:
        raise RuntimeError(f"No artifacts found in {args.export_dir} for mode={args.device_mode}")

    meta = load_meta_from_export(args.export_dir)
    image_size = args.image_size or meta.get("padded_size") or meta.get("image_size")
    if image_size is None:
        raise RuntimeError(
            "Unable to determine image size. Pass --image-size explicitly or export weights with metadata."
        )

    loader, test_count = build_test_loader(args, image_size=image_size)
    print(f"[info] Test split size: {test_count} samples | batch_size={args.batch_size}")

    demo_threshold = args.demo_threshold if args.demo_threshold is not None else args.prob_threshold
    demo_config = DemoConfig(out_dir=args.demo_dir, threshold=demo_threshold) if args.demo_dir else None

    for artifact in cpu_artifacts:
        device = torch.device("cpu")
        model, model_meta, precision = load_model_from_artifact(artifact, device=device, overrides=args)
        warmup_model(model, loader, device, args.warmup_batches, precision)
        elapsed, samples, metrics = benchmark_model(
            model,
            loader,
            device,
            args.max_batches,
            precision,
            artifact,
            demo_config,
            args.prob_threshold,
        )
        summarize_result(artifact, device, model_meta, precision, elapsed, samples, metrics)

    if gpu_artifacts:
        if not torch.cuda.is_available():
            print("[warn] CUDA unavailable, skipping GPU artifacts")
        else:
            device_count = torch.cuda.device_count()
            for artifact in gpu_artifacts:
                for idx in range(device_count):
                    device = torch.device(f"cuda:{idx}")
                    model, model_meta, precision = load_model_from_artifact(artifact, device=device, overrides=args)
                    warmup_model(model, loader, device, args.warmup_batches, precision)
                    elapsed, samples, metrics = benchmark_model(
                        model,
                        loader,
                        device,
                        args.max_batches,
                        precision,
                        artifact,
                        demo_config,
                        args.prob_threshold,
                    )
                    summarize_result(artifact, device, model_meta, precision, elapsed, samples, metrics)


if __name__ == "__main__":
    main()
