"""
Smoke test utility for YOLOv5 TorchScript bundles (.ts files).

The helper loads a TorchScript checkpoint exported via `yolov5/export.py`,
runs it on one or more images, prints the detections, and optionally writes
annotated frames for quick visual inspection.

Example:
    python -m helpers.test_yolov5_torchscript \
        --weights exports/yolov5s-aoi-fourcls/yolov5s-aoi-fourcls.ts \
        --images data/detection/demo/*.jpg \
        --output-dir runs/yolov5s_ts_check
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
Y5_ROOT = PROJECT_ROOT / "yolov5"
if Y5_ROOT.is_dir():
    y5_path = str(Y5_ROOT)
    if y5_path not in sys.path:
        sys.path.append(y5_path)

from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference using a YOLOv5 TorchScript model.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to TorchScript file (.ts)")
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=True,
        help="Image files or glob patterns (evaluation order is alphabetical).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Square inference size in pixels.")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold for NMS.")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=False, help="Use fp16 inference.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "yolov5_ts_eval",
        help="Optional directory for annotated frames.",
    )
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True, help="Write annotated outputs.")
    return parser.parse_args()


def _collect_images(patterns: Iterable[str]) -> List[Path]:
    files: List[str] = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if expanded:
            files.extend(expanded)
        else:
            files.append(pattern)
    paths = [Path(p) for p in sorted(set(files))]
    if not paths:
        raise FileNotFoundError("No images resolved from provided patterns.")
    return paths


def _prepare_image(path: Path, imgsz: int, stride: int, device: torch.device, half: bool) -> tuple[torch.Tensor, np.ndarray]:
    im0 = cv2.imread(str(path))
    if im0 is None:
        raise FileNotFoundError(f"failed to load image: {path}")
    img = letterbox(im0, imgsz, stride=stride, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to NCHW
    img = np.ascontiguousarray(img)
    tensor = torch.from_numpy(img).to(device)
    tensor = tensor.half() if half else tensor.float()
    tensor /= 255.0
    tensor = tensor.unsqueeze(0)
    return tensor, im0


def main() -> None:
    args = parse_args()
    weights_path = args.weights
    if weights_path.suffix == "":
        weights_path = weights_path.with_suffix(".ts")
    if not weights_path.is_file() and weights_path.suffix == ".ts":
        alt = weights_path.with_suffix(".torchscript")
        if alt.is_file():
            weights_path = alt
    if weights_path.suffix not in {".ts", ".torchscript"}:
        raise ValueError("Weights must be a .ts TorchScript file")
    if not weights_path.is_file():
        raise FileNotFoundError(f"TorchScript weights not found: {weights_path}")

    device = select_device(args.device)
    half = args.half and device.type != "cpu"

    model = torch.jit.load(str(weights_path), map_location=device)
    model.eval()
    model.to(device)

    stride = int(getattr(model, "stride", torch.tensor([32])).max())
    names = getattr(model, "names", None)
    imgsz = check_img_size(args.imgsz, s=stride)

    image_paths = _collect_images(args.images)
    if args.save:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0.0
    for path in image_paths:
        img, im0 = _prepare_image(path, imgsz, stride, device, half)
        start = time.perf_counter()
        with torch.no_grad():
            pred = model(img)
        elapsed = time.perf_counter() - start
        total += elapsed

        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        detections = non_max_suppression(
            pred,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            max_det=args.max_det,
        )

        print(f"[info] {path.name}: inference {elapsed*1e3:.1f} ms")
        annotator = Annotator(im0.copy(), line_width=2, example=str(names if names else "0"))
        if detections and detections[0] is not None and len(detections[0]):
            det = detections[0]
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in det.tolist():
                label = f"{int(cls)} {conf:.2f}"
                if names and isinstance(names, (list, tuple)):
                    label = f"{names[int(cls)]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(int(cls), True))
                print(f"    class={int(cls):>2} conf={conf:.3f} box={[round(v,1) for v in xyxy]}")
        else:
            print("    no detections")

        if args.save:
            out_file = args.output_dir / f"{path.stem}_ts.jpg"
            cv2.imwrite(str(out_file), annotator.result())
            print(f"    saved {out_file}")

    if image_paths:
        print(f"[done] processed {len(image_paths)} images, avg {total / len(image_paths):.3f}s per image")


if __name__ == "__main__":
    main()
