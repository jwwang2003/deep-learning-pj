#import "@preview/neural-netz:0.3.0": draw-network

#set page(width: auto, height: auto, margin: 5mm)

= Attention R2 U-Net

An Attention R2 U-Net with recurrent residual blocks, attention-gated skips, and up-convolutions.

#draw-network((
  (type: "input", image: "default", channels: ("1", "H x W"), widths: (0.25,), height: 8, depth: 8, label: "Input", name: "input"),

  (type: "convres", label: "RRCNN 64", channels: ("64", "64"), widths: (0.6, 0.6), height: 8, depth: 8, name: "down1", offset: 1.5),
  (type: "pool", height: 6.5, depth: 6.5, name: "pool1", connection-label: "2x2 maxpool"),

  (type: "convres", label: "RRCNN 128", channels: ("128", "128"), widths: (0.75, 0.75), height: 6.5, depth: 6.5, name: "down2"),
  (type: "pool", height: 5, depth: 5, name: "pool2"),

  (type: "convres", label: "RRCNN 256", channels: ("256", "256"), widths: (0.9, 0.9), height: 5, depth: 5, name: "down3"),
  (type: "pool", height: 3.8, depth: 3.8, name: "pool3"),

  (type: "convres", label: "RRCNN 512", channels: ("512", "512"), widths: (1.05, 1.05), height: 3.8, depth: 3.8, name: "down4"),
  (type: "pool", height: 2.8, depth: 2.8, name: "pool4"),

  (type: "convres", label: "RRCNN 1024", channels: ("1024", "1024"), widths: (1.3, 1.3), height: 2.8, depth: 2.8, name: "bottleneck"),

  (type: "deconv", label: "UpConv 512", channels: ("512", ""), widths: (0.7,), height: 3.8, depth: 3.8, name: "up4", offset: 1.4),
  (type: "custom", label: "Attention Gate", legend: "Attention gate", widths: (0.35,), height: 3.8, depth: 3.8, name: "gate4", show-connection: true),
  (type: "concat", label: "Concat", widths: (0.35,), height: 3.8, depth: 3.8, name: "cat4", connection-label: "skip + gating"),
  (type: "convres", label: "RRCNN 512", channels: ("512", "512"), widths: (1.0, 1.0), height: 3.8, depth: 3.8, name: "dec4"),

  (type: "deconv", label: "UpConv 256", channels: ("256", ""), widths: (0.65,), height: 5, depth: 5, name: "up3", offset: 1.4),
  (type: "custom", label: "Attention Gate", widths: (0.35,), height: 5, depth: 5, name: "gate3", show-connection: true),
  (type: "concat", label: "Concat", widths: (0.35,), height: 5, depth: 5, name: "cat3", connection-label: "skip + gating"),
  (type: "convres", label: "RRCNN 256", channels: ("256", "256"), widths: (0.9, 0.9), height: 5, depth: 5, name: "dec3"),

  (type: "deconv", label: "UpConv 128", channels: ("128", ""), widths: (0.6,), height: 6.5, depth: 6.5, name: "up2", offset: 1.4),
  (type: "custom", label: "Attention Gate", widths: (0.35,), height: 6.5, depth: 6.5, name: "gate2", show-connection: true),
  (type: "concat", label: "Concat", widths: (0.35,), height: 6.5, depth: 6.5, name: "cat2", connection-label: "skip + gating"),
  (type: "convres", label: "RRCNN 128", channels: ("128", "128"), widths: (0.75, 0.75), height: 6.5, depth: 6.5, name: "dec2"),

  (type: "deconv", label: "UpConv 64", channels: ("64", ""), widths: (0.55,), height: 8, depth: 8, name: "up1", offset: 1.4),
  (type: "custom", label: "Attention Gate", widths: (0.35,), height: 8, depth: 8, name: "gate1", show-connection: true),
  (type: "concat", label: "Concat", widths: (0.35,), height: 8, depth: 8, name: "cat1", connection-label: "skip + gating"),
  (type: "convres", label: "RRCNN 64", channels: ("64", "64"), widths: (0.6, 0.6), height: 8, depth: 8, name: "dec1"),

  (type: "conv", label: "1x1 Conv", channels: ("C", ""), widths: (0.25,), height: 8, depth: 8, name: "seg-head"),
  (type: "output", label: "Segmentation Map", channels: ("", ""), height: 8, depth: 8, name: "output", show-connection: false),
),
connections: (
  (from: "down4", to: "gate4", type: "skip", mode: "air", pos: 2.4, touch-layer: true, label: "skip"),
  (from: "down3", to: "gate3", type: "skip", mode: "air", pos: 3.0, touch-layer: true),
  (from: "down2", to: "gate2", type: "skip", mode: "air", pos: 3.6, touch-layer: true),
  (from: "down1", to: "gate1", type: "skip", mode: "air", pos: 4.2, touch-layer: true)
),
palette: "cold",
show-legend: true,
legend-title: "Attention R2 U-Net",
show-relu: true,
scale: 100%,
depth-multiplier: 0.35,
stroke-thickness: 1.1,
)
