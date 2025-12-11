"""UNet model zoo with PyTorch 2.x-friendly implementations.

Usage (PyTorch 2.x):
    >>> import torch
    >>> from unet import AttentionR2UNet, init_weights
    >>> model = AttentionR2UNet(img_ch=3, output_ch=1)
    >>> model = model.to(torch.device("cuda"), dtype=torch.float32)
    >>> init_weights(model, init_type="kaiming")

All models are safe to wrap in torch.compile for speedups.
"""

from .blocks import (
    AttentionBlock,
    ConvBlock,
    RecurrentBlock,
    RRCNNBlock,
    UpConv,
    init_weights,
)
from .att_r2unet import AttU_Net, AttentionR2UNet, AttentionUNet, R2AttU_Net
from .r2unet import R2UNet
from .unet import UNet

__all__ = [
    "UNet",
    "R2UNet",
    "AttentionUNet",
    "AttentionR2UNet",
    "AttU_Net",
    "R2AttU_Net",
    "AttentionBlock",
    "ConvBlock",
    "RecurrentBlock",
    "RRCNNBlock",
    "UpConv",
    "init_weights",
]
