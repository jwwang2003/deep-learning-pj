"""Shared building blocks for UNet variants.

The blocks are written to be friendly with PyTorch 2.x (no deprecated
APIs, batchnorm/conv modules only) and are safe to use with torch.compile.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import init

__all__ = [
    "init_weights",
    "ConvBlock",
    "UpConv",
    "RecurrentBlock",
    "RRCNNBlock",
    "AttentionBlock",
    "conv_block",
    "up_conv",
    "Recurrent_block",
    "RRCNN_block",
    "Attention_block",
]


def init_weights(net: nn.Module, init_type: str = "kaiming", gain: float = 0.02) -> None:
    """Initialize weights in a network using a standard strategy."""

    def init_func(module: nn.Module) -> None:
        classname = module.__class__.__name__
        if hasattr(module, "weight") and ("Conv" in classname or "Linear" in classname):
            if init_type == "normal":
                init.normal_(module.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(module.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(module.weight.data, gain=gain)
            else:
                raise NotImplementedError(f"initialization method [{init_type}] is not implemented")
            if hasattr(module, "bias") and module.bias is not None:
                init.constant_(module.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1 and hasattr(module, "weight"):
            init.normal_(module.weight.data, 1.0, gain)
            init.constant_(module.bias.data, 0.0)

    net.apply(init_func)


class ConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        scale_factor: int = 2,
        mode: str = "bilinear",
        align_corners: Optional[bool] = False,
    ):
        super().__init__()
        align = align_corners if mode in {"linear", "bilinear", "bicubic", "trilinear"} else None
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class RecurrentBlock(nn.Module):
    def __init__(self, ch_out: int, t: int = 2):
        super().__init__()
        if t < 1:
            raise ValueError("t must be >= 1")
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        for _ in range(1, self.t):
            out = self.conv(x + out)
        return out


class RRCNNBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, t: int = 2):
        super().__init__()
        self.rcnn = nn.Sequential(
            RecurrentBlock(ch_out, t=t),
            RecurrentBlock(ch_out, t=t),
        )
        self.conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1x1(x)
        out = self.rcnn(x)
        return x + out


class AttentionBlock(nn.Module):
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int),
        )

        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# Backwards-compatible aliases for legacy code imports
conv_block = ConvBlock
up_conv = UpConv
Recurrent_block = RecurrentBlock
RRCNN_block = RRCNNBlock
Attention_block = AttentionBlock
