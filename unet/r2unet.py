"""Recurrent Residual UNet (R2UNet)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .blocks import RRCNNBlock, UpConv

__all__ = ["R2UNet"]


class R2UNet(nn.Module):
    def __init__(
        self,
        img_ch: int = 3,
        output_ch: int = 1,
        t: int = 2,
        up_mode: str = "bilinear",
        align_corners: Optional[bool] = False,
    ):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.rrcnn1 = RRCNNBlock(ch_in=img_ch, ch_out=64, t=t)
        self.rrcnn2 = RRCNNBlock(ch_in=64, ch_out=128, t=t)
        self.rrcnn3 = RRCNNBlock(ch_in=128, ch_out=256, t=t)
        self.rrcnn4 = RRCNNBlock(ch_in=256, ch_out=512, t=t)
        self.rrcnn5 = RRCNNBlock(ch_in=512, ch_out=1024, t=t)

        self.up5 = UpConv(ch_in=1024, ch_out=512, mode=up_mode, align_corners=align_corners)
        self.up_rrcnn5 = RRCNNBlock(ch_in=1024, ch_out=512, t=t)

        self.up4 = UpConv(ch_in=512, ch_out=256, mode=up_mode, align_corners=align_corners)
        self.up_rrcnn4 = RRCNNBlock(ch_in=512, ch_out=256, t=t)

        self.up3 = UpConv(ch_in=256, ch_out=128, mode=up_mode, align_corners=align_corners)
        self.up_rrcnn3 = RRCNNBlock(ch_in=256, ch_out=128, t=t)

        self.up2 = UpConv(ch_in=128, ch_out=64, mode=up_mode, align_corners=align_corners)
        self.up_rrcnn2 = RRCNNBlock(ch_in=128, ch_out=64, t=t)

        self.conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoding path
        x1 = self.rrcnn1(x)

        x2 = self.maxpool(x1)
        x2 = self.rrcnn2(x2)

        x3 = self.maxpool(x2)
        x3 = self.rrcnn3(x3)

        x4 = self.maxpool(x3)
        x4 = self.rrcnn4(x4)

        x5 = self.maxpool(x4)
        x5 = self.rrcnn5(x5)

        # decoding + concat path
        d5 = self.up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_rrcnn5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_rrcnn4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_rrcnn3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_rrcnn2(d2)

        d1 = self.conv_1x1(d2)

        return d1
