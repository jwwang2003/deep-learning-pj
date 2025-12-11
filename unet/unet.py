"""Plain UNet encoder-decoder."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .blocks import ConvBlock, UpConv

__all__ = ["UNet"]


class UNet(nn.Module):
    def __init__(
        self,
        img_ch: int = 3,
        output_ch: int = 1,
        up_mode: str = "bilinear",
        align_corners: Optional[bool] = False,
    ):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.up5 = UpConv(ch_in=1024, ch_out=512, mode=up_mode, align_corners=align_corners)
        self.up_conv5 = ConvBlock(ch_in=1024, ch_out=512)

        self.up4 = UpConv(ch_in=512, ch_out=256, mode=up_mode, align_corners=align_corners)
        self.up_conv4 = ConvBlock(ch_in=512, ch_out=256)

        self.up3 = UpConv(ch_in=256, ch_out=128, mode=up_mode, align_corners=align_corners)
        self.up_conv3 = ConvBlock(ch_in=256, ch_out=128)

        self.up2 = UpConv(ch_in=128, ch_out=64, mode=up_mode, align_corners=align_corners)
        self.up_conv2 = ConvBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoding path
        x1 = self.conv1(x)

        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)

        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)

        # decoding + concat path
        d5 = self.up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)

        return d1
