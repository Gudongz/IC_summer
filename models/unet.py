"""Vanilla 2D U-Net for binary lesion segmentation.

The model returns raw logits. Apply ``torch.sigmoid`` only for inference or
when calculating metrics; use ``BCEWithLogitsLoss`` during training.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    """Two 3x3 convolutions, each followed by batch normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Down(nn.Module):
    """One encoder stage: 2x2 max pooling followed by double convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(kernel_size=2), DoubleConv(in_channels, out_channels))

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Up(nn.Module):
    """One decoder stage with transposed-convolution upsampling and skip fusion."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        # This also supports input dimensions that are not divisible by 16.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((skip, x), dim=1))


class UNet(nn.Module):
    """Vanilla 2D U-Net for segmentation.

    Args:
        in_channels: Number of input channels. Use 3 for RGB dermoscopy images.
        out_channels: Number of output mask channels. Use 1 for binary lesion masks.
        base_channels: Width of the first encoder stage; later stages double it.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        if min(in_channels, out_channels, base_channels) < 1:
            raise ValueError("in_channels, out_channels, and base_channels must be positive.")

        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)

        self.up1 = Up(c * 16, c * 8, c * 8)
        self.up2 = Up(c * 8, c * 4, c * 4)
        self.up3 = Up(c * 4, c * 2, c * 2)
        self.up4 = Up(c * 2, c, c)
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Return per-pixel logits with spatial size matching ``x``."""
        input_size = x.shape[-2:]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits
