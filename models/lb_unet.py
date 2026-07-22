"""Lightweight boundary-assisted U-Net for binary lesion segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GroupShuffleAttention(nn.Module):
    """Grouped channel/spatial attention followed by channel shuffle."""

    def __init__(self, channels: int, groups: int = 4) -> None:
        super().__init__()
        if channels % groups:
            raise ValueError("channels must be divisible by groups")
        self.groups = groups
        group_channels = channels // groups
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(group_channels, group_channels, 1), nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(group_channels, group_channels, 3, padding=1, groups=group_channels), nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        chunks = x.chunk(self.groups, dim=1)
        attended = [chunk * self.channel_gate(chunk) * self.spatial_gate(chunk) for chunk in chunks]
        x = torch.cat(attended, dim=1)
        batch, channels, height, width = x.shape
        x = x.reshape(batch, self.groups, channels // self.groups, height, width)
        return x.transpose(1, 2).reshape(batch, channels, height, width)


class GSAConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            GroupShuffleAttention(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class PredictionMapAuxiliary(nn.Module):
    """PMA region/boundary prediction and fusion into a skip feature."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.region_head = nn.Conv2d(channels, 1, 1)
        self.boundary_head = nn.Conv2d(channels, 1, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels + 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        region = self.region_head(feature)
        boundary = self.boundary_head(feature)
        fused = self.fuse(torch.cat((feature, torch.sigmoid(region), torch.sigmoid(boundary)), dim=1))
        return fused, region, boundary


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.refine = GSAConv(out_channels + skip_channels, out_channels)
        self.pma = PredictionMapAuxiliary(out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.pma(self.refine(torch.cat((x, skip), dim=1)))


@dataclass
class LBUNetAuxiliaryOutput:
    logits: Tensor
    region_logits: list[Tensor]
    boundary_logits: list[Tensor]


class LBUNet(nn.Module):
    """GSA/PMA-based lightweight U-Net with boundary auxiliary supervision."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        if min(in_channels, out_channels, base_channels) < 1 or base_channels % 4:
            raise ValueError("base_channels must be positive and divisible by 4")
        c = base_channels
        self.enc1 = GSAConv(in_channels, c)
        self.enc2 = GSAConv(c, c * 2)
        self.enc3 = GSAConv(c * 2, c * 4)
        self.enc4 = GSAConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = GSAConv(c * 8, c * 16)
        self.dec4 = DecoderStage(c * 16, c * 8, c * 8)
        self.dec3 = DecoderStage(c * 8, c * 4, c * 4)
        self.dec2 = DecoderStage(c * 4, c * 2, c * 2)
        self.dec1 = DecoderStage(c * 2, c, c)
        self.head = nn.Conv2d(c, out_channels, 1)

    def forward_with_aux(self, x: Tensor) -> LBUNetAuxiliaryOutput:
        input_size = x.shape[-2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4, r4, b4 = self.dec4(b, e4)
        d3, r3, b3 = self.dec3(d4, e3)
        d2, r2, b2 = self.dec2(d3, e2)
        d1, r1, b1 = self.dec1(d2, e1)
        logits = self.head(d1)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return LBUNetAuxiliaryOutput(logits, [r4, r3, r2, r1], [b4, b3, b2, b1])

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_aux(x).logits
