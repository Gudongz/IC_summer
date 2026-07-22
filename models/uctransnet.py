"""UCTransNet-inspired U-Net with channel-wise Transformer skip fusion."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .unet import DoubleConv


class ChannelCrossTransformer(nn.Module):
    """CCT: exchange information between channel tokens from all encoder scales."""

    def __init__(self, channels: tuple[int, ...], token_grid: int = 8, embed_dim: int = 128) -> None:
        super().__init__()
        self.channels = channels
        self.token_grid = token_grid
        patch_dim = token_grid * token_grid
        self.to_token = nn.ModuleList([nn.Linear(patch_dim, embed_dim) for _ in channels])
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 2,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.to_channel = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in channels])

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        tokens: list[Tensor] = []
        counts: list[int] = []
        for feature, projection in zip(features, self.to_token):
            pooled = F.adaptive_avg_pool2d(feature, (self.token_grid, self.token_grid))
            tokens.append(projection(pooled.flatten(2)))
            counts.append(feature.shape[1])
        encoded = self.transformer(torch.cat(tokens, dim=1))
        return [projection(chunk).squeeze(-1) for projection, chunk in zip(self.to_channel, encoded.split(counts, dim=1))]


class ChannelCrossAttention(nn.Module):
    """CCA: decoder context selects useful channels from one encoder skip map."""

    def __init__(self, decoder_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.decoder_to_skip = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(decoder_channels, skip_channels, 1))
        self.fuse = DoubleConv(decoder_channels + skip_channels, out_channels)

    def forward(self, decoder: Tensor, skip: Tensor, ctrans_channel_logits: Tensor) -> Tensor:
        if decoder.shape[-2:] != skip.shape[-2:]:
            decoder = F.interpolate(decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        decoder_logits = self.decoder_to_skip(decoder).flatten(1)
        gate = torch.sigmoid(decoder_logits + ctrans_channel_logits).unsqueeze(-1).unsqueeze(-1)
        return self.fuse(torch.cat((decoder, skip * gate), dim=1))


class UCTransNet(nn.Module):
    """2D binary segmentation network using CCT/CCA instead of direct skips."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c * 8, c * 16)
        self.ctrans = ChannelCrossTransformer((c, c * 2, c * 4, c * 8))
        self.up4, self.dec4 = nn.ConvTranspose2d(c * 16, c * 8, 2, 2), ChannelCrossAttention(c * 8, c * 8, c * 8)
        self.up3, self.dec3 = nn.ConvTranspose2d(c * 8, c * 4, 2, 2), ChannelCrossAttention(c * 4, c * 4, c * 4)
        self.up2, self.dec2 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2), ChannelCrossAttention(c * 2, c * 2, c * 2)
        self.up1, self.dec1 = nn.ConvTranspose2d(c * 2, c, 2, 2), ChannelCrossAttention(c, c, c)
        self.head = nn.Conv2d(c, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        input_size = x.shape[-2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        c1, c2, c3, c4 = self.ctrans([e1, e2, e3, e4])
        d4 = self.dec4(self.up4(b), e4, c4)
        d3 = self.dec3(self.up3(d4), e3, c3)
        d2 = self.dec2(self.up2(d3), e2, c2)
        d1 = self.dec1(self.up1(d2), e1, c1)
        return F.interpolate(self.head(d1), size=input_size, mode="bilinear", align_corners=False)
