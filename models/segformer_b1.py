"""SegFormer-B1 with a binary lesion segmentation decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SegFormerB1(nn.Module):
    """ImageNet-pretrained MiT-B1 encoder plus a lightweight MLP decoder."""

    checkpoint_name = "nvidia/mit-b1"

    def __init__(self, out_channels: int = 1, pretrained: bool = True, decoder_channels: int = 256) -> None:
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerModel
        except ModuleNotFoundError as exc:
            raise ImportError("SegFormer-B1 requires transformers. Install it with: python -m pip install transformers") from exc
        if pretrained:
            self.encoder = SegformerModel.from_pretrained(self.checkpoint_name)
        else:
            config = SegformerConfig(
                num_channels=3, depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512],
                num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1], mlp_ratios=[4, 4, 4, 4],
                drop_path_rate=0.1,
            )
            self.encoder = SegformerModel(config)
        self.register_buffer("input_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.normalize_input = pretrained
        hidden_sizes = self.encoder.config.hidden_sizes
        self.projections = nn.ModuleList([nn.Conv2d(channels, decoder_channels, 1) for channels in hidden_sizes])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels * len(hidden_sizes), decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
        )
        self.head = nn.Conv2d(decoder_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        input_size = x.shape[-2:]
        if self.normalize_input:
            x = (x - self.input_mean) / self.input_std
        outputs = self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
        features = outputs.hidden_states
        target_size = features[0].shape[-2:]
        decoded = [F.interpolate(proj(feature), size=target_size, mode="bilinear", align_corners=False) for proj, feature in zip(self.projections, features)]
        logits = self.head(self.fuse(torch.cat(decoded, dim=1)))
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
