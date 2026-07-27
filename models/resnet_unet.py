"""ImageNet-pretrained ResNet encoders with U-Net-style decoders."""

from __future__ import annotations

from itertools import chain

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .unet import DoubleConv


class DecoderBlock(nn.Module):
    """Upsample, concatenate one encoder skip, then refine the fused feature."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.refine = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat((x, skip), dim=1))


class ResNetUNet(nn.Module):
    """ResNet-34 or ResNet-50 encoder with a five-level U-Net decoder."""

    def __init__(self, depth: int, out_channels: int = 1, pretrained: bool = True) -> None:
        super().__init__()
        if depth not in (34, 50):
            raise ValueError("depth must be 34 or 50")
        try:
            from torchvision.models import ResNet34_Weights, ResNet50_Weights, resnet34, resnet50
        except ModuleNotFoundError as exc:
            raise ImportError("ResNet U-Net requires torchvision. Install a PyTorch-compatible torchvision build.") from exc

        if depth == 34:
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            channels = (64, 64, 128, 256, 512)
            decoder_channels = (256, 128, 64, 64, 32)
        else:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            channels = (64, 256, 512, 1024, 2048)
            decoder_channels = (512, 256, 128, 64, 32)

        self.depth = depth
        self.normalize_input = pretrained
        self.register_buffer("input_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.encoder = nn.ModuleDict(
            {
                "stem": nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),
                "maxpool": backbone.maxpool,
                "layer1": backbone.layer1,
                "layer2": backbone.layer2,
                "layer3": backbone.layer3,
                "layer4": backbone.layer4,
            }
        )

        stem_channels, layer1_channels, layer2_channels, layer3_channels, layer4_channels = channels
        d4, d3, d2, d1, final_channels = decoder_channels
        self.dec4 = DecoderBlock(layer4_channels, layer3_channels, d4)
        self.dec3 = DecoderBlock(d4, layer2_channels, d3)
        self.dec2 = DecoderBlock(d3, layer1_channels, d2)
        self.dec1 = DecoderBlock(d2, stem_channels, d1)
        self.final_up = nn.ConvTranspose2d(d1, final_channels, kernel_size=2, stride=2)
        self.final_refine = DoubleConv(final_channels, final_channels)
        self.head = nn.Conv2d(final_channels, out_channels, kernel_size=1)
        self.encoder_frozen = False

    def encoder_parameters(self):
        return self.encoder.parameters()

    def decoder_parameters(self):
        return chain(
            self.dec4.parameters(), self.dec3.parameters(), self.dec2.parameters(), self.dec1.parameters(),
            self.final_up.parameters(), self.final_refine.parameters(), self.head.parameters(),
        )

    def optimizer_parameter_groups(self, decoder_learning_rate: float, encoder_learning_rate: float):
        return [
            {"params": list(self.decoder_parameters()), "lr": decoder_learning_rate},
            {"params": list(self.encoder_parameters()), "lr": encoder_learning_rate},
        ]

    def set_encoder_trainable(self, trainable: bool) -> bool:
        """Set encoder gradient and BatchNorm behaviour; return whether state changed."""
        frozen = not trainable
        changed = self.encoder_frozen != frozen
        self.encoder_frozen = frozen
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable
        if frozen:
            self.encoder.eval()
        return changed

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        input_size = x.shape[-2:]
        if self.normalize_input:
            x = (x - self.input_mean) / self.input_std

        stem = self.encoder["stem"](x)
        layer1 = self.encoder["layer1"](self.encoder["maxpool"](stem))
        layer2 = self.encoder["layer2"](layer1)
        layer3 = self.encoder["layer3"](layer2)
        layer4 = self.encoder["layer4"](layer3)

        x = self.dec4(layer4, layer3)
        x = self.dec3(x, layer2)
        x = self.dec2(x, layer1)
        x = self.dec1(x, stem)
        x = self.final_refine(self.final_up(x))
        logits = self.head(x)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
