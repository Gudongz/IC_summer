"""RGB-only Task 2 models with one shared encoder and five full decoders."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvRefine(nn.Sequential):
    """Two local refinement convolutions used independently by every decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )


class ResNetUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.refine = ConvRefine(out_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat((x, skip), dim=1))


class ResNetAttributeDecoder(nn.Module):
    """A complete independent U-Net decoder for one dermoscopic attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.dec4 = ResNetUpBlock(512, 256, 256)
        self.dec3 = ResNetUpBlock(256, 128, 128)
        self.dec2 = ResNetUpBlock(128, 64, 64)
        self.dec1 = ResNetUpBlock(64, 64, 64)
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_refine = ConvRefine(32, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, stem: Tensor, layer1: Tensor, layer2: Tensor, layer3: Tensor, layer4: Tensor, input_size: tuple[int, int]) -> Tensor:
        x = self.dec4(layer4, layer3)
        x = self.dec3(x, layer2)
        x = self.dec2(x, layer1)
        x = self.dec1(x, stem)
        x = self.final_refine(self.final_up(x))
        return F.interpolate(self.head(x), size=input_size, mode="bilinear", align_corners=False)


class Task2ResNet34MultiDecoder(nn.Module):
    """Shared ImageNet/Task-1 ResNet34 encoder with five full U-Net decoders."""

    task1_model_name = "resnet34_unet"

    def __init__(self, pretrained: bool = True, num_attributes: int = 5) -> None:
        super().__init__()
        if num_attributes <= 0:
            raise ValueError("num_attributes must be positive")
        try:
            from torchvision.models import ResNet34_Weights, resnet34
        except ModuleNotFoundError as exc:
            raise ImportError("Task2ResNet34MultiDecoder requires torchvision.") from exc
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.normalize_input = pretrained
        self.num_attributes = num_attributes
        self.register_buffer("input_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.encoder = nn.ModuleDict({
            "stem": nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),
            "maxpool": backbone.maxpool, "layer1": backbone.layer1, "layer2": backbone.layer2,
            "layer3": backbone.layer3, "layer4": backbone.layer4,
        })
        self.attribute_decoders = nn.ModuleList(ResNetAttributeDecoder() for _ in range(num_attributes))

    def load_task1_encoder(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        if checkpoint.get("model_name") != self.task1_model_name:
            raise ValueError(f"Task2ResNet34MultiDecoder requires a {self.task1_model_name!r} checkpoint.")
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("Task 1 checkpoint has no model_state_dict.")
        encoder_state = {key.removeprefix("encoder."): value for key, value in state.items() if key.startswith("encoder.")}
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.normalize_input = bool(checkpoint.get("pretrained", checkpoint.get("pretrained_encoder", False)))

    def forward(self, x: Tensor, attribute_indices: Sequence[int] | None = None) -> Tensor:
        input_size = x.shape[-2:]
        if self.normalize_input:
            x = (x - self.input_mean) / self.input_std
        stem = self.encoder["stem"](x)
        layer1 = self.encoder["layer1"](self.encoder["maxpool"](stem))
        layer2 = self.encoder["layer2"](layer1)
        layer3 = self.encoder["layer3"](layer2)
        layer4 = self.encoder["layer4"](layer3)
        indices = range(self.num_attributes) if attribute_indices is None else tuple(attribute_indices)
        if not indices or any(index < 0 or index >= self.num_attributes for index in indices):
            raise ValueError("attribute_indices must contain valid Task 2 decoder indices.")
        return torch.cat(
            [self.attribute_decoders[index](stem, layer1, layer2, layer3, layer4, input_size) for index in indices], dim=1
        )


class ResNet50AttributeDecoder(nn.Module):
    """A complete independent U-Net decoder matched to ResNet50 feature widths."""

    def __init__(self) -> None:
        super().__init__()
        self.dec4 = ResNetUpBlock(2048, 1024, 512)
        self.dec3 = ResNetUpBlock(512, 512, 256)
        self.dec2 = ResNetUpBlock(256, 256, 128)
        self.dec1 = ResNetUpBlock(128, 64, 64)
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_refine = ConvRefine(32, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(
        self,
        stem: Tensor,
        layer1: Tensor,
        layer2: Tensor,
        layer3: Tensor,
        layer4: Tensor,
        input_size: tuple[int, int],
    ) -> Tensor:
        x = self.dec4(layer4, layer3)
        x = self.dec3(x, layer2)
        x = self.dec2(x, layer1)
        x = self.dec1(x, stem)
        x = self.final_refine(self.final_up(x))
        return F.interpolate(self.head(x), size=input_size, mode="bilinear", align_corners=False)


class Task2ResNet50MultiDecoder(nn.Module):
    """Shared ImageNet/Task-1 ResNet50 encoder with five full U-Net decoders."""

    task1_model_name = "resnet50_unet"

    def __init__(self, pretrained: bool = True, num_attributes: int = 5) -> None:
        super().__init__()
        if num_attributes <= 0:
            raise ValueError("num_attributes must be positive")
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except ModuleNotFoundError as exc:
            raise ImportError("Task2ResNet50MultiDecoder requires torchvision.") from exc
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.normalize_input = pretrained
        self.num_attributes = num_attributes
        self.register_buffer("input_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.encoder = nn.ModuleDict({
            "stem": nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),
            "maxpool": backbone.maxpool, "layer1": backbone.layer1, "layer2": backbone.layer2,
            "layer3": backbone.layer3, "layer4": backbone.layer4,
        })
        self.attribute_decoders = nn.ModuleList(ResNet50AttributeDecoder() for _ in range(num_attributes))

    def load_task1_encoder(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        if checkpoint.get("model_name") != self.task1_model_name:
            raise ValueError(f"Task2ResNet50MultiDecoder requires a {self.task1_model_name!r} checkpoint.")
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("Task 1 checkpoint has no model_state_dict.")
        encoder_state = {key.removeprefix("encoder."): value for key, value in state.items() if key.startswith("encoder.")}
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.normalize_input = bool(checkpoint.get("pretrained", checkpoint.get("pretrained_encoder", False)))

    def forward(self, x: Tensor, attribute_indices: Sequence[int] | None = None) -> Tensor:
        input_size = x.shape[-2:]
        if self.normalize_input:
            x = (x - self.input_mean) / self.input_std
        stem = self.encoder["stem"](x)
        layer1 = self.encoder["layer1"](self.encoder["maxpool"](stem))
        layer2 = self.encoder["layer2"](layer1)
        layer3 = self.encoder["layer3"](layer2)
        layer4 = self.encoder["layer4"](layer3)
        indices = range(self.num_attributes) if attribute_indices is None else tuple(attribute_indices)
        if not indices or any(index < 0 or index >= self.num_attributes for index in indices):
            raise ValueError("attribute_indices must contain valid Task 2 decoder indices.")
        return torch.cat(
            [self.attribute_decoders[index](stem, layer1, layer2, layer3, layer4, input_size) for index in indices], dim=1
        )


class SegFormerUpBlock(nn.Module):
    """Upsample and fuse one MiT skip feature for a single attribute decoder."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.refine = ConvRefine(out_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.refine(torch.cat((x, skip), dim=1))


class SegFormerAttributeDecoder(nn.Module):
    """Independent H/32→H/4 hierarchical decoder for one attribute."""

    def __init__(self, hidden_sizes: list[int]) -> None:
        super().__init__()
        c1, c2, c3, c4 = hidden_sizes
        self.deep = ConvRefine(c4, 256)
        self.up3 = SegFormerUpBlock(256, c3, 192)
        self.up2 = SegFormerUpBlock(192, c2, 128)
        self.up1 = SegFormerUpBlock(128, c1, 64)
        self.refine = ConvRefine(64, 64)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, features: tuple[Tensor, ...], input_size: tuple[int, int]) -> Tensor:
        f1, f2, f3, f4 = features
        x = self.deep(f4)
        x = self.up3(x, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)
        return F.interpolate(self.head(self.refine(x)), size=input_size, mode="bilinear", align_corners=False)


class Task2SegFormerB1MultiDecoder(nn.Module):
    """Shared MiT-B1 encoder with five complete attribute-specific decoders."""

    checkpoint_name = "nvidia/mit-b1"
    task1_model_name = "segformer_b1"

    def __init__(self, pretrained: bool = True, num_attributes: int = 5) -> None:
        super().__init__()
        if num_attributes <= 0:
            raise ValueError("num_attributes must be positive")
        try:
            from transformers import SegformerConfig, SegformerModel
        except ModuleNotFoundError as exc:
            raise ImportError("Task2SegFormerB1MultiDecoder requires transformers.") from exc
        if pretrained:
            self.encoder = SegformerModel.from_pretrained(self.checkpoint_name)
        else:
            self.encoder = SegformerModel(SegformerConfig(
                num_channels=3, depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512],
                num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1], mlp_ratios=[4, 4, 4, 4], drop_path_rate=0.1,
            ))
        self.normalize_input = pretrained
        self.num_attributes = num_attributes
        self.register_buffer("input_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.attribute_decoders = nn.ModuleList(
            SegFormerAttributeDecoder(list(self.encoder.config.hidden_sizes)) for _ in range(num_attributes)
        )

    def load_task1_encoder(self, checkpoint_path: str | Path) -> None:
        from .task2_segformer_b1 import Task2SegFormerB1

        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        if checkpoint.get("model_name") != self.task1_model_name:
            raise ValueError(f"Task2SegFormerB1MultiDecoder requires a {self.task1_model_name!r} checkpoint.")
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("Task 1 checkpoint has no model_state_dict.")
        expected_legacy = any(key.startswith("encoder.") for key in self.encoder.state_dict())
        expected_short = any(".attention.q_proj." in key for key in self.encoder.state_dict())
        encoder_state: dict[str, Tensor] = {}
        for key, value in state.items():
            if key.startswith("encoder."):
                key = Task2SegFormerB1._translate_encoder_key(key, to_legacy=expected_legacy)
                key = Task2SegFormerB1._translate_attention_projection_alias(key, use_short_names=expected_short)
                encoder_state[key.removeprefix("encoder.")] = value
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.normalize_input = bool(checkpoint.get("pretrained", checkpoint.get("pretrained_encoder", False)))

    def forward(self, x: Tensor, attribute_indices: Sequence[int] | None = None) -> Tensor:
        input_size = x.shape[-2:]
        if self.normalize_input:
            x = (x - self.input_mean) / self.input_std
        outputs = self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
        features = tuple(outputs.hidden_states)
        indices = range(self.num_attributes) if attribute_indices is None else tuple(attribute_indices)
        if not indices or any(index < 0 or index >= self.num_attributes for index in indices):
            raise ValueError("attribute_indices must contain valid Task 2 decoder indices.")
        return torch.cat([self.attribute_decoders[index](features, input_size) for index in indices], dim=1)
