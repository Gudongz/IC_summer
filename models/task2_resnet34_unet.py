"""Task 2 five-attribute segmentation with a ResNet-34 U-Net backbone."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TASK2_ATTRIBUTE_ORDER = (
    "pigment_network",
    "negative_network",
    "streaks",
    "milia_like_cyst",
    "globules",
)


class ConvNormReLU(nn.Sequential):
    """A decoder convolution that does not share Task 1 modules."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Task2DecoderBlock(nn.Module):
    """Upsample, fuse one skip feature, and refine it for Task 2."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.refine = ConvNormReLU(out_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat((x, skip), dim=1))


class AttributeAttentionHead(nn.Module):
    """One attribute-specific spatial attention head producing one raw logit map."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.attention = nn.Conv2d(channels, 1, kernel_size=1)
        self.output = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, shared_features: Tensor) -> Tensor:
        features = self.refine(shared_features)
        gate = torch.sigmoid(self.attention(features))
        return self.output(features * (1.0 + gate))


class LesionAwareFusion(nn.Module):
    """Use a predicted Task 1 lesion prior to residually gate one RGB feature map."""

    def __init__(self, feature_channels: int, prior_channels: int) -> None:
        super().__init__()
        self.prior_projection = nn.Sequential(
            nn.Conv2d(prior_channels, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )
        self.spatial_gate = nn.Conv2d(feature_channels * 2, 1, kernel_size=1)

    def forward(self, features: Tensor, prior_features: Tensor) -> Tensor:
        if prior_features.shape[-2:] != features.shape[-2:]:
            prior_features = F.interpolate(prior_features, size=features.shape[-2:], mode="bilinear", align_corners=False)
        prior_features = self.prior_projection(prior_features)
        gate = torch.sigmoid(self.spatial_gate(torch.cat((features, prior_features), dim=1)))
        return features * (1.0 + gate)


class ResNetMaskPyramid(nn.Module):
    """Lightweight encoder that aligns a binary lesion prior to ResNet scales."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))

    def forward(self, prior: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        stem = self.stem(prior)
        layer1 = self.layer1(stem)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        return stem, layer1, layer2, layer3, self.layer4(layer3)


class Task2ResNet34UNet(nn.Module):
    """Task 2 network with RGB features and a Task 1 lesion-prior branch.

    The forward result contains raw logits in ``TASK2_ATTRIBUTE_ORDER``.  Apply
    sigmoid independently per channel during loss computation or inference.
    """

    task1_model_name = "resnet34_unet"

    def __init__(self, pretrained: bool = True, num_attributes: int = 5) -> None:
        super().__init__()
        if num_attributes <= 0:
            raise ValueError("num_attributes must be positive")
        try:
            from torchvision.models import ResNet34_Weights, resnet34
        except ModuleNotFoundError as exc:
            raise ImportError("Task2ResNet34UNet requires torchvision.") from exc

        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.num_attributes = num_attributes
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
        self.mask_encoder = ResNetMaskPyramid()
        self.fuse_stem = LesionAwareFusion(64, 64)
        self.fuse_layer1 = LesionAwareFusion(64, 64)
        self.fuse_layer2 = LesionAwareFusion(128, 128)
        self.fuse_layer3 = LesionAwareFusion(256, 256)
        self.fuse_layer4 = LesionAwareFusion(512, 512)

        self.dec4 = Task2DecoderBlock(512, 256, 256)
        self.dec3 = Task2DecoderBlock(256, 128, 128)
        self.dec2 = Task2DecoderBlock(128, 64, 64)
        self.dec1 = Task2DecoderBlock(64, 64, 64)
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_refine = ConvNormReLU(32, 32)
        self.attribute_heads = nn.ModuleList(AttributeAttentionHead(32) for _ in range(num_attributes))

    def load_task1_encoder(self, checkpoint_path: str | Path) -> None:
        """Load only a matching Task 1 ResNet-34 encoder from a checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_name = checkpoint.get("model_name")
        if model_name != self.task1_model_name:
            raise ValueError(
                f"Checkpoint {checkpoint_path} contains {model_name!r}; "
                f"Task2ResNet34UNet requires {self.task1_model_name!r}."
            )
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint {checkpoint_path} has no model_state_dict.")
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        }
        if not encoder_state:
            raise ValueError(f"Checkpoint {checkpoint_path} contains no encoder weights.")
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.normalize_input = bool(checkpoint.get("pretrained", checkpoint.get("pretrained_encoder", False)))

    def forward(self, x: Tensor, lesion_prior: Tensor | None = None) -> Tensor:
        """Return five logits; ``lesion_prior`` is a binary ``B×1×H×W`` map."""
        input_size = x.shape[-2:]
        if lesion_prior is None:
            lesion_prior = x.new_zeros((x.shape[0], 1, *input_size))
        if lesion_prior.ndim != 4 or lesion_prior.shape[1] != 1:
            raise ValueError("lesion_prior must have shape B×1×H×W")
        if lesion_prior.shape[-2:] != input_size:
            lesion_prior = F.interpolate(lesion_prior, size=input_size, mode="nearest")
        if self.normalize_input:
            x = (x - self.input_mean) / self.input_std
        stem = self.encoder["stem"](x)
        layer1 = self.encoder["layer1"](self.encoder["maxpool"](stem))
        layer2 = self.encoder["layer2"](layer1)
        layer3 = self.encoder["layer3"](layer2)
        layer4 = self.encoder["layer4"](layer3)
        prior_stem, prior_layer1, prior_layer2, prior_layer3, prior_layer4 = self.mask_encoder(lesion_prior)
        stem = self.fuse_stem(stem, prior_stem)
        layer1 = self.fuse_layer1(layer1, prior_layer1)
        layer2 = self.fuse_layer2(layer2, prior_layer2)
        layer3 = self.fuse_layer3(layer3, prior_layer3)
        layer4 = self.fuse_layer4(layer4, prior_layer4)

        decoded = self.dec4(layer4, layer3)
        decoded = self.dec3(decoded, layer2)
        decoded = self.dec2(decoded, layer1)
        decoded = self.dec1(decoded, stem)
        decoded = self.final_refine(self.final_up(decoded))
        logits = torch.cat([head(decoded) for head in self.attribute_heads], dim=1)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
