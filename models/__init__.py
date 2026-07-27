"""Task 1 segmentation model registry."""

from torch import nn

from .lb_unet import LBUNet
from .resnet_unet import ResNetUNet
from .segformer_b1 import SegFormerB1
from .task2_resnet34_unet import Task2ResNet34UNet
from .task2_segformer_b1 import Task2SegFormerB1
from .unet import UNet
from .uctransnet import UCTransNet

SUPPORTED_TASK1_MODELS = ("unet", "resnet34_unet", "resnet50_unet", "lb_unet", "segformer_b1", "uctransnet")


def build_task1_model(model_name: str, pretrained: bool = False) -> nn.Module:
    """Build one Task 1 binary-segmentation model by its stable config name."""
    if model_name == "unet":
        return UNet(in_channels=3, out_channels=1)
    if model_name == "resnet34_unet":
        return ResNetUNet(depth=34, out_channels=1, pretrained=pretrained)
    if model_name == "resnet50_unet":
        return ResNetUNet(depth=50, out_channels=1, pretrained=pretrained)
    if model_name == "lb_unet":
        return LBUNet(in_channels=3, out_channels=1)
    if model_name == "segformer_b1":
        return SegFormerB1(out_channels=1, pretrained=pretrained)
    if model_name == "uctransnet":
        return UCTransNet(in_channels=3, out_channels=1)
    raise ValueError(
        f"Unsupported Task 1 model {model_name!r}. Choose one of {SUPPORTED_TASK1_MODELS}."
    )


__all__ = [
    "LBUNet", "ResNetUNet", "SegFormerB1", "UCTransNet", "UNet",
    "Task2ResNet34UNet", "Task2SegFormerB1", "SUPPORTED_TASK1_MODELS", "build_task1_model",
]
