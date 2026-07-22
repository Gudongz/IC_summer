"""Task 1 segmentation model registry."""

from torch import nn

from .lb_unet import LBUNet
from .segformer_b1 import SegFormerB1
from .unet import UNet
from .uctransnet import UCTransNet

SUPPORTED_TASK1_MODELS = ("unet", "lb_unet", "segformer_b1", "uctransnet")


def build_task1_model(model_name: str, pretrained: bool = False) -> nn.Module:
    """Build one Task 1 binary-segmentation model by its stable config name."""
    if model_name == "unet":
        return UNet(in_channels=3, out_channels=1)
    if model_name == "lb_unet":
        return LBUNet(in_channels=3, out_channels=1)
    if model_name == "segformer_b1":
        return SegFormerB1(out_channels=1, pretrained=pretrained)
    if model_name == "uctransnet":
        return UCTransNet(in_channels=3, out_channels=1)
    raise ValueError(
        f"Unsupported Task 1 model {model_name!r}. Choose one of {SUPPORTED_TASK1_MODELS}. "
        "The retired 'resnet_unet' has been replaced by 'lb_unet'."
    )


__all__ = ["LBUNet", "SegFormerB1", "UCTransNet", "UNet", "SUPPORTED_TASK1_MODELS", "build_task1_model"]
