"""Reusable training primitives shared by Task 1 and Task 2 trainers."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    return torch.device(requested if requested == "cpu" or torch.cuda.is_available() else "cpu")


def build_differential_adamw(
    model: nn.Module,
    decoder_learning_rate: float,
    encoder_learning_rate: float | None,
    weight_decay: float,
    use_encoder_group: bool,
) -> AdamW:
    """Use a lower learning rate for an optionally pretrained ``model.encoder``."""
    encoder = getattr(model, "encoder", None)
    if use_encoder_group and isinstance(encoder, nn.Module):
        if encoder_learning_rate is None:
            raise ValueError("A pretrained encoder requires encoder_learning_rate.")
        encoder_params = list(encoder.parameters())
        encoder_ids = {id(parameter) for parameter in encoder_params}
        decoder_params = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
        return AdamW(
            [{"params": decoder_params, "lr": decoder_learning_rate}, {"params": encoder_params, "lr": encoder_learning_rate}],
            weight_decay=weight_decay,
        )
    return AdamW(model.parameters(), lr=decoder_learning_rate, weight_decay=weight_decay)


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    """Set encoder gradients and protect BatchNorm statistics while frozen."""
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, nn.Module):
        return
    for parameter in encoder.parameters():
        parameter.requires_grad = trainable
    if hasattr(model, "encoder_frozen"):
        model.encoder_frozen = not trainable
    if not trainable:
        encoder.eval()


def keep_frozen_encoder_in_eval(model: nn.Module, frozen: bool) -> None:
    encoder = getattr(model, "encoder", None)
    if frozen and isinstance(encoder, nn.Module):
        encoder.eval()


class SegmentationTrainer(ABC):
    """Small hook-based contract for task-specific segmentation trainers.

    The concrete trainer owns data and losses; shared primitives above keep
    optimiser, encoder-fine-tuning and device behaviour consistent.
    """

    @abstractmethod
    def build_model(self) -> nn.Module: ...

    @abstractmethod
    def build_loaders(self): ...

    @abstractmethod
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def compute_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]: ...

    @abstractmethod
    def checkpoint_metadata(self) -> dict[str, object]: ...
