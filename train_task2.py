"""Train Task 2 five-attribute segmentation with fixed per-attribute losses."""

from __future__ import annotations

import json
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from config import load_task2_settings
from data_preprocessing import TASK2_ATTRIBUTES
from models import Task2ResNet34MultiDecoder, Task2ResNet50MultiDecoder, Task2SegFormerB1MultiDecoder
from task2_data import (
    OneVariantPerSourceSampler, Task2SegmentationDataset, build_task2_samples,
    restore_logits_to_canvas,
)
from training_common import (
    SegmentationTrainer, build_differential_adamw, keep_frozen_encoder_in_eval,
    resolve_device, set_encoder_trainable, set_random_seed,
)


def focal_tversky_per_attribute(
    logits: Tensor,
    targets: Tensor,
    alpha: float | Tensor | np.ndarray,
    beta: float | Tensor | np.ndarray,
    gamma: float | Tensor | np.ndarray,
    epsilon: float,
) -> Tensor:
    """Return one differentiable Focal Tversky loss for every attribute channel."""
    probabilities = torch.sigmoid(logits)
    dimensions = (0, 2, 3)
    true_positive = (probabilities * targets).sum(dim=dimensions)
    false_negative = ((1 - probabilities) * targets).sum(dim=dimensions)
    false_positive = (probabilities * (1 - targets)).sum(dim=dimensions)
    alpha_tensor = torch.as_tensor(alpha, dtype=logits.dtype, device=logits.device).reshape(-1)
    beta_tensor = torch.as_tensor(beta, dtype=logits.dtype, device=logits.device).reshape(-1)
    gamma_tensor = torch.as_tensor(gamma, dtype=logits.dtype, device=logits.device).reshape(-1)
    if alpha_tensor.numel() not in (1, logits.shape[1]) or beta_tensor.numel() not in (1, logits.shape[1]) or gamma_tensor.numel() not in (1, logits.shape[1]):
        raise ValueError("Focal Tversky parameters must be scalar or one value per output channel.")
    tversky = (true_positive + epsilon) / (true_positive + alpha_tensor * false_negative + beta_tensor * false_positive + epsilon)
    return (1 - tversky).pow(gamma_tensor)


def dice_per_attribute(logits: Tensor, targets: Tensor, threshold: float) -> tuple[Tensor, Tensor]:
    prediction = (torch.sigmoid(logits) >= threshold).float()
    intersection = (prediction * targets).sum(dim=(0, 2, 3))
    denominator = prediction.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
    return intersection, denominator


def precision_recall_counts(logits: Tensor, targets: Tensor, threshold: float) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-attribute true-positive, false-positive, and false-negative counts."""
    prediction = torch.sigmoid(logits) >= threshold
    target = targets.bool()
    dimensions = (0, 2, 3)
    true_positive = (prediction & target).sum(dim=dimensions)
    false_positive = (prediction & ~target).sum(dim=dimensions)
    false_negative = (~prediction & target).sum(dim=dimensions)
    return true_positive, false_positive, false_negative


def bce_per_attribute(logits: Tensor, targets: Tensor) -> Tensor:
    """Return one BCEWithLogits loss per attribute channel."""
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=(0, 2, 3))


class Task2Trainer(SegmentationTrainer):
    """Task-specific hooks plus the full Task 2 training loop."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.device = resolve_device(settings.device)
        profiles = [settings.attribute_loss.get(attribute, {}) for attribute in TASK2_ATTRIBUTES]
        self.attribute_weights = np.asarray([profile.get("weight", 1.0) for profile in profiles], dtype=np.float64)
        self.alpha = np.asarray([profile.get("alpha", settings.alpha) for profile in profiles], dtype=np.float64)
        self.beta = np.asarray([profile.get("beta", settings.beta) for profile in profiles], dtype=np.float64)
        self.gamma = np.asarray([profile.get("gamma", settings.gamma) for profile in profiles], dtype=np.float64)
        if np.any(self.attribute_weights < 0) or not np.any(self.attribute_weights > 0):
            raise ValueError("Task 2 attribute loss weights must be non-negative, with at least one active channel.")
        if np.any(self.alpha < 0) or np.any(self.beta < 0) or np.any(self.gamma <= 0):
            raise ValueError("Each Task 2 attribute requires alpha/beta >= 0 and gamma > 0.")

    def build_model(self) -> nn.Module:
        if self.settings.model_name in ("task2_resnet34_multidecoder", "task2_resnet34_multidecoder_roi"):
            return Task2ResNet34MultiDecoder(pretrained=self.settings.encoder_initialization == "imagenet")
        if self.settings.model_name in ("task2_resnet50_multidecoder", "task2_resnet50_multidecoder_roi"):
            return Task2ResNet50MultiDecoder(pretrained=self.settings.encoder_initialization == "imagenet")
        if self.settings.model_name in ("task2_segformer_b1_multidecoder", "task2_segformer_b1_multidecoder_roi"):
            return Task2SegFormerB1MultiDecoder(pretrained=self.settings.encoder_initialization == "imagenet")
        raise ValueError(f"Unsupported Task 2 model: {self.settings.model_name}")

    def build_loaders(self) -> tuple[DataLoader, DataLoader, int, int]:
        roi_kwargs = {
            "roi_mask_dir": self.settings.train_roi_mask if self.settings.roi_enabled else None,
            "roi_margin_ratio": self.settings.roi_margin_ratio,
            "roi_minimum_box_side": self.settings.roi_minimum_box_side,
            "full_canvas_size": self.settings.image_size,
        }
        train_samples = build_task2_samples(self.settings.train_input, self.settings.train_gt, self.settings.train_manifest, **roi_kwargs)
        val_samples = build_task2_samples(
            self.settings.val_input,
            self.settings.val_gt,
            roi_mask_dir=self.settings.val_roi_mask if self.settings.roi_enabled else None,
            roi_margin_ratio=self.settings.roi_margin_ratio,
            roi_minimum_box_side=self.settings.roi_minimum_box_side,
            full_canvas_size=self.settings.image_size,
        )
        train_dataset = Task2SegmentationDataset(train_samples, self.settings.image_size)
        val_dataset = Task2SegmentationDataset(val_samples, self.settings.image_size)
        if self.settings.variant_sampling == "one_per_source":
            sampler = OneVariantPerSourceSampler(train_samples, seed=self.settings.random_seed)
        elif self.settings.variant_sampling == "all_variants_weighted":
            sampler = WeightedRandomSampler([sample.sample_weight for sample in train_samples], len(train_samples), replacement=True)
        else:
            raise ValueError("task2.training.variant_sampling must be 'one_per_source' or 'all_variants_weighted'.")
        common = {"batch_size": self.settings.batch_size, "num_workers": self.settings.num_workers, "pin_memory": self.device.type == "cuda"}
        return DataLoader(train_dataset, sampler=sampler, **common), DataLoader(val_dataset, shuffle=False, **common), len(sampler), len(val_samples)

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        focal_losses = focal_tversky_per_attribute(logits, targets, self.alpha, self.beta, self.gamma, self.settings.epsilon)
        bce_losses = bce_per_attribute(logits, targets)
        losses = self.settings.focal_tversky_weight * focal_losses + self.settings.bce_weight * bce_losses
        weights = torch.as_tensor(self.attribute_weights, dtype=logits.dtype, device=logits.device)
        return (losses * weights).sum() / weights.sum()

    def compute_metrics(self, logits: Tensor, targets: Tensor) -> dict[str, float]:
        intersection, denominator = dice_per_attribute(logits, targets, self.settings.prediction_threshold)
        dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
        return {attribute: float(value) for attribute, value in zip(TASK2_ATTRIBUTES, dice)}

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "task": 2,
            "model_name": self.settings.model_name,
            "architecture": "rgb_shared_encoder_five_full_attribute_decoders",
            "roi": {
                "enabled": self.settings.roi_enabled,
                "margin_ratio": self.settings.roi_margin_ratio,
                "minimum_box_side": self.settings.roi_minimum_box_side,
            },
            "task1_checkpoint": str(self.settings.task1_checkpoint) if self.settings.task1_checkpoint is not None else None,
            "loss": {
                "bce_weight": self.settings.bce_weight,
                "focal_tversky_weight": self.settings.focal_tversky_weight,
                "epsilon": self.settings.epsilon,
                "attribute_loss": self.settings.attribute_loss,
            },
        }

    def _encoder_frozen(self, epoch: int) -> bool:
        return epoch <= self.settings.freeze_encoder_epochs

    def configure_encoder(self, model: nn.Module, epoch: int) -> None:
        frozen = self._encoder_frozen(epoch)
        previous = getattr(model, "_task2_encoder_frozen", None)
        model._task2_encoder_frozen = frozen
        set_encoder_trainable(model, not frozen)
        if previous != frozen:
            state = "frozen" if frozen else "unfrozen"
            print(f"  Encoder {state} for epoch {epoch}.")

    def run_epoch(self, model: nn.Module, loader: DataLoader, optimizer: AdamW | None, epoch: int, scaler: torch.amp.GradScaler) -> dict[str, object]:
        training = optimizer is not None
        model.train(training)
        if training:
            keep_frozen_encoder_in_eval(model, bool(getattr(model, "_task2_encoder_frozen", False)))
        losses_sum = np.zeros(len(TASK2_ATTRIBUTES), dtype=np.float64)
        intersection_sum = np.zeros(len(TASK2_ATTRIBUTES), dtype=np.float64)
        denominator_sum = np.zeros(len(TASK2_ATTRIBUTES), dtype=np.float64)
        true_positive_sum = np.zeros(len(TASK2_ATTRIBUTES), dtype=np.float64)
        false_positive_sum = np.zeros(len(TASK2_ATTRIBUTES), dtype=np.float64)
        false_negative_sum = np.zeros(len(TASK2_ATTRIBUTES), dtype=np.float64)
        total_loss = 0.0
        phase = "Train" if training else "Validation"
        progress = tqdm(loader, desc=f"Epoch {epoch:03d}/{self.settings.epochs} {phase}", unit="batch", leave=False)
        with torch.set_grad_enabled(training):
            for images, targets, rois, full_targets in progress:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                    logits = model(images)
                    focal_losses = focal_tversky_per_attribute(logits, targets, self.alpha, self.beta, self.gamma, self.settings.epsilon)
                    bce_losses = bce_per_attribute(logits, targets)
                    per_attribute_loss = self.settings.focal_tversky_weight * focal_losses + self.settings.bce_weight * bce_losses
                    weights = torch.as_tensor(self.attribute_weights, dtype=logits.dtype, device=self.device)
                    loss = (per_attribute_loss * weights).sum() / weights.sum()
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                if training:
                    metric_logits, metric_targets = logits, targets
                else:
                    metric_logits = restore_logits_to_canvas(logits, rois.to(self.device, non_blocking=True))
                    metric_targets = full_targets.to(self.device, non_blocking=True)
                intersections, denominators = dice_per_attribute(metric_logits, metric_targets, self.settings.prediction_threshold)
                true_positive, false_positive, false_negative = precision_recall_counts(
                    metric_logits, metric_targets, self.settings.prediction_threshold
                )
                losses_sum += per_attribute_loss.detach().float().cpu().numpy()
                intersection_sum += intersections.detach().cpu().numpy()
                denominator_sum += denominators.detach().cpu().numpy()
                true_positive_sum += true_positive.detach().cpu().numpy()
                false_positive_sum += false_positive.detach().cpu().numpy()
                false_negative_sum += false_negative.detach().cpu().numpy()
                total_loss += loss.item()
                dice_now = (2 * intersections + 1e-6) / (denominators + 1e-6)
                progress.set_postfix(loss=f"{loss.item():.4f}", mean_dice=f"{dice_now.mean().item():.4f}")
        dice = (2 * intersection_sum + 1e-6) / (denominator_sum + 1e-6)
        precision = (true_positive_sum + 1e-6) / (true_positive_sum + false_positive_sum + 1e-6)
        recall = (true_positive_sum + 1e-6) / (true_positive_sum + false_negative_sum + 1e-6)
        metrics: dict[str, object] = {"total_loss": total_loss / len(loader), "mean_dice": float(dice.mean())}
        for index, attribute in enumerate(TASK2_ATTRIBUTES):
            metrics[f"loss_{attribute}"] = float(losses_sum[index] / len(loader))
            metrics[f"dice_{attribute}"] = float(dice[index])
            metrics[f"precision_{attribute}"] = float(precision[index])
            metrics[f"recall_{attribute}"] = float(recall[index])
        return metrics

    def empty_history(self) -> dict[str, list[float]]:
        history = {"epoch": [], "train_total_loss": [], "val_total_loss": [], "train_mean_dice": [], "val_mean_dice": []}
        for attribute in TASK2_ATTRIBUTES:
            for prefix in ("train_loss", "val_loss", "train_dice", "val_dice", "train_precision", "val_precision", "train_recall", "val_recall"):
                history[f"{prefix}_{attribute}"] = []
        return history

    def save_plot(self, history: dict[str, list[float]]) -> None:
        self.settings.training_plot_path.parent.mkdir(parents=True, exist_ok=True)
        history_path = self.settings.training_plot_path.with_name("history.json")
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        subprocess.run([sys.executable, str(Path(__file__).with_name("plot_task2_training.py")), "--history", str(history_path), "--output", str(self.settings.training_plot_path)], check=False)

    def restore(self, model: nn.Module, optimizer: AdamW) -> tuple[int, float, dict[str, list[float]]]:
        if not self.settings.checkpoint_path.is_file():
            if self.settings.encoder_initialization == "task1":
                model.load_task1_encoder(self.settings.task1_checkpoint)
            else:
                print("Initialising the Task 2 encoder from ImageNet weights; no Task 1 checkpoint is loaded.")
            return 1, -1.0, self.empty_history()
        checkpoint = torch.load(self.settings.checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_name") != self.settings.model_name:
            raise ValueError(f"Checkpoint model {checkpoint.get('model_name')!r} does not match {self.settings.model_name!r}")
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError as exc:
                print(f"Checkpoint optimiser layout differs; using a fresh optimiser ({exc}).")
        if checkpoint.get("loss") != self.checkpoint_metadata()["loss"]:
            print("Task 2 loss settings changed; using the current fixed attribute weights from settings.json.")
        saved_history = checkpoint.get("history")
        history = self.empty_history()
        if isinstance(saved_history, dict) and isinstance(saved_history.get("epoch"), list):
            epochs = len(saved_history["epoch"])
            for key in history:
                if isinstance(saved_history.get(key), list) and len(saved_history[key]) == epochs:
                    history[key] = saved_history[key]
        return int(checkpoint.get("epoch", 0)) + 1, float(checkpoint.get("validation_mean_dice", -1.0)), history

    def latest_checkpoint_path(self) -> Path:
        return self.settings.checkpoint_path.with_name(
            f"{self.settings.checkpoint_path.stem}_latest{self.settings.checkpoint_path.suffix}"
        )

    def checkpoint_payload(
        self,
        model: nn.Module,
        optimizer: AdamW,
        epoch: int,
        validation_metrics: dict[str, object],
        history: dict[str, list[float]],
    ) -> dict[str, object]:
        return {
            **self.checkpoint_metadata(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "validation_mean_dice": validation_metrics["mean_dice"],
            "validation_total_loss": validation_metrics["total_loss"],
            "history": history,
        }

    def train(self) -> None:
        set_random_seed(self.settings.random_seed)
        train_loader, val_loader, train_count, val_count = self.build_loaders()
        model = self.build_model().to(self.device)
        optimizer = build_differential_adamw(model, self.settings.learning_rate, self.settings.encoder_learning_rate, self.settings.weight_decay, use_encoder_group=True)
        scaler = torch.amp.GradScaler(self.device.type, enabled=self.device.type == "cuda")
        self.settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        start_epoch, best_dice, history = self.restore(model, optimizer)
        if start_epoch > self.settings.epochs:
            print(f"Training already completed through epoch {self.settings.epochs}.")
            return
        print(f"Training {self.settings.model_name} on {self.device} ({train_count} train samples per epoch / {val_count} val samples), epochs {start_epoch}-{self.settings.epochs}")
        for epoch in range(start_epoch, self.settings.epochs + 1):
            self.configure_encoder(model, epoch)
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            train_metrics = self.run_epoch(model, train_loader, optimizer, epoch, scaler)
            val_metrics = self.run_epoch(model, val_loader, None, epoch, scaler)
            history["epoch"].append(epoch)
            for phase, metrics in (("train", train_metrics), ("val", val_metrics)):
                history[f"{phase}_total_loss"].append(metrics["total_loss"])
                history[f"{phase}_mean_dice"].append(metrics["mean_dice"])
                for attribute in TASK2_ATTRIBUTES:
                    history[f"{phase}_loss_{attribute}"].append(metrics[f"loss_{attribute}"])
                    history[f"{phase}_dice_{attribute}"].append(metrics[f"dice_{attribute}"])
                    history[f"{phase}_precision_{attribute}"].append(metrics[f"precision_{attribute}"])
                    history[f"{phase}_recall_{attribute}"].append(metrics[f"recall_{attribute}"])
            print(f"Epoch {epoch:03d}: train loss={train_metrics['total_loss']:.4f}, Dice={train_metrics['mean_dice']:.4f}; val loss={val_metrics['total_loss']:.4f}, Dice={val_metrics['mean_dice']:.4f}")
            for phase, metrics in (("Train", train_metrics), ("Val", val_metrics)):
                summary = " | ".join(
                    f"{attribute}: P={metrics[f'precision_{attribute}']:.4f}, R={metrics[f'recall_{attribute}']:.4f}"
                    for attribute in TASK2_ATTRIBUTES
                )
                print(f"  {phase} precision/recall — {summary}")
            if val_metrics["mean_dice"] > best_dice:
                best_dice = val_metrics["mean_dice"]
                torch.save(self.checkpoint_payload(model, optimizer, epoch, val_metrics, history), self.settings.checkpoint_path)
                print(f"  Saved best checkpoint to {self.settings.checkpoint_path}")
            torch.save(self.checkpoint_payload(model, optimizer, epoch, val_metrics, history), self.latest_checkpoint_path())
            self.save_plot(history)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Task 2 five-attribute model selected in settings.json."
    )
    parser.parse_args()
    Task2Trainer(load_task2_settings()).train()


if __name__ == "__main__":
    main()
