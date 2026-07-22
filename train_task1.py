"""Train one registered Task 1 lesion-segmentation model."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
# This import loads OpenCV before PyTorch; see task1_data.py for why.
from task1_data import LesionSegmentationDataset, build_pairs
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import settings
from models import build_task1_model
from task1_metrics import hausdorff_distances


def build_model() -> nn.Module:
    return build_task1_model(settings.model_name, pretrained=settings.pretrained)


def dice_loss(logits: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * intersection + eps) / (denominator + eps)).mean()


def dice_score(logits: Tensor, target: Tensor, threshold: float) -> Tensor:
    prediction = (torch.sigmoid(logits) >= threshold).float()
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    denominator = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def boundary_target(mask: Tensor) -> Tensor:
    """Create a one-pixel-ish binary contour target from a segmentation mask."""
    dilated = nn.functional.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = 1 - nn.functional.max_pool2d(1 - mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded > 0).float()


def lb_auxiliary_loss(auxiliary_output, masks: Tensor, bce: nn.Module) -> Tensor:
    """Region and boundary supervision for LB-UNet's PMA heads."""
    boundary = boundary_target(masks)
    total = masks.new_zeros(())
    for region_logits, edge_logits in zip(
        auxiliary_output.region_logits, auxiliary_output.boundary_logits
    ):
        region_target = nn.functional.interpolate(masks, size=region_logits.shape[-2:], mode="nearest")
        edge_target = nn.functional.interpolate(boundary, size=edge_logits.shape[-2:], mode="nearest")
        total = total + settings.lb_region_loss_weight * (bce(region_logits, region_target) + dice_loss(region_logits, region_target))
        total = total + settings.lb_boundary_loss_weight * bce(edge_logits, edge_target)
    return total / len(auxiliary_output.region_logits)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    bce = nn.BCEWithLogitsLoss()
    total_bce = total_dice_loss = total_loss = total_dice = 0.0
    hausdorff_values: list[float] = []
    hd95_values: list[float] = []
    phase = "Train" if training else "Validation"
    progress = tqdm(loader, desc=f"Epoch {epoch:03d}/{total_epochs} {phase}", unit="batch", leave=False)

    amp_enabled = device.type == "cuda"
    with torch.set_grad_enabled(training):
        for images, masks in progress:
            images, masks = images.to(device), masks.to(device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                auxiliary_output = model.forward_with_aux(images) if training and hasattr(model, "forward_with_aux") else None
                logits = auxiliary_output.logits if auxiliary_output is not None else model(images)
                bce_value = bce(logits, masks)
                dice_loss_value = dice_loss(logits, masks)
                loss = bce_value + dice_loss_value
                if auxiliary_output is not None:
                    loss = loss + lb_auxiliary_loss(auxiliary_output, masks, bce)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_bce += bce_value.item()
            total_dice_loss += dice_loss_value.item()
            total_loss += loss.item()
            total_dice += dice_score(logits, masks, settings.prediction_threshold).item()
            if not training:
                predictions = (torch.sigmoid(logits) >= settings.prediction_threshold).cpu().numpy().astype(bool)
                targets = masks.cpu().numpy().astype(bool)
                for prediction, target in zip(predictions[:, 0], targets[:, 0]):
                    hd, hd95 = hausdorff_distances(prediction, target)
                    hausdorff_values.append(hd)
                    hd95_values.append(hd95)
            postfix = {
                "BCE": f"{bce_value.item():.4f}",
                "DiceLoss": f"{dice_loss_value.item():.4f}",
                "Total": f"{loss.item():.4f}",
                "Dice": f"{dice_score(logits, masks, settings.prediction_threshold).item():.4f}",
            }
            if not training and hausdorff_values:
                postfix["HD"] = f"{np.mean(hausdorff_values):.1f}"
                postfix["HD95"] = f"{np.mean(hd95_values):.1f}"
            progress.set_postfix(postfix)

    metrics = {
        "bce": total_bce / len(loader),
        "dice_loss": total_dice_loss / len(loader),
        "total_loss": total_loss / len(loader),
        "dice": total_dice / len(loader),
    }
    if not training:
        metrics["hd"] = float(np.mean(hausdorff_values))
        metrics["hd95"] = float(np.mean(hd95_values))
    return metrics


def save_training_plot(history: dict[str, list[float]]) -> None:
    """Write all requested training and validation metrics into one PNG figure."""
    epochs = history["epoch"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(epochs, history["train_total_loss"], label="Train")
    axes[0, 0].plot(epochs, history["val_total_loss"], label="Validation")
    axes[0, 0].set(title="Total loss (BCE + Dice loss)", xlabel="Epoch", ylabel="Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history["train_bce"], label="Train BCE")
    axes[0, 1].plot(epochs, history["val_bce"], label="Validation BCE")
    axes[0, 1].set(title="Binary cross-entropy", xlabel="Epoch", ylabel="BCE loss")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, history["train_dice"], label="Train Dice")
    axes[1, 0].plot(epochs, history["val_dice"], label="Validation Dice")
    axes[1, 0].set(title="Dice coefficient", xlabel="Epoch", ylabel="Dice", ylim=(0, 1))
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, history["val_hd"], label="Validation HD")
    axes[1, 1].plot(epochs, history["val_hd95"], label="Validation HD95")
    axes[1, 1].set(title="Boundary distances", xlabel="Epoch", ylabel="Pixels")
    axes[1, 1].legend()

    settings.training_plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(settings.training_plot_path, dpi=160)
    plt.close(figure)


def empty_history() -> dict[str, list[float]]:
    """Create the metric history persisted in the best checkpoint."""
    return {key: [] for key in ("epoch", "train_bce", "val_bce", "train_total_loss", "val_total_loss", "train_dice", "val_dice", "val_hd", "val_hd95")}


def restore_checkpoint(model: nn.Module, optimizer: AdamW) -> tuple[int, float, dict[str, list[float]]]:
    """Restore the configured best checkpoint, if present, for automatic resume."""
    if not settings.best_checkpoint_path.is_file():
        return 1, -1.0, empty_history()

    checkpoint = torch.load(settings.best_checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint.get("model_name")
    if checkpoint_model != settings.model_name:
        raise ValueError(
            f"Checkpoint model is {checkpoint_model!r}, but settings.model_name is "
            f"{settings.model_name!r}. Use a matching checkpoint or rename the old one."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print("Checkpoint has no optimizer state; resuming model weights with a fresh optimizer.")

    history = checkpoint.get("history", empty_history())
    if not all(key in history for key in empty_history()):
        history = empty_history()
        print("Checkpoint has no compatible metric history; starting a new curve history.")
    next_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_dice = float(checkpoint.get("validation_dice", -1.0))
    print(f"Resumed checkpoint: {settings.best_checkpoint_path} (next epoch {next_epoch}, best Dice {best_dice:.4f})")
    return next_epoch, best_dice, history


def main() -> None:
    random.seed(settings.random_seed)
    np.random.seed(settings.random_seed)
    torch.manual_seed(settings.random_seed)

    device_name = settings.device if settings.device == "cpu" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    training_pairs = build_pairs(settings.task1_train_input, settings.task1_train_gt)
    validation_pairs = build_pairs(settings.task1_val_input, settings.task1_val_gt)

    train_loader = DataLoader(
        LesionSegmentationDataset(
            training_pairs,
            settings.image_size,
        ),
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        LesionSegmentationDataset(
            validation_pairs,
            settings.image_size,
        ),
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model().to(device)
    optimizer = AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, best_dice, history = restore_checkpoint(model, optimizer)
    end_epoch = start_epoch + settings.epochs - 1

    print(f"Training {settings.model_name} on {device} ({len(training_pairs)} train / {len(validation_pairs)} val samples), epochs {start_epoch}-{end_epoch}")
    for epoch in range(start_epoch, end_epoch + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, epoch, end_epoch, scaler)
        val_metrics = run_epoch(model, val_loader, None, device, epoch, end_epoch, scaler)
        history["epoch"].append(epoch)
        for key in ("bce", "total_loss", "dice"):
            history[f"train_{key}"].append(train_metrics[key])
            history[f"val_{key}"].append(val_metrics[key])
        history["val_hd"].append(val_metrics["hd"])
        history["val_hd95"].append(val_metrics["hd95"])
        save_training_plot(history)
        print(f"Epoch {epoch:03d}/{end_epoch}: train BCE={train_metrics['bce']:.4f}, loss={train_metrics['total_loss']:.4f}, Dice={train_metrics['dice']:.4f}; val BCE={val_metrics['bce']:.4f}, loss={val_metrics['total_loss']:.4f}, Dice={val_metrics['dice']:.4f}, HD={val_metrics['hd']:.2f}px, HD95={val_metrics['hd95']:.2f}px")
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save({"model_name": settings.model_name, "pretrained": settings.pretrained, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "validation_dice": val_metrics["dice"], "validation_bce": val_metrics["bce"], "validation_total_loss": val_metrics["total_loss"], "validation_hd": val_metrics["hd"], "validation_hd95": val_metrics["hd95"], "history": history}, settings.best_checkpoint_path)
            print(f"  Saved best checkpoint to {settings.best_checkpoint_path}")


if __name__ == "__main__":
    main()
