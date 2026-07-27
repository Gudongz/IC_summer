"""Train one registered Task 1 lesion-segmentation model."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
# This import loads OpenCV before PyTorch; see task1_data.py for why.
from task1_data import LesionSegmentationDataset, OneVariantPerSourceSampler, build_pairs
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import settings
from models import build_task1_model
from task1_metrics import hausdorff_distances
from training_common import build_differential_adamw, set_random_seed


def build_model(pretrained: bool) -> nn.Module:
    """Build a model, downloading encoder weights only for a fresh run."""
    return build_task1_model(settings.model_name, pretrained=pretrained)


def has_pretrained_encoder(model: nn.Module) -> bool:
    """Whether this run should stage-freeze a model's named encoder module."""
    return settings.pretrained and isinstance(getattr(model, "encoder", None), nn.Module)


def encoder_parameters(model: nn.Module) -> list[nn.Parameter]:
    if not has_pretrained_encoder(model):
        return []
    return list(model.encoder.parameters())


def build_optimizer(model: nn.Module) -> AdamW:
    """Build generic differential groups for any pretrained encoder model."""
    encoder_params = encoder_parameters(model)
    return build_differential_adamw(
        model, settings.learning_rate, settings.encoder_learning_rate,
        settings.weight_decay, use_encoder_group=bool(encoder_params),
    )


def report_model_parameters(model: nn.Module) -> None:
    """Print a compact parameter summary before training starts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = total - trainable
    print(
        "Model parameters: "
        f"total={total:,} ({total / 1_000_000:.2f}M), "
        f"trainable={trainable:,} ({trainable / 1_000_000:.2f}M), "
        f"frozen={frozen:,} ({frozen / 1_000_000:.2f}M)"
    )


def configure_encoder_for_epoch(model: nn.Module, epoch: int) -> None:
    """Apply the configured staged encoder fine-tuning policy."""
    if not has_pretrained_encoder(model):
        return
    trainable = epoch > settings.freeze_encoder_epochs
    frozen = not trainable
    previous_frozen = getattr(model, "_pipeline_encoder_frozen", None)
    model._pipeline_encoder_frozen = frozen
    # Retain this existing ResNet runtime flag without making the pipeline
    # dependent on its model-specific helper methods or parameter naming.
    if hasattr(model, "encoder_frozen"):
        model.encoder_frozen = frozen
    for parameter in model.encoder.parameters():
        parameter.requires_grad = trainable
    if frozen:
        model.encoder.eval()
    if previous_frozen != frozen:
        if trainable:
            print(
                f"  Encoder unfrozen for epoch {epoch}: pretrained encoder weights "
                "will now be updated (encoder LR="
                f"{settings.encoder_learning_rate:g})."
            )
        else:
            print(
                f"  Encoder frozen for epoch {epoch}: only decoder/head parameters "
                "will be updated."
            )


def keep_frozen_encoder_in_eval(model: nn.Module) -> None:
    """Protect frozen encoder BatchNorm statistics after model.train(True)."""
    if has_pretrained_encoder(model) and getattr(model, "_pipeline_encoder_frozen", False):
        model.encoder.eval()


def loss_weights_for_epoch(model: nn.Module, epoch: int) -> tuple[float, float]:
    """Return scheduled main BCE and Dice coefficients for one epoch."""
    bce_weight = settings.bce_weight
    if settings.bce_weight_decay_enabled and settings.bce_weight_decay_epochs > 0:
        start_epoch = settings.freeze_encoder_epochs + 1 if has_pretrained_encoder(model) else 1
        progress = min(max(epoch - start_epoch, 0) / settings.bce_weight_decay_epochs, 1.0)
        bce_weight += progress * (settings.bce_weight_decay_target - bce_weight)
    return bce_weight, settings.dice_weight


def training_strategy(model: nn.Module) -> dict[str, object]:
    """Persist the settings which determine optimization and loss behaviour."""
    return {
        "uses_pretrained_encoder": has_pretrained_encoder(model),
        "freeze_encoder_epochs": settings.freeze_encoder_epochs if has_pretrained_encoder(model) else 0,
        "decoder_learning_rate": settings.learning_rate,
        "encoder_learning_rate": settings.encoder_learning_rate if has_pretrained_encoder(model) else None,
        "loss": {
            "bce_weight": settings.bce_weight,
            "dice_weight": settings.dice_weight,
            "bce_weight_decay": {
                "enabled": settings.bce_weight_decay_enabled,
                "target_weight": settings.bce_weight_decay_target,
                "decay_epochs": settings.bce_weight_decay_epochs,
            },
        },
    }


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
    if training:
        keep_frozen_encoder_in_eval(model)
    bce = nn.BCEWithLogitsLoss()
    bce_weight, dice_weight = loss_weights_for_epoch(model, epoch)
    total_bce = total_dice_loss = total_main_loss = total_auxiliary_loss = total_loss = total_dice = 0.0
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
                main_loss = bce_weight * bce_value + dice_weight * dice_loss_value
                auxiliary_loss = masks.new_zeros(())
                if auxiliary_output is not None:
                    # PMA supervision intentionally keeps its own fixed weights.
                    auxiliary_loss = lb_auxiliary_loss(auxiliary_output, masks, bce)
                loss = main_loss + auxiliary_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_bce += bce_value.item()
            total_dice_loss += dice_loss_value.item()
            total_main_loss += main_loss.item()
            total_auxiliary_loss += auxiliary_loss.item()
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
                "wBCE": f"{bce_weight:.3f}",
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
        "main_loss": total_main_loss / len(loader),
        "auxiliary_loss": total_auxiliary_loss / len(loader),
        "total_loss": total_loss / len(loader),
        "dice": total_dice / len(loader),
        "bce_weight": bce_weight,
        "dice_weight": dice_weight,
    }
    if not training:
        metrics["hd"] = float(np.mean(hausdorff_values))
        metrics["hd95"] = float(np.mean(hd95_values))
    return metrics


def save_training_plot(history: dict[str, list[float]]) -> None:
    """Persist metrics and render curves in a process without PyTorch loaded.

    On Windows, importing Matplotlib beside CUDA PyTorch can load conflicting
    OpenMP runtimes. The plotting helper deliberately imports no PyTorch.
    """
    settings.training_plot_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = settings.training_plot_path.with_name("history.json")
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    helper = Path(__file__).with_name("plot_task1_training.py")
    result = subprocess.run(
        [sys.executable, str(helper), "--history", str(history_path), "--output", str(settings.training_plot_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        print(f"  Warning: curve rendering failed (exit {result.returncode}); training will continue.")
        if result.stderr:
            print(result.stderr.strip())


def empty_history() -> dict[str, list[float]]:
    """Create the metric history persisted in the best checkpoint."""
    return {
        key: []
        for key in (
            "epoch", "bce_weight", "dice_weight", "train_bce", "val_bce",
            "train_main_loss", "val_main_loss", "train_auxiliary_loss", "val_auxiliary_loss",
            "train_total_loss", "val_total_loss", "train_dice", "val_dice", "val_hd", "val_hd95",
        )
    }


def upgrade_history(history: object) -> dict[str, list[float]]:
    """Preserve existing curves while adding fields introduced by newer trainers."""
    if not isinstance(history, dict) or not isinstance(history.get("epoch"), list):
        print("Checkpoint has no compatible metric history; starting a new curve history.")
        return empty_history()
    epochs = len(history["epoch"])
    upgraded = empty_history()
    for key in upgraded:
        if isinstance(history.get(key), list) and len(history[key]) == epochs:
            upgraded[key] = history[key]
    # Old checkpoints used total loss as their only main loss. Their fixed
    # objective was BCE + Dice, so the historic coefficients are both one.
    for phase in ("train", "val"):
        main_key, aux_key, total_key = f"{phase}_main_loss", f"{phase}_auxiliary_loss", f"{phase}_total_loss"
        if not upgraded[main_key] and upgraded[total_key]:
            upgraded[main_key] = list(upgraded[total_key])
        if not upgraded[aux_key]:
            upgraded[aux_key] = [0.0] * epochs
    if not upgraded["bce_weight"]:
        upgraded["bce_weight"] = [1.0] * epochs
    if not upgraded["dice_weight"]:
        upgraded["dice_weight"] = [1.0] * epochs
    return upgraded


def restore_checkpoint(model: nn.Module, optimizer: AdamW) -> tuple[int, float, dict[str, list[float]]]:
    """Restore the configured best checkpoint, if present, for automatic resume."""
    if not settings.checkpoint_path.is_file():
        return 1, -1.0, empty_history()

    checkpoint = torch.load(settings.checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint.get("model_name")
    if checkpoint_model != settings.model_name:
        raise ValueError(
            f"Checkpoint model is {checkpoint_model!r}, but settings.model_name is "
            f"{settings.model_name!r}. Use a matching checkpoint or rename the old one."
        )
    # Fresh resume construction deliberately avoids downloading a pretrained
    # encoder. Restore the checkpoint's input-normalization policy explicitly.
    if hasattr(model, "normalize_input"):
        model.normalize_input = bool(checkpoint.get("pretrained", checkpoint.get("pretrained_encoder", False)))
    optimizer_state_is_compatible = True
    if hasattr(model, "load_compatible_state_dict"):
        optimizer_state_is_compatible = not model.load_compatible_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint and optimizer_state_is_compatible:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError as exc:
            print(f"Checkpoint optimizer layout differs from current training strategy; using a fresh optimizer ({exc}).")
    elif not optimizer_state_is_compatible:
        print("Checkpoint model weights were migrated for the installed transformers version; using a fresh optimizer state.")
    else:
        print("Checkpoint has no optimizer state; resuming model weights with a fresh optimizer.")

    saved_strategy = checkpoint.get("training_strategy")
    current_strategy = training_strategy(model)
    if saved_strategy is not None and saved_strategy != current_strategy:
        print("Checkpoint training strategy differs from current settings; continuing with current settings.json.")
    history = upgrade_history(checkpoint.get("history", empty_history()))
    next_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_dice = float(checkpoint.get("validation_dice", -1.0))
    print(f"Resumed checkpoint: {settings.checkpoint_path} (next epoch {next_epoch}, best Dice {best_dice:.4f})")
    return next_epoch, best_dice, history


def latest_checkpoint_path() -> Path:
    """Return the per-epoch checkpoint beside the configured best checkpoint."""
    return settings.checkpoint_path.with_name(
        f"{settings.checkpoint_path.stem}_latest{settings.checkpoint_path.suffix}"
    )


def checkpoint_payload(model: nn.Module, optimizer: AdamW, epoch: int, validation_metrics: dict[str, float], history: dict[str, list[float]]) -> dict[str, object]:
    return {
        "model_name": settings.model_name,
        "pretrained": settings.pretrained,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "validation_dice": validation_metrics["dice"],
        "validation_bce": validation_metrics["bce"],
        "validation_total_loss": validation_metrics["total_loss"],
        "validation_hd": validation_metrics["hd"],
        "validation_hd95": validation_metrics["hd95"],
        "history": history,
        "training_strategy": training_strategy(model),
    }


def main() -> None:
    set_random_seed(settings.random_seed)

    device_name = settings.device if settings.device == "cpu" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    training_pairs = build_pairs(settings.task1_train_input, settings.task1_train_gt)
    validation_pairs = build_pairs(settings.task1_val_input, settings.task1_val_gt)

    train_dataset = LesionSegmentationDataset(training_pairs, settings.image_size)
    if settings.variant_sampling == "one_per_source":
        train_sampler = OneVariantPerSourceSampler(training_pairs, seed=settings.random_seed)
        train_loader = DataLoader(
            train_dataset, batch_size=settings.batch_size, sampler=train_sampler,
            num_workers=settings.num_workers, pin_memory=device.type == "cuda",
        )
    elif settings.variant_sampling == "all_variants":
        train_loader = DataLoader(
            train_dataset, batch_size=settings.batch_size, shuffle=True,
            num_workers=settings.num_workers, pin_memory=device.type == "cuda",
        )
    else:
        raise ValueError("training.variant_sampling must be 'one_per_source' or 'all_variants'.")
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

    # A full local checkpoint already contains the encoder. Avoid a needless
    # Hugging Face download before resuming it, which also supports offline runs.
    fresh_run = not settings.checkpoint_path.is_file()
    model = build_model(pretrained=settings.pretrained and fresh_run).to(device)
    optimizer = build_optimizer(model)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start_epoch, best_dice, history = restore_checkpoint(model, optimizer)
    end_epoch = settings.epochs
    if start_epoch > end_epoch:
        print(f"Training is already complete through epoch {end_epoch}.")
        return

    # Apply the correct frozen/unfrozen state before reporting trainable weights.
    configure_encoder_for_epoch(model, start_epoch)
    report_model_parameters(model)
    print(f"Training {settings.model_name} on {device} ({len(train_loader.sampler)} train samples per epoch / {len(validation_pairs)} val samples), epochs {start_epoch}-{end_epoch}")
    for epoch in range(start_epoch, end_epoch + 1):
        configure_encoder_for_epoch(model, epoch)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, optimizer, device, epoch, end_epoch, scaler)
        val_metrics = run_epoch(model, val_loader, None, device, epoch, end_epoch, scaler)
        history["epoch"].append(epoch)
        history["bce_weight"].append(train_metrics["bce_weight"])
        history["dice_weight"].append(train_metrics["dice_weight"])
        for key in ("bce", "main_loss", "auxiliary_loss", "total_loss", "dice"):
            history[f"train_{key}"].append(train_metrics[key])
            history[f"val_{key}"].append(val_metrics[key])
        history["val_hd"].append(val_metrics["hd"])
        history["val_hd95"].append(val_metrics["hd95"])
        print(
            f"Epoch {epoch:03d}/{end_epoch} (BCE w={train_metrics['bce_weight']:.3f}, "
            f"Dice w={train_metrics['dice_weight']:.3f}): "
            f"train BCE={train_metrics['bce']:.4f}, main={train_metrics['main_loss']:.4f}, "
            f"aux={train_metrics['auxiliary_loss']:.4f}, total={train_metrics['total_loss']:.4f}, "
            f"Dice={train_metrics['dice']:.4f}; val BCE={val_metrics['bce']:.4f}, "
            f"main={val_metrics['main_loss']:.4f}, total={val_metrics['total_loss']:.4f}, "
            f"Dice={val_metrics['dice']:.4f}, HD={val_metrics['hd']:.2f}px, "
            f"HD95={val_metrics['hd95']:.2f}px"
        )
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            print("  Saving best checkpoint...")
            torch.save(checkpoint_payload(model, optimizer, epoch, val_metrics, history), settings.checkpoint_path)
            print(f"  Saved best checkpoint to {settings.checkpoint_path}")
        torch.save(checkpoint_payload(model, optimizer, epoch, val_metrics, history), latest_checkpoint_path())
        print("  Saving training curves...")
        save_training_plot(history)


if __name__ == "__main__":
    main()
