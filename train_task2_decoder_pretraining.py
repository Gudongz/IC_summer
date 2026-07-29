"""Stage-1 Task 2 training: freeze one shared encoder and train one attribute decoder."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import load_task2_settings
from data_preprocessing import TASK2_ATTRIBUTES
from models import Task2ResNet34MultiDecoder, Task2ResNet50MultiDecoder, Task2SegFormerB1MultiDecoder
from task2_data import (
    AttributeCurriculumSampler, OneVariantPerSourceSampler, Task2SegmentationDataset, build_task2_samples,
    manifest_attribute_presence, restore_logits_to_canvas,
)
from training_common import resolve_device, set_random_seed


def build_model(settings) -> nn.Module:
    if settings.model_name in ("task2_resnet34_multidecoder", "task2_resnet34_multidecoder_roi"):
        return Task2ResNet34MultiDecoder(pretrained=settings.encoder_initialization == "imagenet")
    if settings.model_name in ("task2_resnet50_multidecoder", "task2_resnet50_multidecoder_roi"):
        return Task2ResNet50MultiDecoder(pretrained=settings.encoder_initialization == "imagenet")
    if settings.model_name in ("task2_segformer_b1_multidecoder", "task2_segformer_b1_multidecoder_roi"):
        return Task2SegFormerB1MultiDecoder(pretrained=settings.encoder_initialization == "imagenet")
    raise ValueError(f"Unsupported Task 2 model: {settings.model_name}")


def attribute_loss(logits: Tensor, targets: Tensor, settings, attribute_index: int) -> Tensor:
    """Use BCE on all sampled ROIs and Focal Tversky only on positive ROIs."""
    profile = settings.attribute_loss[TASK2_ATTRIBUTES[attribute_index]]
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    positive = targets.flatten(1).any(dim=1)
    if not positive.any():
        return settings.bce_weight * bce
    probabilities = torch.sigmoid(logits[positive])
    positive_targets = targets[positive]
    dimensions = (0, 2, 3)
    true_positive = (probabilities * positive_targets).sum(dim=dimensions)
    false_negative = ((1 - probabilities) * positive_targets).sum(dim=dimensions)
    false_positive = (probabilities * (1 - positive_targets)).sum(dim=dimensions)
    alpha, beta, gamma = (float(profile[key]) for key in ("alpha", "beta", "gamma"))
    tversky = (true_positive + settings.epsilon) / (
        true_positive + alpha * false_negative + beta * false_positive + settings.epsilon
    )
    focal_tversky = (1 - tversky).pow(gamma).mean()
    return settings.bce_weight * bce + settings.focal_tversky_weight * focal_tversky


def binary_counts(logits: Tensor, targets: Tensor, threshold: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    prediction = torch.sigmoid(logits) >= threshold
    target = targets.bool()
    true_positive = (prediction & target).sum()
    false_positive = (prediction & ~target).sum()
    false_negative = (~prediction & target).sum()
    true_negative = (~prediction & ~target).sum()
    return true_positive, false_positive, false_negative, true_negative


def metrics_from_counts(
    true_positive: float,
    false_positive: float,
    false_negative: float,
    true_negative: float,
) -> dict[str, float]:
    epsilon = 1e-6
    precision = (true_positive + epsilon) / (true_positive + false_positive + epsilon)
    recall = (true_positive + epsilon) / (true_positive + false_negative + epsilon)
    dice = (2 * true_positive + epsilon) / (2 * true_positive + false_positive + false_negative + epsilon)
    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def freeze_for_attribute(model: nn.Module, attribute_index: int, training: bool) -> None:
    """Keep the Task-1-initialized encoder and unrelated decoders immutable."""
    model.encoder.eval()
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    for index, decoder in enumerate(model.attribute_decoders):
        target = index == attribute_index
        decoder.train(training and target)
        for parameter in decoder.parameters():
            parameter.requires_grad = target


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: AdamW | None, settings, attribute_index: int, epoch: int, scaler: torch.amp.GradScaler) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    freeze_for_attribute(model, attribute_index, training)
    total_loss = true_positive = false_positive = false_negative = true_negative = 0.0
    phase = "Train" if training else "Validation"
    progress = tqdm(loader, desc=f"{TASK2_ATTRIBUTES[attribute_index]} {phase} {epoch:03d}", unit="batch", leave=False)
    with torch.set_grad_enabled(training):
        for images, roi_targets, rois, full_targets in progress:
            images = images.to(settings.device, non_blocking=True)
            targets = roi_targets[:, attribute_index : attribute_index + 1].to(settings.device, non_blocking=True)
            with torch.autocast(device_type=settings.device.type, enabled=settings.device.type == "cuda"):
                logits = model(images, attribute_indices=(attribute_index,))
                loss = attribute_loss(logits, targets, settings, attribute_index)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                metric_logits, metric_targets = logits, targets
            else:
                metric_logits = restore_logits_to_canvas(logits, rois.to(settings.device, non_blocking=True))
                metric_targets = full_targets[:, attribute_index : attribute_index + 1].to(settings.device, non_blocking=True)
            tp, fp, fn, tn = binary_counts(metric_logits, metric_targets, settings.prediction_threshold)
            total_loss += loss.item()
            true_positive += float(tp.detach().cpu())
            false_positive += float(fp.detach().cpu())
            false_negative += float(fn.detach().cpu())
            true_negative += float(tn.detach().cpu())
            now = metrics_from_counts(true_positive, false_positive, false_negative, true_negative)
            progress.set_postfix(loss=f"{loss.item():.4f}", dice=f"{now['dice']:.4f}")
    return {"loss": total_loss / len(loader), **metrics_from_counts(true_positive, false_positive, false_negative, true_negative)}


def checkpoint_paths(settings, attribute: str) -> tuple[Path, Path, Path, Path, Path]:
    root = settings.decoder_pretraining["checkpoint_root"] / settings.model_name
    root.mkdir(parents=True, exist_ok=True)
    output_dir = settings.decoder_pretraining["output_root"] / settings.model_name / attribute
    return (
        root / f"{attribute}.pt",
        root / f"{attribute}_latest.pt",
        root / "shared_state.pt",
        output_dir / "history.json",
        output_dir / "curves.png",
    )


def empty_history() -> dict[str, list[float]]:
    names = (
        "epoch", "train_loss", "val_loss", "train_dice", "val_dice",
        "train_precision", "val_precision", "train_recall", "val_recall",
        "train_true_positive", "val_true_positive", "train_false_positive", "val_false_positive",
        "train_false_negative", "val_false_negative", "train_true_negative", "val_true_negative",
        "positive_ratio",
    )
    return {name: [] for name in names}


def payload(model: nn.Module, optimizer: AdamW, attribute: str, completed: list[str], epoch: int, best_dice: float, history: dict[str, list[float]], settings) -> dict[str, object]:
    return {
        "task": 2,
        "stage": "decoder_pretraining",
        "model_name": settings.model_name,
        "task1_checkpoint": str(settings.task1_checkpoint) if settings.task1_checkpoint is not None else None,
        "active_attribute": attribute,
        "completed_attributes": completed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_validation_dice": best_dice,
        "history": history,
        "decoder_pretraining": settings.decoder_pretraining,
    }


def restore(model: nn.Module, optimizer: AdamW, attribute: str, best_path: Path, latest_path: Path, shared_path: Path, settings) -> tuple[int, float, list[str], dict[str, list[float]]]:
    if settings.encoder_initialization == "task1":
        model.load_task1_encoder(settings.task1_checkpoint)
    initial_history = empty_history()
    # Stage-1 continuation always restarts from the validation-best decoder.
    # ``*_latest.pt`` is retained for audit/recovery only and is never loaded.
    checkpoint_path = best_path if best_path.is_file() else shared_path if shared_path.is_file() else None
    if checkpoint_path is None:
        if settings.encoder_initialization == "imagenet":
            print("Initialising the Task 2 encoder from ImageNet weights; no Task 1 checkpoint is loaded.")
        return 1, -1.0, [], initial_history
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != settings.model_name:
        raise ValueError(f"Checkpoint model {checkpoint.get('model_name')!r} does not match {settings.model_name!r}.")
    completed = list(checkpoint.get("completed_attributes", []))
    if checkpoint_path == shared_path and attribute in completed:
        raise ValueError(
            f"{attribute} is marked complete but its own best checkpoint is missing; "
            "the decoder cannot be safely resumed."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    if checkpoint_path == best_path:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        saved_history = checkpoint.get("history")
        history = initial_history
        if isinstance(saved_history, dict) and isinstance(saved_history.get("epoch"), list):
            epoch_count = len(saved_history["epoch"])
            for key in history:
                if isinstance(saved_history.get(key), list) and len(saved_history[key]) == epoch_count:
                    history[key] = saved_history[key]
        return int(checkpoint.get("epoch", 0)) + 1, float(checkpoint.get("best_validation_dice", -1.0)), completed, history
    return 1, -1.0, completed, initial_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the Task 2 encoder and pretrain one full attribute decoder.")
    parser.add_argument("--attribute", choices=TASK2_ATTRIBUTES, required=True)
    args = parser.parse_args()
    settings = load_task2_settings()
    settings.device = resolve_device(settings.device)
    set_random_seed(settings.random_seed)
    attribute_index = TASK2_ATTRIBUTES.index(args.attribute)
    config = settings.decoder_pretraining

    roi_kwargs = {
        "roi_mask_dir": settings.train_roi_mask if settings.roi_enabled else None,
        "roi_margin_ratio": settings.roi_margin_ratio,
        "roi_minimum_box_side": settings.roi_minimum_box_side,
        "full_canvas_size": settings.image_size,
    }
    train_samples = build_task2_samples(settings.train_input, settings.train_gt, settings.train_manifest, **roi_kwargs)
    val_samples = build_task2_samples(
        settings.val_input, settings.val_gt,
        roi_mask_dir=settings.val_roi_mask if settings.roi_enabled else None,
        roi_margin_ratio=settings.roi_margin_ratio,
        roi_minimum_box_side=settings.roi_minimum_box_side,
        full_canvas_size=settings.image_size,
    )
    source_presence = manifest_attribute_presence(settings.train_manifest, args.attribute)
    dynamic_sampling = bool(config.get("dynamic_sampling", True))
    if dynamic_sampling:
        sampler = AttributeCurriculumSampler(
            train_samples,
            source_presence,
            settings.random_seed,
            float(config["positive_ratio_start"]),
            float(config["positive_ratio_end"]),
            int(config["ratio_decay_epochs"]),
        )
        sampling_description = (
            f"dynamic positive ratio {float(config['positive_ratio_start']):.1%}"
            f"→{float(config['positive_ratio_end']):.1%}"
        )
    else:
        sampler = OneVariantPerSourceSampler(train_samples, seed=settings.random_seed)
        retained_sources = {sample.source_image_id for sample in train_samples}
        natural_positive_ratio = sum(source_presence[source] for source in retained_sources) / len(retained_sources)
        sampling_description = f"natural positive ratio {natural_positive_ratio:.1%}"
    common = {"batch_size": settings.batch_size, "num_workers": settings.num_workers, "pin_memory": settings.device.type == "cuda"}
    train_loader = DataLoader(Task2SegmentationDataset(train_samples, settings.image_size), sampler=sampler, **common)
    val_loader = DataLoader(Task2SegmentationDataset(val_samples, settings.image_size), shuffle=False, **common)

    model = build_model(settings).to(settings.device)
    freeze_for_attribute(model, attribute_index, training=True)
    optimizer = AdamW(model.attribute_decoders[attribute_index].parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scaler = torch.amp.GradScaler(settings.device.type, enabled=settings.device.type == "cuda")
    best_path, latest_path, shared_path, history_path, plot_path = checkpoint_paths(settings, args.attribute)
    start_epoch, best_dice, completed, history = restore(model, optimizer, args.attribute, best_path, latest_path, shared_path, settings)
    epochs = int(config["epochs_per_attribute"])
    if start_epoch > epochs:
        print(f"{args.attribute} has already completed epoch {start_epoch - 1}; settings target is epoch {epochs}.")
        return
    print(
        f"Stage 1 {args.attribute}: frozen encoder, {len(train_loader.sampler)} sampled source images/epoch, "
        f"{len(val_samples)} natural validation samples, {sampling_description}, epochs {start_epoch}-{epochs}."
    )
    for epoch in range(start_epoch, epochs + 1):
        sampler.set_epoch(epoch)
        current_positive_ratio = sampler.positive_ratio if dynamic_sampling else natural_positive_ratio
        train_metrics = run_epoch(model, train_loader, optimizer, settings, attribute_index, epoch, scaler)
        val_metrics = run_epoch(model, val_loader, None, settings, attribute_index, epoch, scaler)
        history["epoch"].append(epoch)
        history["positive_ratio"].append(current_positive_ratio)
        for phase, metrics in (("train", train_metrics), ("val", val_metrics)):
            for name, value in metrics.items():
                history[f"{phase}_{name}"].append(value)
        print(
            f"Epoch {epoch:03d}: positive={current_positive_ratio:.1%}; "
            f"train Dice={train_metrics['dice']:.4f}, P={train_metrics['precision']:.4f}, R={train_metrics['recall']:.4f}; "
            f"val Dice={val_metrics['dice']:.4f}, P={val_metrics['precision']:.4f}, R={val_metrics['recall']:.4f}"
        )
        state = payload(model, optimizer, args.attribute, completed, epoch, best_dice, history, settings)
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            state["best_validation_dice"] = best_dice
            torch.save(state, best_path)
            print(f"  Saved best {args.attribute} decoder checkpoint to {best_path}")
        torch.save(state, latest_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("plot_task2_decoder_pretraining.py")),
                "--history", str(history_path), "--output", str(plot_path), "--attribute", args.attribute,
            ],
            check=False,
        )

    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    completed = [name for name in completed if name != args.attribute] + [args.attribute]
    torch.save(
        {
            **best_checkpoint,
            "active_attribute": None,
            "completed_attributes": completed,
            "model_state_dict": model.state_dict(),
        },
        shared_path,
    )
    print(f"Completed {args.attribute}; shared sequential state saved to {shared_path}")


if __name__ == "__main__":
    main()
