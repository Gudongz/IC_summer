"""Compare all registered Task 1 checkpoints on validation and sample images."""

from __future__ import annotations

import argparse
import csv
import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from config import settings
from models import SUPPORTED_TASK1_MODELS, build_task1_model
from task1_metrics import dice_coefficient, hausdorff_distances
from task1_visualization import save_prediction_comparison

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EPS = 1e-6


@dataclass(frozen=True)
class ValidationPair:
    image: Path
    mask: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Task 1 model checkpoints on one validation set.")
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_TASK1_MODELS, default=None)
    parser.add_argument("--validation-images", type=Path, default=settings.validation_input)
    parser.add_argument("--validation-masks", type=Path, default=settings.validation_ground_truth)
    parser.add_argument("--validation-manifest", type=Path, default=settings.validation_manifest, help="CSV defining the source-image validation IDs.")
    parser.add_argument("--sample-input", type=Path, default=settings.sample_input, help="Folder whose images will receive per-model masks and comparisons.")
    parser.add_argument("--sample-ground-truth", type=Path, default=settings.sample_ground_truth, help="Matching mask folder for sample comparison images.")
    parser.add_argument("--output-root", type=Path, default=settings.output_root)
    parser.add_argument("--batch-size", type=int, default=settings.evaluation_batch_size)
    parser.add_argument("--skip-samples", action="store_true", help="Evaluate validation metrics only; do not rewrite sample masks or comparison images.")
    parser.add_argument("--device", default=None, help="Defaults to the device configured in settings.json.")
    return parser.parse_args()


def image_paths(folder: Path) -> list[Path]:
    paths = sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise RuntimeError(f"No supported image files found in {folder}")
    return paths


def validation_pairs(image_dir: Path, mask_dir: Path, manifest_path: Path | None) -> list[ValidationPair]:
    requested_ids: set[str] | None = None
    if manifest_path is not None:
        with manifest_path.open(newline="", encoding="utf-8-sig") as file:
            requested_ids = {row["image_id"] for row in csv.DictReader(file) if row.get("split", "val") == "val"}
        if not requested_ids:
            raise RuntimeError(f"No validation IDs found in {manifest_path}")
    pairs = []
    for image_path in image_paths(image_dir):
        if requested_ids is not None and image_path.stem not in requested_ids:
            continue
        mask_path = mask_dir / f"{image_path.stem}_segmentation.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing validation mask for {image_path.name}: {mask_path}")
        pairs.append(ValidationPair(image_path, mask_path))
    return pairs


def mask_array(path: Path, image_size: int, strict_size: bool) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L")) > 127
    if strict_size and mask.shape != (image_size, image_size):
        raise ValueError(f"Expected prepared {image_size}x{image_size} mask: {path}")
    return mask


def load_checkpoint_model(model_name: str, checkpoint_path: Path, device: torch.device) -> nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint for {model_name} not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_name = checkpoint.get("model_name")
    if saved_name != model_name:
        raise ValueError(f"Checkpoint {checkpoint_path} contains {saved_name!r}, expected {model_name!r}.")
    model = build_task1_model(model_name, pretrained=False)
    if hasattr(model, "normalize_input"):
        model.normalize_input = bool(checkpoint.get("pretrained", checkpoint.get("pretrained_encoder", False)))
    if hasattr(model, "load_compatible_state_dict"):
        model.load_compatible_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def soft_dice(probabilities: np.ndarray, target: np.ndarray) -> float:
    """Confidence-aware Dice using unthresholded lesion probabilities."""
    target_float = target.astype(np.float32)
    return float((2 * (probabilities * target_float).sum() + EPS) / (probabilities.sum() + target_float.sum() + EPS))


def restore_logits(logits: Tensor, image: Image.Image, geometry: tuple[int, int, int, int]) -> Tensor:
    """Undo the evaluation resize/pad transform and return original-size logits."""
    left, top, resized_width, resized_height = geometry
    logits = logits[:, :, top : top + resized_height, left : left + resized_width]
    return F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)


def prepare_source_mask(path: Path, geometry: tuple[int, int, int, int]) -> np.ndarray:
    """Resize/pad an original GT mask into the model's 256px coordinate system."""
    left, top, resized_width, resized_height = geometry
    mask = Image.open(path).convert("L")
    resized = mask.resize((resized_width, resized_height), Image.Resampling.NEAREST)
    canvas = Image.new("L", (settings.image_size, settings.image_size), color=0)
    canvas.paste(resized, (left, top))
    return np.asarray(canvas) > 127


def evaluate_validation(model: nn.Module, pairs: list[ValidationPair], device: torch.device, batch_size: int) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    records: list[dict[str, float | str]] = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(pairs), batch_size), desc="Validation", unit="batch", leave=False, ascii=True):
            batch_pairs = pairs[start : start + batch_size]
            originals = [Image.open(pair.image).convert("RGB") for pair in batch_pairs]
            prepared = [prepare_sample_input(image) for image in originals]
            images = torch.stack([tensor for tensor, _ in prepared]).to(device)
            probabilities = torch.sigmoid(model(images)).squeeze(1).cpu().numpy()
            for pair, (_, geometry), probability in zip(batch_pairs, prepared, probabilities):
                target = prepare_source_mask(pair.mask, geometry)
                prediction = probability >= settings.prediction_threshold
                hd, hd95 = hausdorff_distances(prediction, target)
                records.append({"image": pair.image.name, "dice": dice_coefficient(prediction, target), "dice_confidence": soft_dice(probability, target), "hd": hd, "hd95": hd95})
    summary = {key: float(np.mean([record[key] for record in records])) for key in ("dice", "dice_confidence", "hd", "hd95")}
    return summary, records


def prepare_sample_input(image: Image.Image) -> tuple[Tensor, tuple[int, int, int, int]]:
    """Apply inference resize/pad preprocessing and return its reversible geometry."""
    image = image.convert("RGB")
    width, height = image.size
    scale = settings.image_size / max(width, height)
    resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
    canvas = Image.new("RGB", (settings.image_size, settings.image_size), color=(0, 0, 0))
    left, top = (settings.image_size - resized_width) // 2, (settings.image_size - resized_height) // 2
    canvas.paste(image.resize((resized_width, resized_height), Image.Resampling.BILINEAR), (left, top))
    tensor = torch.from_numpy(np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    return tensor, (left, top, resized_width, resized_height)


def predict_sample_outputs(model: nn.Module, image: Image.Image, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    tensor, (left, top, resized_width, resized_height) = prepare_sample_input(image)
    with torch.inference_mode():
        logits = model(tensor.unsqueeze(0).to(device))[:, :, top : top + resized_height, left : left + resized_width]
        logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
    probabilities = torch.sigmoid(logits[0, 0]).cpu().numpy()
    return probabilities, probabilities >= settings.prediction_threshold


def write_csv(path: Path, rows: list[dict[str, float | int | str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_sample_predictions(model: nn.Module, model_name: str, input_dir: Path, ground_truth_dir: Path | None, output_root: Path, device: torch.device) -> int:
    model_root = output_root / "sample_predictions" / model_name
    masks_dir = model_root / "masks"
    comparisons_dir = model_root / "comparisons"
    masks_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in tqdm(image_paths(input_dir), desc=f"Samples ({model_name})", unit="image", leave=False, ascii=True):
        image = Image.open(image_path).convert("RGB")
        probabilities, prediction = predict_sample_outputs(model, image, device)
        Image.fromarray(prediction.astype(np.uint8) * 255).save(masks_dir / f"{image_path.stem}_segmentation.png")
        target = None
        if ground_truth_dir is not None:
            target_path = ground_truth_dir / f"{image_path.stem}_segmentation.png"
            if target_path.is_file():
                target = mask_array(target_path, settings.image_size, strict_size=False)
                if target.shape != prediction.shape:
                    raise ValueError(f"Sample GT size mismatch for {image_path.name}: {target.shape} vs {prediction.shape}")
        save_prediction_comparison(
            image, target, prediction,
            comparisons_dir / f"{image_path.stem}_comparison.png",
            probabilities=probabilities,
        )
        count += 1
    return count


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device_name = args.device or (settings.device if settings.device == "cpu" or torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    pairs = validation_pairs(args.validation_images, args.validation_masks, args.validation_manifest)
    summaries: list[dict[str, float | int | str]] = []
    per_image_rows: list[dict[str, float | str]] = []

    if args.models is None:
        model_names = [name for name in SUPPORTED_TASK1_MODELS if settings.model_profiles[name]["checkpoint_path"].is_file()]
        skipped = [name for name in SUPPORTED_TASK1_MODELS if name not in model_names]
        if skipped:
            print(f"Skipping models without checkpoints: {', '.join(skipped)}")
        if not model_names:
            raise RuntimeError("No configured checkpoints found. Train at least one model or pass --models explicitly.")
    else:
        model_names = args.models

    model_progress = tqdm(model_names, desc="Models", unit="model", ascii=True)
    for model_name in model_progress:
        profile = settings.model_profiles[model_name]
        print(f"Evaluating {model_name}: {profile['checkpoint_path']}")
        model = load_checkpoint_model(model_name, profile["checkpoint_path"], device)
        summary, records = evaluate_validation(model, pairs, device, args.batch_size)
        summaries.append({"model": model_name, "checkpoint": str(profile["checkpoint_path"]), "samples": len(records), **summary})
        per_image_rows.extend({"model": model_name, **record} for record in records)
        exported = 0
        if not args.skip_samples:
            exported = export_sample_predictions(model, model_name, args.sample_input, args.sample_ground_truth, args.output_root, device)
        suffix = "sample export skipped" if args.skip_samples else f"exported {exported} sample images"
        print(f"  Dice={summary['dice']:.4f}, confidence Dice={summary['dice_confidence']:.4f}, HD={summary['hd']:.2f}px, HD95={summary['hd95']:.2f}px; {suffix}")
        model_progress.set_postfix(model=model_name, dice=f"{summary['dice']:.4f}")
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(args.output_root / "validation_model_comparison.csv", summaries, ["model", "checkpoint", "samples", "dice", "dice_confidence", "hd", "hd95"])
    write_csv(args.output_root / "validation_per_image_metrics.csv", per_image_rows, ["model", "image", "dice", "dice_confidence", "hd", "hd95"])
    print(f"Saved comparison reports to {args.output_root}")


if __name__ == "__main__":
    main()
