"""Create lesion-mask predictions from a saved Task 1 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from config import settings
from models import build_task1_model


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_task1_model(checkpoint["model_name"], pretrained=False)
    # The state dict supplies pretrained weights; retain the corresponding
    # ImageNet input normalization without downloading the encoder again.
    if hasattr(model, "normalize_input"):
        model.normalize_input = bool(checkpoint.get("pretrained", checkpoint.get("pretrained_encoder", False)))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def preprocess(image: Image.Image) -> torch.Tensor:
    """Match validation preprocessing: aspect-ratio resize then centered padding."""
    image = image.convert("RGB")
    width, height = image.size
    scale = settings.image_size / max(width, height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (settings.image_size, settings.image_size), color=(0, 0, 0))
    left = (settings.image_size - resized_width) // 2
    top = (settings.image_size - resized_height) // 2
    canvas.paste(resized, (left, top))
    return torch.from_numpy(np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Compute an Otsu threshold for a finite floating-point score map."""
    values = values[np.isfinite(values)]
    if values.size == 0 or np.ptp(values) == 0:
        return float(values[0]) if values.size else 0.0
    histogram, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weight_background = np.cumsum(histogram)
    weight_foreground = histogram.sum() - weight_background
    sum_background = np.cumsum(histogram * centers)
    sum_foreground = sum_background[-1] - sum_background
    mean_background = np.divide(sum_background, weight_background, out=np.zeros_like(sum_background, dtype=float), where=weight_background > 0)
    mean_foreground = np.divide(sum_foreground, weight_foreground, out=np.zeros_like(sum_foreground, dtype=float), where=weight_foreground > 0)
    between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
    return float(centers[np.argmax(between_class_variance)])


def prediction_from_logits(logits: np.ndarray) -> np.ndarray:
    """Threshold a single-image logit map, optionally using Gaussian + Otsu."""
    if settings.use_postprocessing:
        smoothed_logits = gaussian_filter(logits, sigma=settings.postprocess_gaussian_sigma)
        return smoothed_logits > otsu_threshold(smoothed_logits)
    probabilities = 1 / (1 + np.exp(-logits))
    return probabilities >= settings.prediction_threshold


def save_comparison_figure(
    image: Image.Image,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
) -> None:
    """Save a 2x2 visual comparison for one labelled validation image."""
    image_array = np.asarray(image.convert("RGB"))
    figure, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

    axes[0, 0].imshow(image_array)
    axes[0, 0].set_title("Original image")

    axes[0, 1].imshow(image_array)
    axes[0, 1].contour(ground_truth, levels=[0.5], colors="lime", linewidths=1.5)
    axes[0, 1].set_title("Original + GT contour")

    axes[1, 0].imshow(image_array)
    axes[1, 0].contour(prediction, levels=[0.5], colors="orange", linewidths=1.5)
    axes[1, 0].set_title("Original + predicted contour")

    axes[1, 1].imshow(image_array)
    axes[1, 1].contour(ground_truth, levels=[0.5], colors="lime", linewidths=1.5)
    axes[1, 1].contour(prediction, levels=[0.5], colors="red", linewidths=1.5)
    axes[1, 1].set_title("Contour comparison: GT green / prediction red")

    for axis in axes.flat:
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=settings.task1_input, help="Folder of source RGB images.")
    parser.add_argument("--output", type=Path, default=settings.task1_output_folder, help="Folder for predicted PNG masks.")
    parser.add_argument("--checkpoint", type=Path, default=settings.checkpoint_path)
    parser.add_argument("--ground-truth", type=Path, default=settings.task1_gt, help="Optional Task 1 GT-mask folder, used only for visual comparisons.")
    parser.add_argument("--save-comparisons", action=argparse.BooleanOptionalAction, default=settings.save_comparisons, help="Save a 2x2 original/GT/prediction comparison image per sample; requires --ground-truth.")
    args = parser.parse_args()

    if args.save_comparisons and args.ground_truth is None:
        raise ValueError("--save-comparisons requires --ground-truth <task1_gt folder>.")

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}. Run train_task1.py first.")
    device_name = settings.device if settings.device == "cpu" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = load_model(args.checkpoint, device)
    args.output.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in args.input.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        raise RuntimeError(f"No images found in {args.input}")
    with torch.inference_mode():
        total_batches = (len(image_paths) + settings.inference_batch_size - 1) // settings.inference_batch_size
        progress = tqdm(range(0, len(image_paths), settings.inference_batch_size), total=total_batches, desc="Inference", unit="batch")
        for batch_start in progress:
            batch_paths = image_paths[batch_start : batch_start + settings.inference_batch_size]
            originals = [Image.open(image_path).convert("RGB") for image_path in batch_paths]
            batch_inputs = torch.cat([preprocess(original) for original in originals], dim=0).to(device)
            batch_logits = model(batch_inputs)

            for image_path, original, logits in zip(batch_paths, originals, batch_logits.split(1)):
                width, height = original.size
                scale = settings.image_size / max(width, height)
                resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
                left, top = (settings.image_size - resized_width) // 2, (settings.image_size - resized_height) // 2
                logits = logits[:, :, top : top + resized_height, left : left + resized_width]
                logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
                prediction = prediction_from_logits(logits[0, 0].cpu().numpy())
                Image.fromarray(prediction.astype(np.uint8) * 255).save(args.output / f"{image_path.stem}_segmentation.png")
                if args.save_comparisons:
                    target_path = args.ground_truth / f"{image_path.stem}_segmentation.png"
                    if not target_path.is_file():
                        raise FileNotFoundError(f"Ground-truth mask missing: {target_path}")
                    ground_truth = np.asarray(Image.open(target_path).convert("L")) > 0
                    if ground_truth.shape != prediction.shape:
                        raise ValueError(f"Size mismatch for {image_path.name}: GT {ground_truth.shape}, prediction {prediction.shape}")
                    save_comparison_figure(original, ground_truth, prediction, args.output / "comparisons" / f"{image_path.stem}_comparison.png")
            progress.set_postfix(images=min(batch_start + len(batch_paths), len(image_paths)))
    print(f"Saved {len(image_paths)} masks to {args.output}")


if __name__ == "__main__":
    main()
