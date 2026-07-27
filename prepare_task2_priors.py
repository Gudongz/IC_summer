"""Create Task 1 lesion-mask priors for fixed augmented Task 2 splits.

Example:
    conda run -n IC_summer python prepare_task2_priors.py --model segformer_b1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm.auto import tqdm

from config import PROJECT_ROOT, settings
from infer_task1 import load_model, preprocess
from models import SUPPORTED_TASK1_MODELS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TASK2_SPLIT_PATHS = {
    "train": (
        PROJECT_ROOT / "data" / "prepared" / "task2" / "train" / "images",
        PROJECT_ROOT / "data" / "prepared" / "task2" / "train" / "mask",
    ),
    "val": (
        PROJECT_ROOT / "data" / "prepared" / "task2" / "val" / "images",
        PROJECT_ROOT / "data" / "prepared" / "task2" / "val" / "mask",
    ),
}


def checkpoint_for_model(model_name: str) -> Path:
    """Return the configured Task 1 checkpoint for ``model_name``."""
    try:
        return settings.model_profiles[model_name]["checkpoint_path"]
    except KeyError as exc:
        raise ValueError(f"No checkpoint profile is configured for {model_name!r}.") from exc


def validate_checkpoint_model(checkpoint_path: Path, model_name: str) -> None:
    """Reject accidentally pairing a requested model with another checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_model_name = checkpoint.get("model_name")
    if saved_model_name != model_name:
        raise ValueError(
            f"--model is {model_name!r}, but checkpoint {checkpoint_path} contains "
            f"{saved_model_name!r}. Choose a matching checkpoint or change --model."
        )


def restore_prediction(logits: torch.Tensor, original: Image.Image, threshold: float) -> np.ndarray:
    """Undo Task 1 aspect-ratio padding and return an original-size binary mask."""
    width, height = original.size
    scale = settings.image_size / max(width, height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left = (settings.image_size - resized_width) // 2
    top = (settings.image_size - resized_height) // 2
    logits = logits[:, :, top : top + resized_height, left : left + resized_width]
    logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
    probabilities = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()
    return probabilities >= threshold


def generate_split_priors(
    model: torch.nn.Module,
    device: torch.device,
    input_dir: Path,
    output_dir: Path,
    batch_size: int,
    threshold: float,
    split_name: str,
) -> int:
    """Generate priors for one prepared Task 2 split using one loaded model."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Augmented Task 2 image folder not found: {input_dir}")
    image_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise RuntimeError(f"No supported images found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        batches = range(0, len(image_paths), batch_size)
        progress = tqdm(batches, total=(len(image_paths) + batch_size - 1) // batch_size, desc=f"Task 2 {split_name} priors", unit="batch")
        for batch_start in progress:
            batch_paths = image_paths[batch_start : batch_start + batch_size]
            originals = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = torch.cat([preprocess(image) for image in originals], dim=0).to(device)
            batch_logits = model(inputs)
            for image_path, original, logits in zip(batch_paths, originals, batch_logits.split(1)):
                mask = restore_prediction(logits, original, threshold)
                Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / f"{image_path.stem}_segmentation.png")
            progress.set_postfix(images=min(batch_start + len(batch_paths), len(image_paths)))
    print(f"Saved {len(image_paths)} {split_name} priors to {output_dir} (threshold={threshold:g}).")
    return len(image_paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate binary Task 1 lesion priors for augmented Task 2 train/validation splits."
    )
    parser.add_argument(
        "--model", choices=SUPPORTED_TASK1_MODELS, required=True,
        help="Task 1 architecture whose configured checkpoint will be used.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Optional matching Task 1 checkpoint. Defaults to this model's settings.json profile.",
    )
    parser.add_argument("--split", choices=("train", "val", "both"), default="train", help="Prepared Task 2 split(s) to generate when --input/--output are omitted.")
    parser.add_argument("--input", type=Path, default=None, help="Optional custom image folder; requires --output.")
    parser.add_argument("--output", type=Path, default=None, help="Optional custom prior folder; requires --input.")
    parser.add_argument("--batch-size", type=int, default=settings.inference_batch_size)
    parser.add_argument("--threshold", type=float, default=settings.prediction_threshold)
    parser.add_argument("--device", default=None, help="cuda or cpu; defaults to settings.json with CUDA fallback.")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if (args.input is None) != (args.output is None):
        raise ValueError("--input and --output must be supplied together.")

    checkpoint_path = args.checkpoint or checkpoint_for_model(args.model)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Task 1 checkpoint not found: {checkpoint_path}")
    validate_checkpoint_model(checkpoint_path, args.model)

    requested_device = args.device or settings.device
    device = torch.device(requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    if args.input is not None:
        targets = [("custom", args.input, args.output)]
    else:
        split_names = ("train", "val") if args.split == "both" else (args.split,)
        targets = [(name, *TASK2_SPLIT_PATHS[name]) for name in split_names]
    for split_name, input_dir, output_dir in targets:
        generate_split_priors(model, device, input_dir, output_dir, args.batch_size, args.threshold, split_name)


if __name__ == "__main__":
    main()
