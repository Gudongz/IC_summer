"""Final Task 1 evaluation: Dice, Hausdorff distance (HD), and HD95."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from config import settings
from task1_metrics import dice_coefficient, hausdorff_distances


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted Task 1 masks against ground truth.")
    parser.add_argument("--predictions", type=Path, default=settings.task1_output_folder)
    parser.add_argument("--masks", type=Path, default=settings.task1_gt)
    args = parser.parse_args()

    prediction_paths = sorted(args.predictions.glob("*_segmentation.png"))
    if not prediction_paths:
        raise RuntimeError(f"No prediction masks found in {args.predictions}")

    dices, hds, hd95s = [], [], []
    for prediction_path in prediction_paths:
        target_path = args.masks / prediction_path.name
        if not target_path.is_file():
            raise FileNotFoundError(f"Ground-truth mask missing for {prediction_path.name}")
        prediction = np.asarray(Image.open(prediction_path).convert("L")) > 0
        target = np.asarray(Image.open(target_path).convert("L")) > 0
        if prediction.shape != target.shape:
            raise ValueError(f"Size mismatch for {prediction_path.name}: {prediction.shape} vs {target.shape}")
        hd, hd95 = hausdorff_distances(prediction, target)
        dices.append(dice_coefficient(prediction, target))
        hds.append(hd)
        hd95s.append(hd95)

    print(f"Evaluated {len(prediction_paths)} images")
    print(f"Mean Dice: {np.mean(dices):.4f}")
    print(f"Mean HD:   {np.mean(hds):.2f}px")
    print(f"Mean HD95: {np.mean(hd95s):.2f}px")


if __name__ == "__main__":
    main()
