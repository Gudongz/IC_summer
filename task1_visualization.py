"""Pillow-only Task 1 prediction comparison images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def boundary(mask: np.ndarray) -> np.ndarray:
    """Return a one-pixel inner boundary without requiring OpenCV or SciPy."""
    padded = np.pad(mask, 1, constant_values=False)
    eroded = mask & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return mask & ~eroded


def overlay(image: Image.Image, target: np.ndarray | None, prediction: np.ndarray) -> Image.Image:
    array = np.asarray(image.convert("RGB")).copy()
    if target is not None:
        array[boundary(target)] = (0, 255, 80)
    array[boundary(prediction)] = (255, 130, 0)
    return Image.fromarray(array)


def save_prediction_comparison(image: Image.Image, target: np.ndarray | None, prediction: np.ndarray, output_path: Path) -> None:
    """Save one labelled original-image comparison with GT and prediction contours."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contour_image = overlay(image, target, prediction)
    legend_height = 28
    canvas = Image.new("RGB", (contour_image.width, contour_image.height + legend_height), "white")
    canvas.paste(contour_image, (0, 0))
    drawer = ImageDraw.Draw(canvas)
    drawer.line((8, contour_image.height + 10, 26, contour_image.height + 10), fill=(0, 255, 80), width=3)
    drawer.text((31, contour_image.height + 4), "GT (green)", fill="black")
    drawer.line((125, contour_image.height + 10, 143, contour_image.height + 10), fill=(255, 130, 0), width=3)
    drawer.text((148, contour_image.height + 4), "Prediction (orange)", fill="black")
    canvas.save(output_path)
