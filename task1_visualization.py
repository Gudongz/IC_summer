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


def thick_boundary(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """Expand an inner boundary to a visibly thicker contour."""
    contour = boundary(mask)
    for _ in range(width - 1):
        padded = np.pad(contour, 1, constant_values=False)
        contour = (
            contour
            | padded[:-2, 1:-1] | padded[2:, 1:-1]
            | padded[1:-1, :-2] | padded[1:-1, 2:]
        )
    return contour


def overlay(image: Image.Image, target: np.ndarray | None, prediction: np.ndarray) -> Image.Image:
    array = np.asarray(image.convert("RGB")).copy()
    if target is not None:
        array[boundary(target)] = (0, 255, 80)
    array[boundary(prediction)] = (255, 130, 0)
    return Image.fromarray(array)


def contour_comparison_panel(image: Image.Image, target: np.ndarray | None, prediction: np.ndarray, legend_height: int) -> Image.Image:
    """Create the original-image panel with GT and binary-prediction contours."""
    contour_image = overlay(image, target, prediction)
    canvas = Image.new("RGB", (contour_image.width, contour_image.height + legend_height), "white")
    canvas.paste(contour_image, (0, 0))
    drawer = ImageDraw.Draw(canvas)
    drawer.line((8, contour_image.height + 10, 26, contour_image.height + 10), fill=(0, 255, 80), width=3)
    drawer.text((31, contour_image.height + 4), "GT (green)", fill="black")
    drawer.line((125, contour_image.height + 10, 143, contour_image.height + 10), fill=(255, 130, 0), width=3)
    drawer.text((148, contour_image.height + 4), "Prediction (orange)", fill="black")
    return canvas


def save_prediction_comparison(
    image: Image.Image,
    target: np.ndarray | None,
    prediction: np.ndarray,
    output_path: Path,
    probabilities: np.ndarray | None = None,
) -> None:
    """Save contours alone, or a side-by-side contour and probability comparison."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if probabilities is None:
        contour_comparison_panel(image, target, prediction, legend_height=28).save(output_path)
        return

    probability_panel = probability_region_panel(probabilities, target)
    contour_panel = contour_comparison_panel(image, target, prediction, legend_height=probability_panel.height - image.height)
    combined = Image.new("RGB", (contour_panel.width + probability_panel.width, probability_panel.height), "white")
    combined.paste(contour_panel, (0, 0))
    combined.paste(probability_panel, (contour_panel.width, 0))
    combined.save(output_path)


def probability_region_panel(probabilities: np.ndarray, target: np.ndarray | None) -> Image.Image:
    """Create a colour-coded probability panel with confidence interval labels.

    The bins are ``<0.3``, ``0.3-0.4``, ``0.4-0.5``, ``0.5-0.6`` and
    ``>=0.6``. A thick black contour marks the optional ground-truth lesion.
    """
    if probabilities.ndim != 2:
        raise ValueError(f"Expected a 2D probability map, got shape {probabilities.shape}")
    if target is not None and target.shape != probabilities.shape:
        raise ValueError(f"GT/probability shape mismatch: {target.shape} vs {probabilities.shape}")

    colours = np.asarray(
        [
            (37, 99, 235),   # < 0.3: blue
            (37, 201, 204),  # 0.3 - 0.4: cyan
            (248, 214, 67),  # 0.4 - 0.5: yellow
            (245, 144, 48),  # 0.5 - 0.6: orange
            (220, 57, 54),   # >= 0.6: red
        ],
        dtype=np.uint8,
    )
    labels = ("< 0.3", "0.3 - 0.4", "0.4 - 0.5", "0.5 - 0.6", ">= 0.6")
    region_index = np.digitize(probabilities, bins=(0.3, 0.4, 0.5, 0.6), right=False)
    array = colours[region_index]
    if target is not None:
        array[thick_boundary(target, width=2)] = (0, 0, 0)

    region_image = Image.fromarray(array, mode="RGB")
    legend_height = 70
    canvas = Image.new("RGB", (region_image.width, region_image.height + legend_height), "white")
    canvas.paste(region_image, (0, 0))
    drawer = ImageDraw.Draw(canvas)
    for index, (colour, label) in enumerate(zip(colours, labels)):
        row, column = divmod(index, 3)
        x = 8 + column * 86
        y = region_image.height + 8 + row * 21
        drawer.rectangle((x, y, x + 12, y + 12), fill=tuple(colour))
        drawer.text((x + 17, y - 2), label, fill="black")
    if target is not None:
        drawer.line((8, region_image.height + 55, 26, region_image.height + 55), fill="black", width=3)
        drawer.text((31, region_image.height + 49), "GT contour (black)", fill="black")
    return canvas


def save_probability_regions(probabilities: np.ndarray, target: np.ndarray | None, output_path: Path) -> None:
    """Save a standalone probability panel for callers that explicitly need one."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probability_region_panel(probabilities, target).save(output_path)
