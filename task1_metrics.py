"""Final evaluation metrics for binary lesion segmentation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_coefficient(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """Dice coefficient for two boolean 2D masks."""
    prediction, target = prediction.astype(bool), target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    return float((2 * intersection + eps) / (prediction.sum() + target.sum() + eps))


def hausdorff_distances(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return symmetric Hausdorff distance (HD) and robust HD95 in pixels.

    Both-empty masks score 0. If only one mask is empty, the image diagonal is
    used as a finite maximum penalty, allowing dataset-level means to be shown.
    """
    prediction, target = prediction.astype(bool), target.astype(bool)
    if not prediction.any() and not target.any():
        return 0.0, 0.0
    if not prediction.any() or not target.any():
        diagonal = float(np.hypot(*prediction.shape))
        return diagonal, diagonal

    pred_surface = prediction & ~binary_erosion(prediction)
    target_surface = target & ~binary_erosion(target)
    pred_to_target = distance_transform_edt(~target_surface)[pred_surface]
    target_to_pred = distance_transform_edt(~pred_surface)[target_surface]
    distances = np.concatenate((pred_to_target, target_to_pred))
    return float(distances.max()), float(np.percentile(distances, 95))
