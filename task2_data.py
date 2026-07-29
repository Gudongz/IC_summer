"""Fixed-augmentation dataset utilities for Task 2 attribute segmentation."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2  # Import before torch to preserve the project's Windows OpenMP order.
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from data_preprocessing import TASK2_ATTRIBUTES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Task2Sample:
    image_path: Path
    mask_paths: tuple[Path, ...]
    roi: "ROITransform"
    source_image_id: str
    sample_weight: float


@dataclass(frozen=True)
class ROITransform:
    """One square crop in full-canvas pixel coordinates."""

    left: int
    top: int
    right: int
    bottom: int
    canvas_height: int
    canvas_width: int

    def as_tensor(self) -> Tensor:
        return torch.tensor(
            (self.left, self.top, self.right, self.bottom, self.canvas_height, self.canvas_width),
            dtype=torch.int64,
        )


def _source_image_id(stem: str) -> str:
    return stem.split("__aug_", maxsplit=1)[0]


def manifest_weights(manifest_path: Path) -> dict[str, float]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as file:
        return {row["image_id"]: float(row.get("sample_weight") or 1.0) for row in csv.DictReader(file)}


def manifest_attribute_presence(manifest_path: Path, attribute: str) -> dict[str, bool]:
    """Read source-level Task 2 attribute presence labels from a split manifest."""
    column = f"{attribute}_present"
    with manifest_path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows or column not in rows[0]:
        raise ValueError(
            f"{manifest_path} must contain {column!r}; regenerate the split with data_preprocessing_v2.py."
        )
    return {row["image_id"]: bool(int(row.get(column) or 0)) for row in rows}


def full_canvas_roi(height: int, width: int) -> ROITransform:
    return ROITransform(0, 0, width, height, height, width)


def roi_from_binary_mask(mask: np.ndarray, margin_ratio: float, minimum_box_side: int) -> ROITransform | None:
    """Build a square ROI from the largest predicted lesion component only."""
    if not 0 <= margin_ratio < 0.5:
        raise ValueError("Task 2 ROI margin_ratio must be in [0, 0.5).")
    if minimum_box_side <= 0:
        raise ValueError("Task 2 ROI minimum_box_side must be positive.")
    if mask.ndim != 2:
        raise ValueError("Task 2 ROI mask must be a 2D binary image.")
    component_count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype("uint8"), connectivity=8)
    if component_count <= 1:
        return None
    largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    left = int(stats[largest_index, cv2.CC_STAT_LEFT])
    top = int(stats[largest_index, cv2.CC_STAT_TOP])
    right = left + int(stats[largest_index, cv2.CC_STAT_WIDTH])
    bottom = top + int(stats[largest_index, cv2.CC_STAT_HEIGHT])
    raw_side = max(right - left, bottom - top)
    if raw_side < minimum_box_side:
        return None
    height, width = mask.shape
    side = min(int(np.ceil(raw_side * (1 + 2 * margin_ratio))), height, width)
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    roi_left = min(max(int(round(center_x - side / 2)), 0), width - side)
    roi_top = min(max(int(round(center_y - side / 2)), 0), height - side)
    return ROITransform(roi_left, roi_top, roi_left + side, roi_top + side, height, width)


def build_task2_samples(
    image_dir: Path,
    mask_dir: Path,
    manifest_path: Path | None = None,
    roi_mask_dir: Path | None = None,
    roi_margin_ratio: float = 0.10,
    roi_minimum_box_side: int = 32,
    full_canvas_size: int = 256,
) -> list[Task2Sample]:
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected Task 2 image/label folders: {image_dir}, {mask_dir}")
    if roi_mask_dir is not None:
        roi_mask_dir = Path(roi_mask_dir)
        if not roi_mask_dir.is_dir():
            raise FileNotFoundError(f"Expected Task 1 predicted ROI-mask folder: {roi_mask_dir}")
    weights = manifest_weights(manifest_path) if manifest_path is not None else {}
    samples: list[Task2Sample] = []
    empty_roi_count = small_roi_count = 0
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
        mask_paths = tuple(mask_dir / f"{image_path.stem}_attribute_{attribute}.png" for attribute in TASK2_ATTRIBUTES)
        missing = [path.name for path in mask_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Task 2 masks missing for {image_path.name}: {', '.join(missing)}")
        if roi_mask_dir is None:
            roi = full_canvas_roi(full_canvas_size, full_canvas_size)
        else:
            prior_path = roi_mask_dir / f"{image_path.stem}_segmentation.png"
            prior = cv2.imread(str(prior_path), cv2.IMREAD_GRAYSCALE)
            if prior is None:
                raise FileNotFoundError(f"Task 1 predicted ROI mask missing for {image_path.name}: {prior_path.name}")
            roi = roi_from_binary_mask(prior, roi_margin_ratio, roi_minimum_box_side)
            if roi is None:
                if cv2.countNonZero(prior) == 0:
                    empty_roi_count += 1
                else:
                    small_roi_count += 1
                continue
        source_image_id = _source_image_id(image_path.stem)
        samples.append(Task2Sample(image_path, mask_paths, roi, source_image_id, weights.get(source_image_id, 1.0)))
    if not samples:
        raise RuntimeError(f"No supported Task 2 images found in {image_dir}")
    if roi_mask_dir is not None:
        print(
            f"Task 2 ROI samples from {image_dir.parent.name}: retained={len(samples)}, "
            f"skipped_empty={empty_roi_count}, skipped_small={small_roi_count}."
        )
    return samples


class OneVariantPerSourceSampler(Sampler[int]):
    """Select one random fixed augmentation for every source image per epoch."""

    def __init__(self, samples: list[Task2Sample], seed: int) -> None:
        self.seed = seed
        self.epoch = 0
        self.indices_by_source: dict[str, list[int]] = {}
        for index, sample in enumerate(samples):
            self.indices_by_source.setdefault(sample.source_image_id, []).append(index)
        if not self.indices_by_source:
            raise ValueError("OneVariantPerSourceSampler requires at least one sample.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        selected = [rng.choice(indices) for _, indices in sorted(self.indices_by_source.items())]
        rng.shuffle(selected)
        return iter(selected)

    def __len__(self) -> int:
        return len(self.indices_by_source)


class AttributeCurriculumSampler(Sampler[int]):
    """Sample one fixed variant per selected source with a scheduled positive ratio."""

    def __init__(
        self,
        samples: list[Task2Sample],
        source_presence: dict[str, bool],
        seed: int,
        positive_ratio_start: float,
        positive_ratio_end: float,
        ratio_decay_epochs: int,
    ) -> None:
        if not 0 < positive_ratio_start < 1 or not 0 < positive_ratio_end < 1:
            raise ValueError("Attribute curriculum positive ratios must be strictly between 0 and 1.")
        if ratio_decay_epochs < 1:
            raise ValueError("Attribute curriculum ratio_decay_epochs must be positive.")
        self.seed = seed
        self.epoch = 1
        self.positive_ratio_start = positive_ratio_start
        self.positive_ratio_end = positive_ratio_end
        self.ratio_decay_epochs = ratio_decay_epochs
        self.indices_by_source: dict[str, list[int]] = {}
        for index, sample in enumerate(samples):
            self.indices_by_source.setdefault(sample.source_image_id, []).append(index)
        missing = sorted(set(self.indices_by_source) - set(source_presence))
        if missing:
            raise ValueError(f"Attribute manifest lacks {len(missing)} retained source IDs (for example {missing[0]}).")
        self.positive_sources = [source for source in self.indices_by_source if source_presence[source]]
        self.negative_sources = [source for source in self.indices_by_source if not source_presence[source]]
        if not self.positive_sources or not self.negative_sources:
            raise ValueError("Attribute curriculum requires both positive and negative source images.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @property
    def positive_ratio(self) -> float:
        progress = min(max(self.epoch - 1, 0) / max(self.ratio_decay_epochs - 1, 1), 1.0)
        return self.positive_ratio_start + (self.positive_ratio_end - self.positive_ratio_start) * progress

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        total = len(self.indices_by_source)
        positive_count = round(total * self.positive_ratio)
        sources = (
            [rng.choice(self.positive_sources) for _ in range(positive_count)]
            + [rng.choice(self.negative_sources) for _ in range(total - positive_count)]
        )
        indices = [rng.choice(self.indices_by_source[source]) for source in sources]
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.indices_by_source)


class Task2SegmentationDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Read and resize a lesion ROI, while preserving its full-canvas target."""

    def __init__(self, samples: list[Task2Sample], image_size: int) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        sample = self.samples[index]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {sample.image_path}")
        if image.shape[:2] != (self.image_size, self.image_size):
            raise ValueError(f"Expected {self.image_size}x{self.image_size}: {sample.image_path.name}")
        if image.shape[:2] != (sample.roi.canvas_height, sample.roi.canvas_width):
            raise ValueError(f"ROI canvas size mismatch for {sample.image_path.name}")
        masks = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in sample.mask_paths]
        if any(mask is None for mask in masks):
            raise FileNotFoundError(f"Cannot read one or more masks for {sample.image_path.name}")
        if any(mask.shape != (self.image_size, self.image_size) for mask in masks):
            raise ValueError(f"Mask size mismatch for {sample.image_path.name}")
        full_target = np.stack([(mask > 127).astype("float32") for mask in masks], axis=0)
        roi = sample.roi
        image = image[roi.top : roi.bottom, roi.left : roi.right]
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        roi_target = np.stack(
            [
                cv2.resize(mask[roi.top : roi.bottom, roi.left : roi.right], (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                for mask in full_target
            ],
            axis=0,
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        return (
            torch.from_numpy(image.transpose(2, 0, 1)),
            torch.from_numpy(roi_target.astype("float32")),
            roi.as_tensor(),
            torch.from_numpy(full_target),
        )


def restore_logits_to_canvas(logits: Tensor, rois: Tensor, outside_logit: float = -20.0) -> Tensor:
    """Map per-ROI logits back to full canvases; ROI-exterior pixels are negative."""
    if logits.ndim != 4 or rois.ndim != 2 or rois.shape != (logits.shape[0], 6):
        raise ValueError("Expected logits [B,C,H,W] and ROI metadata [B,6].")
    canvas_height, canvas_width = (int(rois[0, 4].item()), int(rois[0, 5].item()))
    if torch.any(rois[:, 4] != canvas_height) or torch.any(rois[:, 5] != canvas_width):
        raise ValueError("A batch must have one common full-canvas size.")
    restored = logits.new_full((logits.shape[0], logits.shape[1], canvas_height, canvas_width), outside_logit)
    for index, roi in enumerate(rois):
        left, top, right, bottom = (int(value.item()) for value in roi[:4])
        restored[index : index + 1, :, top:bottom, left:right] = F.interpolate(
            logits[index : index + 1], size=(bottom - top, right - left), mode="bilinear", align_corners=False
        )
    return restored
