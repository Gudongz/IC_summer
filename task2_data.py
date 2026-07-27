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
from torch.utils.data import Dataset, Sampler

from data_preprocessing import TASK2_ATTRIBUTES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Task2Sample:
    image_path: Path
    mask_paths: tuple[Path, ...]
    lesion_prior_path: Path
    source_image_id: str
    sample_weight: float


def _source_image_id(stem: str) -> str:
    return stem.split("__aug_", maxsplit=1)[0]


def manifest_weights(manifest_path: Path) -> dict[str, float]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as file:
        return {row["image_id"]: float(row.get("sample_weight") or 1.0) for row in csv.DictReader(file)}


def build_task2_samples(image_dir: Path, mask_dir: Path, lesion_prior_dir: Path, manifest_path: Path | None = None) -> list[Task2Sample]:
    image_dir, mask_dir, lesion_prior_dir = Path(image_dir), Path(mask_dir), Path(lesion_prior_dir)
    if not image_dir.is_dir() or not mask_dir.is_dir() or not lesion_prior_dir.is_dir():
        raise FileNotFoundError(f"Expected Task 2 image/label/prior folders: {image_dir}, {mask_dir}, {lesion_prior_dir}")
    weights = manifest_weights(manifest_path) if manifest_path is not None else {}
    samples: list[Task2Sample] = []
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
        mask_paths = tuple(mask_dir / f"{image_path.stem}_attribute_{attribute}.png" for attribute in TASK2_ATTRIBUTES)
        missing = [path.name for path in mask_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Task 2 masks missing for {image_path.name}: {', '.join(missing)}")
        lesion_prior_path = lesion_prior_dir / f"{image_path.stem}_segmentation.png"
        if not lesion_prior_path.is_file():
            raise FileNotFoundError(
                f"Task 1 predicted lesion prior missing for {image_path.name}: {lesion_prior_path.name}. "
                "Run prepare_task2_priors.py for this split before Task 2 training."
            )
        source_image_id = _source_image_id(image_path.stem)
        samples.append(Task2Sample(image_path, mask_paths, lesion_prior_path, source_image_id, weights.get(source_image_id, 1.0)))
    if not samples:
        raise RuntimeError(f"No supported Task 2 images found in {image_dir}")
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


class Task2SegmentationDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Read RGB, Task 1 predicted prior, and five aligned Task 2 masks."""

    def __init__(self, samples: list[Task2Sample], image_size: int) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        sample = self.samples[index]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {sample.image_path}")
        if image.shape[:2] != (self.image_size, self.image_size):
            raise ValueError(f"Expected {self.image_size}x{self.image_size}: {sample.image_path.name}")
        masks = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in sample.mask_paths]
        lesion_prior = cv2.imread(str(sample.lesion_prior_path), cv2.IMREAD_GRAYSCALE)
        if any(mask is None for mask in masks):
            raise FileNotFoundError(f"Cannot read one or more masks for {sample.image_path.name}")
        if any(mask.shape != (self.image_size, self.image_size) for mask in masks):
            raise ValueError(f"Mask size mismatch for {sample.image_path.name}")
        if lesion_prior is None or lesion_prior.shape != (self.image_size, self.image_size):
            raise ValueError(f"Lesion-prior size mismatch for {sample.image_path.name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        target = torch.from_numpy(np.stack([(mask > 127).astype("float32") for mask in masks], axis=0))
        prior = torch.from_numpy((lesion_prior > 127).astype("float32")).unsqueeze(0)
        return torch.from_numpy(image.transpose(2, 0, 1)), prior, target
