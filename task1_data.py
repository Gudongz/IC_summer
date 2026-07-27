"""Dataset utilities for Task 1 lesion segmentation."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Iterable

# Import OpenCV before PyTorch to avoid the Windows OpenMP runtime load-order
# conflict caused by OpenCV being imported lazily after torch.
import cv2  # noqa: F401
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def build_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    """Match ``000001.jpg`` to ``000001_segmentation.png`` and validate pairs."""
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected image directory {image_dir} and mask directory {mask_dir}.")

    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
        mask_path = mask_dir / f"{image_path.stem}_segmentation.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"No Task 1 mask found for {image_path.name}: expected {mask_path.name}")
        pairs.append((image_path, mask_path))
    if not pairs:
        raise RuntimeError(f"No supported images found in {image_dir}.")
    return pairs


class LesionSegmentationDataset(Dataset[tuple[Tensor, Tensor]]):
    """Task 1 dataset reading preprocessed image/mask pairs from disk."""

    def __init__(
        self,
        pairs: Iterable[tuple[Path, Path]],
        image_size: int,
    ) -> None:
        self.pairs = list(pairs)
        if not self.pairs:
            raise ValueError("pairs must not be empty")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Cannot read preprocessed pair: {image_path}, {mask_path}")
        if image.shape[:2] != (self.image_size, self.image_size) or mask.shape != (self.image_size, self.image_size):
            raise ValueError(
                f"Preprocessed pair must be {self.image_size}x{self.image_size}: {image_path.name}"
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        mask = (mask > 127).astype("float32")
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(mask).unsqueeze(0)


class OneVariantPerSourceSampler(Sampler[int]):
    """Choose one random fixed augmentation for every original image each epoch."""

    def __init__(self, pairs: Iterable[tuple[Path, Path]], seed: int) -> None:
        self.seed = seed
        self.epoch = 0
        self.indices_by_source: dict[str, list[int]] = {}
        for index, (image_path, _) in enumerate(pairs):
            source_id = image_path.stem.split("__aug_", maxsplit=1)[0]
            self.indices_by_source.setdefault(source_id, []).append(index)
        if not self.indices_by_source:
            raise ValueError("OneVariantPerSourceSampler requires at least one pair.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        selected = [rng.choice(indices) for _, indices in sorted(self.indices_by_source.items())]
        rng.shuffle(selected)
        return iter(selected)

    def __len__(self) -> int:
        return len(self.indices_by_source)
