"""Task 1/2 dermoscopy data cleaning and paired augmentation.

Install augmentation dependencies when needed:

    python -m pip install "albumentations==2.0.8" opencv-python-headless

Examples:

    python task1_data_pipeline.py audit --task 1
    python task1_data_pipeline.py audit --task 2
    python task1_data_pipeline.py split --task 2
    python task1_data_pipeline.py split --task 1 --reference-split splits/task2_split.csv
    python task1_data_pipeline.py balanced-epoch
    python task1_data_pipeline.py preview --task 2 --image-id 000001
    python data_preprocessing.py prepare-training

The original images and masks are never modified.  ``audit`` writes a CSV
report, while ``preview`` writes a contact sheet showing random augmentations.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_DIR / "data" / "train"
DEFAULT_TASK1_AUDIT_PATH = PROJECT_DIR / "task1_audit.csv"
DEFAULT_TASK2_AUDIT_PATH = PROJECT_DIR / "task2_audit.csv"
DEFAULT_TASK1_PREVIEW_PATH = PROJECT_DIR / "task1_augmentation_preview.jpg"
DEFAULT_TASK2_PREVIEW_PATH = PROJECT_DIR / "task2_augmentation_preview.jpg"
DEFAULT_SPLIT_DIR = PROJECT_DIR / "splits"
DEFAULT_SIZE = 512

# This order is also the channel order returned by load_task2_pair().
TASK2_ATTRIBUTES = (
    "pigment_network",
    "negative_network",
    "streaks",
    "milia_like_cyst",
    "globules",
)

# RGB contour colours used only in the Task 2 preview image.
TASK2_PREVIEW_COLOURS = (
    (255, 60, 60),
    (60, 220, 60),
    (60, 120, 255),
    (255, 220, 60),
    (220, 60, 255),
)


def image_path(data_root: Path, image_id: str) -> Path:
    return data_root / "images" / f"{image_id}.jpg"


def mask_path(data_root: Path, image_id: str) -> Path:
    """Return the Task 1 lesion-mask path (kept for compatibility)."""
    return data_root / "task1_gt" / f"{image_id}_segmentation.png"


def task2_mask_path(data_root: Path, image_id: str, attribute: str) -> Path:
    if attribute not in TASK2_ATTRIBUTES:
        raise ValueError(f"Unknown Task 2 attribute: {attribute}")
    return data_root / "task2_gt" / f"{image_id}_attribute_{attribute}.png"


@lru_cache(maxsize=16)
def _audit_statuses(audit_path_text: str) -> dict[str, str]:
    audit_path = Path(audit_path_text)
    with audit_path.open("r", newline="", encoding="utf-8-sig") as file:
        return {
            row["image_id"]: row.get("status", "").strip().lower()
            for row in csv.DictReader(file)
        }


def validate_audit_status(
    image_id: str,
    audit_path: Path | None,
    include_review: bool = True,
) -> None:
    """Reject samples marked unusable by an existing audit manifest."""
    if audit_path is None or not audit_path.exists():
        return

    statuses = _audit_statuses(str(audit_path.resolve()))
    if image_id not in statuses:
        raise ValueError(f"{image_id} is missing from audit file {audit_path}")

    status = statuses[image_id]
    allowed = {"ok", "review"} if include_review else {"ok"}
    if status not in allowed:
        raise ValueError(
            f"{image_id} has audit status {status!r} in {audit_path}; "
            "sample was not loaded"
        )


def processed_mask_stats(mask: Image.Image, size: int) -> tuple[int, int, int, float]:
    """Measure a binary mask after aspect-ratio resize and square padding.

    Padding is background, so only the resized mask needs to be counted. The
    denominator is the complete ``size x size`` model input, including padding.
    """
    if size < 1:
        raise ValueError("size must be at least 1")

    width, height = mask.size
    scale = size / max(width, height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized_mask = mask.resize(
        (resized_width, resized_height), Image.Resampling.NEAREST
    )
    histogram = resized_mask.histogram()
    lesion_pixels = sum(histogram[128:])
    lesion_ratio = lesion_pixels / (size * size)
    return resized_width, resized_height, lesion_pixels, lesion_ratio


def print_processed_lesion_average(
    rows: list[dict[str, Any]], size: int
) -> None:
    ratios = [
        float(row["processed_lesion_ratio"])
        for row in rows
        if row.get("processed_lesion_ratio") is not None
        and row.get("status") != "reject"
    ]
    if not ratios:
        print("Processed lesion ratio: no valid masks")
        return

    average_ratio = sum(ratios) / len(ratios)
    print(
        f"Average lesion pixel ratio after {size}x{size} preprocessing: "
        f"{average_ratio:.6f} ({average_ratio * 100:.4f}%)"
    )


def audit_task1_dataset(
    data_root: Path, output_path: Path, size: int = DEFAULT_SIZE
) -> None:
    """Check Task 1 image/mask pairing, dimensions and binary masks."""
    images_dir = data_root / "images"
    masks_dir = data_root / "task1_gt"

    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {images_dir} and {masks_dir} to be directories."
        )

    rows: list[dict[str, Any]] = []

    for source_image_path in sorted(images_dir.glob("*.jpg")):
        image_id = source_image_path.stem
        source_mask_path = mask_path(data_root, image_id)
        flags: list[str] = []
        width = 0
        height = 0
        lesion_ratio = 0.0
        processed_width = 0
        processed_height = 0
        processed_lesion_pixels: int | None = None
        processed_lesion_ratio: float | None = None

        try:
            if not source_mask_path.exists():
                raise FileNotFoundError("mask_missing")

            # load() forces a complete decode instead of checking only headers.
            with Image.open(source_image_path) as image:
                image.load()
                width, height = image.size
                if image.mode != "RGB":
                    flags.append(f"image_mode={image.mode}")

            with Image.open(source_mask_path) as source_mask:
                mask_mode = source_mask.mode
                mask = source_mask.convert("L")
                mask.load()

                if mask.size != (width, height):
                    flags.append("size_mismatch")

                histogram = mask.histogram()
                used_values = {
                    value for value, count in enumerate(histogram) if count > 0
                }
                if not used_values.issubset({0, 255}):
                    flags.append("non_binary_mask")

                lesion_ratio = sum(histogram[128:]) / (mask.width * mask.height)
                (
                    processed_width,
                    processed_height,
                    processed_lesion_pixels,
                    processed_lesion_ratio,
                ) = processed_mask_stats(mask, size)
                if lesion_ratio < 0.005:
                    flags.append("very_small_lesion")
                if lesion_ratio > 0.95:
                    flags.append("almost_full_mask")

                if mask_mode != "L":
                    flags.append(f"mask_mode={mask_mode}")

        except Exception as exc:  # Keep auditing the remaining files.
            flags.append(f"read_error={type(exc).__name__}:{exc}")

        structural_error = any(
            flag.startswith(("read_error", "size_mismatch", "non_binary_mask"))
            for flag in flags
        )
        status = "reject" if structural_error else ("review" if flags else "ok")

        rows.append(
            {
                "image_id": image_id,
                "width": width,
                "height": height,
                "lesion_ratio": round(lesion_ratio, 6),
                "processed_canvas": f"{size}x{size}",
                "processed_content_width": processed_width,
                "processed_content_height": processed_height,
                "processed_image_pixels": size * size,
                "processed_lesion_pixels": processed_lesion_pixels,
                "processed_lesion_ratio": (
                    round(processed_lesion_ratio, 8)
                    if processed_lesion_ratio is not None
                    else None
                ),
                "processed_lesion_percent": (
                    round(processed_lesion_ratio * 100, 4)
                    if processed_lesion_ratio is not None
                    else None
                ),
                "status": status,
                "flags": ";".join(flags),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    _audit_statuses.cache_clear()

    counts = Counter(row["status"] for row in rows)
    print(f"Audit complete: {len(rows)} image/mask pairs")
    print(
        f"ok={counts['ok']}, review={counts['review']}, "
        f"reject={counts['reject']}"
    )
    print_processed_lesion_average(rows, size)
    print(f"Report: {output_path.resolve()}")


def audit_task2_dataset(
    data_root: Path, output_path: Path, size: int = DEFAULT_SIZE
) -> None:
    """Audit five Task 2 masks per image; empty masks are valid negatives."""
    images_dir = data_root / "images"
    masks_dir = data_root / "task2_gt"

    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {images_dir} and {masks_dir} to be directories."
        )

    rows: list[dict[str, Any]] = []
    for source_image_path in sorted(images_dir.glob("*.jpg")):
        image_id = source_image_path.stem
        flags: list[str] = []
        width = 0
        height = 0
        row: dict[str, Any] = {"image_id": image_id}

        try:
            with Image.open(source_image_path) as image:
                image.load()
                width, height = image.size
                if image.mode != "RGB":
                    flags.append(f"image_mode={image.mode}")
        except Exception as exc:
            flags.append(f"image_read_error={type(exc).__name__}:{exc}")

        row.update({"width": width, "height": height})

        # Task 2 attributes are local regions. Use the Task 1 lesion mask to
        # measure the complete lesion area after deterministic preprocessing.
        source_lesion_path = mask_path(data_root, image_id)
        processed_lesion_pixels: int | None = None
        processed_lesion_ratio: float | None = None
        processed_width = 0
        processed_height = 0
        try:
            if not source_lesion_path.exists():
                raise FileNotFoundError("lesion_mask_missing")
            with Image.open(source_lesion_path) as source_lesion_mask:
                lesion_mask_mode = source_lesion_mask.mode
                lesion_mask = source_lesion_mask.convert("L")
                lesion_mask.load()

                if lesion_mask.size != (width, height):
                    flags.append("lesion:size_mismatch")
                lesion_histogram = lesion_mask.histogram()
                lesion_values = {
                    value
                    for value, count in enumerate(lesion_histogram)
                    if count > 0
                }
                if not lesion_values.issubset({0, 255}):
                    flags.append("lesion:non_binary_mask")
                if lesion_mask_mode != "L":
                    flags.append(f"lesion:mask_mode={lesion_mask_mode}")

                (
                    processed_width,
                    processed_height,
                    processed_lesion_pixels,
                    processed_lesion_ratio,
                ) = processed_mask_stats(lesion_mask, size)
        except Exception as exc:
            flags.append(f"lesion:read_error={type(exc).__name__}:{exc}")

        row.update(
            {
                "processed_canvas": f"{size}x{size}",
                "processed_content_width": processed_width,
                "processed_content_height": processed_height,
                "processed_image_pixels": size * size,
                "processed_lesion_pixels": processed_lesion_pixels,
                "processed_lesion_ratio": (
                    round(processed_lesion_ratio, 8)
                    if processed_lesion_ratio is not None
                    else None
                ),
                "processed_lesion_percent": (
                    round(processed_lesion_ratio * 100, 4)
                    if processed_lesion_ratio is not None
                    else None
                ),
            }
        )

        for attribute in TASK2_ATTRIBUTES:
            source_mask_path = task2_mask_path(data_root, image_id, attribute)
            ratio = 0.0
            try:
                if not source_mask_path.exists():
                    raise FileNotFoundError("mask_missing")

                with Image.open(source_mask_path) as source_mask:
                    mask_mode = source_mask.mode
                    mask = source_mask.convert("L")
                    mask.load()

                    if mask.size != (width, height):
                        flags.append(f"{attribute}:size_mismatch")

                    histogram = mask.histogram()
                    used_values = {
                        value for value, count in enumerate(histogram) if count > 0
                    }
                    if not used_values.issubset({0, 255}):
                        flags.append(f"{attribute}:non_binary_mask")

                    ratio = sum(histogram[128:]) / (mask.width * mask.height)
                    if mask_mode != "L":
                        flags.append(f"{attribute}:mask_mode={mask_mode}")
            except Exception as exc:
                flags.append(
                    f"{attribute}:read_error={type(exc).__name__}:{exc}"
                )

            row[f"{attribute}_present"] = int(ratio > 0.0)
            row[f"{attribute}_ratio"] = round(ratio, 8)

        structural_error = any(
            marker in flag
            for flag in flags
            for marker in ("read_error", "size_mismatch", "non_binary_mask")
        )
        row["status"] = "reject" if structural_error else ("review" if flags else "ok")
        row["flags"] = ";".join(flags)
        rows.append(row)

    if not rows:
        raise ValueError(f"No JPG images found in {images_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    _audit_statuses.cache_clear()

    counts = Counter(row["status"] for row in rows)
    print(f"Task 2 audit complete: {len(rows)} images, 5 masks per image")
    print(
        f"ok={counts['ok']}, review={counts['review']}, "
        f"reject={counts['reject']}"
    )
    for attribute in TASK2_ATTRIBUTES:
        positives = sum(row[f"{attribute}_present"] for row in rows)
        print(f"{attribute}: {positives}/{len(rows)} positive")
    print_processed_lesion_average(rows, size)
    print(f"Report: {output_path.resolve()}")


def audit_dataset(
    data_root: Path,
    output_path: Path,
    task: int = 1,
    size: int = DEFAULT_SIZE,
) -> None:
    """Audit Task 1 or Task 2 without changing source files."""
    if task == 1:
        audit_task1_dataset(data_root, output_path, size)
    elif task == 2:
        audit_task2_dataset(data_root, output_path, size)
    else:
        raise ValueError("task must be 1 or 2")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _perceptual_hashes(path: Path) -> tuple[int, int, float]:
    """Return dHash, average-hash and aspect ratio for leakage grouping."""
    def flattened_pixels(image: Image.Image) -> list[int]:
        if hasattr(image, "get_flattened_data"):
            return list(image.get_flattened_data())
        return list(image.getdata())

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        width, height = image.size

        difference_pixels = flattened_pixels(
            image.resize((9, 8), Image.Resampling.BILINEAR)
        )
        difference_hash = 0
        for row in range(8):
            for column in range(8):
                left = difference_pixels[row * 9 + column]
                right = difference_pixels[row * 9 + column + 1]
                difference_hash = (difference_hash << 1) | int(right > left)

        average_pixels = flattened_pixels(
            image.resize((8, 8), Image.Resampling.BILINEAR)
        )
        average = sum(average_pixels) / len(average_pixels)
        average_hash = 0
        for pixel in average_pixels:
            average_hash = (average_hash << 1) | int(pixel >= average)

    return difference_hash, average_hash, width / height


def find_duplicate_groups(
    data_root: Path,
    image_ids: list[str],
    max_hash_distance: int = 2,
) -> tuple[dict[str, str], dict[str, int]]:
    """Group perceptually similar images so they cannot cross data splits."""
    parent = {image_id: image_id for image_id in image_ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    if max_hash_distance >= 0:
        signatures = {
            image_id: _perceptual_hashes(image_path(data_root, image_id))
            for image_id in image_ids
        }
        for index, first_id in enumerate(image_ids):
            first_difference, first_average, first_aspect = signatures[first_id]
            for second_id in image_ids[index + 1 :]:
                second_difference, second_average, second_aspect = signatures[second_id]
                aspect_delta = abs(math.log(first_aspect / second_aspect))
                if aspect_delta > 0.08:
                    continue
                if (first_difference ^ second_difference).bit_count() > max_hash_distance:
                    continue
                if (first_average ^ second_average).bit_count() > max_hash_distance:
                    continue
                union(first_id, second_id)

    grouped: dict[str, list[str]] = {}
    for image_id in image_ids:
        grouped.setdefault(find(image_id), []).append(image_id)

    group_for_id: dict[str, str] = {}
    group_size_for_id: dict[str, int] = {}
    ordered_groups = sorted(grouped.values(), key=lambda members: min(members))
    for group_number, members in enumerate(ordered_groups, start=1):
        group_name = f"group_{group_number:06d}"
        for image_id in members:
            group_for_id[image_id] = group_name
            group_size_for_id[image_id] = len(members)
    return group_for_id, group_size_for_id


def _lesion_size_label(row: dict[str, str]) -> str:
    ratio = float(row.get("processed_lesion_ratio") or 0.0)
    if ratio < 0.01:
        return "lesion_tiny"
    if ratio < 0.05:
        return "lesion_small"
    if ratio < 0.25:
        return "lesion_medium"
    if ratio < 0.50:
        return "lesion_large"
    return "lesion_very_large"


def _split_labels(row: dict[str, str], task: int) -> set[str]:
    labels = {_lesion_size_label(row)}
    if task == 2:
        labels.update(
            attribute
            for attribute in TASK2_ATTRIBUTES
            if int(row.get(f"{attribute}_present") or 0) == 1
        )
    return labels


def _group_stratified_split(
    rows: list[dict[str, str]],
    task: int,
    group_for_id: dict[str, str],
    val_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Greedily preserve lesion-size and multilabel distributions by group."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    rows_by_id = {row["image_id"]: row for row in rows}
    members_by_group: dict[str, list[str]] = {}
    for image_id, group_name in group_for_id.items():
        members_by_group.setdefault(group_name, []).append(image_id)

    total_labels: Counter[str] = Counter()
    group_records = []
    for group_name, members in members_by_group.items():
        label_counts: Counter[str] = Counter()
        for image_id in members:
            labels = _split_labels(rows_by_id[image_id], task)
            label_counts.update(labels)
            total_labels.update(labels)
        group_records.append(
            {
                "name": group_name,
                "members": members,
                "size": len(members),
                "labels": label_counts,
            }
        )

    target_size = max(1, min(len(rows) - 1, round(len(rows) * val_fraction)))
    target_labels = {
        label: count * val_fraction for label, count in total_labels.items()
    }

    rng = random.Random(seed)
    rng.shuffle(group_records)

    val_groups: list[dict[str, Any]] = []
    remaining_groups = group_records.copy()
    val_size = 0
    val_labels: Counter[str] = Counter()

    def distribution_cost(size_value: int, labels_value: Counter[str]) -> float:
        size_cost = ((size_value - target_size) / max(target_size, 1)) ** 2
        label_cost = sum(
            (
                (labels_value[label] - target_labels[label])
                / max(target_labels[label], 1.0)
            )
            ** 2
            for label in target_labels
        ) / max(len(target_labels), 1)
        return size_cost + label_cost

    # Select each validation group by the global size + multilabel error after
    # adding it. Random shuffling above makes ties deterministic for ``seed``.
    while val_size < target_size and remaining_groups:
        best_group = min(
            remaining_groups,
            key=lambda group: distribution_cost(
                val_size + group["size"], val_labels + group["labels"]
            ),
        )
        remaining_groups.remove(best_group)
        val_groups.append(best_group)
        val_size += best_group["size"]
        val_labels += best_group["labels"]

    val_ids = {
        image_id for group in val_groups for image_id in group["members"]
    }
    train_ids = set(rows_by_id) - val_ids
    if not train_ids or not val_ids:
        raise RuntimeError("Unable to create non-empty train and validation splits")
    return train_ids, val_ids


def _task2_sample_weights(
    train_rows: list[dict[str, str]], max_weight: float
) -> dict[str, float]:
    """Build moderate inverse-frequency weights for sparse Task 2 labels."""
    total = len(train_rows)
    positive_counts = {
        attribute: sum(
            int(row.get(f"{attribute}_present") or 0) for row in train_rows
        )
        for attribute in TASK2_ATTRIBUTES
    }

    raw_weights: dict[str, float] = {}
    for row in train_rows:
        positive_weights = [
            math.sqrt(total / positive_counts[attribute])
            for attribute in TASK2_ATTRIBUTES
            if positive_counts[attribute] > 0
            and int(row.get(f"{attribute}_present") or 0) == 1
        ]
        raw_weights[row["image_id"]] = min(
            max_weight, max([1.0, *positive_weights])
        )

    mean_weight = sum(raw_weights.values()) / len(raw_weights)
    return {
        image_id: weight / mean_weight for image_id, weight in raw_weights.items()
    }


def sample_manifest_ids(
    manifest_path: Path,
    count: int | None = None,
    seed: int = 42,
) -> list[str]:
    """Sample one balanced Task 2 training epoch from manifest weights."""
    rows = _read_csv_rows(manifest_path)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    if count is not None and count < 1:
        raise ValueError("count must be at least 1")
    sample_count = len(rows) if count is None else count
    rng = random.Random(seed)
    return rng.choices(
        [row["image_id"] for row in rows],
        weights=[float(row.get("sample_weight") or 1.0) for row in rows],
        k=sample_count,
    )


def write_balanced_epoch_manifest(
    manifest_path: Path,
    output_path: Path,
    count: int | None = None,
    seed: int = 42,
) -> Path:
    """Materialize one weighted, with-replacement Task 2 training epoch."""
    rows = _read_csv_rows(manifest_path)
    rows_by_id = {row["image_id"]: row for row in rows}
    sampled_ids = sample_manifest_ids(manifest_path, count=count, seed=seed)
    output_rows = [
        {"epoch_index": index, **rows_by_id[image_id]}
        for index, image_id in enumerate(sampled_ids)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Balanced epoch manifest: {output_path.resolve()}")
    return output_path


def create_split_manifests(
    data_root: Path,
    task: int,
    output_dir: Path = DEFAULT_SPLIT_DIR,
    audit_path: Path | None = None,
    val_fraction: float = 0.20,
    seed: int = 42,
    max_hash_distance: int = 2,
    include_review: bool = True,
    max_sample_weight: float = 5.0,
    size: int = DEFAULT_SIZE,
    reference_split_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Create clean, duplicate-grouped and stratified train/val manifests."""
    if task not in (1, 2):
        raise ValueError("task must be 1 or 2")
    if max_sample_weight < 1.0:
        raise ValueError("max_sample_weight must be at least 1")

    if audit_path is None:
        audit_path = (
            DEFAULT_TASK1_AUDIT_PATH if task == 1 else DEFAULT_TASK2_AUDIT_PATH
        )
    if not audit_path.exists():
        print(f"Audit file not found; creating {audit_path}")
        audit_dataset(data_root, audit_path, task=task, size=size)

    audit_rows = _read_csv_rows(audit_path)
    allowed_statuses = {"ok", "review"} if include_review else {"ok"}
    clean_rows = [
        row for row in audit_rows if row.get("status", "").lower() in allowed_statuses
    ]
    rejected_count = len(audit_rows) - len(clean_rows)
    if len(clean_rows) < 2:
        raise ValueError("Not enough clean samples to create data splits")

    image_ids = sorted(row["image_id"] for row in clean_rows)
    if reference_split_path is not None:
        reference_rows = _read_csv_rows(reference_split_path)
        reference_by_id = {row["image_id"]: row for row in reference_rows}
        missing_reference_ids = sorted(set(image_ids) - set(reference_by_id))
        if missing_reference_ids:
            raise ValueError(
                f"Reference split is missing {len(missing_reference_ids)} clean IDs; "
                f"first missing ID: {missing_reference_ids[0]}"
            )
        group_for_id = {
            image_id: reference_by_id[image_id]["duplicate_group"]
            for image_id in image_ids
        }
        group_size_for_id = {
            image_id: int(reference_by_id[image_id]["duplicate_group_size"])
            for image_id in image_ids
        }
        train_ids = {
            image_id
            for image_id in image_ids
            if reference_by_id[image_id]["split"] == "train"
        }
        val_ids = {
            image_id
            for image_id in image_ids
            if reference_by_id[image_id]["split"] == "val"
        }
        if train_ids | val_ids != set(image_ids):
            raise ValueError("Reference split contains unsupported or missing split values")
    else:
        group_for_id, group_size_for_id = find_duplicate_groups(
            data_root, image_ids, max_hash_distance=max_hash_distance
        )
        train_ids, val_ids = _group_stratified_split(
            clean_rows, task, group_for_id, val_fraction, seed
        )

    rows_by_id = {row["image_id"]: row for row in clean_rows}
    train_rows = [rows_by_id[image_id] for image_id in sorted(train_ids)]
    val_rows = [rows_by_id[image_id] for image_id in sorted(val_ids)]
    sample_weights = (
        _task2_sample_weights(train_rows, max_sample_weight)
        if task == 2
        else {image_id: 1.0 for image_id in train_ids}
    )

    def enrich(row: dict[str, str], split: str) -> dict[str, Any]:
        image_id = row["image_id"]
        return {
            "image_id": image_id,
            "split": split,
            "duplicate_group": group_for_id[image_id],
            "duplicate_group_size": group_size_for_id[image_id],
            "sample_weight": round(
                sample_weights.get(image_id, 1.0) if split == "train" else 1.0,
                6,
            ),
            **{key: value for key, value in row.items() if key != "image_id"},
        }

    enriched_train = [enrich(row, "train") for row in train_rows]
    enriched_val = [enrich(row, "val") for row in val_rows]
    combined_rows = sorted(
        [*enriched_train, *enriched_val], key=lambda row: row["image_id"]
    )

    train_groups = {row["duplicate_group"] for row in enriched_train}
    val_groups = {row["duplicate_group"] for row in enriched_val}
    if train_groups & val_groups:
        raise RuntimeError("Duplicate group leakage detected after splitting")

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / f"task{task}_split.csv"
    train_path = output_dir / f"task{task}_train.csv"
    val_path = output_dir / f"task{task}_val.csv"

    def write_manifest(path: Path, output_rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=output_rows[0].keys())
            writer.writeheader()
            writer.writerows(output_rows)

    write_manifest(combined_path, combined_rows)
    write_manifest(train_path, enriched_train)
    write_manifest(val_path, enriched_val)

    duplicate_images = sum(size_value > 1 for size_value in group_size_for_id.values())
    print(
        f"Task {task} split complete: train={len(enriched_train)}, "
        f"val={len(enriched_val)}, filtered={rejected_count}, "
        f"images_in_duplicate_groups={duplicate_images}"
    )
    if task == 2:
        for attribute in TASK2_ATTRIBUTES:
            train_positive = sum(
                int(row.get(f"{attribute}_present") or 0) for row in train_rows
            )
            val_positive = sum(
                int(row.get(f"{attribute}_present") or 0) for row in val_rows
            )
            print(
                f"{attribute}: train={train_positive}/{len(train_rows)}, "
                f"val={val_positive}/{len(val_rows)}"
            )
    print(f"Combined manifest: {combined_path.resolve()}")
    return combined_path, train_path, val_path


def _augmentation_dependencies():
    try:
        import albumentations as A
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing augmentation dependency. Run:\n"
            'python -m pip install "albumentations==2.0.8" '
            "opencv-python-headless"
        ) from exc
    return A, cv2, np


def build_transforms(size: int = DEFAULT_SIZE, seed: int = 42):
    """Return random training and deterministic validation transforms."""
    A, cv2, _ = _augmentation_dependencies()

    train_transform = A.Compose(
        [
            # Preserve the aspect ratio and keep the complete lesion in frame.
            A.LongestMaxSize(
                max_size=size,
                interpolation=cv2.INTER_AREA,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            A.PadIfNeeded(
                min_height=size,
                min_width=size,
                position="center",
                border_mode=cv2.BORDER_CONSTANT,
                fill=(0, 0, 0),
                fill_mask=0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(-0.05, 0.05),
                rotate=(-30, 30),
                shear=(-3, 3),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(0, 0, 0),
                fill_mask=0,
                p=0.7,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15,
                        contrast_limit=0.15,
                        p=1.0,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=6,
                        sat_shift_limit=10,
                        val_shift_limit=10,
                        p=1.0,
                    ),
                    A.CLAHE(clip_limit=(1.0, 2.0), p=1.0),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.01, 0.04), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.ImageCompression(quality_range=(85, 100), p=1.0),
                ],
                # One of the three effects is used on 20% of samples.
                p=0.2,
            ),
        ],
        mask_interpolation=cv2.INTER_NEAREST,
        strict=True,
        seed=seed,
    )

    validation_transform = A.Compose(
        [
            A.LongestMaxSize(
                max_size=size,
                interpolation=cv2.INTER_AREA,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            A.PadIfNeeded(
                min_height=size,
                min_width=size,
                position="center",
                border_mode=cv2.BORDER_CONSTANT,
                fill=(0, 0, 0),
                fill_mask=0,
            ),
        ],
        mask_interpolation=cv2.INTER_NEAREST,
        strict=True,
    )
    return train_transform, validation_transform


@lru_cache(maxsize=16)
def cached_transforms(size: int = DEFAULT_SIZE, seed: int = 42):
    """Reuse one stateful Compose sequence instead of resetting it per image."""
    return build_transforms(size=size, seed=seed)


def read_pair(data_root: Path, image_id: str):
    """Read an RGB image and its 0/1 segmentation mask."""
    _, cv2, np = _augmentation_dependencies()

    source_image_path = image_path(data_root, image_id)
    source_mask_path = mask_path(data_root, image_id)

    image = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(source_mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {source_image_path}")
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {source_mask_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = (mask > 127).astype(np.uint8)
    return image, mask


def read_task2_pair(data_root: Path, image_id: str):
    """Read one RGB image and five 0/1 Task 2 attribute masks."""
    _, cv2, np = _augmentation_dependencies()
    source_image_path = image_path(data_root, image_id)

    image = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {source_image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = []
    for attribute in TASK2_ATTRIBUTES:
        source_mask_path = task2_mask_path(data_root, image_id, attribute)
        mask = cv2.imread(str(source_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {source_mask_path}")
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"Shape mismatch for {source_mask_path}: "
                f"image={image.shape[:2]}, mask={mask.shape}"
            )
        masks.append((mask > 127).astype(np.uint8))
    return image, masks


def load_pair(
    image_id: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    training: bool = True,
    size: int = DEFAULT_SIZE,
    transform=None,
    audit_path: Path | None = DEFAULT_TASK1_AUDIT_PATH,
    include_review: bool = True,
    seed: int = 42,
):
    """Return a model-ready HWC float image and HW float mask."""
    _, _, np = _augmentation_dependencies()
    validate_audit_status(image_id, audit_path, include_review)
    image, mask = read_pair(data_root, image_id)

    if transform is None:
        train_transform, validation_transform = cached_transforms(size=size, seed=seed)
        transform = train_transform if training else validation_transform
    result = transform(image=image, mask=mask)
    model_image = result["image"].astype(np.float32) / 255.0
    model_mask = (result["mask"] > 0).astype(np.float32)
    return model_image, model_mask


def load_task2_pair(
    image_id: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    training: bool = True,
    size: int = DEFAULT_SIZE,
    transform=None,
    audit_path: Path | None = DEFAULT_TASK2_AUDIT_PATH,
    include_review: bool = True,
    seed: int = 42,
):
    """Return an HWC float image and five Task 2 masks in 5xHxW order."""
    _, _, np = _augmentation_dependencies()
    validate_audit_status(image_id, audit_path, include_review)
    image, masks = read_task2_pair(data_root, image_id)

    if transform is None:
        train_transform, validation_transform = cached_transforms(size=size, seed=seed)
        transform = train_transform if training else validation_transform
    # Albumentations applies identical geometry to every item in ``masks``.
    result = transform(image=image, masks=masks)
    model_image = result["image"].astype(np.float32) / 255.0
    model_masks = np.stack(
        [(mask > 0).astype(np.float32) for mask in result["masks"]], axis=0
    )
    return model_image, model_masks


def load_task_pair(
    task: int,
    image_id: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    training: bool = True,
    size: int = DEFAULT_SIZE,
    transform=None,
    audit_path: Path | None = None,
    include_review: bool = True,
    seed: int = 42,
):
    """Unified loader: Task 1 returns HxW; Task 2 returns 5xHxW masks."""
    if task == 1:
        return load_pair(
            image_id,
            data_root=data_root,
            training=training,
            size=size,
            transform=transform,
            audit_path=audit_path or DEFAULT_TASK1_AUDIT_PATH,
            include_review=include_review,
            seed=seed,
        )
    if task == 2:
        return load_task2_pair(
            image_id,
            data_root=data_root,
            training=training,
            size=size,
            transform=transform,
            audit_path=audit_path or DEFAULT_TASK2_AUDIT_PATH,
            include_review=include_review,
            seed=seed,
        )
    raise ValueError("task must be 1 or 2")


def _fixed_task1_augmentation_transforms(size: int, seed: int):
    """Return deterministic transforms covering every Task 1 augmentation type.

    Unlike ``build_transforms`` (which samples a new random augmentation at
    training time), these transforms are used to write a reproducible training
    set to disk. Every augmentation family has one dedicated variant so that
    no ``OneOf`` branch is silently omitted.
    """
    A, cv2, _ = _augmentation_dependencies()
    preprocess = [
        A.LongestMaxSize(
            max_size=size,
            interpolation=cv2.INTER_AREA,
            mask_interpolation=cv2.INTER_NEAREST,
        ),
        A.PadIfNeeded(
            min_height=size,
            min_width=size,
            position="center",
            border_mode=cv2.BORDER_CONSTANT,
            fill=(0, 0, 0),
            fill_mask=0,
        ),
    ]
    variants = {
        "base": [],
        "hflip": [A.HorizontalFlip(p=1.0)],
        "vflip": [A.VerticalFlip(p=1.0)],
        "affine": [
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(-0.05, 0.05),
                rotate=(-30, 30),
                shear=(-3, 3),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(0, 0, 0),
                fill_mask=0,
                p=1.0,
            )
        ],
        "brightness_contrast": [
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=1.0
            )
        ],
        "hue_saturation_value": [
            A.HueSaturationValue(
                hue_shift_limit=6, sat_shift_limit=10, val_shift_limit=10, p=1.0
            )
        ],
        "clahe": [A.CLAHE(clip_limit=(1.0, 2.0), p=1.0)],
        "gauss_noise": [A.GaussNoise(std_range=(0.01, 0.04), p=1.0)],
        "gaussian_blur": [A.GaussianBlur(blur_limit=(3, 5), p=1.0)],
        "image_compression": [A.ImageCompression(quality_range=(85, 100), p=1.0)],
    }
    return {
        name: A.Compose(
            [*preprocess, *operations],
            mask_interpolation=cv2.INTER_NEAREST,
            strict=True,
            # Each variant gets a stable, distinct random sequence.
            seed=seed + index,
        )
        for index, (name, operations) in enumerate(variants.items())
    }


def prepare_fixed_task1_training_set(
    data_root: Path,
    output_root: Path,
    audit_path: Path,
    split_dir: Path,
    size: int = DEFAULT_SIZE,
    val_fraction: float = 0.10,
    seed: int = 42,
    max_hash_distance: int = 2,
) -> None:
    """Create a leakage-safe fixed Task 1 augmented training dataset.

    The original files are never changed. A grouped train/validation split is
    created first; only training IDs receive the ten fixed augmentation
    variants plus one deterministic base copy (ten samples per source image).
    Validation receives only the
    deterministic base preprocessing.
    """
    audit_task1_dataset(data_root, audit_path, size=size)
    _, train_manifest, val_manifest = create_split_manifests(
        data_root=data_root,
        task=1,
        output_dir=split_dir,
        audit_path=audit_path,
        val_fraction=val_fraction,
        seed=seed,
        max_hash_distance=max_hash_distance,
        include_review=True,
        max_sample_weight=5.0,
        size=size,
    )
    _, cv2, _ = _augmentation_dependencies()
    transforms = _fixed_task1_augmentation_transforms(size=size, seed=seed)
    validation_transform = build_transforms(size=size, seed=seed)[1]
    output_root = output_root.resolve()
    train_images = output_root / "train" / "images"
    train_masks = output_root / "train" / "task1_gt"
    val_images = output_root / "val" / "images"
    val_masks = output_root / "val" / "task1_gt"
    for directory in (train_images, train_masks, val_images, val_masks):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    def write_pair(image, mask, image_target: Path, mask_target: Path) -> None:
        image_ok = cv2.imwrite(
            str(image_target), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )
        mask_ok = cv2.imwrite(str(mask_target), (mask > 0).astype("uint8") * 255)
        if not image_ok or not mask_ok:
            raise OSError(f"Failed to write augmented pair for {image_target.stem}")

    for split_name, manifest, image_dir, mask_dir in (
        ("train", train_manifest, train_images, train_masks),
        ("val", val_manifest, val_images, val_masks),
    ):
        for row in _read_csv_rows(manifest):
            image_id = row["image_id"]
            image, mask = read_pair(data_root, image_id)
            selected_transforms = transforms if split_name == "train" else {"base": validation_transform}
            for variant, transform in selected_transforms.items():
                result = transform(image=image, mask=mask)
                sample_id = f"{image_id}__aug_{variant}"
                image_target = image_dir / f"{sample_id}.png"
                mask_target = mask_dir / f"{sample_id}_segmentation.png"
                write_pair(result["image"], result["mask"], image_target, mask_target)
                rows.append(
                    {
                        "source_image_id": image_id,
                        "split": split_name,
                        "augmentation": variant,
                        "image_path": str(image_target.relative_to(output_root)),
                        "mask_path": str(mask_target.relative_to(output_root)),
                    }
                )

    manifest_path = output_root / "prepared_task1_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    train_count = sum(row["split"] == "train" for row in rows)
    val_count = len(rows) - train_count
    print(
        f"Prepared fixed Task 1 dataset: train={train_count} samples, "
        f"val={val_count} samples, size={size}x{size}"
    )
    print(f"Prepared manifest: {manifest_path}")


def prepare_fixed_task2_training_set(
    data_root: Path,
    output_root: Path,
    audit_path: Path,
    split_dir: Path,
    size: int = DEFAULT_SIZE,
    val_fraction: float = 0.10,
    seed: int = 42,
    max_hash_distance: int = 2,
    reference_split_path: Path | None = None,
) -> None:
    """Create Task 2's fixed augmented set with synchronized five-mask transforms."""
    audit_task2_dataset(data_root, audit_path, size=size)
    _, train_manifest, val_manifest = create_split_manifests(
        data_root=data_root,
        task=2,
        output_dir=split_dir,
        audit_path=audit_path,
        val_fraction=val_fraction,
        seed=seed,
        max_hash_distance=max_hash_distance,
        include_review=True,
        max_sample_weight=5.0,
        size=size,
        reference_split_path=reference_split_path,
    )
    _, cv2, _ = _augmentation_dependencies()
    transforms = _fixed_task1_augmentation_transforms(size=size, seed=seed)
    validation_transform = build_transforms(size=size, seed=seed)[1]
    output_root = output_root.resolve()
    train_images = output_root / "train" / "images"
    train_masks = output_root / "train" / "task2_gt"
    val_images = output_root / "val" / "images"
    val_masks = output_root / "val" / "task2_gt"
    for directory in (train_images, train_masks, val_images, val_masks):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for split_name, manifest, image_dir, mask_dir in (
        ("train", train_manifest, train_images, train_masks),
        ("val", val_manifest, val_images, val_masks),
    ):
        for row in _read_csv_rows(manifest):
            image_id = row["image_id"]
            image, masks = read_task2_pair(data_root, image_id)
            selected_transforms = transforms if split_name == "train" else {"base": validation_transform}
            for variant, transform in selected_transforms.items():
                # Albumentations applies the exact same geometric operation to
                # every entry in ``masks``; this preserves Task 2 label alignment.
                result = transform(image=image, masks=masks)
                sample_id = f"{image_id}__aug_{variant}"
                image_target = image_dir / f"{sample_id}.png"
                image_ok = cv2.imwrite(
                    str(image_target), cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
                )
                if not image_ok:
                    raise OSError(f"Failed to write augmented image {image_target}")
                for attribute, mask in zip(TASK2_ATTRIBUTES, result["masks"]):
                    mask_target = mask_dir / f"{sample_id}_attribute_{attribute}.png"
                    if not cv2.imwrite(str(mask_target), (mask > 0).astype("uint8") * 255):
                        raise OSError(f"Failed to write augmented mask {mask_target}")
                rows.append(
                    {
                        "source_image_id": image_id,
                        "split": split_name,
                        "augmentation": variant,
                        "image_path": str(image_target.relative_to(output_root)),
                    }
                )

    manifest_path = output_root / "prepared_task2_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    train_count = sum(row["split"] == "train" for row in rows)
    val_count = len(rows) - train_count
    print(
        f"Prepared fixed Task 2 dataset: train={train_count} samples, "
        f"val={val_count} samples, size={size}x{size}"
    )
    print(f"Prepared manifest: {manifest_path}")


def prepare_fixed_training_sets(
    data_root: Path,
    output_root: Path,
    task1_audit_path: Path,
    task2_audit_path: Path,
    split_dir: Path,
    size: int = DEFAULT_SIZE,
    val_fraction: float = 0.10,
    seed: int = 42,
    max_hash_distance: int = 2,
) -> None:
    """Prepare leakage-safe fixed augmented datasets for Task 1 and Task 2."""
    prepare_fixed_task1_training_set(
        data_root, output_root / "task1", task1_audit_path, split_dir,
        size, val_fraction, seed, max_hash_distance,
    )
    prepare_fixed_task2_training_set(
        data_root, output_root / "task2", task2_audit_path, split_dir,
        size, val_fraction, seed, max_hash_distance,
        reference_split_path=split_dir / "task1_split.csv",
    )


def save_augmentation_preview(
    data_root: Path,
    image_id: str,
    output_path: Path,
    size: int,
    count: int,
    task: int = 1,
) -> None:
    """Save a Task 1 or Task 2 paired-augmentation contact sheet."""
    _, cv2, np = _augmentation_dependencies()
    train_transform, _ = build_transforms(size=size)
    if task == 1:
        image, mask = read_pair(data_root, image_id)
        transform_inputs = {"image": image, "mask": mask}
    elif task == 2:
        image, masks = read_task2_pair(data_root, image_id)
        transform_inputs = {"image": image, "masks": masks}
    else:
        raise ValueError("task must be 1 or 2")
    tiles = []

    for _ in range(count):
        result = train_transform(**transform_inputs)
        preview = result["image"].copy()
        if task == 1:
            preview_masks = [result["mask"]]
            colours = [(0, 255, 0)]
        else:
            preview_masks = result["masks"]
            colours = TASK2_PREVIEW_COLOURS

        for preview_mask, colour in zip(preview_masks, colours):
            binary_mask = (preview_mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(preview, contours, -1, colour, thickness=2)
        tiles.append(preview)

    columns = min(3, count)
    rows = (count + columns - 1) // columns
    blank = np.zeros_like(tiles[0])
    tiles.extend([blank] * (rows * columns - count))
    contact_sheet = np.concatenate(
        [np.concatenate(tiles[row * columns : (row + 1) * columns], axis=1)
         for row in range(rows)],
        axis=0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(
        str(output_path), cv2.cvtColor(contact_sheet, cv2.COLOR_RGB2BGR)
    )
    if not success:
        raise OSError(f"Failed to write preview: {output_path}")
    print(f"Preview: {output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and augment Task 1/2 dermoscopy masks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="write a data audit CSV")
    audit_parser.add_argument("--task", type=int, choices=(1, 2), default=1)
    audit_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    audit_parser.add_argument("--output", type=Path, default=None)
    audit_parser.add_argument("--size", type=int, default=DEFAULT_SIZE)

    split_parser = subparsers.add_parser(
        "split", help="create clean, grouped and stratified train/val manifests"
    )
    split_parser.add_argument("--task", type=int, choices=(1, 2), required=True)
    split_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    split_parser.add_argument("--audit-path", type=Path, default=None)
    split_parser.add_argument("--output-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    split_parser.add_argument("--val-fraction", type=float, default=0.20)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--near-duplicate-distance", type=int, default=2)
    split_parser.add_argument("--exclude-review", action="store_true")
    split_parser.add_argument("--max-sample-weight", type=float, default=5.0)
    split_parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    split_parser.add_argument(
        "--reference-split",
        type=Path,
        default=None,
        help="reuse split/group assignments from another task manifest",
    )

    epoch_parser = subparsers.add_parser(
        "balanced-epoch", help="sample one weighted Task 2 training epoch"
    )
    epoch_parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_SPLIT_DIR / "task2_train.csv"
    )
    epoch_parser.add_argument("--output", type=Path, default=None)
    epoch_parser.add_argument("--count", type=int, default=None)
    epoch_parser.add_argument("--seed", type=int, default=42)

    preview_parser = subparsers.add_parser(
        "preview", help="save random paired augmentation previews"
    )
    preview_parser.add_argument("--task", type=int, choices=(1, 2), default=1)
    preview_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    preview_parser.add_argument("--image-id", default="000001")
    preview_parser.add_argument("--output", type=Path, default=None)
    preview_parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    preview_parser.add_argument("--count", type=int, default=6)

    prepare_parser = subparsers.add_parser(
        "prepare-training",
        aliases=["prepare-task1-training"],
        help="write fixed, fully augmented Task 1/2 train sets and clean val sets",
    )
    prepare_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    prepare_parser.add_argument(
        "--output-root", type=Path, default=PROJECT_DIR / "data" / "prepared"
    )
    prepare_parser.add_argument(
        "--audit-path", type=Path, default=DEFAULT_TASK1_AUDIT_PATH,
        help="Task 1 audit CSV path",
    )
    prepare_parser.add_argument(
        "--task2-audit-path", type=Path, default=DEFAULT_TASK2_AUDIT_PATH
    )
    prepare_parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    prepare_parser.add_argument("--size", type=int, default=256)
    prepare_parser.add_argument("--val-fraction", type=float, default=0.10)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--near-duplicate-distance", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        default_output = (
            DEFAULT_TASK1_AUDIT_PATH if args.task == 1 else DEFAULT_TASK2_AUDIT_PATH
        )
        audit_dataset(
            args.data_root,
            args.output or default_output,
            task=args.task,
            size=args.size,
        )
    elif args.command == "split":
        create_split_manifests(
            data_root=args.data_root,
            task=args.task,
            output_dir=args.output_dir,
            audit_path=args.audit_path,
            val_fraction=args.val_fraction,
            seed=args.seed,
            max_hash_distance=args.near_duplicate_distance,
            include_review=not args.exclude_review,
            max_sample_weight=args.max_sample_weight,
            size=args.size,
            reference_split_path=args.reference_split,
        )
    elif args.command == "balanced-epoch":
        if args.count is not None and args.count < 1:
            raise ValueError("--count must be at least 1")
        output_path = args.output or (
            args.manifest.parent / f"{args.manifest.stem}_balanced_epoch.csv"
        )
        write_balanced_epoch_manifest(
            args.manifest,
            output_path,
            count=args.count,
            seed=args.seed,
        )
    elif args.command == "preview":
        if args.count < 1:
            raise ValueError("--count must be at least 1")
        default_output = (
            DEFAULT_TASK1_PREVIEW_PATH
            if args.task == 1
            else DEFAULT_TASK2_PREVIEW_PATH
        )
        save_augmentation_preview(
            data_root=args.data_root,
            image_id=args.image_id,
            output_path=args.output or default_output,
            size=args.size,
            count=args.count,
            task=args.task,
        )
    elif args.command in {"prepare-training", "prepare-task1-training"}:
        if not 0.0 < args.val_fraction < 1.0:
            raise ValueError("--val-fraction must be between 0 and 1")
        prepare_fixed_training_sets(
            data_root=args.data_root,
            output_root=args.output_root,
            task1_audit_path=args.audit_path,
            task2_audit_path=args.task2_audit_path,
            split_dir=args.split_dir,
            size=args.size,
            val_fraction=args.val_fraction,
            seed=args.seed,
            max_hash_distance=args.near_duplicate_distance,
        )


if __name__ == "__main__":
    main()
