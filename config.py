"""Read the non-code project configuration from settings.json."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = PROJECT_ROOT / "settings.json"


def _project_path(value: str) -> Path:
    """Resolve a settings.json path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings() -> SimpleNamespace:
    """Load settings, select the active model profile, and resolve paths."""
    with SETTINGS_PATH.open(encoding="utf-8") as file:
        raw = json.load(file)

    model_name = raw["model_name"]
    try:
        profile = raw["models"][model_name]
    except KeyError as exc:
        available = ", ".join(raw.get("models", {}))
        raise ValueError(f"Unknown model_name {model_name!r}. Available models: {available}") from exc

    values = {
        **raw["data"],
        **raw["output"],
        **raw["training"],
        **raw["inference"],
        **raw["evaluation"],
        "model_name": model_name,
        "batch_size": profile["batch_size"],
        "learning_rate": profile["learning_rate"],
        "checkpoint_path": profile["checkpoint_path"],
        "pretrained": profile.get("pretrained", False),
        "encoder_learning_rate": profile.get("encoder_learning_rate"),
        "freeze_encoder_epochs": profile.get("freeze_encoder_epochs", 0),
        "loss": raw["training"].get("loss", {}),
        "inference_batch_size": raw["inference"]["batch_size"],
        "evaluation_batch_size": raw["evaluation"]["batch_size"],
    }
    values["model_profiles"] = {
        name: {
            **candidate,
            "checkpoint_path": _project_path(candidate["checkpoint_path"]),
        }
        for name, candidate in raw["models"].items()
    }
    loss = values["loss"]
    bce_decay = loss.get("bce_weight_decay", {})
    values["bce_weight"] = float(loss.get("bce_weight", 1.0))
    values["dice_weight"] = float(loss.get("dice_weight", 1.0))
    values["bce_weight_decay_enabled"] = bool(bce_decay.get("enabled", False))
    values["bce_weight_decay_target"] = float(bce_decay.get("target_weight", values["bce_weight"]))
    values["bce_weight_decay_epochs"] = int(bce_decay.get("decay_epochs", 0))
    if values["bce_weight"] < 0 or values["dice_weight"] < 0:
        raise ValueError("training.loss BCE and Dice weights must be non-negative.")
    if values["bce_weight_decay_target"] < 0 or values["bce_weight_decay_epochs"] < 0:
        raise ValueError("training.loss BCE decay target and decay_epochs must be non-negative.")
    for key in (
        "task1_input", "task1_gt", "task1_train_input", "task1_train_gt",
        "task1_val_input", "task1_val_gt", "checkpoint_path", "training_root",
        "prediction_root", "validation_input", "validation_ground_truth", "validation_manifest",
        "sample_input", "sample_ground_truth", "output_root",
    ):
        values[key] = _project_path(values[key])
    values["training_plot_path"] = values.pop("training_root") / model_name / "curves.png"
    values["task1_output_folder"] = values.pop("prediction_root") / model_name
    return SimpleNamespace(**values)


def load_task2_settings() -> SimpleNamespace:
    """Load the isolated Task 2 training configuration from settings.json."""
    with SETTINGS_PATH.open(encoding="utf-8") as file:
        raw = json.load(file)
    task2 = raw["task2"]
    model_name = task2["model_name"]
    try:
        profile = task2["models"][model_name]
    except KeyError as exc:
        available = ", ".join(task2.get("models", {}))
        raise ValueError(f"Unknown Task 2 model {model_name!r}. Available models: {available}") from exc
    values = {
        **task2["data"], **task2["training"], **task2["loss"], **task2["dynamic_weights"], **task2["output"],
        "model_name": model_name,
        "batch_size": profile["batch_size"],
        "learning_rate": profile["learning_rate"],
        "encoder_learning_rate": profile["encoder_learning_rate"],
        "freeze_encoder_epochs": profile["freeze_encoder_epochs"],
        "roi_enabled": bool(profile.get("roi_enabled", False)),
        "task1_checkpoint": profile["task1_checkpoint"],
        "checkpoint_path": profile["checkpoint_path"],
    }
    for key in ("train_input", "train_gt", "train_lesion_prior", "train_roi_mask", "val_input", "val_gt", "val_lesion_prior", "val_roi_mask", "train_manifest", "task1_checkpoint", "checkpoint_path", "training_root"):
        values[key] = _project_path(values[key])
    values["training_plot_path"] = values["training_root"] / model_name / "curves.png"
    attribute_loss = values.get("attribute_loss", {})
    if not isinstance(attribute_loss, dict):
        raise ValueError("task2.loss.attribute_loss must be an object keyed by attribute name.")
    values["attribute_loss"] = attribute_loss
    return SimpleNamespace(**values)


settings = load_settings()
