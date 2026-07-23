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
    for key in (
        "task1_input", "task1_gt", "task1_train_input", "task1_train_gt",
        "task1_val_input", "task1_val_gt", "checkpoint_path", "training_root",
        "prediction_root", "sample_input", "sample_ground_truth", "output_root",
    ):
        values[key] = _project_path(values[key])
    values["training_plot_path"] = values.pop("training_root") / model_name / "curves.png"
    values["task1_output_folder"] = values.pop("prediction_root") / model_name
    return SimpleNamespace(**values)


settings = load_settings()
