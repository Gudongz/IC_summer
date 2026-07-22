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
    """Load JSON settings and expose them as attributes."""
    with SETTINGS_PATH.open(encoding="utf-8") as file:
        values = json.load(file)

    for key in ("task1_input", "task1_gt", "task1_train_input", "task1_train_gt", "task1_val_input", "task1_val_gt", "task2_gt", "task1_output_folder", "task2_output_folder", "inference_ground_truth", "checkpoint_folder", "training_plot_path"):
        if values.get(key) is not None:
            values[key] = _project_path(values[key])
    model_name = values["model_name"]
    values["batch_size"] = values.get("model_batch_sizes", {}).get(model_name, values["batch_size"])
    values["learning_rate"] = values.get("model_learning_rates", {}).get(model_name, values["learning_rate"])
    values["checkpoint_folder"] = values["checkpoint_folder"] / model_name
    values["best_checkpoint_path"] = values["checkpoint_folder"] / "best.pt"
    values["training_plot_path"] = values["training_plot_path"] / model_name / "curves.png"
    values["task1_output_folder"] = values["task1_output_folder"] / model_name
    return SimpleNamespace(**values)


settings = load_settings()
