"""Render curves and the validation precision/recall confusion matrix for one Stage-1 decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one Task 2 decoder-pretraining report.")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attribute", required=True)
    args = parser.parse_args()
    with args.history.open(encoding="utf-8-sig") as file:
        history = json.load(file)
    epochs = history["epoch"]
    if not epochs:
        return

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(epochs, history["train_loss"], label="Train")
    axes[0, 0].plot(epochs, history["val_loss"], label="Validation")
    axes[0, 0].set(title=f"{args.attribute}: loss", xlabel="Epoch", ylabel="BCE + Focal Tversky loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history["train_dice"], label="Train")
    axes[0, 1].plot(epochs, history["val_dice"], label="Validation")
    axes[0, 1].set(title="Dice", xlabel="Epoch", ylabel="Dice", ylim=(0, 1))
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, history["train_precision"], label="Train precision")
    axes[1, 0].plot(epochs, history["train_recall"], label="Train recall")
    axes[1, 0].plot(epochs, history["val_precision"], "--", label="Validation precision")
    axes[1, 0].plot(epochs, history["val_recall"], "--", label="Validation recall")
    axes[1, 0].set(title="Precision and recall", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1, 0].legend(fontsize=8)

    count_keys = ("val_true_negative", "val_false_positive", "val_false_negative", "val_true_positive")
    if all(history.get(key) for key in count_keys):
        true_negative = history["val_true_negative"][-1]
        false_positive = history["val_false_positive"][-1]
        false_negative = history["val_false_negative"][-1]
        true_positive = history["val_true_positive"][-1]
        counts = np.asarray(((true_negative, false_positive), (false_negative, true_positive)), dtype=np.float64)
        row_totals = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0)
        image = axes[1, 1].imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        axes[1, 1].set(
            title=(f"Validation confusion matrix (epoch {epochs[-1]})\n"
                   f"Precision={history['val_precision'][-1]:.3f}, Recall={history['val_recall'][-1]:.3f}"),
            xlabel="Predicted class", ylabel="Actual class",
            xticks=(0, 1), xticklabels=("Negative", "Positive"),
            yticks=(0, 1), yticklabels=("Negative", "Positive"),
        )
        for row in range(2):
            for column in range(2):
                axes[1, 1].text(
                    column, row, f"{normalized[row, column]:.1%}\n(n={int(counts[row, column])})",
                    ha="center", va="center", color="white" if normalized[row, column] > 0.5 else "black",
                )
        figure.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Row-normalized rate")
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5, 0.5, "Validation confusion-matrix counts\nwill appear after the next epoch.",
            ha="center", va="center",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
