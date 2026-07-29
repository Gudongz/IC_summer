"""Render Task 2 curves without importing PyTorch in the plotting process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_preprocessing import TASK2_ATTRIBUTES


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Task 2 training curves.")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.history.open(encoding="utf-8-sig") as file:
        history = json.load(file)
    epochs = history["epoch"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].plot(epochs, history["train_total_loss"], label="Train")
    axes[0].plot(epochs, history["val_total_loss"], label="Validation")
    axes[0].set(title="Fixed-weight BCE + Focal Tversky loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_mean_dice"], label="Train")
    axes[1].plot(epochs, history["val_mean_dice"], label="Validation")
    axes[1].set(title="Mean attribute Dice", xlabel="Epoch", ylabel="Dice", ylim=(0, 1))
    axes[1].legend()
    for attribute in TASK2_ATTRIBUTES:
        axes[2].plot(epochs, history[f"val_dice_{attribute}"], label=attribute)
    axes[2].set(title="Validation Dice by attribute", xlabel="Epoch", ylabel="Dice", ylim=(0, 1))
    axes[2].legend(fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
