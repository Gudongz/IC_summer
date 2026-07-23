"""Render Task 1 training curves in a standalone, non-PyTorch process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Task 1 training curves.")
    parser.add_argument("--history", type=Path, required=True, help="Path to the persisted metric history JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Destination PNG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.history.open(encoding="utf-8-sig") as file:
        history = json.load(file)

    epochs = history["epoch"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(epochs, history["train_total_loss"], label="Train")
    axes[0, 0].plot(epochs, history["val_total_loss"], label="Validation")
    axes[0, 0].set(title="Total loss (BCE + Dice loss)", xlabel="Epoch", ylabel="Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history["train_bce"], label="Train BCE")
    axes[0, 1].plot(epochs, history["val_bce"], label="Validation BCE")
    axes[0, 1].set(title="Binary cross-entropy", xlabel="Epoch", ylabel="BCE loss")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, history["train_dice"], label="Train Dice")
    axes[1, 0].plot(epochs, history["val_dice"], label="Validation Dice")
    axes[1, 0].set(title="Dice coefficient", xlabel="Epoch", ylabel="Dice", ylim=(0, 1))
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, history["val_hd"], label="Validation HD")
    axes[1, 1].plot(epochs, history["val_hd95"], label="Validation HD95")
    axes[1, 1].set(title="Boundary distances", xlabel="Epoch", ylabel="Pixels")
    axes[1, 1].legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
