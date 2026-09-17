from __future__ import annotations

from pathlib import Path
import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aerodepth-matplotlib")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from .data import MidAirDataset


def visualize_sample(dataset: MidAirDataset, output: str | Path, index: int = 0, prediction: torch.Tensor | None = None) -> None:
    sample = dataset[index]
    rgb = sample["rgb"].permute(1, 2, 0).numpy()
    depth = sample["depth"].squeeze().numpy()
    valid = sample["valid"].squeeze().numpy()
    columns = 4 if prediction is not None else 2
    fig, axes = plt.subplots(1, columns, figsize=(4 * columns, 3))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    depth_plot = np.where(valid, depth, np.nan)
    im = axes[1].imshow(depth_plot, cmap="magma")
    axes[1].set_title("Ground-truth metric depth")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    if prediction is not None:
        pred = prediction.detach().cpu().squeeze().numpy()
        axes[2].imshow(pred, cmap="magma")
        axes[2].set_title("Predicted depth")
        axes[3].imshow(np.where(valid, np.abs(pred - depth), np.nan), cmap="inferno")
        axes[3].set_title("Absolute error")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def generate_report(input_dir: str | Path, output_dir: str | Path) -> list[str]:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads((input_dir / "results.json").read_text(encoding="utf-8"))["results"]
    names = [r["policy"] for r in data]
    calls = [r["inference_calls"] for r in data]
    skips = [100 * r["skip_rate"] for r in data]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, calls)
    axes[0].set_ylabel("Neural inference calls")
    axes[1].bar(names, skips)
    axes[1].set_ylabel("Skip rate (%)")
    fig.tight_layout()
    comparison = output_dir / "policy_comparison.png"
    fig.savefig(comparison, dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.scatter(skips, [r["metrics"]["absrel"] for r in data])
    for x, y, name in zip(skips, [r["metrics"]["absrel"] for r in data], names):
        axis.annotate(name, (x, y))
    axis.set_xlabel("Inference-call reduction / skip rate (%)")
    axis.set_ylabel("AbsRel")
    fig.tight_layout()
    tradeoff = output_dir / "quality_efficiency.png"
    fig.savefig(tradeoff, dpi=150)
    plt.close(fig)
    adaptive_log = input_dir / "adaptive_frames.csv"
    produced = [str(comparison), str(tradeoff)]
    if adaptive_log.exists():
        with adaptive_log.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        x = np.arange(len(rows))
        fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True)
        for axis, field in zip(axes[:4], ("motion", "visual", "risk", "age_after")):
            axis.plot(x, [float(row[field]) for row in rows])
            axis.set_ylabel(field)
        inferred = [i for i, row in enumerate(rows) if row["infer"].lower() == "true"]
        forced = [i for i, row in enumerate(rows) if row["max_skip_forced"].lower() == "true"]
        axes[4].scatter(inferred, np.ones(len(inferred)), label="infer", marker="|")
        axes[4].scatter(forced, np.zeros(len(forced)), label="MAX_SKIP", marker="x")
        axes[4].legend()
        axes[4].set_xlabel("Replay frame")
        fig.tight_layout()
        timeline = output_dir / "scheduler_timeline.png"
        fig.savefig(timeline, dpi=150)
        plt.close(fig)
        produced.append(str(timeline))
    return produced


def plot_history(history_path: str | Path, output: str | Path) -> None:
    with Path(history_path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(r["epoch"]) for r in rows]
    plt.plot(epochs, [float(r["train_loss"]) for r in rows], label="train")
    plt.plot(epochs, [float(r["val_loss"]) for r in rows], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
