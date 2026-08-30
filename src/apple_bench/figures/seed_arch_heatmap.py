"""Seed x architecture heatmap of val_acc deviations from per-arch mean.

Figure for paper3/main.tex. Each cell shows (val_acc[seed, arch] -
mean_val_acc[arch]) in pp units; diverging colormap with zero centred.
Designed to reveal any across-architecture seed effect: if seed=42
consistently produces lower-than-mean accuracy across multiple
architectures, that row should be reddish across the board.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from apple_bench.figures._style import display_name, figsize

SEEDS = [1729, 42, 7, 2024, 511]
MODELS = [
    "mobilenet_v3_small",
    "efficientnet_lite0",
    "yolov8_cls_n",
    "paper3_efficientnet_b0_gmp",
    "paper4_resnet50",
    "paper2_econv_vit",
]


def from_summaries(runs_root: Path, output_path: Path) -> None:
    """Render the seed x architecture deviation heatmap."""
    matrix = np.zeros((len(SEEDS), len(MODELS)))
    for j, model in enumerate(MODELS):
        d = json.loads(
            (runs_root / f"{model}_int8_multiseed_summary.json").read_text()
        )
        mean = d["val_acc_mean"]
        seed_to_acc = {s["seed"]: s["val_acc"] for s in d["per_seed"]}
        for i, seed in enumerate(SEEDS):
            matrix[i, j] = (seed_to_acc[seed] - mean) * 100.0  # pp

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize(1.0, 0.6))

    vmax = max(abs(matrix.min()), abs(matrix.max()))
    if vmax == 0:
        vmax = 0.1
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")

    for i in range(len(SEEDS)):
        for j in range(len(MODELS)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if abs(v) > vmax * 0.5 else "black")

    ax.set_yticks(range(len(SEEDS)))
    ax.set_yticklabels([f"$s$={s}" for s in SEEDS], fontsize=7)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([display_name(m) for m in MODELS],
                       fontsize=6.5, rotation=30, ha="right")
    ax.set_xlabel("")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("val accuracy deviation from per-arch mean (pp)",
                   fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)
