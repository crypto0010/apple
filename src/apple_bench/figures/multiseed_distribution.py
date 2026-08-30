"""Per-seed TRT INT8 val_acc distribution for MobileNetV3-Small.

Figure for the multi-seed methodology contribution in paper/main.tex.
A horizontal strip plot of the five per-seed accuracies with reference
lines for the FP32 anchor (1.000), the per-seed lower bound, and the
prior single-point estimate (0.8425) annotated with its standard-
deviation distance from the multi-seed mean.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from apple_bench.figures._style import PALETTE, figsize


def from_json(summary_path: Path, output_path: Path,
              prior_single_point: float = 0.8425) -> None:
    """Render the per-seed distribution figure from the multiseed summary JSON."""
    summary = json.loads(summary_path.read_text())
    seeds = [int(s["seed"]) for s in summary["per_seed"]]
    accs = np.asarray([float(s["val_acc"]) for s in summary["per_seed"]])
    mean = float(summary["val_acc_mean"])
    sd = float(summary["val_acc_sd"])
    lo = float(accs.min())

    sigma_distance_from_mean = abs(prior_single_point - mean) / sd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize(1.0, 0.5))

    ax.scatter(accs, [1] * len(accs), s=60, color=PALETTE[0],
               edgecolor="black", linewidth=0.5, zorder=3,
               label=f"per-seed val_acc (n={len(accs)})")

    ax.axvline(mean, color=PALETTE[0], linestyle="--", linewidth=1.0,
               label=f"multi-seed mean = {mean:.4f}")
    ax.axvspan(mean - sd, mean + sd, color=PALETTE[0], alpha=0.12,
               label=f"$\\pm$1 SD ({sd:.4f})")
    ax.axvline(lo, color="gray", linestyle=":", linewidth=0.8,
               label=f"per-seed lower bound = {lo:.4f}")
    ax.axvline(1.000, color="black", linestyle="-", linewidth=0.8,
               label="FP32 anchor = 1.000")
    ax.axvline(prior_single_point, color=PALETTE[3], linestyle="-",
               linewidth=1.2,
               label=(f"prior single-point = {prior_single_point:.4f} "
                      f"({sigma_distance_from_mean:.2f}$\\sigma$ from mean)"))

    # Five seeds span ~0.03 accuracy on an axis ~0.18 wide (the FP32
    # anchor and the prior single-point estimate force the wide range),
    # so the seed tags collide badly on one level and still touch on
    # two. Three levels in accuracy order put same-level neighbours two
    # ranks apart, which clears them at this figure width.
    _TAG_DY = (10, -17, 23)
    for rank, (a, s) in enumerate(sorted(zip(accs, seeds, strict=True))):
        ax.annotate(str(s), (a, 1), textcoords="offset points",
                    xytext=(0, _TAG_DY[rank % 3]),
                    ha="center", fontsize=6.5)

    ax.set_xlabel("MobileNetV3-Small TRT INT8 val accuracy")
    ax.set_yticks([])
    ax.set_ylim(0.5, 1.5)
    x_lo = min(float(accs.min()), prior_single_point) - 0.01
    x_hi = max(float(accs.max()), 1.000) + 0.01
    ax.set_xlim(x_lo, x_hi)
    # Legend below the axes: frameless legends (global style) printed
    # inside the axes overlap the strip of per-seed points.
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.34),
              fontsize=7, ncol=2)

    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)
