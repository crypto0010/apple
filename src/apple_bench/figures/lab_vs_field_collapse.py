"""Lab vs field generalization gap.

Two bars per model: PlantVillage val (lab, clean studio images) and
Plant Pathology 2021 (field, in-the-wild). Annotates the per-model
generalization gap in percentage points. Reproduces Paper 2's gap
visualization at the benchmark scale.

Models with no PP2021 score (e.g. classical pipeline not trained, or
field eval was skipped) are dropped from the figure.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from apple_bench.figures._style import PALETTE, display_name, figsize  # noqa: F401


def render(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep only rows with both a val acc and a PP2021 acc.
    usable = []
    for r in rows:
        val = r.get("val_acc_pv")
        field = r.get("acc_pp2021")
        if val in (None, "", "None") or field in (None, "", "None"):
            continue
        try:
            usable.append((r["model"], float(val), float(field)))
        except (TypeError, ValueError):
            continue

    # Sort by val accuracy descending so the "perfectly trained" models go left.
    usable.sort(key=lambda t: -t[1])

    models = [m for m, _v, _f in usable]
    val = [v for _m, v, _f in usable]
    field = [f for _m, _v, f in usable]
    x = np.arange(len(models))
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=figsize(1.8, 0.65))
    ax.bar(x - bar_width / 2, val, width=bar_width,
           color=PALETTE[0], edgecolor="black", linewidth=0.4,
           label="PlantVillage (lab)")
    ax.bar(x + bar_width / 2, field, width=bar_width,
           color=PALETTE[3], edgecolor="black", linewidth=0.4,
           label="Plant Pathology 2021 (field)")

    # Annotate per-model gap in pp above the higher of the two bars.
    for i, (v, f) in enumerate(zip(val, field, strict=True)):
        gap_pp = (v - f) * 100
        top = max(v, f) + 0.04
        ax.text(x[i], top, f"−{gap_pp:.1f} pp",
                ha="center", va="bottom", fontsize=7, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([display_name(m) for m in models],
                       rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.15)
    ax.set_title("Lab vs. field generalization gap")
    ax.legend(loc="upper right", fontsize=7)

    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)


def from_csv(parity_csv: Path, output_path: Path) -> None:
    with parity_csv.open() as f:
        rows = list(csv.DictReader(f))
    render(rows, output_path)
