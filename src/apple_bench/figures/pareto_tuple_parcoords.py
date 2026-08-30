"""Pareto-tuple parallel-coordinates plot.

Figure for paper3/main.tex. For each (model, backend) row in
jetson_orin_nano_results.csv, plot a polyline across four normalised
axes: Delta_pp (clamped to [0, 30] then min-max scaled), p50_ms,
energy_mj_per_inf, engine_size_kb. Lines coloured by model so
backend-progression within a model is visible.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from apple_bench.figures._style import PALETTE, display_name, figsize

AXES = [
    ("accuracy_delta_pp_vs_fp32", "$|\\Delta_{\\mathrm{pp}}|$ (pp)"),
    ("p50_ms", "$p_{50}$ latency (ms)"),
    ("energy_mj_per_inf", "energy (mJ/inf)"),
    ("engine_size_kb", "engine size (KB)"),
]


def from_csv(edge_csv: Path, output_path: Path) -> None:
    """Render the parallel-coordinates Pareto tuple."""
    rows = []
    with edge_csv.open() as f:
        for row in csv.DictReader(f):
            try:
                vals = [abs(float(row[key])) for key, _ in AXES]
            except (ValueError, KeyError):
                continue
            rows.append({
                "model": row["model"],
                "format": row["format"],
                "vals": vals,
            })

    # Min-max normalise each axis
    axis_vals = np.array([r["vals"] for r in rows])
    axis_min = axis_vals.min(axis=0)
    axis_max = axis_vals.max(axis=0)
    span = np.where(axis_max > axis_min, axis_max - axis_min, 1.0)
    normed = (axis_vals - axis_min) / span

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize(1.0, 0.7))

    models = sorted({r["model"] for r in rows})
    model_color = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    x_pos = list(range(len(AXES)))
    for i, r in enumerate(rows):
        ax.plot(x_pos, normed[i], color=model_color[r["model"]],
                alpha=0.55, linewidth=0.9,
                label=display_name(r["model"]) if r["format"] == "tensorrt_int8" else None)
        # mark INT8 rows with circles for visual emphasis
        if r["format"] == "tensorrt_int8":
            ax.scatter(x_pos, normed[i], color=model_color[r["model"]],
                       s=18, edgecolor="black", linewidth=0.4, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([label for _, label in AXES], fontsize=7,
                       rotation=20, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("min-max normalised value (lower = better)", fontsize=7)
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.85,
              title="model (INT8 rows marked $\\bullet$)",
              title_fontsize=6.5)
    for x in x_pos:
        ax.axvline(x, color="gray", linewidth=0.4, alpha=0.4)

    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)
