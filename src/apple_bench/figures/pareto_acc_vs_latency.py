"""Accuracy vs latency Pareto plot.

X-axis: best_p50_ms (log scale). Y-axis: accuracy_pv. One marker per
(model, format) row in the edge CSV. The Pareto frontier (lowest
latency for each accuracy band) is drawn as a step line through the
dominant points.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from apple_bench.analysis.pareto import pareto_frontier_indices
from apple_bench.figures._style import (  # noqa: F401 -- import for rcParams side effects
    PALETTE,
    display_name,
    figsize,
)

# Distinct marker per format so the figure is readable in black-and-white prints.
FORMAT_MARKERS = {
    "pytorch_fp32": "o",
    "onnxruntime_fp32": "s",
    "tensorrt_fp32": "^",
    "tensorrt_fp16": "D",
    "tensorrt_int8": "v",
}


def render(rows: list[dict], output_path: Path) -> None:
    """Render the figure from in-memory edge rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Drop rows missing accuracy (e.g. a format that failed regression check).
    rows = [r for r in rows if r.get("accuracy_pv") not in ("", None, "None")]

    models = sorted({r["model"] for r in rows})
    model_color = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=figsize(1.4, 0.7))
    for r in rows:
        ax.scatter(
            float(r["p50_ms"]),
            float(r["accuracy_pv"]),
            marker=FORMAT_MARKERS.get(r["format"], "x"),
            color=model_color[r["model"]],
            s=42,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )

    # Pareto frontier across ALL points: minimize latency, maximize accuracy.
    cost = np.array([float(r["p50_ms"]) for r in rows])
    val = np.array([float(r["accuracy_pv"]) for r in rows])
    pareto_idx = pareto_frontier_indices(cost, val)
    # Sort the Pareto points by cost so the step line goes left-to-right.
    pareto_idx.sort(key=lambda i: cost[i])
    ax.plot(
        cost[pareto_idx], val[pareto_idx],
        drawstyle="steps-post",
        color="black", linewidth=1.0, alpha=0.7, zorder=1,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Median latency (ms, batch=1)")
    ax.set_ylabel("Accuracy (PlantVillage val)")
    ax.set_title("Accuracy vs. latency Pareto frontier (Orin Nano)")

    # Build legend handles for models (color) and formats (marker).
    model_handles = [
        plt.Line2D([], [], marker="o", color=model_color[m], linestyle="",
                   markersize=6, label=display_name(m))
        for m in models
    ]
    format_handles = [
        plt.Line2D([], [], marker=mk, color="gray", linestyle="",
                   markersize=6, label=fmt)
        for fmt, mk in FORMAT_MARKERS.items()
    ]
    leg1 = ax.legend(handles=model_handles, title="Model",
                     loc="lower right", fontsize=7, title_fontsize=7)
    ax.add_artist(leg1)
    ax.legend(handles=format_handles, title="Format",
              loc="lower left", fontsize=7, title_fontsize=7)

    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)


def from_csv(edge_csv: Path, output_path: Path) -> None:
    with edge_csv.open() as f:
        rows = list(csv.DictReader(f))
    render(rows, output_path)
