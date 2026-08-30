"""Accuracy vs energy Pareto plot.

Same structure as pareto_acc_vs_latency, but the cost axis is
``energy_mj_per_inf`` (mJ per inference). Highlights that the
lowest-latency model is not always the lowest-energy one — useful
when battery life matters more than wall-clock latency.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from apple_bench.analysis.pareto import pareto_frontier_indices
from apple_bench.figures._style import PALETTE, display_name, figsize  # noqa: F401
from apple_bench.figures.pareto_acc_vs_latency import FORMAT_MARKERS


def render(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        r for r in rows
        if r.get("accuracy_pv") not in ("", None, "None")
        and r.get("energy_mj_per_inf") not in ("", None, "None")
    ]

    models = sorted({r["model"] for r in rows})
    model_color = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=figsize(1.4, 0.7))
    for r in rows:
        ax.scatter(
            float(r["energy_mj_per_inf"]),
            float(r["accuracy_pv"]),
            marker=FORMAT_MARKERS.get(r["format"], "x"),
            color=model_color[r["model"]],
            s=42,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )

    cost = np.array([float(r["energy_mj_per_inf"]) for r in rows])
    val = np.array([float(r["accuracy_pv"]) for r in rows])
    pareto_idx = pareto_frontier_indices(cost, val)
    pareto_idx.sort(key=lambda i: cost[i])
    ax.plot(
        cost[pareto_idx], val[pareto_idx],
        drawstyle="steps-post",
        color="black", linewidth=1.0, alpha=0.7, zorder=1,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Energy (mJ / inference, batch=1)")
    ax.set_ylabel("Accuracy (PlantVillage val)")
    ax.set_title("Accuracy vs. energy Pareto frontier (Orin Nano)")

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
