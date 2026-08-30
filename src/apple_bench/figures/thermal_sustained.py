"""Sustained-throughput thermal plot.

Subplot grid: one panel per model. Each panel shows FPS (left axis)
and tj (right axis) vs minutes. Annotates the timestamp where FPS
drops below 90 % of the initial value (thermal-throttle onset).

Input: a mapping ``model -> list[row]`` where each row has the columns
written by ``scripts/06_thermal_run.py`` (t_seconds, infs_in_window,
fps, tj_C, gr3d_freq_pct).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from apple_bench.figures._style import PALETTE, display_name, figsize  # noqa: F401


def render(model_to_rows: dict[str, list[dict]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = sorted(model_to_rows.keys())
    n = len(models)
    cols = 2
    rows = max(1, math.ceil(n / cols))

    fig, axes = plt.subplots(
        rows, cols, figsize=figsize(2.2, 0.55 * rows / max(1, cols / 2)),
        sharex=True,
    )
    axes = np.atleast_2d(axes)

    for i, model in enumerate(models):
        ax = axes[i // cols][i % cols]
        rows_for = model_to_rows[model]
        if not rows_for:
            ax.set_title(f"{display_name(model)}\n(no data)", fontsize=8)
            continue

        # Each row represents one window of length t_seconds; assume 60 s
        # windows (the default sample_period_s) → x in minutes is row index + 1.
        minutes = np.array([i + 1 for i in range(len(rows_for))])
        fps = np.array([float(r["fps"]) for r in rows_for])
        tj = np.array([
            float(r["tj_C"]) if r.get("tj_C") not in ("", "nan", None) else float("nan")
            for r in rows_for
        ])

        ax.plot(minutes, fps, color=PALETTE[0], linewidth=1.4, label="FPS")
        ax.set_xlabel("Minutes")
        ax.set_ylabel("FPS", color=PALETTE[0])
        ax.tick_params(axis="y", labelcolor=PALETTE[0])

        ax2 = ax.twinx()
        ax2.plot(minutes, tj, color=PALETTE[3], linestyle="--",
                 linewidth=1.0, label="tj (°C)")
        ax2.set_ylabel("tj (°C)", color=PALETTE[3])
        ax2.tick_params(axis="y", labelcolor=PALETTE[3])
        # Twinx adds the right spine back; turn it off again.
        ax2.spines["top"].set_visible(False)

        # Throttle onset: first minute where FPS < 90 % of initial.
        initial = fps[0]
        throttle_idx = next(
            (k for k, v in enumerate(fps) if v < 0.9 * initial),
            None,
        )
        if throttle_idx is not None:
            ax.axvline(minutes[throttle_idx], color="red",
                       linestyle=":", linewidth=0.8, alpha=0.6)
            ax.text(
                minutes[throttle_idx] + 0.5, initial * 0.85,
                f"−10% @ {minutes[throttle_idx]}m",
                fontsize=7, color="red",
            )

        ax.set_title(display_name(model), fontsize=8)

    # Hide any unused subplots in the grid.
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle("Sustained throughput + junction temperature (30 min)",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)


def from_csvs(model_dirs: dict[str, Path], output_path: Path) -> None:
    """Read each model's thermal CSV (typically
    ``runs/full/<model>/thermal_30min.csv``) and call render.
    """
    model_to_rows: dict[str, list[dict]] = {}
    for model, csv_path in model_dirs.items():
        if not csv_path.is_file():
            model_to_rows[model] = []
            continue
        with csv_path.open() as f:
            model_to_rows[model] = list(csv.DictReader(f))
    render(model_to_rows, output_path)
