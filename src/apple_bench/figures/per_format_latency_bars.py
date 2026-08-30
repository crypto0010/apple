"""Per-format latency bar chart.

Grouped bars: x = model (sorted by TRT-INT8 latency ascending),
hue = format. Y-axis: p50 latency in ms (log scale, since the dynamic
range is two orders of magnitude across formats).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from apple_bench.figures._style import PALETTE, display_name, figsize  # noqa: F401

FORMAT_ORDER = [
    "pytorch_fp32",
    "onnxruntime_fp32",
    "tensorrt_fp32",
    "tensorrt_fp16",
    "tensorrt_int8",
]


def render(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a (model, format) -> p50_ms lookup, dropping missing rows.
    lookup: dict[tuple[str, str], float] = {}
    for r in rows:
        try:
            lookup[(r["model"], r["format"])] = float(r["p50_ms"])
        except (KeyError, ValueError, TypeError):
            continue

    models = sorted({m for (m, _f) in lookup})
    # Sort by TRT-INT8 latency if available, else by any minimum format.
    def _sort_key(m: str) -> float:
        if (m, "tensorrt_int8") in lookup:
            return lookup[(m, "tensorrt_int8")]
        return min(v for (mm, _f), v in lookup.items() if mm == m)
    models.sort(key=_sort_key)

    n_models = len(models)
    n_formats = len(FORMAT_ORDER)
    bar_width = 0.85 / n_formats
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=figsize(2.0, 0.55))
    for j, fmt in enumerate(FORMAT_ORDER):
        heights = [lookup.get((m, fmt), 0.0) for m in models]
        ax.bar(
            x + (j - n_formats / 2 + 0.5) * bar_width,
            heights,
            width=bar_width,
            label=fmt,
            color=PALETTE[j % len(PALETTE)],
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(m) for m in models],
                       rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("p50 latency (ms, batch=1, log scale)")
    ax.set_title("Per-format median latency on Orin Nano")
    ax.legend(title="Format", fontsize=7, title_fontsize=7,
              loc="upper left", ncol=2)

    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)


def from_csv(edge_csv: Path, output_path: Path) -> None:
    with edge_csv.open() as f:
        rows = list(csv.DictReader(f))
    render(rows, output_path)
