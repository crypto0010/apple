"""Per-architecture multi-seed sigma_s bar chart.

Figure for paper3/main.tex showing the per-architecture per-seed
standard deviation (sigma_s) of TRT INT8 val accuracy on the
PV apple subset (n=635). Horizontal bars, sorted by sigma_s
descending. Architectures with sigma_s = 0 get a "0.00 pp"
annotation so they are visible despite the zero-length bar.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from apple_bench.figures._style import PALETTE, display_name, figsize

MODELS = [
    "mobilenet_v3_small",
    "efficientnet_lite0",
    "yolov8_cls_n",
    "paper3_efficientnet_b0_gmp",
    "paper4_resnet50",
    "paper2_econv_vit",
]


def from_summaries(runs_root: Path, output_path: Path) -> None:
    """Render the per-architecture sigma bar chart.

    Reads each architecture's multiseed_summary.json from runs/full/
    and plots sigma in pp units (sigma * 100).
    """
    rows = []
    for model in MODELS:
        summary_path = runs_root / f"{model}_int8_multiseed_summary.json"
        d = json.loads(summary_path.read_text())
        rows.append({
            "model": model,
            "sigma_pp": d["val_acc_sd"] * 100.0,
            "delta_pp": d["delta_pp_vs_fp32_mean"],
        })
    rows.sort(key=lambda r: r["sigma_pp"], reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize(1.0, 0.7))

    labels = [display_name(r["model"]) for r in rows]
    sigmas = [r["sigma_pp"] for r in rows]
    y_pos = list(range(len(rows)))

    ax.barh(y_pos, sigmas, color=PALETTE[0], edgecolor="black",
            linewidth=0.5, alpha=0.85, zorder=3)

    for i, s in enumerate(sigmas):
        text_x = s + 0.05 if s > 0 else 0.02
        ax.text(text_x, i, f"  {s:.2f} pp", va="center", fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    # Observed: with the spelled-out "(percentage points)" the saved PDF
    # rendered the label truncated mid-word ("...percentage point").
    # Shortening to "pp" resolved it and matches both the bar labels and
    # the notation tables of the papers that embed this figure.
    ax.set_xlabel("per-seed standard deviation $\\sigma_s$ (pp)")
    ax.set_xlim(0, max(sigmas) * 1.4 if max(sigmas) > 0 else 1.0)
    ax.invert_yaxis()

    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)
