"""Pipeline schematic for the four PyTorch quantisation paths.

Figure for paper2/main.tex showing the four pipelines side-by-side as
boxes-and-arrows diagrams. Failure points are marked with a red FAIL
annotation.

Pipelines:
    1. Legacy FX QAT: Anchor -> prepare_qat_fx -> train -> convert_fx [FAIL]
    2. Legacy FX PTQ: Anchor -> prepare_fx -> calib -> convert_fx [FAIL]
    3. pt2e QAT: Anchor -> export -> prepare_qat_pt2e -> train -> convert_pt2e [FAIL]
    4. pt2e PTQ: Anchor -> export -> prepare_pt2e -> calib -> convert_pt2e (works)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from apple_bench.figures._style import PALETTE, figsize

PIPELINES = [
    ("Legacy FX QAT", [
        ("FP32 anchor", False),
        ("prepare_qat_fx", False),
        ("train", False),
        ("convert_fx", True),
    ]),
    ("Legacy FX PTQ", [
        ("FP32 anchor", False),
        ("prepare_fx", False),
        ("calibrate", False),
        ("convert_fx", True),
    ]),
    ("pt2e QAT", [
        ("FP32 anchor", False),
        ("export_for_training", False),
        ("prepare_qat_pt2e", False),
        ("train + convert", True),
    ]),
    ("pt2e PTQ", [
        ("FP32 anchor", False),
        ("export_for_training", False),
        ("prepare_pt2e", False),
        ("calibrate + convert", False),
    ]),
]

BOX_W = 0.17
BOX_H = 0.08
H_GAP = 0.03
ROW_GAP = 0.22
LEFT_PAD = 0.20
TOP_PAD = 0.92


def render(output_path: Path) -> None:
    """Render the pipeline schematic figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize(1.0, 0.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    for row_idx, (pipeline_label, stages) in enumerate(PIPELINES):
        y = TOP_PAD - row_idx * ROW_GAP

        ax.text(LEFT_PAD - 0.01, y, pipeline_label, ha="right", va="center",
                fontsize=8, fontweight="bold")

        for col_idx, (box_label, is_fail) in enumerate(stages):
            x = LEFT_PAD + col_idx * (BOX_W + H_GAP)
            box_color = PALETTE[3] if is_fail else PALETTE[0]
            alpha = 0.35 if is_fail else 0.18
            box = mpatches.FancyBboxPatch(
                (x, y - BOX_H / 2), BOX_W, BOX_H,
                boxstyle="round,pad=0.01",
                facecolor=box_color, alpha=alpha,
                edgecolor="black", linewidth=0.6,
            )
            ax.add_patch(box)
            ax.text(x + BOX_W / 2, y, box_label,
                    ha="center", va="center", fontsize=5.5)
            if is_fail:
                ax.text(x + BOX_W / 2, y - BOX_H / 2 - 0.025,
                        "FAIL",
                        ha="center", va="top", fontsize=6.5,
                        color="darkred", fontweight="bold")

            if col_idx < len(stages) - 1:
                arrow = mpatches.FancyArrowPatch(
                    (x + BOX_W, y), (x + BOX_W + H_GAP, y),
                    arrowstyle="->", mutation_scale=8,
                    color="black", linewidth=0.6,
                )
                ax.add_patch(arrow)

    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)
