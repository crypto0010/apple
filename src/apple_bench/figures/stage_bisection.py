"""Stage-bisection trajectories for four PyTorch INT8 conversion paths.

Figure for paper/main.tex showing val_acc through the four-stage
diagnostic pipeline:

  S1 : FP32 anchor
  S2 : post-prepare (untrained; observers attached, scales at defaults)
  S2b: post-50-batch calibration
  S3 : post-convert (final INT8 model)

Four lines are overlaid: legacy FX QAT, legacy FX PTQ, pt2e QAT,
pt2e PTQ. A horizontal reference line marks the TRT vendor PTQ
multi-seed mean (0.9131) from runs/full/.../multiseed_summary.json.

Source data points are reproduced from:
    scripts/debug_qat_stages.py
    scripts/debug_ptq_stages.py
    scripts/debug_pt2e_stages.py
    scripts/debug_pt2e_ptq.py

Each script writes per-stage val_acc to stdout deterministically.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from apple_bench.figures._style import PALETTE, figsize

STAGES = ["S1", "S2", "S2b", "S3"]

PATHS = {
    "legacy FX QAT":  [1.0000, 0.5512, 0.3937, 0.0866],
    "legacy FX PTQ":  [1.0000, 1.0000, 1.0000, 0.0866],
    "pt2e QAT":       [1.0000, 0.6472, 0.9024, None],   # S3 raises KeyError
    "pt2e PTQ":       [1.0000, 1.0000, 1.0000, 0.6016],
}

TRT_PTQ_REFERENCE = 0.9131  # 5-seed mean from multiseed_summary.json
FP32_REFERENCE = 1.0000

# Y-anchor for "KeyError" annotation on missing-data stages.
# Placed slightly above the bottom of the plot (set_ylim lower bound is -0.02).
MISSING_Y = 0.05


def render(output_path: Path) -> None:
    """Render the bisection trajectories figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize(1.0, 0.7))

    x = list(range(len(STAGES)))
    for i, (label, ys) in enumerate(PATHS.items()):
        xs_present = [xv for xv, y in zip(x, ys, strict=True) if y is not None]
        ys_present = [y for y in ys if y is not None]
        ax.plot(xs_present, ys_present, marker="o", linewidth=1.4,
                color=PALETTE[i], label=label)
        # Mark missing-data points (pt2e QAT S3 KeyError) with an x.
        for xv, y in zip(x, ys, strict=True):
            if y is None:
                ax.annotate("KeyError", (xv, MISSING_Y),
                            ha="center", va="bottom", fontsize=7,
                            color=PALETTE[i])
                ax.plot([xv], [MISSING_Y], marker="x", color=PALETTE[i],
                        markersize=10, mew=2)

    ax.axhline(TRT_PTQ_REFERENCE, color="black", linestyle="--",
               linewidth=0.8,
               label=f"TRT vendor PTQ mean = {TRT_PTQ_REFERENCE:.4f}")
    ax.axhline(FP32_REFERENCE, color="gray", linestyle=":", linewidth=0.8,
               label=f"FP32 anchor = {FP32_REFERENCE:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(STAGES)
    ax.set_xlabel("pipeline stage")
    ax.set_ylabel("val accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.2, len(STAGES) - 0.5)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)

    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)
