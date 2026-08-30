"""Per-architecture flip-risk plot using directly-measured sigma_s.

Figure for paper3/main.tex. Each architecture's |Delta_pp - tau|
(gate-margin) is shown as a horizontal bar; its 2*sigma_s envelope is
overlaid as an error bar. Architectures whose envelope crosses zero
(i.e. |Delta - tau| < 2 sigma_s) are flip-risk; others are safe.

Per the actual multi-seed sweep, 0 of 6 architectures are at flip-risk
under directly-measured per-architecture sigma. This INVALIDATES Paper
2's "5 of 6" claim, which used the inherited sigma = 1.14 pp from
MobileNetV3-Small for all rows.

Layout choices (v2): architecture names are on the y-axis (no inline
labels overlapping the bars), and the numeric annotation
"|Delta-tau|=X, 2sigma=Y" is placed at the right edge of the plot
on each row so the data and the labels never collide.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from apple_bench.figures._style import PALETTE, display_name, figsize

TAU_GATE = 2.0
MODELS = [
    "mobilenet_v3_small",
    "efficientnet_lite0",
    "yolov8_cls_n",
    "paper3_efficientnet_b0_gmp",
    "paper4_resnet50",
    "paper2_econv_vit",
]


def from_summaries(runs_root: Path, output_path: Path) -> None:
    """Render the per-architecture flip-risk plot."""
    rows = []
    for model in MODELS:
        d = json.loads(
            (runs_root / f"{model}_int8_multiseed_summary.json").read_text()
        )
        sigma_pp = d["val_acc_sd"] * 100.0
        # Magnitude, matching the Delta_pp column of the flip-risk tables
        # in the papers that embed this figure. Note the summary JSON uses
        # negative-is-loss, so paper2_econv_vit's +14.95 is an accuracy
        # *gain* over a collapsed FP32 anchor, not a regression; the table
        # captions flag that row. The flip-risk verdict is unaffected.
        delta_pp = abs(d["delta_pp_vs_fp32_mean"])
        margin = abs(delta_pp - TAU_GATE)
        flip_risk = margin <= 2 * sigma_pp
        rows.append({
            "model": model,
            "delta_pp": delta_pp,
            "margin": margin,
            "sigma_pp": sigma_pp,
            "flip_risk": flip_risk,
        })

    rows.sort(key=lambda r: r["margin"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize(1.0, 0.75))

    x_max = max(r["margin"] + 2 * r["sigma_pp"] for r in rows) * 1.05
    annotation_x = x_max * 1.02

    y_pos = list(range(len(rows)))
    for i, r in enumerate(rows):
        bar_color = PALETTE[3] if r["flip_risk"] else PALETTE[0]
        ax.barh(i, r["margin"], color=bar_color, edgecolor="black",
                linewidth=0.5, alpha=0.85, zorder=3)
        ax.errorbar(r["margin"], i, xerr=2 * r["sigma_pp"], fmt="none",
                    ecolor="black", capsize=4, capthick=1.0, zorder=4)
        ax.text(annotation_x, i,
                f"$|\\Delta-\\tau|={r['margin']:.2f}$, "
                f"$2\\sigma_s={2*r['sigma_pp']:.2f}$",
                va="center", ha="left", fontsize=7)

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8,
               label=r"gate boundary (flip-risk if $|\Delta-\tau|<2\sigma_s$)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([display_name(r["model"]) for r in rows], fontsize=8)
    ax.set_xlabel(r"$|\Delta_{\mathrm{pp}}-\tau|$ (gate-margin, pp)")
    ax.set_xlim(-0.5, x_max * 1.55)
    # Frameless legends (_style.py sets legend.frameon False) printed
    # inside the axes overprint the per-bar gate-margin labels with nothing
    # behind them to separate the two; place the legend below the axes.
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), fontsize=7)

    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)
