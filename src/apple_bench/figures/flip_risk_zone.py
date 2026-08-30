"""Per-architecture flip-risk visualisation.

Figure for paper2/main.tex showing each architecture's single-point
INT8 accuracy regression against the formalised flip-risk zone
[tau - k*sigma_s, tau + k*sigma_s]. Architectures whose |Delta_pp - tau|
falls inside the zone are flagged because their single-seed pass/fail
verdict against the MLPerf 2 pp regression gate is sensitive to
calibration draw.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from apple_bench.figures._style import PALETTE, display_name, figsize

TAU_GATE = 2.0      # MLPerf regression-gate threshold (pp)
SIGMA_S = 1.14      # per-seed sd from paper2 Table I aggregate row
K_SIGMA = 2.0       # k-sigma envelope multiplier. k=2 is the
                    # informal 95% confidence zone used in the
                    # published figure (see paper2 sec:flip-risk).
                    # k=1 catches no rows in the present dataset; k=2
                    # catches 5 of 6.


def from_csv(edge_csv: Path, output_path: Path) -> None:
    """Render the flip-risk zone figure from the Jetson edge CSV."""
    rows = []
    with edge_csv.open() as f:
        for row in csv.DictReader(f):
            if row["format"] == "tensorrt_int8":
                rows.append({
                    "model": row["model"],
                    "delta": float(row["accuracy_delta_pp_vs_fp32"]),
                })

    rows.sort(key=lambda r: r["delta"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize(1.0, 0.78))

    zone_lo = TAU_GATE - K_SIGMA * SIGMA_S
    zone_hi = TAU_GATE + K_SIGMA * SIGMA_S
    ax.axvspan(zone_lo, zone_hi, color=PALETTE[3], alpha=0.18,
               label=f"flip-risk zone $\\tau \\pm {K_SIGMA:.0f}\\sigma_s$ "
                     f"= [{zone_lo:.2f}, {zone_hi:.2f}] pp")
    ax.axvline(TAU_GATE, color="black", linestyle="--", linewidth=0.8,
               label=f"MLPerf gate $\\tau$ = {TAU_GATE:.1f} pp")

    # Anchor every label at a common x beyond the shaded zone rather than
    # at the bar tip: short bars end inside the zone, so tip-anchored text
    # would start on top of its own bar and run across the gate line.
    annotation_x = zone_hi + 0.6

    for i, r in enumerate(rows):
        in_zone = abs(r["delta"] - TAU_GATE) <= K_SIGMA * SIGMA_S
        bar_color = PALETTE[3] if in_zone else PALETTE[0]
        ax.barh(i, r["delta"], color=bar_color, edgecolor="black",
                linewidth=0.5, alpha=0.85, zorder=3)
        label = display_name(r["model"])
        ax.text(max(r["delta"], annotation_x), i,
                f"  {label}: {r['delta']:+.2f} pp",
                va="center", fontsize=7, ha="left")

    ax.set_yticks([])
    ax.set_xlabel("INT8 accuracy regression $\\Delta_{\\mathrm{pp}}$ "
                  "(positive = loss)")
    ax.set_xlim(-2, 30)
    # Legend sits below the axes: the global style disables legend
    # frames (_style.py sets legend.frameon False), so an in-axes legend
    # printed over the bar labels with nothing behind it to separate them.
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16),
              fontsize=7, ncol=1)

    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)
