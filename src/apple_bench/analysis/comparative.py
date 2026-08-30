"""Headline comparative table: papers' claims vs our measurements.

A central artifact of the paper: one row per model with
(claimed PV acc, ours PV acc, Δ, ours field acc, claimed deployable,
actually deployable on Orin, best edge format, best p50 ms, mean board mW).

Claimed numbers are constants extracted from the four 2025 papers and
verified against the PDFs + ``docs/reproduction-report.md``. Measured
numbers come from A.1's parity_table.csv and A.2's
jetson_orin_nano_results.csv.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

# Hard-coded claims from the four 2025 papers; verified against
# docs/reproduction-report.md (Phase A.1 reproduction context).
PAPER_CLAIMS: dict[str, dict] = {
    "paper1_aco_svm": {
        "citation": "Li et al. 2025 (Smart Agric. Tech.)",
        "claimed_acc_pv": 0.925,
        "claimed_deployable": False,  # paper makes no edge claim
        "claimed_latency_ms": None,
    },
    "paper2_econv_vit": {
        "citation": "Huang et al. 2025 (Info. Proc. in Agric.)",
        "claimed_acc_pv": 0.992,
        "claimed_acc_field": 0.793,  # only paper to test in-the-wild
        "claimed_deployable": False,
        "claimed_latency_ms": 33.0,  # on Tesla P40 (datacenter GPU)
    },
    "paper3_efficientnet_b0_gmp": {
        "citation": "Ali et al. 2025 (Sci. Reports)",
        "claimed_acc_pv": 0.9978,
        "claimed_deployable": True,  # claims drone/sprayer suitability
        "claimed_latency_ms": None,
    },
    "paper4_resnet50": {
        "citation": "Rohith et al. 2025 (Multimedia Tools)",
        "claimed_acc_pv": 0.989,  # validation, no held-out test
        "claimed_deployable": False,  # real-time listed as future work
        "claimed_latency_ms": None,
    },
}


@dataclass
class ComparativeRow:
    model: str
    citation: str
    claimed_acc_pv: float
    ours_acc_pv: float | None
    delta_pv_pp: float | None
    ours_acc_pp2021: float | None
    ours_acc_al9: float | None
    claimed_deployable: bool
    actually_deployable: bool
    best_format: str | None
    best_p50_ms: float | None
    mean_power_mw: float | None


def _to_float(s: str | None) -> float | None:
    if s in (None, "", "None"):
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def build_comparative_table(
    parity_csv: Path,
    edge_csv: Path,
    deployable_threshold_ms: float = 100.0,
    accuracy_tolerance_pp: float = 2.0,
) -> list[ComparativeRow]:
    """Join parity and edge CSVs into one row per model in ``PAPER_CLAIMS``.

    ``best_format`` is the lowest-p50 row whose ``accuracy_delta_pp_vs_fp32``
    is within ``accuracy_tolerance_pp`` (matching the Phase A.2 regression
    gate). This avoids reporting an "INT8 best" for models like
    MobileNetV3-Small whose INT8 lost 15.75 pp.
    """
    with parity_csv.open() as f:
        parity = {row["model"]: row for row in csv.DictReader(f)}
    with edge_csv.open() as f:
        edge_rows = list(csv.DictReader(f))

    out: list[ComparativeRow] = []
    for model, claim in PAPER_CLAIMS.items():
        p = parity.get(model)
        # Filter edges to those whose INT8 regression passed (or that have
        # no delta because they ARE the FP32 anchor).
        candidates = []
        for r in edge_rows:
            if r["model"] != model:
                continue
            delta = _to_float(r.get("accuracy_delta_pp_vs_fp32"))
            if delta is None or delta <= accuracy_tolerance_pp:
                candidates.append(r)
        best = min(candidates, key=lambda r: float(r["p50_ms"])) if candidates else None

        ours_pv = _to_float(p["val_acc_pv"]) if p else None
        out.append(ComparativeRow(
            model=model,
            citation=claim["citation"],
            claimed_acc_pv=claim["claimed_acc_pv"],
            ours_acc_pv=ours_pv,
            delta_pv_pp=(
                (claim["claimed_acc_pv"] - ours_pv) * 100
                if ours_pv is not None else None
            ),
            ours_acc_pp2021=_to_float(p["acc_pp2021"]) if p else None,
            ours_acc_al9=_to_float(p["acc_al9"]) if p else None,
            claimed_deployable=claim["claimed_deployable"],
            actually_deployable=bool(
                best and float(best["p50_ms"]) < deployable_threshold_ms
            ),
            best_format=best["format"] if best else None,
            best_p50_ms=float(best["p50_ms"]) if best else None,
            mean_power_mw=_to_float(best.get("mean_power_mw")) if best else None,
        ))
    return out
