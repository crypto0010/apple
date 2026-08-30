"""Tests for the comparative table builder."""

from __future__ import annotations

from pathlib import Path

from apple_bench.analysis.comparative import (
    PAPER_CLAIMS,
    build_comparative_table,
)

PARITY_CSV = """model,params,val_acc_pv,acc_pp2021,ci_pp2021,acc_al9,ci_al9,train_minutes
mobilenet_v3_small,1500000,0.99,0.44,"(0.43, 0.45)",0.61,"(0.60, 0.62)",7.0
paper2_econv_vit,85000000,0.66,0.05,"(0.04, 0.05)",0.05,"(0.04, 0.05)",105.0
paper3_efficientnet_b0_gmp,4300000,1.00,0.39,"(0.38, 0.40)",0.48,"(0.47, 0.49)",9.3
paper4_resnet50,23500000,1.00,0.48,"(0.47, 0.49)",0.71,"(0.70, 0.72)",12.7
"""

EDGE_CSV = """model,format,p50_ms,p95_ms,p99_ms,mean_ms,std_ms,mean_power_mw,energy_mj_per_inf,accuracy_pv,accuracy_delta_pp_vs_fp32,engine_size_kb,nvpmodel
paper3_efficientnet_b0_gmp,tensorrt_fp16,2.7,4.0,5.3,2.8,0.5,12130.0,34.3,1.0,0.0,10274,2
paper3_efficientnet_b0_gmp,tensorrt_int8,2.5,3.5,4.8,2.6,0.5,9045.0,30.0,0.80,20.0,6271,2
paper4_resnet50,tensorrt_int8,2.7,3.5,4.8,2.8,0.4,12100.0,33.0,1.0,0.0,24091,2
paper2_econv_vit,tensorrt_int8,21.6,26.7,28.0,22.7,1.8,16600.0,376.0,0.50,0.25,98711,2
"""


def test_build_table_picks_best_format_respecting_accuracy_gate(tmp_path: Path) -> None:
    parity = tmp_path / "parity.csv"
    parity.write_text(PARITY_CSV)
    edge = tmp_path / "edge.csv"
    edge.write_text(EDGE_CSV)

    rows = build_comparative_table(parity, edge)
    by_model = {r.model: r for r in rows}

    # paper3's INT8 row in this synthetic edge CSV lost 20 pp (way over
    # the 2 pp gate); should fall back to FP16 even though INT8 has a
    # lower p50.
    assert by_model["paper3_efficientnet_b0_gmp"].best_format == "tensorrt_fp16"
    assert by_model["paper3_efficientnet_b0_gmp"].best_p50_ms == 2.7

    # ResNet50's INT8 had zero loss; should pick it.
    assert by_model["paper4_resnet50"].best_format == "tensorrt_int8"

    # All four paper models appear (paper1 has no parity row → ours_acc_pv None).
    assert set(by_model.keys()) == {"paper1_aco_svm", "paper2_econv_vit",
                                     "paper3_efficientnet_b0_gmp", "paper4_resnet50"}
    assert by_model["paper1_aco_svm"].ours_acc_pv is None
    assert by_model["paper1_aco_svm"].best_format is None


def test_paper_claims_keys_match_known_models() -> None:
    """PAPER_CLAIMS keys must match the model registry names used in the CSVs."""
    expected = {"paper1_aco_svm", "paper2_econv_vit",
                "paper3_efficientnet_b0_gmp", "paper4_resnet50"}
    assert set(PAPER_CLAIMS.keys()) == expected


def test_paper_claims_each_entry_has_required_fields() -> None:
    required = {"citation", "claimed_acc_pv", "claimed_deployable"}
    for model, claim in PAPER_CLAIMS.items():
        missing = required - claim.keys()
        assert not missing, f"{model}: missing {missing}"
