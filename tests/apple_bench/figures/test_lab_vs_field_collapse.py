"""Smoke tests for lab_vs_field_collapse."""

from pathlib import Path

from apple_bench.figures.lab_vs_field_collapse import render

SYNTHETIC = [
    {"model": "mobilenet_v3_small", "val_acc_pv": "1.0",  "acc_pp2021": "0.44"},
    {"model": "paper4_resnet50",    "val_acc_pv": "1.0",  "acc_pp2021": "0.48"},
    {"model": "paper2_econv_vit",   "val_acc_pv": "0.66", "acc_pp2021": "0.05"},
    {"model": "yolov8_cls_n",       "val_acc_pv": "0.99", "acc_pp2021": "0.45"},
    # Row with missing field acc should be dropped.
    {"model": "untrained",          "val_acc_pv": "0.50", "acc_pp2021": ""},
]


def test_render_produces_pdf(tmp_path: Path) -> None:
    out = tmp_path / "gap.pdf"
    render(SYNTHETIC, out)
    assert out.stat().st_size > 5 * 1024
    assert out.with_suffix(".png").stat().st_size > 5 * 1024
