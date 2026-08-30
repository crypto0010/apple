"""Smoke tests for pareto_acc_vs_energy."""

from pathlib import Path

from apple_bench.figures.pareto_acc_vs_energy import render

SYNTHETIC = [
    {"model": "mobilenet_v3_small", "format": "tensorrt_fp16",
     "energy_mj_per_inf": "16.8", "accuracy_pv": "1.0"},
    {"model": "mobilenet_v3_small", "format": "tensorrt_int8",
     "energy_mj_per_inf": "14.2", "accuracy_pv": "0.84"},
    {"model": "paper4_resnet50", "format": "tensorrt_int8",
     "energy_mj_per_inf": "33.4", "accuracy_pv": "1.0"},
    {"model": "paper2_econv_vit", "format": "tensorrt_int8",
     "energy_mj_per_inf": "376.0", "accuracy_pv": "0.50"},
]


def test_render_produces_pdf_and_png(tmp_path: Path) -> None:
    pdf = tmp_path / "pareto_energy.pdf"
    render(SYNTHETIC, pdf)
    assert pdf.stat().st_size > 5 * 1024
    assert pdf.with_suffix(".png").stat().st_size > 5 * 1024
