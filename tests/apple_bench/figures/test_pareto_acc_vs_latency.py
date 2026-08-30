"""Smoke tests for pareto_acc_vs_latency."""

from pathlib import Path

from apple_bench.figures.pareto_acc_vs_latency import from_csv, render

SYNTHETIC = [
    {"model": "mobilenet_v3_small", "format": "tensorrt_fp16",
     "p50_ms": "2.0", "accuracy_pv": "1.0"},
    {"model": "mobilenet_v3_small", "format": "tensorrt_int8",
     "p50_ms": "1.8", "accuracy_pv": "0.84"},
    {"model": "paper4_resnet50", "format": "tensorrt_int8",
     "p50_ms": "2.7", "accuracy_pv": "1.0"},
    {"model": "paper2_econv_vit", "format": "tensorrt_int8",
     "p50_ms": "21.6", "accuracy_pv": "0.50"},
]


def test_render_produces_pdf_and_png(tmp_path: Path) -> None:
    pdf = tmp_path / "pareto.pdf"
    render(SYNTHETIC, pdf)
    assert pdf.is_file()
    assert pdf.stat().st_size > 5 * 1024
    png = pdf.with_suffix(".png")
    assert png.is_file()
    assert png.stat().st_size > 5 * 1024


def test_from_csv_path_works(tmp_path: Path) -> None:
    import csv
    edge = tmp_path / "edge.csv"
    with edge.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "format", "p50_ms", "accuracy_pv"])
        w.writeheader()
        w.writerows(SYNTHETIC)
    out = tmp_path / "from_csv.pdf"
    from_csv(edge, out)
    assert out.stat().st_size > 5 * 1024
