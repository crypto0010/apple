"""Smoke tests for per_format_latency_bars."""

from pathlib import Path

from apple_bench.figures.per_format_latency_bars import render

SYNTHETIC = [
    {"model": "yolov8_cls_n", "format": "tensorrt_int8", "p50_ms": "1.5"},
    {"model": "yolov8_cls_n", "format": "tensorrt_fp16", "p50_ms": "1.6"},
    {"model": "yolov8_cls_n", "format": "pytorch_fp32", "p50_ms": "10.9"},
    {"model": "paper4_resnet50", "format": "tensorrt_int8", "p50_ms": "2.7"},
    {"model": "paper4_resnet50", "format": "tensorrt_fp16", "p50_ms": "2.6"},
    {"model": "paper4_resnet50", "format": "pytorch_fp32", "p50_ms": "18.6"},
]


def test_render_produces_pdf(tmp_path: Path) -> None:
    out = tmp_path / "bars.pdf"
    render(SYNTHETIC, out)
    assert out.stat().st_size > 5 * 1024
    assert out.with_suffix(".png").stat().st_size > 5 * 1024
