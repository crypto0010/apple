"""Smoke tests for thermal_sustained figure."""

from pathlib import Path

from apple_bench.figures.thermal_sustained import render


def _synthetic_rows(initial_fps: float, drop_at: int | None) -> list[dict]:
    rows = []
    for i in range(30):
        fps = initial_fps * 0.8 if drop_at is not None and i >= drop_at else initial_fps
        rows.append({
            "t_seconds": "60", "infs_in_window": str(int(fps * 60)),
            "fps": str(round(fps, 2)), "tj_C": "65.0", "gr3d_freq_pct": "30",
        })
    return rows


def test_render_with_two_models(tmp_path: Path) -> None:
    data = {
        "mobilenet_v3_small": _synthetic_rows(800.0, drop_at=None),
        "paper4_resnet50": _synthetic_rows(350.0, drop_at=5),  # throttles at 5 min
    }
    out = tmp_path / "thermal.pdf"
    render(data, out)
    assert out.stat().st_size > 5 * 1024
    assert out.with_suffix(".png").stat().st_size > 5 * 1024


def test_render_with_empty_model_panel(tmp_path: Path) -> None:
    data = {
        "mobilenet_v3_small": _synthetic_rows(800.0, drop_at=None),
        "missing_model": [],
    }
    out = tmp_path / "thermal_partial.pdf"
    render(data, out)
    assert out.stat().st_size > 5 * 1024
