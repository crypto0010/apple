"""Tests for sustained_run thermal logger."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from apple_bench.bench.thermal import GR3D_RE, TJ_RE, sustained_run

REAL_TEGRASTATS_LINE = (
    "05-15-2026 21:25:52 RAM 3709/7607MB GR3D_FREQ 12%@[306] "
    "cpu@66.656C tj@66.687C VDD_IN 6426mW/6426mW"
)


def test_tj_regex_parses_real_line() -> None:
    m = TJ_RE.search(REAL_TEGRASTATS_LINE)
    assert m is not None
    assert float(m.group(1)) == 66.687


def test_gr3d_regex_parses_real_line() -> None:
    m = GR3D_RE.search(REAL_TEGRASTATS_LINE)
    assert m is not None
    assert int(m.group(1)) == 12
    assert int(m.group(2)) == 306


class _TickRunner:
    name = "tick"

    def __init__(self) -> None:
        self.calls = 0

    def warmup(self, x: np.ndarray, n: int = 10) -> None:
        pass

    def run(self, x: np.ndarray) -> np.ndarray:
        self.calls += 1
        return x


def test_sustained_run_writes_csv_rows(tmp_path: Path, monkeypatch) -> None:
    # Mock tegrastats so the test runs on any host.
    monkeypatch.setattr(
        "apple_bench.bench.thermal._read_tegrastats_oneshot", lambda: (55.0, 30)
    )

    csv_path = tmp_path / "thermal.csv"
    runner = _TickRunner()
    x = np.zeros((1, 1), dtype=np.float32)
    # 6 seconds total, sample every 2 seconds → 3 rows.
    sustained_run(runner, x, duration_minutes=0.1, output_csv=csv_path, sample_period_s=2.0)

    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == "t_seconds,infs_in_window,fps,tj_C,gr3d_freq_pct"
    # At least 2 data rows (3 if the loop catches the 3rd boundary).
    assert len(lines) >= 3
    assert runner.calls > 0
