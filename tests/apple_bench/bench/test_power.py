"""Tests for tegrastats VDD_IN parser + TegraPowerLogger."""

from __future__ import annotations

from apple_bench.bench.power import VDD_IN_RE, PowerSample, TegraPowerLogger

REAL_TEGRASTATS_LINE = (
    "05-15-2026 21:25:52 RAM 3709/7607MB (lfb 21x4MB) SWAP 138/3804MB "
    "(cached 0MB) CPU [34%@729,15%@729,36%@729,36%@729,17%@729,15%@729] "
    "EMC_FREQ 5%@2133 GR3D_FREQ 12%@[306] NVDEC off NVJPG off NVJPG1 off "
    "VIC off OFA off APE 200 cpu@66.656C soc2@65.406C soc0@66.437C "
    "gpu@65.875C tj@66.687C soc1@66.687C VDD_IN 6426mW/6426mW "
    "VDD_CPU_GPU_CV 1220mW/1220mW VDD_SOC 1695mW/1695mW"
)


def test_vdd_in_regex_parses_real_line() -> None:
    m = VDD_IN_RE.search(REAL_TEGRASTATS_LINE)
    assert m is not None
    assert int(m.group(1)) == 6426


def test_mean_power_mw_averages_samples() -> None:
    logger = TegraPowerLogger()
    logger.samples = [
        PowerSample(timestamp=0.0, vdd_in_mw=6000),
        PowerSample(timestamp=0.1, vdd_in_mw=7000),
        PowerSample(timestamp=0.2, vdd_in_mw=8000),
    ]
    assert logger.mean_power_mw() == 7000.0


def test_mean_power_mw_empty_returns_zero() -> None:
    logger = TegraPowerLogger()
    assert logger.mean_power_mw() == 0.0


def test_logger_collects_real_samples_when_tegrastats_available() -> None:
    """Integration test — captures at least one VDD_IN sample on a Jetson."""
    import shutil
    import time

    import pytest

    if shutil.which("tegrastats") is None:
        pytest.skip("tegrastats not on PATH — non-Jetson host")

    logger = TegraPowerLogger(interval_ms=200)
    logger.start()
    time.sleep(1.5)
    logger.stop()
    assert len(logger.samples) >= 2
    # Idle Orin draws ~5-9 W on VDD_IN.
    mean_w = logger.mean_power_mw() / 1000
    assert 3.0 <= mean_w <= 30.0, f"implausible mean power: {mean_w} W"
