"""Tests for measure_latency — uses a sleep-based stub runner."""

import time

import numpy as np

from apple_bench.bench.latency import LatencyResult, measure_latency


class _SleepRunner:
    name = "sleep"

    def __init__(self, sleep_ms: float) -> None:
        self.sleep_ms = sleep_ms

    def warmup(self, x: np.ndarray, n: int = 10) -> None:
        for _ in range(n):
            time.sleep(self.sleep_ms / 1000)

    def run(self, x: np.ndarray) -> np.ndarray:
        time.sleep(self.sleep_ms / 1000)
        return x


def test_measure_latency_percentiles_match_sleep_target() -> None:
    runner = _SleepRunner(sleep_ms=5.0)
    x = np.zeros(1, dtype=np.float32)
    result = measure_latency(runner, x, warmup=2, n_runs=30)

    assert isinstance(result, LatencyResult)
    assert result.runner_name == "sleep"
    assert result.n_runs == 30
    # Sleeps tend to overshoot slightly on Linux; allow a generous lower
    # bound and a 2x upper bound to keep the test robust on a busy host.
    assert 4.5 <= result.p50_ms <= 12.0, f"p50={result.p50_ms}"
    assert result.p95_ms >= result.p50_ms
    assert result.p99_ms >= result.p95_ms
