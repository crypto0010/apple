"""Latency benchmark primitive: warmup, then N timed runs, return percentiles."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from apple_bench.runners.base import Runner


@dataclass
class LatencyResult:
    runner_name: str
    n_runs: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float

    def as_dict(self) -> dict:
        return {
            "runner": self.runner_name,
            "n_runs": self.n_runs,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
        }


def measure_latency(
    runner: Runner,
    sample_input: np.ndarray,
    warmup: int = 20,
    n_runs: int = 200,
) -> LatencyResult:
    runner.warmup(sample_input, n=warmup)
    times = np.empty(n_runs)
    for i in range(n_runs):
        t0 = time.perf_counter()
        runner.run(sample_input)
        times[i] = (time.perf_counter() - t0) * 1000
    return LatencyResult(
        runner_name=runner.name,
        n_runs=n_runs,
        p50_ms=float(np.percentile(times, 50)),
        p95_ms=float(np.percentile(times, 95)),
        p99_ms=float(np.percentile(times, 99)),
        mean_ms=float(times.mean()),
        std_ms=float(times.std()),
    )
