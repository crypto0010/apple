"""Tests for the multi-seed aggregation helpers (Paper 2 Task 10)."""

from __future__ import annotations

from apple_bench.analysis.multiseed import (
    SeedStats,
    paired_wilcoxon_seeds,
    seed_stats,
)


def test_seed_stats_returns_mean_sd_n():
    s = seed_stats([0.95, 0.96, 0.94, 0.97, 0.95])
    assert isinstance(s, SeedStats)
    assert abs(s.mean - 0.954) < 1e-3
    # ddof=1 sample SD: sqrt(sum((x-mean)^2) / (n-1))
    # values = [.95,.96,.94,.97,.95]; squared deviations
    # from .954 = .000016, .000036, .000196, .000256, .000016 -> sum .00052
    # var = .00052 / 4 = .00013 ; sd = sqrt(.00013) = 0.01140175
    assert abs(s.sd - 0.0114) < 1e-3
    assert s.n == 5


def test_seed_stats_handles_single_value():
    s = seed_stats([0.5])
    assert s.mean == 0.5
    assert s.sd == 0.0
    assert s.n == 1


def test_paired_wilcoxon_detects_consistent_shift():
    # QAT consistently higher than PTQ across 5 seeds.
    ptq = [0.80, 0.81, 0.79, 0.82, 0.80]
    qat = [0.98, 0.99, 0.98, 0.99, 0.98]
    p = paired_wilcoxon_seeds(ptq, qat)
    assert 0.0 <= p <= 0.10  # consistent ranks -> small p-value


def test_paired_wilcoxon_rejects_no_difference():
    # No shift between paired observations -> p should NOT be tiny.
    a = [0.80, 0.81, 0.79, 0.82, 0.80]
    b = [0.80, 0.82, 0.78, 0.81, 0.81]  # roughly the same
    p = paired_wilcoxon_seeds(a, b)
    assert p > 0.1
