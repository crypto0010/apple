"""Tests for bootstrap CI + paired Wilcoxon."""

import numpy as np

from apple_bench.analysis.stats import CI95, bootstrap_mean_ci, paired_wilcoxon


def test_bootstrap_ci_contains_population_mean() -> None:
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=5.0, scale=1.0, size=500)
    ci = bootstrap_mean_ci(samples, n_boot=2000)
    assert isinstance(ci, CI95)
    assert ci.lower < 5.0 < ci.upper
    # Sample mean should be inside its own CI.
    assert ci.lower <= ci.point <= ci.upper
    # CI should be tight on 500 samples (within ~0.2 of true mean).
    assert ci.upper - ci.lower < 0.3


def test_bootstrap_ci_is_reproducible_with_seed() -> None:
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    a = bootstrap_mean_ci(samples, n_boot=500, seed=7)
    b = bootstrap_mean_ci(samples, n_boot=500, seed=7)
    assert (a.point, a.lower, a.upper) == (b.point, b.lower, b.upper)


def test_paired_wilcoxon_detects_difference() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, size=200)
    b = a + 0.5  # paired, shifted positive
    pval = paired_wilcoxon(a, b)
    assert pval < 0.001


def test_paired_wilcoxon_does_not_flag_when_no_shift() -> None:
    """Wilcoxon ranks the sign of differences, not their magnitude.

    A truly-random pair with no consistent shift should yield a high p-value;
    a consistent shift of *any* size (even 1e-9) rejects. This test uses
    independent draws — there is no shift to detect.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, size=200)
    b = rng.normal(0.0, 1.0, size=200)
    pval = paired_wilcoxon(a, b)
    assert pval > 0.05
