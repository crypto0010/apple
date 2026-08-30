"""Aggregation helpers for multi-seed experiments (Paper 2).

Paper 1's L1 threat to validity flagged the single-training-seed
methodology as a source of measurement uncertainty.  Paper 2 reports
every QAT result as a per-seed distribution; these helpers compute
mean / standard deviation / paired-test p-values from those
distributions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class SeedStats:
    """Distributional summary of a per-seed measurement series."""

    mean: float
    sd: float  # sample SD with ddof=1; 0.0 when n==1
    n: int


def seed_stats(values: Sequence[float]) -> SeedStats:
    """Mean + sample SD (ddof=1) + count from a per-seed series."""
    arr = np.asarray(values, dtype=float)
    return SeedStats(
        mean=float(arr.mean()),
        sd=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        n=int(arr.size),
    )


def paired_wilcoxon_seeds(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value for matched seeds.

    The intended usage is to test whether the two measurement series
    (e.g. PTQ-INT8 accuracy vs QAT-INT8 accuracy) differ systematically
    across seeds while paired by training seed.  The null hypothesis is
    that the per-seed differences are symmetric about zero.
    """
    res = wilcoxon(np.asarray(a), np.asarray(b), zero_method="wilcox")
    return float(res.pvalue)
