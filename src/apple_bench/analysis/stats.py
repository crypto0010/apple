"""Bootstrap CIs and paired comparison tests for benchmark results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon


@dataclass
class CI95:
    point: float
    lower: float
    upper: float


def bootstrap_mean_ci(
    samples: np.ndarray, n_boot: int = 5000, seed: int = 1729,
) -> CI95:
    """95 % percentile-bootstrap CI for the mean of ``samples``."""
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples)
    n = len(samples)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = samples[idx].mean()
    return CI95(
        point=float(samples.mean()),
        lower=float(np.percentile(boots, 2.5)),
        upper=float(np.percentile(boots, 97.5)),
    )


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value for a vs b."""
    return float(wilcoxon(a, b).pvalue)
