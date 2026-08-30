"""Pareto frontier computation."""

from __future__ import annotations

import numpy as np


def pareto_frontier_indices(
    cost: np.ndarray,
    value: np.ndarray,
    *,
    minimize_cost: bool = True,
    maximize_value: bool = True,
) -> list[int]:
    """Indices of Pareto-optimal points.

    By default lower cost + higher value is better. Flip ``minimize_cost`` /
    ``maximize_value`` to invert either axis. A point is Pareto-optimal iff
    no other point weakly beats it on both axes and strictly beats on at
    least one.
    """
    n = len(cost)
    c = -np.asarray(cost) if not minimize_cost else np.asarray(cost)
    v = -np.asarray(value) if not maximize_value else np.asarray(value)
    keep: list[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if (
                c[j] <= c[i]
                and v[j] >= v[i]
                and (c[j] < c[i] or v[j] > v[i])
            ):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep
