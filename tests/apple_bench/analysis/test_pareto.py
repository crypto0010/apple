"""Tests for Pareto frontier computation."""

import numpy as np

from apple_bench.analysis.pareto import pareto_frontier_indices


def test_simple_pareto_all_kept() -> None:
    # No point dominates another → all five Pareto-optimal.
    cost = np.array([1.0, 2.0, 3.0, 1.5, 2.5])
    value = np.array([0.5, 0.7, 0.9, 0.6, 0.8])
    idx = sorted(pareto_frontier_indices(cost, value))
    assert idx == [0, 1, 2, 3, 4]


def test_dominated_point_excluded() -> None:
    cost = np.array([1.0, 2.0, 1.5])
    value = np.array([0.5, 0.9, 0.4])  # idx 2 dominated by idx 0
    idx = sorted(pareto_frontier_indices(cost, value))
    assert idx == [0, 1]


def test_maximize_cost_flag() -> None:
    # Higher cost is better here.
    cost = np.array([1.0, 2.0, 3.0])
    value = np.array([0.5, 0.7, 0.9])
    idx = sorted(pareto_frontier_indices(cost, value, minimize_cost=False))
    # Lowest-cost point is dominated by all others when higher cost is better.
    assert 0 not in idx


def test_duplicate_points_kept() -> None:
    cost = np.array([1.0, 1.0, 2.0])
    value = np.array([0.5, 0.5, 0.9])
    # Duplicates: neither strictly dominates, so both are kept.
    idx = sorted(pareto_frontier_indices(cost, value))
    assert idx == [0, 1, 2]
