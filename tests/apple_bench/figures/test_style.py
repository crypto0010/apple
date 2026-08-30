"""Smoke tests for shared style module."""

import matplotlib as mpl

from apple_bench.figures._style import PALETTE, figsize


def test_palette_has_eight_colorblind_colors() -> None:
    assert len(PALETTE) == 8
    for r, g, b in PALETTE:
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0


def test_rcparams_applied() -> None:
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.right"] is False
    assert mpl.rcParams["savefig.dpi"] == 300


def test_figsize_scales_linearly() -> None:
    w1, h1 = figsize(1.0)
    w2, h2 = figsize(2.0)
    assert w2 == 2 * w1
    assert h2 == 2 * h1
