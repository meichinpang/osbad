"""Tests for :mod:`osbad.viz`.

The visualization module is mostly figure-producing code. Here we cover the
pure ``calculate_bubble_size_ratio`` transform and a smoke test that
``hist_boxplot`` returns a Matplotlib Axes using the Agg backend.
"""
import matplotlib
import numpy as np
import pandas as pd
import pytest

import osbad.viz as bviz


class TestCalculateBubbleSizeRatio:
    """Tests for :func:`osbad.viz.calculate_bubble_size_ratio`."""

    def test_returns_standardized_series(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bviz.calculate_bubble_size_ratio(series)
        assert len(result) == len(series)
        assert np.isclose(np.mean(result), 0.0, atol=1e-9)

    def test_zero_at_mean(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bviz.calculate_bubble_size_ratio(series)
        # Value equal to the mean (3.0) maps to a ratio of 0.
        assert np.isclose(result.iloc[2], 0.0)

    def test_accepts_ndarray(self):
        result = bviz.calculate_bubble_size_ratio(
            np.array([10.0, 20.0, 30.0]))
        assert len(result) == 3


class TestHistBoxplot:
    """Smoke test for :func:`osbad.viz.hist_boxplot`."""

    def test_returns_axes(self):
        data = pd.Series(np.linspace(0, 1, 50))
        ax = bviz.hist_boxplot(data)
        assert isinstance(ax, matplotlib.axes.Axes)
        matplotlib.pyplot.close("all")
