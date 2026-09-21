"""Shared pytest fixtures and configuration for the OSBAD test suite."""
# Use a non-interactive backend so figure-producing functions never try to
# open a GUI window during the test run.
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    """Deterministic NumPy random generator."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def univariate_with_outlier():
    """1D feature array with a single obvious high outlier at index 10."""
    data = np.array(
        [10.0, 11.0, 9.5, 10.5, 10.2, 9.8, 10.1, 9.9, 10.3, 10.0, 100.0])
    return data


@pytest.fixture
def cycle_dataframe():
    """Minimal per-cycle dataframe used by CycleScaling.

    Two cycles, each with a monotonically varying feature so that the
    per-cycle scaling and difference calculations are well defined.
    """
    rows = []
    for cycle in (1, 2):
        for point in range(30):
            rows.append(
                {
                    "cell_index": "cellA",
                    "cycle_index": cycle,
                    "discharge_capacity": 1.0 + point * 0.01 + cycle * 0.1,
                    "voltage": 3.0 + point * 0.02 + cycle * 0.05,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def benchmark_dataframe():
    """Benchmark dataset with per-point rows, cycles and a true label.

    Cycle 2 is flagged as a true outlier cycle (``outlier == 1``).
    """
    rows = []
    for cycle in (1, 2, 3):
        for point in range(5):
            rows.append(
                {
                    "cycle_index": cycle,
                    "cell_index": "cellA",
                    "voltage": 3.0 + point * 0.01,
                    "discharge_capacity": 1.0 + point * 0.01,
                    "outlier": 1 if cycle == 2 else 0,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def features_2d():
    """2D feature matrix with one clear outlier point."""
    features = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [-0.1, 0.05],
            [0.05, -0.1],
            [0.0, 0.1],
            [10.0, 10.0],
        ]
    )
    centroid = np.array([0.0, 0.0])
    return features, centroid
