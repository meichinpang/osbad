"""Tests for :mod:`osbad.model`."""
import numpy as np
import pandas as pd
import pytest

import osbad.config as bconf
from osbad.model import ModelRunner


@pytest.fixture
def input_features():
    """Feature dataframe with the columns expected by ModelRunner."""
    n = 20
    cycle = np.arange(n)
    return pd.DataFrame(
        {
            "cell_index": ["cellA"] * n,
            "cycle_index": cycle,
            "log_max_diff_dQ": np.linspace(-3.0, 0.0, n),
            "log_max_diff_dV": np.linspace(-2.0, 1.0, n),
        }
    )


@pytest.fixture
def runner(input_features, tmp_path, monkeypatch):
    """A ModelRunner whose artifacts are written to a temp directory."""
    monkeypatch.setattr(bconf, "PIPELINE_OUTPUT_DIR", tmp_path)
    return ModelRunner(
        cell_label="cellA",
        df_input_features=input_features,
        selected_feature_cols=("log_max_diff_dQ", "log_max_diff_dV"),
    )


class TestCreateModelXInput:
    """Tests for :meth:`ModelRunner.create_model_x_input`."""

    def test_shape(self, runner, input_features):
        xdata = runner.create_model_x_input()
        assert xdata.shape == (len(input_features), 2)

    def test_values_match_columns(self, runner, input_features):
        xdata = runner.create_model_x_input()
        assert np.allclose(xdata[:, 0], input_features["log_max_diff_dQ"])
        assert np.allclose(xdata[:, 1], input_features["log_max_diff_dV"])


class TestPredOutlierIndicesFromProba:
    """Tests for :meth:`ModelRunner.pred_outlier_indices_from_proba`."""

    def test_indices_and_scores(self, runner):
        proba = np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.4, 0.6],
                [0.95, 0.05],
            ]
        )
        indices, scores = runner.pred_outlier_indices_from_proba(
            proba, threshold=0.6)
        assert list(indices) == [1, 2]
        assert np.allclose(scores, [0.8, 0.6])

    def test_no_outliers_above_threshold(self, runner):
        proba = np.array([[0.9, 0.1], [0.8, 0.2]])
        indices, scores = runner.pred_outlier_indices_from_proba(
            proba, threshold=0.5)
        assert len(indices) == 0
        assert len(scores) == 0


class TestCreate2dMeshGrid:
    """Tests for :meth:`ModelRunner.create_2d_mesh_grid`."""

    def test_shapes(self, runner):
        runner.create_model_x_input()
        xx, yy, meshgrid = runner.create_2d_mesh_grid()
        assert xx.shape == (100, 100)
        assert yy.shape == (100, 100)
        assert meshgrid.shape == (100 * 100, 2)

    def test_square_grid_shares_bounds(self, runner):
        runner.create_model_x_input()
        xx, yy, _ = runner.create_2d_mesh_grid(square_grid=True)
        assert np.isclose(xx.min(), yy.min())
        assert np.isclose(xx.max(), yy.max())


class TestProxyEvaluateIndices:
    """Tests for :meth:`ModelRunner.proxy_evaluate_indices`."""

    def test_returns_two_scores(self, runner):
        cycle_idx = np.arange(20).reshape(-1, 1)
        features = (2.0 * cycle_idx.ravel()).astype(float)
        features[5] = 500.0  # inject an outlier
        loss_score, inlier_score = runner.proxy_evaluate_indices(
            pred_indices=np.array([5]),
            cycle_idx=cycle_idx,
            features=features,
        )
        assert isinstance(loss_score, float)
        assert isinstance(inlier_score, float)

    def test_inlier_score_reflects_removed_points(self, runner):
        cycle_idx = np.arange(20).reshape(-1, 1)
        features = (2.0 * cycle_idx.ravel()).astype(float)
        features[5] = 500.0
        _, inlier_score = runner.proxy_evaluate_indices(
            pred_indices=np.array([5]),
            cycle_idx=cycle_idx,
            features=features,
        )
        # One of 20 points removed leaves 19/20 as inliers.
        assert np.isclose(inlier_score, 19 / 20)


class TestEvaluateIndices:
    """Tests for :meth:`ModelRunner.evaluate_indices`."""

    def test_recall_precision_range(self, runner, benchmark_dataframe):
        recall, precision = runner.evaluate_indices(
            df_benchmark_dataset=benchmark_dataframe,
            pred_indices=np.array([2]),
        )
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= precision <= 1.0

    def test_perfect_prediction(self, runner, benchmark_dataframe):
        recall, precision = runner.evaluate_indices(
            df_benchmark_dataset=benchmark_dataframe,
            pred_indices=np.array([2]),
        )
        assert recall == 1.0
        assert precision == 1.0


class TestSplitList:
    """Tests for the private :meth:`ModelRunner._split_list` helper."""

    def test_short_list_returns_str(self, runner):
        result = runner._split_list([1, 2, 3], chunk_size=5)
        assert isinstance(result, str)
        assert result == "[1, 2, 3]"

    def test_long_list_is_chunked(self, runner):
        result = runner._split_list(list(range(12)), chunk_size=5)
        assert isinstance(result, str)
        assert result.count("\n") == 2
