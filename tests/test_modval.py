"""Tests for :mod:`osbad.modval`."""
import matplotlib
import numpy as np
import pandas as pd
import pytest

import osbad.modval as modval


class TestEvaluatePredOutliers:
    """Tests for :func:`osbad.modval.evaluate_pred_outliers`."""

    def test_columns_and_one_row_per_cycle(self, benchmark_dataframe):
        df = modval.evaluate_pred_outliers(
            df_benchmark=benchmark_dataframe,
            outlier_cycle_index=np.array([2]),
        )
        assert set(df.columns) == {
            "cycle_index", "true_outlier", "pred_outlier"}
        assert len(df) == benchmark_dataframe["cycle_index"].nunique()

    def test_true_labels_match_benchmark(self, benchmark_dataframe):
        df = modval.evaluate_pred_outliers(
            df_benchmark=benchmark_dataframe,
            outlier_cycle_index=np.array([2]),
        )
        true_map = df.set_index("cycle_index")["true_outlier"].to_dict()
        assert true_map == {1: 0, 2: 1, 3: 0}

    def test_predicted_labels_reflect_input(self, benchmark_dataframe):
        df = modval.evaluate_pred_outliers(
            df_benchmark=benchmark_dataframe,
            outlier_cycle_index=np.array([2]),
        )
        pred_map = df.set_index("cycle_index")["pred_outlier"].to_dict()
        assert pred_map == {1: 0, 2: 1, 3: 0}

    def test_no_predicted_outliers(self, benchmark_dataframe):
        df = modval.evaluate_pred_outliers(
            df_benchmark=benchmark_dataframe,
            outlier_cycle_index=np.array([]),
        )
        assert (df["pred_outlier"] == 0).all()


class TestEvalModelPerformance:
    """Tests for :func:`osbad.modval.eval_model_performance`."""

    def _perfect_eval(self):
        return pd.DataFrame(
            {
                "true_outlier": [0, 1, 0, 1],
                "pred_outlier": [0, 1, 0, 1],
            }
        )

    def test_returns_single_row(self):
        df = modval.eval_model_performance(
            model_name="iforest",
            selected_cell_label="cellA",
            df_eval_outliers=self._perfect_eval(),
        )
        assert len(df) == 1

    def test_metric_columns_present(self):
        df = modval.eval_model_performance(
            model_name="iforest",
            selected_cell_label="cellA",
            df_eval_outliers=self._perfect_eval(),
        )
        for col in (
            "ml_model",
            "cell_index",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "mcc_score",
        ):
            assert col in df.columns

    def test_perfect_prediction_scores(self):
        df = modval.eval_model_performance(
            model_name="iforest",
            selected_cell_label="cellA",
            df_eval_outliers=self._perfect_eval(),
        )
        assert df.loc[0, "accuracy"] == 1.0
        assert df.loc[0, "precision"] == 1.0
        assert df.loc[0, "recall"] == 1.0
        assert df.loc[0, "f1_score"] == 1.0

    def test_metadata_preserved(self):
        df = modval.eval_model_performance(
            model_name="iforest",
            selected_cell_label="cellA",
            df_eval_outliers=self._perfect_eval(),
        )
        assert df.loc[0, "ml_model"] == "iforest"
        assert df.loc[0, "cell_index"] == "cellA"


class TestGenerateConfusionMatrix:
    """Tests for :func:`osbad.modval.generate_confusion_matrix`."""

    def test_returns_axes(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        ax = modval.generate_confusion_matrix(y_true, y_pred)
        assert isinstance(ax, matplotlib.axes.Axes)
        matplotlib.pyplot.close("all")
