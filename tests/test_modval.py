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


class TestSummarizeEvalMetrics:
    """Tests for :func:`osbad.modval.summarize_eval_metrics`."""

    def _per_cell_metrics(self):
        """Per-cell metrics for two models over three cells.

        ``iforest`` scores are constant across cells, so its spread is
        exactly zero. ``knn`` varies, so its spread is non-zero. The
        model order is deliberately not alphabetical.
        """
        rows = []
        for cell in ("cellA", "cellB", "cellC"):
            rows.append(
                {
                    "ml_model": "iforest",
                    "cell_index": cell,
                    "accuracy": 0.8,
                    "precision": 0.8,
                    "recall": 0.8,
                    "f1_score": 0.8,
                    "mcc_score": 0.8,
                }
            )
        for cell, score in zip(("cellA", "cellB", "cellC"), (0.2, 0.5, 0.8)):
            rows.append(
                {
                    "ml_model": "knn",
                    "cell_index": cell,
                    "accuracy": score,
                    "precision": score,
                    "recall": score,
                    "f1_score": score,
                    "mcc_score": score,
                }
            )
        return pd.DataFrame(rows)

    def test_one_row_per_model(self):
        df = modval.summarize_eval_metrics(self._per_cell_metrics())
        assert len(df) == 2
        assert set(df["ml_model"]) == {"iforest", "knn"}

    def test_expected_columns(self):
        df = modval.summarize_eval_metrics(self._per_cell_metrics())
        expected = {"ml_model", "n_cells"}
        for metric in (
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "mcc_score",
        ):
            expected.add(f"avg_{metric}")
            expected.add(f"std_{metric}")
        assert set(df.columns) == expected

    def test_n_cells_counts_unique_cells(self):
        df = modval.summarize_eval_metrics(self._per_cell_metrics())
        assert (df["n_cells"] == 3).all()

    def test_mean_values(self):
        df = modval.summarize_eval_metrics(self._per_cell_metrics())
        summary = df.set_index("ml_model")
        assert summary.loc["iforest", "avg_accuracy"] == pytest.approx(0.8)
        assert summary.loc["knn", "avg_accuracy"] == pytest.approx(0.5)

    def test_constant_metrics_have_zero_std(self):
        df = modval.summarize_eval_metrics(self._per_cell_metrics())
        summary = df.set_index("ml_model")
        for metric in (
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "mcc_score",
        ):
            assert summary.loc["iforest", f"std_{metric}"] == pytest.approx(
                0.0)

    def test_sample_std_is_the_default(self):
        """The default ddof=1 must give the sample standard deviation."""
        df = modval.summarize_eval_metrics(self._per_cell_metrics())
        summary = df.set_index("ml_model")
        assert summary.loc["knn", "std_accuracy"] == pytest.approx(0.3)

    def test_ddof_zero_gives_population_std(self):
        df = modval.summarize_eval_metrics(
            self._per_cell_metrics(), ddof=0)
        summary = df.set_index("ml_model")
        assert summary.loc["knn", "std_accuracy"] == pytest.approx(
            np.sqrt(0.06))

    def test_model_order_of_appearance_is_preserved(self):
        """Row order must follow the input, not alphabetical order.

        The comparison plots label their axis with a hard-coded model
        order, so re-sorting the rows here would silently mislabel the
        bars.
        """
        df = modval.summarize_eval_metrics(self._per_cell_metrics())
        assert df["ml_model"].to_list() == ["iforest", "knn"]

    def test_duplicate_cells_are_dropped(self):
        df_metrics = self._per_cell_metrics()
        df_duplicated = pd.concat([df_metrics, df_metrics.head(1)])
        df = modval.summarize_eval_metrics(df_duplicated)
        summary = df.set_index("ml_model")
        assert summary.loc["iforest", "n_cells"] == 3
        assert summary.loc["iforest", "avg_accuracy"] == pytest.approx(0.8)

    def test_duplicates_retained_when_disabled(self):
        df_metrics = self._per_cell_metrics()
        df_duplicated = pd.concat([df_metrics, df_metrics.head(1)])
        df = modval.summarize_eval_metrics(
            df_duplicated, drop_duplicate_cells=False)
        summary = df.set_index("ml_model")
        # The duplicated row is aggregated, but cellA is still one cell.
        assert summary.loc["iforest", "n_cells"] == 3

    def test_single_cell_std_is_nan(self):
        """A single cell has no spread to report with ddof=1."""
        df_metrics = self._per_cell_metrics()
        df_single = df_metrics[df_metrics["cell_index"] == "cellA"]
        df = modval.summarize_eval_metrics(df_single)
        assert df["n_cells"].eq(1).all()
        assert df["std_accuracy"].isna().all()

    def test_missing_column_raises_keyerror(self):
        df_metrics = self._per_cell_metrics().drop(columns=["mcc_score"])
        with pytest.raises(KeyError, match="mcc_score"):
            modval.summarize_eval_metrics(df_metrics)

    def test_input_dataframe_not_modified(self):
        df_metrics = self._per_cell_metrics()
        df_before = df_metrics.copy()
        modval.summarize_eval_metrics(df_metrics)
        pd.testing.assert_frame_equal(df_metrics, df_before)
