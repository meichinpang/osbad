"""Tests for :mod:`osbad.database`.

Only the label-handling helpers of :class:`BenchDB` are exercised here since
they do not require a live DuckDB database. Dataset loading and plotting are
integration concerns tied to on-disk databases and are out of scope.
"""
import numpy as np
import pytest

import osbad.config as bconf
from osbad.database import BenchDB


@pytest.fixture
def benchdb(tmp_path, monkeypatch):
    """A BenchDB instance whose artifacts land in a temp directory."""
    monkeypatch.setattr(bconf, "PIPELINE_OUTPUT_DIR", tmp_path)
    return BenchDB(
        input_db_filepath=str(tmp_path / "missing.db"),
        cell_label="cellA",
    )


class TestDropLabels:
    """Tests for :meth:`BenchDB.drop_labels`."""

    def test_removes_outlier_column(self, benchdb, benchmark_dataframe):
        result = benchdb.drop_labels(benchmark_dataframe)
        assert "outlier" not in result.columns

    def test_filter_col_keeps_only_selected(self, benchdb, benchmark_dataframe):
        filter_col = ["cell_index", "cycle_index", "voltage"]
        result = benchdb.drop_labels(benchmark_dataframe, filter_col=filter_col)
        assert list(result.columns) == filter_col

    def test_row_count_preserved(self, benchdb, benchmark_dataframe):
        result = benchdb.drop_labels(benchmark_dataframe)
        assert len(result) == len(benchmark_dataframe)


class TestGetTrueOutlierCycleIndex:
    """Tests for :meth:`BenchDB.get_true_outlier_cycle_index`."""

    def test_returns_outlier_cycles(self, benchdb, benchmark_dataframe):
        result = benchdb.get_true_outlier_cycle_index(benchmark_dataframe)
        assert isinstance(result, np.ndarray)
        assert set(result) == {2}

    def test_no_outliers_returns_empty(self, benchdb, benchmark_dataframe):
        df = benchmark_dataframe.copy()
        df["outlier"] = 0
        result = benchdb.get_true_outlier_cycle_index(df)
        assert len(result) == 0
