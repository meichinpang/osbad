"""Tests for :mod:`osbad.scaler`."""
import numpy as np
import pandas as pd
import pytest

from osbad.scaler import CycleScaling


class TestMedianIQRScaling:
    """Tests for :meth:`CycleScaling.median_IQR_scaling`."""

    def test_default_columns(self, cycle_dataframe):
        scaler = CycleScaling(cycle_dataframe)
        df = scaler.median_IQR_scaling(variable="discharge_capacity")
        assert set(df.columns) == {"scaled_discharge_capacity", "cycle_index"}

    def test_validate_adds_intermediate_columns(self, cycle_dataframe):
        scaler = CycleScaling(cycle_dataframe)
        df = scaler.median_IQR_scaling(
            variable="discharge_capacity", validate=True)
        for col in (
            "discharge_capacity",
            "cycle_median",
            "median_square",
            "IQR",
            "median_square_IQR_ratio",
            "scaled_discharge_capacity",
            "cycle_index",
        ):
            assert col in df.columns

    def test_row_count_preserved(self, cycle_dataframe):
        scaler = CycleScaling(cycle_dataframe)
        df = scaler.median_IQR_scaling(variable="voltage")
        assert len(df) == len(cycle_dataframe)

    def test_all_cycles_present(self, cycle_dataframe):
        scaler = CycleScaling(cycle_dataframe)
        df = scaler.median_IQR_scaling(variable="voltage")
        assert set(df["cycle_index"].unique()) == {1, 2}


class TestMaxDiffPerCycle:
    """Tests for :meth:`CycleScaling.calculate_max_diff_per_cycle`."""

    def test_columns_and_row_per_cycle(self, cycle_dataframe):
        scaler = CycleScaling(cycle_dataframe)
        scaled = scaler.median_IQR_scaling(variable="discharge_capacity")
        df = scaler.calculate_max_diff_per_cycle(
            scaled, variable_name="scaled_discharge_capacity")
        assert set(df.columns) == {"max_diff", "log_max_diff", "cycle_index"}
        assert len(df) == 2

    def test_max_diff_is_non_negative(self, cycle_dataframe):
        scaler = CycleScaling(cycle_dataframe)
        scaled = scaler.median_IQR_scaling(variable="discharge_capacity")
        df = scaler.calculate_max_diff_per_cycle(
            scaled, variable_name="scaled_discharge_capacity")
        assert (df["max_diff"] >= 0).all()

    def test_log_matches_max_diff(self, cycle_dataframe):
        scaler = CycleScaling(cycle_dataframe)
        scaled = scaler.median_IQR_scaling(variable="discharge_capacity")
        df = scaler.calculate_max_diff_per_cycle(
            scaled, variable_name="scaled_discharge_capacity")
        assert np.allclose(df["log_max_diff"], np.log(df["max_diff"]))


class TestMaxFeatureDerivativePerCycle:
    """Tests for :meth:`CycleScaling.calculate_max_feature_derivative_per_cycle`."""

    def test_columns_and_row_per_cycle(self, cycle_dataframe):
        scaler = CycleScaling(cycle_dataframe)
        df = scaler.calculate_max_feature_derivative_per_cycle(
            Xfeature=cycle_dataframe["discharge_capacity"],
            Yfeature=cycle_dataframe["voltage"],
            cycle_index=cycle_dataframe["cycle_index"],
        )
        assert set(df.columns) == {"max_diff", "log_max_diff", "cycle_index"}
        assert len(df) == 2

    def test_derivative_matches_linear_slope(self, cycle_dataframe):
        # dV = 0.02 per point and dQ = 0.01 per point, so dV/dQ == 2.0.
        scaler = CycleScaling(cycle_dataframe)
        df = scaler.calculate_max_feature_derivative_per_cycle(
            Xfeature=cycle_dataframe["discharge_capacity"],
            Yfeature=cycle_dataframe["voltage"],
            cycle_index=cycle_dataframe["cycle_index"],
        )
        assert np.allclose(df["max_diff"], 2.0)
