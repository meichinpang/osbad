"""Tests for :mod:`osbad.stats`."""
import numpy as np
import pandas as pd
import pytest

import osbad.stats as bstats


class TestOutlierDetectors:
    """Grouped tests for the threshold-based outlier detectors.

    Every detector shares the same contract: it returns a tuple of
    ``(outlier_indices, lower_limit, upper_limit)`` where the indices are a
    NumPy array flagging the single obvious high outlier in the fixture.
    """

    @pytest.mark.parametrize(
        "detector",
        [
            bstats._compute_sd_outliers,
            bstats._compute_mad_outliers,
            bstats._compute_iqr_outliers,
            bstats._compute_zscore_outliers,
            bstats._compute_modified_z_outliers,
        ],
    )
    def test_returns_three_tuple(self, detector, univariate_with_outlier):
        result = detector(univariate_with_outlier)
        assert isinstance(result, tuple)
        assert len(result) == 3

    @pytest.mark.parametrize(
        "detector",
        [
            bstats._compute_sd_outliers,
            bstats._compute_mad_outliers,
            bstats._compute_iqr_outliers,
            bstats._compute_zscore_outliers,
            bstats._compute_modified_z_outliers,
        ],
    )
    def test_indices_is_ndarray(self, detector, univariate_with_outlier):
        indices, _, _ = detector(univariate_with_outlier)
        assert isinstance(indices, np.ndarray)

    @pytest.mark.parametrize(
        "detector",
        [
            bstats._compute_sd_outliers,
            bstats._compute_mad_outliers,
            bstats._compute_iqr_outliers,
            bstats._compute_zscore_outliers,
            bstats._compute_modified_z_outliers,
        ],
    )
    def test_flags_known_outlier(self, detector, univariate_with_outlier):
        indices, _, _ = detector(univariate_with_outlier)
        # The final element (value 100) is the injected outlier.
        assert 10 in indices

    @pytest.mark.parametrize(
        "detector",
        [
            bstats._compute_sd_outliers,
            bstats._compute_mad_outliers,
            bstats._compute_iqr_outliers,
            bstats._compute_zscore_outliers,
            bstats._compute_modified_z_outliers,
        ],
    )
    def test_limits_are_ordered(self, detector, univariate_with_outlier):
        _, lower, upper = detector(univariate_with_outlier)
        assert lower < upper

    def test_sd_limits_symmetric_about_mean(self, univariate_with_outlier):
        _, lower, upper = bstats._compute_sd_outliers(univariate_with_outlier)
        mean = np.mean(univariate_with_outlier)
        assert np.isclose((lower + upper) / 2, mean)

    def test_zscore_limits_symmetric(self, univariate_with_outlier):
        _, lower, upper = bstats._compute_zscore_outliers(
            univariate_with_outlier, zscore_threshold=3)
        assert lower == -3
        assert upper == 3


class TestMadFactor:
    """Tests for :func:`osbad.stats._calculate_mad_factor`."""

    def test_returns_positive_float(self, univariate_with_outlier):
        factor = bstats._calculate_mad_factor(univariate_with_outlier)
        assert isinstance(factor, float)
        assert factor > 0

    def test_explicit_mad_factor_is_used(self, univariate_with_outlier):
        # Providing a mad_factor should bypass the auto-calculation and still
        # return a valid three-tuple.
        indices, lower, upper = bstats._compute_mad_outliers(
            univariate_with_outlier, mad_factor=1.4826)
        assert isinstance(indices, np.ndarray)
        assert lower < upper


class TestFeatureStatistics:
    """Tests for the descriptive-statistics helpers."""

    def test_calculate_zscore_mean_near_zero(self, univariate_with_outlier):
        zscore = bstats.calculate_zscore(univariate_with_outlier)
        assert np.isclose(np.mean(zscore), 0.0, atol=1e-9)

    def test_calculate_zscore_preserves_length(self, univariate_with_outlier):
        zscore = bstats.calculate_zscore(univariate_with_outlier)
        assert len(zscore) == len(univariate_with_outlier)

    def test_calculate_feature_stats_rows(self, univariate_with_outlier):
        df = bstats.calculate_feature_stats(univariate_with_outlier)
        assert isinstance(df, pd.DataFrame)
        assert list(df.index) == ["max", "min", "mean", "std"]

    def test_calculate_feature_stats_named_column(self, univariate_with_outlier):
        df = bstats.calculate_feature_stats(
            univariate_with_outlier, new_col_name="feat")
        assert list(df.columns) == ["feat"]

    def test_calculate_feature_stats_values(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        df = bstats.calculate_feature_stats(data, new_col_name="feat")
        assert df.loc["max", "feat"] == 5.0
        assert df.loc["min", "feat"] == 1.0
        assert df.loc["mean", "feat"] == 3.0


class TestOutlierRegistry:
    """Tests for the :data:`osbad.stats.outlier_method` registry."""

    def test_registry_keys(self):
        assert set(bstats.outlier_method) == {
            "sd", "mad", "iqr", "zscore", "mod_zscore"}

    @pytest.mark.parametrize(
        "key", ["sd", "mad", "iqr", "zscore", "mod_zscore"])
    def test_config_has_callable_and_params(self, key):
        config = bstats.outlier_method[key]
        assert callable(config.compute)
        assert isinstance(config.params, dict)

    def test_compute_via_registry(self, univariate_with_outlier):
        config = bstats.outlier_method["sd"]
        indices, lower, upper = config.compute(
            univariate_with_outlier, **config.params)
        assert 10 in indices
        assert lower < upper
