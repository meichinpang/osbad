"""Tests for :mod:`osbad.dbad`."""
import matplotlib
import numpy as np
import pytest

import osbad.dbad as dbad


class TestCalculateDistance:
    """Tests for :func:`osbad.dbad.calculate_distance`."""

    def test_norm_without_max_returns_tuple(self, features_2d):
        features, centroid = features_2d
        result = dbad.calculate_distance(
            metric_name="euclidean",
            features=features,
            centroid=centroid,
            norm=True,
        )
        assert isinstance(result, tuple)
        distances, max_distance = result
        assert isinstance(distances, np.ndarray)
        assert np.isclose(distances.max(), 1.0)
        assert max_distance > 0

    def test_norm_with_max_returns_array(self, features_2d):
        features, centroid = features_2d
        distances = dbad.calculate_distance(
            metric_name="euclidean",
            features=features,
            centroid=centroid,
            max_distance=5.0,
            norm=True,
        )
        assert isinstance(distances, np.ndarray)

    def test_unnormalized_returns_list(self, features_2d):
        features, centroid = features_2d
        distances = dbad.calculate_distance(
            metric_name="euclidean",
            features=features,
            centroid=centroid,
            norm=False,
        )
        assert isinstance(distances, list)
        assert len(distances) == len(features)

    def test_manhattan_metric(self, features_2d):
        features, centroid = features_2d
        distances = dbad.calculate_distance(
            metric_name="manhattan",
            features=features,
            centroid=centroid,
            norm=False,
        )
        # Manhattan distance of the outlier point [10, 10] from origin is 20.
        assert np.isclose(distances[-1], 20.0)

    def test_minkowski_metric_with_p(self, features_2d):
        features, centroid = features_2d
        distances = dbad.calculate_distance(
            metric_name="minkowski",
            features=features,
            centroid=centroid,
            p=2,
            norm=False,
        )
        assert len(distances) == len(features)

    def test_mahalanobis_metric_with_inv_cov(self, features_2d):
        features, centroid = features_2d
        inv_cov = np.linalg.inv(np.cov(features, rowvar=False))
        distances = dbad.calculate_distance(
            metric_name="mahalanobis",
            features=features,
            centroid=centroid,
            inv_cov_matrix=inv_cov,
            norm=False,
        )
        assert len(distances) == len(features)


class TestPredictOutliers:
    """Tests for :func:`osbad.dbad.predict_outliers`."""

    def test_returns_four_tuple(self, features_2d):
        features, centroid = features_2d
        distances, _ = dbad.calculate_distance(
            metric_name="euclidean",
            features=features,
            centroid=centroid,
            norm=True,
        )
        result = dbad.predict_outliers(distances, features, mad_threshold=3)
        assert len(result) == 4

    def test_flags_known_outlier(self, features_2d):
        features, centroid = features_2d
        distances, _ = dbad.calculate_distance(
            metric_name="euclidean",
            features=features,
            centroid=centroid,
            norm=True,
        )
        indices, out_dist, out_feat, max_limit = dbad.predict_outliers(
            distances, features, mad_threshold=3)
        assert 5 in indices
        assert len(out_dist) == len(indices)
        assert len(out_feat) == len(indices)
        assert np.isscalar(max_limit) or isinstance(max_limit, float)


class TestPlots:
    """Smoke tests for the plotting helpers in :mod:`osbad.dbad`."""

    def test_plot_hist_distance_returns_axes(self, features_2d):
        features, centroid = features_2d
        distances, _ = dbad.calculate_distance(
            metric_name="euclidean",
            features=features,
            centroid=centroid,
            norm=True,
        )
        ax = dbad.plot_hist_distance(distances, threshold=0.5)
        assert isinstance(ax, matplotlib.axes.Axes)
        matplotlib.pyplot.close("all")
