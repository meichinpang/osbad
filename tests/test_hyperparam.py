"""Tests for :mod:`osbad.hyperparam`."""
import numpy as np
import pytest

import osbad.hyperparam as hp


class TestAggregateParamMethod:
    """Tests for :func:`osbad.hyperparam.aggregate_param_method`."""

    def test_median(self):
        assert hp.aggregate_param_method([500, 300, 250, 400, 200], "median") \
            == 300.0

    def test_mean(self):
        assert hp.aggregate_param_method([1, 2, 3, 4], "mean") == 2.5

    def test_median_int(self):
        result = hp.aggregate_param_method([500, 300, 250, 400, 200],
                                           "median_int")
        assert result == 300
        assert isinstance(result, int)

    def test_mean_int(self):
        result = hp.aggregate_param_method([1, 2, 4], "mean_int")
        assert isinstance(result, int)

    def test_mode(self):
        values = ["manhattan", "manhattan", "euclidean",
                  "manhattan", "minkowski"]
        assert hp.aggregate_param_method(values, "mode") == "manhattan"

    def test_unsupported_method_raises(self):
        with pytest.raises(ValueError):
            hp.aggregate_param_method([1, 2, 3], "unsupported")


class TestTradeOffTrialsDetection:
    """Tests for :func:`osbad.hyperparam.trade_off_trials_detection`."""

    @pytest.fixture
    def study(self):
        optuna = pytest.importorskip("optuna")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            x = trial.suggest_float("x", 0.0, 1.0)
            return x, 1.0 - x

        study = optuna.create_study(
            directions=["maximize", "maximize"],
            sampler=optuna.samplers.TPESampler(seed=1),
        )
        study.optimize(objective, n_trials=12)
        return study

    def test_returns_list_of_trials(self, study):
        optuna = pytest.importorskip("optuna")
        result = hp.trade_off_trials_detection(study)
        assert isinstance(result, list)
        assert all(
            isinstance(trial, optuna.trial.FrozenTrial) for trial in result)

    def test_trials_share_objective_values(self, study):
        result = hp.trade_off_trials_detection(study)
        # Every returned trial should share the same (rounded) objective pair.
        pairs = {
            (round(trial.values[0], 4), round(trial.values[1], 4))
            for trial in result
        }
        assert len(pairs) == 1
