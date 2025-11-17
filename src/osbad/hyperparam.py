"""
Hyperparameter tuning utilities for PyOD models with Optuna.

This module provides building blocks to define search spaces, create model
instances, run Optuna studies, summarize/aggregate best trials, visualize
Pareto fronts (recall vs. precision), and export results to CSV files. It
covers six anomaly-detection models: Isolation Forest, KNN, GMM, LOF, PCA,
and AutoEncoder.

Key features:
    - ``ModelConfigDataClass``: Frozen dataclass bundling the Optuna
      search-space function (``hp_space``), model configuration with
      tuned hyperparameters (``model_param``), model configuration without
      hyperparameters tuning (``baseline_model_param``) and the probability
      column index for outliers (``proba_col``).
    - ``MODEL_CONFIG``: Registry mapping model IDs (``iforest``, ``knn``,
      ``gmm``, ``lof``, ``pca``, ``autoencoder``) to their
      ``ModelConfigDataClass``.
    - ``objective``: Generic Optuna objective function that can handle both
      supervised hyperparameter tuning and unsupervised hyperparameter tuning.
      Samples params, builds the model, predicts outliers, and returns
      performance metrics.
    - ``trade_off_trials_detection``: Finds best compromised solutions or
      trials based on the most frequent pair of objective values.
    - ``aggregate_param_method``: Aggregate a list of values via ``median``,
      ``median_int``, or ``mode`` (with deterministic tie-breaking by
      ``mode``).
    - ``aggregate_best_trials``: Collect parameters from ``study.best_trials``
      and produce a single-row DataFrame of aggregated hyperparameters.
    - ``plot_proxy_pareto_front``: Plot the Pareto front for proxy metrics
      (loss score vs. inlier score), annotates best compromised solution, 
      and save to the artifacts folder.
    - ``evaluate_hp_perfect_score_pct``: Compute the percentage of trials
      with perfect recall and precision (value == 1) and log per-trial
      scores.
    - ``plot_pareto_front``: Plot the Pareto front (recall vs. precision),
      annotate perfect-score percentages, and save to the artifacts folder.
    - ``export_current_hyperparam``: Append best hyperparameters for a cell
      to a CSV (skips if already present) and return the updated DataFrame.
    - ``export_current_model_metrics``: Append evaluation metrics for a
      (model, cell) pair to a CSV (skips if already present) and return the
      updated DataFrame.

Configuration:
    - ``RANDOM_STATE``: Shared random seed used by model factories.
    - ``bconf.PIPELINE_OUTPUT_DIR``: Base directory where per-cell figures
      are saved.

.. code-block::

    import osbad.hyperparam as hp
"""
# Standard library
import logging
import os
import pathlib
import sys
from dataclasses import dataclass
from importlib import reload
from statistics import mode
from typing import Any, Callable, Dict, List, Literal, Tuple, Union, Optional

# Third-party libraries
import fireducks.pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import optuna
from matplotlib import rcParams
import pyod
import torch
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.gmm import GMM
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.pca import PCA

rcParams["text.usetex"] = True

# Custom osbad library for anomaly detection
import osbad.config as bconf
from osbad.config import CustomFormatter
from osbad.model import ModelRunner

# Initiate logging for methods --------------------------------------------

def _customize_logger(
    logger_name: str,
    output_log: bool=True) -> logging.Logger:
    """
    Create and configure a custom logger.

    This function initializes a logger with the specified name and
    configures its logging level based on ``output_log``. Logs are
    formatted with ``CustomFormatter`` and streamed to ``sys.stdout``.
    Existing handlers are cleared to prevent duplicate outputs.

    Args:
        logger_name (str): Name assigned to the logger instance.
        output_log (bool, optional): If True, sets logging level to
            ``logging.INFO``. If False, sets logging level to
            ``logging.WARNING``. Defaults to True.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger_name = logger_name
    logger = logging.getLogger(logger_name)

    # Logging levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    if output_log:
        logger_level = logging.INFO
    else:
        logger_level = logging.WARNING

    logger.setLevel(logger_level)

    # use sys.stdout to output logging to notebook without red boxes around log
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setLevel(logger_level)

    _ch.setFormatter(CustomFormatter())

    # To remove duplicated output from loggers
    # https://stackoverflow.com/questions/7173033/duplicate-log-output-when-using-python-logging-module
    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.addHandler(_ch)

    return logger

# Shared seed
RANDOM_STATE = 42

# Device for AutoEncoder model
AE_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu")


# Config plumbing ------------------------------------------------------------

# HpSpaceFuncType is a type alias
# a function that takes an optuna.trial.Trial and returns a dict of
# hyperparameters
HpSpaceFuncType = Callable[[optuna.trial.Trial], Dict[str, Any]]
"""
A type alias for a function that takes an ``optuna.trial.Trial`` as input and
returns a dictionary mapping hyperparameter names ``str`` to their suggested
values ``Any``.

Example:
    .. code-block::

        def knn_hp_space(trial: optuna.trial.Trial) -> Dict[str, Any]:

            hyperparam_dict =  {
                "contamination": trial.suggest_float(
                    "contamination", 0.0, 0.5),
                "n_neighbors": trial.suggest_int(
                    "n_neighbors", 2, 50, step=2),
                "method": trial.suggest_categorical(
                    "method", ["largest", "mean", "median"]),
                "metric": trial.suggest_categorical(
                    "metric", ["minkowski", "euclidean", "manhattan"]),
                "threshold": trial.suggest_float(
                    "threshold", 0.0, 1.0),
            }

            return hyperparam_dict
"""

# ----------------------------------------------------------------------------
# ModelParamFuncType is a type alias
# a function that takes that dict of hyperparameters and
# returns a model instance
ModelParamFuncType  = Callable[[Dict[str, Any]], Any]
"""
A type alias for a function type that takes a dictionary of hyperparameters
as input ``Dict[str, Any]`` and returns a model instance
(e.g., KNN, IForest, GMM) ``Any``.

Example:
    .. code-block::

        input_hyperparam_dict = {
            "contamination": 0.1,
            "n_neighbors": 10,
            "method": "mean",
            "metric": "euclidean",
            "threshold": 0.5,
        }

        def knn_model_param(param: Dict[str, Any]) -> Any:

            model_instance = KNN(
                contamination=param["contamination"],
                n_neighbors=param["n_neighbors"],
                method=param["method"],
                metric=param["metric"],
                n_jobs=-1,
            )

            return model_instance

        # output will be:
        # KNN(contamination=0.1, n_neighbors=10, method="mean",
        # metric="euclidean", n_jobs=-1)
"""

PyODModelType = Union[
    pyod.models.iforest.IForest,
    pyod.models.knn.KNN,
    pyod.models.gmm.GMM,
    pyod.models.lof.LOF,
    pyod.models.pca.PCA,
    pyod.models.auto_encoder.AutoEncoder]
"""Type alias for supported PyOD anomaly detection models.

This alias unifies the set of PyOD estimators commonly used in the
benchmarking pipeline, enabling consistent type annotations and
improved readability.

Supported models:
    * ``IForest``: Isolation Forest model.
    * ``KNN``: K-Nearest Neighbors–based outlier detector.
    * ``GMM``: Gaussian Mixture Model for density-based detection.
    * ``LOF``: Local Outlier Factor for neighborhood-based detection.
    * ``PCA``: Principal Component Analysis for subspace-based detection.
    * ``AutoEncoder``: Neural network–based autoencoder for
      reconstruction-based detection.
"""

# ----------------------------------------------------------------------------
# Declares an immutable container class.
# Once created, its fields can’t be changed.
@dataclass(frozen=True)
class ModelConfigDataClass:
    """
    Immutable container class for the model configuration.

    Stores the search space function for Optuna trials, the model
    factory function, and the probability column index used for
    PyOD estimators.
    """
    hp_space: HpSpaceFuncType
    """
    Function that defines the hyperparameter search space for an Optuna trial.
    """

    model_param: ModelParamFuncType
    """
    Function that builds a model instance from a set of hyperparameters.
    """

    # Model instance without hyperparameter tuning
    baseline_model_param: Optional[Callable[[], PyODModelType]] = None
    """
    Function that builds a model instance without hyperparameter tuning.
    """

    # For PyOD estimators:
    # column 0 = probability of being inlier (normal)
    # column 1 = probability of being outlier
    proba_col: int = 1
    """
    Index of the probability column in PyOD estimators. Column 0 is the inlier
    probability, and column 1 is the outlier probability. Defaults to 1.
    """

# Model registry -------------------------------------------------------------
# Taking external hp schema as input to the hyperparameter tuning

def _suggest_float(trial, schema, name: str):
    params = {**schema.get(name)}
    suggest_float_trial = trial.suggest_float(
        name,
        low=params["low"],
        high=params["high"])

    return suggest_float_trial

def _suggest_int(trial, schema, name: str):
    params = {**schema.get(name)}
    suggest_int_trial = trial.suggest_int(
        name,
        low=params["low"],
        high=params["high"])

    return suggest_int_trial

def _suggest_categorical(trial, schema, name: str):
    """
    Suggest a categorical parameter using a list from schema.

    Args:
        trial (optuna.trial.Trial): Optuna trial object used to suggest
            parameters.
        schema (dict): Dictionary containing hyperparameter schema. For
            categorical parameters, the value should be a list of choices.
        name (str): Name of the hyperparameter to retrieve.

    Returns:
        Any: The value selected from the list of categorical choices.

    Raises:
        ValueError: If the schema value for the given name is not a list.
    """
    choices = schema.get(name)

    if not isinstance(choices, list):
        raise ValueError(
            f"Expected a list of choices for '{name}', "
            f"but got {type(choices).__name__}"
        )

    return trial.suggest_categorical(name, choices)

# ----------------------------------------------------------------------------
# Reload the config module to refresh in-memory variables
# especially after updating parameters during runtime
reload(bconf)

DATA_SOURCE: Literal["tohoku", "severson"] = bconf.HP_DATA_SOURCE
"""
Global selector for the active hyperparameter data source.

Loaded from ``bconf.HP_DATA_SOURCE`` during import to keep module
state aligned with the currently configured dataset registry.
"""

def _grab(model: str):
    """
    Retrieve hyperparameter configuration for a given model and data source.

    This function dynamically retrieves the hyperparameter configuration
    dictionary for a specified model by constructing the attribute name
    based on the model identifier and the global DATA_SOURCE setting.

    Args:
        model (str): The model identifier (e.g., "iforest", "knn", "gmm",
            "lof", "pca", "autoencoder").

    Returns:
        Dict[str, Any]: The hyperparameter configuration dictionary for the
            specified model and data source combination.

    Raises:
        AttributeError: If the constructed attribute name does not exist in
            the bconf module.

    Example:
        .. code-block::

            DATA_SOURCE = "tohoku"
            _IFOREST_HP_CONFIG = _grab("iforest")
    """
    suffix = DATA_SOURCE.lower()
    attr = f"_{model}_hp_config_{suffix}"
    return getattr(bconf, attr)

_IFOREST_HP_CONFIG = _grab("iforest")
_KNN_HP_CONFIG = _grab("knn")
_GMM_HP_CONFIG = _grab("gmm")
_LOF_HP_CONFIG = _grab("lof")
_PCA_HP_CONFIG = _grab("pca")
_AUTOENCODER_HP_CONFIG = _grab("autoencoder")
"""
Hyperparameter schema cache for supported anomaly detection models.

Each constant stores the preloaded hyperparameter configuration for a
specific PyOD estimator under the active ``DATA_SOURCE`` selection.
"""

MODEL_CONFIG: Dict[str, ModelConfigDataClass] = {
    "iforest": ModelConfigDataClass(
        hp_space=lambda trial: {
            "contamination": _suggest_float(
                trial, _IFOREST_HP_CONFIG, "contamination"),
            "n_estimators": _suggest_int(
                trial, _IFOREST_HP_CONFIG, "n_estimators"),
            "max_samples": _suggest_int(
                trial, _IFOREST_HP_CONFIG, "max_samples"),
            "threshold": _suggest_float(
                trial, _IFOREST_HP_CONFIG, "threshold"),
        },
        model_param=lambda param: IForest(
            behaviour="new",
            contamination=param["contamination"],
            n_estimators=param["n_estimators"],
            max_samples=param["max_samples"],
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        baseline_model_param=lambda: IForest(
            behaviour="new",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    ),
    "knn": ModelConfigDataClass(
        hp_space=lambda trial: {
            "contamination": _suggest_float(
                trial, _KNN_HP_CONFIG, "contamination"),
            "n_neighbors": _suggest_int(
                trial, _KNN_HP_CONFIG, "n_neighbors"),
            "method": _suggest_categorical(
                trial, _KNN_HP_CONFIG, "method"),
            "metric": _suggest_categorical(
                trial, _KNN_HP_CONFIG, "metric"),
            "threshold": _suggest_float(
                trial, _KNN_HP_CONFIG, "threshold"),
        },
        model_param=lambda param: KNN(
            contamination=param["contamination"],
            n_neighbors=param["n_neighbors"],
            method=param["method"],
            metric=param["metric"],
            n_jobs=-1
        ),
        baseline_model_param=lambda: KNN(
            n_jobs=-1
        ),
    ),
    "gmm": ModelConfigDataClass(
        hp_space=lambda trial: {
            "n_components": _suggest_int(
                trial, _GMM_HP_CONFIG, "n_components"),
            "covariance_type": _suggest_categorical(
                trial, _GMM_HP_CONFIG, "covariance_type"),
            "init_param": _suggest_categorical(
                trial, _GMM_HP_CONFIG, "init_param"),
            "contamination": _suggest_float(
                trial, _GMM_HP_CONFIG, "contamination"),
            "threshold": _suggest_float(
                trial, _GMM_HP_CONFIG, "threshold"),
        },
        model_param=lambda param: GMM(
            n_components=param["n_components"],
            covariance_type=param["covariance_type"],
            init_params=param["init_param"],
            contamination=param["contamination"],
            random_state=RANDOM_STATE
        ),
        baseline_model_param=lambda: GMM(
            n_components=2,
            random_state=RANDOM_STATE
        ),
    ),
    "lof": ModelConfigDataClass(
        hp_space=lambda trial: {
            "n_neighbors": _suggest_int(
                trial, _LOF_HP_CONFIG, "n_neighbors"),
            "leaf_size": _suggest_int(
                trial, _LOF_HP_CONFIG, "leaf_size"),
            "metric": _suggest_categorical(
                trial, _LOF_HP_CONFIG, "metric"),
            "contamination": _suggest_float(
                trial, _LOF_HP_CONFIG, "contamination"),
            "threshold": _suggest_float(
                trial, _LOF_HP_CONFIG, "threshold"),
        },
        model_param=lambda param: LOF(
            n_neighbors=param["n_neighbors"],
            leaf_size=param["leaf_size"],
            metric=param["metric"],
            contamination=param["contamination"],
            novelty=True,
            n_jobs=-1
        ),
        baseline_model_param=lambda: LOF(
            novelty=True,
            n_jobs=-1
        ),
    ),
    "pca": ModelConfigDataClass(
        hp_space=lambda trial: {
            "n_components": _suggest_int(
                trial, _PCA_HP_CONFIG, "n_components"),
            "contamination": _suggest_float(
                trial, _PCA_HP_CONFIG, "contamination"),
            "threshold": _suggest_float(
                trial, _PCA_HP_CONFIG, "threshold"),
        },
        model_param=lambda param: PCA(
            n_components=param["n_components"],
            contamination=param["contamination"]
        ),
        baseline_model_param=lambda: PCA(
            n_components=2,
        ),
    ),
    "autoencoder": ModelConfigDataClass(
        hp_space=lambda trial: {
            "batch_size": _suggest_int(
                trial, _AUTOENCODER_HP_CONFIG, "batch_size"),
            "epoch_num": _suggest_int(
                trial, _AUTOENCODER_HP_CONFIG, "epoch_num"),
            "learning_rate": _suggest_float(
                trial, _AUTOENCODER_HP_CONFIG, "learning_rate"),
            "dropout_rate": _suggest_float(
                trial, _AUTOENCODER_HP_CONFIG, "dropout_rate"),
            "threshold": _suggest_float(
                trial, _AUTOENCODER_HP_CONFIG, "threshold"),
        },
        # hidden_neuron_list = The number of neurons per hidden layers.
        # So the network has the structure as
        # [feature_size, 64, 32, 32, 64, feature_size]
        model_param=lambda param: AutoEncoder(
            batch_size=param["batch_size"],
            epoch_num=param["epoch_num"],
            lr=param["learning_rate"],
            dropout_rate=param["dropout_rate"],
            hidden_neuron_list=[2, 64, 32, 32, 64, 2],
            optimizer_name="adam",
            device=AE_DEVICE,
            random_state=RANDOM_STATE
        ),
        baseline_model_param=lambda: AutoEncoder(
            hidden_neuron_list=[2, 64, 32, 32, 64, 2],
            device=AE_DEVICE,
            optimizer_name="adam",
            random_state=RANDOM_STATE
        ),
    ),
}
"""
Dictionary mapping model identifiers to their configurations.

Each entry contains a ModelConfigDataClass object that defines the search
space for Optuna hyperparameter optimization (`hp_space`) and a
factory function (`model_param`) to create the corresponding model
with the chosen hyperparameters.

The following model identifiers are supported:
    - "iforest": Isolation Forest
    - "knn": k-Nearest Neighbors
    - "gmm": Gaussian Mixture Model
    - "lof": Local Outlier Factor
    - "pca": Principal Component Analysis
    - "autoencoder": AutoEncoder

Args:
    key (str): Model identifier.
    value (ModelConfigDataClass): Configuration for searching hyperparameters
        and to instantiate the corresponding model creation based on the
        hyperparameter search space.
"""

# Generic Optuna objective ---------------------------------------------------


def objective(
    trial: optuna.trial.Trial,
    model_id: Literal["iforest","knn","gmm","lof","pca","autoencoder"],
    df_feature_dataset: pd.DataFrame,
    selected_feature_cols: list,
    selected_cell_label: str,
    df_benchmark_dataset: Optional[pd.DataFrame] = None,
) -> Tuple[float, float]:

    """
    Optimize model hyperparameters using Optuna trial.

    This function evaluates a given anomaly detection model by sampling
    hyperparameters from the trial, training the model, predicting
    outliers, and computing evaluation metrics. If a benchmark dataset
    is provided, recall and precision are computed. Otherwise, proxy
    evaluation metrics are used based on cycle index and model input
    features.

    Args:
        trial (optuna.trial.Trial): Optuna trial object used to
            suggest hyperparameters.
        model_id (Literal): Identifier of the model to optimize. Must be
            one of "iforest", "knn", "gmm", "lof", "pca", or
            "autoencoder".
        df_feature_dataset (pd.DataFrame): Feature dataset containing
            model input features.
        selected_feature_cols (list): List of selected feature column names.
        selected_cell_label (str): Label identifying the cell for which
            the model is being trained.
        df_benchmark_dataset (Optional[pd.DataFrame]): Benchmark dataset
            used to evaluate predicted outliers. If None, proxy evaluation
            is performed.

    Returns:
        Tuple[float, float]: If benchmark dataset is provided, returns
        recall and precision scores. Otherwise, returns loss score and
        inliers score from proxy evaluation.

    Example:
        .. code-block::

            import optuna
            import osbad.hyperparam as hp

            # Reload the hp module to refresh in-memory variables
            # especially after updating parameters
            from importlib import reload
            reload(hp)

            # Check if the schema in the script has been updated
            # based on the latest updated constraints
            print("Current hyperparameter config:")
            print(hp._IFOREST_HP_CONFIG)
            print("-"*70)

            # Instantiate an optuna study for iForest model
            sampler = optuna.samplers.TPESampler(seed=42)

            selected_feature_cols = (
                "log_max_diff_dQ",
                "log_max_diff_dV")

            if_study = optuna.create_study(
                study_name="iforest_hyperparam",
                sampler=sampler,
                directions=["maximize","maximize"])

            if_study.optimize(
                lambda trial: hp.objective(
                    trial,
                    model_id="iforest",
                    df_feature_dataset=df_features_per_cell,
                    selected_feature_cols=selected_feature_cols,
                    df_benchmark_dataset=df_selected_cell,
                    selected_cell_label=selected_cell_label),
                n_trials=20)
    """

    cfg = MODEL_CONFIG[model_id]
    params = cfg.hp_space(trial)

    runner = ModelRunner(
        cell_label=selected_cell_label,
        df_input_features=df_feature_dataset,
        selected_feature_cols=selected_feature_cols)

    if df_benchmark_dataset is not None:

        Xdata = runner.create_model_x_input()
        model = cfg.model_param(params)
        model.fit(Xdata)

        proba = model.predict_proba(Xdata)  # shape (n_samples, 2)
        (pred_outlier_indices,
        pred_outlier_score) = runner.pred_outlier_indices_from_proba(
            proba=proba,
            threshold=params["threshold"],
            outlier_col=cfg.proba_col
        )
        recall, precision = runner.evaluate_indices(
            df_benchmark_dataset, pred_outlier_indices)

        return recall, precision

    else:
        # Xdata: with cycle index and two anomaly detection features
        Xdata = runner.create_model_x_input()

        #extract cycle index predictor for regression proxy
        cycle_idx = Xdata[:,0].reshape(-1,1)

        #extract target features for Pyod model and regression proxy
        features = Xdata[:,1:]

        model = cfg.model_param(params)
        model.fit(features)

        proba = model.predict_proba(features)  # shape (n_samples, 2)
        (pred_outlier_indices,
         pred_outlier_score) = runner.pred_outlier_indices_from_proba(
            proba=proba,
            threshold=params["threshold"],
            outlier_col=cfg.proba_col
        )

        loss_score, inliers_score = runner.proxy_evaluate_indices(
            pred_outlier_indices, cycle_idx, features)

        if inliers_score < 0.5:
            raise optuna.TrialPruned()

        return loss_score, inliers_score

def trade_off_trials_detection(
        study: optuna.study.Study,
        ) -> List[optuna.trial.FrozenTrial]:
    """
    Identifies and returns the list of trials from the best trials in an
    Optuna study that share the most frequently occurring combination of 
    objective values.

    This function is useful in multi-objective optimization scenarios where 
    multiple trials may be considered optimal. It selects the subset of trials
    that exhibit the most common trade-off between the two objectives.

    Args:
        study (optuna.study.Study): An Optuna study object containing completed 
            trials.
    Returns:
        List[optuna.trial.FrozenTrial]: A list of FrozenTrial objects that 
        match the most frequent pair of objective values among the best trials.
    """
    obj_scores = [(trial.values[0], trial.values[1])
                for trial in study.best_trials]
    
    freq_counter = Counter(obj_scores)

    high_frq_obj_score = list(freq_counter.most_common(1)[0][0])

    optimal_trials_list = [trial for trial in study.best_trials
                                if round(trial.values[0], 4) ==
                                round(high_frq_obj_score[0], 4)
                                and round(trial.values[1], 4) ==
                                round(high_frq_obj_score[1], 4)]

    return optimal_trials_list

# Generic aggregation of best trials -----------------------------------------

Agg = Literal["median", "mean", "median_int", "mean_int", "mode"]
"""
Type alias for the aggregation methods.

Represents allowed strategies for aggregating a list of values:

    - "median": Returns the median as a float.
    - "mean": Returns the mean as a float.
    - "median_int": Returns the median as an integer.
    - "mean_int": Returns the mean as an integer.
    - "mode": Returns the most frequent value.
"""

def aggregate_param_method(values: List[Any], how: Agg):
    """
    Aggregate a list of values using the given method.

    Supports median and mean as float, median_int and mean_int as integer
    and mode for strings. Raises ValueError if an unsupported method is
    provided.

    Args:
        values (List[Any]): List of values to aggregate.
        how (Agg): Aggregation method, one of ``median``, ``mean``,
            ``median_int``, ``mean_int`` or ``mode``.

    Returns:
        Any: Aggregated result based on the specified method.

    Raises:
        ValueError: If ``how`` is not a supported aggregation method.

    Example:
        .. code-block::

            >>> aggregate_param_method([500, 300, 250, 400, 200], "median")
            300.0
            >>> aggregate_param_method([500, 300, 250, 400, 200], "median_int")
            300
            >>> aggregate_param_method(
                ['manhattan', 'manhattan', 'euclidean',
                'manhattan', 'minkowski'], "mode")
            'manhattan'

    .. Note::

        If there is a tie in the most frequent parameter, for example,
        ``method`` = ``['largest','largest','median','median','mean']``,
        the first most frequent parameter ``largest`` will be chosen.
        """
    if how == "median":
        return float(np.median(values))
    if how == "mean":
        return float(np.mean(values))
    if how == "median_int":
        return int(np.median(values))
    if how == "mean_int":
        return int(np.mean(values))
    if how == "mode":
        return mode(values)
    raise ValueError(how)

def aggregate_best_trials(
    best_trials: List[optuna.trial.FrozenTrial],
    cell_label: str,
    model_id: str,
    schema: Dict[str, Agg],
) -> pd.DataFrame:
    """
    Aggregate parameters from the best Optuna trials.

    Collects hyperparameters from the best trials of a study and
    aggregates them using rules defined in the schema. Each parameter
    is reduced to a single representative value using median,
    median_int, or mode.

    Args:
        best_trials (List[optuna.trial.FrozenTrial]): A list of best
            trials obtained using Pareto optimization or the additional
            frequency analysis step in case of proxy hyperparameter tuning
            method.
        cell_label (str): Identifier for the experimental cell.
        model_id (str): Identifier of the ML-model. Allowed values are
            "iforest", "knn", "gmm", "lof", "pca", "autoencoder".
        schema (Dict[str, Agg]): Mapping of parameter names to
            aggregation strategies. Allowed values are "median",
            "median_int", and "mode".

    Returns:
        pd.DataFrame: A single-row DataFrame containing the model ID,
        cell label, and aggregated hyperparameters.

    Example:
        .. code-block::

            schema_knn = {
                "contamination": "median",
                "n_neighbors": "median_int",
                "method": "mode",
                "metric": "mode",
                "threshold": "median"}

            df_knn = hp.aggregate_best_trials(
                study.best_trials,
                cell_label=selected_cell_label,
                model_id="knn",
                schema=schema_knn)
        """

    if not best_trials:
        return pd.DataFrame(
            [{"ml_model": model_id, "cell_index": cell_label}])

    # An example schema for knn
    # schema_knn = {
    #     "contamination": "median",
    #     "n_neighbors": "median_int",
    #     "method": "mode",
    #     "metric": "mode",
    #     "threshold": "median",
    # }

    collected_param_dict = {k: [] for k in schema.keys()}

    # Initialize an empty dict to store best trials hyperparameter
    # where k correspond to the keys from the schema:
    # collected_param_dict = {
    #   'contamination': [], 'n_neighbors': [],
    #   'method': [], 'metric': [], 'threshold': []}

    # Extract the parameters from the best trial of the optuna optimization
    # And store them in collected_param_dict
    for tr in best_trials:
        for p in schema.keys():
            collected_param_dict[p].append(tr.params[p])

    # If there are 8 best trials output from the optuna study,
    # collected_param_dict will store the param of each best trial in the dict

    # collected_param_dict = {
    # 'contamination': [0.21597250932105788, 0.15230688458668534,
    #                   0.17837666334679464, 0.31164906341377896,
    #                   0.40562173451259353, 0.4049830638482713,
    #                   0.1941324729699589, 0.42551617277539],
    # 'n_neighbors': [16, 6, 16, 18, 36, 8, 40, 10],
    # 'method': ['largest','largest','median','median','largest',
    #            'mean', 'median', 'mean'],
    # 'metric': ['manhattan', 'manhattan', 'euclidean', 'manhattan',
    #            'manhattan', 'euclidean', 'euclidean', 'minkowski'],
    # 'threshold': [0.19967378215835974, 0.2587799816000169,
    #               0.1987156815341724, 0.4722149251619493,
    #               0.3476334304965827, 0.18131949650344986,
    #               0.4523263128776084, 0.1499228256998617]}

    # Some hyperparameter in PyOD require specific type
    # For example:
    # For iForest: n_estimators and max_samples must have the type int
    # For knn: method and metric must have the type str
    # Use the aggregare_param_method to calculate the median parameter or
    # most frequent param out of the best trials.

    # aggregated =  {
    #   'contamination': 0.2638107863674184,
    #   'n_neighbors': 16,
    #   'method': 'largest',
    #   'metric': 'manhattan',
    #   'threshold': 0.22922688187918833}

    aggregated = {}
    for name, how in schema.items():
        aggregated[name] = aggregate_param_method(
            collected_param_dict[name],
            how)

    # Standardized output keys
    out = {"ml_model": model_id, "cell_index": cell_label}
    out.update(aggregated)
    return pd.DataFrame([out])

def plot_proxy_pareto_front(
    model_study: optuna.study.study.Study,
    best_trials_list: List[optuna.trial.FrozenTrial],
    selected_cell_label: str,
    fig_title: str) -> None:

    """
    Plot and save the Pareto front of proxy evaluation metrics
    i.e. loss scores vs. inliers count scores.

    This function generates a Pareto front plot from an Optuna study
    that optimizes for both regression loss and predicted inlier count score.
    The plot also highlights the best compromised solution or trade-off 
    solution obtained using frequency analysis out of all pareto-optimal 
    trials.

    Args:
        model_study (optuna.study.study.Study): Optuna study object
            containing trials with recall and precision scores as
            objectives.
        best_trials_list (List[optuna.trial.FrozenTrial]): A list of best
            trials obtained using frequency analysis step in case of proxy
            hyperparameter tuning method.
        selected_cell_label (str): Identifier for the evaluated cell,
            used to generate the output file path.
        fig_title (str): Title of the plot and basis for the output file
            name.
        output_log_status (bool, optional): If True, enables logging of
            intermediate evaluation steps. Defaults to False.

    Returns:
        None: The function saves the Pareto front plot as a PNG file in
        the artifacts directory associated with the selected cell.

    Example:
        .. code-block::

            hp.plot_proxy_pareto_front(
                if_study,
                selected_cell_label,
                fig_title="Isolation Forest Pareto Front")
    """
    # extracting loss score value and inlier score value at knee point
    loss_infl = best_trials_list[0].values[0]
    inlier_infl = best_trials_list[0].values[1]

    # plotting pareto front with proxy metrics
    axplot = optuna.visualization.matplotlib.plot_pareto_front(
        model_study,
        target_names=[
            "pred_mse_score",
            "pred_inlier_score"])

    axplot.scatter(
        loss_infl, inlier_infl,
        marker='X',
        color='green',
        label='Best Compromise Solution')

    #axplot.axis("equal")
    axplot.set_xlim([0,1.05])

    axplot.set_xlabel(
        "Regression loss score",
        color="black",
        fontsize=14)

    axplot.set_ylabel(
        "Inlier count score",
        color="black",
        fontsize=14)

    axplot.set_title(
        fig_title,
        fontsize=16)

    axplot.legend(
        loc='right',
        fontsize=10)

    axplot.patch.set_facecolor("white")
    axplot.spines["bottom"].set_color("black")
    axplot.spines["top"].set_color("black")
    axplot.spines["left"].set_color("black")
    axplot.spines["right"].set_color("black")

    axplot.grid(
        color="grey",
        linestyle="-",
        linewidth=0.25,
        alpha=0.7)

    # convert the fig title to snake_case for filepath standardization
    output_fig_filename = (
        fig_title.casefold().replace(" ","_")
        + "_" + selected_cell_label
        + ".png")

    selected_cell_artifacts_dir = bconf.artifacts_output_dir(
        selected_cell_label)

    fig_output_path = (
        selected_cell_artifacts_dir.joinpath(output_fig_filename))

    plt.savefig(
        fig_output_path,
        dpi=600,
        bbox_inches="tight")


def evaluate_hp_perfect_score_pct(
    model_study: optuna.study.study.Study,
    output_log_bool: bool=True):
    """
    Evaluate percentage of trials with the perfect recall and precision score.

    This function analyzes an Optuna study and calculates the percentage
    of trials that achieved a perfect recall score of 1 and a perfect
    precision score of 1. Trial-level recall and precision values are
    logged for inspection. The function provides an overview of how often
    hyperparameter trials reach ideal performance.

    Args:
        model_study (optuna.study.study.Study): Optuna study object
            containing hyperparameter optimization trials. Each trial is
            expected to have recall and precision as its objective values.
        output_log_bool (bool, optional): If True, enables logging output.
            Defaults to True.

    Returns:
        Tuple[float, float]: A tuple containing:
            - recall_score_pct (float): Percentage of trials with recall=1.
            - precision_score_pct (float): Percentage of trials with
              precision=1.

    Example:
        .. code-block::

            sampler = optuna.samplers.TPESampler(seed=42)

            selected_feature_cols = (
                "log_max_diff_dQ",
                "log_max_diff_dV")

            if_study = optuna.create_study(
                study_name="iforest_hyperparam",
                sampler=sampler,
                directions=["maximize","maximize"])

            if_study.optimize(
                lambda trial: hp.objective(
                    trial,
                    model_id="iforest",
                    df_feature_dataset=df_features_per_cell,
                    selected_feature_cols=selected_feature_cols,
                    df_benchmark_dataset=df_selected_cell,
                    selected_cell_label=selected_cell_label),
                n_trials=20)

            (recall_score_pct,
            precision_score_pct) = hp.evaluate_hp_perfect_score_pct(
                model_study=if_study)
    """
    total_hp_trial = len(model_study.trials)

    logger = _customize_logger(
        logger_name="hp_perfect_score_pct",
        output_log=output_log_bool)

    logger.info(f"Total trial count: {total_hp_trial}")

    # ------------------------------------------------------------------
    # extract the recall and precision score per trial
    recall_score_list = []
    precision_score_list = []

    for trial_idx in range(total_hp_trial):
        eval_score = model_study.trials[trial_idx].values

        recall_score_per_trial = eval_score[0]
        precision_score_per_trial = eval_score[1]


        logger.info(f"Trial {trial_idx}:\n" +
            f"Recall score per trial: {recall_score_per_trial}\n" +
            f"Precision score per trial: {precision_score_per_trial}")
        logger.info("-"*70)
        recall_score_list.append(recall_score_per_trial)
        precision_score_list.append(precision_score_per_trial)

    # ------------------------------------------------------------------
    # Calculate the percentage of recall score = 1 among all trials
    recall_score_arr = np.array(recall_score_list)

    # boolean mask for score equals to 1
    mask = (recall_score_arr == 1)

    # count score equals to 1
    perfect_score_count = np.sum(mask)
    recall_score_pct = ((perfect_score_count/total_hp_trial)*100)
    logger.info(f"Percentage perfect recall score: {recall_score_pct}")

    # ------------------------------------------------------------------
    # Calculate the percentage of precision score = 1 among all trials
    precision_score_arr = np.array(precision_score_list)

    # boolean mask for score equals to 1
    mask = (precision_score_arr == 1)

    # count score equals to 1
    perfect_score_count = np.sum(mask)
    precision_score_pct = ((perfect_score_count/total_hp_trial)*100)
    logger.info(f"Percentage perfect precision score: {precision_score_pct}")
    logger.info("*"*100)

    return recall_score_pct, precision_score_pct

def plot_pareto_front(
    model_study: optuna.study.study.Study,
    selected_cell_label: str,
    fig_title: str,
    output_log_status: bool=False) -> None:
    """
    Plot and save the Pareto front of recall vs. precision scores.

    This function generates a Pareto front plot from an Optuna study
    that optimizes for both recall and precision. The plot includes an
    annotation showing the percentage of trials with perfect recall and
    precision scores. The figure is customized with labels, legends, and
    formatting before being saved as a PNG file.

    Args:
        model_study (optuna.study.study.Study): Optuna study object
            containing trials with recall and precision scores as
            objectives.
        selected_cell_label (str): Identifier for the evaluated cell,
            used to generate the output file path.
        fig_title (str): Title of the plot and basis for the output file
            name.
        output_log_status (bool, optional): If True, enables logging of
            intermediate evaluation steps. Defaults to False.

    Returns:
        None: The function saves the Pareto front plot as a PNG file in
        the artifacts directory associated with the selected cell.

    Example:
        .. code-block::

            hp.plot_pareto_front(
                if_study,
                selected_cell_label,
                fig_title="Isolation Forest Pareto Front")
        """

    (recall_score_pct,
     precision_score_pct) = evaluate_hp_perfect_score_pct(
     model_study,
     output_log_bool=output_log_status)


    axplot = optuna.visualization.matplotlib.plot_pareto_front(
        model_study,
        target_names=[
            "pred_recall_score",
            "pred_precision_score"])

    axplot.axis("equal")

    axplot.set_xlabel(
        "Recall score",
        color="black",
        fontsize=14)

    axplot.set_ylabel(
        "Precision score",
        color="black",
        fontsize=14)

    axplot.set_title(
        fig_title,
        fontsize=16)

    axplot.legend(
        loc='upper right',
        fontsize=10)

    axplot.annotate(
        text=(
            f"Perfect recall score: {np.round(recall_score_pct)}\%\n" +
            f"Perfect precision score: {np.round(precision_score_pct)}\%"),
        xy=(0.25,0.3),
        xytext=(0.25,0.3),
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="yellow",
            ec="yellow",
            alpha=0.2))

    axplot.set_xlim([0,1.4])

    axplot.patch.set_facecolor("white")
    axplot.spines["bottom"].set_color("black")
    axplot.spines["top"].set_color("black")
    axplot.spines["left"].set_color("black")
    axplot.spines["right"].set_color("black")

    axplot.grid(
        color="grey",
        linestyle="-",
        linewidth=0.25,
        alpha=0.7)

    # convert the fig title to snake_case for filepath standardization
    output_fig_filename = (
        fig_title.casefold().replace(" ","_")
        + "_" + selected_cell_label
        + ".png")

    selected_cell_artifacts_dir = bconf.artifacts_output_dir(
        selected_cell_label)

    fig_output_path = (
        selected_cell_artifacts_dir.joinpath(output_fig_filename))

    plt.savefig(
        fig_output_path,
        dpi=600,
        bbox_inches="tight")

def export_current_hyperparam(
    df_best_param_current_cell: pd.DataFrame,
    selected_cell_label: str,
    export_csv_filepath: Union[pathlib.PosixPath, str],
    if_exists: Literal["replace", "keep"] = "replace",
    output_log_bool: bool=True):
    """
    Export best hyperparameters for a cell to a CSV file.

    This function manages the storage of best hyperparameters for a
    specified cell. If the cell's hyperparameters already exist in the
    CSV file, the behavior depends on ``if_exists``: either the rows are
    replaced or kept. If the cell is not found, a new row is created.
    Logging tracks the export status and duplication handling.

    Args:
        df_best_param_current_cell (pd.DataFrame): DataFrame containing the
            best hyperparameters for the current cell.
        selected_cell_label (str): Identifier for the evaluated cell.
        export_csv_filepath (Union[pathlib.PosixPath, str]): Path to the
            CSV file where hyperparameters are stored.
        if_exists (Literal["replace", "keep"], optional): Action to take
            if the cell already exists in the CSV. Defaults to
            ``"replace"``.
        output_log_bool (bool, optional): If True, enables logging output.
            Defaults to True.

    Returns:
        pd.DataFrame: Updated DataFrame containing hyperparameters from both
        existing records and the current cell.

    Raises:
        ValueError: If ``if_exists`` is not ``"replace"`` or ``"keep"``.

    Example:
        .. code-block::

            # Export current hyperparameters to CSV
            hyperparam_filepath =  PIPELINE_OUTPUT_DIR.joinpath(
                "hyperparams_autoencoder_tohoku.csv")

            hp.export_current_hyperparam(
                df_autoencoder_hyperparam,
                selected_cell_label,
                export_csv_filepath=hyperparam_filepath,
                if_exists="replace")
    """

    logger = _customize_logger(
        logger_name="export_current_hyperparam",
        output_log=output_log_bool)

    if os.path.exists(export_csv_filepath):
        df_csv = pd.read_csv(
            export_csv_filepath)

        # Status check if the current model hyperparam
        # already exists in the hyperparam inventories
        existed_before = (
            df_csv["cell_index"].astype(str).eq(selected_cell_label).any())

        logger.info("Have the hyperparam for "
                + f"{selected_cell_label} been evaluated?")
        logger.info(existed_before)
        logger.info("*"*70)
    else:
        df_csv = None
        existed_before = False

    # Decide action based on if_exists
    if existed_before:
        if if_exists == "replace":
            logger.info(f"Replacing existing rows for {selected_cell_label}.")

            # Exclude row that has the same selected_cell_label
            df_without_cell = df_csv.loc[
                ~df_csv["cell_index"].astype(str).eq(selected_cell_label)]

            df_updated_hyperparam = pd.concat(
                [df_without_cell,
                 df_best_param_current_cell],
                 axis=0,
                 ignore_index=True)
            action = "replaced"
        elif if_exists == "keep":
            logger.info(f"Keeping existing rows for {selected_cell_label};"
                        + "incoming rows ignored.")
            df_updated_hyperparam = df_csv.copy()
            action = "kept"
        else:
            raise ValueError('if_exists must be "replace" or "keep"')
    else:
        logger.info(f"Creating new row for cell {selected_cell_label}.")
        df_updated_hyperparam = pd.concat(
            [df_csv,
             df_best_param_current_cell],
             axis=0,
             ignore_index=True)
        action = "created"

    logger.info(
        f"Hyperparameters for {selected_cell_label} "
        + f"have been {action} in the CSV file."
    )

    # Export metrics to CSV
    df_updated_hyperparam.to_csv(
        export_csv_filepath,
        index=False)

    return df_updated_hyperparam

def export_current_model_metrics(
    model_name: str,
    selected_cell_label: str,
    df_current_eval_metrics: pd.DataFrame,
    export_csv_filepath: Union[pathlib.PosixPath, str],
    if_exists: Literal["replace", "keep"] = "replace",
    output_log_bool: bool = True,
):
    """
    Export current model evaluation metrics to a CSV file.

    This function checks if the evaluation metrics for a given model and
    cell are already stored in the CSV file. Based on the ``if_exists``
    flag, it either replaces existing rows, keeps them unchanged, or
    creates a new entry. Logging tracks the status of the export process.

    Args:
        model_name (str): Name of the machine learning model.
        selected_cell_label (str): Identifier for the evaluated cell.
        df_current_eval_metrics (pd.DataFrame): DataFrame containing
            current evaluation metrics to be exported.
        export_csv_filepath (Union[pathlib.PosixPath, str]): Path to the
            CSV file where evaluation metrics are stored.
        if_exists (Literal["replace", "keep"], optional): Action to take
            if the model-cell combination already exists in the CSV.
            Defaults to "replace".
        output_log_bool (bool, optional): If True, enables logging output.
            Defaults to True.

    Returns:
        pd.DataFrame: Updated DataFrame containing evaluation metrics from
        both existing and current evaluations.

    Raises:
        ValueError: If ``if_exists`` is not "replace" or "keep".

    Example:
        .. code-block::

            # Export current metrics to CSV
            hyperparam_eval_filepath =  Path.cwd().joinpath(
                "eval_metrics_hp_single_cell_tohoku.csv")

            hp.export_current_model_metrics(
                model_name="iforest",
                selected_cell_label=selected_cell_label,
                df_current_eval_metrics=df_current_eval_metrics,
                export_csv_filepath=hyperparam_eval_filepath,
                if_exists="replace")
    """
    logger = _customize_logger(
        logger_name="export_current_model_metrics",
        output_log=output_log_bool
    )

    if os.path.exists(export_csv_filepath):
        df_csv = pd.read_csv(export_csv_filepath)

        # Check if this model-cell pair already exists
        existed_before = (
            (df_csv["ml_model"].astype(str) == model_name) &
            (df_csv["cell_index"].astype(str) == selected_cell_label)
        ).any()

        logger.info("Have the metrics for "
                    f"{model_name} on cell {selected_cell_label} "
                    "been evaluated before?")
        logger.info(existed_before)
        logger.info("-" * 70)
    else:
        df_csv = None
        existed_before = False

    # Decide action based on if_exists
    if existed_before:
        if if_exists == "replace":
            logger.info(f"Replacing existing row for {model_name}, "
                        f"{selected_cell_label}.")

            # Exclude row that has the same model_name and selected_cell_label
            df_without_entry = df_csv.loc[
                ~(
                    (df_csv["ml_model"].astype(str) == model_name) &
                    (df_csv["cell_index"].astype(str) == selected_cell_label)
                )
            ]

            df_updated_metrics = pd.concat(
                [df_without_entry, df_current_eval_metrics],
                axis=0,
                ignore_index=True
            )
            action = "replaced"

        elif if_exists == "keep":
            logger.info(f"Keeping existing row for {model_name}, "
                        f"{selected_cell_label}; incoming metrics ignored.")
            df_updated_metrics = df_csv.copy()
            action = "kept"

        else:
            raise ValueError('if_exists must be "replace" or "keep"')
    else:
        logger.info(f"Creating new row for {model_name}, "
                    f"{selected_cell_label}.")
        df_updated_metrics = pd.concat(
            [df_csv, df_current_eval_metrics],
            axis=0,
            ignore_index=True
        )
        action = "created"

    logger.info(f"Metrics for {model_name}, {selected_cell_label} have been "
                f"{action} in the CSV file.")

    df_updated_metrics.to_csv(export_csv_filepath, index=False)

    return df_updated_metrics