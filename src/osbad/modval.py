"""
Evaluation utilities for benchmarking anomaly detection models.

This module provides functions to compare predicted anomalous cycles
against ground-truth labels from a benchmarking dataset. It includes
utilities for aligning predictions with labels, visualizing results via
confusion matrices, and summarizing performance metrics.

Key features:
    - ``evaluate_pred_outliers``: Aligns predicted outlier indices with the
      benchmarking dataset and produces a DataFrame containing cycle-wise
      true and predicted outlier labels.
    - ``generate_confusion_matrix``: Generates a customized confusion
      matrix heatmap, highlighting correct predictions in palegreen and
      misclassifications in salmon.
    - ``eval_model_performance``: Computes and prints standard evaluation
      metrics (accuracy, precision, recall, F1-score, Matthews correlation
      coefficient) and returns them in a single-row DataFrame.

    .. code-block:: python

        import osbad.modval as modval
"""
# Standard library
from typing import Union

# Third-party libraries
import fireducks.pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.stats import t
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

rcParams["text.usetex"] = True


def evaluate_pred_outliers(
    df_benchmark: pd.DataFrame,
    outlier_cycle_index: np.ndarray) -> pd.DataFrame:
    """
    Evaluate the predicted outliers against the true outliers for each
    cycle in a new dataframe.

    Args:
        df_benchmark (pd.DataFrame): Benchmarking dataset of the selected
            cell.
        outlier_cycle_index (np.ndarray): Predicted outliers from
            statistical methods or ML models.

    Returns:
        pd.DataFrame: A dataframe with predicted outliers and true
        outliers from the benchmarking dataset for each cycle.

    Example:
        .. code-block::

            df_eval_outlier_sd_dV = modval.evaluate_pred_outliers(
                df_benchmark=df_selected_cell,
                outlier_cycle_index=std_outlier_dV_index)
    """
    # create a copy without the true labels
    df_pred_outliers = df_benchmark[
        ["cycle_index",
        "cell_index",
        "voltage",
        "discharge_capacity"]].copy()

    # Define the default label as zero
    df_pred_outliers["outlier_pred"] = 0

    # Conditional update the label based on predicted outlier_cycle_index
    outlier_conditions = [
        df_pred_outliers["cycle_index"].isin(outlier_cycle_index)]
    outlier_values = [1]
    df_pred_outliers["outlier_pred"] = np.select(
        outlier_conditions,
        outlier_values)

    unique_cycle_count = df_benchmark["cycle_index"].unique()

    true_outlier_cycle_label = []
    pred_outlier_cycle_label = []

    for cycle_count in unique_cycle_count:

        # Extract the outlier label per cycle from the
        # benchmarking dataset for true outliers
        df_cycle_true = df_benchmark[
            df_benchmark["cycle_index"] == cycle_count]
        if df_cycle_true["outlier"].unique() == 1:
            true_cycle_label = 1
            true_outlier_cycle_label.append(true_cycle_label)
        else:
            true_cycle_label = 0
            true_outlier_cycle_label.append(true_cycle_label)

        # Extract outlier label per cycle
        # For predicted outliers
        df_cycle_pred = df_pred_outliers[
            df_pred_outliers["cycle_index"] == cycle_count]
        if df_cycle_pred["outlier_pred"].unique() == 1:
            pred_cycle_label = 1
            pred_outlier_cycle_label.append(pred_cycle_label)
        else:
            pred_cycle_label = 0
            pred_outlier_cycle_label.append(pred_cycle_label)

    outlier_eval_dict = {
        "cycle_index": unique_cycle_count,
        "true_outlier": true_outlier_cycle_label,
        "pred_outlier": pred_outlier_cycle_label
    }

    df_eval_outlier = pd.DataFrame.from_dict(outlier_eval_dict)

    return df_eval_outlier


def generate_confusion_matrix(
    y_true: Union[pd.Series,np.ndarray],
    y_pred: Union[pd.Series,np.ndarray]) -> matplotlib.axes._axes.Axes:
    """
    Generate a custom confusion matrix for true and false predictions,
    where the color palegreen indicates true predictions, whereas the
    color salmon denotes false predictions.

    Args:
        y_true (pd.Series | np.ndarray): True outliers from the
            benchmarking dataset.
        y_pred (pd.Series | np.ndarray): Predicted outliers from the
            statistical methods or ML models.

    Returns:
        matplotlib.axes._axes.Axes: Matplotlib axes for additional
        external customization.

    Example:
        .. code-block::

            df_eval_outlier_sd_dV = modval.evaluate_pred_outliers(
                df_benchmark=df_selected_cell,
                outlier_cycle_index=std_outlier_dV_index)

            axplot = modval.generate_confusion_matrix(
                y_true=np.array(df_eval_outlier_sd_dV["true_outlier"]),
                y_pred=np.array(df_eval_outlier_sd_dV["pred_outlier"]))

            fig_title=(r"SD on $\\Delta V_\\textrm{scaled,max,cyc}$\\newline")
            axplot.set_title(fig_title + "\\n", fontsize=16)

            plt.show()
    """

    # Ref:
    # https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
    # https://stackoverflow.com/questions/73709628/labelling-both-percentages-and-absolute-values-on-the-cells-in-seaborn-heatmap-c


    import seaborn as sns

    conf_matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.set(font_scale=1)

    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
    group_pct = [str(np.round(value,2)) + "\\%" for
        value in (conf_matrix.flatten()/np.sum(conf_matrix))*100]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(
        group_names,
        group_counts,
        group_pct)]

    labels = np.asarray(labels, dtype=object).reshape(2,2)

    ax = sns.heatmap(
        np.eye(2),
        annot=labels,
        fmt='',
        cbar=False,
        annot_kws={"fontsize":18},
        cmap=sns.color_palette(["salmon", "palegreen"], as_cmap=True))

    ax.set_xlabel(r"Predicted anomalous cycle", labelpad=20, fontsize=16)
    ax.set_ylabel(r"True anomalous cycle", labelpad=20, fontsize=16)

    return ax

def eval_model_performance(
    model_name,
    selected_cell_label: str,
    df_eval_outliers: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate and summarize model performance metrics.

    This function computes model performance metrics (accuracy,
    precision, recall, F1-score, and Matthews correlation coefficient)
    using ground-truth and predicted outlier labels. It prints each
    metric to the console and returns the results as a one-row DataFrame
    for the specified model and cell.

    Args:
        model_name (str): Name of the machine learning model being
            evaluated.
        selected_cell_label (str): Identifier for the evaluated cell.
        df_eval_outliers (pd.DataFrame): DataFrame containing two columns:
            - ``true_outlier``: Ground-truth outlier labels.
            - ``pred_outlier``: Predicted outlier labels.

    Returns:
        pd.DataFrame: Single-row DataFrame with the evaluation metrics and
        metadata including ``ml_model`` and ``cell_index``.

    Example:
        .. code-block::

            df_current_eval_metrics = modval.eval_model_performance(
                model_name="iforest",
                selected_cell_label=selected_cell_label,
                df_eval_outliers=df_eval_outlier)

    .. note::

        - Both ``true_outlier`` and ``pred_outlier`` must be binary
          labels where ``0`` = inlier and ``1`` = outlier.
    """

    y_true=df_eval_outliers["true_outlier"]
    y_pred=df_eval_outliers["pred_outlier"]

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")

    precision = precision_score(y_true, y_pred)
    print(f"Precision: {precision}")

    recall = recall_score(y_true, y_pred)
    print(f"Recall: {recall}")

    f1score = f1_score(y_true, y_pred)
    print(f"F1-score: {f1score}")

    mcc_score = matthews_corrcoef(y_true, y_pred)
    print(f"MCC-score: {mcc_score}")
    print("*"*100)

    eval_dict = {
        "ml_model": model_name,
        "cell_index": selected_cell_label,
        "accuracy": [accuracy],
        "precision": [precision],
        "recall": [recall],
        "f1_score": [f1score],
        "mcc_score": [mcc_score]
    }

    df_current_eval_metrics = pd.DataFrame.from_dict(eval_dict)

    return df_current_eval_metrics


