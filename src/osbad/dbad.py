"""
This module provides tools for Distance-Based Anomaly Detection (DBAD)
in multivariate datasets. It includes utilities for computing distances
using various metrics, identifying outliers using robust statistical methods,
and visualizing the results through histograms and contour maps.

Key Features:
    - **Distance Computation**: Supports Euclidean, Manhattan, Minkowski, and
      Mahalanobis metrics for calculating distances between data points and a
      centroid. Also supports calculation normalized distances.
    - **Outlier Detection**: Implements Median Absolute Deviation (MAD)-based
      outlier detection for non-gaussian distributions of distances.
    - **Visualization**:
        - Histogram of distances with threshold indication.
        - 2D contour map showing distance scores, decision boundaries, and
          annotated outliers.

Intended Use:
    This module is designed for exploratory data analysis, benchmarking
    anomaly detection in battery cycle data, and other applications where data
    is unimodal and identifying deviations from a central tendency is critical.

Example Workflow:
    1. Compute distances using ``calculate_distance``.
    2. Detect outliers with ``predict_outliers``.
    3. Visualize results using ``plot_hist_distance`` and
       ``plot_distance_score_map``.

Note:
    - Ensure that input data is properly scaled and shaped before using the
      functions.
    - Visualization for 2D contour map showing distance scores, assumes 2D
      feature space for plotting purposes.
"""


# Libraries for plotting
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import rcParams

# libraries for distance based anomaly detection
from scipy.spatial import distance
from osbad.stats import (_compute_mad_outliers, _compute_sd_outliers,
_compute_modified_z_outliers)

rcParams["text.usetex"] = True

from typing import Any, Callable, Dict, List, Literal, Tuple, Union, Optional

distance_metrics: Dict[str, Callable[..., float]] = {
    "euclidean": distance.euclidean,
    "manhattan": distance.cityblock,
    "minkowski": distance.minkowski,
    "mahalanobis": distance.mahalanobis,
}

def calculate_distance(
        metric_name: Literal["euclidean",
                             "manhattan",
                             "minkowski",
                             "mahalanobis"],
        features: np.ndarray,
        centroid: np.ndarray,
        p: int=None,
        inv_cov_matrix: np.ndarray=None,
        max_distance: float=None,
        norm: bool=True,
        ) -> np.ndarray:

    """
    Calculates the distance between each point in a feature set and a given
    centroid using the specified distance metric.

    Args:
        metric_name (Literal): The name of the distance metric to use.
        Options include:
            * ``euclidean``: Euclidean distance
            * ``manhattan``: Manhattan (L1) distance
            * ``minkowski``: Minkowski distance (requires ``p``)
            * ``mahalanobis``: Mahalanobis distance
              (requires ``inv_cov_matrix``)

        features (np.ndarray): A 2D array of shape (n_samples, n_features)
            representing the dataset.

        centroid (np.ndarray): A 1D array of shape (n_features,) representing
            the centroid point of the data distribution.

        p (int, optional): The order parameter for Minkowski distance. Required
            if `metric_name` is ``minkowski``.

        inv_cov_matrix (np.ndarray, optional): The inverse of the covariance
            matrix. Required if `metric_name` is ``mahalanobis``.

        max_distance (float, optional): If provided, distances are normalized
            by dividing by the value of ``max_distance``. Else, the maximum
            value of calculated distance is used.

        norm (bool): If True, distance is normalized.

    Returns:
        np.ndarray: A 1D array of distances between each feature vector and the
        centroid.   

    Raises:
        ValueError: If required parameters (``p`` for Minkowski or
        ``inv_cov_matrix`` for Mahalanobis) are not provided.
    """
    metric = distance_metrics[metric_name]
    if metric_name == "minkowski":
        distance = [metric(point, centroid, p)
                    for point in features]
    elif metric_name == "mahalanobis":
        distance = [metric(point, centroid, inv_cov_matrix)
                    for point in features]

    else:
        distance = [metric(point, centroid)
                    for point in features]

    if norm:
        if max_distance:
            distance = np.array(distance)/max_distance
            return distance
        else:
            max_distance = max(distance)
            distance = np.array(distance)/max_distance
            return distance, max_distance
    else:
        return distance


def predict_outliers(distance: np.ndarray,
                     features: np.ndarray,
                     mad_threshold: float=3,
                     ) -> tuple:

    """
    Detects outliers in a dataset using the Median Absolute Deviation (MAD)
    method.

    This function identifies data points whose distance from a reference
    (e.g., centroid) exceeds a threshold based on the MAD, which is a robust
    measure of statistical dispersion. It is particularly effective for
    detecting anomalies in skewed or non-Gaussian distributions.

    Args:
        distance (np.ndarray): A 1D array of distance values for each data
            point, computed using ``calculate_distance`` function based on
            selected distance metric.

        features (np.ndarray): A 2D array of shape (n_samples, n_features)
            representing the original feature vectors corresponding to each
            distance value.

        mad_threshold (float): The threshold multiplier for MAD-based outlier
            detection. Default is 3. A higher value results in fewer points
            being classified as outliers.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, float]: A tuple containing:
            - mad_outlier_indices (np.ndarray): Indices of the detected
              outliers.
            - outlier_distance (np.ndarray): Distance values of the outliers.
            - outlier_features (np.ndarray): Feature vectors corresponding to
              the outliers.
            - mad_max_limit (float): The upper limit used for outlier detection
              based on MAD.

    .. note::

        This function relies on ``_compute_mad_outliers`` from `stats` module,
        which calculates the MAD factor if not provided and applies the
        threshold to identify anomalous points.
    """


    #ourlier_dict = {}

    # (SD_outlier_indices,
    #  SD_min_limit,
    #  SD_max_limit) = _compute_sd_outliers(distance,
    #                                     std_dev_threshold=3,
    #                                     )

    (mad_outlier_indices,
     mad_min_limit,
     mad_max_limit) = _compute_mad_outliers(distance,
                                    mad_threshold=mad_threshold,
                                    mad_factor=None,
                                    )
    # (mzs_outlier_indices,
    #  mzs_min_limit,
    #  mzs_max_limit) = _compute_modified_z_outliers(distance,
    #                                     mod_zscore_threshold=3.5)


    # ourlier_dict["SD"] = (SD_outlier_indices,
    #                       SD_min_limit,
    #                       SD_max_limit)

    # ourlier_dict["MAD"] = (mad_outlier_indices,
    #                       mad_min_limit,
    #                       mad_max_limit)

    # ourlier_dict["MZS"] = (mzs_outlier_indices,
    #                       mzs_min_limit,
    #                       mzs_max_limit)

    outlier_features = features[mad_outlier_indices]
    outlier_distance = distance[mad_outlier_indices]

    # return ourlier_dict

    return (mad_outlier_indices,
            outlier_distance,
            outlier_features,
            mad_max_limit)

def plot_hist_distance(distance: np.ndarray,
                       #outlier_indices: np.ndarray,
                       threshold: float,
                       ) -> mpl.axes._axes.Axes:

    """
    Plots a histogram of distances from a centroid and highlights the outlier
    threshold. The histogram displays the distribution of distances and a
    vertical dashed red line marks the outlier threshold.

    Args:
        distance (np.ndarray): A 1D array of distance values for each data
            point.

        outlier_indices (np.ndarray): Indices of the data points identified as
            outliers.

        threshold (float): The distance threshold used to classify outliers
            (e.g., based on MAD).

    Returns:
        mpl.axes._axes.Axes: The matplotlib Axes object containing the
        histogram plot.
    """

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(distance, color="b",
        edgecolor="black",
        bins=200)

    ax.grid(
        color="grey",
        linestyle="-",
        linewidth=0.25,
        alpha=0.7)

    ax.axvline(threshold, linestyle="--",
               color='r', label='MAD Threshold')

    # # Add an arrow and text annotations
    # for i, cycle in enumerate(outlier_indices):
    #     ax.annotate(text=f"N{cycle}",
    #                 xy=(distance[cycle], 5),
    #                 xytext=(distance[cycle], 20),
    #                 fontsize=10,
    #                 arrowprops=dict(facecolor='black',
    #                                 shrink=0.05,
    #                                 width=0.5,
    #                                 headwidth=5))

    ax.set_xlabel("Distance from Centroid", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend()

    return ax

def plot_distance_score_map(
        meshgrid_distance: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray,
        features: np.ndarray,
        xoutliers: pd.Series,
        youtliers: pd.Series,
        centroid: np.ndarray,
        threshold: np.ndarray,
        pred_outlier_indices: np.ndarray,
        norm: bool=True,
        ) -> mpl.axes._axes.Axes:


    """
    Plots a 2D contour map of distance scores over a meshgrid with a decision
    boundary and highlights inliers, outliers, and the centroid.

    This function visualizes the distance landscape using a filled contour plot
    across a 2D mesh grid. It creates a contour plot showing anomaly distances,
    a dashed decision boundary at the specified threshold, and highlights the
    predicted anomalous cycles.

    Args:
        meshgrid_distance (np.ndarray): Flattened array of distance scores
            computed over a meshgrid.

        xx (np.ndarray): Meshgrid array for the x-axis.

        yy (np.ndarray): Meshgrid array for the y-axis.

        features (np.ndarray): 2D array of shape (n_samples, 2) representing
            the feature vectors.

        xoutliers (pd.Series): Series containing x-coordinates of detected
            outliers.

        youtliers (pd.Series): Series containing y-coordinates of detected
            outliers.

        centroid (np.ndarray): 1D array of shape (2,) representing the centroid
            coordinates.

        threshold (np.ndarray): Distance threshold used to identify outliers.

        pred_outlier_indices (np.ndarray): Indices of predicted outliers to be
            annotated on the plot.

        norm (bool): If normalized distances are used for contour plot set
        to `True`. Else `False`.

    Returns:
        mpl.axes._axes.Axes: The matplotlib Axes object containing the contour
        and scatter plot.

    .. Note::
        This function assumes the dataset contains exactly two features, which
        are used for 2D plotting.

    .. code-block::

        xx, yy, meshgrid = runner.create_2d_mesh_grid()

        grid_euclidean_dist = dbad.calculate_distance(
                                    metric_name="euclidean",
                                    features=meshgrid,
                                    centroid=centroid
                                    )

        axplot = dbad.plot_distance_score_map(
            meshgrid_distance = grid_euclidean_dist,
            xx = xx,
            yy = yy,
            features=features,
            xoutliers= df_outliers["feature1"],
            youtliers= df_outliers["feature2"],
            centroid=centroid,
            threshold= euclidean_threshold,
            pred_outlier_indices= pred_outlier_indices,
            norm=True
            )
    """

    zz_grid_dist = meshgrid_distance.reshape(xx.shape)

    selected_colormap = cm.RdBu_r

    fig, ax = plt.subplots(figsize=(8,5))

    # Reset the sns settings
    mpl.rcParams.update(mpl.rcParamsDefault)
    rcParams["text.usetex"] = True

    # The contour plot using the model on the grid
    ax.contourf(
        xx,
        yy,
        zz_grid_dist,
        cmap=selected_colormap,
        levels=30,
        vmin=0,
        vmax=1
        )

    ax.contour(
        xx,
        yy,
        zz_grid_dist,
        levels=[threshold],
        linewidths=2,
        linestyles="dashed",
        colors='black',
        )

    # Set the limits for the colorbar
    cbar_limit = plt.cm.ScalarMappable(cmap=selected_colormap)
    cbar_limit.set_array(zz_grid_dist)
    cbar_limit.set_clim(0., 1)

    cbar = plt.colorbar(cbar_limit, ax = ax, shrink=0.9)
    if norm:
        cbar.ax.set_ylabel(
            'Normalized distance from centroid',
            fontsize=14)
    else:
        cbar.ax.set_ylabel(
            'Distance from centroid',
            fontsize=14)

    ax.scatter(features[:,0],
                features[:,1],
                s=10,
                alpha=1,
                marker='o',
                c='black',
                label='Inliers')

    ax.scatter(centroid[0],
                centroid[1],
                marker='x',
                s=100,
                alpha=1,
                color='r',
                label='Centroid')

    ax.scatter(xoutliers,
                youtliers,
                color='gold',
                edgecolors='black',
                s=150,
                alpha=1,
                zorder=2,
                marker='*',
                label='Outliers')

    # Text beside each flagged cycle to label the
    # anomalous cycle
    if len(pred_outlier_indices) != 0:
        for cycle in pred_outlier_indices:
            dQ_text_position = xoutliers.loc[cycle]
            dV_text_position = youtliers.loc[cycle]

            # print(f"Anomalous cycle: {cycle}")
            # print(f"dQ text position: {dQ_text_position}")
            # print(f"dV text position: {dV_text_position}")

            ax.text(
                # x-position of the text
                # Add an offset of 0.1 so that the text
                # does not overlap with the outlier symbol
                x = dQ_text_position + 0.1,
                # y-position of the text
                y = dV_text_position,
                # text-string is the cycle number
                s = cycle,
                horizontalalignment='left',
                size=12,
                color='black',
                weight='bold',
                )
                # print("*"*70)

        # Textbox for the legend to label anomalous cycles ---------------
        # properties for bbox
        props = dict(
            boxstyle='round',
            facecolor='white',
            alpha=0.8)

        # Create textbox to annotate anomalous cycle
        textstr = '\n'.join((
            r"\textbf{Predicted anomalous cycles:}",
            f"{str(pred_outlier_indices)}"))

        # first text value corresponds to the left right
        # alignment starting from left
        # second second value corresponds to up down
        # alignment starting from bottom
        ax.text(
            0.75, 0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            # ha means right alignment of the text
            ha="center", va='top',
            bbox=props)

    ax.set_xlabel(
        r"$\log(\Delta Q_\textrm{scaled,max,cyc)}\;\textrm{[Ah]}$",
        fontsize=12)
    ax.set_ylabel(
        r"$\log(\Delta V_\textrm{scaled,max,cyc})\;\textrm{[V]}$",
        fontsize=12)

    return ax
