import numpy as np
from scipy.spatial import distance

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from matplotlib import cm
from matplotlib import rcParams

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
        features: np.ndarray) -> np.ndarray:
    
    # calculating the centroid uisng mean or median
    #centroid = np.mean(features, axis=0) 
    centroid = np.median(features, axis=0)

    metric = distance_metrics[metric_name]
    euclidean_dist = [metric(point, centroid) for point in features]

    return np.array(euclidean_dist)

def calculate_threshold(distance: np.ndarray) -> float:

    mean_dist = np.mean(distance)
    std_dist = np.std(distance)

    threshold = mean_dist + 2 * std_dist

    return threshold

def predict_outliers(distance: np.ndarray,
                     threshold: float,
                     features: np.ndarray) -> tuple:

    pred_outlier_indices = np.where(distance > threshold)[0]

    outlier_features = features[pred_outlier_indices]
    outlier_distance = distance[pred_outlier_indices]

    return pred_outlier_indices, outlier_distance, outlier_features

def plot_hist_distance(distance: np.ndarray,
                       threshold: float) -> Figure:

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(distance, color="b",
        edgecolor="black",
        bins=100)

    ax.grid(
        color="grey",
        linestyle="-",
        linewidth=0.25,
        alpha=0.7)

    ax.axvline(threshold, linestyle="--", color='r')

    ax.set_xlabel("Euclidean Distance from Centroid", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

    return fig

def plot_distance_score_map(
        meshgrid_distance: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray,
        features: np.ndarray,
        threshold: np.ndarray,
        outlier_features: np.ndarray) -> Figure: 
    
    centroid_median = np.median(features, axis=0)

    zz_grid_euclidean_dist = meshgrid_distance.reshape(xx.shape)

    mpl.rcParams.update(mpl.rcParamsDefault)
    rcParams["text.usetex"] = True

    selected_colormap = cm.RdBu_r

    fig, ax = plt.subplots(figsize=(8,5))

    # The contour plot using the model on the grid
    ax.contourf(
        xx,
        yy,
        zz_grid_euclidean_dist,
        cmap=selected_colormap,
        vmin=0,
        vmax=1)

    ax.contour(
        xx,
        yy,
        zz_grid_euclidean_dist,
        levels=[threshold],
        linewidths=2,
        linestyles="dashed",
        colors='black')

    # Set the limits for the colorbar
    cbar_limit = plt.cm.ScalarMappable(cmap=selected_colormap)
    cbar_limit.set_array(zz_grid_euclidean_dist)
    #cbar_limit.set_clim(0., 1.)

    cbar = plt.colorbar(cbar_limit, ax = ax, shrink=0.9)
    cbar.ax.set_ylabel(
        'Outlier Distance',
        fontsize=14)

    ax.scatter(features[:,0], 
                features[:,1], 
                alpha=0.5, 
                label='Inliers')

    ax.scatter(centroid_median[0], 
                centroid_median[1], 
                marker='x', 
                s=50, color='r', 
                label='Centroid')

    ax.scatter(outlier_features[:,0], 
                outlier_features[:,1], 
                color='r', s=20, 
                label='Outliers')


    # Add legend and title
    # plt.legend(handles=[scatter_data, scatter_centroid, scatter_outliers, contour_proxy],
    #            labels=['Data Points', 'Centroid', 'Outliers', 'Manhattan Threshold Boundary'])
    ax.set_title('Outlier Detection using Euclidean Distance')
    ax.set_xlabel('dQ feature')
    ax.set_ylabel('dV feature')

    return fig
