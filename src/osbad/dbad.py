import numpy as np
import pandas as pd
from scipy.spatial import distance

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from matplotlib import cm
from matplotlib import rcParams

from osbad.stats import _compute_mad_outliers

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
        centroid: np.ndarray) -> np.ndarray:
    
    metric = distance_metrics[metric_name]
    distance = [metric(point, centroid) for point in features]

    return np.array(distance)

# def calculate_threshold(distance: np.ndarray) -> float:

#     median_dist = np.median(distance)
#     std_dist = np.std(distance)

#     threshold = mean_dist + 2 * std_dist

#     return threshold

def predict_outliers(distance: np.ndarray,
                     features: np.ndarray) -> tuple:
    
    (pred_outlier_indices,
     mad_min_limit,
     mad_max_limit) = _compute_mad_outliers(distance, 
                                            mad_threshold=3,
                                            mad_factor=None)

    outlier_features = features[pred_outlier_indices]
    outlier_distance = distance[pred_outlier_indices]

    return (pred_outlier_indices, 
            outlier_distance,
            outlier_features, 
            mad_max_limit)

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
        xoutliers: pd.Series,
        youtliers: pd.Series,
        centroid: np.ndarray,
        threshold: np.ndarray,
        pred_outlier_indices: np.ndarray,
        ) -> Figure: 
    
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
        #vmin=0,
        #vmax=1
        )

    ax.contour(
        xx,
        yy,
        zz_grid_dist,
        levels=[threshold],
        linewidths=2,
        linestyles="dashed",
        colors='black')

    # Set the limits for the colorbar
    cbar_limit = plt.cm.ScalarMappable(cmap=selected_colormap)
    cbar_limit.set_array(zz_grid_dist)
    #cbar_limit.set_clim(0., 1.)

    cbar = plt.colorbar(cbar_limit, ax = ax, shrink=0.9)
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
                edgecolors='red', 
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
                size='medium',
                color='black',
                weight='bold')
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

    # Add legend and title
    # plt.legend(handles=[scatter_data, scatter_centroid, scatter_outliers, contour_proxy],
    #            labels=['Data Points', 'Centroid', 'Outliers', 'Manhattan Threshold Boundary'])
    ax.set_title('Outlier Detection using Euclidean Distance')
    ax.set_xlabel(
        r"$\log(\Delta Q_\textrm{scaled,max,cyc)}\;\textrm{[Ah]}$",
        fontsize=12)
    ax.set_ylabel(
        r"$\log(\Delta V_\textrm{scaled,max,cyc})\;\textrm{[V]}$",
        fontsize=12)

    return fig
