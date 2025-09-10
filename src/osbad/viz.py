"""
The methods outlined in this module visualize cycle data with and without
anomalies.

.. code-block::

    import osbad.viz as bviz
"""
# Third-party libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from scipy import stats
from scipy.stats import norm, probplot

rcParams["text.usetex"] = True

# Custom osbad library for anomaly detection
import osbad.config as bconf
from osbad.scaler import CycleScaling


# _color_map = matplotlib.colormaps.get_cmap("RdYlBu_r")
_color_map = mpl.colormaps.get_cmap("Spectral_r")

def plot_cycle_data(
    xseries: pd.Series,
    yseries: pd.Series,
    cycle_index_series: pd.Series,
    xoutlier: pd.Series=None,
    youtlier:pd. Series=None) -> mpl.axes._axes.Axes:
    """
    Create scatter plot for the cycling data including colormap, colorbar and
    the option to plot outliers.

    Args:
        xseries (pd.Series): Data for x-axis (e.g. capacity data);
        yseries (pd.Series): Data for y-axis (e.g. voltage data);
        cycle_index_series (pd.Series): Data for cycle count;
        xoutlier (pd.Series, optional): Anomalous x-data. Defaults to None.
        youtlier (pd.Series, optional): Anomalous y-data. Defaults to None.

    Returns:
        mpl.axes._axes.Axes: Matplotlib axes for additional external
        customization.

    Example::

        # Anomalous cycle has label = 1
        # Normal cycle has label = 0
        # true outliers from benchmarking dataset
        df_true_outlier = df_selected_cell_without_labels[
            df_selected_cell_without_labels.cycle_index.isin(
                true_outlier_cycle_index)]

        # Plot normal cycles with true outliers
        axplot = bviz.plot_cycle_data(
            xseries=df_selected_cell_without_labels["discharge_capacity"],
            yseries=df_selected_cell_without_labels["voltage"],
            cycle_index_series=df_selected_cell_without_labels["cycle_index"],
            xoutlier=df_true_outlier["discharge_capacity"],
            youtlier=df_true_outlier["voltage"])

        axplot.set_xlabel(
            r"Discharge capacity [Ah]",
            fontsize=14)
        axplot.set_ylabel(
            r"Discharge voltage [V]",
            fontsize=14)

        axplot.set_title(
            f"Cell {selected_cell_label}",
            fontsize=16)

        plt.show()
    """
    min_cycle_count = cycle_index_series.min()
    max_cycle_count = cycle_index_series.max()

    # figsize=(width, height)
    fig, ax = plt.subplots(figsize=(10,6))

    # Reset the sns settings
    mpl.rcParams.update(mpl.rcParamsDefault)
    rcParams["text.usetex"] = True

    # scatterplot for all data
    ax.scatter(
        xseries,
        yseries,
        s=10,
        marker="o",
        c=cycle_index_series,
        vmin=min_cycle_count,
        vmax=max_cycle_count,
        cmap=_color_map)

    # scatterplot to highlight outliers
    ax.scatter(
        xoutlier,
        youtlier,
        s=10,
        marker="o",
        c="black")

    # Create the colorbar
    smap = plt.cm.ScalarMappable(
        cmap=_color_map)

    smap.set_clim(
        vmin=min_cycle_count,
        vmax=max_cycle_count)

    cbar = fig.colorbar(
        smap,
        ax=ax)

    cbar.ax.tick_params(labelsize=11)
    cbar.ax.set_ylabel(
        'Number of cycles',
        rotation=90,
        labelpad=15,
        fontdict = {"size":14})

    xlabel = xseries.name
    ylabel = yseries.name

    ax.set_xlabel(
        xlabel,
        fontsize=12)
    ax.set_ylabel(
        ylabel,
        fontsize=12)

    ax.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.7)

    return ax

def hist_boxplot(
        df_var: pd.Series) -> mpl.axes._axes.Axes:
    """
    Create boxplot and histogram of a given feature in the same chart.

    Args:
        df_var (pd.Series): Feature to create the boxplot and histogram.

    Returns:
        mpl.axes._axes.Axes: Matplotlib axes for additional external
        customization.

    Example::

        # Plot the histogram and boxplot of the scaled data
        ax_hist = bviz.hist_boxplot(
            df_var=df_capacity_med_scaled["scaled_discharge_capacity"])

        ax_hist.set_xlabel(
            r"Discharge capacity [Ah]",
            fontsize=12)
        ax_hist.set_ylabel(
            r"Count",
            fontsize=12)

        plt.show()
    """
    f, (ax_box, ax_hist) = plt.subplots(
        2,
        sharex=True,
        gridspec_kw={"height_ratios": (0.50, 0.85)})

    sns.boxplot(
        x=df_var,
        ax=ax_box,
        color="orange")

    sns.histplot(
        data=df_var,
        ax=ax_hist,
        color="orange")

    ax_box.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.7)
    ax_hist.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.7)

    return ax_hist


def scatterhist(
    xseries: pd.Series,
    yseries: pd.Series,
    cycle_index_series: pd.Series,
    selected_cell_label=None) -> mpl.axes._axes.Axes:
    """
    Create scatterplot with histogram to display the distribution for
    x-axis and y-axis.

    Args:
        xseries (pd.Series): Data for x-axis (e.g. capacity data);
        yseries (pd.Series): Data for y-axis (e.g. voltage data);
        cycle_index_series (pd.Series): Data for cycle count;

    Returns:
        mpl.axes._axes.Axes: Matplotlib axes for additional external
        customization.

    Example::

        axplot = bviz.scatterhist(
            xseries=df_selected_cell_without_labels["discharge_capacity"],
            yseries=df_selected_cell_without_labels["voltage"],
            cycle_index_series=df_selected_cell_without_labels["cycle_index"])

        axplot.set_xlabel(
            r"Capacity [Ah]",
            fontsize=12)
        axplot.set_ylabel(
            r"Voltage [V]",
            fontsize=12)

        plt.show()
    """
    min_cycle_count = cycle_index_series.min()
    max_cycle_count = cycle_index_series.max()

    # figsize=(width, height)
    fig = plt.figure(figsize=(7, 7))

    # Reset the sns settings
    mpl.rcParams.update(mpl.rcParamsDefault)
    rcParams["text.usetex"] = True

    gs = fig.add_gridspec(
        2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
        left=0.1, right=0.9, bottom=0.1, top=0.9,
        wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(
        xseries,
        yseries,
        s=10,
        marker="o",
        c=cycle_index_series,
        vmin=min_cycle_count,
        vmax=max_cycle_count,
        cmap=_color_map)

    xlabel = xseries.name
    ylabel = yseries.name

    ax.set_xlabel(
        xlabel,
        fontsize=12)
    ax.set_ylabel(
        ylabel,
        fontsize=12)

    ax.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax_histx = fig.add_subplot(gs[0, 0])
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.hist(xseries, bins=40, color="salmon")
    ax_histx.axis("off")

    if selected_cell_label:
        ax_histx.set_title(f"Cell {selected_cell_label}",
            fontsize=12)

    ax_histy = fig.add_subplot(gs[1, 1])
    ax_histy.hist(
        yseries,
        bins=40,
        orientation='horizontal',
        color="grey")
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.axis("off")


    return ax

def plot_explain_scaling(
    df_scaled_capacity: pd.DataFrame,
    df_scaled_voltage: pd.DataFrame,
    extracted_cell_label: str,
    xoutlier: pd.Series=None,
    youtlier: pd.Series=None):
    """
    Visual explanation of the scaling effects on the selected feature.

    Args:
        df_scaled_capacity (pd.DataFrame): Scaled capacity dataframe.
        df_scaled_voltage (pd.DataFrame): Scaled voltage dataframe.
        extracted_cell_label (str): Cell-ID of the selected experiment.
        xoutlier (pd.Series, optional): A series of anomalous xdata.
                                        Defaults to None.
        youtlier (pd.Series, optional): A series of anomalous ydata.
                                        Defaults to None.
    """
    min_cycle_count = df_scaled_capacity["cycle_index"].min()
    max_cycle_count = df_scaled_capacity["cycle_index"].max()

    # figsize=(width, height)
    fig = plt.figure(figsize=(12,8))

    # Reset the sns settings
    mpl.rcParams.update(mpl.rcParamsDefault)
    rcParams["text.usetex"] = True

    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # plot capacity-voltage curve ----------------------------------
    ax1 = fig.add_subplot(gs[0])
    ax1.tick_params(labelbottom=True, labelleft=True)

    ax1.scatter(
        df_scaled_capacity["discharge_capacity"],
        df_scaled_voltage["voltage"],
        s=10,
        marker="o",
        c=df_scaled_capacity["cycle_index"],
        vmin=min_cycle_count,
        vmax=max_cycle_count,
        cmap=_color_map)

    ax1.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.7)
    ax1.set_xlabel(
        r"Capacity, $Q$ [Ah]",
        fontsize=12)
    ax1.set_ylabel(
        r"Voltage, $V$ [V]",
        fontsize=12)

    # plot scaled capacity-voltage curve ---------------------------
    ax2 = fig.add_subplot(gs[1])
    ax2.tick_params(labelbottom=True, labelleft=True)

    ax2.scatter(
        df_scaled_capacity["scaled_discharge_capacity"],
        df_scaled_voltage["scaled_voltage"],
        s=10,
        marker="o",
        c=df_scaled_capacity["cycle_index"],
        vmin=min_cycle_count,
        vmax=max_cycle_count,
        cmap=_color_map)

    ax2.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.7)
    ax2.set_xlabel(
        r"Scaled capacity, $Q_\textrm{scaled}$ [Ah]",
        fontsize=12)
    ax2.set_ylabel(
        r"Scaled voltage, $V_\textrm{scaled}$ [V]",
        fontsize=12)

    # plot voltage-capacity curve with detected outliers -----------

    ax3 = fig.add_subplot(gs[2])
    ax3.tick_params(labelbottom=True, labelleft=True)

    ax3.scatter(
        df_scaled_capacity["discharge_capacity"],
        df_scaled_voltage["voltage"],
        s=10,
        linestyle='dashed',
        marker="o",
        linewidth=1,
        c=df_scaled_capacity["cycle_index"],
        vmin=min_cycle_count,
        vmax=max_cycle_count,
        cmap=_color_map)

    ax3.scatter(
        xoutlier,
        youtlier,
        s=10,
        linestyle='dashed',
        marker="o",
        linewidth=1,
        c="black")

    ax3.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.7)
    ax3.set_xlabel(
        r"Capacity, $Q$ [Ah]",
        fontsize=12)
    ax3.set_ylabel(
        r"Voltage, $V$ [V]",
        fontsize=12)

    # plot median square -------------------------------------------
    ax4 = fig.add_subplot(gs[3])
    ax4.tick_params(labelbottom=True, labelleft=True)

    ax4.scatter(
        df_scaled_capacity["median_square"],
        df_scaled_voltage["median_square"],
        s=10,
        marker="o",
        c=df_scaled_capacity["cycle_index"],
        vmin=min_cycle_count,
        vmax=max_cycle_count,
        cmap=_color_map)

    ax4.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.7)
    ax4.set_xlabel(
        r"Median square capacity, $Q^{2}_\textrm{med}$ [Ah$^{2}$]",
        fontsize=12)
    ax4.set_ylabel(
        r"Median square voltage, $V^{2}_\textrm{med}$ [V$^{2}$]",
        fontsize=12)

    # plot IQR --------------------------------------------------

    ax5 = fig.add_subplot(gs[4])
    ax5.tick_params(labelbottom=True, labelleft=True)

    ax5.scatter(
        df_scaled_capacity["IQR"],
        df_scaled_voltage["IQR"],
        s=10,
        marker="o",
        c=df_scaled_capacity["cycle_index"],
        vmin=min_cycle_count,
        vmax=max_cycle_count,
        cmap=_color_map)

    ax5.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.7)
    ax5.set_xlabel(
        r"IQR capacity, $Q_\textrm{IQR}$ [Ah]",
        fontsize=12)
    ax5.set_ylabel(
        r"IQR voltage, $V_\textrm{IQR}$ [V]",
        fontsize=12)

    # plot median/IQR ratio --------------------------------------
    ax6 = fig.add_subplot(gs[5])
    ax6.tick_params(labelbottom=True, labelleft=True)

    ax6.scatter(
        df_scaled_capacity["median_square_IQR_ratio"],
        df_scaled_voltage["median_square_IQR_ratio"],
        s=10,
        marker="o",
        c=df_scaled_capacity["cycle_index"],
        vmin=min_cycle_count,
        vmax=max_cycle_count,
        cmap=_color_map)

    ax6.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.7)
    ax6.set_xlabel(
        r"Median square capacity-IQR-ratio [Ah]",
        fontsize=12)
    ax6.set_ylabel(
        r"Median square voltage-IQR-ratio [V]",
        fontsize=12)

    # Create the colorbar -------------------------------------------
    # Map the colorbar to chosen colormap
    smap = plt.cm.ScalarMappable(
        cmap=_color_map)

    smap.set_clim(
        vmin=min_cycle_count,
        vmax=max_cycle_count)

    # Create a common standalone colorbar axes for all subplots
    fig.subplots_adjust(right=0.82)
    # dimensions of the colorbar axes (left, bottom, width, height)
    cbar_axes = fig.add_axes([0.85, 0.15, 0.025, 0.7])

    cbar = fig.colorbar(
        smap,
        cax=cbar_axes)

    cbar.ax.tick_params(labelsize=11)
    cbar.ax.set_ylabel(
        'Number of cycles',
        rotation=90,
        labelpad=15,
        fontdict = {"size":14})

    ax2.set_title("Statistical feature transformation for cell "
                  + f"{extracted_cell_label}\n", fontsize=14)

    fig_output_title = ("fig_output/explain_feature_transformation_"
                        + extracted_cell_label
                        + ".png")
    plt.savefig(
        fig_output_title,
        dpi=600,
        bbox_inches="tight")
    plt.show()

def compare_hist_limits(
    df_variable,
    df_norm_variable,
    upper_limit,
    lower_limit):

    fig = plt.figure(figsize=(10,6))

    # Reset the sns settings
    mpl.rcParams.update(mpl.rcParamsDefault)
    rcParams["text.usetex"] = True

    gs = fig.add_gridspec(1, 2, wspace=0.2)

    ax1 = fig.add_subplot(gs[0])
    ax1.hist(
        df_variable,
        bins=50,
        color="salmon",
        label="Data with outliers")
    ax1.axvline(
        x=lower_limit,
        color="black",
        linestyle="--",
        label="IQR lower limit")
    ax1.axvline(
        x=upper_limit,
        color="black",
        linestyle="--",
        label="IQR upper limit")
    ax1.grid(
        color="grey",
        linestyle="-",
        linewidth=0.25,
        alpha=0.7)
    ax1.legend()

    ax2 = fig.add_subplot(gs[1])
    ax2.hist(
        df_norm_variable,
        bins=50,
        color="salmon",
        label="Data without outliers")
    ax2.axvline(
        x=lower_limit,
        color="black",
        linestyle="--",
        label="IQR lower limit")
    ax2.axvline(
        x=upper_limit,
        color="black",
        linestyle="--",
        label="IQR upper limit")
    ax2.grid(
        color="grey",
        linestyle="-",
        linewidth=0.25,
        alpha=0.7)
    ax2.legend()

    return (ax1, ax2)

def plot_quantiles(
    xdata: pd.Series|np.ndarray,
    ax: mpl.axes._axes.Axes,
    fit=False,
    validate=False) -> mpl.axes._axes.Axes:
    """
    Adapt the probplot method from scipy stats to create the probability plot
    of a selected feature so that the feature distribution can be
    compared to the theoretical quantiles of a normal distribution.

    Args:
        xdata (pd.Series | np.ndarray): Selected feature.
        ax (mpl.axes._axes.Axes): Matplotlib axes from a subplot.
        fit (bool, optional): If True, create a straight line fit through the
                              probability plot. Defaults to False.
        validate (bool, optional): If True, compare adapted visualization
                                   method with scipy's implementation.
                                   Defaults to False.

    Returns:
        mpl.axes._axes.Axes: Matplotlib axes for additional external
        customization.

    .. Note::

        The straight dotted line in the probability plot indicates a perfect
        fit to the normal distribution. If most data points fall approximately
        along the straight line, it implies that the feature are consistent
        with the normal distribution. Anomalies would appear as points far
        away from the main cluster and the straight line fit. If points
        deviate significantly in the tails, this suggests heavier tails
        compared to the theoretical normal distribution.

    Example::

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

        ax1 = bp.plot_quantiles(
            xdata=np.array(df_max_dV["max_diff"]),
            ax=ax1,
            fit=True,
            validate=False)

        ax1.set_title("Normality check before removing outliers")

        ax2 = bp.plot_quantiles(
            xdata=np.array(df_max_dV_2nd_iter["max_diff"]),
            ax=ax2,
            fit=True,
            validate=False)

        ax2.set_title("Normality check after removing outliers")

        plt.show()
    """
    # Adapt the probplot method from scipy stats so we can plot the
    # probability plot for different cycles using different color scale
    # to denote the cycles (if needed)
    # Link:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html#scipy.stats.probplot


    def _calc_uniform_order_statistic_medians(n):
        v = np.empty(n, dtype=np.float64)
        v[-1] = 0.5**(1.0 / n)
        v[0] = 1 - v[-1]
        i = np.arange(2, n)
        v[1:-1] = (i - 0.3175) / (n + 0.365)
        return v

    osm_uniform = _calc_uniform_order_statistic_medians(len(xdata))

    osm = norm.ppf(osm_uniform)
    osr = np.sort(xdata)
    slope, intercept, r_value, p_value, std_err = stats.linregress(osm,osr)
    r_value_plot = np.around(r_value,2)

    ax.scatter(
        osm,
        osr,
        s=20,
        marker="o",
        color="orange")


    if fit:
        ax.plot(
            osm,
            slope*osm + intercept,
            color='black',
            linewidth=3,
            linestyle=":")

    # Compare adapted visualization method with scipy's implementation
    if validate:
        probplot(xdata, fit=True, plot=ax)

    ax.set_xlabel(
        r"Theoretical quantiles",
        fontsize=12)
    ax.set_ylabel(
        r"Ordered values",
        fontsize=12)

    ax.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.7)

    # Adapt from:
    # https://matplotlib.org/3.3.4/gallery/recipes/placing_text_boxes.html
    textstr = '\n'.join((
        r"\textbf{R-square:}",
        f"{r_value_plot}"))

    # properties for bbox
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # first 0.10 corresponds to the left-right alignment starting from left
    # second 0.95 corresponds to up-down alignment starting from bottom
    ax.text(
        0.10, 0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        # ha means left alignment of the text
        ha="left", va='top',
        bbox=props)

    return ax

def plot_histogram_with_distribution_fit(
    df_variable: pd.Series|np.ndarray,
    method="norm") -> mpl.axes._axes.Axes:
    """
    Plot the histogram of the selected feature with its distribution fit.

    Args:
        df_variable (pd.Series | np.ndarray): Selected feature.
        method (str, optional): Fit the feature data with either a normal
                                distribution "norm" or a lognormal
                                distribution "lognorm". Defaults to "norm".

    Returns:
        mpl.axes._axes.Axes: Matplotlib axes for additional external
        customization.
    """
    # fit normal distribution
    if method == "norm":
        mean, std = stats.norm.fit(df_variable, loc=0)
        pdf_dist = stats.norm.pdf(df_variable, mean, std)

        fig_label = "Normal distribution fit"

    elif method == "lognorm":
        # fit lognormal distribution
        lognorm_param = stats.lognorm.fit(df_variable)
        pdf_dist = stats.lognorm.pdf(
            df_variable,
            lognorm_param[0],
            lognorm_param[1],
            lognorm_param[2])

        fig_label = "Lognormal distribution fit"

    fig, ax = plt.subplots(figsize=(8, 4))

    # Reset the sns settings
    mpl.rcParams.update(mpl.rcParamsDefault)
    rcParams["text.usetex"] = True

    # bins = auto
    # Minimum bin width between the ‘sturges’ and ‘fd’ estimators.
    # Provides good all-around performance.
    # See Ref [1] and Ref [2]

    # density = True
    # If True, draw and return a probability density
    # each bin will display the bin's raw count divided by
    # the total number of counts and the bin width
    # (density = counts / (sum(counts) * np.diff(bins))),
    # so that the area under the histogram integrates to 1
    # See Ref [1]
    ax.hist(
        df_variable,
        bins='auto',
        density=True,
        color="salmon")

    ax.scatter(
        df_variable,
        pdf_dist,
        c="black",
        label=fig_label)

    ax.legend(
        loc="upper right",
        fontsize=12)

    ax.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.7)

    return ax

def plot_bubble_chart(
    xseries: pd.Series,
    yseries: pd.Series,
    bubble_size: np.ndarray|pd.Series,
    unique_cycle_count: np.ndarray|pd.Series=None,
    cycle_outlier_idx_label: np.ndarray=None) -> mpl.axes._axes.Axes:
    """
    Plot the bubble chart of each feature with scalable bubble size ratio
    depending on the anomaly score.

    Args:
        xseries (pd.Series): Data to be plotted on the x-axis of the bubble
                             chart.
        yseries (pd.Series): Data to be plotted on the y-axis of the bubble
                             chart.
        bubble_size (np.ndarray|pd.Series): Calculated bubble size depending
                                            on the anomaly score.
        unique_cycle_count (np.ndarray|pd.Series, optional): Unique cycle
                                                             count of the
                                                             selected cell.
                                                             Defaults to None.
        cycle_outlier_idx_label (np.ndarray, optional): The index of anomalous
                                                        cycles. Defaults
                                                        to None.

    Returns:
        mpl.axes._axes.Axes: Matplotlib axes for additional external
        customization.

    .. code-block::

        # Plot the bubble chart and label the outliers
        axplot = bviz.plot_bubble_chart(
            xseries=df_features_per_cell["log_max_diff_dQ"],
            yseries=df_features_per_cell["log_max_diff_dV"],
            bubble_size=bubble_size,
            unique_cycle_count=unique_cycle_count,
            cycle_outlier_idx_label=true_outlier_cycle_index)

        axplot.set_title(
            f"Cell {selected_cell_label}", fontsize=13)

        axplot.set_xlabel(
            r"$\\log(\\Delta Q_\\textrm{scaled,max,cyc)}\\;\\textrm{[Ah]}$",
            fontsize=12)
        axplot.set_ylabel(
            r"$\\log(\\Delta V_\\textrm{scaled,max,cyc})\\;\\textrm{[V]}$",
            fontsize=12)

        output_fig_filename = (
            "log_bubble_plot_"
            + selected_cell_label
            + ".png")

        fig_output_path = (
            selected_cell_artifacts_dir.joinpath(output_fig_filename))

        plt.savefig(
            fig_output_path,
            dpi=200,
            bbox_inches="tight")

        plt.show()
    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    rcParams["text.usetex"] = True

    fig, ax = plt.subplots(1,1)

    if isinstance(xseries, np.ndarray):
        xseries = pd.Series(xseries)

    # Define the boundaries of the grid
    # Extend the grid boundaries by -1 and +1 to
    # ensure full coverage
    min_xrange = np.round(xseries.min() - 1)
    max_xrange = np.round(xseries.max() + 1)
    min_yrange = np.round(yseries.min() - 1)
    max_yrange = np.round(yseries.max() + 1)

    # scatterplot for all data
    ax.scatter(
        xseries,
        yseries,
        s=np.abs(bubble_size)*100,
        alpha=0.5,
        marker="o",
        c="salmon")

    if unique_cycle_count is not None:

        # if unique_cycle_count or xseries has the type np.ndarray
        # change into pd.Series so that we can update the
        # index of the series to match the cycle number

        if isinstance(unique_cycle_count, np.ndarray):
            unique_cycle_count = pd.Series(unique_cycle_count)

        # Update the index of the series to match the unique_cycle_count
        # Especially after some anomalous cycles have been removed
        xseries.index = unique_cycle_count
        unique_cycle_count.index = unique_cycle_count

        for cycle in unique_cycle_count:
            if cycle in cycle_outlier_idx_label:
                print(f"Potential anomalous cycle: {cycle}")
                print(f"x-position of the text: {xseries[int(cycle)]}")
                print(f"y-position of the text: {yseries[int(cycle)]}")
                print("-"*70)
                ax.text(
                    # x-position of the text
                    x = xseries[int(cycle)],
                    # y-position of the text
                    y = yseries[int(cycle)],
                    # text-string is the cycle number
                    s = unique_cycle_count[int(cycle)],
                    horizontalalignment='center',
                    size='medium',
                    color='black',
                    weight='bold')

        # properties for bbox
        props = dict(
            boxstyle='round',
            facecolor='wheat',
            alpha=0.5)

        # Create textbox to annotate anomalous cycle
        textstr = '\n'.join((
            r"\textbf{Anomalous cycles:}",
            f"{cycle_outlier_idx_label}"))

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

    ax.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.7)

    # Define square grid with equal distance for x-axis and y-axis
    # so that the visualization of decision boundaries with predicted
    # outliers can be more intuitive
    # and the distance is not distorted due to unequal grid points
    min_ax = np.min([min_xrange, min_yrange])
    max_ax = np.max([max_xrange, max_yrange])

    ax.set_xlim([min_ax, max_ax])
    ax.set_ylim([min_ax, max_ax])

    # (max_ax + stepsize) to include endpoint for np.arange
    stepsize = 1
    xticks = np.arange(min_ax, max_ax+stepsize, stepsize)
    yticks = np.arange(min_ax, max_ax+stepsize, stepsize)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    return ax



def plot_scale_capacity(
    df_selected_cell_without_labels: pd.DataFrame,
    selected_cell_label: str) -> pd.DataFrame:
    """
    Implement median-IQR-scaling to the cell capacity dataset and plot
    the corresponding histogram-boxplot of the scaled feature.

    Args:
        df_selected_cell_without_labels (pd.DataFrame): Selected features
                                                        without
                                                        true labels.

    Returns:
        pd.DataFrame: Scaled cell capacity.
    """
    # Reset the sns settings
    mpl.rcParams.update(mpl.rcParamsDefault)
    rcParams["text.usetex"] = True

    scaler = CycleScaling(df_selected_cell_without_labels)

    df_capacity_med_scaled = scaler.median_IQR_scaling(
        variable="discharge_capacity",
        validate=True)

    ax_hist = hist_boxplot(
        df_var=df_capacity_med_scaled["scaled_discharge_capacity"])

    ax_hist.set_xlabel(
        r"Discharge capacity, $Q_\textrm{dis}$ [Ah]",
        fontsize=12)
    ax_hist.set_ylabel(
        r"Count",
        fontsize=12)

    output_fig_filename = (
        "scaled_capacity_"
        + selected_cell_label
        + ".png")

    selected_cell_artifacts_dir = bconf.artifacts_output_dir(
        selected_cell_label)

    fig_output_path = (
        selected_cell_artifacts_dir.joinpath(output_fig_filename))

    plt.savefig(
        fig_output_path,
        dpi=200,
        bbox_inches="tight")

    if bconf.SHOW_FIG_STATUS:
        plt.show()

    return df_capacity_med_scaled

def plot_scale_voltage(
    df_selected_cell_without_labels: pd.DataFrame,
    selected_cell_label: str) -> pd.DataFrame:
    """
    Implement median-IQR-scaling to the cell voltage dataset and plot
    the corresponding histogram-boxplot of the scaled feature.

    Args:
        df_selected_cell_without_labels (pd.DataFrame): Selected features
                                                        without
                                                        true labels.

    Returns:
        pd.DataFrame: Scaled cell voltage.
    """
    # Reset the sns settings
    mpl.rcParams.update(mpl.rcParamsDefault)
    rcParams["text.usetex"] = True



    df_voltage_med_scaled = scaler.median_IQR_scaling(
        variable="voltage",
        validate=True)

    ax_hist = hist_boxplot(
        df_var=df_voltage_med_scaled["scaled_voltage"])

    ax_hist.set_xlabel(
        r"Scaled voltage, $V_\textrm{dis}$ [V]",
        fontsize=12)
    ax_hist.set_ylabel(
        r"Count",
        fontsize=12)

    output_fig_filename = (
        "scaled_voltage_"
        + selected_cell_label
        + ".png")

    selected_cell_artifacts_dir = bconf.artifacts_output_dir(
        selected_cell_label)

    fig_output_path = (
        selected_cell_artifacts_dir.joinpath(output_fig_filename))

    plt.savefig(
        fig_output_path,
        dpi=200,
        bbox_inches="tight")

    if bconf.SHOW_FIG_STATUS:
        plt.show()

    return df_voltage_med_scaled