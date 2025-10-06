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
from typing import Union

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

    Example:
        .. code-block::

            # Anomalous cycle has label = 1
            # Normal cycle has label = 0
            # true outliers from benchmarking dataset
            df_true_outlier = df_selected_cell_no_label[
                df_selected_cell_no_label.cycle_index.isin(
                    true_outlier_cycle_index)]

            # Plot normal cycles with true outliers
            axplot = bviz.plot_cycle_data(
                xseries=df_selected_cell_no_label["discharge_capacity"],
                yseries=df_selected_cell_no_label["voltage"],
                cycle_index_series=df_selected_cell_no_label["cycle_index"],
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
        s=20,
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

    ax.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.7)

    return ax

def hist_boxplot(
    df_variable: Union[pd.Series, np.ndarray]) -> mpl.axes._axes.Axes:
    """
    Plot a combined boxplot and histogram of a feature.

    This function generates a two-part visualization for a given
    feature: a boxplot on the top (to show distribution, median,
    and potential outliers) and a histogram on the bottom (to show
    frequency distribution). Both plots share the same x-axis for
    easier comparison.

    Args:
        df_variable (Union[pd.Series, np.ndarray]): Input feature
            values as a pandas Series or NumPy array.

    Returns:
        mpl.axes._axes.Axes: Matplotlib histogram axes. This allows
        additional external customization, such as setting labels or
        titles.

    Example:
        .. code-block::

            # Plot the histogram and boxplot of the scaled data
            ax_hist = bviz.hist_boxplot(
                df_var=df_capacity_med_scaled["scaled_discharge_capacity"])

            ax_hist.set_xlabel(
                r"Discharge capacity, $Q_\\textrm{dis}$ [Ah]",
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
        x=df_variable,
        ax=ax_box,
        color="orange")

    sns.histplot(
        data=df_variable,
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
    Plot a scatter plot with marginal histograms for two features.

    This function creates a joint visualization consisting of:
        - A central scatter plot of two features, color-coded by cycle index.
        - A histogram of the x-series above the scatter plot.
        - A histogram of the y-series to the right of the scatter plot.

    The marginal histograms provide additional insight into the
    distributions of each variable, while the scatter plot shows
    their relationship.

    Args:
        xseries (pd.Series): Data for the x-axis.
        yseries (pd.Series): Data for the y-axis.
        cycle_index_series (pd.Series): Series of cycle indices used
            for color mapping in the scatter plot.
        selected_cell_label (str, optional): Label for the cell,
            displayed as the plot title. Defaults to None.

    Returns:
        mpl.axes._axes.Axes: Matplotlib scatter plot axes. Enables
        further customization (e.g., labels, annotations).

    Example:
        .. code-block::

            axplot = bviz.scatterhist(
                xseries=df_selected_cell_no_label["discharge_capacity"],
                yseries=df_selected_cell_no_label["voltage"],
                cycle_index_series=df_selected_cell_no_label["cycle_index"],
                selected_cell_label=selected_cell_label)

            axplot.set_xlabel(
                r"Capacity, $Q_\\textrm{dis}$ [Ah]",
                fontsize=12)
            axplot.set_ylabel(
                r"Voltage, $V_\\textrm{dis}$ [V]",
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
    selected_cell_label: str,
    xoutlier: pd.Series=None,
    youtlier: pd.Series=None):
    """
    Visualize statistical scaling transformations for a cell's cycles.

    This function creates a 2×3 grid of subplots to illustrate how scaling
    and statistical feature transformations are applied to cycling data.
    It plots original and scaled capacity-voltage curves, highlights detected
    outliers, and visualizes derived features such as median-square, IQR,
    and median/IQR ratio. A shared colorbar indicates cycle progression.

    Args:
        df_scaled_capacity (pd.DataFrame): DataFrame containing cycle-based
            capacity features. Must include the following columns:
            ``["discharge_capacity", "scaled_discharge_capacity",
            "median_square", "IQR", "median_square_IQR_ratio",
            "cycle_index"]``.
        df_scaled_voltage (pd.DataFrame): DataFrame containing cycle-based
            voltage features. Must include the following columns:
            ``["voltage", "scaled_voltage", "median_square", "IQR",
            "median_square_IQR_ratio"]``.
        selected_cell_label (str): Identifier of the evaluated cell, used
            for titling the plots and naming the output file.
        xoutlier (pd.Series, optional): X-coordinates of outlier points to
            highlight in the voltage-capacity plot. Defaults to None.
        youtlier (pd.Series, optional): Y-coordinates of outlier points to
            highlight in the voltage-capacity plot. Defaults to None.

    Returns:
        None: The function saves the resulting figure as a PNG file and
        displays it.

    Example:
        .. code-block::

            # Path to DuckDB file
            db_filepath = str(
                Path.cwd()
                .parent
                .joinpath("database","train_dataset_severson.db"))

            selected_cell_label = "2017-05-12_5_4C-70per_3C_CH17"

            # Import the BenchDB class
            # Load only the dataset based on the selected cell
            benchdb = BenchDB(
                db_filepath,
                selected_cell_label)

            # Extract true outliers cycle index from benchmarking dataset
            true_outlier_cycle_index = benchdb.get_true_outlier_cycle_index(
                df_selected_cell)

            # Anomalous cycle has label = 1
            # Normal cycle has label = 0
            # true outliers from benchmarking dataset
            df_true_outlier = df_selected_cell_without_labels[
                df_selected_cell_without_labels.cycle_index.isin(
                    true_outlier_cycle_index)]

            # Instantiate the CycleScaling class
            scaler = CycleScaling(
                df_selected_cell=df_selected_cell_without_labels)

            # Implement median IQR scaling on the discharge capacity data
            df_capacity_med_scaled = scaler.median_IQR_scaling(
                variable="discharge_capacity",
                validate=True)

            # Implement median IQR scaling on the discharge voltage data
            df_voltage_med_scaled = scaler.median_IQR_scaling(
                variable="voltage",
                validate=True)

            bviz.plot_explain_scaling(
                df_scaled_capacity=df_capacity_med_scaled,
                df_scaled_voltage=df_voltage_med_scaled,
                extracted_cell_label=selected_cell_label,
                xoutlier=df_true_outlier["discharge_capacity"],
                youtlier=df_true_outlier["voltage"]
            )

    .. note::

        - The plots include:
            1. Raw capacity-voltage curve.
            2. Scaled capacity-voltage curve.
            3. Capacity-voltage curve with detected outliers.
            4. Median-square transformation.
            5. Interquartile range (IQR).
            6. Median/IQR ratio.
        - All scatter plots are color-coded by ``cycle_index`` using a
          shared colorbar.
        - Figures are saved under ``fig_output/`` with a filename based on
          the cell label.
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
                  + f"{selected_cell_label}\n", fontsize=14)

    output_fig_filename = (
        f"explain_feature_transformation_"
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

    Example:
        .. code-block::

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

            ax1 = bviz.plot_quantiles(
                xdata=df_max_dV["max_diff"],
                ax=ax1,
                fit=True,
                validate=False)

            ax1.set_title(
                "Normality check before removing outliers")

            ax2 = bviz.plot_quantiles(
                xdata=df_max_dV_2nd_iter["max_diff"],
                ax=ax2,
                fit=True,
                validate=False)

            ax2.set_title(
                "Normality check after removing detected outliers")

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
    df_variable: Union[pd.Series, np.ndarray],
    method="norm") -> mpl.axes._axes.Axes:
    """
    Plot a histogram of feature values with fitted distribution overlay.

    This function visualizes the distribution of a selected feature by
    plotting its histogram and overlaying a fitted probability density
    function (PDF). The feature can be fitted with either a normal or
    lognormal distribution, based on the ``method`` argument.

    Args:
        df_variable (pd.Series | np.ndarray): Input feature data.
        method (str, optional): Distribution type for fitting. Must be:

            - ``norm``: Fit using ``scipy.stats.norm.fit`` and overlay a
              normal distribution.
            - ``lognorm``: Fit using ``scipy.stats.lognorm.fit`` and
              overlay a lognormal distribution. Defaults to ``"norm"``.

    Returns:
        mpl.axes._axes.Axes: Matplotlib axes object containing the
        histogram and fitted distribution plot. Can be customized
        further externally.

    Example:
        .. code-block::

            # Plot with normal fit
            axplot = bviz.plot_histogram_with_distribution_fit(
                df_variable=df_max_dV_2nd_iter["max_diff"],
                method="norm")

            axplot.set_xlabel(
                r"$\Delta V_\\textrm{scaled,max,cyc}\\;\\textrm{[V]}$",
                fontsize=12)

            axplot.set_ylabel('Probability', fontsize=12)

            plt.show()

            # Plot with lognormal fit
            axplot = bviz.plot_histogram_with_distribution_fit(
                df_variable=np.array(df_max_dQ_2nd_iter["max_diff"]),
                method="lognorm")

            axplot.set_xlabel(
                r"$\Delta Q_\\textrm{scaled,max,cyc}\\;\\textrm{[V]}$",
                fontsize=12)

            axplot.set_ylabel('Probability', fontsize=12)

            plt.show()

    .. note::

        - Histogram uses ``bins="auto"`` for automatic bin width
          selection, balancing performance across distributions.
        - The histogram is normalized (``density=True``) so that the
          total area integrates to 1, matching the fitted PDF.
        - For ``"norm"`` fitting, see ``scipy.stats.norm.fit``.
        - For ``"lognorm"`` fitting, see ``scipy.stats.lognorm.fit``.
    """
    # Reset the sns settings
    # from confusion matrix
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    rcParams["text.usetex"] = True

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

def calculate_bubble_size_ratio(
    df_variable: pd.Series|np.ndarray) -> pd.Series:
    """
    Calculate the bubble size of the feature in the bubble plot depending
    on the anomaly score by using the feature standardization method.

    Args:
        df_variable (pd.Series | np.ndarray): Selected feature.

    Returns:
        pd.Series: Calculated bubble size of the feature.

    Example:
        .. code-block::

            df_bubble_size_dQ = bviz.calculate_bubble_size_ratio(
                df_variable=df_max_dQ["max_diff_dQ"])

            df_bubble_size_dV = bviz.calculate_bubble_size_ratio(
                df_variable=df_max_dV["max_diff"])
    """

    mean_var = np.mean(df_variable)
    print(f"Feature mean: {mean_var}")

    max_var = np.max(df_variable)
    print(f"Feature max: {max_var}")

    min_var = np.min(df_variable)
    print(f"Feature min: {min_var}")

    std_var = np.std(df_variable, ddof=1)
    print(f"Feature std: {std_var}")
    print("*"*70)

    scaling_ratio = (df_variable - mean_var)/(std_var)

    return scaling_ratio

def plot_bubble_chart(
    xseries: pd.Series,
    yseries: pd.Series,
    bubble_size: np.ndarray|pd.Series,
    unique_cycle_count: np.ndarray|pd.Series=None,
    cycle_outlier_idx_label: np.ndarray=None,
    square_grid:bool =False) -> mpl.axes._axes.Axes:
    """
    Plot the bubble chart of each feature with scalable bubble size ratio
    depending on the anomaly score.

    Args:
        xseries (pd.Series):
            Data to be plotted on the x-axis of the bubble chart.
        yseries (pd.Series):
            Data to be plotted on the y-axis of the bubble chart.
        bubble_size (np.ndarray|pd.Series):
            Calculated bubble size depending on the anomaly score.
        unique_cycle_count (np.ndarray|pd.Series, optional):
            Unique cycle count of the selected cell. Defaults to None.
        cycle_outlier_idx_label (np.ndarray, optional):
            The index of anomalous cycles. Defaults to None.
        square_grid (bool, optional):
            Define square grid with equal distance for x-axis and y-axis.
            Defaults to False.

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
            cycle_outlier_idx_label=true_outlier_cycle_index,
            square_grid=True)

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

        if cycle_outlier_idx_label is not None:
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

    if square_grid:

        # Define the boundaries of the grid
        # Extend the grid boundaries by -1 and +1 to
        # ensure full coverage
        min_xrange = np.round(xseries.min() - 1)
        max_xrange = np.round(xseries.max() + 1)
        min_yrange = np.round(yseries.min() - 1)
        max_yrange = np.round(yseries.max() + 1)

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

def plot_multiple_outlier_cycles(
    df_selected_cell: pd.DataFrame,
    potential_outlier_cycles: list,
    selected_cell_label: str) -> None:
    """
    Plot and annotate multiple potential outlier cycles.

    This function creates a grid of subplots, each highlighting one of
    the potential outlier cycles from the given cell dataset. Cycling
    data for all cycles is shown with a colormap, and the specified
    outlier cycles are annotated with text boxes. A shared colorbar
    indicates the cycle index range. The figure is saved to the cell’s
    artifacts directory and displayed interactively.

    Args:
        df_selected_cell (pd.DataFrame): Cycling dataset for the cell,
            containing ``discharge_capacity``, ``voltage`` and ``cycle_index``.
        potential_outlier_cycles (list): List of cycle indices to be
            highlighted as potential outliers.
        selected_cell_label (str): Identifier of the evaluated cell, used
            in the plot title and output filename.

    Returns:
        None: The function saves and displays the generated plot.

    Example:
        .. code-block::

            # Get the cell-ID from cell_inventory
            selected_cell_label = "2017-05-12_5_4C-70per_3C_CH17"

            bviz.plot_multiple_outliers(
                df_selected_cell,
                potential_outlier_cycles= [0, 40, 147, 148],
                selected_cell_label=selected_cell_label)
    """

    # Reset the sns settings
    # from confusion matrix
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    rcParams["text.usetex"] = True

    total_subplot = len(potential_outlier_cycles)

    if total_subplot > 2:
        total_rows = total_subplot // 3
        total_cols = total_subplot // total_rows
        ylabel_offset = 0.075
    else:
        total_rows = 1
        total_cols = total_subplot
        ylabel_offset = -0.005

    print(f"Total subplot: {total_subplot}")
    print(f"Total rows: {total_rows}")
    print(f"Total columns: {total_cols}")
    print("*"*70)

    # figsize = (width, height)
    fig = plt.figure(figsize=(
        4*total_cols,
        4*total_rows))

    # 1 row, columns depending on number of potential outlier cycles
    gs = fig.add_gridspec(total_rows, total_cols, hspace=0.2, wspace=0.2)
    axs = {}

    # properties for bbox
    props = dict(
        boxstyle='round',
        facecolor='wheat',
        alpha=0.5)

    color_map = mpl.colormaps.get_cmap("Spectral_r")


    for outlier_cycle_idx in range(len(potential_outlier_cycles)):

        print("Plot outlier cycle: "
              + f"{potential_outlier_cycles[outlier_cycle_idx]}")

        # select a specific cell that matches the cell index
        df_check_outlier = df_selected_cell[
            (df_selected_cell["cycle_index"]
            == potential_outlier_cycles[outlier_cycle_idx])]

        potential_outlier_cycle_index = (
            df_check_outlier["cycle_index"].unique())

        # Create a new axis for each subplot loop
        axs[outlier_cycle_idx] = fig.add_subplot(gs[outlier_cycle_idx])

        # Calculate the max cycle
        max_cycle_count = df_selected_cell["cycle_index"].max()

        # scatterplot for all data
        axs[outlier_cycle_idx].scatter(
            df_selected_cell["discharge_capacity"],
            df_selected_cell["voltage"],
            s=10,
            marker="o",
            c=df_selected_cell["cycle_index"],
            vmin=0,
            vmax=max_cycle_count,
            cmap=color_map)

        # scatterplot to highlight outliers
        axs[outlier_cycle_idx].scatter(
            df_check_outlier["discharge_capacity"],
            df_check_outlier["voltage"],
            s=10,
            marker="o",
            c="black")

        # Create a grid for all subplots
        axs[outlier_cycle_idx].grid(
            color="grey",
            linestyle="-",
            linewidth=0.25,
            alpha=0.7)

        # Create textbox to annotate anomalous cycle
        textstr = '\n'.join((
            r"\textbf{Anomalous cycle:}",
            f"{potential_outlier_cycle_index}"))

        # first 0.95 corresponds to the left right alignment
        # starting from left
        # second 0.95 corresponds to up down alignment
        # starting from bottom
        axs[outlier_cycle_idx].text(
            0.95, 0.95,
            textstr,
            transform=axs[outlier_cycle_idx].transAxes,
            fontsize=9,
            # ha means right alignment of the text
            ha="right", va='top',
            bbox=props)

    # Create common x-axis and y-axis
    fig.supxlabel(
        r"Discharge capacity, $Q_\textrm{dis}$ [Ah]",
        y=-0.02,
        fontsize=14)
    fig.supylabel(
        r"Discharge voltage, $V_\textrm{dis}$ [V]",
        x=ylabel_offset,
        fontsize=14)

    # Create a common standalone colorbar axes for all subplots
    fig.subplots_adjust(right=0.82)

    # dimensions of the colorbar axes (left, bottom, width, height)
    cbar_axes = fig.add_axes([0.85, 0.15, 0.025, 0.7])

    smap = plt.cm.ScalarMappable(
        cmap=color_map)

    # vmax is determined by the max cycles from all cells
    smap.set_clim(
        vmin=0,
        vmax=max_cycle_count)

    cbar = fig.colorbar(
        smap,
        cax=cbar_axes)

    cbar.ax.tick_params(labelsize=11)
    cbar.ax.set_ylabel(
        'Max number of cycles',
        rotation=90,
        labelpad=15,
        fontdict = {"size":14})

    output_fig_filename = (
        "evaluate_potential_outliers_"
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

    plt.show()

def plot_single_outlier_cycle(
    df_selected_cell: pd.DataFrame,
    selected_cycle_index: int,
    selected_cell_label: str):
    """
    Plot and annotate a single cycle as a potential outlier.

    This function visualizes the voltage vs. discharge capacity for the
    selected cell and highlights one cycle specified by its cycle index.
    The chosen cycle is annotated as a potential outlier on the plot.
    The figure is saved to the artifacts directory for the selected cell
    and displayed interactively.

    Args:
        df_selected_cell (pd.DataFrame): Cycling dataset for the cell,
            containing ``discharge_capacity``, ``voltage`` and
            ``cycle index``.
        selected_cycle_index (int): Cycle index to highlight as a
            potential outlier.
        selected_cell_label (str): Identifier of the evaluated cell,
            used in the plot title and output filename.

    Returns:
        None: The function saves and displays the generated plot.

    Example:
        .. code-block::

            # Get the cell-ID from cell_inventory
            selected_cell_label = "2017-05-12_5_4C-70per_3C_CH17"

            bviz.plot_single_outlier(
                df_selected_cell,
                selected_cycle_index=147,
                selected_cell_label=selected_cell_label)
    """
    # Reset the sns settings
    # from confusion matrix
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    rcParams["text.usetex"] = True


    # Check if specific cycle are anomalous or not
    df_check_outlier = df_selected_cell[
        (df_selected_cell["cycle_index"] == selected_cycle_index)]

    check_outlier_cycle_index = df_check_outlier["cycle_index"].unique()

    # Plot normal cycles with potential outliers
    axplot = plot_cycle_data(
        xseries=df_selected_cell["discharge_capacity"],
        yseries=df_selected_cell["voltage"],
        cycle_index_series=df_selected_cell["cycle_index"],
        xoutlier=df_check_outlier["discharge_capacity"],
        youtlier=df_check_outlier["voltage"])

    axplot.set_xlabel(
        r"Discharge capacity, $Q_\textrm{dis}$ [Ah]",
        fontsize=12)
    axplot.set_ylabel(
        r"Discharge voltage, $V_\textrm{dis}$ [V]",
        fontsize=12)

    axplot.set_title(f"Cell {selected_cell_label}",
                    fontsize=12)

    # Create textbox to annotate anomalous cycle
    textstr = '\n'.join((
        r"\textbf{Potential anomalous cycle:}",
        f"{check_outlier_cycle_index}"))

    # properties for bbox
    props = dict(
        boxstyle='round',
        facecolor='wheat',
        alpha=0.5)

    # first 0.95 corresponds to the left right alignment starting from left
    # second 0.95 corresponds to up down alignment starting from bottom
    axplot.text(
        0.95, 0.95,
        textstr,
        transform=axplot.transAxes,
        fontsize=12,
        # ha means right alignment of the text
        ha="right", va='top',
        bbox=props)

    output_fig_filename = (
        f"evaluate_potential_outlier_cycle_{selected_cycle_index}_"
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


    plt.show()