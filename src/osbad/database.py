"""
Utilities for loading and visualizing benchmarking data from DuckDB.

This module defines :class:`BenchDB`, a helper for accessing battery cell
datasets stored in DuckDB, removing labels for modeling, extracting ground-
truth outlier indices, and plotting cycling curves. Figures are saved to a
per-cell artifacts directory under ``bconf.PIPELINE_OUTPUT_DIR`` and may be
shown interactively if ``bconf.SHOW_FIG_STATUS`` is enabled.

Key features:
    - ``load_benchmark_dataset``: Load the training or test dataset
      for the selected cell from a DuckDB database (including the label
      column).
    - ``drop_labels``: Remove the ``outlier`` label column and,
      optionally, keep only a specified subset of columns.
    - ``get_true_outlier_cycle_index``: Retrieve the cycle indices
      labeled as anomalous (``outlier == 1``) for the selected cell.
    - ``plot_cycle_data``: Plot discharge voltage vs. capacity
      cycles; optionally highlight and annotate known anomalous cycles.
      Saves the figure to the cell's artifacts directory and can display it.
    - ``load_features_db``: Load precomputed features (train or
      test) for the selected cell from a DuckDB features database.

Example:
    .. code-block::

        from osbad.database import BenchDB
"""
# Standard library
import os
import pathlib
from pathlib import Path
from typing import Union

# Third-party libraries
import duckdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

rcParams["text.usetex"] = True

# Custom osbad library for anomaly detection
import osbad.config as bconf
import osbad.viz as bviz


class BenchDB:
    """Load and analyze benchmarking datasets for a single cell.

    The ``BenchDB`` class provides utilities for accessing and managing
    benchmarking datasets stored in DuckDB. It supports loading training
    and test datasets, extracting features, removing labels, plotting
    cycling data, and retrieving ground-truth outlier indices. Figures
    are saved to a per-cell artifacts directory, with optional display
    enabled via configuration.

    Args:
        input_db_filepath (str): Path to the DuckDB benchmarking database
            file.
        cell_label (str): Label of the cell to be analyzed.
        show_fig_status (bool, optional): Whether to show generated
            figures interactively. Defaults to False when running the
            pipeline in the batch mode for multiple cells.
    """
    def __init__(
        self,
        input_db_filepath: str,
        cell_label: str):

        self._db_filepath = input_db_filepath
        self._selected_cell_label = cell_label
        self._show_fig_status = bconf.SHOW_FIG_STATUS

        # create a new folder for each evaluated cell
        # store all figures output for each evaluated
        # cell into its corresponding folder
        self._selected_cell_artifacts = bconf.PIPELINE_OUTPUT_DIR.joinpath(
            self._selected_cell_label)
        if not os.path.exists(self._selected_cell_artifacts):
            os.mkdir(self._selected_cell_artifacts)

    def load_benchmark_dataset(
        self,
        dataset_type="train") -> pd.DataFrame: # type: ignore
        """
        Load benchmarking dataset for the selected cell from DuckDB.

        This function connects to the DuckDB benchmarking database, loads
        either the training or test dataset, and filters the records for the
        selected cell label. The resulting DataFrame contains all cycles for
        that cell, including true outlier labels.

        Args:
            dataset_type (str, optional): Type of dataset to load. Must be
                one of:
                - ``"train"``: Load training dataset from
                ``df_train_dataset_sv``.
                - ``"test"``: Load test dataset from
                ``df_test_dataset_sv``.
                Defaults to ``"train"``.

        Returns:
            pd.DataFrame: Benchmarking dataset filtered for the selected
            cell label.

        Raises:
            AssertionError: If the selected cell label is not found in the
                database.

        Example:
            .. code-block::

                # Path to the DuckDB instance:
                # "train_dataset_severson.db"
                db_filepath = (
                    Path.cwd()
                    .parent.parent
                    .joinpath("database","train_dataset_severson.db"))

                # Get the cell-ID from cell_inventory
                selected_cell_label = "2017-05-12_5_4C-70per_3C_CH17"

                # Import the BenchDB class
                # Load only the dataset based on the selected cell
                benchdb = BenchDB(
                    db_filepath,
                    selected_cell_label)

                # load the benchmarking dataset
                df_selected_cell = benchdb.load_benchmark_dataset(
                    dataset_type="train")
        """
        if os.path.exists(self._db_filepath):
            print("Database is found in the given filepath.")
            print(f"Loading benchmarking dataset now...")
            # Create a DuckDB connection
            con = duckdb.connect(
                self._db_filepath,
                read_only=True)

            if dataset_type == "train":
                df_duckdb = con.execute(
                    "SELECT * FROM df_train_dataset_sv").fetchdf()
            elif dataset_type == "test":
                df_duckdb = con.execute(
                    "SELECT * FROM df_test_dataset_sv").fetchdf()

            # Filter dataset for specific selected cell only
            assert (self._selected_cell_label
                in df_duckdb["cell_index"].unique()), (
                f"{self._selected_cell_label} does not exist in database")

            df_selected_cell = df_duckdb[
                df_duckdb["cell_index"] == self._selected_cell_label]

            print("*"*100)

            return df_selected_cell

        else:
            print("Filepath is not valid. Please ensure that database "
                  + "can be found in the given filepath.")

    def drop_labels(
        self,
        df_selected_cell: pd.DataFrame,
        filter_col: list = None) -> pd.DataFrame: # type: ignore
        """
        Remove true outlier labels from a cell cycling dataset.

        This function drops the ``outlier`` column from the benchmarking
        dataset for the selected cell. Optionally, it can also filter the
        dataset to retain only the specified columns.

        Args:
            df_selected_cell (pd.DataFrame): Input cycling dataset for a
                single cell, including the ``outlier`` label column.
            filter_col (list, optional): List of column names to retain after
                dropping labels. If None, all remaining columns are returned.
                Defaults to None.

        Returns:
            pd.DataFrame: DataFrame without the ``outlier`` column. If
            ``filter_col`` is provided, only the specified columns are kept.

        Example:
            .. code-block::

                # Path to the DuckDB instance: "train_dataset_severson.db"
                db_filepath = (
                    Path.cwd()
                    .parent.parent
                    .joinpath("database","train_dataset_severson.db"))
                print(db_filepath)

                # Import the BenchDB class
                # Load only the dataset based on the selected cell
                benchdb = BenchDB(
                    db_filepath,
                    selected_cell_label)

                # load the benchmarking dataset
                df_selected_cell = benchdb.load_benchmark_dataset(
                    dataset_type="train")

                if df_selected_cell is not None:

                    filter_col = [
                        "cell_index",
                        "cycle_index",
                        "discharge_capacity",
                        "voltage"]

                    # Drop true labels from the benchmarking dataset
                    # and filter for selected columns only
                    benchdb.drop_labels(
                        df_selected_cell,
                        filter_col)
            """
        # Drop the label from the benchmarking dataset
        df_selected_cell_no_label = df_selected_cell.drop(
            "outlier", axis=1).reset_index(drop=True)

        if filter_col is not None:
            df_selected_cell_no_label_filtered = df_selected_cell_no_label[
                filter_col].reset_index(drop=True)

            return df_selected_cell_no_label_filtered

        else:
            return df_selected_cell_no_label

    def get_true_outlier_cycle_index(
        self,
        df_selected_cell: pd.DataFrame) -> np.ndarray:
        """
        Extract true outlier labels from the benchmarking dataset.

        Args:
            df_selected_cell (pd.DataFrame): Cell cycling dataset based on
                selected cell index.

        Returns:
            np.ndarray: True outliers labels from the benchmarking dataset.

        Example:
            .. code-block::

                # Extract true outliers cycle index
                # from benchmarking dataset
                true_outlier_cycle_idx = battdb.get_true_outlier_cycle_index(
                    df_selected_cell)
                print(f"True outlier cycle index: {true_outlier_cycle_idx}")
        """
        # Anomalous cycle has label = 1
        # Normal cycle has label = 0
        # true outliers from benchmarking dataset
        df_true_outlier = df_selected_cell[
            df_selected_cell["outlier"] == 1]

        # Get the cycle index of anomalous cycle
        true_outlier_cycle_index = df_true_outlier["cycle_index"].unique()

        return true_outlier_cycle_index


    def plot_cycle_data(
        self,
        df_selected_cell_without_labels: pd.DataFrame,
        true_outlier_cycle_index: list=None): # type: ignore
        """
        Visualize discharge voltage vs. capacity cycles for a cell.

        This function plots cycling data for the selected cell. If a list of
        true outlier cycle indices is provided, those cycles are highlighted
        and annotated on the plot. Otherwise, no annotations of anomalies will
        be shown. The figure is saved in the cell’s artifacts directory.

        Args:
            df_selected_cell_without_labels (pd.DataFrame): DataFrame
                containing discharge capacity, voltage, and cycle index for
                the selected cell (without labels).
            true_outlier_cycle_index (list, optional): List of cycle indices
                known to be anomalous. If None, cycles are plotted without
                outlier highlights. Defaults to None.

        Returns:
            matplotlib.axes.Axes: Axes object containing the cycle plot.

        Example:
            .. code-block::

                # Plot cell data with true anomalies
                # If the true outlier cycle index is not known,
                # cycling data will be plotted without labels
                benchdb.plot_cycle_data(
                    df_selected_cell_without_labels,
                    true_outlier_cycle_index)
        """
        # Reset the sns settings
        mpl.rcParams.update(mpl.rcParamsDefault)
        rcParams["text.usetex"] = True

        if true_outlier_cycle_index is not None:
            # Anomalous cycle has label = 1
            # Normal cycle has label = 0
            # true outliers from benchmarking dataset
            df_true_outlier = df_selected_cell_without_labels[
                df_selected_cell_without_labels.cycle_index.isin(
                    true_outlier_cycle_index)]

            # Plot normal cycles with true outliers
            axplot = bviz.plot_cycle_data(
                xseries=(
                    df_selected_cell_without_labels["discharge_capacity"]),
                yseries=df_selected_cell_without_labels["voltage"],
                cycle_index_series=(
                    df_selected_cell_without_labels["cycle_index"]),
                xoutlier=df_true_outlier["discharge_capacity"],
                youtlier=df_true_outlier["voltage"])

            # Create textbox to annotate anomalous cycle
            textstr = '\n'.join((
                r"\textbf{Cycle index with anomalies:}",
                f"{true_outlier_cycle_index}"))

            # properties for bbox
            props = dict(
                boxstyle='round',
                facecolor='wheat',
                alpha=0.5)

            # first 0.95 corresponds to the left right alignment starting
            # from left, second 0.95 corresponds to up down alignment
            # starting from bottom
            axplot.text(
                0.95, 0.95,
                textstr,
                transform=axplot.transAxes,
                fontsize=12,
                # ha means right alignment of the text
                ha="right", va='top',
                bbox=props)
        else:
            # Plot normal cycles without outliers
            axplot = bviz.plot_cycle_data(
                xseries=(
                    df_selected_cell_without_labels["discharge_capacity"]),
                yseries=df_selected_cell_without_labels["voltage"],
                cycle_index_series=(
                    df_selected_cell_without_labels["cycle_index"]))

        axplot.set_xlabel(
            r"Discharge capacity, $Q_\textrm{dis}$ [Ah]",
            fontsize=14)
        axplot.set_ylabel(
            r"Discharge voltage, $V_\textrm{dis}$ [V]",
            fontsize=14)

        axplot.set_title(
            f"Cell {self._selected_cell_label}",
            fontsize=16)

        # Use pdf for figures in the paper, but loading pdf output
        # can take time
        # Use png for normal plot
        output_fig_filename = (
            "plot_cycles_"
            + self._selected_cell_label
            + ".png")

        fig_output_path = (
            self._selected_cell_artifacts.joinpath(output_fig_filename))

        plt.savefig(
            fig_output_path,
            dpi=200,
            bbox_inches="tight")

        if self._show_fig_status:
            plt.show()

    def load_features_db(
        self,
        db_features_filepath: Union[pathlib.PosixPath, str],
        dataset_type: str):
        """
        Load features for the selected cell from a DuckDB database.

        This function connects to a DuckDB database containing precomputed
        features for battery cells. It supports loading either training or
        test datasets, filters the data for the selected cell label, and
        returns the resulting feature DataFrame.

        Args:
            db_features_filepath (Union[pathlib.PosixPath, str]): Path to the
                DuckDB features database file.
            dataset_type (str): Type of dataset to load. Must be one of:
                - ``"train"``: Load training features from
                ``df_train_features_sv``.
                - ``"test"``: Load test features from
                ``df_test_features_sv``.

        Returns:
            pd.DataFrame: DataFrame containing features for the selected cell.

        Raises:
            AssertionError: If the selected cell label is not found in the
                database.

        Example:
            .. code-block::

                # Define the filepath to ``train_features_severson.db``
                # DuckDB instance.
                db_features_filepath = (
                    Path.cwd()
                    .parent.parent
                    .joinpath("database","train_features_severson.db"))

                # Load only the training features dataset
                df_features_per_cell = benchdb.load_features_db(
                    db_features_filepath,
                    dataset_type="train")

                unique_cycle_count = (
                    df_features_per_cell["cycle_index"].unique())
        """

        if os.path.exists(db_features_filepath):
            print("Features database is found in the given filepath.")


            # Create a DuckDB connection
            con = duckdb.connect(
                db_features_filepath,
                read_only=True)

            if dataset_type == "train":
                df_merge_features_train = con.execute(
                    "SELECT * FROM df_train_features_sv").fetchdf()
            elif dataset_type == "test":
                df_merge_features_train = con.execute(
                    "SELECT * FROM df_test_features_sv").fetchdf()

        # Filter dataset for specific selected cell only
        assert (self._selected_cell_label
            in df_merge_features_train["cell_index"].unique()), (
            f"{self._selected_cell_label} does not exist in database")

        df_features_per_cell = (df_merge_features_train[
            df_merge_features_train["cell_index"] ==
            self._selected_cell_label]
            .reset_index(drop=True))

        print(f"Features database is loaded.")
        print("*"*100)
        return df_features_per_cell