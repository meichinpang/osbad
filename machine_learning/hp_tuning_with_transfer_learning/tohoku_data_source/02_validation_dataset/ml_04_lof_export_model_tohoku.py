# Standard library
import os
import pprint
from pathlib import Path

# Third-party libraries
import duckdb
from joblib import dump
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.covariance import EmpiricalCovariance
from statistics import mode

# Custom osbad library for anomaly detection
import osbad.config as bconf
import osbad.hyperparam as hp
import osbad.modval as modval
import osbad.viz as bviz
from osbad.database import BenchDB
from osbad.model import ModelRunner


# ---------------------------------------------------------------------------
# Use pathlib to define OS independent
# filepath navigation
# Export current hyperparameters to CSV
hyperparam_filepath =  Path.cwd().joinpath(
    "hp_04_lof_hyperparam_tohoku.csv")

# Export current metrics to CSV
hyperparam_eval_metrics_filepath =  Path.cwd().joinpath(
    "eval_metrics_tohoku_train_multiple_cells.csv")

# --------------------------------------------------------------------------
# Load only the training dataset

# Path to the DuckDB instance:
# "tohoku_benchmark_dataset.db" inside the database folder
DB_DIR = bconf.DB_DIR
db_filepath = DB_DIR.joinpath("tohoku_benchmark_dataset.db")

# Create a DuckDB connection
con = duckdb.connect(
    db_filepath,
    read_only=True)

# Load all training dataset from duckdb
df_duckdb = con.execute(
    "SELECT * FROM df_tohoku_dataset").fetchdf()

# Drop the additional index column
df_duckdb = df_duckdb.drop(
    columns="__index_level_0__",
    errors="ignore")

selected_cell_label = "cell_num_1"

# Extract only the last digit for plotting purposes
cell_num = selected_cell_label.split("_")[-1]

# Define grid offset size for anomaly score map
GRID_OFFSET_SIZE = 1

if __name__ == "__main__":

    print(f"Evaluating cell now: {selected_cell_label}")

    # -------------------------------------------------------------------
    # Create a subfolder to store fig output
    # corresponding to each cell-index
    selected_cell_artifacts_dir = bconf.artifacts_output_dir(
        selected_cell_label)

    # -------------------------------------------------------------------
    # Import the BenchDB class
    # Load only the dataset based on the selected cell
    benchdb = BenchDB(
        db_filepath,
        selected_cell_label)

    # Filter dataset for specific selected cell only
    df_selected_cell = df_duckdb[
        df_duckdb["cell_index"] == selected_cell_label]

    # Drop the outlier labels
    df_selected_cell_without_labels = df_selected_cell.drop(
        "outlier", axis=1).reset_index(drop=True)

    # print a subset of the dataframe
    # for diagnostics running in terminals
    print(df_selected_cell_without_labels.head(10).to_markdown())
    print("-"*70)

    # ----------------------------------------------------------------
    # Calculate maximum capacity per cycle
    max_cap_per_cycle = (
        df_selected_cell_without_labels
            .groupby(["cycle_index"])["discharge_capacity"].max())
    max_cap_per_cycle.name = "max_discharge_capacity"

    unique_cycle_index = (
        df_selected_cell_without_labels["cycle_index"].unique())
    df_cycle_index = pd.Series(unique_cycle_index, name="cycle_index")

    # ----------------------------------------------------------------
    # Plot capacity fade without labels
    axplot = bviz.plot_cycle_data(
        xseries=unique_cycle_index,
        yseries=max_cap_per_cycle,
        cycle_index_series=unique_cycle_index)

    axplot.set_xlabel(
        r"Cycle index",
        fontsize=14)

    axplot.set_ylabel(
        r"Maximum discharge capacity [mAh/g]",
        fontsize=14)

    axplot.set_title(
        f"Cell-{cell_num}",
        fontsize=16)

    output_fig_filename = (
        "export_model_cap_fade_without_labels_"
        + selected_cell_label
        + ".png")

    fig_output_path = (
        selected_cell_artifacts_dir.joinpath(output_fig_filename))

    plt.savefig(
        fig_output_path,
        dpi=600,
        bbox_inches="tight")

    plt.close()

    # ----------------------------------------------------------------
    # Feature: Normalized Mahalanobis distance
    # Calculate Mahalanobis distance from unique_cycle_index and
    # max_cap_per_cycle

    # Input features for Mahalanobis distance
    df_features_per_cell = pd.concat(
        [df_cycle_index,
            max_cap_per_cycle],
        axis=1)

    Xdata = df_features_per_cell.values

    cov = EmpiricalCovariance().fit(Xdata)
    mahal_dist = cov.mahalanobis(Xdata)

    df_maha_dist = pd.Series(
        mahal_dist,
        name="mahal_dist")

    df_merge_features = pd.concat(
        [df_features_per_cell,
            df_maha_dist], axis=1)

    # Calculate maximum mahal_dist to
    # normalize the distance calculation
    max_mahal_dist = (
        df_merge_features["mahal_dist"].max())

    df_merge_features["norm_mahal_dist"] = (
        df_merge_features["mahal_dist"]/max_mahal_dist)

    # Only use the max_discharge_capacity and norm_mahal_dist as the
    # input features to track sudden capacity drop
    selected_feature_cols = (
        "max_discharge_capacity",
        "norm_mahal_dist")

    print(df_merge_features.head(10).to_markdown())
    print("-"*70)

    # --------------------------------------------------------------------
    # Read hyperparameters values from CSV file
    df_hyperparam_from_csv = pd.read_csv(hyperparam_filepath)

    # Fit with mean hyperparameters from the training dataset
    # The 'n_neighbors' parameter of LocalOutlierFactor must be an int
    avg_n_neighbors = (
        int(np.mean(
            df_hyperparam_from_csv["n_neighbors"])))

    mode_metric = mode(
        df_hyperparam_from_csv["metric"])

    # The 'leaf_size' parameter of LocalOutlierFactor must be an int
    median_leaf_size = (
        int(np.mean(df_hyperparam_from_csv["leaf_size"])))

    avg_contamination = (
        np.mean(df_hyperparam_from_csv["contamination"]))

    avg_threshold = np.mean(
        df_hyperparam_from_csv["threshold"])

    param_dict = {
        'ml_model': 'lof',
        'cell_index': selected_cell_label,
        'contamination': avg_contamination,
        'n_neighbors': avg_n_neighbors,
        'leaf_size': median_leaf_size,
        'metric': mode_metric,
        'threshold': avg_threshold}


    print("Parameter dictionary:")
    pprint.pprint(param_dict)
    print("-"*70)

    # -------------------------------------------------------------------
    # Run the model with best trial parameters
    cfg = hp.MODEL_CONFIG["lof"]

    runner = ModelRunner(
        cell_label=selected_cell_label,
        df_input_features=df_merge_features,
        selected_feature_cols=selected_feature_cols
    )

    Xdata = runner.create_model_x_input()

    model = cfg.model_param(param_dict)
    print(model)
    model.fit(Xdata)
    proba = model.predict_proba(Xdata)

    (pred_outlier_indices,
        pred_outlier_score) = runner.pred_outlier_indices_from_proba(
        proba=proba,
        threshold=param_dict["threshold"],
        outlier_col=cfg.proba_col
    )

    # -------------------------------------------------------------------
    # Get df_outliers_pred
    df_outliers_pred = (df_merge_features[
        df_merge_features["cycle_index"]
        .isin(pred_outlier_indices)].copy())

    df_outliers_pred["outlier_prob"] = pred_outlier_score

    # -------------------------------------------------------------------
    # Predict anomaly score map
    axplot = runner.predict_anomaly_score_map(
        selected_model=model,
        model_name="Local Outlier Factor (LOF)",
        xoutliers=df_outliers_pred["max_discharge_capacity"],
        youtliers=df_outliers_pred["norm_mahal_dist"],
        pred_outliers_index=pred_outlier_indices,
        threshold=param_dict["threshold"],
        square_grid=False,
        grid_offset=GRID_OFFSET_SIZE
    )

    axplot.set_xlabel(
        r"Maximum discharge capacity per cycle",
        fontsize=12)
    axplot.set_ylabel(
        r"Normalized Mahalanobis distance",
        fontsize=12)

    output_fig_filename = (
        f"export_model_lof_grid_offset_size_{GRID_OFFSET_SIZE}_"
        + selected_cell_label
        + ".png")

    fig_output_path = (
        selected_cell_artifacts_dir
        .joinpath(output_fig_filename))

    plt.savefig(
        fig_output_path,
        dpi=600,
        bbox_inches="tight")

    plt.close()

    # -------------------------------------------------------------------
    # Model performance evaluation
    df_eval_outlier = modval.evaluate_pred_outliers(
        df_benchmark=df_selected_cell,
        outlier_cycle_index=pred_outlier_indices)

    # -------------------------------------------------------------------
    # Confusion Matrix
    axplot = modval.generate_confusion_matrix(
        y_true=df_eval_outlier["true_outlier"],
        y_pred=df_eval_outlier["pred_outlier"])

    axplot.set_title(
        "Local Outlier Factor (LOF)",
        fontsize=16)

    output_fig_filename = (
        "export_model_conf_matrix_lof_"
        + selected_cell_label
        + ".png")

    fig_output_path = (
        selected_cell_artifacts_dir
        .joinpath(output_fig_filename))

    plt.savefig(
        fig_output_path,
        dpi=600,
        bbox_inches="tight")

    plt.close()

    # -------------------------------------------------------------------
    # Evaluate model performance
    df_current_eval_metrics = modval.eval_model_performance(
        model_name="lof",
        selected_cell_label=selected_cell_label,
        df_eval_outliers=df_eval_outlier)

    # -------------------------------------------------------------------
    # Export model performance metrics to CSV output
    hp.export_current_model_metrics(
        model_name="lof",
        selected_cell_label=selected_cell_label,
        df_current_eval_metrics=df_current_eval_metrics,
        export_csv_filepath=hyperparam_eval_metrics_filepath,
        if_exists="replace")

    # -------------------------------------------------------------------
    # Plot predicted anomalies

    axplot = benchdb.plot_cycle_data(
        df_selected_cell_without_labels,
        pred_outlier_indices)

    axplot.set_title(
        f"Cell-{cell_num}: Predicted Anomalies with LOF",
        fontsize=16)

    output_fig_filename = (
        "export_model_lof_pred_cycles_with_outliers_"
        + selected_cell_label
        + ".png")

    fig_output_path = (
        selected_cell_artifacts_dir
        .joinpath(output_fig_filename))

    plt.savefig(
        fig_output_path,
        dpi=600,
        bbox_inches="tight")

    plt.close()

    # -------------------------------------------------------------------
    # Plot capacity fade with labels

    pred_cap_outlier = max_cap_per_cycle[
        max_cap_per_cycle
            .index.isin(pred_outlier_indices)]

    axplot = bviz.plot_cycle_data(
        xseries=unique_cycle_index,
        yseries=max_cap_per_cycle,
        cycle_index_series=unique_cycle_index,
        xoutlier=pred_cap_outlier.index,
        youtlier=pred_cap_outlier)

    axplot.set_xlabel(
        r"Cycle index",
        fontsize=14)
    axplot.set_ylabel(
        r"Maximum discharge capacity [mAh/g]",
        fontsize=14)

    axplot.set_title(
        f"Cell-{cell_num}: Predicted Anomalies with LOF",
        fontsize=16)

    # Create textbox to annotate anomalous cycle
    textstr = '\n'.join((
        r"Cycle index with anomalies:",
        f"{list(pred_cap_outlier.index)}"))

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

    output_fig_filename = (
        "export_model_lof_pred_cap_fade_with_outliers_"
        + selected_cell_label
        + ".png")

    fig_output_path = (
        selected_cell_artifacts_dir
        .joinpath(output_fig_filename))

    plt.savefig(
        fig_output_path,
        dpi=600,
        bbox_inches="tight")

    plt.close()


    # -----------------------------------------------------------------------
    # Export model using average parameters trained on selected_cell_label
    # Save all exported models into one directory
    EXPORT_MODEL_DIR = Path.cwd().joinpath("exported_models_dir")
    if not os.path.exists(EXPORT_MODEL_DIR):
        os.mkdir(EXPORT_MODEL_DIR)

    export_model_filepath = EXPORT_MODEL_DIR.joinpath(
        f"lof_train_{selected_cell_label}.joblib")
    dump(model, export_model_filepath)


    print(f"EXPORTED {export_model_filepath} TRAINED "
            + f"ON CELL {selected_cell_label}")
    print("*"*170)