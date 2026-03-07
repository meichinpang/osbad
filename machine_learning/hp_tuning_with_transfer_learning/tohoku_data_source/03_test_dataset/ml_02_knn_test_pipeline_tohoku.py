# Standard library
import pprint
from pathlib import Path

# Third-party libraries
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import load
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

# Import frozen hyperparameters from the validation dataset
current_path = Path(__file__).resolve()
hyperparam_filepath = current_path.parent.parent.joinpath(
    "02_validation_dataset",
    "hp_02_knn_hyperparam_tohoku.csv")

# Export current metrics to CSV in the current working dir
hyperparam_eval_metrics_filepath =  Path.cwd().joinpath(
    "eval_metrics_tohoku_test_multiple_cells.csv")

# --------------------------------------------------------------------------
# Path to the DuckDB instance:
# "tohoku_benchmark_dataset.db" inside the database folder
DB_DIR = bconf.DB_DIR
db_filepath = DB_DIR.joinpath("tohoku_benchmark_dataset.db")

# Create a DuckDB connection
con = duckdb.connect(
    db_filepath,
    read_only=True)

# Load all dataset from duckdb
df_duckdb = con.execute(
    "SELECT * FROM df_tohoku_dataset").fetchdf()

# Drop the additional index column
df_duckdb = df_duckdb.drop(
    columns="__index_level_0__",
    errors="ignore")

# Cell-index for test ML models
unique_cell_index_test = [
    'cell_num_7',
    'cell_num_8',
    'cell_num_9',
    'cell_num_10']

test_cell_count = len(unique_cell_index_test)

print(f"Cell index for testing: {unique_cell_index_test}")
print(f"Test cell count: {test_cell_count}")
print("-"*70)

# Define grid offset size for anomaly score map
GRID_OFFSET_SIZE = 1

# --------------------------------------------------------------------------
# Load trained model
# Please change the cell-index name here if you change the cell-idx used for
# training and exporting the model
TRAINED_MODEL_FILEPATH = current_path.parent.parent.joinpath(
    "02_validation_dataset",
    "exported_models_dir",
    "knn_train_cell_num_1.joblib")

trained_model = load(TRAINED_MODEL_FILEPATH)
print("Trained model configuration:")
print(trained_model)
print("-"*70)

if __name__ == "__main__":

    for idx, selected_cell_label in enumerate(unique_cell_index_test):
        print(f"Evaluating cell now: {selected_cell_label}")

        # -------------------------------------------------------------------
        # Create a subfolder to store fig output
        # corresponding to each cell-index
        selected_cell_artifacts_dir = bconf.artifacts_output_dir(
            selected_cell_label)

        # Extract only the last digit for plotting purposes
        cell_num = selected_cell_label[-1]

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
            "cap_fade_without_labels_"
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

        # ----------------------------------------------------------------
        # Test K Nearest Neighbors (KNN)
        # Read hyperparameters values from CSV file
        cfg = hp.MODEL_CONFIG["knn"]

        df_hyperparam_from_csv = pd.read_csv(hyperparam_filepath)

        # Get average hyperparameters tuned from the training dataset
        avg_threshold = np.mean(
            df_hyperparam_from_csv["threshold"])

        runner = ModelRunner(
            cell_label=selected_cell_label,
            df_input_features=df_merge_features,
            selected_feature_cols=selected_feature_cols
        )

        # Create Xdata from test dataset
        Xdata = runner.create_model_x_input()

        # Predict with trained model
        proba = trained_model.predict_proba(Xdata)

        (pred_outlier_indices,
            pred_outlier_score) = runner.pred_outlier_indices_from_proba(
            proba=proba,
            threshold=avg_threshold,
            outlier_col=cfg.proba_col)

        # -------------------------------------------------------------------
        # Get df_outliers_pred
        df_outliers_pred = (df_merge_features[
            df_merge_features["cycle_index"]
            .isin(pred_outlier_indices)].copy())

        df_outliers_pred["outlier_prob"] = pred_outlier_score

        # -------------------------------------------------------------------
        # Predict anomaly score map
        axplot = runner.predict_anomaly_score_map(
            selected_model=trained_model,
            model_name="K Nearest Neighbors (KNN)",
            xoutliers=df_outliers_pred["max_discharge_capacity"],
            youtliers=df_outliers_pred["norm_mahal_dist"],
            pred_outliers_index=pred_outlier_indices,
            threshold=avg_threshold,
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
            f"knn_grid_offset_size_{GRID_OFFSET_SIZE}_"
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
            "K Nearest Neighbors (KNN)",
            fontsize=16)

        output_fig_filename = (
            "conf_matrix_knn_"
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
            model_name="knn",
            selected_cell_label=selected_cell_label,
            df_eval_outliers=df_eval_outlier)

        # -------------------------------------------------------------------
        # Export model performance metrics to CSV output
        hp.export_current_model_metrics(
            model_name="knn",
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
            f"Cell-{cell_num}: Predicted Anomalies with KNN",
            fontsize=16)

        output_fig_filename = (
            "knn_pred_cycles_with_outliers_"
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
            f"Cell-{cell_num}: Predicted Anomalies with KNN",
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
            "knn_pred_cap_fade_with_outliers_"
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


        print(f"END OF TEST CELL EVALUATION {selected_cell_label}")
        print("*"*150)