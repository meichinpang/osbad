# Standard library
import pprint
from pathlib import Path

# Third-party libraries
import duckdb
import fireducks.pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import optuna
from matplotlib import rcParams
from statistics import mode

rcParams["text.usetex"] = True

# Custom osbad library for anomaly detection
import osbad.config as bconf
import osbad.hyperparam as hp
import osbad.modval as modval
import osbad.viz as bviz
from osbad.database import BenchDB
from osbad.model import ModelRunner

# ---------------------------------------------------------------------------
# Path to the database directory
DB_DIR = bconf.DB_DIR

# Import frozen hyperparameters from the validation dataset
current_path = Path(__file__).resolve()
hyperparam_filepath = current_path.parent.parent.joinpath(
    "02_validation_dataset",
    "hp_04_lof_hyperparam_proxy_severson.csv")

# Export current metrics to CSV in the current working dir
hyperparam_eval_metrics_filepath =  Path.cwd().joinpath(
    "eval_metrics_severson_test_multiple_cells.csv")

# --------------------------------------------------------------------------
# Load only the test dataset
# Path to the DuckDB instance:
# "osbad/database/test_dataset_severson.db"
db_filepath = DB_DIR.joinpath("test_dataset_severson.db")

# Define the filepath to ``test_features_severson.db``
# "osbad/database/test_features_severson.db"
db_features_filepath = DB_DIR.joinpath("test_features_severson.db")

# Create a DuckDB connection
con = duckdb.connect(
    db_filepath,
    read_only=True)

# Load all test dataset from duckdb
df_duckdb = con.execute(
    "SELECT * FROM df_test_dataset_sv").fetchdf()

unique_cell_index_test = df_duckdb["cell_index"].unique()
print(f"Unique cell index: {unique_cell_index_test}")

test_cell_count = len(unique_cell_index_test)
print(f"Test cell count: {test_cell_count}")
print("-"*70)

if __name__ == "__main__":

    for idx, selected_cell_label in enumerate(unique_cell_index_test):
        print("Evaluating cell now:")
        print(idx, selected_cell_label)

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

        # load the benchmarking dataset
        df_selected_cell = benchdb.load_benchmark_dataset(
            dataset_type="test")

        if df_selected_cell is not None:

            filter_col = [
                "cell_index",
                "cycle_index",
                "discharge_capacity",
                "voltage"]

            # Drop true labels from the benchmarking dataset
            # and filter for selected columns only
            df_selected_cell_without_labels = benchdb.drop_labels(
                df_selected_cell,
                filter_col)

            # print a subset of the dataframe
            # for diagnostics running in terminals
            print(df_selected_cell_without_labels.head(10).to_markdown())
            print("-"*70)

        # --------------------------------------------------------------------
        # Plot cycle data without labels
        # If the true outlier cycle index is not known,
        # cycling data will be plotted without labels
        benchdb.plot_cycle_data(
            df_selected_cell_without_labels)

        output_fig_filename = (
            "cycle_data_without_labels_"
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

        # --------------------------------------------------------------------
        # Custom features transformation pipeline
        # Load only the test features dataset
        df_features_per_cell = benchdb.load_features_db(
            db_features_filepath,
            dataset_type="test")

        print(df_features_per_cell.head(10).to_markdown())
        print("-"*70)

        unique_cycle_count = (
            df_features_per_cell["cycle_index"].unique())

        # --------------------------------------------------------------------
        # Test Local Outlier Factor (LOF)
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
        # Run the model with average best trial parameters
        # (frozen from the training dataset)
        cfg = hp.MODEL_CONFIG["lof"]

        selected_feature_cols = (
            "log_max_diff_dQ",
            "log_max_diff_dV")

        runner = ModelRunner(
            cell_label=selected_cell_label,
            df_input_features=df_features_per_cell,
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
        print(f"\n***Predicted outlier cycle index:***")
        print(pred_outlier_indices)
        print("\n")

        # -------------------------------------------------------------------
        # Get df_outliers_pred
        df_outliers_pred = df_features_per_cell[
            df_features_per_cell["cycle_index"].isin(
                pred_outlier_indices)].copy()

        df_outliers_pred["outlier_prob"] = pred_outlier_score

        # -------------------------------------------------------------------
        # Predict anomaly score map
        axplot = runner.predict_anomaly_score_map(
            selected_model=model,
            model_name="Local Outlier Factor (LOF)",
            xoutliers=df_outliers_pred["log_max_diff_dQ"],
            youtliers=df_outliers_pred["log_max_diff_dV"],
            pred_outliers_index=pred_outlier_indices,
            threshold=param_dict["threshold"]
        )

        axplot.set_xlabel(
            r"$\log(\Delta Q_\textrm{scaled,max,cyc)}\;\textrm{[Ah]}$",
            fontsize=12)
        axplot.set_ylabel(
            r"$\log(\Delta V_\textrm{scaled,max,cyc})\;\textrm{[V]}$",
            fontsize=12)

        output_fig_filename = (
            "lof_"
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
            "conf_matrix_lof_"
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
        # Finally: check with true labels
        # Extract true outliers cycle index from benchmarking dataset
        true_outlier_cycle_index = benchdb.get_true_outlier_cycle_index(
            df_selected_cell)
        print(f"True outlier cycle index:")
        print(true_outlier_cycle_index)

        # Plot cell data with true anomalies
        # If the true outlier cycle index is not known,
        # cycling data will be plotted without labels
        benchdb.plot_cycle_data(
            df_selected_cell_without_labels,
            true_outlier_cycle_index)

        output_fig_filename = (
            "cycle_data_with_labels_"
            + selected_cell_label
            + ".png")

        fig_output_path = (
            selected_cell_artifacts_dir.joinpath(output_fig_filename))

        plt.savefig(
            fig_output_path,
            dpi=600,
            bbox_inches="tight")

        plt.close()

        # -------------------------------------------------------------------
        # Plot the bubble chart and label the true outliers
        # Calculate the bubble size ratio for plotting
        df_bubble_size_dQ = bviz.calculate_bubble_size_ratio(
            df_variable=df_features_per_cell["max_diff_dQ"])

        df_bubble_size_dV = bviz.calculate_bubble_size_ratio(
            df_variable=df_features_per_cell["max_diff_dV"])

        bubble_size = (
            np.abs(df_bubble_size_dV)
            * np.abs(df_bubble_size_dQ))

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
            r"$\log(\Delta Q_\textrm{scaled,max,cyc)}\;\textrm{[Ah]}$",
            fontsize=12)
        axplot.set_ylabel(
            r"$\log(\Delta V_\textrm{scaled,max,cyc})\;\textrm{[V]}$",
            fontsize=12)

        output_fig_filename = (
            "log_bubble_plot_"
            + selected_cell_label
            + ".png")

        fig_output_path = (
            selected_cell_artifacts_dir.joinpath(output_fig_filename))

        plt.savefig(
            fig_output_path,
            dpi=600,
            bbox_inches="tight")

        plt.close()

        print(f"END OF TEST CELL EVALUATION {selected_cell_label}")
        print("*"*170)