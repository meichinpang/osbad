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
from sklearn.covariance import EmpiricalCovariance

rcParams["text.usetex"] = True

# Custom osbad library for anomaly detection
import osbad.config as bconf
import osbad.hyperparam as hp
import osbad.modval as modval
import osbad.viz as bviz
from osbad.database import BenchDB
from osbad.model import ModelRunner


# ---------------------------------------------------------------------------
# Define a global variable to save pipeline artifacts
# Use pathlib to define OS independent
# filepath navigation

# Export current hyperparameters to CSV
hyperparam_filepath =  bconf.PIPELINE_OUTPUT_DIR.joinpath(
    "hyperparams_iforest_tohoku.csv")

# Export current metrics to CSV
hyperparam_eval_metrics_filepath =  bconf.PIPELINE_OUTPUT_DIR.joinpath(
    "eval_metrics_tohoku.csv")

# --------------------------------------------------------------------------
# Load only the training dataset

# Path to the DuckDB instance:
# "tohoku_benchmark_dataset.db" inside the database folder
db_filepath = (
    Path.cwd()
    .parent.parent.parent
    .joinpath("database","tohoku_benchmark_dataset.db"))

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

# Take only the first five cells for training
unique_cell_index_train = df_duckdb["cell_index"].unique()[:5]
print(unique_cell_index_train)

training_cell_count = len(unique_cell_index_train)
print(f"Training cell count: {training_cell_count}")

if __name__ == "__main__":

    for idx, selected_cell_label in enumerate(unique_cell_index_train):
        print("Evaluating cell now:")
        print(idx, selected_cell_label)

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
        # Update the hyperparameter config during runtime
        total_cycle_count = len(unique_cycle_index)
        print(f"Total cycle count per cell: {total_cycle_count}")

        hp_config_iforest = {
            "contamination": {"low": 0.0, "high": 0.5},
            "n_estimators": {"low": 100, "high": 500},
            "max_samples": {"low": 100, "high": total_cycle_count},
            "threshold": {"low": 0.0, "high": 1.0}
        }

        # Save the updated hp config into a json file
        iforest_hp_config_filepath = (
            Path.cwd()
            .parent.parent.parent
            .joinpath(
                "machine_learning",
                "hp_config_schema",
                "tohoku_hp_config",
                "iforest_hp_config.json"))

        bconf.create_json_hp_config(
            iforest_hp_config_filepath,
            hp_dict=hp_config_iforest)

        # --------------------------------------------------------------------
        # Hyperparameter tuning with Bayesian optimization

        # Reload the hp module to refresh in-memory variables
        # especially after updating parameters
        from importlib import reload
        reload(hp)

        # Check if the schema in the script has been updated
        # based on the current constraints specified during runtime
        print("Current hyperparameter config:")
        print(hp.IFOREST_HP_CONFIG)
        print("-"*70)

        # Instantiate an optuna study for iForest model
        sampler = optuna.samplers.TPESampler(seed=42)

        selected_feature_cols = (
            "max_discharge_capacity",
            "norm_mahal_dist")

        if_study = optuna.create_study(
            study_name="iforest_hyperparam",
            sampler=sampler,
            directions=["maximize","maximize"])

        if_study.optimize(
            lambda trial: hp.objective(
                trial,
                model_id="iforest",
                df_feature_dataset=df_merge_features,
                selected_feature_cols=selected_feature_cols,
                df_benchmark_dataset=df_selected_cell,
                selected_cell_label=selected_cell_label),
            n_trials=20)

        # -------------------------------------------------------------------
        # Aggregate best trials
        schema_iforest = {
            "threshold": "median",
            "contamination": "median",
            "n_estimators": "median_int",
            "max_samples": "median_int",
        }

        df_iforest_hyperparam = hp.aggregate_best_trials(
            if_study.best_trials,
            cell_label=selected_cell_label,
            model_id="iforest",
            schema=schema_iforest)

        # -------------------------------------------------------------------
        # Plot Pareto Front
        hp.plot_pareto_front(
            if_study,
            selected_cell_label,
            fig_title="Isolation Forest Pareto Front")

        # -------------------------------------------------------------------
        # Export current hyperparameters to CSV
        hp.export_current_hyperparam(
            df_iforest_hyperparam,
            selected_cell_label,
            export_csv_filepath=hyperparam_filepath,
            if_exists="replace")

        # -------------------------------------------------------------------
        # Read hyperparameter from stored CSV
        df_hyperparam_from_csv = pd.read_csv(hyperparam_filepath)

        df_param_per_cell = df_hyperparam_from_csv[
            df_hyperparam_from_csv["cell_index"] == selected_cell_label]

        # Create a dict for best trial parameters
        param_dict = df_param_per_cell.iloc[0].to_dict()
        pprint.pp(param_dict)

        # -------------------------------------------------------------------
        # Run the model with best trial parameters
        cfg = hp.MODEL_CONFIG["iforest"]

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
            model_name="Isolation Forest",
            xoutliers=df_outliers_pred["max_discharge_capacity"],
            youtliers=df_outliers_pred["norm_mahal_dist"],
            pred_outliers_index=pred_outlier_indices,
            threshold=param_dict["threshold"],
            square_grid=False,
            grid_offset=1
        )

        axplot.set_xlabel(
            r"Maximum discharge capacity per cycle",
            fontsize=12)
        axplot.set_ylabel(
            r"Normalized Mahalanobis distance",
            fontsize=12)

        output_fig_filename = (
            "iforest_"
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
            "Isolation Forest",
            fontsize=16)

        output_fig_filename = (
            "conf_matrix_iforest_"
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
            model_name="iforest",
            selected_cell_label=selected_cell_label,
            df_eval_outliers=df_eval_outlier)

        # -------------------------------------------------------------------
        # Export model performance metrics to CSV output
        hp.export_current_model_metrics(
            model_name="iforest",
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
        axplot = benchdb.plot_cycle_data(
            df_selected_cell_without_labels,
            true_outlier_cycle_index)

        axplot.set_title(
            f"Cell-{cell_num}",
            fontsize=16)

        output_fig_filename = (
            "cycling_data_with_labels_"
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

        df_true_outliers = (df_merge_features[
            df_merge_features["cycle_index"]
            .isin(true_outlier_cycle_index)].copy())

        axplot = bviz.plot_cycle_data(
            xseries=unique_cycle_index,
            yseries=max_cap_per_cycle,
            cycle_index_series=unique_cycle_index,
            xoutlier=df_true_outliers["cycle_index"],
            youtlier=df_true_outliers["max_discharge_capacity"])

        axplot.set_xlabel(
            r"Cycle index",
            fontsize=14)
        axplot.set_ylabel(
            r"Maximum discharge capacity [mAh/g]",
            fontsize=14)

        axplot.set_title(
            f"Cell-{cell_num}",
            fontsize=16)

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

        output_fig_filename = (
            "cap_fade_with_labels_"
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


        print(f"END OF CELL EVALUATION {selected_cell_label}")
        print("*"*150)