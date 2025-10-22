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

rcParams["text.usetex"] = True

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
    "hp_01_iforest_hyperparam_proxy_severson.csv")

# Export current metrics to CSV
hyperparam_eval_metrics_filepath =  Path.cwd().joinpath(
    "eval_metrics_severson_train_multiple_cells.csv")

# --------------------------------------------------------------------------
# Path to database directory
DB_DIR = bconf.DB_DIR

# Path to the DuckDB instance:
# "osbad/database/train_dataset_severson.db"
db_filepath = DB_DIR.joinpath("train_dataset_severson.db")

# Define the filepath to ``train_features_severson.db``
# "osbad/database/train_features_severson.db"
db_features_filepath = DB_DIR.joinpath("train_features_severson.db")

# Create a DuckDB connection
con = duckdb.connect(
    db_filepath,
    read_only=True)

# Load all training dataset from duckdb
df_duckdb = con.execute(
    "SELECT * FROM df_train_dataset_sv").fetchdf()

unique_cell_index_train = df_duckdb["cell_index"].unique()
print(unique_cell_index_train)

# unique_cell_index_train = ['2017-05-12_4C-80per_4C_CH6']
# print(unique_cell_index_train)

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

        # -------------------------------------------------------------------
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
        # Load only the training features dataset
        df_features_per_cell = benchdb.load_features_db(
            db_features_filepath,
            dataset_type="train")
        print(df_features_per_cell.head(10).to_markdown())
        print("-"*70)

        unique_cycle_count = (
            df_features_per_cell["cycle_index"].unique())

        # --------------------------------------------------------------------
        # Hyperparameter tuning with Bayesian optimization

        # Update the HP config for max_samples depending on the cycle numbers
        total_cycle_count = len(
            df_selected_cell_without_labels["cycle_index"].unique())

        hp_config_iforest = {
            "contamination": {"low": 0.0, "high": 0.5},
            "n_estimators": {"low": 100, "high": 500},
            "max_samples": {"low": 100, "high": total_cycle_count},
            "threshold": {"low": 0.0, "high": 1.0}
        }

        iforest_hp_config_filepath = (
            Path.cwd()
            .parents[3]
            .joinpath(
                "machine_learning",
                "hp_config_schema",
                "severson_hp_config",
                "iforest_hp_config.json"))

        bconf.create_json_hp_config(
            iforest_hp_config_filepath,
            hp_dict=hp_config_iforest)

        # Reload the hp module to refresh in-memory variables
        # especially after updating parameters
        from importlib import reload
        reload(hp)

        # Check if the schema in the script has been updated
        # based on the current constraints specified
        # from the notebook
        print("Current hyperparameter config:")
        print(hp._IFOREST_HP_CONFIG)
        print("-"*70)

        # --------------------------------------------------------------------
        sampler = optuna.samplers.TPESampler(seed=42)

        selected_feature_cols = (
            "cycle_index",
            "log_max_diff_dQ",
            "log_max_diff_dV")

        # Instantiate an optuna study
        if_study = optuna.create_study(
            study_name="iforest_hyperparam",
            sampler=sampler,
            directions=["minimize","maximize"])

        if_study.optimize(
            lambda trial: hp.objective(
                trial,
                model_id="iforest",
                df_feature_dataset=df_features_per_cell,
                selected_feature_cols=selected_feature_cols,
                #df_benchmark_dataset=df_selected_cell,
                selected_cell_label=selected_cell_label),
            n_trials=100)

        # -------------------------------------------------------------------
        # Aggregate best trials
        schema_iforest = {
            "threshold": "median",
            "contamination": "median",
            "n_estimators": "median_int",
            "max_samples": "median_int",
        }

        trade_off_trials_list = hp.trade_off_trials_detection(
            if_study)

        df_iforest_hyperparam = hp.aggregate_best_trials(
            trade_off_trials_list,
            cell_label=selected_cell_label,
            model_id="iforest",
            schema=schema_iforest)

        # -------------------------------------------------------------------
        # Plot Pareto Front
        hp.plot_proxy_pareto_front(
            if_study,
            trade_off_trials_list,
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
            model_name="Isolation Forest",
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

        print(f"END OF TRAIN CELL EVALUATION {selected_cell_label}")
        print("*"*170)