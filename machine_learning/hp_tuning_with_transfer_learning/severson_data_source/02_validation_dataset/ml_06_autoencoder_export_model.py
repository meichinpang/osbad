# Standard library
import os
import pprint
from pathlib import Path
from joblib import dump, load

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
# Use pathlib to define OS independent
# filepath navigation
# Export current hyperparameters to CSV
hyperparam_filepath =  Path.cwd().joinpath(
    "hp_06_autoencoder_hyperparam_severson.csv")

# Export current metrics to CSV
hyperparam_eval_metrics_filepath =  Path.cwd().joinpath(
    "eval_metrics_severson_train_export_model.csv")

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

training_cell_count = len(unique_cell_index_train)
print(f"Training cell count: {training_cell_count}")

selected_cell_label = "2017-05-12_5_4C-70per_3C_CH17"
print(f"Train model with average hyperparameters on {selected_cell_label}")

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
    # Read hyperparameters values from CSV file
    df_hyperparam_from_csv = pd.read_csv(hyperparam_filepath)

    # Fit with mean hyperparameters from the training dataset
    avg_batch_size = (
        int(np.mean(
            df_hyperparam_from_csv["batch_size"])))

    avg_epoch_num = (
        int(np.mean(df_hyperparam_from_csv["epoch_num"])))

    avg_learning_rate = (
        np.mean(df_hyperparam_from_csv["learning_rate"]))

    avg_dropout_rate = (
        np.mean(df_hyperparam_from_csv["dropout_rate"]))

    avg_threshold = (
        np.mean(df_hyperparam_from_csv["threshold"]))

    param_dict = {
        'ml_model': 'autoencoder',
        'cell_index': selected_cell_label,
        'batch_size': avg_batch_size,
        'epoch_num': avg_epoch_num,
        'learning_rate': avg_learning_rate,
        'dropout_rate': avg_dropout_rate,
        'threshold': avg_threshold}

    print("Parameter dictionary:")
    pprint.pprint(param_dict)
    print("-"*70)

    # --------------------------------------------------------------------
    selected_feature_cols = (
        "log_max_diff_dQ",
        "log_max_diff_dV")

    # -------------------------------------------------------------------
    # Run the model with best trial parameters
    cfg = hp.MODEL_CONFIG["autoencoder"]

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
        model_name="Autoencoder",
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
        "export_model_autoencoder_"
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
        "Autoencoder",
        fontsize=16)

    output_fig_filename = (
        "export_model_conf_matrix_autoencoder_"
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
        model_name="autoencoder",
        selected_cell_label=selected_cell_label,
        df_eval_outliers=df_eval_outlier)

    # -------------------------------------------------------------------
    # Export model performance metrics to CSV output
    hp.export_current_model_metrics(
        model_name="autoencoder",
        selected_cell_label=selected_cell_label,
        df_current_eval_metrics=df_current_eval_metrics,
        export_csv_filepath=hyperparam_eval_metrics_filepath,
        if_exists="replace")

    # Export model using average parameters trained on selected_cell_label
    # Save all exported models into one directory
    EXPORT_MODEL_DIR = Path.cwd().joinpath("exported_models_dir")
    if not os.path.exists(EXPORT_MODEL_DIR):
        os.mkdir(EXPORT_MODEL_DIR)

    export_model_filepath = EXPORT_MODEL_DIR.joinpath(
        f"autoencoder_train_{selected_cell_label}.joblib")
    dump(model, export_model_filepath)


    print(f"EXPORTED {export_model_filepath} TRAINED "
            + f"ON CELL {selected_cell_label}")
    print("*"*170)