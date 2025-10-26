# Standard library
import pprint
from joblib import dump, load
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
# Path to the database directory
DB_DIR = bconf.DB_DIR

# Import frozen hyperparameters from the validation dataset
current_path = Path(__file__).resolve()
hyperparam_filepath = current_path.parent.parent.joinpath(
    "02_validation_dataset",
    "hp_04_lof_hyperparam_severson.csv")

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

# --------------------------------------------------------------------------
# Load trained model
# Please change the cell-index name here if you change the cell-idx used for
# training and exporting the model
TRAINED_MODEL_FILEPATH = current_path.parent.parent.joinpath(
    "02_validation_dataset",
    "exported_models_dir",
    "lof_train_2017-05-12_5_4C-70per_3C_CH17.joblib")

trained_model = load(TRAINED_MODEL_FILEPATH)
print("Trained model configuration:")
print(trained_model)
print("-"*70)


for idx, selected_cell_label in enumerate(unique_cell_index_test):

    print(f"Evaluating cell-{idx} now: {selected_cell_label}")

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
    # Test Local Outlier Factor (LOF) model
    # Read hyperparameters values from CSV file
    df_hyperparam_from_csv = pd.read_csv(hyperparam_filepath)

    # Get the average threshold tuned from the training dataset
    avg_threshold = np.mean(
        df_hyperparam_from_csv["threshold"])

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

    # Create Xdata from test dataset
    Xdata = runner.create_model_x_input()

    # Predict with trained model
    proba = trained_model.predict_proba(Xdata)

    (pred_outlier_indices,
        pred_outlier_score) = runner.pred_outlier_indices_from_proba(
        proba=proba,
        threshold=avg_threshold,
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
        selected_model=trained_model,
        model_name="Local Outlier Factor (LOF)",
        xoutliers=df_outliers_pred["log_max_diff_dQ"],
        youtliers=df_outliers_pred["log_max_diff_dV"],
        pred_outliers_index=pred_outlier_indices,
        threshold=avg_threshold
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

    print(f"END OF TEST CELL EVALUATION {selected_cell_label}")
    print("*"*170)