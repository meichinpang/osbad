# STEP-1: LOAD LIBRARIES
# Standard library
import os
from pathlib import Path

# Third-party libraries
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Custom osbad library for anomaly detection
import osbad.config as bconf
from osbad.database import BenchDB

# Path to database directory
DB_DIR = bconf.DB_DIR

db_filepath = DB_DIR.joinpath("train_dataset_severson.db")

BASE_DIR = Path.cwd().parent.parent
OUT_TABLE_SAMPLE = BASE_DIR.joinpath("docs", "sample_data")
OUT_FIGS = BASE_DIR.joinpath("docs", "docs_figure")
SAMPLE_ROWS = 50

Path.mkdir(OUT_TABLE_SAMPLE, exist_ok=True)

# Create a DuckDB connection
con = duckdb.connect(
    db_filepath,
    read_only=True)

# Load all training dataset from duckdb
df_duckdb = con.execute(
    "SELECT * FROM df_train_dataset_sv").fetchdf()

# Get the cell index of training dataset
unique_cell_index_train = df_duckdb["cell_index"].unique()
print(f"Unique cell index: {unique_cell_index_train}")

# Get the cell-ID from cell_inventory
selected_cell_label = "2017-05-12_5_4C-70per_3C_CH17"

# -------------------------------------------------------------------------
# STEP-3: LOAD BENCHMARKING DATASET

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

   # Extract true outliers cycle index from benchmarking dataset
   true_outlier_cycle_index = benchdb.get_true_outlier_cycle_index(
      df_selected_cell)
   print(f"True outlier cycle index:")
   print(true_outlier_cycle_index)

# # Save sample rows as CSV for .. csv-table::
sample_html_path = os.path.join(OUT_TABLE_SAMPLE, "severson_sample.html")
# sample_html_path.parent.mkdir(parents=True, exist_ok=True)
html_table = df_selected_cell.head(SAMPLE_ROWS).to_html(
    classes=["docutils", "dataframe"],  # reuse Sphinx styling
    index=False,
    border=0
)

with open(sample_html_path, "w", encoding="utf-8") as f:
    f.write(html_table)

print("Sample dataset saved in HTML format")


# Histograms for each numeric column
selected_cols = ["voltage", "discharge_capacity", "current", 
                 "internal_resistance", "temperature"] 
n = len(selected_cols)

fig, axes = plt.subplots(1, n, figsize=(5*n, 4), dpi=120, constrained_layout=True)

for ax, col in zip(axes, selected_cols):
    ax.hist(df_selected_cell[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
    ax.set_title(f"{col}")

plt.savefig(OUT_FIGS.joinpath("hist_features.png"))
plt.close(fig)


