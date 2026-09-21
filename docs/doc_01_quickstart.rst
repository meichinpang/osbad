Quickstart
=============

This project is implemented and managed using UV environment, which is an
extremely fast Python package and project manager, written in Rust.

**Step-1: Install uv**

Follow the `official Astral installation documentation
<https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1/>`_ to
install uv.

**Step-2: Setup Git LFS for your system**

This project uses Git LFS to manage large files such as datasets. Without Git LFS, 
you will only be able to download the text pointers instead of the actual large files, 
which will lead to errors when you run the benchmarking scripts.

Example error message when Git LFS is not installed:

.. code-block:: text

   IOException: IO Error: The file "train_dataset_severson.db" exists, but it is not a 
   valid DuckDB database file!

If you are using Git LFS for the first time, you can follow the `Git LFS installation guide
<https://github.com/git-lfs/git-lfs/wiki/Installation>`_ to install Git LFS.

.. code-block:: bash

  # On Ubuntu/Debian
  sudo apt install git-lfs

  # Then initialize Git LFS in your system
  git lfs install

**Step-3: Clone osbad from the GitHub Repository**

* Clone the osbad repository to access the example notebooks and scripts
* Sync dependencies and activate the virtual environment
* Pull the large dataset files with Git LFS

.. code-block:: bash

   # Clone the osbad repository to access the example notebooks and scripts
   git clone git@github.com:meichinpang/osbad.git

   # Change into the cloned osbad repository
   cd osbad

   # Sync Dependencies
   uv sync

   # Activate the virtual environment
   # On macOS/Linux:
   source .venv/bin/activate

   # On Windows (Command Prompt):
   .venv\Scripts\activate.bat

   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1

   # Pull the large dataset files with Git LFS
   git lfs pull


To test ``osbad`` installation, run the script ``test_osbad_installation.py`` 
in the root directory of the project. This script imports the osbad package 
and prints the current version to confirm that the installation is successful.

.. code-block:: bash

   python test_osbad_installation.py

If the installation is successful, you should see an output similar to the following:

.. code-block:: bash

   Hello from osbad!
   osbad current version: X.Y.Z
   OSBAD package installation is successful!

   # Then you can start Jupyter Notebook
   jupyter notebook

In the Jupyter browser UI, navigate to the following notebook to run the
osbad workflow for the Isolation Forest model on the Severson dataset:

``machine_learning/hp_tuning_with_transfer_learning/severson_data_source/01_train_dataset/ml_01_iforest_hyperparam_severson.ipynb``

Typical Workflow
-------------------

The notebook
``ml_01_iforest_hyperparam_severson.ipynb``
demonstrates the end-to-end osbad workflow for the Isolation Forest model:

1. **Import libraries**: Load ``osbad``, ``optuna``, ``duckdb``, ``pandas``,
   and ``matplotlib``.
2. **Load the benchmarking dataset**: Connect to the Severson training
   dataset (``train_dataset_severson.db``) via DuckDB and select a cell
   for analysis.
3. **Drop true labels**: Remove ground-truth anomaly labels from the
   dataset to simulate an unsupervised setting.
4. **Plot raw cycle data**: Visualize discharge capacity vs. voltage
   curves without labels.
5. **Load features database**: Import pre-computed features from
   ``train_features_severson.db``.
6. **Hyperparameter tuning**: Run Bayesian optimization with Optuna
   (TPE sampler, 20 trials) to find the best Isolation Forest
   hyperparameters (contamination, n_estimators, max_samples, threshold).
7. **Aggregate best trials**: Extract the median hyperparameters from
   the Pareto-optimal trials and export them to CSV.
8. **Train the model**: Fit an Isolation Forest with the best trial
   parameters and predict anomaly scores.
9. **Visualize anomaly score map**: Plot the decision boundary and
   predicted outliers in feature space.
10. **Evaluate model performance**: Generate a confusion matrix and
    compute performance metrics (accuracy, precision, recall, F1-score,
    and Matthews correlation coefficient).
11. **Export evaluation metrics**: Save the model performance results
    to a CSV file.
12. **Verify with true labels**: Compare predicted outliers against
    the ground-truth labels using cycle plots and bubble charts.


Issues and Troubleshooting
----------------------------
* If you encounter errors related to missing files or invalid database
  files, ensure that you have Git LFS installed and have pulled the
  large dataset files correctly.
* If you see errors about missing Python packages, make sure you have
  activated the virtual environment with ``source .venv/bin/activate`` (or the
  appropriate command for your operating system) and that you have run
  ``uv sync`` to install all dependencies.
* For any other issues, please open an issue on the
  `OSBAD Issue Tracker <https://github.com/meichinpang/osbad/issues>`_
  with details about the error message and your system configuration.

Next Steps
--------------

* See :doc:`doc_02_dataset_overview` for an overview of the datasets
  included in this benchmarking project.
* Explore :doc:`doc_03_models_overview` for details on the models
  included in this benchmarking study.

.. * Run comparisons in :doc:`doc_04_benchmarking` to evaluate model
     performance.