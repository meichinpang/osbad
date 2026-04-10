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

.. code-block:: python

   IOException: IO Error: The file "train_dataset_severson.db" exists, but it is not a 
   valid DuckDB database file!

If you are using Git LFS for the first time, you can follow the `Git LFS installation guide
<https://github.com/git-lfs/git-lfs/wiki/Installation>`_ to install Git LFS.

.. code-block:: bash

  # On Ubuntu/Debian
  sudo apt install git-lfs

  # Then initialize and pull
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

   # Then you can start Jupyter Notebook
   jupyter notebook

In the Jupyter browser UI, navigate to the following notebook to run the
osbad workflow for the Isolation Forest model on the Severson dataset:

``machine_learning/hp_tuning_with_transfer_learning/severson_data_source/01_train_dataset/ml_01_iforest_hyperparam_severson.ipynb``

Typical Workflow
-------------------

1. Load the benchmarking dataset and features database
2. For unsupervised ML models, run hyperparameter tuning using Bayesian
   optimization.
3. Train models with best trial hyperparameters
4. Evaluate with model performance with confusion matrix and model performance
   KPI such as accuracy, precision, recall, F1-score and Matthew correlation
   coefficient.

Next Steps
--------------

* See :doc:`doc_02_dataset_overview` for an overview of the datasets
  included in this benchmarking project.
* Explore :doc:`doc_03_models_overview` for details on the models
  included in this benchmarking study.

.. * Run comparisons in :doc:`doc_04_benchmarking` to evaluate model
     performance.