Quickstart
=============

This project is implemented and managed using UV environment, which is an
extremely fast Python package and project manager, written in Rust.

**Step-1: Install uv**

Follow the `official Astral installation documentation
<https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1/>`_ to
install uv.

**Step-2: Install osbad from PyPi**

* Creates a new project folder named ``osbad`` with a basic ``pyproject.toml``
  and ``README.md`` and switch into the project directory on your terminal.
* Create a uv virtual environment within the project directory.
* Add the ``osbad`` dependency by running ``uv add osbad``. This step updates
  your ``pyproject.toml`` under ``[project].dependencies``.
* Generate a lockfile which pins exact versions of all dependencies for
  reproducibility.
* Sync dependencies by installing all dependencies listed in ``uv.lock`` and
  ensures your virtual environment matches the locked versions.

.. code-block:: bash

   # Initialize a new project
   uv init osbad

   # Switch into the project directory
   cd osbad

   # Create a uv virtual environment within the project directory
   uv venv

   # Add the osbad dependency
   uv add osbad

   # Generate a Lockfile
   uv lock

   # Sync Dependencies
   uv sync

To test ``osbad`` installation, replace the script in ``main.py`` with

.. code-block:: python

  # Test osbad installation:
  # osbad for open-source benchmark of anomaly detection
  from importlib.metadata import version
  import osbad

  def main():
      print("Hello from osbad!")
      osbad_current_version = version("osbad")
      print(f"osbad current version: {osbad_current_version}")
      print(f"OSBAD package installation is successful!")

  if __name__ == "__main__":
      main()

On your terminal where ``main.py`` is located, run

.. code-block:: bash

   # Execute the python script with uv run
   uv run main.py

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

* See :doc:`dataset`
* Explore :doc:`models`
.. * Run comparisons in :doc:`benchmarking`
