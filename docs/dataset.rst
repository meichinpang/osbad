Datasets Guide
################

Overview
======================
In this open-source anomaly detection benchmarking project, we only work with
publicly available published dataset:

* :doc:`severson_dataset`
* :doc:`tohoku_dataset`

Data management toolstack
============================

**DuckDB**: The data management is this project is implemented using DuckDB
(Link: https://duckdb.org/). To access the dataset curated in this project,
you would need to first install duckdb:

.. code-block::

   uv pip install duckdb


Git Large File Storage (LFS)
==============================
Because large files cannot be committed and versioned directly in Git
repositories, we used Git Large File Storage (LFS) for versioning large
dataset in this project (Link: https://git-lfs.com/). You can follow the
documentation there to setup Git LFS for your local development.


Contributions
================
We welcome contributions to enhance our open-source anomaly detection
benchmarking suite. New datasets, use-cases, or extensions to existing
repositories are highly encouraged. If you are interested in contributing,
feel free to reach out or submit a pull request.


Example Dataset
==================
.. toctree::
   :maxdepth: 2
   :caption: List of Dataset

   severson_dataset
   tohoku_dataset
