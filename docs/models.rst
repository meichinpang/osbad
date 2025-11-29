Models Guide
################

List of Benchmarked Models
==================================

As of today, available models included in this benchmarking study are

* Statistical models

  - Standard Deviation
  - Median Absolute Deviation (MAD)
  - Interquartile Range (IQR)
  - Z-score
  - Modified Z-score

* Distance-based models

  - Euclidean Distance
  - Manhattan Distance
  - Minkowski Distance
  - Mahalanobis Distance

* Machine-learning models

  - Isolation Forest
  - K-Nearest Neighbors (KNN)
  - Gaussian Mixture Models (GMM)
  - Local Outlier Factor (LOF)
  - Principal Component Analysis (PCA)
  - Autoencoders (AE)

Hyperparameter Tuning
=========================

In this study, we implemented hyperparameter tuning for unsupervised
anomaly detection models using two different methods:

  (1) Leveraging meta-learning with labelled outliers to train a model with
      hyperparameter tuning (training dataset) and subsequenty predict the
      model performance on new, unlabeled datasets (test dataset) and
  (2) Using the output from the unsupervised model (e.g., predicted inliers
      from all anomaly detection models) as features for a supervised task,
      and adjust the unsupervised model's hyperparameters to maximize
      the performance of the downstream supervised model.

Instead of blindly trying random hyperparameters as in random search or
computing expensive hyperparameter searching as in grid search,
we implemented Bayesian optimization to build a probabilistic model of the
objective function and to choose the most promising hyperparameters.


Example
==================
.. toctree::
   :maxdepth: 2
   :caption: List of Model Tutorials

   severson_iforest_example
   tohoku_iforest_example
   severson_dbad_euclidean_example
   severson_knn_proxy_regr_example