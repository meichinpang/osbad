# Standard library
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Union

# Third-party libraries
import numpy as np
import pandas as pd


ArrayLike = Union[pd.Series, np.ndarray]
"""
Type alias for array-like inputs.

Represents data structures that can be treated as arrays in numerical
and analytical operations. This alias is used for type hints to accept
both pandas Series and NumPy ndarray objects.
"""

# ----------------------------------------------------------------------------
# Stats Method dataclass to declare the type

@dataclass(frozen=True)
class OutlierMethodConfig:
    """
    Immutable configuration for a statistical outlier detector.

    Stores::

      - compute: the detector implementation accepting
        (X_variable, **stats_params_dict)
      - params: default statistical parameters
    """
    compute: Callable[[Any], Tuple]
    params: Dict[str, Any]

# ----------------------------------------------------------------------------
# Implementation of the stats outlier detection methods as _private_methods

def _compute_sd_outliers(
    df_variable: ArrayLike,
    std_dev_threshold: float = 3.0,
    ddof: int = 1) -> tuple:
    """
    Detect outliers using the standard deviation rule.

    This function computes the mean and standard deviation of the input
    variable and identifies outliers that fall outside the range defined
    by ± ``std_dev_threshold`` standard deviations from the mean.

    Args:
        df_variable (ArrayLike): Input data as a pandas Series or NumPy
            ndarray.
        std_dev_threshold (float, optional): Number of standard deviations
            from the mean to define outlier thresholds. Defaults to 3.0.
        ddof (int, optional): Delta degrees of freedom for standard
            deviation calculation. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Indices of detected outliers.
            - float: Lower bound (mean - std_dev_threshold * std).
            - float: Upper bound (mean + std_dev_threshold * std).

    .. note::

        For anomaly detection in battery cycling protocols, deviations in
        features such as ``max_diff_dQ`` or ``max_diff_dV`` beyond 3
        standard deviations from the mean may indicate potential
        anomalous cycles.
    """

    # Calculate the mean and std deviation
    # from the max diff feature
    feature_mean = np.mean(df_variable)
    feature_std = np.std(df_variable, ddof=ddof)
    print(f"SD feature mean: {feature_mean}")
    print(f"SD feature std: {feature_std}")

    # Mix and max limit
    # defined as 3-std deviation from the
    # distribution mean
    SD_min_limit = feature_mean - std_dev_threshold*feature_std
    SD_max_limit = feature_mean + std_dev_threshold*feature_std

    print(f"SD lower bound: {SD_min_limit}")
    print(f"SD upper bound: {SD_max_limit}")

    std_outlier_index = np.where(
        (df_variable > SD_max_limit) |
        (df_variable < SD_min_limit))
    print(f"Std anomalous cycle index: {std_outlier_index[0]}")

    if isinstance(std_outlier_index, tuple):
        # convert tuple into numpy array
        return (std_outlier_index[0], SD_min_limit,SD_max_limit)
    else:
        return (std_outlier_index, SD_min_limit,SD_max_limit)


def _calculate_mad_factor(
    df_variable: ArrayLike,
    ddof:int =1):
    """
    Calculate the scaling factor for Median Absolute Deviation (MAD).

    This function estimates the MAD scaling factor by transforming the
    input data into z-scores, computing the 75th percentile of the
    standardized distribution, and taking the reciprocal of its value.
    The factor is used to normalize MAD for robust outlier detection.

    Args:
        df_variable (ArrayLike): Input feature data as a pandas Series
            or NumPy array.
        ddof (int, optional): Delta degrees of freedom used when
            calculating the standard deviation. Defaults to 1.

    Returns:
        float: The calculated MAD scaling factor.
    """

    # Transform the distribution to have a mean of zero
    # and std-deviation of one
    mean_var = np.mean(df_variable)
    std_var = np.std(df_variable, ddof=ddof)
    var_zscore = (df_variable - mean_var)/std_var
    mean_zscore = np.mean(var_zscore)
    std_zscore = np.std(var_zscore, ddof=1)
    print(f"Feature z-score mean: {np.round(mean_zscore,2)}")
    print(f"Feature z-score std. deviation: {np.round(std_zscore,2)}")

    # Calculate 75th percentile of the standard distribution
    Q3_std_distribution = np.quantile(var_zscore, 0.75)

    # MAD-factor: 1/75th percentile of the standard distribution
    # Here, we use the absolute value
    mad_factor = np.abs(1/Q3_std_distribution)

    return mad_factor


def _compute_mad_outliers(
    df_variable: ArrayLike,
    mad_threshold: float = 3.0,
    mad_factor: float=None,
    ddof: int =1) -> Tuple:
    """
    Detect outliers using the Median Absolute Deviation (MAD) method.

    This function identifies outliers in a dataset by computing the
    median, absolute deviations, and applying the MAD thresholding
    rule. By default, the MAD factor is calculated dynamically if not
    provided, ensuring robustness against skewed or non-Gaussian
    distributions. A scaling parameter ``mad_threshold`` determines
    how many MADs away from the median a value must be to be flagged
    as an outlier.

    Args:
        df_variable (ArrayLike): Input feature data as a pandas Series
            or NumPy array.
        mad_threshold (float, optional): Scaling factor for defining
            thresholds. Outliers are flagged if they fall outside
            ``median ± mad_threshold * MAD``. Defaults to 3.0.
        mad_factor (float, optional): Scaling factor for MAD. If None,
            it is estimated automatically using the distribution.
            Defaults to None.
        ddof (int, optional): Delta degrees of freedom for standard
            deviation used in estimating the MAD factor. Defaults to 1.

    Returns:
        Tuple: A tuple containing:
            - np.ndarray: Indices of detected outliers.
            - float: Lower MAD threshold.
            - float: Upper MAD threshold.

    .. note::

        - MAD is more robust to extreme values than the standard
          deviation method.
        - Outliers are flagged if they fall outside
          ``median ± mad_threshold * MAD``.
        - The ``mad_factor`` plays an important role to determine the
          corresponding MAD-score. If the underlying data distribution is
          Gaussian, then we can assume that ``mad_factor`` = 1.4826.
        - If we would like to relax the assumption about the normality
          of a feature distribution, then ``mad_factor`` can be
          calculated from the reciprocal of the 75th-percentile of a
          standard distribution, which means a distribution with a
          mean of zero and a standard deviation of one).
    """

    # Calculate the median of the feature
    median = np.median(df_variable)
    print(f"Feature median: {median}")

    # Calculate absolute deviation from the median
    abs_deviations = np.abs(df_variable - median)

    if mad_factor is None:

        mad_factor = _calculate_mad_factor(
            df_variable,
            ddof)

    # Calculate MAD-score
    MAD = mad_factor*np.median(abs_deviations)
    print(f"MAD: {MAD}")

    # Calculate upper MAD limit
    MAD_min_limit = median - mad_threshold*MAD
    print(f"MAD min limit: {MAD_min_limit}")

    # Calculate lower MAD limit
    MAD_max_limit = median + mad_threshold*MAD
    print(f"MAD max limit: {MAD_max_limit}")

    MAD_outlier_index = np.where(
        (df_variable < MAD_min_limit) |
        (df_variable > MAD_max_limit))

    if isinstance(MAD_outlier_index, tuple):
        # convert tuple into numpy array
        return (MAD_outlier_index[0], MAD_min_limit, MAD_max_limit)
    else:
        return (MAD_outlier_index, MAD_min_limit, MAD_max_limit)

def _compute_iqr_outliers(
    df_variable: ArrayLike,
    iqr_threshold: float = 1.5):
    """
    Detect outliers using the Interquartile Range (IQR) method.

    This function computes the first quartile (Q1), third quartile (Q3),
    and the interquartile range (IQR = Q3 - Q1).

    Args:
        df_variable (ArrayLike): Input feature data as a pandas Series or
            NumPy array.
        iqr_threshold (float, optional): Multiplier applied to the IQR for
            defining outlier bounds. Defaults to 1.5.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Indices of detected outliers.
            - float: Lower IQR threshold.
            - float: Upper IQR threshold.

    .. note::

        - Outliers are flagged if they fall outside
          ``[Q1 - iqr_threshold*IQR, Q3 + iqr_threshold*IQR]``.
    """
    quartiles = np.quantile(
        df_variable,
        [0.25, 0.5, 0.75])
    Q1 = quartiles[0]
    Q3 = quartiles[2]
    IQR = Q3 - Q1

    IQR_min_limit = Q1 - iqr_threshold*IQR
    IQR_max_limit = Q3 + iqr_threshold*IQR

    print(f"IQR lower limit: {IQR_min_limit}")
    print(f"IQR upper limit: {IQR_max_limit}")

    IQR_outlier_index = np.where(
        (df_variable < IQR_min_limit) |
        (df_variable> IQR_max_limit))

    if isinstance(IQR_outlier_index, tuple):
        # convert tuple into numpy array
        return (IQR_outlier_index[0], IQR_min_limit, IQR_max_limit)
    else:
        return (IQR_outlier_index, IQR_min_limit, IQR_max_limit)

def _compute_zscore_outliers(
    df_variable: ArrayLike,
    zscore_threshold: float=3,
    ddof:int =1) -> tuple:
    """
    Detect outliers using the Z-score method.

    This function standardizes the input variable using its mean and
    standard deviation, then identifies values that fall outside a given
    Z-score threshold. It assumes the data is approximately normally
    distributed, where most values lie within ±3 standard deviations from
    the mean.

    Args:
        df_variable (ArrayLike): Input feature data as a pandas Series or
            NumPy array.
        zscore_threshold (float, optional): Threshold for flagging outliers
            based on Z-scores. Defaults to 3.
        ddof (int, optional): Delta degrees of freedom used in the standard
            deviation calculation. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Indices of detected outliers.
            - float: Lower Z-score threshold.
            - float: Upper Z-score threshold.

    .. note::

        - After Z-transformation, the feature mean should be close to 0,
          and the standard deviation close to 1.
        - Outliers are flagged if they fall outside
          ``[-zscore_threshold, zscore_threshold]``.
    """
    # Calculate the mean and std deviation
    # from the max diff feature
    feature_mean = np.mean(df_variable)

    # use unbiased estimator with the bessel correction factor (n-1) when
    # using the std deviation method from numpy (ddof=1)
    feature_std = np.std(df_variable, ddof=ddof)
    print(f"Feature mean: {feature_mean}")
    print(f"Feature std: {feature_std}")

    feature_zscore = (df_variable - feature_mean)/feature_std
    print("-"*70)
    print("After Z-transformation, feature mean should be close to 0 "
            + "and feature std should be close to 1.")
    print(f"Zscore feature mean: {np.mean(feature_zscore)}")
    print(f"Zscore feature std: {np.std(feature_zscore, ddof=1)}")

    zscore_min_limit = - zscore_threshold
    zscore_max_limit = zscore_threshold

    zscore_outlier_index = np.where(
        (feature_zscore > zscore_max_limit) |
        (feature_zscore < zscore_min_limit))
    print(f"Zscore anomalous cycle index: {zscore_outlier_index}")

    if isinstance(zscore_outlier_index , tuple):
        # convert tuple into numpy array
        return (zscore_outlier_index[0],
                zscore_min_limit,
                zscore_max_limit)
    else:
        return (zscore_outlier_index,
                zscore_min_limit,
                zscore_max_limit)

def _compute_modified_z_outliers(
    df_variable: ArrayLike,
    mad_factor: float=None,
    mod_zscore_threshold = 3.5,
    ddof: int = 1) -> tuple:
    """
    Detect outliers using the Modified Z-Score method.

    This method computes the modified z-score, which is based on the
    median and the Median Absolute Deviation (MAD).

    Args:
        df_variable (ArrayLike): Input feature data as a pandas Series
            or NumPy array.
        mad_factor (float, optional): Scaling factor for MAD. If None,
            it is estimated automatically. Defaults to None.
        mod_zscore_threshold (float, optional): Cutoff for the modified
            z-score. Values with modified z greater than this threshold
            are flagged as outliers. Defaults to 3.5.
        ddof (int, optional): Delta degrees of freedom for standard
            deviation used in estimating the MAD factor. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Indices of detected outliers.
            - float: Lower modified z-score threshold.
            - float: Upper modified z-score threshold.

    .. note::

        - Modified Z-Score method is more robust than the standard z-score
          method for datasets with skewed or non-Gaussian distributions.
          Outliers are flagged if their modified z-score exceeds the
          specified threshold.
    """
    # Calculate the median of the feature
    median = np.median(df_variable)
    print(f"Feature median: {median}")

    # Calculate absolute deviation from the median
    abs_deviations = np.abs(df_variable - median)

    if mad_factor is None:

        mad_factor = _calculate_mad_factor(
            df_variable,
            ddof)

    # Calculate MAD-score
    MAD = mad_factor*np.median(abs_deviations)
    print(f"MAD: {MAD}")

    modified_zscore = (df_variable - median)/MAD

    # Modified z-score lower limit
    modified_zmin_limit = - mod_zscore_threshold
    print(f"Modified Zmin limit: {modified_zmin_limit}")

    # Modified z-score upper limit
    modified_zmax_limit = mod_zscore_threshold
    print(f"Modified Zmax limit: {modified_zmax_limit}")

    modified_zoutlier_index = np.where(
        (modified_zscore < modified_zmin_limit) |
        (modified_zscore > modified_zmax_limit))

    if isinstance(modified_zoutlier_index, tuple):
        # convert tuple into numpy array
        return (
            modified_zoutlier_index[0],
            modified_zmin_limit,
            modified_zmax_limit)
    else:
        return (modified_zoutlier_index,
                modified_zmin_limit,
                modified_zmax_limit)

# ----------------------------------------------------------------------------
# Statitical anomaly detection registry

outlier_method: Dict[str, OutlierMethodConfig] = {
    "sd": OutlierMethodConfig(
        compute=_compute_sd_outliers,
        params={"std_dev_threshold": 3.0, "ddof": 1},
    ),
    "mad": OutlierMethodConfig(
        compute=_compute_mad_outliers,
        params={"mad_factor": None, "mad_threshold": 3.0, "ddof": 1},
    ),
    "iqr": OutlierMethodConfig(
        compute=_compute_iqr_outliers,
        params={"iqr_threshold": 1.5},
    ),
    "zscore": OutlierMethodConfig(
        compute=_compute_zscore_outliers,
        params={"zscore_threshold": 3.0, "ddof": 1},
    ),
    "mod_zscore": OutlierMethodConfig(
        compute=_compute_modified_z_outliers,
        params={"mad_factor": None, "mod_zscore_threshold": 3.5, "ddof": 1},
    ),
}
"""
Dictionary mapping outlier-detector identifiers to their configs.

Identifiers:
  - "sd":          Standard Deviation
  - "mad":         Median Absolute Deviation
  - "iqr":         Interquartile range
  - "zscore":      Z-score
  - "mod_zscore":  Modified Z-score

Example:
    .. code-block::

        # (1): Anomaly detection with standard deviation
        # Access the dict of parameters
        sd_param_dict = bstats.outlier_method["sd"].params

        # Predict the anomalous cycle using standard dev method
        # and the corresponding stats parameters
        (SD_outlier_dV_index,
         SD_min_limit_dV,
         SD_max_limit_dV) = bstats.outlier_method["sd"].compute(
            df_max_dV["max_diff"],
            **sd_param_dict)

        # (2): Anomaly detection with MAD
        mad_param_dict = bstats.outlier_method["mad"].params

        (MAD_outlier_index_dV,
         MAD_min_limit_dV,
         MAD_max_limit_dV) = bstats.outlier_method["mad"].compute(
            df_max_dV["max_diff"],
            **mad_param_dict)

        # To update the statistical parameters or threshold
        # Create a copy of the default dict parameter
        mad_param_dict_const = bstats.outlier_method["mad"].params.copy()

        # Update the dict value to be 1.4826
        mad_param_dict_const["mad_factor"] = 1.4826

        # Use the updated param_dict in the outlier method
        (MAD_outlier_index_dV_const,
         MAD_min_limit_dV_const,
         MAD_max_limit_dV_const) = bstats.outlier_method["mad"].compute(
            df_max_dV["max_diff"],
            **mad_param_dict_const)
        """
# ----------------------------------------------------------------------------

def calculate_zscore(
    df_variable: pd.Series|np.ndarray) -> pd.Series|np.ndarray:
    """
    Calculate the Z-score of the selected feature.

    Args:
        df_variable (pd.Series | np.ndarray): Selected feature.

    Returns:
        pd.Series|np.ndarray: Z-score of selected feature.
    """
    # Calculate the mean and std deviation
    # from the max diff feature
    feature_mean = np.mean(df_variable)
    feature_std = np.std(df_variable, ddof=1)
    print(f"Feature mean: {feature_mean}")
    print(f"Feature std: {feature_std}")

    feature_zscore = (df_variable - feature_mean)/feature_std
    print("After Z-transformation, feature mean should be close to 0 "
            + "and feature std should be close to 1.")
    print(f"Zscore feature mean: {np.mean(feature_zscore)}")
    print(f"Zscore feature std: {np.std(feature_zscore, ddof=1)}")

    return feature_zscore

def calculate_feature_stats(
    df_variable: ArrayLike,
    new_col_name: str=None) -> pd.DataFrame:
    """
    Calculate descriptive statistics for a given feature.

    This function computes the mean, minimum, maximum, and standard
    deviation of the input variable. The results are returned as a
    pandas DataFrame, optionally labeled with a custom column name.

    Args:
        df_variable (pd.Series | np.ndarray): Input data series or
            array for which statistics are calculated.
        new_col_name (str, optional): Optional name for the resulting
            column in the output DataFrame. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with statistics (max, min, mean, std)
        as rows. If ``new_col_name`` is provided, the statistics are
        stored under that column name.
    """
    mean_var = np.mean(df_variable)
    print(f"Feature mean: {mean_var}")

    max_var = np.max(df_variable)
    print(f"Feature max: {max_var}")

    min_var = np.min(df_variable)
    print(f"Feature min: {min_var}")

    std_var = np.std(df_variable, ddof=1)
    print(f"Feature std: {std_var}")
    print("*"*70)

    feature_dict = {
        "max": [np.round(max_var, 4)],
        "min": [np.round(min_var, 4)],
        "mean": [np.round(mean_var, 4)],
        "std": [np.round(std_var, 4)],
    }

    df_feature_stats = pd.DataFrame.from_dict(feature_dict).T

    if new_col_name:
        df_feature_stats.columns = [new_col_name]

    return df_feature_stats