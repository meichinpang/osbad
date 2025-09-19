from dataclasses import dataclass
from typing import (
    Callable, Dict, Optional, Tuple, Union, Any)
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Types

ArrayLike = Union[pd.Series, np.ndarray]
"""
Type alias for array-like inputs.

Represents data structures that can be treated as arrays in numerical
and analytical operations. This alias is used for type hints to accept
both pandas Series and NumPy ndarray objects.
"""

# ----------------------------------------------------------------------------
# Method config container (mirrors ModelConfigDataClass)

@dataclass(frozen=True)
class OutlierMethodConfig:
    """
    Immutable configuration for a statistical outlier detector.

    Stores::

      - compute: the detector implementation accepting (x, **params)
      - params: default statistical parameters
    """
    # compute: ComputeFunc
    compute: Callable[[Any], Tuple]
    params: Dict[str, Any]

# ----------------------------------------------------------------------------
# Implementations

def compute_sd_outliers(
    df_variable: ArrayLike,
    k: float = 3.0,
    ddof: int = 1) -> tuple:
    """
    Detect outliers using the standard deviation rule.

    This function computes the mean and standard deviation of the input
    variable and identifies outliers that fall outside the range defined
    by ± ``k`` standard deviations from the mean.

    Args:
        df_variable (ArrayLike): Input data as a pandas Series or NumPy
            ndarray.
        k (float, optional): Number of standard deviations from the mean
            to define outlier thresholds. Defaults to 3.0.
        ddof (int, optional): Delta degrees of freedom for standard
            deviation calculation. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Indices of detected outliers.
            - float: Lower bound (mean - k * std).
            - float: Upper bound (mean + k * std).

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
    SD_min_limit = feature_mean - k*feature_std
    SD_max_limit = feature_mean + k*feature_std

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
    MAD_factor = np.abs(1/Q3_std_distribution)

    return MAD_factor


def compute_mad_outliers(
    df_variable: ArrayLike,
    k: float = 3.0,
    MAD_factor: float=None,
    ddof: int =1) -> Tuple:
    """
    Detect outliers using the Median Absolute Deviation (MAD) method.

    This function identifies outliers in a dataset by computing the
    median, absolute deviations, and applying the MAD thresholding
    rule. By default, the MAD factor is calculated dynamically if not
    provided, ensuring robustness against skewed or non-Gaussian
    distributions. A scaling parameter ``k`` determines how many MADs
    away from the median a value must be to be flagged as an outlier.

    Args:
        df_variable (ArrayLike): Input feature data as a pandas Series
            or NumPy array.
        k (float, optional): Scaling factor for defining thresholds.
            Outliers are flagged if they fall outside
            ``median ± k * MAD``. Defaults to 3.0.
        MAD_factor (float, optional): Scaling factor for MAD. If None,
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
        - The parameter ``k`` plays a similar role to the z-score
          threshold in the standard deviation method.
        - Outliers are flagged if they fall outside
          ``median ± k * MAD``.
        - The MAD-factor plays an important role to determine the
          corresponding MAD-score. If the underlying data distribution is
          Gaussian, then we can assume that MAD-factor = 1.4826.
        - If we would like to relax the assumption about the normality
          of a feature distribution, then MAD-factor can be
          calculated from the reciprocal of the 75th-percentile of a
          standard distribution, which means a distribution with a
          mean of zero and a standard deviation of one).
    """

    # Calculate the median of the feature
    median = np.median(df_variable)
    print(f"Feature median: {median}")

    # Calculate absolute deviation from the median
    abs_deviations = np.abs(df_variable - median)

    if MAD_factor is None:

        MAD_factor = _calculate_mad_factor(
            df_variable,
            ddof)
        # Transform the distribution to have a mean of zero
        # and std-deviation of one
        # mean_var = np.mean(df_variable)
        # std_var = np.std(df_variable, ddof=1)
        # var_zscore = (df_variable - mean_var)/std_var
        # mean_zscore = np.mean(var_zscore)
        # std_zscore = np.std(var_zscore, ddof=1)
        # print(f"Feature z-score mean: {np.round(mean_zscore,2)}")
        # print(f"Feature z-score std. deviation: {np.round(std_zscore,2)}")

        # # Calculate 75th percentile of the standard distribution
        # Q3_std_distribution = np.quantile(var_zscore, 0.75)

        # # MAD-factor: 1/75th percentile of the standard distribution
        # # Here, we use the absolute value
        # MAD_factor = np.abs(1/Q3_std_distribution)

    # Calculate MAD-score
    MAD = MAD_factor*np.median(abs_deviations)
    print(f"MAD: {MAD}")

    # Calculate upper MAD limit
    MAD_min_limit = median - k*MAD
    print(f"MAD min limit: {MAD_min_limit}")

    # Calculate lower MAD limit
    MAD_max_limit = median + k*MAD
    print(f"MAD max limit: {MAD_max_limit}")

    MAD_outlier_index = np.where(
        (df_variable < MAD_min_limit) |
        (df_variable > MAD_max_limit))

    if isinstance(MAD_outlier_index, tuple):
        # convert tuple into numpy array
        return (MAD_outlier_index[0], MAD_min_limit, MAD_max_limit)
    else:
        return (MAD_outlier_index, MAD_min_limit, MAD_max_limit)

def compute_modified_z_outliers(
    df_variable: ArrayLike,
    MAD_factor: float=None,
    threshold = 3.5,
    ddof: int = 1) -> tuple:
    """
    Detect outliers using the Modified Z-Score method.

    This method computes the modified z-score, which is based on the
    median and the Median Absolute Deviation (MAD).

    Args:
        df_variable (ArrayLike): Input feature data as a pandas Series
            or NumPy array.
        MAD_factor (float, optional): Scaling factor for MAD. If None,
            it is estimated automatically. Defaults to None.
        threshold (float, optional): Cutoff for the modified z-score.
            Values with modified z greater than this threshold are flagged as
            outliers. Defaults to 3.5.
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

    if MAD_factor is None:

        MAD_factor = _calculate_mad_factor(
            df_variable,
            ddof)

        # # Transform the distribution to have a mean of zero
        # # and std-deviation of one
        # mean_var = np.mean(df_variable)
        # std_var = np.std(df_variable, ddof=1)
        # var_zscore = (df_variable - mean_var)/std_var
        # mean_zscore = np.mean(var_zscore)
        # std_zscore = np.std(var_zscore, ddof=1)
        # print(f"Feature z-score mean: {np.round(mean_zscore,2)}")
        # print(f"Feature z-score std. deviation: {np.round(std_zscore,2)}")

        # # Calculate 75th percentile of the standard distribution
        # Q3_std_distribution = np.quantile(var_zscore, 0.75)

        # # MAD-factor: 1/75th percentile of the standard distribution
        # # Here, we use the absolute value
        # MAD_factor = np.abs(1/Q3_std_distribution)

    # Calculate MAD-score
    MAD = MAD_factor*np.median(abs_deviations)
    print(f"MAD: {MAD}")

    modified_zscore = (df_variable - median)/MAD

    # Modified z-score lower limit
    modified_zmin_limit = - threshold
    print(f"Modified Zmin limit: {modified_zmin_limit}")

    # Modified z-score upper limit
    modified_zmax_limit = threshold
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
        compute=compute_sd_outliers,
        params={"k": 3.0, "ddof": 1},
    ),
    "mad": OutlierMethodConfig(
        compute=compute_mad_outliers,
        params={"MAD_factor": None, "k": 3.0, "ddof": 1},
    ),
    "modified_z": OutlierMethodConfig(
        compute=compute_modified_z_outliers,
        params={"mad_factor": None, "threshold": 3.5, "ddof": 1},
    ),
}
"""
Dictionary mapping outlier-detector identifiers to their configs.

Identifiers:
  - "sd":          Standard Deviation
  - "mad":         Median Absolute Deviation
  - "modified_z":  Modified Z-score

.. code-block::

    # 1) Use defaults (baseline)
    res = outlier_method["sd"].compute(df["feature"], **outlier_method["sd"].params)
    print(res.idx, res.lower, res.upper)

    # 2) With tuned params from Optuna
    params = outlier_method["modified_z"].hp_space(trial)    # inside an objective
    res = outlier_method["modified_z"].compute(df["feature"], **params)
"""

# ----------------------------------------------------------------------------
# Optional thin wrappers to keep backward-compatible function names

def calculate_feature_stats(
    df_variable: pd.Series|np.ndarray,
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

def calculate_sd_outliers(
        df_variable: ArrayLike) -> Tuple[np.ndarray, float, float]:
    res = outlier_method["sd"].compute(
        df_variable,
        **outlier_method["sd"].params)
    return (res.idx, res.lower, res.upper)

def calculate_MAD_outliers(
    df_variable: ArrayLike,
    MAD_factor: Optional[float] = None
) -> Tuple[np.ndarray, float, float]:
    params = dict(outlier_method["mad"].params)
    if MAD_factor is not None:
        params["mad_factor"] = MAD_factor
    res = outlier_method["mad"].compute(df_variable, **params)
    return (res.idx, res.lower, res.upper)

def calculate_modified_zscore_outliers(
    df_variable: ArrayLike,
    MAD_factor: Optional[float] = None
) -> Tuple[np.ndarray, float, float]:
    params = dict(outlier_method["modified_z"].params)
    if MAD_factor is not None:
        params["mad_factor"] = MAD_factor
    res = outlier_method["modified_z"].compute(
        df_variable, **params)
    return (res.idx, res.lower, res.upper)
