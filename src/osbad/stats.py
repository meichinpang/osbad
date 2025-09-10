"""
The methods outlined in this module implement a suite of statistical methods
to detect anomalies across multiple cycles in the given battery
cycling protocols.


.. code-block::

    import osbad.stats as bstats
"""

import fireducks.pandas as pd
import numpy as np

def calculate_IQR_bound(
    df_variable: pd.Series) -> tuple:
    """
    Calculate the IQR first and third quantile with the corresponding IQR
    lower and upper bounds.

    Args:
        df_variable (pd.Series): Selected feature.

    Returns:
        tuple: IQR first and third quantile with the corresponding IQR
        lower and upper bounds.
    """
    Q1 = df_variable.quantile(0.25)
    print(f"IQR first quantile Q1: {Q1}")

    Q3 = df_variable.quantile(0.75)
    print(f"IQR third quantile Q3: {Q3}")

    IQR = (Q3 - Q1)

    lower_bound = Q1 - 1.5*IQR
    print(f"IQR lower bound: {lower_bound}")

    upper_bound = Q3 + 1.5*IQR
    print(f"IQR upper bound: {upper_bound}")

    return (Q1, Q3, lower_bound, upper_bound)

def calculate_sd_outliers(
    df_variable: pd.Series|np.ndarray) -> tuple:
    """
    Use the 3 standard deviation method to detect outliers.

    Args:
        df_variable (pd.Series | np.ndarray): Selected feature.

    Returns:
        tuple: Potential anomalous cycle index detected by the standard
        deviation method with the calculated lower and upper limits.

    Example::

        (std_outlier_dQ_index,
        SD_min_limit_dQ,
        SD_max_limit_dQ) = sd.calculate_sd_outliers(df_max_dQ["max_diff"])
    """

    # Calculate the mean and std deviation
    # from the max diff feature
    feature_mean = np.mean(df_variable)
    feature_std = np.std(df_variable, ddof=1)
    print(f"SD feature mean: {feature_mean}")
    print(f"SD feature std: {feature_std}")

    # Mix and max limit
    # defined as 3-std deviation from the
    # distribution mean
    SD_min_limit = feature_mean - 3*feature_std
    SD_max_limit = feature_mean + 3*feature_std

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

def calculate_bubble_size_ratio(
    df_variable: pd.Series|np.ndarray) -> pd.Series:
    """
    Calculate the bubble size of the feature in the bubble plot depending
    on the anomaly score by using the feature standardization method.

    Args:
        df_variable (pd.Series | np.ndarray): Selected feature.

    Returns:
        pd.Series: Calculated bubble size of the feature.

    Example::

        df_bubble_size_dQ = sd.calculate_bubble_size_ratio(
            df_variable=df_max_dQ["max_diff_dQ"])

        df_bubble_size_dV = sd.calculate_bubble_size_ratio(
            df_variable=df_max_dV["max_diff"])
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

    scaling_ratio = (df_variable - mean_var)/(std_var)

    return scaling_ratio

def calculate_feature_stats(
    df_variable: pd.Series|np.ndarray,
    new_col_name: str=None) -> pd.DataFrame:
    """
    Calculate the statistics (max, min, mean and standard deviation) of
    the selected feature.

    Args:
        df_variable (pd.Series | np.ndarray): Selected feature.
        new_col_name (str, optional): New column name for the calculated
                                        feature statistics.
                                        Defaults to None.

    Returns:
        pd.DataFrame: Calculated statistics of the selected feature.

    Example::

        max_dV_feature_stats = sd.calculate_feature_stats(
            np.array(df_max_dV["max_diff"]),
            new_col_name="dV")
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

def calculate_MAD_outliers(
    df_variable: pd.Series|np.ndarray,
    MAD_factor: float=None):
    """
    Use Median Absolute Deviation (MAD) method to detect outliers.

    Args:
        df_variable (pd.Series | np.ndarray): Selected feature.
        MAD_factor (float): MAD-factor in the MAD equation.

    Returns:
        tuple: Potential anomalous cycle index detected by the MAD method
        with the calculated lower and upper limits.

    Note::

        One should note that the MAD-factor plays an important role to
        determine the corresponding MAD-score. If the underlying data
        distribution is Gaussian, then we can assume that
        MAD-factor = 1.4826.

        If one would like to relax the assumption about the normality
        of a feature distribution, then MAD-factor can be
        calculated from the reciprocal of the 75th-percentile of a
        standard distribution, which means a distribution with a
        mean of zero and a standard deviation of one).

    Example::

        # If MAD_factor is None, the MAD-factor will be calculated
        # from 1/Q3 of std-distribution
        (MAD_outlier_index_dV_norm_factor,
            MAD_min_limit_dV_norm_factor,
            MAD_max_limit_dV_norm_factor) = sd.calculate_MAD_outliers(
            df_max_dV["max_diff"],
            MAD_factor=1.4826)
    """
    # Calculate the median of the feature
    median = np.median(df_variable)
    print(f"Feature median: {median}")

    # Calculate absolute deviation from the median
    abs_deviations = np.abs(df_variable - median)

    if MAD_factor is None:
        # Transform the distribution to have a mean of zero
        # and std-deviation of one
        mean_var = np.mean(df_variable)
        std_var = np.std(df_variable, ddof=1)
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

    # Calculate MAD-score
    MAD = MAD_factor*np.median(abs_deviations)
    print(f"MAD: {MAD}")

    # Calculate upper MAD limit
    MAD_min_limit = median - 3*MAD
    print(f"MAD min limit: {MAD_min_limit}")

    # Calculate lower MAD limit
    MAD_max_limit = median + 3*MAD
    print(f"MAD max limit: {MAD_max_limit}")

    MAD_outlier_index = np.where(
        (df_variable < MAD_min_limit) |
        (df_variable > MAD_max_limit))

    if isinstance(MAD_outlier_index, tuple):
        # convert tuple into numpy array
        return (MAD_outlier_index[0], MAD_min_limit, MAD_max_limit)
    else:
        return (MAD_outlier_index, MAD_min_limit, MAD_max_limit)

def calculate_modified_zscore_outliers(
    df_variable: pd.Series|np.ndarray,
    MAD_factor: float=None) -> tuple:
    """
    Use modified Z-score method to detect outliers.

    Args:
        df_variable (pd.Series | np.ndarray): Selected feature.
        MAD_factor (float): MAD-factor in the MAD equation.

    Returns:
        tuple: Potential anomalous cycle index detected by the modified
        Zscore outlier detection method with the calculated lower
        and upper limits.

    Example::

        # If MAD_factor is None, the MAD-factor will be calculated
        # from 1/Q3 of std-distribution
        (MOD_Zoutlier_index_dV,
        MOD_Zmin_limit_dV,
        MOD_Zmax_limit_dV) = sd.calculate_modified_zscore_outliers(
            df_max_dV["max_diff"],
            MAD_factor=None)
    """
    # Calculate the median of the feature
    median = np.median(df_variable)
    print(f"Feature median: {median}")

    # Calculate absolute deviation from the median
    abs_deviations = np.abs(df_variable - median)

    if MAD_factor is None:
        # Transform the distribution to have a mean of zero
        # and std-deviation of one
        mean_var = np.mean(df_variable)
        std_var = np.std(df_variable, ddof=1)
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

    # Calculate MAD-score
    MAD = MAD_factor*np.median(abs_deviations)
    print(f"MAD: {MAD}")

    modified_zscore = (df_variable - median)/MAD

    # Modified z-score lower limit
    modified_zmin_limit = -3.5
    print(f"Modified Zmin limit: {modified_zmin_limit}")

    # Modified z-score upper limit
    modified_zmax_limit = 3.5
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

def calculate_zscore_outliers(
    df_variable: pd.Series|np.ndarray) -> tuple:
    """
    Use Z-score method to detect outliers.

    Args:
        df_variable (pd.Series | np.ndarray): Selected feature.

    Returns:
        tuple: Potential anomalous cycle index detected by the Zscore
        method with the calculated lower and upper limits.

    Example::

        (zscore_outlier_index_dV,
            zscore_min_limit,
            zscore_max_limit) = sd.calculate_zscore_outliers(
            df_max_dV["max_diff"])

    """
    # Calculate the mean and std deviation
    # from the max diff feature
    feature_mean = np.mean(df_variable)

    # use unbiased estimator with the bessel correction factor (n-1) when
    # using the std deviation method from numpy (ddof=1)
    feature_std = np.std(df_variable, ddof=1)
    print(f"Feature mean: {feature_mean}")
    print(f"Feature std: {feature_std}")

    feature_zscore = (df_variable - feature_mean)/feature_std
    print("After Z-transformation, feature mean should be close to 0 "
            + "and feature std should be close to 1.")
    print(f"Zscore feature mean: {np.mean(feature_zscore)}")
    print(f"Zscore feature std: {np.std(feature_zscore, ddof=1)}")

    zscore_min_limit = -3
    zscore_max_limit = 3

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


def calculate_IQR_outliers(
    df_variable: pd.Series|np.ndarray):
    """
    Use the Interquartile Range (IQR) method to detect outliers.

    Args:
        df_variable (pd.Series | np.ndarray): Selected feature.

    Returns:
        tuple: Potential anomalous cycle index detected by the IQR method
        with the calculated lower and upper limits.

    Example::

        (IQR_outlier_index_dV,
        IQR_min_limit_dV,
        IQR_max_limit_dV) = sd.calculate_IQR_outliers(
            df_variable=df_max_dV["max_diff"])
    """


    quartiles = np.quantile(
        df_variable,
        [0.25, 0.5, 0.75])
    Q1 = quartiles[0]
    Q3 = quartiles[2]
    IQR = Q3 - Q1

    IQR_min_limit = Q1 - 1.5*IQR
    IQR_max_limit = Q3 + 1.5*IQR

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


