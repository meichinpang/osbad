"""
The methods outlined in this module implement statistical feature
transformations before passing the input features into the anomaly detection
methods in this benchmarking study.


.. code-block::

    from osbad.scaler import CycleScaling
"""
import fireducks.pandas as pd
import numpy as np


class CycleScaling:
    """
    Implement statistical feature transformation methods on the selected
    dataframe.

    .. code-block::

        # Only select the relevant features for the models while excluding
        # the true labels from the benchmarking dataset.
        df_selected_cell_without_labels = df_selected_cell[
            ["cell_index",
             "cycle_index",
             "discharge_capacity",
             "voltage"]].reset_index(drop=True)

        # Instantiate the CycleScaling class for dataset without labels
        scaler = CycleScaling(
            df_selected_cell=df_selected_cell_without_labels)

    .. note::

        True labels are stored in the dataframe as
        ``df_selected_cell["outlier"]``.
    """
    def __init__(
        self,
        df_selected_cell: pd.DataFrame):
        """

        Args:
            df_selected_cell (pd.DataFrame): The dataframe of selected cell.
        """
        self._df_selected_cell = df_selected_cell

    def median_IQR_scaling(
        self,
        variable: str,
        validate: bool=False) -> pd.DataFrame:
        """
        Implement median-IQR-scaling on the selected feature from the
        dataframe to help with the marginal histogram separation of
        abnormal cycles from normal cycles.

        Args:
            variable (str): Variable or feature to implement with the
                            median-IQR-scaling method.
            validate (bool, optional): Validate and visually inspect if the
                                       scaling are performed correctly.
                                       If True, this method will return
                                       additional columns with intermediate
                                       calculation step results.
                                       Defaults to False.

        Returns:
            pd.DataFrame: Scaled variable with the corresponding cycle index.

        Example::

            # Instantiate the CycleScaling class
            scaler = CycleScaling(
                df_selected_cell=df_selected_cell_without_labels)

            # Implement median IQR scaling on the discharge capacity data
            df_capacity_med_scaled = scaler.median_IQR_scaling(
                variable="discharge_capacity",
                validate=True)
        """
        unique_cycle_count = self._df_selected_cell["cycle_index"].unique()

        var_scaled_list = []

        for k, cycle_count in enumerate(unique_cycle_count):
            df_cycle = self._df_selected_cell[
                self._df_selected_cell["cycle_index"] == cycle_count].copy()

            # Implement custom IQR scaling for each cycle
            IQR = (df_cycle[variable].quantile(0.75)
                - df_cycle[variable].quantile(0.25))

            median = np.median(df_cycle[variable])
            median_square = np.median(df_cycle[variable])**2

            median_square_IQR_ratio = median_square/IQR

            # use median_square to keep the physical unit consistent
            var_scaled = (df_cycle[variable] - median_square_IQR_ratio)
            # var_scaled = (df_cycle[variable] - median)**2/IQR

            if validate:

                var_scaled_dict = {
                    "variable": df_cycle[variable],
                    "cycle_median": median,
                    "median_square": median_square,
                    "IQR": IQR,
                    "median_square_IQR_ratio": median_square_IQR_ratio,
                    "scaled_variable": var_scaled,
                    "cycle_index": cycle_count
                }

                df_scaled = pd.DataFrame.from_dict(var_scaled_dict)
                var_scaled_list.append(df_scaled)
            else:
                var_scaled_dict = {
                    "scaled_variable": var_scaled,
                    "cycle_index": cycle_count
                }

                df_scaled = pd.DataFrame.from_dict(var_scaled_dict)
                var_scaled_list.append(df_scaled)

        df_variable_median_scaled = pd.concat(var_scaled_list)
        new_col_name = "scaled_" + variable
        df_variable_median_scaled = df_variable_median_scaled.rename(
            columns={
                "variable": variable,
                "scaled_variable": new_col_name})

        return df_variable_median_scaled


    def calculate_max_diff_per_cycle(
        self,
        df_scaled: pd.DataFrame,
        variable_name: str) -> pd.DataFrame:
        """
        Calculate the maximum feature difference per cycle to transform
        collective anomalies of a given cycle into cycle-wise point anomalies.
        If continuous abnormal voltage and current measurements are recorded
        in a cycle, the specific cycle will be labelled as an anomalous cycle.

        Args:
            df_scaled (pd.DataFrame): The dataframe with scaled feature.
            variable_name (str): Name of the feature or variable in the
                                 dataframe.

        Returns:
            pd.DataFrame: Maximum feature difference per cycle with the
            corresponding cycle index.

        .. Note::

            While the cycle index at the beginning may be the same as the
            natural index of the dataframe, do not use the natural index of
            the dataframe to label the cycle number. This is because the
            natural index may change if some anomalous cycless are removed
            from the dataframe.

        Example::

            # maximum scaled capacity difference per cycle
            df_max_dQ = scaler.calculate_max_diff_per_cycle(
                df_scaled=df_capacity_med_scaled,
                variable_name="scaled_discharge_capacity")

            # maximum scaled voltage difference per cycle
            df_max_dV = scaler.calculate_max_diff_per_cycle(
                df_scaled=df_voltage_med_scaled,
                variable_name="scaled_voltage")
        """
        # Get the unique cycle count from the dataset
        unique_cycle_count = df_scaled["cycle_index"].unique()

        max_diff_list = []

        for k, cycle_count in enumerate(unique_cycle_count):
            # print(cycle_count)

            # Drop the first and last 10 data point
            df_cycle = df_scaled[
                df_scaled["cycle_index"] == cycle_count].iloc[10:-10]

            # Calculate the pointwise feature difference
            feature_diff = np.diff(df_cycle[variable_name])

            # Replace any inf or nan values with zeros
            updated_diff = np.where(
                (np.isinf(feature_diff) | np.isnan(feature_diff)),
                0,
                feature_diff)

            # Calculate max diff per cycle
            max_diff = np.max(updated_diff)

            # Calculate log max diff per cycle
            log_max_diff = np.log(max_diff)

            # Create a dict to keep track of max_dV and
            # the corresponding cycle index
            diff_dict = {
                "max_diff": [max_diff],
                "log_max_diff": [log_max_diff],
                "cycle_index": cycle_count
            }

            df_max_diff = pd.DataFrame.from_dict(diff_dict)
            max_diff_list.append(df_max_diff)


        df_max_diff_all = pd.concat(
            max_diff_list,
            axis=0,
            ignore_index=True)

        return df_max_diff_all

    def calculate_max_feature_derivative_per_cycle(
        self,
        Xfeature: pd.Series,
        Yfeature: pd.Series,
        cycle_index: pd.Series) -> pd.DataFrame:
        """
        Calculate the derivative of Yfeature and Xfeature (dYdX)

        Args:
            Xfeature (pd.Series): Feature to be considered as denominator.
            Yfeature (pd.Series): Feature to be considered as numerator.
            cycle_index (pd.Series): Cycle index of selected cell.

        Returns:
            pd.DataFrame: Calculate max feature derivative (dYdX) per cycle.
        """

        df_merge_scaled_feature = pd.concat([
            Xfeature,
            Yfeature,
            cycle_index], axis=1)

        # Get the unique cycle count from the dataset
        unique_cycle_count = cycle_index.unique()

        max_diff_list = []

        for k, cycle_count in enumerate(unique_cycle_count):

            # Drop the first and last 10 data point
            df_cycle = (df_merge_scaled_feature[
                df_merge_scaled_feature["cycle_index"] == cycle_count]
                .iloc[10:-10])

            # Calculate the pointwise feature difference
            # Note:
            # df_cycle.iloc[:,1] denotes scaled_voltage (dY = dV)
            # df_cycle.iloc[:,0] denotes scaled_discharge_capacity (dX = dQ)
            numerator_feature_diff = np.diff(df_cycle.iloc[:,1])
            denominator_feature_diff = np.diff(df_cycle.iloc[:,0])

            feature_diff = numerator_feature_diff/denominator_feature_diff

            # Replace any inf or nan values with zeros
            updated_diff = np.where(
                (np.isinf(feature_diff) | np.isnan(feature_diff)),
                0,
                feature_diff)

            # Calculate max diff per cycle
            max_diff = np.max(updated_diff)

            # Calculate log max diff per cycle
            log_max_diff = np.log(max_diff)

            # Create a dict to keep track of max_dV and
            # the corresponding cycle index
            diff_dict = {
                "max_diff": [max_diff],
                "log_max_diff": [log_max_diff],
                "cycle_index": cycle_count
            }

            df_max_diff = pd.DataFrame.from_dict(diff_dict)
            max_diff_list.append(df_max_diff)


        df_max_diff_all = pd.concat(
            max_diff_list,
            axis=0,
            ignore_index=True)

        return df_max_diff_all