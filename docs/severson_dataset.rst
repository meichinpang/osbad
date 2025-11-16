MIT/Stanford Data-driven Prediction of Battery Cycle Life Dataset
####################################################################

The first dataset used in this project is extracted from the publication by

   Severson, K.A., Attia, P.M., Jin, N. et al. Data-driven prediction of
   battery cycle life before capacity degradation.
   Nat Energy 4, 383–391 (2019). https://doi.org/10.1038/s41560-019-0356-8

The raw dataset was found to be contaminated with different anomalies.
Due to the large number of cells used for the experiments in their works,
we have only used the experimental dataset from 46 cells in this study.
Each cell has an average of 845 cycles, which is more than sufficient for
benchmarking anomaly detection algorithms in the present study.
In addition, we have also enriched the raw dataset by manually
labelling normal (denoted as 0) vs anomalous cycle (denoted as 1)
for each cycle across all 46 cells.

.. image:: docs_figure/outliers_multiple_cells.png
   :height: 650px
   :width: 950 px
   :alt: cell cycling dataset with anomalies from severson dataset
   :align: center

Battery Chemistry Description
==============================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Property
     - MIT/Stanford
   * - Positive electrode
     - LFP (LiFePO₄)
   * - Electrolyte
     - Liquid electrolyte
   * - Negative electrode
     - Graphite
   * - Number of cells
     - 46 cells
   * - Nominal capacity
     - 1.1 Ah
   * - Discharging C-rates
     - 4 C (1C ≈ 1.1 A)
   * - ΔSOC in %
     - 100%
   * - Theoretical voltage limits
     - 2.0 V - 3.6 V
   * - Operating temperature
     - 30 °C
   * - Anomalous data type
     - Discharge-capacity profile
   * - Data source
     - Severson et al.

Train and Test Dataset
==========================

The dataset from 46 cells is split into training dataset
(``train_dataset_severson.db``) and test dataset
(``test_dataset_severson.db``). The features used for anomaly
detection were further extracted and saved in another database,
so that the pipeline can be automated for all cells in a leaner manner.
The same protocols used for creating the training features are applied to
create the test features.

Standard Schema
==========================
* ``test_time``: Experimental test time in [seconds];
* ``cycle_index``: Discharge cycle index of the experiment;
* ``cell_index``: Identifier for the tested cell;
* ``voltage``: Measured voltage during the experiment in [V];
* ``discharge_capacity``: Cell discharge capacity measured in [Ah];
* ``current``: Applied at the given test step measured in [A];
* ``internal_resistance``: Measured internal resistance of the system [Ohm];
* ``temperature``: Recorded temperature during the experiment in [°C];
* ``outlier``: Boolean flag (0/1) marking whether the data point is an
  outlier;

