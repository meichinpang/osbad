Tohoku Dataset
################

The second dataset used in this study is contributed by students at Tohoku
University. In this dataset, we are interested in detecting capacity
degradation anomalies in the cycling data of solid-state lithium-ion batteries.
For example, in the figure below, we can observe a sudden drop in the
discharge capacity at the cycle index 79, 429 and 476, which indicates an
anomaly in the battery's performance. This capacity degradation anomaly is an
example of point anomaly that we aim to detect using our proposed OSBAD
framework.

.. image:: docs_figure/iforest_pred_cap_fade_with_outliers_cell_num_1.png
   :height: 450px
   :width: 650 px
   :alt: cell cycling dataset from ``Cell 1``
   :align: center


The dataset consists of cycling data from 10 cells, each identified by a
unique cell index. The cycling data includes measurements such as ``voltage``
and ``discharge_capacity`` over multiple charge-discharge cycles.

Battery Chemistry Description
==============================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Property
     - Tohoku
   * - Positive electrode
     - NMC523 (LiNi₀.₅Co₀.₂Mn₀.₃O₂)
   * - Electrolyte
     - Solid electrolyte (Li₆PS₅Cl)
   * - Negative electrode
     - In/InLi
   * - Number of cells
     - 10 cells
   * - Nominal capacity
     - 100 mAh/g
   * - Discharging C-rates
     - 0.1 C (1C ≈ 233 μA)
   * - ΔSOC in %
     - 100%
   * - Theoretical voltage limits
     - 3.0 V - 4.3 V vs. Li⁺/Li
   * - Operating temperature
     - 25 °C
   * - Anomalous data type
     - Discharge-capacity profile
   * - Data source
     - This paper


Feature Description
====================

.. list-table::
   :header-rows: 1
   :widths: 15 10 35 20 20

   * - Feature
     - Type
     - Description
     - Range/Values
     - Anomaly Relevance
   * - ``discharge_capacity``
     - Float
     - Discharge capacity measured during the discharge phase (mAh)
     - 0.0 - 105.66
     - High - Primary indicator of capacity fade anomalies
   * - ``cycle_index``
     - Integer
     - Sequential charge-discharge cycle number
     - 0 - 499
     - High - Temporal reference for anomaly detection
   * - ``voltage``
     - Float
     - Discharge voltage measured during cycling (V)
     - 1.25 - 3.63
     - Medium - Voltage measurement including anomalies.
   * - ``cell_index``
     - String
     - Unique identifier for each battery cell
     - Cell-specific labels
     - Low - Grouping variable for per-cell analysis
   * - ``outlier``
     - Integer
     - Ground-truth anomaly label (0=normal, 1=anomalous)
     - 0, 1
     - High - True label for validation and evaluation

.. important::

   The ``discharge_capacity`` is the primary feature of interest for detecting
   capacity degradation anomalies, while ``cycle_index`` provides the temporal
   context. The ``voltage`` feature can also provide insights into battery
   health, but is secondary to capacity measurements. The ``cell_index``
   allows for analysis on a per-cell basis, and the ``outlier`` label is used
   for validating the performance of anomaly detection methods.

