# Variable Metadata — Column Reference

Each row in the vessel metadata file (e.g. `grimaldi_metadata.csv`) describes one sensor signal. This document explains what each column means and how its value is calculated.

---

## Columns

### `variable_name`
The standardised name used for the signal (e.g. `Main_Engine_Power_kW`). Where the source file uses a different name, it has been renamed to this standard form.

---

### `raw_column_name`
The original column name exactly as it appears in the source data file (e.g. `ME Shaft Power (kW)`). Identical to `variable_name` when no renaming was applied.

---

### `feature_group`
Which analysis category the signal belongs to: `engine_propulsion`, `navigation`, `draft`, `weather`, or `trip`.

---

### `unit`
Physical unit of measurement (e.g. `kn`, `kW`, `m`, `kg/h`). Empty if unknown.

---

### `column_description`
A short description of what the signal measures (e.g. "Main engine shaft power output").

---

### `physical_limits`
The physically possible range for the signal — values outside this range are engineering impossibilities and indicate sensor errors. Formatted as:
- `"0.0 – 25.0"` — lower and upper bound both known
- `">= 0.0"` — only a lower bound is known
- `"<= 4500.0"` — only an upper bound is known
- `""` — bounds unknown

---

### `normal_operating_range`
The typical range seen while the vessel is underway (speed above 0.5 kn), shown as the 10th and 90th percentile: `"Q10=X,  Q90=Y"`. This excludes port and idle periods so that zeros don't distort the range for signals like power and fuel.

---

### `stable_or_highly_dynamic`
How quickly the signal changes relative to its full range:
- `stable` — changes slowly (e.g. draft, GPS position)
- `dynamic` — changes rapidly (e.g. engine power, speed)

The median step-to-step rate of change is divided by the signal's full range. If this ratio is below 0.0001 the signal is `stable`, otherwise `dynamic`. Steps more than 5 minutes apart are excluded.

$$\text{ratio} = \frac{\text{median}\!\left(\dfrac{|v_{i+1} - v_i|}{\Delta t_i}\right)}{v_{\max} - v_{\min}}$$

---

### `observed_min` / `observed_max`
Minimum and maximum values actually seen in the dataset, rounded to 4 decimal places.

---

### `n_missing`
Raw count of records with no value.

---

### `pct_missing`
Percentage of records with no value for this signal, rounded to 3 decimal places.

---

### `n_consecutive_duplicates`
Number of records where the value is identical to the previous record. A small number is normal. A very large number may mean the sensor stopped transmitting fresh data.

---

### `n_spikes`
Number of readings where the value changes abnormally fast — more than 3× the rate of the 99th-percentile step in the dataset. Only consecutive readings within 5 minutes of each other are compared (larger time gaps are ignored).

---

### `n_frozen_periods`
Total number of data points that fall inside runs of identical consecutive values lasting 20 minutes or more. On a signal that should be changing (such as power or speed), this indicates the sensor has stopped updating and is repeating the last recorded value. A time-based threshold is used rather than a fixed count so that the criterion is consistent across data sources with different sampling rates.

---

### `suspicious_unit_flag`
`True` if the recorded values look too large for the declared unit:
- Declared `m` but maximum exceeds 100 → data is probably in mm or cm
- Declared `kn` but maximum exceeds 80 → data is probably in km/h or m/s

If flagged, check the original data documentation before using the signal.

---

### `seasonal_effects`
`True` if the signal's monthly averages vary by more than 20% of the overall average. This suggests the signal is influenced by season or route, and models may need to account for that. 

---

### `hard_filter`
Rules for values that must be removed before any analysis. These are hard physical impossibilities. Example: `remove_negative; remove_gt_4500.0`.

---

### `soft_warning`
Flags for values that are suspicious but not necessarily wrong — investigate before deciding whether to keep or remove them. Example: `23_phys_violations; 47_rate_spikes`.

---

### `interpolation_allowed`
`True` if all three conditions are met:

| Condition | Threshold | Reason |
|-----------|-----------|--------|
| Signal behaviour | `stable` | Slowly-changing signals (draft, position) can be safely estimated between readings |
| Overall missingness | < 5% of all records | Too many missing values means gaps are structural, not occasional |
| Longest single continuous gap | ≤ 10 minutes | A gap longer than this means the sensor was offline; interpolating across it fabricates data |

Signals that change rapidly (power, speed, fuel) are never interpolated — the true value during any gap is unknown.

---

### `remove_repeated_values`
`True` when any frozen run of 20 minutes or more is detected. Those stuck runs should be replaced with gaps during data cleaning.

---

### `treat_low_values_as_idle`
`True` for signals where zero is a legitimate operating state — the vessel is docked or the engine is off. Signals in this category: engine power, fuel consumption, GPS speed, shaft RPM.

Zero readings for these signals are not anomalies; they represent normal port operations and should be analysed separately from underway data.

---

### `notes`
Free-text notes on how the signal behaves while the vessel is in port or at berth.

---

## Thresholds Used in Calculations

| Setting | Value | Effect |
|---------|-------|--------|
| Minimum frozen run duration | 20 minutes | Shorter identical runs are ignored |
| Spike multiplier | 3.0 × 99th-percentile step | Sets the threshold for flagging a rate spike |
| Maximum gap for spike checks | 300 s | Steps longer than 5 min are skipped |
| Seasonal swing threshold | 20% of overall mean | Triggers `seasonal_effects = True` |
| Interpolation missing cap | < 5% of records | More missing than this → not interpolated |
| Interpolation max single continuous gap | ≤ 10 minutes | Longer gap → not interpolated |
| Repeated-value removal threshold | any freeze ≥ 20 min | Triggers `remove_repeated_values = True` |
| Suspicious unit — metres | max > 100 | Probably mm or cm |
| Suspicious unit — knots | max > 80 | Probably km/h or m/s |
