# Twinship — Maritime Vessel Anomaly Detection

Unsupervised anomaly detection system for maritime vessel sensor data. Analyses time-series data from sensors to detect operational anomalies in propulsion, navigation, and fuel consumption patterns.

---

## Table of Contents

- [Overview](#overview)
- [Supported Vessels](#supported-vessels)
- [Key Features](#key-features)
- [Pipeline Architecture](#pipeline-architecture)
- [Detection Methods](#detection-methods)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Output Artefacts](#output-artefacts)
- [Evaluation Strategy](#evaluation-strategy)

---

## Overview

The system runs a **6-stage modular pipeline** that takes raw vessel sensor data and produces:

- Detected anomalies stored in a **SQLite database**
- **10 diagnostic plots** covering time-series overlays, spatial GPS tracks, cluster analysis, and sequential anomaly structures
- Per-method comparison across SVD, Isolation Forest, and Autoencoder
- A **majority-vote ensemble** that requires ≥2/3 methods to agree before flagging an anomaly

The approach is fully **unsupervised** — no labelled anomaly data is required.

---

## Supported Vessels

| Vessel | IMO / MMSI | Sampling | Raw format |
|---|---|---|---|
| **Stenaline** (MMSI 9235517) | 9235517 | ~14 s | Monthly CSVs (`9235517_YYYYMM.csv`) |
| **Stenateknik** (MMSI 9685475) | 9685475 | ~14 s | Monthly CSVs (`9685475_YYYYMM.csv`) |
| **Grimaldi** – EUROCARGO GENOVA | — | ~120 s | Excel export (`.xlsx`, 56 columns) |

Vessel detection is automatic: the pipeline inspects the data file path for the keywords `stenaline`, `stenateknik`, or `grimaldi` and applies the correct column mapping and physical limits.

---

## Key Features

The table below lists the exact raw column names from each vessel dataset and the standard pipeline name they are mapped to.

### Shaft Power

| Standard name | Stenaline | Stenateknik | Grimaldi |
|---|---|---|---|
| `Main_Engine_Power_kW` | `PropulsionPowerTotal` | `ME Shaft Power (kW)` | `SHAFT POWER[kW]` |

### Fuel Consumption

| Standard name | Stenaline | Stenateknik | Grimaldi |
|---|---|---|---|
| `Fuel_Consumption_rate` | `FuelMassFlowMETotal` (kg/h) | `ME Fuel Mass Net (kg/hr)` | `ME FLOWMETER FLOW RATE [mt/h]` |

### Shaft RPM

| Standard name | Stenaline | Stenateknik | Grimaldi |
|---|---|---|---|
| `Speed_rpm` | — *(not available)* | `ME Shaft Speed (rpm)` | `SHAFT SPEED [rpm]` |

### Ship Speed

| Standard name | Description | Stenaline | Stenateknik | Grimaldi |
|---|---|---|---|---|
| `GPSSpeed_kn` | Speed over ground (SOG) | `SOG` | `Ship Speed GPS (knot)` | `SPEED OVER GROUND [kn]` |
| `SpeedLog_kn` | Speed through water (STW) | — *(not available)* | `Ship Speed Log (knot)` | `SPEED THROUGH WATER [kn]` |

### Draft

| Standard name | Description | Stenaline | Stenateknik | Grimaldi |
|---|---|---|---|---|
| `DRAFTAFT` | Draft aft | `DraftAftDynamic` | `Draft Aft (m)` | `DRAFTAFT[m]` |
| `DRAFTFWD` | Draft forward | `DraftFwdDynamic` | `Draft Fwd (m)` | `DRAFTFOR[m]` |

### Wind

| Standard name | Description | Stenaline | Stenateknik | Grimaldi |
|---|---|---|---|---|
| `RelWindSpeed_kn` | Apparent wind speed | `AWS` | `Wind Speed Rel. (knot)` | `WIND SPEED_1` |
| `RelWindAngle_deg` | Apparent wind direction | `AWA` | `Wind Dir. Rel. (deg)` | `WIND DIR_1` |

### Wave & Current *(Grimaldi only)*

| Standard name | Raw column |
|---|---|
| `Wave_Height_m` | `WAVE HEIGHT [m]` |
| `Wave_Period_s` | `WAVE PERIOD [sec]` |
| `Swell_Height_m` | `SWELL HEIGHT [m]` |
| `Current_Speed_kn` | `CURRENT SPEED [kn]` |
| `Current_Direction_deg` | `CURRENT DIRECTION [°]` |

Wave and current data are **not available** in the Stenaline or Stenateknik raw exports.

---

## Pipeline Architecture

```
Stage 01 — EDA
    └─ Generates summary statistics, correlation heatmaps, histograms
       Output: results/01_eda/

Stage 02 — Generate Variable Metadata
    └─ Derives per-variable metadata (unit, physical limits, hard/soft filter bounds)
    └─ Full mode: reloads raw CSV and re-runs Phase 7 of EDARunner
    └─ Patch mode (default): updates unit/limits from cached univariate_report.csv
       Output: results/01_eda/{vessel_id}/{vessel_id}_metadata.csv

Stage 03 — Data Cleaning
    └─ Handles sentinel values, flags constraint violations
    └─ Isolates records with missing feature data → logs to anomalies.db (missing_records table)
    └─ Saves only valid rows to results/03_cleaning/cleaned_data.pkl

Stage 04 — Clustering
    └─ K-Means (k=4) to identify operational modes
    └─ Modes: AtRest_Loaded, HighSpeed, MediumSpeed, AtRest_Unloaded
       Output: results/04_clustering/

Stage 05 — Anomaly Detection
    └─ Per-cluster SVD, Isolation Forest, and/or Autoencoder
    └─ Majority-vote ensemble (≥2/n methods)
    └─ Generates 10 diagnostic plots
       Output: results/05_anomaly_detection/

Stage 06 — Classification
    └─ Labels each anomaly by type (constraint violation vs. multivariate outlier)
       Output: results/06_classification/

Stage 07 — Saving
    └─ Persists anomalies to anomalies.db, exports CSV and HTML report
       Output: results/07_saving/
```

---

## Detection Methods

### Clustered SVD (Primary)
Fits a Truncated SVD within each operational cluster. Points with reconstruction error above the 95th percentile are flagged.

- Captures mode-specific normal behaviour (e.g. zero power at berth is normal, not anomalous)
- Fast (~0.2 s fit time)
- **Recall on known violations: 62.5%**

### Clustered Isolation Forest
Tree-based outlier detection per operational cluster using `contamination=5%`.

- No distance calculations — robust to high-dimensional data
- **Recall on known violations: 54.2%**

### Clustered Autoencoder (PyTorch)
Neural network autoencoder per cluster:
```
Encoder: input_dim → 16 → 8 → latent_dim(3)
Decoder: latent_dim → 8 → 16 → input_dim
Loss: MSE reconstruction error
```
- Captures non-linear feature relationships
- **Recall on known violations: 58.3%**

### Ensemble (Majority Vote)
A point is flagged as an anomaly only if **≥2 out of 3 methods** agree. This reduces false positives while maintaining sensitivity to true anomalies.

---

## Results

Full run with all three methods (50 AE epochs):

| Method | Anomalies | Rate | Recall |
|---|---|---|---|
| SVD | 2,165 | 5.00% | 62.5% |
| Isolation Forest | 2,165 | 5.00% | 54.2% |
| Autoencoder | 2,165 | 5.00% | 58.3% |
| **Ensemble (≥2/3)** | **1,723** | **3.98%** | **54.2%** |

---

## Project Structure

```
AnomalyDetection_V1/
│
├── data1.csv                        # Raw vessel sensor data
├── anomalies.db                     # SQLite: anomalies, detection_runs, missing_records
├── requirements.txt
│
├── pipeline/
│   ├── run_pipeline.py              # Full pipeline orchestrator (CLI + programmatic)
│   ├── stage_01_eda.py
│   ├── stage_02_gen_metadata.py     # Generate per-vessel variable metadata CSVs
│   ├── stage_03_cleaning.py
│   ├── stage_04_clustering.py
│   ├── stage_05_anomaly_detection.py
│   ├── stage_06_classification.py
│   ├── stage_07_saving.py
│   └── regen_metadata.py            # Standalone metadata regeneration helper
│
├── src/
│   ├── models/
│   │   ├── clustered_svd_detector.py   # Primary SVD anomaly detector
│   │   ├── svd_detector.py
│   │   └── clustering.py               # K-Means operational mode clustering
│   ├── database/
│   │   └── anomaly_db.py               # SQLite interface (anomalies + missing records)
│   ├── eda/
│   │   └── eda_runner.py
│   ├── classification/
│   │   └── anomaly_classifier.py
│   ├── saving/
│   │   └── results_saver.py
│   └── preprocessing/
│       └── data_loader.py
│
├── benchmark/
│   └── methods/
│       ├── base_detector.py
│       ├── isolation_forest.py         # ClusteredIsolationForest
│       └── autoencoder.py              # ClusteredAutoencoder (PyTorch)
│
├── results/                            # Generated outputs (gitignored)
│   ├── 01_eda/
│   ├── 03_cleaning/
│   ├── 04_clustering/
│   ├── 05_anomaly_detection/
│   │   └── plots/                      # 10 diagnostic plots
│   ├── 06_classification/
│   └── 07_saving/
│
└── eda_out_redo/                        # Legacy EDA report and plots
```

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone <repo-url>
cd AnomalyDetection_V1

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install torch matplotlib seaborn plotly
```

---

## Usage

### Run the full pipeline (all 3 detection methods)

```bash
python pipeline/run_pipeline.py --methods svd iforest autoencoder
```

### Run with SVD only (fastest)

```bash
python pipeline/run_pipeline.py --methods svd
```

### Skip EDA on re-runs

```bash
python pipeline/run_pipeline.py --skip-eda --methods svd iforest autoencoder
```

### Run a single stage

```bash
python pipeline/run_pipeline.py --only-stage 04 --methods svd iforest autoencoder
```

### Run stage 05 directly with custom settings

```bash
python pipeline/stage_05_anomaly_detection.py \
    --methods svd iforest autoencoder \
    --ae-epochs 100 \
    --threshold-pct 95 \
    --n-clusters 4
```

### All CLI options

| Flag | Default | Description |
|---|---|---|
| `--data` | `data1.csv` | Path to input CSV (vessel detected from path) |
| `--skip-eda` | `False` | Skip Stage 01 EDA |
| `--skip-metadata` | `False` | Skip Stage 02 Generate Metadata |
| `--metadata-full-mode` | `False` | Stage 02: reload raw CSV instead of patch mode |
| `--only-stage` | `None` | Run one stage only (e.g. `05`) |
| `--methods` | `svd iforest autoencoder` | Detection methods to use |
| `--n-clusters` | `4` | Number of operational clusters |
| `--n-components` | `3` | SVD latent components |
| `--threshold-pct` | `95.0` | Anomaly threshold percentile |
| `--ae-epochs` | `50` | Autoencoder training epochs |
| `--ae-encoding-dim` | `3` | Autoencoder latent dimension |
| `--if-n-estimators` | `100` | Isolation Forest trees |
| `--db` | `anomalies.db` | SQLite database path |
| `--random-state` | `42` | Random seed |

### Programmatic use

```python
from pipeline.run_pipeline import run_pipeline

results = run_pipeline({
    'skip_eda':      True,
    'methods':       ['svd', 'iforest', 'autoencoder'],
    'threshold_pct': 95.0,
    'ae_epochs':     50,
})

print(f"Ensemble anomalies: {results['05_detection']['n_anomalies']}")
```

---

## Output Artefacts

### Diagnostic Plots (`results/04_anomaly_detection/plots/`)

| File | Description |
|---|---|
| `score_distributions.png` | Anomaly score histograms per method with p95 threshold |
| `recall_comparison.png` | Per-method and ensemble recall bar chart |
| `method_agreement_heatmap.png` | Pairwise method agreement % |
| `timeseries_all_features.png` | All 8 features over time with anomaly overlay |
| `cluster_anomaly_scatter.png` | Key feature-pair scatters coloured by operational cluster |
| `anomaly_feature_distributions.png` | Box plots: Normal vs Anomaly per feature |
| `anomaly_sequence_structure.png` | Run-length spans, histogram, and rolling 1-hour anomaly rate |
| `spatial_anomalies.png` | GPS vessel track coloured by cluster (left) and anomaly score (right) |
| `anomaly_rate_by_cluster.png` | Anomaly count and rate per operational mode (bar + pie) |
| `power_speed_anomalies.png` | Speed vs Power and Speed vs Fuel scatter with anomaly score intensity |

Paths are under `results/05_anomaly_detection/plots/`.

### Database Tables (`anomalies.db`)

| Table | Contents |
|---|---|
| `anomalies` | All detected anomalies with scores, timestamps, feature values, and type classification |
| `detection_runs` | Metadata per pipeline run (method, threshold, anomaly count, recall) |
| `missing_records` | Records removed during cleaning due to missing feature data |

---

## Evaluation Strategy

Since no labelled anomaly data is available, evaluation uses:

1. **Domain constraint recall** — checks if the model catches the 78 EDA-identified constraint violations (negative power, extreme trim, invalid draft)
2. **Cross-method agreement** — anomalies detected by multiple independent methods are more likely to be genuine
3. **Temporal clustering** — genuine sensor failures tend to be persistent; isolated single-point flags may be noise
4. **Physical plausibility** — flagged points should deviate from the known power–speed relationship (r = 0.95)

---
