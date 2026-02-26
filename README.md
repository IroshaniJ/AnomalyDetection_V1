# Twinship — Maritime Vessel Anomaly Detection

Unsupervised anomaly detection system for maritime vessel sensor data, developed as part of the **Twinship** project at [SINTEF](https://www.sintef.no). Analyses time-series data from onboard sensors to detect operational anomalies in propulsion, navigation, and fuel consumption patterns.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
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

## Dataset

| Property | Value |
|---|---|
| File | `data1.csv` |
| Records | 43,280 |
| Features | 15 columns |
| Time range | March 2020 (2020-03-01 → 2020-03-31) |
| Sampling interval | ~60 s median (gaps up to 789 s) |
| Missing data | `Date` column only (0.1%, 45 records) |

### Key Features Used for Detection

| Feature | Description |
|---|---|
| `GPSSpeed_kn` | Vessel speed over ground |
| `Main_Engine_Power_kW` | Main engine output power |
| `Speed_rpm` | Engine shaft RPM |
| `Fuel_Consumption_t_per_day` | Fuel burn rate |
| `Avg_draft_m` | Mean hull draft |
| `Trim_m` | Vessel trim (fore/aft difference) |
| `TrueWindSpeed_kn` | Meteorological wind speed |
| `RelWindSpeed_kn` | Apparent (relative) wind speed |

### Known Data Quality Issues

| Violation | Count | Handling |
|---|---|---|
| `Main_Engine_Power_kW < 0` | 23 | Flagged as `negative_power` |
| `Trim_m` outside ±5 m | 19 | Flagged as `extreme_trim` |
| `DRAFTFWD == -9999` (sentinel) | 18 | Replaced with `NaN` |
| `Avg_draft_m ≤ 0` | 18 | Flagged as `invalid_draft` |

---

## Pipeline Architecture

```
Stage 01 — EDA
    └─ Generates summary statistics, correlation heatmaps, histograms
       Output: eda_out_redo/

Stage 02 — Data Cleaning
    └─ Handles sentinel values, flags constraint violations
    └─ Isolates records with missing feature data → logs to anomalies.db (missing_records table)
    └─ Saves only valid rows to results/02_cleaning/cleaned_data.pkl

Stage 03 — Clustering
    └─ K-Means (k=4) to identify operational modes
    └─ Modes: AtRest_Loaded, HighSpeed, MediumSpeed, AtRest_Unloaded
       Output: results/03_clustering/

Stage 04 — Anomaly Detection
    └─ Per-cluster SVD, Isolation Forest, and/or Autoencoder
    └─ Majority-vote ensemble (≥2/n methods)
    └─ Generates 10 diagnostic plots
       Output: results/04_anomaly_detection/

Stage 05 — Classification
    └─ Labels each anomaly by type (constraint violation vs. multivariate outlier)
       Output: results/05_classification/

Stage 06 — Saving
    └─ Persists anomalies to anomalies.db, exports CSV and HTML report
       Output: results/06_saving/
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
│   ├── stage_02_cleaning.py
│   ├── stage_03_clustering.py
│   ├── stage_04_anomaly_detection.py
│   ├── stage_05_classification.py
│   └── stage_06_saving.py
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
│   ├── 02_cleaning/
│   ├── 03_clustering/
│   ├── 04_anomaly_detection/
│   │   └── plots/                      # 10 diagnostic plots
│   ├── 05_classification/
│   └── 06_saving/
│
└── eda_out_redo/                        # EDA report and plots
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

### Run stage 04 directly with custom settings

```bash
python pipeline/stage_04_anomaly_detection.py \
    --methods svd iforest autoencoder \
    --ae-epochs 100 \
    --threshold-pct 95 \
    --n-clusters 4
```

### All CLI options

| Flag | Default | Description |
|---|---|---|
| `--data` | `data1.csv` | Path to input CSV |
| `--skip-eda` | `False` | Skip Stage 01 EDA |
| `--only-stage` | `None` | Run one stage only (e.g. `04`) |
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

print(f"Ensemble anomalies: {results['stage04']['n_anomalies']}")
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
