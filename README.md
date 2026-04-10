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

The system runs a **7-stage modular pipeline** that takes raw vessel sensor data and produces:

- Detected anomalies stored in a **SQLite database**
- Diagnostic plots covering time-series overlays, spatial GPS tracks, cluster analysis, and anomaly structures
- A combined ensemble that merges results from multiple detection methods

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
    └─ Full mode: reloads raw CSV; Patch mode (default): updates from cached stats
       Output: results/01_eda/{vessel_id}/{vessel_id}_metadata.csv

Stage 03 — Data Cleaning
       Output: results/03_cleaning/

Stage 04 — Clustering
       Output: results/04_clustering/

Stage 05 — Anomaly Detection
       Output: results/05_anomaly_detection/

Stage 06 — Classification
       Output: results/06_classification/

Stage 07 — Saving
       Output: results/07_saving/
```

---

## Detection Methods

Stage 05 runs multiple unsupervised anomaly detection methods per operational cluster and combines their outputs into a majority-vote ensemble. The methods and their implementation details are not included in the public repository.

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
│   ├── stage_03_cleaning.py         # [not included in public repo]
│   ├── stage_04_clustering.py       # [not included in public repo]
│   ├── stage_05_anomaly_detection.py # [not included in public repo]
│   ├── stage_06_classification.py   # [not included in public repo]
│   ├── stage_07_saving.py           # [not included in public repo]
│   └── regen_metadata.py            # Standalone metadata regeneration helper
│
├── src/
│   ├── models/                      # [not included in public repo]
│   ├── database/                    # [not included in public repo]
│   ├── eda/
│   │   └── eda_runner.py
│   ├── classification/              # [not included in public repo]
│   ├── saving/                      # [not included in public repo]
│   └── preprocessing/               # [not included in public repo]
│
├── benchmark/
│   └── methods/                     # [not included in public repo]
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

### Diagnostic Plots

Stage 05 generates diagnostic plots under `results/05_anomaly_detection/plots/` covering anomaly score distributions, feature time-series overlays, GPS spatial tracks, cluster breakdowns, and power–speed relationships.

### Database Tables (`anomalies.db`)

| Table | Contents |
|---|---|
| `anomalies` | Detected anomalies with scores, timestamps, and feature values |
| `detection_runs` | Metadata per pipeline run |
| `missing_records` | Records removed during cleaning due to missing feature data |

---

## Evaluation Strategy

The system is fully unsupervised. Evaluation is based on domain constraint recall (known sensor violations), cross-method agreement, and physical plausibility checks against expected vessel operating relationships.

---
