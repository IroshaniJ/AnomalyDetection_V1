# Twinship Anomaly Detection - Usage Guide

## Quick Start

```bash
# 1. Set up environment (first time only)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the full modular pipeline (recommended)
python pipeline/run_pipeline.py

# 3. Or run the classic single-script detection
python detect_anomalies.py
```

---

## Modular Pipeline (Recommended)

The codebase is organised as a **6-stage pipeline**. Each stage can be run
standalone or chained via the orchestrator.

```
pipeline/
├── run_pipeline.py              ← Full orchestrator
├── stage_01_eda.py              ← Stage 01: Exploratory Data Analysis
├── stage_02_cleaning.py         ← Stage 02: Data Cleaning & Feature Extraction
├── stage_03_clustering.py       ← Stage 03: Operational Mode Clustering
├── stage_04_anomaly_detection.py← Stage 04: SVD Anomaly Detection
├── stage_05_classification.py   ← Stage 05: Anomaly Classification
└── stage_06_saving.py           ← Stage 06: Save to DB + Reports
```

### Run the Full Pipeline

```bash
source .venv/bin/activate

# Default run (all stages)
python pipeline/run_pipeline.py

# Skip EDA (faster re-runs when data hasn't changed)
python pipeline/run_pipeline.py --skip-eda

# Custom settings
python pipeline/run_pipeline.py \
    --data data1.csv \
    --n-clusters 4 \
    --n-components 3 \
    --threshold-pct 95 \
    --model-version clustered_svd_v1 \
    --db anomalies.db
```

**Orchestrator flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data1.csv` | Input CSV file |
| `--skip-eda` | `False` | Skip Stage 01 (EDA) |
| `--only-stage` | `None` | Run only one stage, e.g. `--only-stage 04` |
| `--n-clusters` | `4` | Number of operational mode clusters |
| `--n-components` | `3` | SVD components per cluster |
| `--threshold-pct` | `95.0` | Anomaly threshold percentile |
| `--random-state` | `42` | Random seed |
| `--model-version` | `clustered_svd_v1` | Model artefact tag |
| `--db` | `anomalies.db` | SQLite database path |

### Run Individual Stages

Each stage reads from the previous stage's `results/` folder and can also
be invoked independently with its own `--output-dir` flag.

#### Stage 01 — EDA
```bash
python pipeline/stage_01_eda.py
python pipeline/stage_01_eda.py --data data1.csv --output-dir results/01_eda
```
Outputs: `results/01_eda/` — summary_stats.csv, missing_values.csv,
constraint_violations.csv, plots/

#### Stage 02 — Data Cleaning
```bash
python pipeline/stage_02_cleaning.py
python pipeline/stage_02_cleaning.py --data data1.csv --output-dir results/02_cleaning
```
Outputs: `results/02_cleaning/` — cleaned_data.parquet, feature_matrix.npy,
valid_mask.npy, cleaning_report.csv

#### Stage 03 — Clustering
```bash
python pipeline/stage_03_clustering.py
python pipeline/stage_03_clustering.py --n-clusters 4
```
Outputs: `results/03_clustering/` — cluster_labels.npy, clusterer.pkl,
cluster_summary.csv, cluster_names.json, plots/

#### Stage 04 — Anomaly Detection
```bash
python pipeline/stage_04_anomaly_detection.py
python pipeline/stage_04_anomaly_detection.py --threshold-pct 95 --n-components 3
```
Outputs: `results/04_anomaly_detection/` — predictions.npy, scores.npy,
annotated_data.parquet, detection_summary.csv, plots/

#### Stage 05 — Anomaly Classification
```bash
python pipeline/stage_05_classification.py
python pipeline/stage_05_classification.py --detection-dir results/04_anomaly_detection
```
Outputs: `results/05_classification/` — classified_anomalies.csv,
anomaly_type_breakdown.csv, full_classified_data.parquet, plots/

#### Stage 06 — Saving
```bash
python pipeline/stage_06_saving.py
python pipeline/stage_06_saving.py --db anomalies.db
```
Outputs: `anomalies.db` (updated), `results/06_saving/` — all_anomalies.csv,
pipeline_report.txt, plots/

### Programmatic Use
```python
from pipeline.run_pipeline import run_pipeline

results = run_pipeline({
    'data':          'data1.csv',
    'skip_eda':      True,
    'n_clusters':    4,
    'threshold_pct': 95.0,
})

# Access per-stage results
print(results['04_detection']['n_anomalies'])
print(results['06_saving']['report'])
```

---

## Source Library (`src/`)

```
src/
├── preprocessing/
│   └── data_loader.py          # load_and_clean(), get_feature_columns()
├── eda/
│   └── eda_runner.py           # EDARunner class
├── models/
│   ├── clustering.py           # OperationalModeClusterer, GMMOperationalModeClusterer
│   ├── svd_detector.py         # SVDAnomalyDetector (single-cluster)
│   └── clustered_svd_detector.py # ClusteredSVDAnomalyDetector (primary model)
├── classification/
│   └── anomaly_classifier.py   # AnomalyClassifier — rule + statistical typing
├── saving/
│   └── results_saver.py        # ResultsSaver — DB + CSV + report
└── database/
    └── anomaly_db.py           # AnomalyDatabase — SQLite operations
```

---

## Commands

### Run Detection

```bash
# Activate environment
source .venv/bin/activate

# Basic run (uses defaults)
python detect_anomalies.py

# Custom parameters
python detect_anomalies.py --data data1.csv \
                           --clusters 4 \
                           --components 3 \
                           --threshold-pct 95 \
                           --model-version clustered_svd_v1
```

**Parameters:**
| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data1.csv` | Input data file |
| `--clusters` | `4` | Number of operational mode clusters |
| `--components` | `3` | Number of SVD components per cluster |
| `--threshold-pct` | `95.0` | Percentile for anomaly threshold (95 = top 5% flagged) |
| `--db` | `anomalies.db` | SQLite database path |
| `--model-dir` | `models` | Directory for saved models |
| `--model-version` | `clustered_svd_v1` | Model version tag |

### Analyze Results

```bash
source .venv/bin/activate

# Generate all plots, CSVs, and reports
python analyze_results.py

# Custom output directory
python analyze_results.py --output-dir my_results
```

**Parameters:**
| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data1.csv` | Original data file |
| `--db` | `anomalies.db` | Anomaly database path |
| `--model` | `models/clustered_svd_v1.pkl` | Trained model path |
| `--output-dir` | `results` | Output directory for plots and CSVs |

**Generated Outputs:**
| File | Description |
|------|-------------|
| `anomalies_all.csv` | All detected anomalies |
| `anomalies_summary.csv` | Summary by anomaly type |
| `anomalies_top100.csv` | Top 100 highest-scoring anomalies |
| `anomalies_negative_power.csv` | Negative power violations |
| `anomalies_extreme_trim.csv` | Extreme trim violations |
| `cluster_summary.csv` | Per-cluster statistics |
| `detection_runs.csv` | Detection run history |
| `analysis_report.txt` | Text summary report |
| `cluster_analysis.png` | Operational mode cluster analysis |
| `anomaly_score_distribution.png` | Score histogram and box plot |
| `anomaly_timeline.png` | Anomalies over time |
| `feature_scatter_anomalies.png` | Feature relationships (speed-power, etc.) |
| `anomaly_type_breakdown.png` | Type distribution bar/pie charts |
| `correlation_heatmap.png` | Feature correlation matrix |
| `gps_track_anomalies.png` | Vessel track with anomalies highlighted |
| `top_anomalies_heatmap.png` | Feature values for top anomalies |

### Query Anomaly Database

```bash
source .venv/bin/activate

# View anomaly summary by type
python -c "
from src.database.anomaly_db import AnomalyDatabase
db = AnomalyDatabase('anomalies.db')
print(db.get_anomaly_summary())
"

# View top 20 anomalies
python -c "
from src.database.anomaly_db import AnomalyDatabase
db = AnomalyDatabase('anomalies.db')
print(db.get_all_anomalies(limit=20))
"

# View unreviewed anomalies
python -c "
from src.database.anomaly_db import AnomalyDatabase
db = AnomalyDatabase('anomalies.db')
print(db.get_unreviewed_anomalies(limit=10))
"

# View detection run history
python -c "
from src.database.anomaly_db import AnomalyDatabase
db = AnomalyDatabase('anomalies.db')
print(db.get_detection_runs())
"
```

### Open Database with SQLite CLI

```bash
sqlite3 anomalies.db

# Useful SQLite commands:
.tables                              # List all tables
.schema anomalies                    # Show table structure
SELECT COUNT(*) FROM anomalies;      # Count anomalies
SELECT * FROM anomalies LIMIT 10;    # View first 10 records
SELECT * FROM detection_runs;        # View run history
.quit                                # Exit
```

### Use Saved Model for Inference

```python
from src.preprocessing.data_loader import load_and_clean, preprocess_features
from src.models.svd_detector import SVDAnomalyDetector
import joblib

# Load model and scaler
detector = SVDAnomalyDetector.load('models/svd_v1.pkl')
scaler = joblib.load('models/svd_v1_scaler.pkl')

# Load new data
df = load_and_clean('new_data.csv')
X_scaled, _, valid_mask = preprocess_features(df, scaler=scaler)

# Predict
predictions = detector.predict(X_scaled)
scores = detector.score_samples(X_scaled)

# -1 = anomaly, 1 = normal
anomalies = df[valid_mask][predictions == -1]
```

## Project Structure

```
AnomalyDetection_V1/
│
├── pipeline/                                ← Modular pipeline (entry points)
│   ├── __init__.py
│   ├── run_pipeline.py                      ← Full orchestrator
│   ├── stage_01_eda.py                      ← EDA & data quality
│   ├── stage_02_cleaning.py                 ← Data cleaning & feature extraction
│   ├── stage_03_clustering.py               ← Operational mode clustering
│   ├── stage_04_anomaly_detection.py        ← SVD anomaly detection
│   ├── stage_05_classification.py           ← Anomaly type classification
│   └── stage_06_saving.py                   ← Persist to DB + reports
│
├── src/                                     ← Library (reusable components)
│   ├── preprocessing/
│   │   └── data_loader.py                   # load_and_clean(), get_feature_columns()
│   ├── eda/
│   │   └── eda_runner.py                    # EDARunner
│   ├── models/
│   │   ├── clustering.py                    # K-Means & GMM clusterers
│   │   ├── svd_detector.py                  # Single-cluster SVD detector
│   │   └── clustered_svd_detector.py        # Primary model (clustered SVD)
│   ├── classification/
│   │   └── anomaly_classifier.py            # AnomalyClassifier
│   ├── saving/
│   │   └── results_saver.py                 # ResultsSaver
│   └── database/
│       └── anomaly_db.py                    # AnomalyDatabase (SQLite)
│
├── benchmark/                               ← Alternative method benchmarks
│   ├── methods/
│   │   ├── base_detector.py
│   │   ├── isolation_forest.py              # ClusteredIsolationForest
│   │   └── autoencoder.py                   # ClusteredAutoencoder (PyTorch)
│   ├── benchmark_framework.py
│   └── run_benchmark.py
│
├── results/                                 ← Pipeline outputs (auto-created)
│   ├── 01_eda/
│   ├── 02_cleaning/
│   ├── 03_clustering/
│   ├── 04_anomaly_detection/
│   ├── 05_classification/
│   └── 06_saving/
│
├── models/                                  ← Saved model artefacts (.pkl)
├── data1.csv                                ← Input data
├── anomalies.db                             ← SQLite anomaly database
├── requirements.txt
├── USAGE.md                                 ← This file
│
└── (legacy scripts, kept for compatibility)
    ├── detect_anomalies.py
    ├── analyze_results.py
    ├── deep_analysis.py
    ├── compare_clustering.py
    └── compare_clustering_anomaly.py
```

## Output Interpretation

### Anomaly Types

| Type | Description |
|------|-------------|
| `multivariate_outlier` | Unusual combination of features (SVD reconstruction error) |
| `negative_power` | Engine power < 0 (sensor error) |
| `extreme_trim` | Trim outside ±5m bounds |
| `invalid_draft` | Draft ≤ 0 (sentinel value) |

### Anomaly Scores

- **Lower score** = More normal
- **Higher score** = More anomalous
- Records above the threshold (95th percentile by default) are flagged

## Troubleshooting

**ModuleNotFoundError:**
```bash
source .venv/bin/activate  # Make sure venv is active
```

**Database locked:**
```bash
# Close any other connections (SQLite viewers, other scripts)
```

**Reset database:**
```bash
rm anomalies.db
python detect_anomalies.py  # Creates fresh database
```
