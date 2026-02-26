# Copilot Instructions for Twinship Anomaly Detection

## Project Overview
Maritime vessel anomaly detection system for the **Twinship** project (SINTEF). Analyzes time-series sensor data from ships to detect operational anomalies in propulsion, navigation, and fuel consumption patterns.

## Dataset Summary
- **Source**: [data1.csv](../data1.csv) — 43,280 records, 15 features, ~5MB
- **Time range**: March 2020 (2020-03-01 to 2020-03-31)
- **Sampling**: ~60 seconds median interval (gaps up to 789 sec exist)
- **Missing data**: Only `Date` column has missing values (0.1%, 45 records)

## Feature Reference

| Feature | Description | Range | Notes |
|---------|-------------|-------|-------|
| `GPSSpeed_kn` | Vessel speed | 0–17.2 kn | Mean 6.6 kn, bimodal (at-rest vs moving) |
| `GPS_LAT`, `GPS_LON` | Position | 51–60°N, -0.25–27°E | High correlation (0.89) suggests route pattern |
| `Main_Engine_Power_kW` | Engine output | -31 to 4072 kW | **23 negative values are anomalies** |
| `Speed_rpm` | Engine RPM | 0–728 | Correlates strongly with power (0.89) |
| `Fuel_Consumption_t_per_day` | Fuel rate | 0–90.3 t/day | Heavy right skew, max is outlier |
| `DRAFTAFT`, `DRAFTFWD` | Hull draft | mm | **18 sentinel values (-9999)** |
| `Avg_draft_m` | Mean draft | -4.99 to 8.2 m | **18 non-positive values are anomalies** |
| `Trim_m` | Vessel trim | -5.1 to 10.1 m | **19 values outside ±5m bounds** |
| `TrueWindSpeed_kn` | Meteorological wind | 0–14.7 kn | |
| `RelWindSpeed_kn` | Apparent wind | 0–20.7 kn | Correlates with speed (0.70) |

## Critical Data Quality Issues
From EDA constraint checks — address these before modeling:

```python
# Known anomalies to handle in preprocessing
CONSTRAINT_VIOLATIONS = {
    'Main_Engine_Power_kW < 0': 23,      # 0.05% — sensor errors
    'Trim_m outside [-5, 5]': 19,        # 0.04% — extreme trim
    'DRAFTFWD <= 0': 18,                 # Sentinel -9999 values
    'Avg_draft_m <= 0': 18,              # Derived from bad draft
}
```

## Key Correlations for Anomaly Detection
Use these relationships to detect deviations:

| Relationship | Correlation | Use Case |
|--------------|-------------|----------|
| `GPSSpeed_kn` ↔ `Main_Engine_Power_kW` | **0.95** | Power-speed anomalies (hull fouling, loading) |
| `Main_Engine_Power_kW` ↔ `Speed_rpm` | 0.89 | Engine efficiency monitoring |
| `Main_Engine_Power_kW` ↔ `Fuel_Consumption_t_per_day` | **0.72** | Fuel efficiency anomalies |
| `TrueWindSpeed_kn` ↔ `RelWindSpeed_kn` | 0.85 | Wind sensor validation |
| `DRAFTAFT` ↔ `Avg_draft_m` | 0.98 | Draft sensor consistency |

## Recommended Preprocessing Pipeline

```python
import pandas as pd
import numpy as np

def load_and_clean(filepath='data1.csv'):
    df = pd.read_csv(filepath, parse_dates=['Date'])
    
    # Handle sentinel values
    df.loc[df['DRAFTFWD'] == -9999, 'DRAFTFWD'] = np.nan
    
    # Flag constraint violations
    df['anomaly_negative_power'] = df['Main_Engine_Power_kW'] < 0
    df['anomaly_extreme_trim'] = df['Trim_m'].abs() > 5
    df['anomaly_draft'] = df['Avg_draft_m'] <= 0
    
    # Derive operational state
    df['is_moving'] = df['GPSSpeed_kn'] > 0.5
    
    return df
```

## EDA Artifacts
Reference visualizations in [eda_out_redo/](../eda_out_redo/):
- `plots/corr_heatmap.png` — Feature correlation matrix
- `plots/hist_*.png` — Distribution histograms
- `plots_extra/power_vs_speed.png` — Key relationship scatter
- `plots_extra/gps_track.png` — Vessel route visualization
- `report.html` — Full EDA report

## Anomaly Detection Approaches
Based on data characteristics (unsupervised — no labeled anomalies available):

1. **Clustering first**: Identify operational modes (at-rest, slow, medium, high speed)
2. **Per-cluster SVD**: Apply SVD anomaly detection within each cluster
3. **Constraint rules**: Catch known violations (negative power, extreme trim, invalid draft)
4. **Combined scoring**: Merge SVD anomalies with constraint-based flags

## Recommended Algorithms

### Primary Approach: Clustered SVD

```python
# Step 1: Cluster into operational modes using K-Means
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
import numpy as np

def clustered_svd_anomaly_detection(df, feature_cols, n_clusters=4, n_components=3, threshold_percentile=95):
    """
    Detect anomalies using clustering + per-cluster SVD.
    
    Approach:
    1. Cluster data into operational modes (at-rest, slow, medium, high speed)
    2. Fit SVD within each cluster to learn mode-specific normal patterns
    3. Anomalies are points with high reconstruction error in their cluster
    """
    X = df[feature_cols].values
    
    # Step 1: Cluster into operational modes
    cluster_scaler = MinMaxScaler()
    X_scaled_cluster = cluster_scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled_cluster)
    
    # Step 2: Per-cluster SVD anomaly detection
    all_scores = np.zeros(len(X))
    all_thresholds = {}
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        X_cluster = X[mask]
        
        # Min-max scale within cluster
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        # SVD decomposition
        svd = TruncatedSVD(n_components=min(n_components, X_scaled.shape[1]-1), random_state=42)
        X_reduced = svd.fit_transform(X_scaled)
        X_reconstructed = svd.inverse_transform(X_reduced)
        
        # Reconstruction error
        errors = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
        all_scores[mask] = errors
        all_thresholds[cluster_id] = np.percentile(errors, threshold_percentile)
    
    # Step 3: Flag anomalies (per-cluster threshold)
    anomalies = np.zeros(len(X), dtype=bool)
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        anomalies[mask] = all_scores[mask] > all_thresholds[cluster_id]
    
    return anomalies, all_scores, cluster_labels, all_thresholds

# Usage
FEATURE_COLS = [
    'GPSSpeed_kn', 'Main_Engine_Power_kW', 'Speed_rpm',
    'Fuel_Consumption_t_per_day', 'Avg_draft_m', 'Trim_m',
    'TrueWindSpeed_kn', 'RelWindSpeed_kn'
]
anomalies, scores, clusters, thresholds = clustered_svd_anomaly_detection(df, FEATURE_COLS)
```

**Why Clustered SVD:**
- Different operational modes have different normal patterns
- At-rest vessel: zero power/speed is normal
- Cruising vessel: high power/speed is normal
- Per-cluster SVD captures mode-specific relationships
- Reduces false positives from mode transitions

**Typical Clusters Identified:**
| Cluster | Description | Characteristics |
|---------|-------------|-----------------|
| Mode_0_AtRest | Vessel at berth | Speed ≈ 0, Power ≈ 0 |
| Mode_1_SlowSpeed | Maneuvering | Speed < 3 kn, Variable power |
| Mode_2_MediumSpeed | Transit | Speed 3-10 kn |
| Mode_3_HighSpeed | Cruising | Speed > 10 kn, High power |

**Alternative methods (for comparison):**
```python
# Statistical methods (IMPLEMENTED)
from benchmark.methods.isolation_forest import ClusteredIsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# Deep learning for time-series (IMPLEMENTED)
from benchmark.methods.autoencoder import ClusteredAutoencoder  # PyTorch-based
```

**Model Selection Guidelines:**
- Start with **Clustered SVD** — captures mode-specific normal behavior (primary method)
- **Augment with Clustered Autoencoder** — PyTorch-based deep learning for complex multivariate patterns
- **Augment with Clustered Isolation Forest** — tree-based outlier detection per operational mode
- Use **LOF** for local density-based anomalies (e.g., unusual power at specific speeds)
- Apply **LSTM-based** methods if temporal sequence patterns matter

### Isolation Forest (Implemented)

A **Clustered Isolation Forest** approach is implemented for comparison:

```python
from benchmark.methods.isolation_forest import ClusteredIsolationForest

# Isolation Forest per cluster with contamination=5%
detector = ClusteredIsolationForest(
    n_clusters=4,
    contamination=0.05,
    n_estimators=100,
    threshold_percentile=95.0
)
detector.fit(X, feature_names)
predictions = detector.predict(X)  # -1 = anomaly
scores = detector.score_samples(X)  # isolation score (lower = more anomalous)
```

**When to use Isolation Forest:**
- Fast baseline comparison (tree-based, no distance calculations)
- Works well with high-dimensional data
- Robust to irrelevant features

### Deep Learning Augmentation (Implemented)

The SVD approach is augmented with a **Clustered Autoencoder** for unsupervised anomaly scoring:

```python
from benchmark.methods.autoencoder import ClusteredAutoencoder

# PyTorch Autoencoder architecture per cluster:
# Encoder: input_dim → 16 → 8 → 3 (latent)
# Decoder: 3 → 8 → 16 → input_dim
# Training: 100 epochs, MSE loss, Adam optimizer

detector = ClusteredAutoencoder(
    n_clusters=4,
    latent_dim=3,
    threshold_percentile=95.0
)
detector.fit(X, feature_names)
predictions = detector.predict(X)  # -1 = anomaly
scores = detector.score_samples(X)  # reconstruction error
```

**Benchmark Results (All Methods):**
| Method | Recall on Known Violations | Fit Time | Agreement with SVD |
|--------|---------------------------|----------|-------------------|
| Clustered SVD | 62.5% | 0.22s | - |
| Clustered Isolation Forest | 54.2% | 1.05s | 94.0% |
| Clustered Autoencoder | 58.3% | 43.1s | 94.0% |

**When to use Autoencoder over SVD:**
- Complex non-linear relationships between features
- Higher-dimensional data where linear SVD may underfit
- When ensemble voting (SVD + Autoencoder agreement) is desired

**Ensemble approach (recommended for production):**
```python
# Flag as anomaly if BOTH methods agree
ensemble_anomaly = (svd_predictions == -1) & (ae_predictions == -1)
# Or use voting: anomaly if majority of methods flag it
```

## Evaluation Strategy (Unsupervised)

Without labeled data, use these approaches:

```python
# 1. Domain constraint validation
def evaluate_known_anomalies(df, predictions):
    """Check if model catches EDA-identified issues"""
    known_bad = (
        (df['Main_Engine_Power_kW'] < 0) |
        (df['Avg_draft_m'] <= 0) |
        (df['DRAFTFWD'] == -9999)
    )
    recall_known = (predictions[known_bad] == -1).mean()
    return recall_known  # Should be high

# 2. Stability analysis
# - Compare results across different contamination rates (0.5%, 1%, 2%)
# - Ensemble multiple models, flag points detected by majority

# 3. Physical plausibility
# - Anomalies should cluster in time (sensor failures are often persistent)
# - Check if flagged points violate known physics (power-speed relationship)
```

**Key Metrics:**
- Recall on known constraint violations (should catch all 78 EDA-flagged records)
- Anomaly rate consistency across clusters/operating modes
- Temporal clustering of detected anomalies

## Deployment Considerations

```python
# Preprocessing must match training
FEATURE_COLS = [
    'GPSSpeed_kn', 'Main_Engine_Power_kW', 'Speed_rpm',
    'Fuel_Consumption_t_per_day', 'Avg_draft_m', 'Trim_m',
    'TrueWindSpeed_kn', 'RelWindSpeed_kn'
]

# Save artifacts
import joblib
joblib.dump(model, 'models/clustered_svd_v1.pkl')

# Inference pipeline using ClusteredSVDAnomalyDetector
from src.models.clustered_svd_detector import ClusteredSVDAnomalyDetector

def detect_anomalies(new_data, model_path='models/clustered_svd_v1.pkl'):
    detector = ClusteredSVDAnomalyDetector.load(model_path)
    X = new_data[FEATURE_COLS].values
    predictions = detector.predict(X)  # -1 = anomaly, 1 = normal
    scores = detector.score_samples(X)
    cluster_labels = detector.get_cluster_labels(X)
    return predictions, scores, cluster_labels
```

## Anomaly Database

Store detected anomalies for tracking, analysis, and model improvement:

```python
import sqlite3
import pandas as pd
from datetime import datetime

def init_anomaly_db(db_path='anomalies.db'):
    """Initialize SQLite database for anomaly storage."""
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_timestamp TEXT NOT NULL,
            data_timestamp TEXT NOT NULL,
            anomaly_score REAL NOT NULL,
            threshold REAL NOT NULL,
            model_version TEXT,
            GPSSpeed_kn REAL,
            GPS_LAT REAL,
            GPS_LON REAL,
            Main_Engine_Power_kW REAL,
            Speed_rpm REAL,
            Fuel_Consumption_t_per_day REAL,
            Avg_draft_m REAL,
            Trim_m REAL,
            anomaly_type TEXT,
            reviewed INTEGER DEFAULT 0,
            notes TEXT
        )
    ''')
    conn.commit()
    return conn

def log_anomalies(df_anomalies, scores, threshold, model_version='svd_v1', db_path='anomalies.db'):
    """Log detected anomalies to database."""
    conn = sqlite3.connect(db_path)
    
    records = []
    for idx, row in df_anomalies.iterrows():
        records.append({
            'detection_timestamp': datetime.utcnow().isoformat(),
            'data_timestamp': str(row.get('Date', '')),
            'anomaly_score': float(scores[idx]) if idx < len(scores) else None,
            'threshold': float(threshold),
            'model_version': model_version,
            'GPSSpeed_kn': row.get('GPSSpeed_kn'),
            'GPS_LAT': row.get('GPS_LAT'),
            'GPS_LON': row.get('GPS_LON'),
            'Main_Engine_Power_kW': row.get('Main_Engine_Power_kW'),
            'Speed_rpm': row.get('Speed_rpm'),
            'Fuel_Consumption_t_per_day': row.get('Fuel_Consumption_t_per_day'),
            'Avg_draft_m': row.get('Avg_draft_m'),
            'Trim_m': row.get('Trim_m'),
            'anomaly_type': classify_anomaly(row),
            'reviewed': 0,
            'notes': None
        })
    
    pd.DataFrame(records).to_sql('anomalies', conn, if_exists='append', index=False)
    conn.close()

def classify_anomaly(row):
    """Classify anomaly type based on constraint violations."""
    types = []
    if row.get('Main_Engine_Power_kW', 0) < 0:
        types.append('negative_power')
    if abs(row.get('Trim_m', 0)) > 5:
        types.append('extreme_trim')
    if row.get('Avg_draft_m', 1) <= 0:
        types.append('invalid_draft')
    return ','.join(types) if types else 'multivariate_outlier'

def get_anomaly_summary(db_path='anomalies.db'):
    """Retrieve anomaly statistics."""
    conn = sqlite3.connect(db_path)
    return pd.read_sql('''
        SELECT anomaly_type, COUNT(*) as count, 
               AVG(anomaly_score) as avg_score,
               MIN(data_timestamp) as first_seen,
               MAX(data_timestamp) as last_seen
        FROM anomalies
        GROUP BY anomaly_type
    ''', conn)
```

**Database Schema Fields:**
- `anomaly_score` / `threshold` — For tuning and analysis
- `model_version` — Track which model detected the anomaly
- `anomaly_type` — Classification (constraint violation vs multivariate outlier)
- `reviewed` — Manual review flag for building labeled dataset
- `notes` — Human annotations for future supervised learning

**Production Checklist:**
- [ ] Handle missing timestamps (interpolate or drop if gap > 2min)
- [ ] Replace sentinel values (-9999) before inference
- [ ] Log anomaly scores, not just binary predictions
- [ ] Set up alerting thresholds based on anomaly density over time windows
- [ ] Regularly export reviewed anomalies for model retraining
