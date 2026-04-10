"""
Stage 06 — Saving Results
=========================
Persists the final anomaly results to SQLite, writes CSV exports,
and produces a plain-text pipeline report.

Reads from ``results/05_classification/full_classified_data.parquet``
(falls back to ``results/04_anomaly_detection/annotated_data.parquet``).

Writes:
  • ``anomalies.db``                       – SQLite database (updated)
  • ``results/06_saving/all_anomalies.csv``
  • ``results/06_saving/full_dataset_with_labels.csv``
  • ``results/06_saving/pipeline_report.txt``
  • ``results/06_saving/plots/``

Standalone usage:
    python pipeline/stage_06_saving.py
    python pipeline/stage_06_saving.py --db my_anomalies.db

As a pipeline step:
    from pipeline.stage_06_saving import run
    result = run(config)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Ensure project root is on sys.path ────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.saving.results_saver import ResultsSaver


# ── Public API ─────────────────────────────────────────────────────────────────

def run(config: Optional[dict] = None) -> dict:
    """
    Execute the saving stage.

    Args:
        config: Optional dict with keys:
            ``classification_dir`` – stage-05 output dir (default: 'results/05_classification')
            ``detection_dir``      – stage-04 output dir (default: 'results/04_anomaly_detection')
            ``output_dir``         – where to write      (default: 'results/06_saving')
            ``db``                 – SQLite path          (default: 'anomalies.db')
            ``model_version``      – model tag            (default: 'clustered_svd_v1')
            ``data``               – source CSV filename  (default: 'data1.csv')
            ``avg_threshold``      – anomaly threshold    (default: auto from data)
            ``n_components``       – SVD components       (default: 3)
            ``threshold_pct``      – threshold percentile (default: 95.0)

    Returns:
        dict with keys: run_id, n_anomalies_logged, anomalies_csv,
        full_dataset_csv, report, db_path, output_dir.
    """
    config = config or {}
    classification_dir = Path(config.get('classification_dir', 'results/05_classification'))
    detection_dir      = Path(config.get('detection_dir',      'results/04_anomaly_detection'))
    output_dir         = Path(config.get('output_dir',         'results/06_saving'))
    db_path            = config.get('db',            'anomalies.db')
    model_version      = config.get('model_version', 'clustered_svd_v1')
    data_file          = config.get('data',          'data1.csv')
    n_components       = int(config.get('n_components',   3))
    threshold_pct      = float(config.get('threshold_pct', 95.0))

    print("\n" + "─" * 60)
    print("STAGE 06 — SAVING RESULTS")
    print("─" * 60)

    t0 = time.time()

    # ── Load data ──────────────────────────────────────────────────────────────
    df = _load_classified(classification_dir, detection_dir)
    print(f"  [Saving] {len(df):,} records loaded  "
          f"({int(df['is_anomaly'].astype(bool).sum()):,} anomalies)")

    # ── Resolve threshold ──────────────────────────────────────────────────────
    if 'avg_threshold' in config:
        avg_threshold = float(config['avg_threshold'])
    elif 'anomaly_score' in df.columns:
        anomalies = df[df['is_anomaly'].astype(bool)]
        avg_threshold = float(anomalies['anomaly_score'].quantile(0.05)) if len(anomalies) > 0 else 0.1
    else:
        avg_threshold = 0.1

    # ── Save ───────────────────────────────────────────────────────────────────
    saver = ResultsSaver(
        output_dir=str(output_dir),
        db_path=db_path,
        model_version=model_version,
    )
    result_meta = saver.save(
        df=df,
        threshold=avg_threshold,
        n_components=n_components,
        threshold_percentile=threshold_pct,
        data_file=data_file,
    )

    elapsed = time.time() - t0
    print(f"  [Saving] Done in {elapsed:.1f}s  →  {output_dir}/  +  {db_path}")

    return {
        **result_meta,
        'output_dir':  str(output_dir),
        'elapsed_sec': round(elapsed, 2),
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_classified(classification_dir: Path, detection_dir: Path) -> pd.DataFrame:
    """Load the best available classified / annotated data."""
    classified_pq = classification_dir / 'full_classified_data.pkl'
    if classified_pq.exists():
        print(f"  [Saving] Loading classified data from {classified_pq}")
        return pd.read_pickle(classified_pq)

    # Fallback to stage-04 output (no classification)
    annotated_pq = detection_dir / 'annotated_data.pkl'
    if annotated_pq.exists():
        print(f"  [Saving] Classified data not found — loading from {annotated_pq}")
        df = pd.read_pickle(annotated_pq)
        # Add minimal classification columns
        df['primary_type'] = 'multivariate_outlier'
        df.loc[~df['is_anomaly'].astype(bool), 'primary_type'] = 'normal'
        return df

    raise FileNotFoundError(
        "No annotated data found. Run stage_04_anomaly_detection.py first."
    )


# ── CLI entry-point ────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Stage 06 – Save Results to Database and Files'
    )
    parser.add_argument('--classification-dir', default='results/05_classification')
    parser.add_argument('--detection-dir',      default='results/04_anomaly_detection')
    parser.add_argument('--output-dir',         default='results/06_saving')
    parser.add_argument('--db',                 default='anomalies.db',
                        help='SQLite database path (default: anomalies.db)')
    parser.add_argument('--model-version',      default='clustered_svd_v1')
    parser.add_argument('--data',               default='data1.csv')
    parser.add_argument('--n-components',       type=int,   default=3)
    parser.add_argument('--threshold-pct',      type=float, default=95.0)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    result = run({
        'classification_dir': args.classification_dir,
        'detection_dir':      args.detection_dir,
        'output_dir':         args.output_dir,
        'db':                 args.db,
        'model_version':      args.model_version,
        'data':               args.data,
        'n_components':       args.n_components,
        'threshold_pct':      args.threshold_pct,
    })

    print("\nSummary:")
    print(f"  Anomalies logged : {result['n_anomalies_logged']:,}")
    print(f"  Database         : {result['db_path']}")
    print(f"  Report           : {result['report']}")
    print(f"\n  Artefacts in:    {result['output_dir']}/")
