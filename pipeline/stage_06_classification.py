"""
Stage 05 — Anomaly Classification
==================================
Classifies detected anomalies into human-readable types using rule-based and
statistical criteria:
  • negative_power        – engine power < 0 kW
  • extreme_trim          – |Trim_m| > 5 m
  • invalid_draft         – Avg_draft_m ≤ 0
  • high_speed_low_power  – fast vessel with negligible engine power
  • low_speed_high_power  – vessel at rest with high engine power
  • fuel_efficiency_outlier – abnormal power/fuel ratio
  • multivariate_outlier  – SVD anomaly with no rule match

Reads from ``results/04_anomaly_detection/annotated_data.parquet``.

Writes:
  • ``results/05_classification/classified_anomalies.csv``
  • ``results/05_classification/anomaly_type_breakdown.csv``
  • ``results/05_classification/plots/``

Standalone usage:
    python pipeline/stage_05_classification.py
    python pipeline/stage_05_classification.py --detection-dir results/04_anomaly_detection

As a pipeline step:
    from pipeline.stage_05_classification import run
    result = run(config)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Ensure project root is on sys.path ────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.classification.anomaly_classifier import AnomalyClassifier


# ── Public API ─────────────────────────────────────────────────────────────────

def run(config: Optional[dict] = None) -> dict:
    """
    Execute the anomaly classification stage.

    Args:
        config: Optional dict with keys:
            ``detection_dir`` – stage-04 output dir (default: 'results/04_anomaly_detection')
            ``output_dir``    – where to write      (default: 'results/05_classification')

    Returns:
        dict with keys: n_anomalies, type_breakdown, anomalies_csv,
        breakdown_csv, output_dir.
    """
    config = config or {}
    detection_dir = Path(config.get('detection_dir', 'results/04_anomaly_detection'))
    output_dir    = Path(config.get('output_dir',    'results/05_classification'))

    print("\n" + "─" * 60)
    print("STAGE 05 — ANOMALY CLASSIFICATION")
    print("─" * 60)

    t0 = time.time()

    # ── Load annotated data ────────────────────────────────────────────────────
    df = _load_annotated(detection_dir)
    n_anomalies = int(df['is_anomaly'].astype(bool).sum())
    print(f"  [Classification] {len(df):,} total records, {n_anomalies:,} anomalies")

    # ── Classify ───────────────────────────────────────────────────────────────
    classifier = AnomalyClassifier(output_dir=str(output_dir))
    df = classifier.classify(df)

    # ── Save results ───────────────────────────────────────────────────────────
    result_meta = classifier.save_results(df)

    # Also persist the full annotated+classified DataFrame
    full_classified_pq = output_dir / 'full_classified_data.pkl'
    df.to_pickle(full_classified_pq)

    elapsed = time.time() - t0
    print(f"  [Classification] Done in {elapsed:.1f}s  →  {output_dir}/")

    return {
        'n_anomalies':      n_anomalies,
        'type_breakdown':   result_meta['type_breakdown'],
        'anomalies_csv':    result_meta['anomalies_csv'],
        'breakdown_csv':    result_meta['breakdown_csv'],
        'full_parquet':     str(full_classified_pq),
        'pie_plot':         result_meta['pie_plot'],
        'bar_plot':         result_meta['bar_plot'],
        'ts_plot':          result_meta['ts_plot'],
        'output_dir':       str(output_dir),
        'elapsed_sec':      round(elapsed, 2),
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_annotated(detection_dir: Path) -> pd.DataFrame:
    """Load annotated parquet from stage_04 output."""
    pq_path = detection_dir / 'annotated_data.pkl'
    if pq_path.exists():
        print(f"  [Classification] Loading from {pq_path}")
        return pd.read_pickle(pq_path)

    raise FileNotFoundError(
        f"Stage-04 output not found at {pq_path}.\n"
        "Run stage_04_anomaly_detection.py first, or pass --detection-dir."
    )


# ── CLI entry-point ────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Stage 05 – Anomaly Classification'
    )
    parser.add_argument('--detection-dir', default='results/04_anomaly_detection',
                        help='Stage-04 output directory')
    parser.add_argument('--output-dir',    default='results/05_classification',
                        help='Output directory (default: results/05_classification)')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    result = run({
        'detection_dir': args.detection_dir,
        'output_dir':    args.output_dir,
    })

    print("\nSummary:")
    print(f"  Total anomalies classified: {result['n_anomalies']:,}")
    print("  Type breakdown:")
    for row in result['type_breakdown']:
        print(f"    {row['primary_type']:<30}: {row['count']:>5}  ({row['pct']:.1f}%)")
    print(f"\n  Artefacts in:  {result['output_dir']}/")
