"""
Stage 02 — Data Cleaning
========================
Loads the raw CSV, applies the cleaning pipeline (sentinel replacement,
constraint-flag columns, ``is_moving`` state), isolates records that cannot
be used for modelling (NaN in any feature column), logs them to
``anomalies.db`` as *missing records*, then saves only the valid rows:

  • ``results/02_cleaning/cleaned_data.pkl``      – valid rows only (no NaN in features)
  • ``results/02_cleaning/feature_matrix.npy``    – feature matrix (float64)
  • ``results/02_cleaning/valid_mask.npy``         – boolean mask (all True — kept for API compat)
  • ``results/02_cleaning/missing_records.csv``   – rows excluded (saved locally too)
  • ``results/02_cleaning/cleaning_report.csv``   – before/after quality stats

Standalone usage:
    python pipeline/stage_02_cleaning.py
    python pipeline/stage_02_cleaning.py --data data1.csv --output results/02_cleaning

As a pipeline step:
    from pipeline.stage_02_cleaning import run
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

from src.preprocessing.data_loader import load_and_clean, get_feature_columns
from src.database.anomaly_db import AnomalyDatabase


# ── Public API ─────────────────────────────────────────────────────────────────

def run(config: Optional[dict] = None) -> dict:
    """
    Execute the data cleaning stage.

    Args:
        config: Optional dict with keys:
            ``data``       – path to input CSV (default: 'data1.csv')
            ``output_dir`` – where to write outputs (default: 'results/02_cleaning')
            ``db``         – SQLite DB path (default: 'anomalies.db')

    Returns:
        dict with keys: n_records, n_valid, n_missing, feature_cols, output_dir,
        cleaned_pkl, feature_matrix_npy, valid_mask_npy, missing_csv.
    """
    config = config or {}
    data_path  = config.get('data', 'data1.csv')
    output_dir = Path(config.get('output_dir', 'results/02_cleaning'))
    db_path    = config.get('db', 'anomalies.db')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "─" * 60)
    print("STAGE 02 — DATA CLEANING")
    print("─" * 60)

    t0 = time.time()

    # ── Load & clean ───────────────────────────────────────────────────────────
    print(f"  [Cleaning] Loading {data_path} …")
    df = load_and_clean(data_path)
    n_total = len(df)
    print(f"  [Cleaning] {n_total:,} records loaded")

    # ── Separate valid vs missing rows ─────────────────────────────────────────
    feature_cols = get_feature_columns()
    valid_mask   = ~df[feature_cols].isna().any(axis=1)
    df_valid     = df[valid_mask].reset_index(drop=True)
    df_missing   = df[~valid_mask].reset_index(drop=True)

    n_valid   = len(df_valid)
    n_missing = len(df_missing)
    print(f"  [Cleaning] Valid records  : {n_valid:,} / {n_total:,}")
    print(f"  [Cleaning] Missing records: {n_missing:,}  (excluded — NaN in feature columns)")

    # ── Log missing records to DB ──────────────────────────────────────────────
    missing_csv = output_dir / 'missing_records.csv'
    if n_missing > 0:
        db = AnomalyDatabase(db_path)
        n_logged = db.log_missing_records(
            df_missing, source_file=data_path, feature_cols=feature_cols
        )
        print(f"  [Cleaning] Logged {n_logged} missing records → {db_path} (missing_records table)")
        df_missing.to_csv(missing_csv, index=False)
    else:
        print(f"  [Cleaning] No missing records found.")
        pd.DataFrame().to_csv(missing_csv, index=False)

    # ── Build feature matrix from valid rows ───────────────────────────────────
    X_valid = df_valid[feature_cols].values.astype(np.float64)
    # valid_mask saved as all-True since cleaned_data.pkl now only has valid rows
    valid_mask_out = np.ones(n_valid, dtype=bool)

    # ── Constraint summary (over valid rows only) ──────────────────────────────
    flags = {
        'flag_negative_power': int(df_valid['flag_negative_power'].sum()),
        'flag_extreme_trim':   int(df_valid['flag_extreme_trim'].sum()),
        'flag_invalid_draft':  int(df_valid['flag_invalid_draft'].sum()),
        'flag_any_constraint': int(df_valid['flag_any_constraint'].sum()),
    }
    for flag, count in flags.items():
        print(f"  [Cleaning]   {flag}: {count}")

    # ── Save artefacts ─────────────────────────────────────────────────────────
    cleaned_pkl        = output_dir / 'cleaned_data.pkl'
    feature_matrix_npy = output_dir / 'feature_matrix.npy'
    valid_mask_npy     = output_dir / 'valid_mask.npy'

    df_valid.to_pickle(cleaned_pkl)
    np.save(feature_matrix_npy, X_valid)
    np.save(valid_mask_npy, valid_mask_out)

    # ── Cleaning report CSV ────────────────────────────────────────────────────
    pd.DataFrame([
        {'metric': 'total_records',   'value': n_total},
        {'metric': 'valid_records',   'value': n_valid},
        {'metric': 'missing_records', 'value': n_missing},
        {'metric': 'n_features',      'value': len(feature_cols)},
        *[{'metric': k, 'value': v} for k, v in flags.items()],
    ]).to_csv(output_dir / 'cleaning_report.csv', index=False)

    elapsed = time.time() - t0
    print(f"  [Cleaning] Done in {elapsed:.1f}s  →  {output_dir}/")

    return {
        'n_records':          n_total,
        'n_valid':            n_valid,
        'n_missing':          n_missing,
        'feature_cols':       feature_cols,
        'cleaned_pkl':        str(cleaned_pkl),
        'feature_matrix_npy': str(feature_matrix_npy),
        'valid_mask_npy':     str(valid_mask_npy),
        'missing_csv':        str(missing_csv),
        'output_dir':         str(output_dir),
        'flags':              flags,
        'elapsed_sec':        round(elapsed, 2),
    }


# ── CLI entry-point ────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Stage 02 – Data Cleaning'
    )
    parser.add_argument('--data',       default='data1.csv',
                        help='Path to input CSV (default: data1.csv)')
    parser.add_argument('--output-dir', default='results/02_cleaning',
                        help='Output directory (default: results/02_cleaning)')
    parser.add_argument('--db',         default='anomalies.db',
                        help='SQLite DB path (default: anomalies.db)')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    result = run({'data': args.data, 'output_dir': args.output_dir, 'db': args.db})

    print("\nSummary:")
    print(f"  Total records   : {result['n_records']:,}")
    print(f"  Valid records   : {result['n_valid']:,}")
    print(f"  Missing removed : {result['n_missing']:,}  → anomalies.db (missing_records table)")
    print(f"  Features        : {result['feature_cols']}")
    print(f"\n  Artefacts in:   {result['output_dir']}/")
    print(f"    cleaned_data.pkl         (valid rows only)")
    print(f"    feature_matrix.npy       ({result['n_valid']}×{len(result['feature_cols'])})")
    print(f"    valid_mask.npy")
    print(f"    missing_records.csv")
    print(f"    cleaning_report.csv")
