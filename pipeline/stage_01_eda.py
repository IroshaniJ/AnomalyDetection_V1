"""
Stage 01 — Exploratory Data Analysis (EDA)
==========================================
Analyses the raw CSV and writes summary statistics, constraint violation counts,
correlation heatmap, distribution histograms, and time-series plots.

Standalone usage:
    python pipeline/stage_01_eda.py
    python pipeline/stage_01_eda.py --data data1.csv --output results/01_eda

As a pipeline step:
    from pipeline.stage_01_eda import run
    result = run(config)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# ── Ensure project root is on sys.path ────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.eda.eda_runner import EDARunner


# ── Public API ─────────────────────────────────────────────────────────────────

def run(config: Optional[dict] = None) -> dict:
    """
    Execute the EDA stage.

    Args:
        config: Optional dict with keys:
            ``data``       – path to input CSV (default: 'data1.csv')
            ``output_dir`` – where to write outputs (default: 'results/01_eda')

    Returns:
        dict with keys: n_records, n_features, date_range,
        constraint_violations, output_dir.
    """
    config = config or {}
    data_path  = config.get('data', 'data1.csv')
    output_dir = config.get('output_dir', 'results/01_eda')

    print("\n" + "─" * 60)
    print("STAGE 01 — EDA")
    print("─" * 60)

    t0 = time.time()
    runner = EDARunner(output_dir=output_dir)
    result = runner.run(data_path=data_path)
    elapsed = time.time() - t0

    print(f"  [EDA] Done in {elapsed:.1f}s  →  {output_dir}/")
    result['elapsed_sec'] = round(elapsed, 2)
    return result


# ── CLI entry-point ────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Stage 01 – Exploratory Data Analysis'
    )
    parser.add_argument('--data',       default='data1.csv',
                        help='Path to input CSV (default: data1.csv)')
    parser.add_argument('--output-dir', default='results/01_eda',
                        help='Output directory (default: results/01_eda)')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    result = run({'data': args.data, 'output_dir': args.output_dir})

    print("\nSummary:")
    print(f"  Records analysed : {result['n_records']:,}")
    print(f"  Date range       : {result['date_range'][0]}  →  {result['date_range'][1]}")
    print("  Constraint violations:")
    for check, count in result['constraint_violations'].items():
        print(f"    {check}: {count}")
    print(f"\n  All outputs in:  {result['output_dir']}/")
