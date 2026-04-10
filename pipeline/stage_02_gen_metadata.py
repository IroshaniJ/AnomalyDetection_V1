"""
Stage 02 — Generate Variable Metadata
======================================
Runs per-vessel variable metadata generation (Phase 7) after EDA.  For each
vessel that has an existing ``results/01_eda/{vessel_id}/`` directory the
stage either:

  FULL  — raw data file is available (``--data`` flag or ``data`` config key)
           → reloads the CSV and re-runs ``EDARunner._variable_metadata``
           → produces the most accurate n_spikes / n_frozen_periods counts

  PATCH — no raw data available (default)
           → re-derives ``unit``, ``physical_limits``, ``hard_filter``, and
             ``soft_warning`` from the cached ``univariate_report.csv`` and the
             current ``VesselConfig`` physical limits
           → all other columns are preserved from the existing metadata CSV

Output per vessel
-----------------
  ``results/01_eda/{vessel_id}/{vessel_id}_metadata.csv``

Standalone usage:
    python pipeline/stage_02_gen_metadata.py
    python pipeline/stage_02_gen_metadata.py --data path/to/raw.csv
    python pipeline/stage_02_gen_metadata.py --vessel stenateknik

As a pipeline step:
    from pipeline.stage_02_gen_metadata import run
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

from pipeline.regen_metadata import run as _regen_run


# ── Public API ─────────────────────────────────────────────────────────────────

def run(config: Optional[dict] = None) -> dict:
    """
    Execute the generate-metadata stage.

    Calls ``pipeline.regen_metadata.run`` to update the
    ``{vessel_id}_metadata.csv`` files produced by Stage 01.

    Args:
        config: Optional dict with keys:
            ``data``       – path to raw CSV; when supplied, full-mode
                             regeneration runs for that vessel (default: None)
            ``vessel``     – restrict patch mode to a single vessel id
                             (default: None — all vessels)
            ``eda_dir``    – EDA output root (default: 'results/01_eda');
                             used only for reporting

    Returns:
        dict with keys: vessels_updated, output_paths, output_dir, elapsed_sec.
    """
    config = config or {}
    data_path  = config.get('data',    None)
    vessel_id  = config.get('vessel',  None)
    eda_dir    = config.get('eda_dir', 'results/01_eda')

    print("\n" + "─" * 60)
    print("STAGE 02 — GENERATE VARIABLE METADATA")
    print("─" * 60)

    t0 = time.time()
    vessel_paths = _regen_run(data_path=data_path, vessel_id=vessel_id)
    elapsed = time.time() - t0

    if vessel_paths:
        for vid, path in vessel_paths.items():
            print(f"  [gen_metadata] {vid} → {path}")
    else:
        print("  [gen_metadata] No metadata files updated.")

    print(f"  [gen_metadata] Done in {elapsed:.1f}s")

    return {
        'vessels_updated': list(vessel_paths.keys()),
        'output_paths':    {k: str(v) for k, v in vessel_paths.items()},
        'output_dir':      eda_dir,
        'elapsed_sec':     round(elapsed, 2),
    }


# ── CLI entry-point ────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Stage 02 — Generate vessel variable metadata CSVs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Patch all vessels (no raw data needed)
  python pipeline/stage_02_gen_metadata.py

  # Full regeneration from raw CSV
  python pipeline/stage_02_gen_metadata.py --data path/to/9685475_202407.csv

  # Patch one vessel only
  python pipeline/stage_02_gen_metadata.py --vessel stenateknik
        """
    )
    parser.add_argument('--data',   metavar='PATH', default=None,
                        help='Raw CSV path (triggers full mode for that vessel)')
    parser.add_argument('--vessel', metavar='ID',   default=None,
                        help='Restrict patch mode to this vessel id')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run({'data': args.data, 'vessel': args.vessel})
