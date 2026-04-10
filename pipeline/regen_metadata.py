"""
Regenerate vessel metadata CSVs (Phase 7 output).

Two operating modes — chosen automatically per vessel:

  FULL  — raw CSV is found → EDARunner._variable_metadata runs on live data.
           Produces the most accurate n_spikes / n_frozen_periods counts.

  PATCH — no raw CSV → derive unit / physical_limits / hard_filter /
           soft_warning directly from the cached univariate_report.csv that
           was written by a previous full EDA run.  All other columns
           (descriptions, dynamics, interpolation flags …) are preserved.

Usage
-----
    python pipeline/regen_metadata.py                       # all vessels
    python pipeline/regen_metadata.py --vessel stenateknik  # one vessel
    python pipeline/regen_metadata.py --test                # smoke-test only

Standalone or as a pipeline step:
    from pipeline.regen_metadata import run
    run()                            # patch mode for all vessels
    run(data_path='data/stena/…')    # full mode for one vessel
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import numpy as np


# ── Helpers shared between full & patch modes ─────────────────────────────────

def _unit_from_raw_col(raw_col: str) -> str:
    """Extract unit from column names with square-bracket or trailing-paren notation.

    Examples
    --------
    'SPEED THROUGH WATER [kn]'  → 'kn'
    'ME Fuel Mass Net (kg/hr)'  → 'kg/hr'
    'Blr DO Temp In (Â°C)'      → '°C'
    'AE Emerg DO Mode ()'       → ''
    """
    def _clean(s: str) -> str:
        s = s.strip().replace('Â°', '°').replace('\u00b0', '°')
        return '' if (not s or s.isdigit() or len(s) > 20) else s

    m = re.search(r'\[([^\]]+)\]', raw_col)
    if m:
        return _clean(m.group(1))
    m = re.search(r'\(([^)]*)\)\s*$', raw_col)
    if m:
        return _clean(m.group(1))
    return ''


# Columns where negative values must be hard-removed (physically impossible)
_STRICT_NEG = {
    'Main_Engine_Power_kW', 'Fuel_Consumption_rate', 'GPSSpeed_kn',
    'DRAFTAFT', 'DRAFTFWD', 'RelWindSpeed_kn', 'Speed_rpm',
}


def _estimate_violations_below(u_row: pd.Series, lo: float, n_missing: int) -> int:
    """Estimate n values < lo from cached quantile data."""
    if u_row is None:
        return 0
    n_valid = int(u_row.get('n_total', 0)) - n_missing
    if n_valid <= 0:
        return 0
    if lo == 0.0:
        return int(u_row.get('n_negative', 0))
    for frac, qcol in ((0.01, 'q01'), (0.05, 'q05'), (0.10, 'q10'),
                       (0.25, 'q25'), (0.50, 'q50')):
        if qcol in u_row and pd.notna(u_row[qcol]) and float(u_row[qcol]) >= lo:
            return int(n_valid * frac)
    return 0


def _estimate_violations_above(u_row: pd.Series, hi: float,
                                n_missing: int, obs_max: float) -> int:
    """Estimate n values > hi from cached quantile data."""
    if u_row is None or obs_max <= hi:
        return 0
    n_valid = int(u_row.get('n_total', 0)) - n_missing
    if n_valid <= 0:
        return 0
    for frac, qcol in ((0.01, 'q99'), (0.05, 'q95'), (0.10, 'q90'),
                       (0.25, 'q75')):
        if qcol in u_row and pd.notna(u_row[qcol]) and float(u_row[qcol]) > hi:
            return int(n_valid * frac)
    return int(n_valid * 0.01)   # < 10% violation, estimate ~1%


# ── Patch mode ────────────────────────────────────────────────────────────────

def _patch_from_cache(vessel_id: str, cfg) -> Path:
    """Update unit / physical_limits / hard_filter / soft_warning in-place."""
    meta_path = Path(f'results/01_eda/{vessel_id}/{vessel_id}_metadata.csv')
    uni_path  = Path(f'results/01_eda/{vessel_id}/univariate_report.csv')

    if not meta_path.exists():
        raise FileNotFoundError(f'Metadata CSV not found: {meta_path}')

    meta = pd.read_csv(meta_path)
    uni  = (pd.read_csv(uni_path).set_index('signal')
            if uni_path.exists() else pd.DataFrame())

    changed = 0
    for idx, row in meta.iterrows():
        col     = str(row['variable_name'])
        raw_col = str(row['raw_column_name'])

        # 1. Unit from raw column name
        new_unit = _unit_from_raw_col(raw_col) or (
            row['unit'] if pd.notna(row['unit']) else '')

        # 2. Physical limits from VesselConfig
        vlo, vhi = cfg.physical_limits.get(col, (None, None))

        if vlo is not None and vhi is not None:
            new_phys = f'{vlo} \u2013 {vhi}'
        elif vlo is not None:
            new_phys = f'>= {vlo}'
        elif vhi is not None:
            new_phys = f'<= {vhi}'
        else:
            new_phys = row['physical_limits'] if pd.notna(row['physical_limits']) else ''

        # 3. Hard filter
        hard_parts = []
        if col in _STRICT_NEG and vlo == 0.0:
            hard_parts.append('remove_negative')
        if vhi is not None:
            hard_parts.append(f'remove_gt_{vhi}')
        if vlo is not None and vlo != 0.0:
            hard_parts.append(f'remove_lt_{vlo}')
        new_hard = '; '.join(hard_parts) or 'none'

        # 4. Soft warning — recompute physical violation count
        obs_min = float(row['observed_min']) if pd.notna(row.get('observed_min')) else None
        obs_max = float(row['observed_max']) if pd.notna(row.get('observed_max')) else None
        u_row   = uni.loc[col] if (not uni.empty and col in uni.index) else None
        n_miss  = int(row.get('n_missing', 0))

        n_viol = 0
        if vlo is not None and obs_min is not None and obs_min < vlo:
            n_viol += _estimate_violations_below(u_row, vlo, n_miss)
        if vhi is not None and obs_max is not None and obs_max > vhi:
            n_viol += _estimate_violations_above(u_row, vhi, n_miss, obs_max)

        old_soft = str(row['soft_warning']) if pd.notna(row['soft_warning']) else 'none'
        soft_parts = [p.strip() for p in old_soft.split(';')
                      if p.strip() not in ('none', '') and '_phys_violations' not in p]
        if n_viol > 0:
            soft_parts.insert(0, f'{n_viol}_phys_violations')
        new_soft = '; '.join(soft_parts) or 'none'

        updates = {
            'unit': new_unit, 'physical_limits': new_phys,
            'hard_filter': new_hard, 'soft_warning': new_soft,
        }
        for k, v in updates.items():
            if str(meta.at[idx, k]) != str(v):
                meta.at[idx, k] = v
                changed += 1

    meta.to_csv(meta_path, index=False)
    print(f'  [{vessel_id}] patch mode — {changed} cell(s) updated → {meta_path}')
    return meta_path


# ── Full mode ─────────────────────────────────────────────────────────────────

def _regen_from_raw(vessel_id: str, cfg, raw_path: str) -> Path:
    """Full regeneration: load raw CSV and re-run EDARunner._variable_metadata."""
    from src.eda.eda_runner import EDARunner
    from src.preprocessing.data_loader import load_raw

    vessel_dir = Path(f'results/01_eda/{vessel_id}')
    vessel_dir.mkdir(parents=True, exist_ok=True)

    print(f'  [{vessel_id}] Loading {raw_path} …')
    t0 = time.time()
    df = load_raw(raw_path)
    print(f'  [{vessel_id}] {len(df):,} rows × {df.shape[1]} cols '
          f'in {time.time() - t0:.1f}s')

    runner = EDARunner(output_dir='results/01_eda')
    runner._vessel_cfg = cfg

    print(f'  [{vessel_id}] Running _variable_metadata …')
    t1 = time.time()
    out = runner._variable_metadata(df, vessel_dir)
    print(f'  [{vessel_id}] full mode — done in {time.time() - t1:.1f}s → {out}')
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def run(data_path: Optional[str] = None, vessel_id: Optional[str] = None) -> dict:
    """Regenerate metadata for all (or one) vessel(s).

    Parameters
    ----------
    data_path:
        Path to the raw CSV for a single vessel.  When supplied the function
        runs in full mode for the vessel whose config matches the path.
        If omitted every vessel with an existing output directory is patched
        from its cached univariate_report.csv.
    vessel_id:
        Optional override to restrict which vessel to process when running
        in patch mode (no data_path).

    Returns
    -------
    dict  {vessel_id: Path}
    """
    from src.vessel_config import STENALINE, STENATEKNIK, GRIMALDI, detect_vessel

    all_configs = [STENALINE, STENATEKNIK, GRIMALDI]
    results: dict = {}

    if data_path:
        cfg = detect_vessel(data_path)
        results[cfg.vessel_id] = _regen_from_raw(cfg.vessel_id, cfg, data_path)
        return results

    for cfg in all_configs:
        vid = cfg.vessel_id
        if vessel_id and vid != vessel_id:
            continue
        if not Path(f'results/01_eda/{vid}/{vid}_metadata.csv').exists():
            print(f'  SKIP {vid}: no metadata CSV found')
            continue
        results[vid] = _patch_from_cache(vid, cfg)

    return results


# ── Smoke test ────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """Verify _variable_metadata runs without errors on a synthetic dataset."""
    from src.eda.eda_runner import EDARunner

    class _FakeCfg:
        vessel_id = 'test_vessel'
        vessel_name = 'Test Vessel'
        column_map = {}
        derived_cols = {}
        expected_interval_s = 120
        physical_limits: dict = {}
        fuel_power_slope = 4.7
        fuel_power_offset_low = -4500.0
        fuel_power_offset_high = 3500.0

    n = 200
    df = pd.DataFrame({
        'Date':                  pd.date_range('2020-01-01', periods=n,
                                               freq='2min', tz='UTC'),
        'GPSSpeed_kn':           np.random.uniform(0, 20, n),
        'Main_Engine_Power_kW':  np.random.uniform(0, 4000, n),
        'Speed_rpm':             np.random.uniform(0, 120, n),
        'Ship_Course_deg':       np.random.uniform(0, 360, n),
        'Heading_deg':           np.random.uniform(0, 360, n),
    })

    out_dir = Path('results/01_eda/test_vessel')
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = EDARunner(output_dir='results/01_eda')
    runner._vessel_cfg = _FakeCfg()
    out = runner._variable_metadata(df, out_dir)
    print(f'  smoke test PASSED → {out}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data',   metavar='PATH',
                        help='Raw CSV path (triggers full mode for that vessel)')
    parser.add_argument('--vessel', metavar='ID',
                        help='Restrict patch mode to this vessel id')
    parser.add_argument('--test',   action='store_true',
                        help='Run smoke test only')
    args = parser.parse_args()

    if args.test:
        print('Running smoke test …')
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            _smoke_test()
        sys.exit(0)

    print('\n' + '─' * 60)
    print('Regenerate vessel metadata CSVs')
    print('─' * 60)
    t0 = time.time()
    results = run(data_path=args.data, vessel_id=args.vessel)
    print(f'\nDone in {time.time() - t0:.1f}s  ({len(results)} vessel(s) updated)')
