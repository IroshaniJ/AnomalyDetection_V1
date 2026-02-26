"""
Full Pipeline Orchestrator
==========================
Chains all six stages end-to-end:

    Stage 01 → EDA
    Stage 02 → Data Cleaning
    Stage 03 → Clustering
    Stage 04 → Anomaly Detection
    Stage 05 → Anomaly Classification
    Stage 06 → Saving (SQLite + CSV + report)

Usage
─────
Run the complete pipeline with defaults:
    python pipeline/run_pipeline.py

Skip EDA (faster re-runs):
    python pipeline/run_pipeline.py --skip-eda

Override data file or thresholds:
    python pipeline/run_pipeline.py --data path/to/data.csv --threshold-pct 97

Run only a specific stage:
    python pipeline/run_pipeline.py --only-stage 04

Programmatic use:
    from pipeline.run_pipeline import run_pipeline
    results = run_pipeline({'threshold_pct': 97, 'skip_eda': True})
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

from pipeline.stage_01_eda              import run as run_eda
from pipeline.stage_02_cleaning         import run as run_cleaning
from pipeline.stage_03_clustering       import run as run_clustering
from pipeline.stage_04_anomaly_detection import run as run_detection
from pipeline.stage_05_classification   import run as run_classification
from pipeline.stage_06_saving           import run as run_saving


# ── Public API ─────────────────────────────────────────────────────────────────

def run_pipeline(config: Optional[dict] = None) -> dict:
    """
    Execute the full anomaly detection pipeline.

    Args:
        config: Optional dict with any combination of the keys below.
            Global:
                ``data``             – path to input CSV (default: 'data1.csv')
                ``skip_eda``         – skip Stage 01 (default: False)
                ``only_stage``       – run only this stage, e.g. '04' (default: None)
            Stage-specific (forwarded to each stage):
                ``n_clusters``       – K-Means clusters (default: 4)
                ``n_components``     – SVD components  (default: 3)
                ``threshold_pct``    – anomaly percentile threshold (default: 95.0)
                ``random_state``     – random seed (default: 42)
                ``model_version``    – model artefact tag (default: 'clustered_svd_v1')
                ``db``               – SQLite path (default: 'anomalies.db')
            Output directories (defaults shown):
                ``eda_dir``              → results/01_eda
                ``cleaning_dir``         → results/02_cleaning
                ``clustering_dir``       → results/03_clustering
                ``detection_dir``        → results/04_anomaly_detection
                ``classification_dir``   → results/05_classification
                ``saving_dir``           → results/06_saving

    Returns:
        dict mapping stage name → that stage's result dict.
    """
    config = config or {}
    t_total = time.time()

    # ── Resolve shared settings ────────────────────────────────────────────────
    data          = config.get('data',          'data1.csv')
    skip_eda      = bool(config.get('skip_eda', False))
    only_stage    = str(config.get('only_stage', '')).zfill(2) if config.get('only_stage') else None
    n_clusters      = int(config.get('n_clusters',      4))
    n_components    = int(config.get('n_components',    3))
    threshold_pct   = float(config.get('threshold_pct', 95.0))
    random_state    = int(config.get('random_state',    42))
    model_version   = config.get('model_version', 'clustered_svd_v1')
    db_path         = config.get('db', 'anomalies.db')
    det_methods     = config.get('methods', ['svd', 'iforest', 'autoencoder'])
    if_n_estimators = int(config.get('if_n_estimators', 100))
    ae_epochs       = int(config.get('ae_epochs',       50))
    ae_encoding_dim = int(config.get('ae_encoding_dim', 3))

    eda_dir            = config.get('eda_dir',            'results/01_eda')
    cleaning_dir       = config.get('cleaning_dir',       'results/02_cleaning')
    clustering_dir     = config.get('clustering_dir',     'results/03_clustering')
    detection_dir      = config.get('detection_dir',      'results/04_anomaly_detection')
    classification_dir = config.get('classification_dir', 'results/05_classification')
    saving_dir         = config.get('saving_dir',         'results/06_saving')

    _banner()
    results: dict = {}

    # ── Stage 01 — EDA ─────────────────────────────────────────────────────────
    if _should_run('01', only_stage, skip_eda):
        results['01_eda'] = run_eda({'data': data, 'output_dir': eda_dir})
    else:
        _skip('01', 'EDA', skip_eda)

    # ── Stage 02 — Cleaning ────────────────────────────────────────────────────
    if _should_run('02', only_stage):
        results['02_cleaning'] = run_cleaning({
            'data':       data,
            'output_dir': cleaning_dir,
            'db':         db_path,
        })
    else:
        _skip('02', 'Cleaning')

    # ── Stage 03 — Clustering ──────────────────────────────────────────────────
    if _should_run('03', only_stage):
        results['03_clustering'] = run_clustering({
            'cleaning_dir': cleaning_dir,
            'data':         data,
            'output_dir':   clustering_dir,
            'n_clusters':   n_clusters,
            'random_state': random_state,
        })
    else:
        _skip('03', 'Clustering')

    # ── Stage 04 — Anomaly Detection ──────────────────────────────────────────
    if _should_run('04', only_stage):
        results['04_detection'] = run_detection({
            'cleaning_dir':    cleaning_dir,
            'data':            data,
            'output_dir':      detection_dir,
            'methods':         det_methods,
            'n_clusters':      n_clusters,
            'n_components':    n_components,
            'threshold_pct':   threshold_pct,
            'random_state':    random_state,
            'model_version':   model_version,
            'if_n_estimators': if_n_estimators,
            'ae_epochs':       ae_epochs,
            'ae_encoding_dim': ae_encoding_dim,
        })
    else:
        _skip('04', 'Anomaly Detection')

    # ── Stage 05 — Classification ──────────────────────────────────────────────
    if _should_run('05', only_stage):
        results['05_classification'] = run_classification({
            'detection_dir': detection_dir,
            'output_dir':    classification_dir,
        })
    else:
        _skip('05', 'Classification')

    # ── Stage 06 — Saving ──────────────────────────────────────────────────────
    if _should_run('06', only_stage):
        # Carry avg_threshold from detection stage if available
        avg_thr = results.get('04_detection', {}).get('avg_threshold_val', None)
        saving_cfg: dict = {
            'classification_dir': classification_dir,
            'detection_dir':      detection_dir,
            'output_dir':         saving_dir,
            'db':                 db_path,
            'model_version':      model_version,
            'data':               data,
            'n_components':       n_components,
            'threshold_pct':      threshold_pct,
        }
        if avg_thr is not None:
            saving_cfg['avg_threshold'] = avg_thr
        results['06_saving'] = run_saving(saving_cfg)
    else:
        _skip('06', 'Saving')

    # ── Final summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    _final_summary(results, elapsed)

    return results


# ── Internal helpers ───────────────────────────────────────────────────────────

def _banner():
    print("\n" + "═" * 60)
    print("  TWINSHIP — ANOMALY DETECTION PIPELINE")
    print("═" * 60)


def _should_run(stage: str, only_stage: Optional[str], skip: bool = False) -> bool:
    if skip:
        return False
    if only_stage:
        return stage == only_stage
    return True


def _skip(stage: str, name: str, explicit: bool = False):
    reason = '(--skip-eda)' if explicit else '(--only-stage specified)'
    print(f"\n  [SKIPPED] Stage {stage} — {name} {reason}")


def _final_summary(results: dict, elapsed: float):
    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE")
    print("═" * 60)

    stage_labels = {
        '01_eda':            'EDA',
        '02_cleaning':       'Cleaning',
        '03_clustering':     'Clustering',
        '04_detection':      'Anomaly Detection',
        '05_classification': 'Classification',
        '06_saving':         'Saving',
    }

    for key, label in stage_labels.items():
        if key in results:
            r = results[key]
            t = r.get('elapsed_sec', '?')
            print(f"  ✓  Stage {key[:2]} — {label:<22}  ({t}s)")
        else:
            print(f"  –  Stage {key[:2]} — {label:<22}  (skipped)")

    # High-level stats
    if '02_cleaning' in results:
        r = results['02_cleaning']
        print(f"\n  Records processed : {r['n_records']:,}  "
              f"(valid: {r['n_valid']:,})")

    if '04_detection' in results:
        r = results['04_detection']
        print(f"  Anomalies found   : {r['n_anomalies']:,}  "
              f"({r['anomaly_rate']:.2f}%)")
        for _m, _rec in r.get('recall_per_method', {}).items():
            print(f"  Recall ({_m:<13}): {_rec:.1%}")
        if 'recall_ensemble' in r:
            print(f"  Recall (ensemble )  : {r['recall_ensemble']:.1%}")

    if '06_saving' in results:
        r = results['06_saving']
        print(f"  DB logged         : {r['n_anomalies_logged']:,} records → {r['db_path']}")
        print(f"  Report            : {r['report']}")

    print(f"\n  Total elapsed     : {elapsed:.1f}s")
    print("═" * 60 + "\n")


# ── CLI entry-point ────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Twinship Full Anomaly Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python pipeline/run_pipeline.py

  # Skip EDA for faster re-runs
  python pipeline/run_pipeline.py --skip-eda

  # Run only anomaly detection stage
  python pipeline/run_pipeline.py --only-stage 04

  # Custom settings
  python pipeline/run_pipeline.py --data data1.csv --n-clusters 5 --threshold-pct 97
        """
    )
    parser.add_argument('--data',            default='data1.csv')
    parser.add_argument('--skip-eda',        action='store_true',
                        help='Skip Stage 01 (EDA) — useful for re-runs')
    parser.add_argument('--only-stage',      type=str, default=None,
                        help='Run only this stage number (01–06), e.g. --only-stage 04')
    parser.add_argument('--n-clusters',      type=int,   default=4)
    parser.add_argument('--n-components',    type=int,   default=3)
    parser.add_argument('--threshold-pct',   type=float, default=95.0)
    parser.add_argument('--random-state',    type=int,   default=42)
    parser.add_argument('--model-version',   default='clustered_svd_v1')
    parser.add_argument('--db',              default='anomalies.db')
    parser.add_argument('--methods',         nargs='+',
                        default=['svd', 'iforest', 'autoencoder'],
                        choices=['svd', 'iforest', 'autoencoder'],
                        help='Detection methods to run (default: all three)')
    parser.add_argument('--if-n-estimators', type=int,   default=100,
                        help='Isolation Forest number of trees')
    parser.add_argument('--ae-epochs',       type=int,   default=50,
                        help='Autoencoder training epochs')
    parser.add_argument('--ae-encoding-dim', type=int,   default=3,
                        help='Autoencoder latent dimension')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run_pipeline({
        'data':            args.data,
        'skip_eda':        args.skip_eda,
        'only_stage':      args.only_stage,
        'n_clusters':      args.n_clusters,
        'n_components':    args.n_components,
        'threshold_pct':   args.threshold_pct,
        'random_state':    args.random_state,
        'model_version':   args.model_version,
        'db':              args.db,
        'methods':         args.methods,
        'if_n_estimators': args.if_n_estimators,
        'ae_epochs':       args.ae_epochs,
        'ae_encoding_dim': args.ae_encoding_dim,
    })
