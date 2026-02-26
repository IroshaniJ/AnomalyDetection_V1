"""
Stage 04 — Anomaly Detection
============================
Runs three detection methods in parallel on the same operational mode clusters:

  1. Clustered SVD       – per-cluster truncated SVD reconstruction error (fast, linear)
  2. Clustered IForest   – per-cluster Isolation Forest (tree-based, non-linear)
  3. Clustered AE        – per-cluster Autoencoder reconstruction error (deep, non-linear)

Then combines results via **majority-vote ensemble** (flagged by ≥2/3 methods).
Constraint violations (negative power, invalid draft, extreme trim) are always
included in the final ``is_anomaly`` flag regardless of model votes.

Reads from ``results/02_cleaning/`` (cleaned data + feature matrix).

Writes:
  • ``results/04_anomaly_detection/predictions_svd.npy``
  • ``results/04_anomaly_detection/predictions_iforest.npy``
  • ``results/04_anomaly_detection/predictions_autoencoder.npy``
  • ``results/04_anomaly_detection/predictions_ensemble.npy``   ← majority vote
  • ``results/04_anomaly_detection/scores_svd.npy``
  • ``results/04_anomaly_detection/scores_iforest.npy``
  • ``results/04_anomaly_detection/scores_autoencoder.npy``
  • ``results/04_anomaly_detection/cluster_labels.npy``
  • ``results/04_anomaly_detection/detector_svd.pkl``           (= <model_version>.pkl)
  • ``results/04_anomaly_detection/detector_iforest.pkl``
  • ``results/04_anomaly_detection/detector_autoencoder.pkl``
  • ``results/04_anomaly_detection/annotated_data.pkl``         ← full df + all labels
  • ``results/04_anomaly_detection/detection_summary.csv``
  • ``results/04_anomaly_detection/method_comparison.csv``
  • ``results/04_anomaly_detection/plots/``

Standalone usage:
    python pipeline/stage_04_anomaly_detection.py
    python pipeline/stage_04_anomaly_detection.py --methods svd iforest autoencoder
    python pipeline/stage_04_anomaly_detection.py --methods svd iforest --threshold-pct 95
    python pipeline/stage_04_anomaly_detection.py --ae-epochs 30 --if-n-estimators 100

As a pipeline step:
    from pipeline.stage_04_anomaly_detection import run
    result = run(config)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Ensure project root is on sys.path ────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.clustered_svd_detector import ClusteredSVDAnomalyDetector
from src.preprocessing.data_loader import get_feature_columns
from benchmark.methods.isolation_forest import ClusteredIsolationForest
from benchmark.methods.autoencoder import ClusteredAutoencoder


# ── Public API ─────────────────────────────────────────────────────────────────

def run(config: Optional[dict] = None) -> dict:
    """
    Execute the anomaly detection stage with multiple methods.

    Args:
        config: Optional dict with keys:
            ``cleaning_dir``      – stage-02 output dir  (default: 'results/02_cleaning')
            ``data``              – fallback raw CSV      (default: 'data1.csv')
            ``output_dir``        – where to write        (default: 'results/04_anomaly_detection')
            ``methods``           – list of methods to run: any subset of
                                    ['svd', 'iforest', 'autoencoder']
                                    (default: all three)
            ``n_clusters``        – clusters (default: 4)
            ``n_components``      – SVD components per cluster (default: 3)
            ``threshold_pct``     – anomaly threshold percentile (default: 95.0)
            ``random_state``      – random seed (default: 42)
            ``model_version``     – artefact tag (default: 'clustered_svd_v1')
            ``if_n_estimators``   – IForest trees (default: 100)
            ``ae_epochs``         – Autoencoder training epochs (default: 50)
            ``ae_encoding_dim``   – Autoencoder latent dim (default: 3)

    Returns:
        dict with per-method results + ensemble summary.
    """
    config = config or {}
    cleaning_dir    = Path(config.get('cleaning_dir',   'results/02_cleaning'))
    data_path       = config.get('data', 'data1.csv')
    output_dir      = Path(config.get('output_dir', 'results/04_anomaly_detection'))
    methods         = config.get('methods', ['svd', 'iforest', 'autoencoder'])
    n_clusters      = int(config.get('n_clusters',    4))
    n_components    = int(config.get('n_components',  3))
    threshold_pct   = float(config.get('threshold_pct', 95.0))
    random_state    = int(config.get('random_state',  42))
    model_version   = config.get('model_version', 'clustered_svd_v1')
    if_n_estimators = int(config.get('if_n_estimators', 100))
    ae_epochs       = int(config.get('ae_epochs', 50))
    ae_encoding_dim = int(config.get('ae_encoding_dim', 3))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)

    print("\n" + "─" * 60)
    print("STAGE 04 — ANOMALY DETECTION")
    print(f"  Methods: {methods}")
    print("─" * 60)

    t0 = time.time()

    # ── Load data ──────────────────────────────────────────────────────────────
    df_full, X, valid_mask, feature_cols = _load_data(cleaning_dir, data_path)
    print(f"  [Detection] Feature matrix: {X.shape[0]:,} × {X.shape[1]}")

    # ── Run each method ────────────────────────────────────────────────────────
    method_predictions = {}   # method → np.array of -1/1
    method_scores      = {}   # method → np.array of floats
    method_clusters    = {}   # method → np.array of ints
    method_thresholds  = {}   # method → dict{cluster_id: threshold}
    method_times       = {}   # method → seconds
    method_recall      = {}   # method → recall on known violations

    constraint_mask = _get_constraint_mask(df_full, valid_mask)

    if 'svd' in methods:
        preds, scores, clusters, thresholds, elapsed = _run_svd(
            X, feature_cols, n_clusters, n_components, threshold_pct,
            random_state, output_dir, model_version
        )
        method_predictions['svd'] = preds
        method_scores['svd']      = scores
        method_clusters['svd']    = clusters
        method_thresholds['svd']  = thresholds
        method_times['svd']       = elapsed
        method_recall['svd']      = _recall(preds, constraint_mask)
        _print_method_summary('SVD', preds, method_recall['svd'], elapsed)

    if 'iforest' in methods:
        preds, scores, clusters, thresholds, elapsed = _run_iforest(
            X, n_clusters, threshold_pct, if_n_estimators, random_state,
            output_dir
        )
        method_predictions['iforest'] = preds
        method_scores['iforest']      = scores
        method_clusters['iforest']    = clusters
        method_thresholds['iforest']  = thresholds
        method_times['iforest']       = elapsed
        method_recall['iforest']      = _recall(preds, constraint_mask)
        _print_method_summary('Isolation Forest', preds, method_recall['iforest'], elapsed)

    if 'autoencoder' in methods:
        preds, scores, clusters, thresholds, elapsed = _run_autoencoder(
            X, n_clusters, threshold_pct, ae_encoding_dim, ae_epochs,
            random_state, output_dir
        )
        method_predictions['autoencoder'] = preds
        method_scores['autoencoder']      = scores
        method_clusters['autoencoder']    = clusters
        method_thresholds['autoencoder']  = thresholds
        method_times['autoencoder']       = elapsed
        method_recall['autoencoder']      = _recall(preds, constraint_mask)
        _print_method_summary('Autoencoder', preds, method_recall['autoencoder'], elapsed)

    # ── Ensemble (majority vote) ───────────────────────────────────────────────
    all_preds_list = list(method_predictions.values())
    if len(all_preds_list) >= 2:
        votes = np.stack([p == -1 for p in all_preds_list], axis=1).sum(axis=1)
        threshold_votes = max(1, len(all_preds_list) // 2 + 1)   # strict majority
        ensemble_preds = np.where(votes >= threshold_votes, -1, 1)
    elif len(all_preds_list) == 1:
        ensemble_preds = all_preds_list[0].copy()
        threshold_votes = 1
    else:
        raise ValueError("No methods produced predictions.")

    ensemble_recall = _recall(ensemble_preds, constraint_mask)
    n_ensemble = int((ensemble_preds == -1).sum())
    print(f"\n  [Ensemble]   {n_ensemble:,} anomalies  ({n_ensemble/len(ensemble_preds)*100:.2f}%)  "
          f"recall={ensemble_recall:.1%}  (majority ≥{threshold_votes}/{len(all_preds_list)})")

    # ── Annotate full DataFrame ────────────────────────────────────────────────
    # Use SVD cluster labels if available, otherwise first available method
    primary_clusters = method_clusters.get('svd',
                       method_clusters.get('iforest',
                       method_clusters.get('autoencoder')))

    # SVD cluster names for readable labels
    svd_pkl = output_dir / f'{model_version}.pkl'
    cluster_names = _load_cluster_names(svd_pkl) if 'svd' in methods else {}

    df_full = _annotate_df(
        df_full, valid_mask, feature_cols,
        method_predictions, method_scores,
        ensemble_preds, primary_clusters, cluster_names
    )

    # ── Save numpy artefacts ───────────────────────────────────────────────────
    for method, preds in method_predictions.items():
        np.save(output_dir / f'predictions_{method}.npy', preds)
        np.save(output_dir / f'scores_{method}.npy',      method_scores[method])

    np.save(output_dir / 'predictions_ensemble.npy', ensemble_preds)
    np.save(output_dir / 'cluster_labels.npy', primary_clusters)

    df_full.to_pickle(output_dir / 'annotated_data.pkl')

    # ── Detection summary CSV ──────────────────────────────────────────────────
    summary_df = _build_detection_summary(
        method_predictions, method_scores, method_thresholds, method_recall,
        method_times, ensemble_preds, ensemble_recall,
        primary_clusters, cluster_names
    )
    summary_df.to_csv(output_dir / 'detection_summary.csv', index=False)

    # ── Method comparison CSV ──────────────────────────────────────────────────
    comparison_df = _build_comparison_table(
        method_predictions, method_scores, method_recall, method_times
    )
    comparison_df.to_csv(output_dir / 'method_comparison.csv', index=False)

    # ── Plots ──────────────────────────────────────────────────────────────────
    plots_dir = output_dir / 'plots'
    score_plot      = _plot_score_distributions(method_scores, method_predictions, plots_dir)
    recall_plot     = _plot_recall_comparison(method_recall, ensemble_recall, plots_dir)
    agree_plot      = _plot_agreement_heatmap(method_predictions, plots_dir)
    ts_plot         = _plot_anomalies_timeseries(df_full, feature_cols, plots_dir)
    cluster_scatter = _plot_cluster_anomaly_scatter(df_full, feature_cols, plots_dir)
    anomaly_dist    = _plot_anomaly_feature_distribution(df_full, feature_cols, plots_dir)
    seq_plot        = _plot_anomaly_sequence_structure(df_full, plots_dir)
    spatial_plot    = _plot_spatial_anomalies(df_full, plots_dir)
    rate_plot       = _plot_anomaly_rate_by_cluster(df_full, plots_dir)
    power_speed     = _plot_power_speed_anomalies(df_full, plots_dir)

    # ── Average threshold (SVD primary; fallback to mean of all) ──────────────
    if 'svd' in method_thresholds and method_thresholds['svd']:
        avg_threshold = float(np.mean(list(method_thresholds['svd'].values())))
    else:
        all_thr = [v for d in method_thresholds.values() for v in (d.values() if isinstance(d, dict) else [])]
        avg_threshold = float(np.mean(all_thr)) if all_thr else 0.1

    elapsed_total = time.time() - t0
    print(f"\n  [Detection] Done in {elapsed_total:.1f}s  →  {output_dir}/")

    return {
        'methods_run':       methods,
        'n_anomalies':       n_ensemble,
        'anomaly_rate':      round(n_ensemble / len(ensemble_preds) * 100, 3),
        'recall_per_method': method_recall,
        'recall_ensemble':   ensemble_recall,
        'avg_threshold':     round(avg_threshold, 6),
        'avg_threshold_val': avg_threshold,
        'cluster_names':     cluster_names,
        'predictions_npy':   str(output_dir / 'predictions_ensemble.npy'),
        'scores_npy':        str(output_dir / 'scores_svd.npy') if 'svd' in methods
                             else str(output_dir / f"scores_{methods[0]}.npy"),
        'annotated_pkl':     str(output_dir / 'annotated_data.pkl'),
        'detection_summary': str(output_dir / 'detection_summary.csv'),
        'method_comparison': str(output_dir / 'method_comparison.csv'),
        'score_plot':        score_plot,
        'recall_plot':       recall_plot,
        'agree_plot':        agree_plot,
        'ts_plot':           ts_plot,
        'cluster_scatter':   cluster_scatter,
        'anomaly_dist':      anomaly_dist,
        'seq_plot':          seq_plot,
        'spatial_plot':      spatial_plot,
        'rate_plot':         rate_plot,
        'power_speed':       power_speed,
        'output_dir':        str(output_dir),
        'elapsed_sec':       round(elapsed_total, 2),
    }


# ── Method runners ─────────────────────────────────────────────────────────────

def _run_svd(X, feature_cols, n_clusters, n_components, threshold_pct,
             random_state, output_dir, model_version):
    print(f"\n  ── Method: SVD (n_clusters={n_clusters}, n_components={n_components}, "
          f"threshold_pct={threshold_pct}) ──")
    t0 = time.time()
    detector = ClusteredSVDAnomalyDetector(
        n_clusters=n_clusters,
        n_components=n_components,
        threshold_percentile=threshold_pct,
        random_state=random_state,
    )
    predictions, scores, cluster_labels = detector.fit_predict(X, feature_cols)
    # Save under both the versioned name and a stable name
    detector.save(str(output_dir / f'{model_version}.pkl'))
    detector.save(str(output_dir / 'detector_svd.pkl'))

    cluster_names = detector.get_cluster_names()
    print(f"  [SVD] Per-cluster breakdown:")
    for cid, cname in cluster_names.items():
        mask_c  = cluster_labels == cid
        total_c = int(mask_c.sum())
        anom_c  = int((predictions[mask_c] == -1).sum())
        thr_c   = detector.cluster_thresholds.get(cid, float('nan'))
        print(f"        {cname:<35}: {anom_c:>4}/{total_c:<6}  ({anom_c/total_c*100:.1f}%)  "
              f"threshold={thr_c:.6f}")

    return predictions, scores, cluster_labels, detector.cluster_thresholds, time.time() - t0


def _run_iforest(X, n_clusters, threshold_pct, n_estimators, random_state, output_dir):
    print(f"\n  ── Method: Isolation Forest (n_clusters={n_clusters}, "
          f"n_estimators={n_estimators}) ──")
    t0 = time.time()
    contamination = max(0.001, min(0.5, 1.0 - threshold_pct / 100.0))
    detector = ClusteredIsolationForest(
        n_clusters=n_clusters,
        threshold_percentile=threshold_pct,
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    detector.fit(X)
    predictions = detector.predict(X)
    scores      = detector.score_samples(X)
    detector.save(str(output_dir / 'detector_iforest.pkl'))
    return predictions, scores, detector.cluster_labels_, detector.cluster_thresholds_, time.time() - t0


def _run_autoencoder(X, n_clusters, threshold_pct, encoding_dim, epochs,
                     random_state, output_dir):
    print(f"\n  ── Method: Autoencoder (n_clusters={n_clusters}, "
          f"encoding_dim={encoding_dim}, epochs={epochs}) ──")
    t0 = time.time()
    detector = ClusteredAutoencoder(
        n_clusters=n_clusters,
        threshold_percentile=threshold_pct,
        encoding_dim=encoding_dim,
        epochs=epochs,
        random_state=random_state,
    )
    detector.fit(X)
    predictions = detector.predict(X)
    scores      = detector.score_samples(X)
    detector.save(str(output_dir / 'detector_autoencoder.pkl'))
    return predictions, scores, detector.cluster_labels_, detector.cluster_thresholds_, time.time() - t0


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_data(cleaning_dir: Path, data_path: str):
    """Return (df_full, X_valid, valid_mask_bool_series, feature_cols)."""
    feature_cols = get_feature_columns()

    pkl_path  = cleaning_dir / 'cleaned_data.pkl'
    npy_path  = cleaning_dir / 'feature_matrix.npy'
    mask_path = cleaning_dir / 'valid_mask.npy'

    if pkl_path.exists() and npy_path.exists() and mask_path.exists():
        print(f"  [Detection] Loading cleaned data from {cleaning_dir}/")
        df_full        = pd.read_pickle(pkl_path)
        X              = np.load(npy_path)
        valid_mask_arr = np.load(mask_path).astype(bool)
        valid_mask     = pd.Series(valid_mask_arr, index=df_full.index)
    else:
        print(f"  [Detection] Stage-02 output not found — loading from {data_path}")
        from src.preprocessing.data_loader import load_and_clean
        df_full     = load_and_clean(data_path)
        mask_series = ~df_full[feature_cols].isna().any(axis=1)
        valid_mask  = mask_series
        X = df_full.loc[valid_mask, feature_cols].values.astype(np.float64)

    return df_full, X, valid_mask, feature_cols


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_constraint_mask(df_full: pd.DataFrame, valid_mask) -> np.ndarray:
    """Boolean mask over valid rows indicating known constraint violations."""
    if 'flag_any_constraint' in df_full.columns:
        return df_full.loc[valid_mask, 'flag_any_constraint'].values.astype(bool)
    return np.zeros(int(valid_mask.sum()), dtype=bool)


def _recall(predictions: np.ndarray, constraint_mask: np.ndarray) -> float:
    if constraint_mask.sum() == 0:
        return 0.0
    return float((predictions[constraint_mask] == -1).mean())


def _print_method_summary(name: str, predictions: np.ndarray,
                           recall: float, elapsed: float):
    n    = int((predictions == -1).sum())
    rate = n / len(predictions) * 100
    print(f"  [{name:<20}] {n:,} anomalies ({rate:.2f}%)  "
          f"recall={recall:.1%}  [{elapsed:.1f}s]")


def _load_cluster_names(pkl_path: Path) -> dict:
    if not pkl_path.exists():
        return {}
    import joblib
    try:
        data = joblib.load(pkl_path)
        clusterer = data.get('clusterer') if isinstance(data, dict) else getattr(data, 'clusterer', None)
        if clusterer and hasattr(clusterer, 'cluster_names'):
            return clusterer.cluster_names
    except Exception:
        pass
    return {}


def _annotate_df(df_full, valid_mask, feature_cols,
                  method_predictions, method_scores,
                  ensemble_preds, primary_clusters, cluster_names):
    """Add all method columns and ensemble to the full DataFrame."""
    df_full = df_full.copy()

    # Initialise columns for each method
    for method in method_predictions:
        df_full[f'is_{method}_anomaly'] = False
        df_full[f'score_{method}']      = float('nan')

    df_full['anomaly_score']       = float('nan')
    df_full['cluster_id']          = -1
    df_full['cluster_name']        = 'Unknown'
    df_full['is_svd_anomaly']      = False   # kept for downstream compat
    df_full['is_ensemble_anomaly'] = False

    # Fill valid rows
    for method, preds in method_predictions.items():
        df_full.loc[valid_mask, f'is_{method}_anomaly'] = preds == -1
        df_full.loc[valid_mask, f'score_{method}']      = method_scores[method]

    # Use SVD score as primary 'anomaly_score'; fallback to first available
    primary_score_col = (
        'score_svd' if 'score_svd' in df_full.columns
        else next((f'score_{m}' for m in method_predictions), None)
    )
    if primary_score_col:
        df_full['anomaly_score'] = df_full[primary_score_col]

    # Cluster labels
    df_full.loc[valid_mask, 'cluster_id']   = primary_clusters
    df_full.loc[valid_mask, 'cluster_name'] = pd.array(
        [cluster_names.get(int(c), f'Cluster_{c}') for c in primary_clusters]
    )

    # Ensemble & backward-compat SVD alias
    df_full.loc[valid_mask, 'is_ensemble_anomaly'] = ensemble_preds == -1
    df_full.loc[valid_mask, 'is_svd_anomaly'] = (
        method_predictions.get('svd', ensemble_preds) == -1
    )

    # Final combined flag: ensemble OR known constraint violations
    if 'flag_any_constraint' in df_full.columns:
        df_full['is_anomaly'] = df_full['is_ensemble_anomaly'] | df_full['flag_any_constraint']
    else:
        df_full['is_anomaly'] = df_full['is_ensemble_anomaly']

    return df_full


def _build_detection_summary(method_predictions, method_scores, method_thresholds,
                               method_recall, method_times,
                               ensemble_preds, ensemble_recall,
                               primary_clusters, cluster_names) -> pd.DataFrame:
    """Per-cluster × per-method anomaly counts."""
    rows = []
    unique_clusters = np.unique(primary_clusters)

    for cid in unique_clusters:
        mask_c = primary_clusters == cid
        cname  = cluster_names.get(int(cid), f'Cluster_{cid}')
        total  = int(mask_c.sum())

        row = {'cluster_id': int(cid), 'cluster_name': cname, 'total': total}
        for method, preds in method_predictions.items():
            n_anom = int((preds[mask_c] == -1).sum())
            row[f'{method}_anomalies'] = n_anom
            row[f'{method}_rate_pct']  = round(n_anom / total * 100, 2) if total > 0 else 0.0
            thr = method_thresholds.get(method, {})
            row[f'{method}_threshold'] = (
                thr.get(int(cid), float('nan')) if isinstance(thr, dict) else float('nan')
            )

        n_ens = int((ensemble_preds[mask_c] == -1).sum())
        row['ensemble_anomalies'] = n_ens
        row['ensemble_rate_pct']  = round(n_ens / total * 100, 2) if total > 0 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def _build_comparison_table(method_predictions, method_scores,
                              method_recall, method_times) -> pd.DataFrame:
    """High-level method comparison table."""
    rows = []
    for method, preds in method_predictions.items():
        n      = int((preds == -1).sum())
        scores = method_scores[method]
        rows.append({
            'method':       method,
            'n_anomalies':  n,
            'anomaly_rate': round(n / len(preds) * 100, 3),
            'recall_known': round(method_recall.get(method, 0.0) * 100, 1),
            'avg_score':    round(float(scores.mean()), 6),
            'max_score':    round(float(scores.max()),  6),
            'fit_time_sec': round(method_times.get(method, 0.0), 2),
        })
    return pd.DataFrame(rows).sort_values('recall_known', ascending=False)


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _plot_score_distributions(method_scores: dict, method_predictions: dict,
                                out_dir: Path) -> str:
    n = len(method_scores)
    if n == 0:
        return ''

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colours = {'svd': 'steelblue', 'iforest': 'forestgreen', 'autoencoder': 'darkorange'}

    for ax, (method, scores) in zip(axes, method_scores.items()):
        preds  = method_predictions[method]
        normal = scores[preds == 1]
        anom   = scores[preds == -1]
        cap    = np.percentile(scores, 99.5)
        bins   = np.linspace(0, cap, 70)

        ax.hist(normal, bins=bins, alpha=0.7, color=colours.get(method, 'steelblue'),
                label=f'Normal (n={len(normal):,})', density=True)
        ax.hist(anom,   bins=bins, alpha=0.7, color='tomato',
                label=f'Anomaly (n={len(anom):,})', density=True)
        thr_line = np.percentile(scores, 95)
        ax.axvline(thr_line, color='black', linestyle='--', lw=1.2, alpha=0.7, label='p95')
        ax.set_title(method.upper(), fontsize=11, fontweight='bold')
        ax.set_xlabel('Anomaly Score', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle('Anomaly Score Distributions by Method', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'score_distributions.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


def _plot_recall_comparison(method_recall: dict, ensemble_recall: float,
                              out_dir: Path) -> str:
    labels  = list(method_recall.keys()) + ['ensemble']
    values  = [method_recall[m] * 100 for m in method_recall] + [ensemble_recall * 100]
    colours = ['steelblue' if l != 'ensemble' else 'tomato' for l in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=colours, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    ax.set_ylim(0, max(values) * 1.25 + 5 if values else 110)
    ax.set_ylabel('Recall on Known Violations (%)', fontsize=11)
    ax.set_title('Recall on Known Constraint Violations by Method', fontsize=12, fontweight='bold')
    ax.axhline(100, color='gray', linestyle='--', lw=0.8, alpha=0.5)
    plt.tight_layout()
    path = out_dir / 'recall_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


def _plot_agreement_heatmap(method_predictions: dict, out_dir: Path) -> str:
    if len(method_predictions) < 2:
        return ''

    methods = list(method_predictions.keys())
    n = len(methods)
    agree = np.zeros((n, n))

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            agree[i, j] = (method_predictions[m1] == method_predictions[m2]).mean() * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(agree, vmin=80, vmax=100, cmap='YlGn')
    plt.colorbar(im, ax=ax, label='Agreement (%)')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([m.upper() for m in methods], fontsize=10)
    ax.set_yticklabels([m.upper() for m in methods], fontsize=10)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{agree[i, j]:.1f}%', ha='center', va='center',
                    fontsize=11, fontweight='bold')

    ax.set_title('Pairwise Prediction Agreement', fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'method_agreement_heatmap.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


# ── 1. Time-series: all key features with anomaly overlay ─────────────────────

def _plot_anomalies_timeseries(df: pd.DataFrame, feature_cols: list,
                                out_dir: Path) -> str:
    """Multi-panel time-series — one row per feature, anomalies highlighted."""
    if 'Date' not in df.columns:
        return ''

    plot_cols = [c for c in feature_cols if c in df.columns]
    n = len(plot_cols)
    if n == 0:
        return ''

    df_ts = df.dropna(subset=['Date']).sort_values('Date').copy()
    anom  = df_ts['is_anomaly'].astype(bool) if 'is_anomaly' in df_ts else pd.Series(False, index=df_ts.index)

    fig, axes = plt.subplots(n, 1, figsize=(18, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colours = plt.cm.tab10.colors

    for ax, col, c in zip(axes, plot_cols, colours):
        vals = df_ts[col]
        ax.plot(df_ts['Date'], vals, lw=0.6, alpha=0.55, color=c, label=col)
        anom_dates = df_ts.loc[anom, 'Date']
        anom_vals  = df_ts.loc[anom, col]
        ax.scatter(anom_dates, anom_vals, s=12, color='tomato', zorder=5,
                   alpha=0.8, label='Anomaly')
        ax.set_ylabel(col, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.4)

    axes[-1].set_xlabel('Date', fontsize=10)
    fig.suptitle('Feature Time-Series with Anomaly Overlay', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'timeseries_all_features.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


# ── 2. Cluster × Anomaly scatter grid (pairwise key features) ─────────────────

def _plot_cluster_anomaly_scatter(df: pd.DataFrame, feature_cols: list,
                                   out_dir: Path) -> str:
    """2×2 scatter grid of the most discriminating feature pairs, coloured by cluster, anomalies marked."""
    pairs = [
        ('GPSSpeed_kn',          'Main_Engine_Power_kW'),
        ('Main_Engine_Power_kW', 'Fuel_Consumption_t_per_day'),
        ('Speed_rpm',            'Main_Engine_Power_kW'),
        ('Avg_draft_m',          'GPSSpeed_kn'),
    ]
    # filter to pairs that actually exist
    pairs = [(x, y) for x, y in pairs if x in df.columns and y in df.columns]
    if not pairs:
        return ''

    n_pairs = len(pairs)
    ncols = 2
    nrows = (n_pairs + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes = np.array(axes).flatten()

    cmap = matplotlib.colormaps['tab10']
    cluster_col = 'cluster_name' if 'cluster_name' in df.columns else None
    clusters = sorted(df[cluster_col].dropna().unique()) if cluster_col else ['All']
    colour_map = {c: cmap(i / max(len(clusters), 1)) for i, c in enumerate(clusters)}

    anom_col = 'is_anomaly' if 'is_anomaly' in df.columns else None

    for ax, (xcol, ycol) in zip(axes, pairs):
        sub = df[[xcol, ycol]].copy()
        if cluster_col:
            sub['_cluster'] = df[cluster_col].fillna('Unknown')
        if anom_col:
            sub['_anom'] = df[anom_col].astype(bool)

        # plot normal points by cluster
        if cluster_col:
            for cname in clusters:
                mask = (sub['_cluster'] == cname) & (~sub.get('_anom', pd.Series(False, index=sub.index)))
                ax.scatter(sub.loc[mask, xcol], sub.loc[mask, ycol],
                           s=4, alpha=0.25, color=colour_map[cname], label=str(cname))
        else:
            normal_mask = ~sub.get('_anom', pd.Series(False, index=sub.index))
            ax.scatter(sub.loc[normal_mask, xcol], sub.loc[normal_mask, ycol],
                       s=4, alpha=0.25, color='steelblue', label='Normal')

        # overlay anomalies
        if anom_col:
            anom_mask = sub['_anom']
            ax.scatter(sub.loc[anom_mask, xcol], sub.loc[anom_mask, ycol],
                       s=30, alpha=0.85, color='tomato', zorder=6,
                       edgecolors='darkred', linewidths=0.4, label='Anomaly')

        ax.set_xlabel(xcol, fontsize=9)
        ax.set_ylabel(ycol, fontsize=9)
        ax.set_title(f'{xcol} vs {ycol}', fontsize=10, fontweight='bold')
        handles, labels = ax.get_legend_handles_labels()
        # deduplicate
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, markerscale=2)

    for ax in axes[len(pairs):]:
        ax.set_visible(False)

    fig.suptitle('Cluster & Anomaly Scatter — Key Feature Pairs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'cluster_anomaly_scatter.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


# ── 3. Anomaly feature distribution (violin / box per feature) ────────────────

def _plot_anomaly_feature_distribution(df: pd.DataFrame, feature_cols: list,
                                        out_dir: Path) -> str:
    """Box-plot comparison: normal vs anomaly distribution for each feature."""
    plot_cols = [c for c in feature_cols if c in df.columns]
    if not plot_cols or 'is_anomaly' not in df.columns:
        return ''

    df_plot = df[plot_cols + ['is_anomaly']].copy()
    df_plot['Group'] = df_plot['is_anomaly'].map({True: 'Anomaly', False: 'Normal', 1: 'Anomaly', 0: 'Normal'})

    n = len(plot_cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, col in zip(axes, plot_cols):
        groups = [df_plot.loc[df_plot['Group'] == g, col].dropna().values
                  for g in ['Normal', 'Anomaly']]
        bp = ax.boxplot(groups, tick_labels=['Normal', 'Anomaly'], patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(linewidth=0.8),
                        flierprops=dict(marker='.', markersize=2, alpha=0.3))
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][0].set_alpha(0.6)
        if len(bp['boxes']) > 1:
            bp['boxes'][1].set_facecolor('tomato')
            bp['boxes'][1].set_alpha(0.7)
        ax.set_title(col, fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle('Feature Distributions: Normal vs Anomaly', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'anomaly_feature_distributions.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


# ── 4. Sequential anomaly structure (run-length & gap analysis) ───────────────

def _plot_anomaly_sequence_structure(df: pd.DataFrame, out_dir: Path) -> str:
    """
    Shows:
      - top panel: anomaly score time-series with run-length coloured spans
      - middle panel: inter-anomaly gap distribution
      - bottom panel: rolling anomaly rate (1-hour window)
    """
    if 'Date' not in df.columns or 'is_anomaly' not in df.columns:
        return ''

    df_ts = df[['Date', 'anomaly_score', 'is_anomaly']].dropna(subset=['Date']).sort_values('Date').copy()
    df_ts['is_anomaly'] = df_ts['is_anomaly'].astype(bool)

    fig, axes = plt.subplots(3, 1, figsize=(18, 12))

    # ── Panel 1: score timeline with consecutive-run shading ──────────────────
    ax = axes[0]
    ax.plot(df_ts['Date'], df_ts['anomaly_score'], lw=0.6, color='steelblue', alpha=0.5)

    # identify consecutive anomaly runs
    in_run = False
    run_start = None
    run_lengths = []
    for _, row in df_ts.iterrows():
        if row['is_anomaly'] and not in_run:
            in_run = True
            run_start = row['Date']
            run_len = 1
        elif row['is_anomaly'] and in_run:
            run_len += 1
        elif not row['is_anomaly'] and in_run:
            ax.axvspan(run_start, row['Date'], alpha=0.25, color='tomato')
            run_lengths.append(run_len)
            in_run = False
    if in_run:
        run_lengths.append(run_len)

    ax.set_ylabel('Anomaly Score', fontsize=10)
    ax.set_title('Anomaly Score — Consecutive Run Spans', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', labelsize=8)

    # ── Panel 2: run-length distribution ──────────────────────────────────────
    ax2 = axes[1]
    if run_lengths:
        max_run = max(run_lengths)
        bins = np.arange(0.5, min(max_run + 1, 51) + 0.5, 1)
        ax2.hist(run_lengths, bins=bins, color='tomato', alpha=0.75, edgecolor='white')
        ax2.set_xlabel('Consecutive Anomaly Run Length (# records)', fontsize=10)
        ax2.set_ylabel('Count', fontsize=10)
        ax2.set_title(f'Consecutive Anomaly Run-Length Distribution  '
                      f'(total runs: {len(run_lengths)}, max: {max_run})', fontsize=11, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No anomaly runs', ha='center', va='center', transform=ax2.transAxes)

    # ── Panel 3: rolling 1-hour anomaly rate ──────────────────────────────────
    ax3 = axes[2]
    df_roll = df_ts.set_index('Date')['is_anomaly'].astype(float)
    window = '1h'
    try:
        rolling_rate = df_roll.rolling(window, min_periods=1).mean() * 100
        ax3.fill_between(rolling_rate.index, rolling_rate.values,
                         alpha=0.6, color='darkorange', label='1-h rolling rate')
        ax3.axhline(5, color='gray', linestyle='--', lw=0.8, alpha=0.6, label='5% ref line')
        ax3.set_ylabel('Anomaly Rate (%)', fontsize=10)
        ax3.set_xlabel('Date', fontsize=10)
        ax3.set_title('Rolling 1-Hour Anomaly Rate', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9)
    except Exception:
        ax3.set_visible(False)

    fig.suptitle('Sequential Anomaly Structure', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'anomaly_sequence_structure.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


# ── 5. Spatial anomaly map (GPS track coloured by anomaly) ────────────────────

def _plot_spatial_anomalies(df: pd.DataFrame, out_dir: Path) -> str:
    """GPS track plot with anomalies highlighted and score used as marker size."""
    if 'GPS_LAT' not in df.columns or 'GPS_LON' not in df.columns:
        return ''

    df_sp = df[['GPS_LAT', 'GPS_LON', 'is_anomaly', 'anomaly_score',
                'cluster_name']].dropna(subset=['GPS_LAT', 'GPS_LON']).copy()
    if df_sp.empty:
        return ''

    df_sp['is_anomaly'] = df_sp['is_anomaly'].astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ── Left: track coloured by cluster ───────────────────────────────────────
    ax = axes[0]
    cmap = matplotlib.colormaps['tab10']
    clusters = sorted(df_sp['cluster_name'].dropna().unique()) if 'cluster_name' in df_sp else []
    colour_map = {c: cmap(i / max(len(clusters), 1)) for i, c in enumerate(clusters)}

    if clusters:
        for cname in clusters:
            mask = df_sp['cluster_name'] == cname
            ax.scatter(df_sp.loc[mask, 'GPS_LON'], df_sp.loc[mask, 'GPS_LAT'],
                       s=3, alpha=0.3, color=colour_map[cname], label=str(cname))
    else:
        ax.scatter(df_sp['GPS_LON'], df_sp['GPS_LAT'], s=3, alpha=0.3, color='steelblue')

    # overlay anomalies
    anom_mask = df_sp['is_anomaly']
    ax.scatter(df_sp.loc[anom_mask, 'GPS_LON'], df_sp.loc[anom_mask, 'GPS_LAT'],
               s=25, alpha=0.85, color='tomato', zorder=6,
               edgecolors='darkred', linewidths=0.4, label='Anomaly')
    ax.set_xlabel('Longitude', fontsize=10); ax.set_ylabel('Latitude', fontsize=10)
    ax.set_title('GPS Track — Coloured by Cluster', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, markerscale=2)

    # ── Right: anomaly score as colour intensity on the route ─────────────────
    ax2 = axes[1]
    scores = df_sp['anomaly_score'].fillna(0).values
    sc = ax2.scatter(df_sp['GPS_LON'], df_sp['GPS_LAT'],
                     c=scores, cmap='YlOrRd', s=4, alpha=0.6,
                     vmin=np.percentile(scores, 5), vmax=np.percentile(scores, 99))
    plt.colorbar(sc, ax=ax2, label='Anomaly Score')
    ax2.set_xlabel('Longitude', fontsize=10); ax2.set_ylabel('Latitude', fontsize=10)
    ax2.set_title('GPS Track — Anomaly Score Intensity', fontsize=11, fontweight='bold')

    fig.suptitle('Spatial Anomaly Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'spatial_anomalies.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


# ── 6. Anomaly rate per cluster (bar + pie) ───────────────────────────────────

def _plot_anomaly_rate_by_cluster(df: pd.DataFrame, out_dir: Path) -> str:
    """Grouped bar chart of anomaly counts per cluster, with anomaly rate labels."""
    if 'cluster_name' not in df.columns or 'is_anomaly' not in df.columns:
        return ''

    summary = (df.groupby('cluster_name')['is_anomaly']
               .agg(['sum', 'count'])
               .rename(columns={'sum': 'anomalies', 'count': 'total'})
               .reset_index())
    summary['rate'] = summary['anomalies'] / summary['total'] * 100
    summary = summary.sort_values('rate', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # bar chart
    ax = axes[0]
    x = np.arange(len(summary))
    w = 0.4
    b1 = ax.bar(x - w/2, summary['total'],    width=w, label='Total',    color='steelblue', alpha=0.7)
    b2 = ax.bar(x + w/2, summary['anomalies'], width=w, label='Anomaly', color='tomato',    alpha=0.85)
    for bar, rate in zip(b2, summary['rate']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9, color='darkred')
    ax.set_xticks(x)
    ax.set_xticklabels(summary['cluster_name'], rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Record count', fontsize=10)
    ax.set_title('Anomaly Count per Operational Cluster', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    # pie chart — share of anomalies
    ax2 = axes[1]
    wedge_colours = plt.cm.tab10.colors[:len(summary)]
    ax2.pie(summary['anomalies'], labels=summary['cluster_name'],
            autopct='%1.1f%%', colors=wedge_colours,
            startangle=140, textprops={'fontsize': 9})
    ax2.set_title('Anomaly Share by Cluster', fontsize=11, fontweight='bold')

    fig.suptitle('Anomaly Rate by Operational Mode', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'anomaly_rate_by_cluster.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


# ── 7. Power–Speed anomaly scatter (key diagnostic plot) ─────────────────────

def _plot_power_speed_anomalies(df: pd.DataFrame, out_dir: Path) -> str:
    """
    Two-panel diagnostic:
      left  — Speed vs Power coloured by ensemble anomaly
      right — Speed vs Fuel Consumption coloured by anomaly score
    """
    needed = ['GPSSpeed_kn', 'Main_Engine_Power_kW',
              'Fuel_Consumption_t_per_day', 'is_anomaly', 'anomaly_score']
    if not all(c in df.columns for c in needed):
        return ''

    df_p = df[needed].dropna(subset=['GPSSpeed_kn', 'Main_Engine_Power_kW']).copy()
    df_p['is_anomaly'] = df_p['is_anomaly'].astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left: Speed vs Power, anomaly overlay ─────────────────────────────────
    ax = axes[0]
    normal_mask = ~df_p['is_anomaly']
    ax.scatter(df_p.loc[normal_mask, 'GPSSpeed_kn'],
               df_p.loc[normal_mask, 'Main_Engine_Power_kW'],
               s=3, alpha=0.2, color='steelblue', label='Normal')
    ax.scatter(df_p.loc[df_p['is_anomaly'], 'GPSSpeed_kn'],
               df_p.loc[df_p['is_anomaly'], 'Main_Engine_Power_kW'],
               s=20, alpha=0.85, color='tomato', zorder=6,
               edgecolors='darkred', linewidths=0.3, label='Anomaly')
    ax.set_xlabel('GPS Speed (kn)', fontsize=10)
    ax.set_ylabel('Main Engine Power (kW)', fontsize=10)
    ax.set_title('Speed vs Engine Power', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    # ── Right: Speed vs Fuel, colour = anomaly score ──────────────────────────
    ax2 = axes[1]
    df_f = df_p.dropna(subset=['Fuel_Consumption_t_per_day', 'anomaly_score'])
    scores = df_f['anomaly_score'].values
    sc = ax2.scatter(df_f['GPSSpeed_kn'], df_f['Fuel_Consumption_t_per_day'],
                     c=scores, cmap='YlOrRd', s=5, alpha=0.5,
                     vmin=np.percentile(scores, 5), vmax=np.percentile(scores, 97))
    plt.colorbar(sc, ax=ax2, label='Anomaly Score')
    ax2.set_xlabel('GPS Speed (kn)', fontsize=10)
    ax2.set_ylabel('Fuel Consumption (t/day)', fontsize=10)
    ax2.set_title('Speed vs Fuel Consumption — Score Intensity', fontsize=11, fontweight='bold')

    fig.suptitle('Power–Speed–Fuel Anomaly Diagnostics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'power_speed_anomalies.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


# ── CLI entry-point ────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Stage 04 – Anomaly Detection (SVD + Isolation Forest + Autoencoder)'
    )
    parser.add_argument('--cleaning-dir',    default='results/02_cleaning')
    parser.add_argument('--data',            default='data1.csv')
    parser.add_argument('--output-dir',      default='results/04_anomaly_detection')
    parser.add_argument('--methods',         nargs='+',
                        default=['svd', 'iforest', 'autoencoder'],
                        choices=['svd', 'iforest', 'autoencoder'],
                        help='Methods to run (default: all three)')
    parser.add_argument('--n-clusters',      type=int,   default=4)
    parser.add_argument('--n-components',    type=int,   default=3,
                        help='SVD components per cluster')
    parser.add_argument('--threshold-pct',   type=float, default=95.0)
    parser.add_argument('--random-state',    type=int,   default=42)
    parser.add_argument('--model-version',   default='clustered_svd_v1')
    parser.add_argument('--if-n-estimators', type=int,   default=100,
                        help='Isolation Forest number of trees')
    parser.add_argument('--ae-epochs',       type=int,   default=50,
                        help='Autoencoder training epochs')
    parser.add_argument('--ae-encoding-dim', type=int,   default=3,
                        help='Autoencoder latent dimension')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    result = run({
        'cleaning_dir':    args.cleaning_dir,
        'data':            args.data,
        'output_dir':      args.output_dir,
        'methods':         args.methods,
        'n_clusters':      args.n_clusters,
        'n_components':    args.n_components,
        'threshold_pct':   args.threshold_pct,
        'random_state':    args.random_state,
        'model_version':   args.model_version,
        'if_n_estimators': args.if_n_estimators,
        'ae_epochs':       args.ae_epochs,
        'ae_encoding_dim': args.ae_encoding_dim,
    })

    print("\nSummary:")
    print(f"  Methods run       : {result['methods_run']}")
    print(f"  Ensemble anomalies: {result['n_anomalies']:,}  ({result['anomaly_rate']:.2f}%)")
    print(f"  Recall per method:")
    for m, r in result['recall_per_method'].items():
        print(f"    {m:<15}: {r:.1%}")
    print(f"  Ensemble recall   : {result['recall_ensemble']:.1%}")
    print(f"\n  Artefacts in:     {result['output_dir']}/")
