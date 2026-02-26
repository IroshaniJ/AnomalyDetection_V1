"""
Stage 03 — Clustering (Operational Mode Identification)
=======================================================
Fits a K-Means clusterer on the cleaned feature matrix to identify
vessel operational modes (at-rest loaded/unloaded, medium speed, high speed).

Reads from ``results/02_cleaning/`` (or raw CSV as fallback).

Writes:
  • ``results/03_clustering/cluster_labels.npy``    – integer label per valid row
  • ``results/03_clustering/cluster_names.json``    – cluster_id → name mapping
  • ``results/03_clustering/cluster_summary.csv``   – counts, centres, percentages
  • ``results/03_clustering/clusterer.pkl``          – fitted OperationalModeClusterer
  • ``results/03_clustering/plots/``                 – visualisation plots

Standalone usage:
    python pipeline/stage_03_clustering.py
    python pipeline/stage_03_clustering.py --n-clusters 4 --output results/03_clustering

As a pipeline step:
    from pipeline.stage_03_clustering import run
    result = run(config)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Ensure project root is on sys.path ────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.clustering import OperationalModeClusterer
from src.preprocessing.data_loader import get_feature_columns


# ── Public API ─────────────────────────────────────────────────────────────────

def run(config: Optional[dict] = None) -> dict:
    """
    Execute the clustering stage.

    Args:
        config: Optional dict with keys:
            ``cleaning_dir`` – where stage_02 wrote its outputs
                               (default: 'results/02_cleaning')
            ``data``         – fallback CSV path if cleaning_dir not found
            ``output_dir``   – where to write outputs (default: 'results/03_clustering')
            ``n_clusters``   – number of operational mode clusters (default: 4)
            ``random_state`` – random seed (default: 42)

    Returns:
        dict with keys: n_clusters, cluster_names, cluster_summary_csv,
        clusterer_pkl, labels_npy, output_dir.
    """
    config = config or {}
    cleaning_dir = Path(config.get('cleaning_dir', 'results/02_cleaning'))
    data_path    = config.get('data', 'data1.csv')
    output_dir   = Path(config.get('output_dir', 'results/03_clustering'))
    n_clusters   = int(config.get('n_clusters', 4))
    random_state = int(config.get('random_state', 42))

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    print("\n" + "─" * 60)
    print("STAGE 03 — CLUSTERING")
    print("─" * 60)

    t0 = time.time()

    # ── Load feature matrix ────────────────────────────────────────────────────
    X, feature_cols = _load_features(cleaning_dir, data_path)
    print(f"  [Clustering] Feature matrix: {X.shape[0]:,} × {X.shape[1]}")

    # ── Fit clusterer ──────────────────────────────────────────────────────────
    print(f"  [Clustering] Fitting K-Means (n_clusters={n_clusters}) …")
    clusterer = OperationalModeClusterer(
        n_clusters=n_clusters,
        random_state=random_state,
    )
    labels = clusterer.fit_predict(X, feature_cols)

    # ── Print cluster summary ──────────────────────────────────────────────────
    summary = clusterer.get_cluster_summary(feature_cols)
    print(f"  [Clustering] Cluster assignments:")
    for _, row in summary.iterrows():
        print(f"        Cluster {row['cluster_id']:d} | {row['name']:<35} | "
              f"{row['count']:>6,} samples  ({row['percentage']:.1f}%)")

    # ── Save artefacts ─────────────────────────────────────────────────────────
    labels_npy    = output_dir / 'cluster_labels.npy'
    clusterer_pkl = output_dir / 'clusterer.pkl'
    summary_csv   = output_dir / 'cluster_summary.csv'
    names_json    = output_dir / 'cluster_names.json'

    np.save(labels_npy, labels)
    clusterer.save(clusterer_pkl)
    summary.to_csv(summary_csv, index=False)

    with open(names_json, 'w') as f:
        json.dump(clusterer.cluster_names, f, indent=2)

    # ── Visualisations ────────────────────────────────────────────────────────
    pie_path   = _plot_cluster_pie(summary, plots_dir)
    scat_path  = _plot_cluster_scatter(X, labels, clusterer.cluster_names,
                                       feature_cols, plots_dir)

    elapsed = time.time() - t0
    print(f"  [Clustering] Done in {elapsed:.1f}s  →  {output_dir}/")

    return {
        'n_clusters':        n_clusters,
        'cluster_names':     clusterer.cluster_names,
        'labels_npy':        str(labels_npy),
        'clusterer_pkl':     str(clusterer_pkl),
        'cluster_summary_csv': str(summary_csv),
        'pie_plot':          pie_path,
        'scatter_plot':      scat_path,
        'output_dir':        str(output_dir),
        'elapsed_sec':       round(elapsed, 2),
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_features(cleaning_dir: Path, data_path: str):
    """Load feature matrix from stage_02 output or fall back to raw CSV."""
    npy_path = cleaning_dir / 'feature_matrix.npy'
    feature_cols = get_feature_columns()

    if npy_path.exists():
        print(f"  [Clustering] Loading feature matrix from {npy_path}")
        X = np.load(npy_path)
        return X, feature_cols

    # Fallback: re-run cleaning inline
    print(f"  [Clustering] Stage-02 output not found — loading from {data_path}")
    from src.preprocessing.data_loader import load_and_clean
    df = load_and_clean(data_path)
    valid_mask = ~df[feature_cols].isna().any(axis=1)
    X = df.loc[valid_mask, feature_cols].values.astype(np.float64)
    return X, feature_cols


def _plot_cluster_pie(summary: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(
        summary['count'],
        labels=summary['name'],
        autopct='%1.1f%%',
        startangle=140,
    )
    ax.set_title('Operational Mode Distribution', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'cluster_distribution_pie.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


def _plot_cluster_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    names: dict,
    feature_cols: list,
    out_dir: Path,
) -> str:
    """Power vs Speed scatter coloured by cluster."""
    speed_idx = next((i for i, c in enumerate(feature_cols) if 'gps' in c.lower() and 'speed' in c.lower()), 0)
    power_idx = next((i for i, c in enumerate(feature_cols) if 'power' in c.lower()), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = plt.cm.tab10.colors

    for cid, cname in names.items():
        mask = labels == cid
        ax.scatter(X[mask, speed_idx], X[mask, power_idx],
                   alpha=0.3, s=5, color=colours[cid % 10], label=cname)

    ax.set_xlabel(feature_cols[speed_idx], fontsize=11)
    ax.set_ylabel(feature_cols[power_idx], fontsize=11)
    ax.set_title('Operational Mode Clusters — Power vs Speed', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, markerscale=4)
    plt.tight_layout()
    path = out_dir / 'cluster_scatter_power_speed.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


# ── CLI entry-point ────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Stage 03 – Operational Mode Clustering'
    )
    parser.add_argument('--cleaning-dir', default='results/02_cleaning',
                        help='Stage-02 output directory (default: results/02_cleaning)')
    parser.add_argument('--data',         default='data1.csv',
                        help='Fallback CSV path (default: data1.csv)')
    parser.add_argument('--output-dir',   default='results/03_clustering',
                        help='Output directory (default: results/03_clustering)')
    parser.add_argument('--n-clusters',   type=int, default=4,
                        help='Number of clusters (default: 4)')
    parser.add_argument('--random-state', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    result = run({
        'cleaning_dir': args.cleaning_dir,
        'data':         args.data,
        'output_dir':   args.output_dir,
        'n_clusters':   args.n_clusters,
        'random_state': args.random_state,
    })

    print("\nSummary:")
    print(f"  Clusters found  : {result['n_clusters']}")
    for cid, cname in result['cluster_names'].items():
        print(f"    [{cid}] {cname}")
    print(f"\n  Artefacts in:   {result['output_dir']}/")
