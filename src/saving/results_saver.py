"""
Results saver — persists anomalies to SQLite, writes final summary CSVs, and
produces a human-readable HTML/text report.

Wraps ``AnomalyDatabase`` and adds pipeline-level orchestration.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Allow imports from project root when run as a module
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.database.anomaly_db import AnomalyDatabase


class ResultsSaver:
    """
    Persist pipeline outputs to disk and SQLite.

    Handles:
      - Logging each detection run to ``detection_runs`` table
      - Logging individual anomaly records to ``anomalies`` table
      - Writing final CSV and HTML summary reports
      - Producing cluster-anomaly distribution bar chart
    """

    def __init__(
        self,
        output_dir: str = 'results/06_saving',
        db_path: str = 'anomalies.db',
        model_version: str = 'clustered_svd_v1',
    ):
        self.output_dir = Path(output_dir)
        self.db_path = db_path
        self.model_version = model_version

    def save(
        self,
        df: pd.DataFrame,
        threshold: float,
        n_components: int = 3,
        threshold_percentile: float = 95.0,
        data_file: str = 'data1.csv',
    ) -> dict:
        """
        Persist full pipeline results.

        Args:
            df: Full DataFrame with ``is_anomaly``, ``anomaly_score``,
                ``cluster_name``, ``primary_type``, and original features.
            threshold: Average anomaly threshold used during detection.
            n_components: SVD components (for run metadata).
            threshold_percentile: Threshold percentile (for run metadata).
            data_file: Source data filename.

        Returns:
            dict with saved file paths and DB row counts.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        db = AnomalyDatabase(self.db_path)
        anomalies = df[df['is_anomaly'].astype(bool)].copy()

        # ── 1. Log detection run ───────────────────────────────────────────────
        run_id = db.log_detection_run(
            model_version=self.model_version,
            data_file=data_file,
            total_records=len(df),
            anomalies_detected=len(anomalies),
            threshold=threshold,
            n_components=n_components,
            threshold_percentile=threshold_percentile,
        )
        print(f"  [Saving] Detection run logged (run_id={run_id})")

        # ── 2. Log anomaly records ─────────────────────────────────────────────
        scores = anomalies['anomaly_score'].fillna(threshold).values.tolist()
        n_logged = db.log_anomalies(
            df_anomalies=anomalies,
            scores=scores,
            threshold=threshold,
            model_version=self.model_version,
        )
        print(f"  [Saving] {n_logged:,} anomaly records → {self.db_path}")

        # ── 3. Write CSV exports ───────────────────────────────────────────────
        anomalies_csv = self.output_dir / 'all_anomalies.csv'
        anomalies.to_csv(anomalies_csv, index=False)

        full_csv = self.output_dir / 'full_dataset_with_labels.csv'
        df.to_csv(full_csv, index=False)
        print(f"  [Saving] CSVs written to {self.output_dir}/")

        # ── 4. Summary report ──────────────────────────────────────────────────
        summary = db.get_anomaly_summary()
        runs = db.get_detection_runs()

        report_path = self._write_report(df, anomalies, summary, runs)
        print(f"  [Saving] Report → {report_path}")

        # ── 5. Cluster-anomaly plot ────────────────────────────────────────────
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        cluster_plot = self._plot_cluster_anomaly_distribution(df, plots_dir)

        return {
            'run_id': run_id,
            'n_anomalies_logged': n_logged,
            'anomalies_csv': str(anomalies_csv),
            'full_dataset_csv': str(full_csv),
            'report': report_path,
            'cluster_plot': cluster_plot,
            'db_path': self.db_path,
        }

    def _write_report(
        self,
        df: pd.DataFrame,
        anomalies: pd.DataFrame,
        summary: pd.DataFrame,
        runs: pd.DataFrame,
    ) -> str:
        """Write a plain-text summary report."""
        lines = [
            '=' * 70,
            'TWINSHIP ANOMALY DETECTION — PIPELINE REPORT',
            '=' * 70,
            '',
            f'Total records:        {len(df):,}',
            f'Total anomalies:      {len(anomalies):,}  ({len(anomalies)/len(df)*100:.2f}%)',
        ]

        if 'Date' in df.columns:
            lines += [
                f'Date range:           {df["Date"].min()}  →  {df["Date"].max()}',
            ]

        lines += ['', '── Anomaly Type Breakdown ──────────────────────────', '']
        if not summary.empty:
            lines.append(summary.to_string(index=False))
        else:
            lines.append('(no data)')

        if 'cluster_name' in anomalies.columns:
            lines += ['', '── Anomalies per Cluster ───────────────────────────', '']
            cluster_counts = (
                anomalies.groupby('cluster_name')
                .agg(n=('cluster_name', 'size'),
                     avg_score=('anomaly_score', 'mean'))
                .reset_index()
            )
            lines.append(cluster_counts.to_string(index=False))

        lines += ['', '── Detection Run History ───────────────────────────', '']
        if not runs.empty:
            cols = ['run_timestamp', 'model_version', 'total_records', 'anomalies_detected']
            cols = [c for c in cols if c in runs.columns]
            lines.append(runs[cols].tail(5).to_string(index=False))

        lines += ['', '=' * 70]

        report_path = self.output_dir / 'pipeline_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))

        return str(report_path)

    def _plot_cluster_anomaly_distribution(self, df: pd.DataFrame, out_dir: Path) -> str:
        if 'cluster_name' not in df.columns:
            return ''

        cluster_stats = (
            df.groupby('cluster_name')
            .agg(total=('is_anomaly', 'size'),
                 anomalies=('is_anomaly', 'sum'))
            .reset_index()
        )
        cluster_stats['normal'] = cluster_stats['total'] - cluster_stats['anomalies']
        cluster_stats['anomaly_rate'] = (
            cluster_stats['anomalies'] / cluster_stats['total'] * 100
        ).round(2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Stacked bar: normal vs anomaly counts
        x = range(len(cluster_stats))
        ax1.bar(x, cluster_stats['normal'], label='Normal', color='steelblue', alpha=0.8)
        ax1.bar(x, cluster_stats['anomalies'], bottom=cluster_stats['normal'],
                label='Anomaly', color='tomato', alpha=0.9)
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(cluster_stats['cluster_name'], rotation=20, ha='right', fontsize=9)
        ax1.set_ylabel('Records', fontsize=11)
        ax1.set_title('Normal vs Anomaly Count per Cluster', fontsize=12, fontweight='bold')
        ax1.legend()

        # Anomaly rate bar
        ax2.bar(x, cluster_stats['anomaly_rate'], color='tomato', alpha=0.8)
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(cluster_stats['cluster_name'], rotation=20, ha='right', fontsize=9)
        ax2.set_ylabel('Anomaly Rate (%)', fontsize=11)
        ax2.set_title('Anomaly Rate per Cluster', fontsize=12, fontweight='bold')
        for i, rate in enumerate(cluster_stats['anomaly_rate']):
            ax2.text(i, rate + 0.1, f'{rate:.1f}%', ha='center', fontsize=9)

        plt.suptitle('Cluster-Level Anomaly Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = out_dir / 'cluster_anomaly_distribution.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(path)
