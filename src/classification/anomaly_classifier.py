"""
Anomaly classification — assigns human-readable type labels to detected anomalies
and produces breakdown plots.

Classification hierarchy
────────────────────────
1. Constraint-based (rule-driven, deterministic):
   • negative_power   – Main_Engine_Power_kW < 0
   • extreme_trim     – |Trim_m| > 5 m
   • invalid_draft    – Avg_draft_m <= 0

2. Cluster-context (statistical, data-driven):
   • high_speed_low_power – speed > 8 kn but power < 200 kW
   • low_speed_high_power – speed < 1 kn but power > 500 kW
   • fuel_efficiency_outlier – anomalous power/fuel ratio for cluster

3. Catch-all:
   • multivariate_outlier – SVD reconstruction error only, no rule matches
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


class AnomalyClassifier:
    """
    Classify anomalies into human-readable types.

    Expects a DataFrame that already contains:
      - All original feature columns
      - ``is_anomaly``       (bool)
      - ``anomaly_score``    (float)
      - ``cluster_name``     (str)
      - Constraint flags: ``flag_negative_power``, ``flag_extreme_trim``,
                           ``flag_invalid_draft``, ``flag_any_constraint``
    """

    # ── Rule thresholds ────────────────────────────────────────────────────────
    HIGH_SPEED_THRESHOLD = 8.0       # kn
    LOW_POWER_THRESHOLD  = 200.0     # kW (when moving fast)
    IDLE_SPEED_THRESHOLD = 1.0       # kn (considered at rest)
    HIGH_POWER_IDLE      = 500.0     # kW (unexpected while at rest)

    def __init__(self, output_dir: str = 'results/05_classification'):
        self.output_dir = Path(output_dir)

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add an ``anomaly_type`` column and a ``primary_type`` column.

        Args:
            df: DataFrame with anomaly flags (full dataset; non-anomaly rows
                receive ``anomaly_type = 'normal'``).

        Returns:
            DataFrame with new columns: ``anomaly_type``, ``primary_type``.
        """
        df = df.copy()
        df['anomaly_type'] = 'normal'
        df['primary_type'] = 'normal'

        # Only classify rows flagged as anomalies
        mask = df['is_anomaly'].astype(bool)

        types_list = []
        primary_list = []

        for _, row in df[mask].iterrows():
            types, primary = self._classify_row(row)
            types_list.append(','.join(types))
            primary_list.append(primary)

        df.loc[mask, 'anomaly_type'] = types_list
        df.loc[mask, 'primary_type'] = primary_list

        return df

    # ── Private helpers ────────────────────────────────────────────────────────

    def _classify_row(self, row: pd.Series):
        """Return (list_of_type_strings, primary_type_string) for one anomaly."""
        types = []

        # 1. Constraint-based (highest priority)
        power = self._safe(row, 'Main_Engine_Power_kW')
        trim  = self._safe(row, 'Trim_m')
        draft = self._safe(row, 'Avg_draft_m')
        speed = self._safe(row, 'GPSSpeed_kn')

        if power is not None and power < 0:
            types.append('negative_power')
        if trim is not None and abs(trim) > 5:
            types.append('extreme_trim')
        if draft is not None and draft <= 0:
            types.append('invalid_draft')

        # 2. Cluster-context rules
        if (speed is not None and power is not None
                and speed > self.HIGH_SPEED_THRESHOLD
                and power < self.LOW_POWER_THRESHOLD):
            types.append('high_speed_low_power')

        if (speed is not None and power is not None
                and speed < self.IDLE_SPEED_THRESHOLD
                and power > self.HIGH_POWER_IDLE):
            types.append('low_speed_high_power')

        fuel = self._safe(row, 'Fuel_Consumption_t_per_day')
        if (power is not None and fuel is not None
                and power > 0 and fuel > 0):
            ratio = power / fuel
            if ratio > 200 or ratio < 5:   # rough domain bounds (kW / (t/day))
                types.append('fuel_efficiency_outlier')

        # 3. Catch-all
        if not types:
            types.append('multivariate_outlier')

        # Primary = first type in the list (priority order from above)
        primary = types[0]
        return types, primary

    @staticmethod
    def _safe(row: pd.Series, col: str) -> Optional[float]:
        val = row.get(col)
        if val is None:
            return None
        try:
            f = float(val)
            return None if np.isnan(f) else f
        except (TypeError, ValueError):
            return None

    def save_results(self, df: pd.DataFrame) -> dict:
        """
        Save classified anomalies CSV, type breakdown CSV, and plots.

        Args:
            df: Output of ``classify()``.

        Returns:
            dict with output paths and counts.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        anomalies = df[df['is_anomaly'].astype(bool)].copy()

        # ── CSV artefacts ──────────────────────────────────────────────────────
        anomalies_path = self.output_dir / 'classified_anomalies.csv'
        anomalies.to_csv(anomalies_path, index=False)

        # Type breakdown
        breakdown = (
            anomalies.groupby('primary_type')
            .agg(count=('primary_type', 'size'),
                 avg_score=('anomaly_score', 'mean'),
                 max_score=('anomaly_score', 'max'))
            .reset_index()
            .sort_values('count', ascending=False)
        )
        breakdown['pct'] = (breakdown['count'] / len(anomalies) * 100).round(2)
        breakdown_path = self.output_dir / 'anomaly_type_breakdown.csv'
        breakdown.to_csv(breakdown_path, index=False)

        # ── Plots ──────────────────────────────────────────────────────────────
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        pie_path = self._plot_type_pie(breakdown, plots_dir)
        bar_path = self._plot_score_by_type(anomalies, plots_dir)
        ts_path  = self._plot_anomalies_over_time(df, plots_dir)

        print(f"  [Classification] {len(anomalies):,} anomalies classified")
        for _, row in breakdown.iterrows():
            print(f"        {row['primary_type']}: {row['count']} ({row['pct']:.1f}%)")

        return {
            'n_anomalies': len(anomalies),
            'type_breakdown': breakdown.to_dict('records'),
            'anomalies_csv': str(anomalies_path),
            'breakdown_csv': str(breakdown_path),
            'pie_plot': pie_path,
            'bar_plot': bar_path,
            'ts_plot':  ts_path,
        }

    def _plot_type_pie(self, breakdown: pd.DataFrame, out_dir: Path) -> str:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(
            breakdown['count'],
            labels=breakdown['primary_type'],
            autopct='%1.1f%%',
            startangle=140,
        )
        ax.set_title('Anomaly Type Distribution', fontsize=13, fontweight='bold')
        plt.tight_layout()
        path = out_dir / 'anomaly_type_pie.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(path)

    def _plot_score_by_type(self, anomalies: pd.DataFrame, out_dir: Path) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))
        order = (anomalies.groupby('primary_type')['anomaly_score']
                 .median().sort_values(ascending=False).index)

        data = [anomalies.loc[anomalies['primary_type'] == t, 'anomaly_score'].values
                for t in order]
        ax.boxplot(data, labels=order, vert=True, patch_artist=True)
        ax.set_xlabel('Anomaly Type', fontsize=11)
        ax.set_ylabel('Anomaly Score (Reconstruction Error)', fontsize=11)
        ax.set_title('Anomaly Score Distribution by Type', fontsize=13, fontweight='bold')
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        path = out_dir / 'score_by_type_boxplot.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(path)

    def _plot_anomalies_over_time(self, df: pd.DataFrame, out_dir: Path) -> str:
        if 'Date' not in df.columns:
            return ''
        anomalies = df[df['is_anomaly'].astype(bool)].copy()
        anomalies['date'] = pd.to_datetime(anomalies['Date'], format='ISO8601',
                                           utc=True).dt.date

        daily = (anomalies.groupby(['date', 'primary_type'])
                 .size().unstack(fill_value=0))

        fig, ax = plt.subplots(figsize=(14, 5))
        daily.plot(kind='bar', stacked=True, ax=ax, width=0.8)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Anomaly Count', fontsize=11)
        ax.set_title('Daily Anomaly Counts by Type', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.tight_layout()
        path = out_dir / 'anomalies_over_time.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(path)
