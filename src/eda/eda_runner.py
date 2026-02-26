"""
EDA runner for maritime vessel data.

Produces summary statistics, distribution histograms, a correlation heatmap,
and a constraint-violation report — all written to an output directory.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional


# ── Feature groups ────────────────────────────────────────────────────────────
NUMERIC_COLS = [
    'GPSSpeed_kn', 'GPS_LAT', 'GPS_LON',
    'Main_Engine_Power_kW', 'Speed_rpm',
    'Fuel_Consumption_t_per_day',
    'DRAFTAFT', 'DRAFTFWD', 'Avg_draft_m', 'Trim_m',
    'TrueWindSpeed_kn', 'RelWindSpeed_kn',
]

FEATURE_COLS = [
    'GPSSpeed_kn', 'Main_Engine_Power_kW', 'Speed_rpm',
    'Fuel_Consumption_t_per_day', 'Avg_draft_m', 'Trim_m',
    'TrueWindSpeed_kn', 'RelWindSpeed_kn',
]

CONSTRAINT_CHECKS = {
    'Negative engine power (< 0 kW)': lambda df: df['Main_Engine_Power_kW'] < 0,
    'Extreme trim (|Trim_m| > 5 m)':  lambda df: df['Trim_m'].abs() > 5,
    'Invalid DRAFTFWD sentinel (-9999)': lambda df: df['DRAFTFWD'] == -9999,
    'Invalid Avg_draft_m (<= 0 m)':   lambda df: df['Avg_draft_m'] <= 0,
}


class EDARunner:
    """Run exploratory data analysis on a vessel data CSV."""

    def __init__(self, output_dir: str = 'results/01_eda'):
        self.output_dir = Path(output_dir)

    def run(self, data_path: str = 'data1.csv') -> dict:
        """
        Execute full EDA pipeline and save all artefacts.

        Args:
            data_path: Path to input CSV file.

        Returns:
            dict with summary statistics and constraint counts.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        print(f"  [EDA] Loading data from {data_path}")
        df = pd.read_csv(data_path, parse_dates=['Date'])
        print(f"  [EDA] {len(df):,} records, {df.shape[1]} columns")

        # ── 1. Summary statistics ─────────────────────────────────────────────
        stats = self._summary_stats(df)
        stats.to_csv(self.output_dir / 'summary_stats.csv')
        print(f"  [EDA] Summary stats → {self.output_dir / 'summary_stats.csv'}")

        # ── 2. Missing values report ──────────────────────────────────────────
        missing = self._missing_report(df)
        missing.to_csv(self.output_dir / 'missing_values.csv')
        print(f"  [EDA] Missing values → {self.output_dir / 'missing_values.csv'}")

        # ── 3. Constraint violations ──────────────────────────────────────────
        violations = self._constraint_violations(df)
        violations.to_csv(self.output_dir / 'constraint_violations.csv', index=False)
        print(f"  [EDA] Constraint violations → {self.output_dir / 'constraint_violations.csv'}")
        for _, row in violations.iterrows():
            print(f"        {row['check']}: {row['count']} records ({row['pct']:.2f}%)")

        # ── 4. Correlation heatmap ────────────────────────────────────────────
        corr_path = self._plot_correlation(df, plots_dir)
        print(f"  [EDA] Correlation heatmap → {corr_path}")

        # ── 5. Distribution histograms ────────────────────────────────────────
        hist_path = self._plot_distributions(df, plots_dir)
        print(f"  [EDA] Distribution histograms → {hist_path}")

        # ── 6. Key scatter: power vs speed ───────────────────────────────────
        scatter_path = self._plot_power_vs_speed(df, plots_dir)
        print(f"  [EDA] Power-vs-speed scatter → {scatter_path}")

        # ── 7. Time-series overview ───────────────────────────────────────────
        ts_path = self._plot_timeseries(df, plots_dir)
        print(f"  [EDA] Time-series overview → {ts_path}")

        # ── 8. Temporal coverage ──────────────────────────────────────────────
        coverage = self._temporal_coverage(df)
        coverage.to_csv(self.output_dir / 'temporal_coverage.csv', index=False)

        result = {
            'n_records': len(df),
            'n_features': df.shape[1],
            'date_range': (str(df['Date'].min()), str(df['Date'].max())),
            'constraint_violations': violations.set_index('check')['count'].to_dict(),
            'output_dir': str(self.output_dir),
        }

        print(f"  [EDA] Complete. Outputs in {self.output_dir}/")
        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Descriptive statistics for numeric columns."""
        cols = [c for c in NUMERIC_COLS if c in df.columns]
        return df[cols].describe().T.round(4)

    def _missing_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count and percentage of missing values per column."""
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / len(df) * 100).round(3)
        return pd.DataFrame({
            'missing_count': missing_count,
            'missing_pct': missing_pct,
        }).sort_values('missing_count', ascending=False)

    def _constraint_violations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate domain constraint checks."""
        rows = []
        for label, check_fn in CONSTRAINT_CHECKS.items():
            try:
                mask = check_fn(df)
                count = int(mask.sum())
                rows.append({
                    'check': label,
                    'count': count,
                    'pct': round(count / len(df) * 100, 4),
                })
            except Exception as e:
                rows.append({'check': label, 'count': -1, 'pct': -1})
        return pd.DataFrame(rows)

    def _plot_correlation(self, df: pd.DataFrame, out_dir: Path) -> str:
        """Save correlation heatmap."""
        cols = [c for c in FEATURE_COLS if c in df.columns]
        corr = df[cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            vmin=-1, vmax=1, center=0, linewidths=0.5,
            annot_kws={'size': 9}, ax=ax
        )
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = out_dir / 'correlation_heatmap.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(path)

    def _plot_distributions(self, df: pd.DataFrame, out_dir: Path) -> str:
        """Save a grid of histograms for all feature columns."""
        cols = [c for c in FEATURE_COLS if c in df.columns]
        n = len(cols)
        ncols = 4
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            data = df[col].dropna()
            axes[i].hist(data, bins=60, color='steelblue', alpha=0.8, edgecolor='none')
            axes[i].set_title(col, fontsize=10, fontweight='bold')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Count')
            axes[i].tick_params(labelsize=8)

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle('Feature Distributions', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = out_dir / 'feature_distributions.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(path)

    def _plot_power_vs_speed(self, df: pd.DataFrame, out_dir: Path) -> str:
        """Save power vs speed scatter highlighting constraint violations."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Normal points
        normal = df[df['Main_Engine_Power_kW'] >= 0]
        ax.scatter(
            normal['GPSSpeed_kn'], normal['Main_Engine_Power_kW'],
            alpha=0.3, s=5, c='steelblue', label='Normal'
        )

        # Negative power (anomalies)
        bad = df[df['Main_Engine_Power_kW'] < 0]
        if len(bad) > 0:
            ax.scatter(
                bad['GPSSpeed_kn'], bad['Main_Engine_Power_kW'],
                alpha=0.9, s=30, c='red', zorder=5, label=f'Negative power (n={len(bad)})'
            )

        ax.set_xlabel('GPS Speed (kn)', fontsize=11)
        ax.set_ylabel('Main Engine Power (kW)', fontsize=11)
        ax.set_title('Engine Power vs GPS Speed', fontsize=13, fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--', lw=1)
        ax.legend(fontsize=9)
        plt.tight_layout()

        path = out_dir / 'power_vs_speed.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(path)

    def _plot_timeseries(self, df: pd.DataFrame, out_dir: Path) -> str:
        """Save time-series overview for key features."""
        plot_cols = [c for c in ['GPSSpeed_kn', 'Main_Engine_Power_kW',
                                  'Fuel_Consumption_t_per_day', 'Avg_draft_m']
                     if c in df.columns]

        df_ts = df[['Date'] + plot_cols].dropna(subset=['Date']).sort_values('Date')

        fig, axes = plt.subplots(len(plot_cols), 1, figsize=(16, 3 * len(plot_cols)),
                                  sharex=True)
        if len(plot_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, plot_cols):
            ax.plot(df_ts['Date'], df_ts[col], lw=0.6, color='steelblue', alpha=0.8)
            ax.set_ylabel(col, fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[0].set_title('Time-Series Overview', fontsize=13, fontweight='bold')
        axes[-1].set_xlabel('Date', fontsize=10)
        plt.tight_layout()

        path = out_dir / 'timeseries_overview.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(path)

    def _temporal_coverage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute daily record counts and sampling gaps."""
        df_ts = df[['Date']].dropna().copy()
        df_ts['Date'] = pd.to_datetime(df_ts['Date'], format='ISO8601', utc=True)
        df_ts = df_ts.sort_values('Date')
        df_ts['date'] = df_ts['Date'].dt.date
        daily_counts = df_ts.groupby('date').size().reset_index(name='records')

        gaps = df_ts['Date'].diff().dt.total_seconds().dropna()
        daily_counts['median_gap_sec'] = gaps.median()
        daily_counts['max_gap_sec'] = gaps.max()

        return daily_counts
