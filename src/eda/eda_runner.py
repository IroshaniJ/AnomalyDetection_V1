"""
EDA runner for maritime vessel data.

Produces summary statistics, distribution histograms, a correlation heatmap,
and a constraint-violation report — all written to an output directory.
"""
import re
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

# ── Default feature groups (used when vessel_config.feature_groups is empty) ──
# Keys map to the five analysis categories used throughout EDA plots/reports.
DEFAULT_FEATURE_GROUPS: dict = {
    'engine_propulsion': [
        'Main_Engine_Power_kW', 'Speed_rpm', 'ME_Shaft_Torque_kNm',
        'ME_Shaft_Thrust_kN', 'Propeller_Pitch_pct',
        'Fuel_Consumption_rate', 'Fuel_Consumption_t_per_day',
        'DG1_Power_kW', 'DG2_Power_kW', 'DG3_Power_kW', 'DG4_Power_kW',
        'GE1_Power_kW', 'GE2_Power_kW', 'GE3_Power_kW', 'GE4_Power_kW',
    ],
    'navigation': [
        'GPSSpeed_kn', 'SpeedLog_kn', 'GPS_LAT', 'GPS_LON',
        'Ship_Course_deg', 'Heading_deg',
    ],
    'draft': [
        'DRAFTAFT', 'DRAFTFWD', 'Avg_draft_m', 'Trim_m',
        'Draft_Mid_Port_m', 'Draft_Mid_Stbd_m', 'Depth_of_Water_m',
        'Displacement_m',
    ],
    'weather': [
        'RelWindSpeed_kn', 'RelWindAngle_deg', 'TrueWindSpeed_kn',
        'Wave_Height_m', 'Wave_Period_s', 'Wave_Direction_deg',
        'Swell_Height_m', 'Swell_Period_s', 'Swell_Direction_deg',
        'Sea_Force', 'Sea_Direction_deg',
        'Current_Speed_kn', 'Current_Direction_deg',
        'Air_Temp_C', 'Pressure_mb', 'Humidity_pct', 'Visibility_m',
    ],
    'trip': [
        'Nautical_Miles',
    ],
}

# Human-readable labels for group keys
_GROUP_LABELS: dict = {
    'engine_propulsion': 'Engine & Propulsion',
    'navigation':        'Navigation',
    'draft':             'Draft',
    'weather':           'Weather',
    'trip':              'Trip',
    'other':             'Other',
}


def _group_for_col(col: str, vessel_cfg=None) -> str:
    """Return the feature-group name for a given standard column name.

    Lookup order:
    1. Vessel-specific ``feature_groups`` (from VesselConfig)
    2. ``DEFAULT_FEATURE_GROUPS``
    3. ``'other'`` fallback
    """
    if vessel_cfg is not None:
        for group, cols in getattr(vessel_cfg, 'feature_groups', {}).items():
            if col in cols:
                return group
    for group, cols in DEFAULT_FEATURE_GROUPS.items():
        if col in cols:
            return group
    return 'other'

CONSTRAINT_CHECKS = {
    'Negative engine power (< 0 kW)': lambda df: df['Main_Engine_Power_kW'] < 0,
    'Extreme trim (|Trim_m| > 5 m)':  lambda df: df['Trim_m'].abs() > 5,
    'Invalid DRAFTFWD sentinel (-9999)': lambda df: df['DRAFTFWD'] == -9999,
    'Invalid Avg_draft_m (<= 0 m)':   lambda df: df['Avg_draft_m'] <= 0,
}


# ── Per-signal physical expectations for univariate EDA ───────────────────────
UNIVARIATE_SIGNAL_PROFILES: dict = {
    'Main_Engine_Power_kW': {
        'label': 'Propulsion Power', 'unit': 'kW',
        'phys_lo': 0.0, 'phys_hi': 45_000.0,
        'negative_is_anomaly': True,
        'alias': ['PropulsionPowerTotal'],
        'expect': (
            'Expected: bimodal — low band at port/idle, high band in transit. '
            'Look for: negative values, impossible maxima, flat-lines, sudden jumps.'
        ),
    },
    'Fuel_Consumption_rate': {
        'label': 'Fuel Mass Flow', 'unit': 'kg/h',
        'phys_lo': 0.0, 'phys_hi': 12_000.0,
        'negative_is_anomaly': True,
        'alias': ['FuelMassFlowMETotal', 'Fuel_Consumption_t_per_day'],
        'expect': (
            'Expected: strongly related to power; separate bands by phase. '
            'Look for: negatives, zeros during sailing, noisy spikes, flat-line, mismatch with power.'
        ),
    },
    'GPSSpeed_kn': {
        'label': 'Speed over Ground', 'unit': 'kn',
        'phys_lo': 0.0, 'phys_hi': 25.0,
        'negative_is_anomaly': True,
        'alias': ['SOG'],
        'expect': (
            'Expected: repeated route profile; stable cruising band in transit; lower near port. '
            'Look for: zeros/near-zeros, sudden jumps, unstable oscillations.'
        ),
    },
    'DRAFTAFT': {
        'label': 'Draft Aft', 'unit': 'm',
        'phys_lo': 3.0, 'phys_hi': 10.0, 'negative_is_anomaly': True,
        'alias': ['DraftAftDynamic'],
        'expect': (
            'Expected: slow variation, clustered around normal loading condition. '
            'Look for: unrealistic values, sudden jumps, large fore-aft difference, drift over months.'
        ),
    },
    'DRAFTFWD': {
        'label': 'Draft Fwd', 'unit': 'm',
        'phys_lo': 3.0, 'phys_hi': 10.0, 'negative_is_anomaly': True,
        'alias': ['DraftFwdDynamic'],
        'expect': (
            'Expected: slow variation, clustered around normal loading condition. '
            'Look for: unrealistic values, sudden jumps, large fore-aft difference, drift over months.'
        ),
    },
    'RelWindSpeed_kn': {
        'label': 'Apparent Wind Speed', 'unit': 'kn',
        'phys_lo': 0.0, 'phys_hi': 60.0, 'negative_is_anomaly': True,
        'alias': ['AWS'],
        'expect': (
            'Expected: route-direction effects; different patterns on opposite legs. '
            'Look for: impossible values, repeated values, sensor wrap-around.'
        ),
    },
    'RelWindAngle_deg': {
        'label': 'Apparent Wind Angle', 'unit': '\u00b0',
        'phys_lo': -180.0, 'phys_hi': 360.0, 'negative_is_anomaly': False,
        'alias': ['AWA'],
        'expect': (
            'Expected: clusters around specific headings/legs; different on each leg. '
            'Look for: impossible angles, repeated values, wrap-around (0/360 jump).'
        ),
    },
    'Avg_draft_m': {
        'label': 'Average Draft', 'unit': 'm',
        'phys_lo': 3.0, 'phys_hi': 10.0, 'negative_is_anomaly': True,
        'alias': [],
        'expect': 'Mean of DraftAft and DraftFwd. Drift or sudden jumps reflect sensor behaviour.',
    },
    'Trim_m': {
        'label': 'Trim', 'unit': 'm',
        'phys_lo': -5.0, 'phys_hi': 5.0, 'negative_is_anomaly': False,
        'alias': [],
        'expect': 'Expected: \u00b11\u20132 m normal trim variation. Look for: values outside \u00b15 m (extreme trim).',
    },
    'TrueWindSpeed_kn': {
        'label': 'True Wind Speed', 'unit': 'kn',
        'phys_lo': 0.0, 'phys_hi': 60.0, 'negative_is_anomaly': True,
        'alias': [],
        'expect': 'Validate against RelWindSpeed. Negative or impossible values are faults.',
    },
    'Speed_rpm': {
        'label': 'Shaft Speed', 'unit': 'rpm',
        'phys_lo': 0.0, 'phys_hi': 160.0, 'negative_is_anomaly': True,
        'alias': ['ME Shaft Speed (rpm)'],
        'expect': (
            'Expected: strong correlation with propulsion power. '
            'Look for: negative values, sudden drops/spikes, constant-value flat-lines.'
        ),
    },
    'ME_Shaft_Torque_kNm': {
        'label': 'Shaft Torque', 'unit': 'kNm',
        'phys_lo': 0.0, 'phys_hi': 2_000.0, 'negative_is_anomaly': True,
        'alias': [],
        'expect': 'Expected: proportional to shaft power / speed. Spikes or negatives indicate sensor fault.',
    },
}


class EDARunner:
    """Run exploratory data analysis on a vessel data CSV."""

    def __init__(self, output_dir: str = 'results/01_eda'):
        self.output_dir = Path(output_dir)

    # ── Column discovery helpers ───────────────────────────────────────────────

    @staticmethod
    def _numeric_cols(df: pd.DataFrame, preferred: Optional[list] = None) -> list:
        """
        Return columns to analyse numerically.

        Priority:
        1. Intersection of *preferred* (e.g. NUMERIC_COLS) with df columns.
        2. If that is empty, fall back to all numeric columns in df
           except helper columns added by the loader.
        """
        preferred = preferred or NUMERIC_COLS
        known = [c for c in preferred if c in df.columns]
        if known:
            return known
        exclude = {'source_file'}
        return [c for c in df.select_dtypes(include='number').columns
                if c not in exclude]

    @staticmethod
    def _date_col(df: pd.DataFrame) -> Optional[str]:
        """Return the name of the first datetime-like column, or None."""
        # Explicit datetime columns
        for col in df.select_dtypes(include=['datetime', 'datetimetz']).columns:
            return col
        # Columns whose name looks like a timestamp
        for col in df.columns:
            if col.lower() in ('date', 'timestamp', 'time', 'datetime'):
                return col
        return None

    def run(self, data_path: str = 'data1.csv') -> dict:
        """
        Execute full EDA pipeline and save all artefacts.

        Args:
            data_path: Path to input CSV file.

        Returns:
            dict with summary statistics and constraint counts.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Detect vessel and create a vessel-specific sub-directory so that
        # running EDA on Stenaline and Stenateknik data never overwrites each
        # other's outputs.
        from src.vessel_config import detect_vessel
        vessel_cfg = detect_vessel(data_path)
        self._vessel_cfg = vessel_cfg          # used by plot helpers below
        vessel_dir = self.output_dir / vessel_cfg.vessel_id
        vessel_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = vessel_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        audit_dir = vessel_dir / 'audit'
        audit_dir.mkdir(exist_ok=True)

        print(f"  [EDA] Loading data from {data_path}")
        print(f"  [EDA] Vessel            : {vessel_cfg.vessel_name}")
        print(f"  [EDA] Output directory  : {vessel_dir}/")
        from src.preprocessing.data_loader import load_raw
        df = load_raw(data_path)
        print(f"  [EDA] {len(df):,} records, {df.shape[1]} columns")

        # ══ PHASE 0 — DATA AUDIT (must run before all other analysis) ════════
        print("\n  [EDA] ── Phase 0: Data Audit ──────────────────────────────")

        # ── 0A. Time-axis checks ──────────────────────────────────────────────
        time_audit = self._time_axis_audit(df, audit_dir)
        print(f"  [EDA] Time-axis audit → {audit_dir}/time_axis_*")
        _summarise_time_audit(time_audit)

        # ── 0B. Missingness audit ─────────────────────────────────────────────
        miss_audit = self._missingness_audit(df, audit_dir)
        print(f"  [EDA] Missingness audit → {audit_dir}/missingness_*")

        # ── 0C. Unit sanity check ─────────────────────────────────────────────
        unit_audit = self._unit_sanity_check(df, audit_dir)
        print(f"  [EDA] Unit sanity check → {audit_dir}/unit_sanity.csv")
        _summarise_unit_audit(unit_audit)

        print("  [EDA] ── Phase 0 complete ──────────────────────────────────\n")

        # ══ PHASE 1 — STANDARD EDA ═══════════════════════════════════════════
        # ── 1. Summary statistics ─────────────────────────────────────────────
        stats = self._summary_stats(df)
        stats.to_csv(vessel_dir / 'summary_stats.csv')
        print(f"  [EDA] Summary stats → {vessel_dir / 'summary_stats.csv'}")

        # ── 2. Missing values report ──────────────────────────────────────────
        missing = self._missing_report(df)
        missing.to_csv(vessel_dir / 'missing_values.csv')
        print(f"  [EDA] Missing values → {vessel_dir / 'missing_values.csv'}")

        # ── 3. Constraint violations ──────────────────────────────────────────
        violations = self._constraint_violations(df)
        violations.to_csv(vessel_dir / 'constraint_violations.csv', index=False)
        print(f"  [EDA] Constraint violations → {vessel_dir / 'constraint_violations.csv'}")
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

        # ── 7. Time-series overview (stacked rows) ───────────────────────────
        ts_path = self._plot_timeseries(df, plots_dir)
        print(f"  [EDA] Time-series overview → {ts_path}")

        # ── 8. Temporal coverage ──────────────────────────────────────────────
        coverage = self._temporal_coverage(df)
        coverage.to_csv(vessel_dir / 'temporal_coverage.csv', index=False)

        # ── 9. Univariate EDA (per-signal deep-dive) ─────────────────────────
        uni_dir = self._univariate_eda(df, plots_dir, vessel_dir=vessel_dir)
        print(f"  [EDA] Univariate EDA      → {uni_dir}/")

        # ── 10. Grouped EDA (distributions and correlations by category) ──────
        print("\n  [EDA] ── Phase 5b: Grouped EDA ────────────────────────────")
        grp_dist_path = self._plot_grouped_distributions(df, plots_dir)
        if grp_dist_path:
            print(f"  [EDA] Grouped distributions → {grp_dist_path}")
        grp_corr_path = self._plot_grouped_correlations(df, plots_dir)
        if grp_corr_path:
            print(f"  [EDA] Grouped correlations  → {grp_corr_path}")

        # ── 11. Bivariate EDA (domain-knowledge-guided relationships) ─────────
        print("\n  [EDA] ── Phase 6: Bivariate EDA (A–D) ─────────────────────")
        biv_dir = self._bivariate_eda(df, plots_dir, vessel_dir=vessel_dir)
        print(f"  [EDA] Bivariate EDA       → {biv_dir}/\n")

        # ── 12. Variable metadata (data dictionary + EDA findings) ────────────
        print("  [EDA] ── Phase 7: Variable Metadata ────────────────────────")
        self._variable_metadata(df, vessel_dir)

        date_col = self._date_col(df)
        if date_col:
            date_range = (str(df[date_col].min()), str(df[date_col].max()))
        else:
            date_range = ('N/A', 'N/A')

        result = {
            'n_records': len(df),
            'n_features': df.shape[1],
            'date_range': date_range,
            'vessel_id': vessel_cfg.vessel_id,
            'vessel_name': vessel_cfg.vessel_name,
            'constraint_violations': violations.set_index('check')['count'].to_dict(),
            'output_dir': str(vessel_dir),
            'audit': {
                'time_axis':  time_audit,
                'missingness': miss_audit,
                'unit_sanity': unit_audit,
            },
        }

        print(f"  [EDA] Complete. Outputs in {vessel_dir}/")
        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Descriptive statistics for numeric columns."""
        cols = self._numeric_cols(df, NUMERIC_COLS)
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
        cfg = getattr(self, '_vessel_cfg', None)
        preferred = cfg.feature_cols if (cfg and cfg.feature_cols) else FEATURE_COLS
        cols = self._numeric_cols(df, preferred)
        # Guard: keep only columns that actually have a numeric dtype to avoid
        # string-conversion errors when a vessel's raw data contains mixed types
        # (e.g. '2.445833333 N' in position columns).
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            return ''
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
        cfg = getattr(self, '_vessel_cfg', None)
        preferred = cfg.feature_cols if (cfg and cfg.feature_cols) else FEATURE_COLS
        cols = self._numeric_cols(df, preferred)
        # Keep only columns with a true numeric dtype (guards against object
        # columns that contain mixed strings/numbers from some vessel sources).
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
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
        """Propulsion power vs vessel speed scatter with anomaly highlighting."""
        # ── Column discovery ──────────────────────────────────────────────────
        # Try normalised names first, then raw names that may exist before
        # normalisation was applied.
        _POWER_CANDS = ['Main_Engine_Power_kW', 'PropulsionPowerTotal']
        _SPEED_CANDS = ['GPSSpeed_kn', 'SOG']

        power_col = next((c for c in _POWER_CANDS if c in df.columns), None)
        speed_col = next((c for c in _SPEED_CANDS if c in df.columns), None)

        # Last resort: ask vessel config
        if power_col is None or speed_col is None:
            cfg = getattr(self, '_vessel_cfg', None)
            if cfg:
                for c in cfg.feature_cols:
                    if power_col is None and ('power' in c.lower() or 'propuls' in c.lower()):
                        if c in df.columns:
                            power_col = c
                    if speed_col is None and ('speed' in c.lower() or 'sog' in c.lower()):
                        if c in df.columns:
                            speed_col = c

        path = out_dir / 'power_vs_speed.png'
        if power_col is None or speed_col is None:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5,
                    f'Power/speed columns not found.\n'
                    f'Tried power: {_POWER_CANDS}\n'
                    f'Tried speed: {_SPEED_CANDS}\n'
                    f'Available: {list(df.columns[:12])}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9)
            ax.axis('off')
            fig.savefig(path, dpi=100)
            plt.close(fig)
            return str(path)

        # ── Data preparation ──────────────────────────────────────────────────
        sub = df[[speed_col, power_col]].dropna()
        power = pd.to_numeric(sub[power_col], errors='coerce')
        speed = pd.to_numeric(sub[speed_col], errors='coerce')
        valid = power.notna() & speed.notna()
        power, speed = power[valid], speed[valid]

        n_neg    = int((power < 0).sum())
        n_zero_s = int(((speed == 0) & (power > 0)).sum())  # power with no movement

        # ── Plot ──────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 7))

        # Normal points
        normal_mask = power >= 0
        ax.scatter(speed[normal_mask], power[normal_mask],
                   alpha=0.25, s=4, c='steelblue', rasterized=True,
                   label=f'Normal  (n={normal_mask.sum():,})')

        # Negative-power anomalies
        if n_neg > 0:
            ax.scatter(speed[~normal_mask], power[~normal_mask],
                       alpha=0.85, s=25, c='red', zorder=5,
                       label=f'Negative power  (n={n_neg})')

        # Zero-reference line
        ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.6)

        # Polynomial trend line on normal data (helps spot hull-fouling deviations)
        try:
            if normal_mask.sum() > 50:
                from numpy.polynomial.polynomial import polyfit as _polyfit
                _s = speed[normal_mask].values
                _p = power[normal_mask].values
                _sx = np.sort(_s)
                _coef = np.polyfit(_s, _p, deg=3)
                ax.plot(_sx, np.polyval(_coef, _sx),
                        color='darkorange', lw=2, ls='-', alpha=0.7,
                        label='Cubic trend (expected envelope)')
        except Exception:
            pass

        # Power-at-zero-speed highlight
        if n_zero_s > 0:
            ax.scatter(speed[(speed == 0) & (power > 0)],
                       power[(speed == 0) & (power > 0)],
                       alpha=0.6, s=18, c='orange', zorder=4,
                       label=f'Power > 0 at rest  (n={n_zero_s})')

        ax.set_xlabel(f'{speed_col}  (knots)', fontsize=11)
        ax.set_ylabel(f'{power_col}  (kW)', fontsize=11)
        ax.set_title(
            f'Propulsion power vs speed\n'
            f'Vessel: {getattr(getattr(self, "_vessel_cfg", None), "vessel_name", "")}  '
            f'|  {len(speed):,} records',
            fontsize=12, fontweight='bold',
        )
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.25)

        # Annotation box
        ax.text(
            0.98, 0.05,
            f'Negative power    : {n_neg:,}\n'
            f'Power at rest     : {n_zero_s:,}\n'
            f'Max power         : {power.max():,.0f} kW\n'
            f'Max speed         : {speed.max():.1f} kn',
            transform=ax.transAxes, fontsize=8, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9),
        )

        plt.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(path)

    def _plot_timeseries(self, df: pd.DataFrame, out_dir: Path) -> str:
        """Time-series overview for the vessel's key operational signals."""
        # ── Column selection ──────────────────────────────────────────────────
        # Priority 1: use the vessel's own feature_cols (set by vessel_config)
        cfg = getattr(self, '_vessel_cfg', None)
        if cfg and cfg.feature_cols:
            plot_cols = [c for c in cfg.feature_cols if c in df.columns]
        else:
            # Priority 2: try all known schema variants, explicitly exclude
            # GPS lat/lon (useless as a time series)
            _CANDIDATES = [
                'GPSSpeed_kn', 'SOG',
                'Main_Engine_Power_kW', 'PropulsionPowerTotal',
                'Fuel_Consumption_rate', 'FuelMassFlowMETotal',
                'Fuel_Consumption_t_per_day',
                'Avg_draft_m', 'Trim_m',
                'RelWindSpeed_kn', 'TrueWindSpeed_kn',
                'RelWindAngle_deg',
            ]
            _seen, plot_cols = set(), []
            for c in _CANDIDATES:
                if c in df.columns and c not in _seen:
                    plot_cols.append(c)
                    _seen.add(c)
                    if len(plot_cols) == 6:
                        break

        # Priority 3: any numeric col that is not a coordinate / helper
        _EXCLUDE = {'GPS_LAT', 'GPS_LON', 'LAT', 'LONG', 'source_file'}
        if not plot_cols:
            plot_cols = [
                c for c in df.select_dtypes(include='number').columns
                if c not in _EXCLUDE
            ][:6]

        date_col = self._date_col(df)
        path = out_dir / 'timeseries_overview.png'

        if date_col is None or not plot_cols:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.text(0.5, 0.5,
                    'No date column or no plottable feature columns found.',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.axis('off')
            fig.savefig(path, dpi=100)
            plt.close(fig)
            return str(path)

        # ── Build plot ────────────────────────────────────────────────────────
        df_ts = (df[[date_col] + plot_cols]
                 .copy()
                 .dropna(subset=[date_col])
                 .sort_values(date_col))
        # Ensure timestamp is parsed
        df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce', utc=True)
        df_ts = df_ts.dropna(subset=[date_col])

        n_panels = len(plot_cols)
        vessel_name = cfg.vessel_name if cfg else ''
        fig, axes = plt.subplots(
            n_panels, 1,
            figsize=(16, max(3, 2.6 * n_panels)),
            sharex=True,
        )
        if n_panels == 1:
            axes = [axes]

        _COLORS = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple', 'sienna']
        for i, (ax, col) in enumerate(zip(axes, plot_cols)):
            col_data = pd.to_numeric(df_ts[col], errors='coerce')
            ax.plot(df_ts[date_col], col_data, lw=0.5,
                    color=_COLORS[i % len(_COLORS)], alpha=0.8, rasterized=True)
            # IQR shading for anomaly visibility
            q1, q3 = col_data.quantile(0.01), col_data.quantile(0.99)
            ax.axhspan(q1, q3, alpha=0.06, color=_COLORS[i % len(_COLORS)])
            ax.set_ylabel(col, fontsize=8)
            ax.grid(True, alpha=0.25)
            n_valid = int(col_data.notna().sum())
            n_miss  = int(col_data.isna().sum())
            ax.text(1.002, 0.5,
                    f'n={n_valid:,}\nmiss={n_miss:,}\n'
                    f'med={col_data.median():.2g}',
                    transform=ax.transAxes, fontsize=7, va='center',
                    color='dimgray')

        axes[0].set_title(
            f'Time-series overview — {vessel_name}\n'
            f'Showing {n_panels} key operational signals',
            fontsize=12, fontweight='bold',
        )
        axes[-1].set_xlabel('Time', fontsize=9)
        plt.tight_layout(rect=[0, 0, 0.96, 1])

        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(path)

    def _temporal_coverage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute daily record counts and sampling gaps."""
        date_col = self._date_col(df)
        if date_col is None:
            return pd.DataFrame(columns=['date', 'records', 'median_gap_sec', 'max_gap_sec'])

        df_ts = df[[date_col]].dropna().copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col], utc=True, errors='coerce')
        df_ts = df_ts.dropna().sort_values(date_col)
        df_ts['date'] = df_ts[date_col].dt.date
        daily_counts = df_ts.groupby('date').size().reset_index(name='records')

        gaps = df_ts[date_col].diff().dt.total_seconds().dropna()
        daily_counts['median_gap_sec'] = gaps.median()
        daily_counts['max_gap_sec'] = gaps.max()

        return daily_counts

    # ──────────────────────────────────────────────────────────────────────────

    def _univariate_eda(
        self,
        df: pd.DataFrame,
        plots_dir: Path,
        vessel_dir: Optional[Path] = None,
    ) -> str:
        """
        Phase 4 — Univariate EDA: per-signal 6-panel deep-dive.

        For each feature signal produces a 6-panel figure:
          [0,0]  Histogram + KDE  (physical limits marked)
          [0,1]  Empirical CDF + quantile reference table
          [1,0]  Time-series sample  (downsampled; outliers highlighted in red)
          [1,1]  Rate-of-change distribution  (first-difference / time)
          [2,0]  Monthly boxplots  (long-term trends across calendar months)
          [2,1]  Weekly boxplots   (ISO week — short-term / schedule variation)

        Outputs
        -------
        plots/univariate/{signal}.png  — one 6-panel figure per signal
        univariate_report.csv          — quantiles + outlier counts per signal
        """
        from datetime import datetime as _pdt

        uni_dir = plots_dir / 'univariate'
        uni_dir.mkdir(exist_ok=True)

        cfg      = getattr(self, '_vessel_cfg', None)
        date_col = self._date_col(df)

        # ── Columns to analyse ────────────────────────────────────────────────
        if cfg and cfg.feature_cols:
            target_cols = [c for c in cfg.feature_cols if c in df.columns]
        else:
            target_cols = self._numeric_cols(df, FEATURE_COLS)
        # Always include draft and angle columns if present
        _XTRA = ('DRAFTAFT', 'DRAFTFWD', 'RelWindAngle_deg', 'AWA', 'Trim_m')
        for _xc in _XTRA:
            if _xc in df.columns and _xc not in target_cols:
                target_cols.append(_xc)

        # ── Parse timestamps once ─────────────────────────────────────────────
        ts_utc: Optional[pd.Series] = None
        if date_col:
            ts_utc = pd.to_datetime(df[date_col], errors='coerce', utc=True)

        N_MAX_TS = 8_000   # max scatter points in the time-series panel
        report_rows: list = []

        for col in target_cols:
            if col not in df.columns:
                continue

            # ── 1. Numeric series ──────────────────────────────────────────────
            raw   = pd.to_numeric(df[col], errors='coerce')
            valid = raw.dropna()
            n_total   = len(raw)
            n_valid   = len(valid)
            n_missing = n_total - n_valid
            if n_valid < 20:
                continue

            # ── 2. Profile lookup ──────────────────────────────────────────────
            profile = UNIVARIATE_SIGNAL_PROFILES.get(col)
            if profile is None:
                for _std, _p in UNIVARIATE_SIGNAL_PROFILES.items():
                    if col in _p.get('alias', []):
                        profile = _p
                        break
            if profile is None:
                profile = {
                    'label': col, 'unit': '',
                    'phys_lo': None, 'phys_hi': None,
                    'negative_is_anomaly': False, 'alias': [], 'expect': '',
                }
            label   = profile['label']
            unit    = profile['unit']
            phys_lo = profile.get('phys_lo')
            phys_hi = profile.get('phys_hi')

            # ── 3. Statistics ──────────────────────────────────────────────────
            q01, q05, q10, q25, q50, q75, q90, q95, q99 = valid.quantile(
                [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            ).values
            vmean  = float(valid.mean())
            vstd   = float(valid.std())
            vmin   = float(valid.min())
            vmax   = float(valid.max())

            n_neg      = int((valid < 0).sum())
            n_zero     = int((valid == 0).sum())
            n_phys_out = 0
            if phys_lo is not None:
                n_phys_out += int((valid < phys_lo).sum())
            if phys_hi is not None:
                n_phys_out += int((valid > phys_hi).sum())

            # ── 4. Temporal grouping frame (month + ISO week) ─────────────────
            df_grp: Optional[pd.DataFrame] = None
            if ts_utc is not None:
                df_grp = (
                    pd.DataFrame({'v': raw, 'ts': ts_utc})
                    .dropna(subset=['ts'])
                    .copy()
                )
                df_grp['v'] = pd.to_numeric(df_grp['v'], errors='coerce')
                df_grp = (
                    df_grp.dropna(subset=['v'])
                    .sort_values('ts')
                    .reset_index(drop=True)
                )
                df_grp['month']    = df_grp['ts'].dt.strftime('%Y-%m')
                df_grp['iso_week'] = df_grp['ts'].dt.strftime('%G-W%V')

            # ── 5. Rate of change (normalised by elapsed seconds) ─────────────
            roc_label = f'\u0394{unit}/s' if ts_utc is not None else f'\u0394{unit}/step'
            if df_grp is not None and len(df_grp) > 1:
                _dt_s = df_grp['ts'].diff().dt.total_seconds()
                _dv   = df_grp['v'].diff()
                _ok   = (_dt_s > 0) & (_dt_s <= 300)  # consecutive, within 5 min
                roc   = (_dv[_ok] / _dt_s[_ok]).dropna()
            else:
                roc = valid.diff().dropna()

            roc_abs    = roc.abs()
            roc_median = float(roc_abs.median()) if len(roc) > 0 else 0.0
            roc_p99    = float(roc_abs.quantile(0.99)) if len(roc) > 1 else 0.0

            # ── 6. Report row ─────────────────────────────────────────────────
            report_rows.append({
                'signal': col,
                'feature_group': _group_for_col(col, cfg),
                'label': label, 'unit': unit,
                'n_total': n_total, 'n_valid': n_valid, 'n_missing': n_missing,
                'pct_missing': round(n_missing / n_total * 100, 2),
                'mean': round(vmean, 4), 'std': round(vstd, 4),
                'min': round(vmin, 4),
                'q01': round(q01, 4), 'q05': round(q05, 4),
                'q10': round(q10, 4), 'q25': round(q25, 4),
                'q50': round(q50, 4), 'q75': round(q75, 4),
                'q90': round(q90, 4), 'q95': round(q95, 4),
                'q99': round(q99, 4),
                'max': round(vmax, 4),
                'n_negative': n_neg, 'n_zero': n_zero,
                'n_physical_outliers': n_phys_out,
                'roc_median_per_s': round(roc_median, 8),
                'roc_p99_per_s':    round(roc_p99, 8),
            })

            # ── 7. Six-panel figure ───────────────────────────────────────────
            vessel_name = getattr(cfg, 'vessel_name', '') if cfg else ''
            fig = plt.figure(figsize=(20, 17))
            fig.suptitle(
                f'Univariate EDA  \u2015  {label}  [{unit}]\n'
                f'n={n_valid:,}  |  missing={n_missing:,} ({n_missing / n_total * 100:.1f}%)'
                f'  |  {vessel_name}',
                fontsize=13, fontweight='bold', y=0.995,
            )
            gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.30,
                                  top=0.945, bottom=0.07)
            ax_hist = fig.add_subplot(gs[0, 0])
            ax_ecdf = fig.add_subplot(gs[0, 1])
            ax_ts   = fig.add_subplot(gs[1, 0])
            ax_roc  = fig.add_subplot(gs[1, 1])
            ax_mon  = fig.add_subplot(gs[2, 0])
            ax_wk   = fig.add_subplot(gs[2, 1])

            # ─ (0,0) Histogram + KDE ─────────────────────────────────────────
            _vlo = float(valid.quantile(0.001))
            _vhi = float(valid.quantile(0.999))
            if _vhi > _vlo + 1e-12:
                _clipped = valid.clip(lower=_vlo, upper=_vhi)
                _nbins   = min(100, max(20, int(n_valid ** 0.38)))
                ax_hist.hist(_clipped, bins=_nbins, density=True,
                             color='steelblue', alpha=0.55, edgecolor='none',
                             label='Histogram (density)')
                try:
                    from scipy.stats import gaussian_kde as _kde_fn
                    _kde_obj = _kde_fn(_clipped.values, bw_method='silverman')
                    _xs = np.linspace(_vlo, _vhi, 400)
                    ax_hist.plot(_xs, _kde_obj(_xs), lw=2.0, color='navy', label='KDE')
                except Exception:
                    pass
                _q25_labeled = False
                for _qv, _ql, _lc in [
                    (q25, 'Q25/Q75', 'darkorange'),
                    (q75, None,      'darkorange'),
                ]:
                    if _vlo <= _qv <= _vhi:
                        ax_hist.axvline(_qv, color=_lc, lw=1.4, ls='--', alpha=0.85,
                                        label=_ql if not _q25_labeled else None)
                        _q25_labeled = True
                if _vlo <= q50 <= _vhi:
                    ax_hist.axvline(q50, color='red', lw=2.0, ls='-', alpha=0.9,
                                    label='Median')
                if phys_lo is not None and _vlo < phys_lo <= _vhi:
                    ax_hist.axvline(phys_lo, color='darkred', lw=2.0, ls='-.',
                                    label=f'Phys.lo={phys_lo}')
                if phys_hi is not None and _vlo <= phys_hi < _vhi:
                    ax_hist.axvline(phys_hi, color='darkred', lw=2.0, ls='-.',
                                    label=f'Phys.hi={phys_hi}')
                ax_hist.legend(fontsize=7, loc='upper right', ncol=1)
            else:
                ax_hist.text(0.5, 0.5, 'Constant / near-constant value',
                             ha='center', va='center',
                             transform=ax_hist.transAxes, fontsize=10, color='gray')
            ax_hist.text(
                0.02, 0.97,
                f'Negative       : {n_neg:,}\n'
                f'Zero           : {n_zero:,}\n'
                f'Physical viol. : {n_phys_out:,}',
                transform=ax_hist.transAxes, fontsize=7.5, va='top', ha='left',
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#fffbe6',
                          edgecolor='goldenrod', alpha=0.95),
            )
            ax_hist.set_xlabel(f'{label} [{unit}]', fontsize=9)
            ax_hist.set_ylabel('Density', fontsize=9)
            ax_hist.set_title('Histogram + KDE', fontsize=10, fontweight='bold')
            ax_hist.grid(True, alpha=0.2)

            # ─ (0,1) Empirical CDF + quantile reference ───────────────────────
            _sv = np.sort(valid.values)
            _cp = np.arange(1, len(_sv) + 1) / len(_sv)
            if len(_sv) > 5_000:
                _ii = np.round(np.linspace(0, len(_sv) - 1, 5_000)).astype(int)
                _sv = _sv[_ii]
                _cp = _cp[_ii]
            ax_ecdf.plot(_sv, _cp, lw=1.5, color='steelblue')
            ax_ecdf.fill_between(_sv, _cp, alpha=0.07, color='steelblue')
            for _qv, _qp, _qc in [
                (q01, 0.01, '#bbbbbb'), (q05, 0.05, '#999999'),
                (q25, 0.25, 'darkorange'), (q50, 0.50, 'red'),
                (q75, 0.75, 'darkorange'), (q95, 0.95, '#999999'),
                (q99, 0.99, '#bbbbbb'),
            ]:
                ax_ecdf.plot([vmin, _qv, _qv], [_qp, _qp, 0.0],
                             color=_qc, lw=0.8, ls='--', alpha=0.7)
            ax_ecdf.text(
                0.97, 0.03,
                f'min  = {vmin:>12.4g}\n'
                f' Q1% = {q01:>12.4g}\n'
                f' Q5% = {q05:>12.4g}\n'
                f'Q10% = {q10:>12.4g}\n'
                f'Q25% = {q25:>12.4g}\n'
                f'Q50% = {q50:>12.4g}  \u2190 median\n'
                f'Q75% = {q75:>12.4g}\n'
                f'Q90% = {q90:>12.4g}\n'
                f'Q95% = {q95:>12.4g}\n'
                f'Q99% = {q99:>12.4g}\n'
                f'max  = {vmax:>12.4g}\n'
                f'\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
                f'mean = {vmean:>12.4g}\n'
                f' std = {vstd:>12.4g}',
                transform=ax_ecdf.transAxes, fontsize=7.5, va='bottom', ha='right',
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                          edgecolor='lightgray', alpha=0.97),
            )
            ax_ecdf.set_xlabel(f'{label} [{unit}]', fontsize=9)
            ax_ecdf.set_ylabel('Cumulative probability', fontsize=9)
            ax_ecdf.set_title('Empirical CDF + quantile reference',
                              fontsize=10, fontweight='bold')
            ax_ecdf.set_ylim(-0.01, 1.03)
            ax_ecdf.grid(True, alpha=0.2)

            # ─ (1,0) Time-series sample ─────────────────────────────────────
            if ts_utc is not None:
                _df_ts = (
                    pd.DataFrame({'v': raw, 'ts': ts_utc})
                    .dropna(subset=['ts'])
                    .copy()
                )
                _df_ts['v'] = pd.to_numeric(_df_ts['v'], errors='coerce')
                _df_ts = _df_ts.sort_values('ts').reset_index(drop=True)
                if len(_df_ts) > N_MAX_TS:
                    _step = max(1, len(_df_ts) // N_MAX_TS)
                    _df_ts = _df_ts.iloc[::_step].copy()
                _tv  = _df_ts['v']
                _tt  = _df_ts['ts']
                _out = pd.Series(False, index=_tv.index)
                if phys_lo is not None:
                    _out = _out | (_tv < phys_lo)
                if phys_hi is not None:
                    _out = _out | (_tv > phys_hi)
                ax_ts.scatter(_tt[~_out], _tv[~_out], s=1, alpha=0.2,
                              c='steelblue', rasterized=True, label='Normal')
                if _out.any():
                    ax_ts.scatter(_tt[_out], _tv[_out], s=5, alpha=0.65,
                                  c='crimson', rasterized=True,
                                  label=f'Outlier ({int(_out.sum()):,})')
                ax_ts.axhline(q50, color='darkorange', lw=0.8, ls='--',
                              alpha=0.45, label='Median')
                if phys_lo is not None:
                    ax_ts.axhline(phys_lo, color='darkred', lw=1.0, ls='-.', alpha=0.55)
                if phys_hi is not None:
                    ax_ts.axhline(phys_hi, color='darkred', lw=1.0, ls='-.', alpha=0.55)
                ax_ts.legend(fontsize=7, loc='upper right', markerscale=4)
                ax_ts.set_xlabel('Time', fontsize=9)
            else:
                _vi = valid.reset_index(drop=True)
                if len(_vi) > N_MAX_TS:
                    _vi = _vi.sample(N_MAX_TS, random_state=42).sort_index()
                ax_ts.scatter(range(len(_vi)), _vi.values, s=1.5, alpha=0.25,
                              c='steelblue', rasterized=True)
                ax_ts.set_xlabel('Row index', fontsize=9)
            ax_ts.set_ylabel(f'{label} [{unit}]', fontsize=9)
            ax_ts.set_title('Time-series sample  (outliers in red)',
                            fontsize=10, fontweight='bold')
            ax_ts.grid(True, alpha=0.2)

            # ─ (1,1) Rate-of-change distribution ────────────────────────────
            if len(roc) > 20:
                _roc_lo = float(roc.quantile(0.005))
                _roc_hi = float(roc.quantile(0.995))
                _roc_c  = roc.clip(lower=_roc_lo, upper=_roc_hi)
                _nb_roc = min(80, max(20, int(len(_roc_c) ** 0.35)))
                ax_roc.hist(_roc_c, bins=_nb_roc, color='seagreen', alpha=0.65,
                            edgecolor='none', label=f'n={len(roc):,} pairs')
                ax_roc.axvline(0, color='gray', lw=1.2, ls='--', alpha=0.6)
                if roc_p99 > 0:
                    ax_roc.axvline( roc_p99, color='crimson', lw=1.5, ls=':',
                                    label=f'\u00b1p99(|\u0394V|)={roc_p99:.3g}')
                    ax_roc.axvline(-roc_p99, color='crimson', lw=1.5, ls=':')
                ax_roc.text(
                    0.02, 0.97,
                    f'med|RoC| = {roc_median:.4g} {roc_label}\n'
                    f'p99|RoC| = {roc_p99:.4g} {roc_label}',
                    transform=ax_roc.transAxes, fontsize=8, va='top', ha='left',
                    family='monospace',
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='#f0fff0',
                              edgecolor='gray', alpha=0.9),
                )
                ax_roc.legend(fontsize=7, loc='upper right')
            else:
                ax_roc.text(0.5, 0.5, 'Insufficient data for\nrate-of-change analysis',
                            ha='center', va='center', transform=ax_roc.transAxes,
                            fontsize=10, color='gray')
            ax_roc.set_title('Rate-of-change distribution',
                             fontsize=10, fontweight='bold')
            ax_roc.set_xlabel(f'Rate of change [{roc_label}]', fontsize=9)
            ax_roc.set_ylabel('Count', fontsize=9)
            ax_roc.grid(True, alpha=0.2)

            # ─ (2,0) Monthly boxplots ─────────────────────────────────────────
            if df_grp is not None and df_grp['month'].nunique() >= 2:
                _months  = sorted(df_grp['month'].unique())
                _bp_data = [
                    df_grp.loc[df_grp['month'] == m, 'v'].values
                    for m in _months
                ]
                _bp_m = ax_mon.boxplot(
                    _bp_data, labels=_months, patch_artist=True,
                    medianprops=dict(color='red', lw=2),
                    flierprops=dict(marker='.', ms=2.5, alpha=0.25, color='crimson'),
                    whiskerprops=dict(lw=1.2),
                    boxprops=dict(alpha=0.55),
                )
                _cmap_m = plt.colormaps.get_cmap('Blues')
                _nm = len(_months)
                for _k, _box in enumerate(_bp_m['boxes']):
                    _box.set_facecolor(_cmap_m(0.35 + 0.55 * _k / max(1, _nm - 1)))
                _meds_m = [float(np.median(d)) for d in _bp_data if len(d) > 0]
                ax_mon.plot(range(1, len(_meds_m) + 1), _meds_m,
                            color='navy', lw=1.5, ls='--', marker='o', ms=4,
                            alpha=0.7, label='Median trend')
                ax_mon.axhline(q50, color='darkorange', lw=0.8, ls=':', alpha=0.45)
                ax_mon.set_xticklabels(_months, rotation=30, ha='right', fontsize=8)
                ax_mon.legend(fontsize=7)
            else:
                _msg_m = ('No timestamp for grouping' if df_grp is None
                          else f'Only {df_grp["month"].nunique()} month(s) \u2014 need \u2265 2')
                ax_mon.text(0.5, 0.5, _msg_m, ha='center', va='center',
                            transform=ax_mon.transAxes, fontsize=10, color='gray')
            ax_mon.set_ylabel(f'{label} [{unit}]', fontsize=9)
            ax_mon.set_title('Monthly boxplots', fontsize=10, fontweight='bold')
            ax_mon.grid(True, alpha=0.2, axis='y')

            # ─ (2,1) Weekly boxplots (ISO week) ──────────────────────────────
            if df_grp is not None and df_grp['iso_week'].nunique() >= 2:
                _weeks   = sorted(df_grp['iso_week'].unique())
                _wp_data = [
                    df_grp.loc[df_grp['iso_week'] == w, 'v'].values
                    for w in _weeks
                ]
                _wbw  = max(0.3, min(0.85, 18.0 / len(_weeks)))
                _bp_w = ax_wk.boxplot(
                    _wp_data,
                    labels=[w.split('-W')[-1] for w in _weeks],
                    patch_artist=True, widths=_wbw,
                    medianprops=dict(color='red', lw=1.5),
                    flierprops=dict(marker='.', ms=1.5, alpha=0.15, color='crimson'),
                    whiskerprops=dict(lw=0.9),
                    boxprops=dict(alpha=0.45),
                    showfliers=(len(_weeks) <= 25),
                )
                _cmap_w = plt.colormaps.get_cmap('Greens')
                _nw = len(_weeks)
                for _k, _box in enumerate(_bp_w['boxes']):
                    _box.set_facecolor(_cmap_w(0.3 + 0.6 * _k / max(1, _nw - 1)))
                _wmeds = [
                    float(np.median(d)) if len(d) > 0 else np.nan
                    for d in _wp_data
                ]
                ax_wk.plot(range(1, len(_wmeds) + 1), _wmeds,
                           color='darkgreen', lw=1.5, ls='--', marker='o', ms=3,
                           alpha=0.7, label='Median trend')
                ax_wk.axhline(q50, color='darkorange', lw=0.8, ls=':', alpha=0.45)
                # Month separator vlines + labels using ISO calendar
                _prev_m: Optional[str] = None
                for _ki, _wstr in enumerate(_weeks):
                    _yr_s, _wn_s = _wstr.split('-W')
                    _monday = _pdt.fromisocalendar(int(_yr_s), int(_wn_s), 1)
                    _this_m = _monday.strftime('%b')
                    if _this_m != _prev_m:
                        ax_wk.axvline(_ki + 1, color='gray', lw=0.7, ls='--',
                                      alpha=0.35)
                        ax_wk.text(_ki + 1.1, 1.0, _this_m, fontsize=7,
                                   color='dimgray', va='top',
                                   transform=ax_wk.get_xaxis_transform())
                        _prev_m = _this_m
                _xt_step = max(1, len(_weeks) // 16)
                _xt      = list(range(0, len(_weeks), _xt_step))
                ax_wk.set_xticks([x + 1 for x in _xt])
                ax_wk.set_xticklabels(
                    [_weeks[x] for x in _xt], rotation=35, ha='right', fontsize=7
                )
                ax_wk.legend(fontsize=7)
            else:
                _msg_w = ('No timestamp for grouping' if df_grp is None
                          else f'Only {df_grp["iso_week"].nunique()} week(s) \u2014 need \u2265 2')
                ax_wk.text(0.5, 0.5, _msg_w, ha='center', va='center',
                           transform=ax_wk.transAxes, fontsize=10, color='gray')
            ax_wk.set_ylabel(f'{label} [{unit}]', fontsize=9)
            ax_wk.set_title('Weekly boxplots (ISO week)', fontsize=10, fontweight='bold')
            ax_wk.grid(True, alpha=0.2, axis='y')

            # "What to look for" footer note
            _expect = profile.get('expect', '')
            if _expect:
                fig.text(0.5, 0.002, _expect, ha='center', va='bottom',
                         fontsize=8, color='dimgray', style='italic', wrap=True)

            safe_col = re.sub(r'[/\\:*?"<>|]', '_', col)
            fig_path = uni_dir / f'{safe_col}.png'
            fig.savefig(fig_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"  [EDA/Univariate] {col:<38s} \u2192 plots/univariate/{safe_col}.png")

        # ── Save summary CSV ──────────────────────────────────────────────────
        if report_rows:
            _rdir     = vessel_dir if vessel_dir is not None else plots_dir.parent
            _rep_path = _rdir / 'univariate_report.csv'
            pd.DataFrame(report_rows).to_csv(_rep_path, index=False)
            print(f"  [EDA/Univariate] Summary \u2192 {_rep_path}")

        return str(uni_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 5b — Grouped EDA plots (distributions + correlations by category)
    # ══════════════════════════════════════════════════════════════════════════

    def _plot_grouped_distributions(
        self,
        df: pd.DataFrame,
        plots_dir: Path,
    ) -> str:
        """
        One figure with one section per feature group.

        For each group a row of histograms is produced; groups are stacked
        vertically and separated by a bold category header.  This gives a
        quick visual overview of all signals organised by topic.

        Output
        ------
        plots/grouped_distributions.png
        """
        cfg = getattr(self, '_vessel_cfg', None)

        # Build ordered list of (group_key, [cols_in_df])
        all_groups: dict = {}
        for group_key in ['engine_propulsion', 'navigation', 'draft', 'weather', 'trip']:
            cols_in_group = [
                c for c in df.columns
                if _group_for_col(c, cfg) == group_key
                and pd.api.types.is_numeric_dtype(df[c])
                and c not in ('GPS_LAT', 'GPS_LON')   # lat/lon as histograms are rarely useful
                and df[c].notna().sum() >= 20
            ]
            if cols_in_group:
                all_groups[group_key] = cols_in_group

        if not all_groups:
            return ''

        # Layout: 4 columns; one row per signal; group header rows inserted
        NCOLS = 4
        _COLORS = {
            'engine_propulsion': '#2471a3',
            'navigation':        '#1a8a50',
            'draft':             '#8e5b22',
            'weather':           '#7d3c98',
            'trip':              '#c0392b',
            'other':             '#7f8c8d',
        }

        # Count total histogram panels
        total_panels = sum(len(cols) for cols in all_groups.values())
        # Extra header rows (1 per group)
        total_header_rows = len(all_groups)
        ncols = NCOLS
        # Each group occupies ceil(len(cols)/NCOLS) data rows + 1 header row
        total_rows = sum(
            1 + (len(cols) + ncols - 1) // ncols
            for cols in all_groups.values()
        )

        fig = plt.figure(figsize=(ncols * 4.5, total_rows * 2.8))
        vessel_name = getattr(cfg, 'vessel_name', '') if cfg else ''
        fig.suptitle(
            f'Feature Distributions by Category — {vessel_name}',
            fontsize=14, fontweight='bold', y=1.002,
        )

        # Manual subplot placement via gridspec
        gs = fig.add_gridspec(total_rows, ncols, hspace=0.55, wspace=0.35)

        row_idx = 0
        for group_key, cols in all_groups.items():
            group_label = _GROUP_LABELS.get(group_key, group_key)
            color = _COLORS.get(group_key, '#7f8c8d')

            # Header: span all columns
            ax_hdr = fig.add_subplot(gs[row_idx, :])
            ax_hdr.set_facecolor(color)
            ax_hdr.text(
                0.01, 0.5, group_label,
                transform=ax_hdr.transAxes,
                fontsize=11, fontweight='bold', va='center', color='white',
            )
            ax_hdr.set_xticks([])
            ax_hdr.set_yticks([])
            row_idx += 1

            # Histogram panels
            for panel_idx, col in enumerate(cols):
                cur_row = row_idx + panel_idx // ncols
                cur_col = panel_idx % ncols
                ax = fig.add_subplot(gs[cur_row, cur_col])
                data = df[col].dropna()
                if len(data) > 0:
                    ax.hist(
                        data.clip(
                            lower=float(data.quantile(0.01)),
                            upper=float(data.quantile(0.99)),
                        ),
                        bins=50, color=color, alpha=0.75, edgecolor='none',
                    )
                ax.set_title(col, fontsize=8, fontweight='bold', color=color, pad=2)
                ax.set_ylabel('Count', fontsize=7)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.2)

            group_data_rows = (len(cols) + ncols - 1) // ncols
            row_idx += group_data_rows

        path = plots_dir / 'grouped_distributions.png'
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        return str(path)

    def _plot_grouped_correlations(
        self,
        df: pd.DataFrame,
        plots_dir: Path,
    ) -> str:
        """
        One correlation heatmap per non-empty feature group.

        Each group heatmap shows pairwise Pearson correlations only for the
        signals belonging to that group (plus cross-group signals like speed
        appear in every relevant group for context).

        Output
        ------
        plots/grouped_correlations.png
        """
        cfg = getattr(self, '_vessel_cfg', None)
        vessel_name = getattr(cfg, 'vessel_name', '') if cfg else ''

        _COLORS = {
            'engine_propulsion': 'Blues',
            'navigation':        'Greens',
            'draft':             'Oranges',
            'weather':           'Purples',
            'trip':              'Reds',
            'other':             'Greys',
        }
        # Anchor signals that appear in every group heatmap for cross-group context
        _ANCHORS = ['Main_Engine_Power_kW', 'GPSSpeed_kn']

        group_plots = []
        for group_key in ['engine_propulsion', 'navigation', 'draft', 'weather', 'trip']:
            group_cols = [
                c for c in df.columns
                if _group_for_col(c, cfg) == group_key
                and pd.api.types.is_numeric_dtype(df[c])
                and df[c].notna().sum() >= 20
            ]
            # Add anchors if they exist and are not already in the group
            for anchor in _ANCHORS:
                if anchor in df.columns and anchor not in group_cols:
                    group_cols.append(anchor)

            group_cols = [c for c in group_cols if pd.api.types.is_numeric_dtype(df[c])]
            if len(group_cols) < 2:
                continue

            corr = df[group_cols].corr()
            group_plots.append((group_key, corr))

        if not group_plots:
            return ''

        n_groups = len(group_plots)
        # Arrange in a 1×n_groups row (or wrap at 3)
        ncols = min(3, n_groups)
        nrows = (n_groups + ncols - 1) // ncols
        max_cols = max(len(c.columns) for _, c in group_plots)
        fig_w = min(8, max(4, max_cols * 1.0)) * ncols
        fig_h = min(8, max(4, max_cols * 1.0)) * nrows

        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
        axes = np.array(axes).flatten()

        for i, (group_key, corr) in enumerate(group_plots):
            ax = axes[i]
            group_label = _GROUP_LABELS.get(group_key, group_key)
            cmap = _COLORS.get(group_key, 'Greys')
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(
                corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdYlGn', vmin=-1, vmax=1, center=0,
                linewidths=0.4, annot_kws={'size': 8}, ax=ax,
                cbar_kws={'shrink': 0.7},
            )
            ax.set_title(group_label, fontsize=11, fontweight='bold', pad=6)
            ax.tick_params(labelsize=8)

        # Hide unused axes
        for j in range(len(group_plots), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f'Grouped Correlation Heatmaps — {vessel_name}',
            fontsize=13, fontweight='bold', y=1.01,
        )
        plt.tight_layout()

        path = plots_dir / 'grouped_correlations.png'
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        return str(path)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 6 — Bivariate EDA (domain-knowledge-guided relationships)
    # ══════════════════════════════════════════════════════════════════════════

    def _bivariate_eda(
        self,
        df: pd.DataFrame,
        plots_dir: Path,
        vessel_dir: Optional[Path] = None,
    ) -> str:
        """
        Phase 6 — Bivariate EDA: four domain-critical relationships.

        A. Power vs Speed          — displacement law, draft & wind colouring,
                                     hexbin density, monthly facets
        B. Fuel flow vs Power      — efficiency trend, monthly slope, residuals
        C. Draft vs Speed/Power    — loading effect on resistance
        D. Wind vs Power/Fuel      — weather penalty, AWA-sector comparison

        Outputs
        -------
        plots/bivariate/A_power_vs_speed.png
        plots/bivariate/B_fuel_vs_power.png
        plots/bivariate/C_draft_vs_power.png
        plots/bivariate/D_wind_vs_power.png
        bivariate_report.csv
        """
        biv_dir = plots_dir / 'bivariate'
        biv_dir.mkdir(exist_ok=True)

        cfg      = getattr(self, '_vessel_cfg', None)
        date_col = self._date_col(df)

        # ── Column lookup helpers ─────────────────────────────────────────────
        def _col(*cands):
            for c in cands:
                if c in df.columns:
                    return c
            return None

        POWER  = _col('Main_Engine_Power_kW', 'PropulsionPowerTotal')
        SPEED  = _col('GPSSpeed_kn', 'SOG')
        FUEL   = _col('Fuel_Consumption_rate', 'FuelMassFlowMETotal',
                      'Fuel_Consumption_t_per_day')
        DRAFT  = _col('Avg_draft_m')
        DAFT   = _col('DRAFTAFT')
        DFWD   = _col('DRAFTFWD')
        TRIM   = _col('Trim_m')
        AWS    = _col('RelWindSpeed_kn', 'AWS')
        AWA    = _col('RelWindAngle_deg', 'AWA')
        LAT    = _col('GPS_LAT', 'LAT', 'Latitude (deg)')
        LON    = _col('GPS_LON', 'LON', 'Longitude (deg)')

        # Vessel label for suptitles
        vessel_name = getattr(cfg, 'vessel_name', '') if cfg else ''

        # Timestamp series (UTC) used for monthly grouping
        ts_utc: Optional[pd.Series] = None
        if date_col:
            ts_utc = pd.to_datetime(df[date_col], errors='coerce', utc=True)

        # Shared colour cycle for months
        _MONTH_CMAP = plt.cm.tab10

        # Numeric helper ──────────────────────────────────────────────────────
        def _num(col):
            return pd.to_numeric(df[col], errors='coerce') if col else None

        pwr  = _num(POWER)
        spd  = _num(SPEED)
        fuel = _num(FUEL)
        dft  = _num(DRAFT)
        trim = _num(TRIM)
        aws  = _num(AWS)
        awa  = _num(AWA)

        report_rows: list = []

        # ══════════════════════════════════════════════════════════════════════
        # A. Power vs Speed
        # ══════════════════════════════════════════════════════════════════════
        if POWER and SPEED:
            _mask = pwr.notna() & spd.notna() & (pwr >= 0) & (spd >= 0)
            _pw = pwr[_mask].values
            _sp = spd[_mask].values

            n_pts = len(_pw)
            SMAX  = 15_000   # scatter cap
            _idx  = (
                np.random.default_rng(42).choice(n_pts, SMAX, replace=False)
                if n_pts > SMAX else np.arange(n_pts)
            )
            _pw_s, _sp_s = _pw[_idx], _sp[_idx]

            # Cubic trend on full data
            _poly_ok = n_pts > 100
            _px = np.linspace(0, _sp.max(), 200) if _poly_ok else None
            if _poly_ok:
                try:
                    _coeffs = np.polyfit(_sp, _pw, 3)
                    _py = np.polyval(_coeffs, _px)
                    _py = np.clip(_py, 0, None)
                except Exception:
                    _poly_ok = False

            fig, axes = plt.subplots(2, 3, figsize=(21, 13))
            fig.suptitle(
                f'Bivariate EDA  A — Power vs Speed\n{vessel_name}',
                fontsize=13, fontweight='bold', y=1.0,
            )

            # A1: plain scatter
            ax = axes[0, 0]
            ax.scatter(_sp_s, _pw_s, s=3, alpha=0.25, c='steelblue', rasterized=True)
            if _poly_ok:
                ax.plot(_px, _py, lw=2.0, color='darkorange', label='Cubic trend')
                ax.legend(fontsize=8)
            ax.set_xlabel(f'Speed ({SPEED}) [kn]'); ax.set_ylabel(f'Power ({POWER}) [kW]')
            ax.set_title('A1  Plain scatter', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25)

            # A2: coloured by draft
            ax = axes[0, 1]
            if dft is not None:
                _dft_s = dft[_mask].values[_idx]
                _valid_dft = np.isfinite(_dft_s)
                sc = ax.scatter(
                    _sp_s[_valid_dft], _pw_s[_valid_dft],
                    c=_dft_s[_valid_dft], cmap='RdYlGn_r',
                    vmin=np.nanpercentile(_dft_s, 5),
                    vmax=np.nanpercentile(_dft_s, 95),
                    s=3, alpha=0.4, rasterized=True,
                )
                plt.colorbar(sc, ax=ax, label=f'{DRAFT} [m]', fraction=0.04)
            else:
                ax.scatter(_sp_s, _pw_s, s=3, alpha=0.25, c='steelblue', rasterized=True)
                ax.text(0.5, 0.98, 'Draft column not available',
                        ha='center', va='top', transform=ax.transAxes,
                        fontsize=9, color='gray')
            ax.set_xlabel(f'Speed [kn]'); ax.set_ylabel(f'Power [kW]')
            ax.set_title('A2  Coloured by mean draft', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25)

            # A3: coloured by apparent wind speed
            ax = axes[0, 2]
            if aws is not None:
                _aws_s = aws[_mask].values[_idx]
                _valid_aws = np.isfinite(_aws_s) & (_aws_s >= 0)
                sc = ax.scatter(
                    _sp_s[_valid_aws], _pw_s[_valid_aws],
                    c=_aws_s[_valid_aws], cmap='YlOrRd',
                    vmin=0,
                    vmax=np.nanpercentile(_aws_s[_valid_aws], 95),
                    s=3, alpha=0.4, rasterized=True,
                )
                plt.colorbar(sc, ax=ax, label=f'{AWS} [kn]', fraction=0.04)
            else:
                ax.scatter(_sp_s, _pw_s, s=3, alpha=0.25, c='steelblue', rasterized=True)
                ax.text(0.5, 0.98, 'Wind column not available',
                        ha='center', va='top', transform=ax.transAxes,
                        fontsize=9, color='gray')
            ax.set_xlabel(f'Speed [kn]'); ax.set_ylabel(f'Power [kW]')
            ax.set_title('A3  Coloured by apparent wind speed', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25)

            # A4: hexbin density
            ax = axes[1, 0]
            if n_pts > 50:
                hb = ax.hexbin(_sp, _pw, gridsize=60, cmap='Blues',
                               mincnt=1, bins='log', rasterized=True)
                plt.colorbar(hb, ax=ax, label='log10(count)', fraction=0.04)
            ax.set_xlabel(f'Speed [kn]'); ax.set_ylabel(f'Power [kW]')
            ax.set_title('A4  Hexbin density (log scale)', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25)

            # A5: monthly scatter
            ax = axes[1, 1]
            if ts_utc is not None:
                _months = ts_utc[_mask].dt.strftime('%Y-%m')
                _unique_m = sorted(_months[np.isin(np.arange(len(_months)), _idx)].dropna().unique())
                _cmap_m = plt.cm.get_cmap('tab20', len(_unique_m))
                for _mi, _m in enumerate(_unique_m):
                    _mk = (_months.values[_idx] == _m)
                    ax.scatter(_sp_s[_mk], _pw_s[_mk], s=3, alpha=0.35,
                               color=_cmap_m(_mi), label=_m, rasterized=True)
                if len(_unique_m) <= 14:
                    ax.legend(fontsize=7, markerscale=3, loc='upper left',
                              ncol=2, framealpha=0.7)
            else:
                ax.scatter(_sp_s, _pw_s, s=3, alpha=0.25, c='steelblue', rasterized=True)
            ax.set_xlabel(f'Speed [kn]'); ax.set_ylabel(f'Power [kW]')
            ax.set_title('A5  Monthly colour split', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25)

            # A6: voyage direction proxy — latitude diff sign (N vs S leg)
            #     fall back to scatter with text if no position data
            ax = axes[1, 2]
            _leg_done = False
            if LAT is not None and ts_utc is not None:
                _lat = pd.to_numeric(df[LAT], errors='coerce')
                _lat_ok = _lat[_mask]
                _dlat = _lat_ok.diff().fillna(0)
                _northing = (_dlat >= 0)
                _n_sp = _sp[_northing.values]; _n_pw = _pw[_northing.values]
                _s_sp = _sp[~_northing.values]; _s_pw = _pw[~_northing.values]
                if len(_n_sp) > 10 and len(_s_sp) > 10:
                    _cap = 5_000
                    _rng = np.random.default_rng(7)
                    def _sample(a, b):
                        n = min(len(a), _cap)
                        i = _rng.choice(len(a), n, replace=False)
                        return a[i], b[i]
                    _ns, _np2 = _sample(_n_sp, _n_pw)
                    _ss, _sp2 = _sample(_s_sp, _s_pw)
                    ax.scatter(_ns, _np2, s=3, alpha=0.3, c='royalblue',
                               label='Northing leg', rasterized=True)
                    ax.scatter(_ss, _sp2, s=3, alpha=0.3, c='tomato',
                               label='Southing leg', rasterized=True)
                    ax.legend(fontsize=8, markerscale=3)
                    _leg_done = True
            if not _leg_done:
                ax.scatter(_sp_s, _pw_s, s=3, alpha=0.25, c='steelblue', rasterized=True)
                ax.text(0.5, 0.98, 'Position data needed for leg split',
                        ha='center', va='top', transform=ax.transAxes,
                        fontsize=9, color='gray')
            ax.set_xlabel(f'Speed [kn]'); ax.set_ylabel(f'Power [kW]')
            ax.set_title('A6  Voyage leg direction (N vs S)', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25)

            fig.text(0.5, 0.0,
                     'Expected: smooth P ∝ V³ curve.  '
                     'Anomalies: high power at low speed (fouling/shallow?), '
                     'clear month-to-month shift, separate clusters.',
                     ha='center', fontsize=8, color='dimgray', style='italic')
            fig.tight_layout(rect=[0, 0.025, 1, 1])
            _fpath = biv_dir / 'A_power_vs_speed.png'
            fig.savefig(_fpath, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"  [EDA/Bivariate] A — Power vs Speed         → {_fpath.name}")

            # Stats summary
            _at_rest  = int((spd[_mask] < 0.5).sum())
            _hi_pwr_lo_spd = int(((pwr[_mask] > pwr[_mask].quantile(0.75)) &
                                   (spd[_mask] < spd[_mask].quantile(0.25))).sum())
            report_rows.append({
                'section': 'A_power_vs_speed',
                'n_valid': int(_mask.sum()),
                'n_at_rest_speed': _at_rest,
                'n_high_power_low_speed': _hi_pwr_lo_spd,
                'speed_p50': round(float(np.percentile(_sp, 50)), 3),
                'power_p50': round(float(np.percentile(_pw, 50)), 3),
                'notes': 'high_power_low_speed = P>Q75 & V<Q25',
            })

        # ══════════════════════════════════════════════════════════════════════
        # B. Fuel flow vs Power
        # ══════════════════════════════════════════════════════════════════════
        if POWER and FUEL:
            _mask_b = pwr.notna() & fuel.notna() & (pwr >= 0) & (fuel >= 0)
            _fw = fuel[_mask_b].values
            _pw2 = pwr[_mask_b].values

            # Linear trend on full data
            _lfit_ok = len(_pw2) > 50
            if _lfit_ok:
                try:
                    from numpy.polynomial import polynomial as _P
                    _b_coeffs = np.polyfit(_pw2, _fw, 1)
                    _px2 = np.linspace(0, _pw2.max(), 200)
                    _fy = np.polyval(_b_coeffs, _px2)
                    _resid = _fw - np.polyval(_b_coeffs, _pw2)
                except Exception:
                    _lfit_ok = False

            # Monthly slopes
            _monthly_slopes: dict = {}
            if ts_utc is not None and _lfit_ok:
                _ts_b  = ts_utc[_mask_b]
                _months_b = _ts_b.dt.strftime('%Y-%m')
                for _m in sorted(_months_b.dropna().unique()):
                    _mk = (_months_b == _m).values
                    if _mk.sum() < 50:
                        continue
                    try:
                        _c = np.polyfit(_pw2[_mk], _fw[_mk], 1)
                        _monthly_slopes[_m] = round(float(_c[0]), 6)
                    except Exception:
                        pass

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(
                f'Bivariate EDA  B — Fuel Flow vs Power\n{vessel_name}',
                fontsize=13, fontweight='bold',
            )

            # B1: scatter + linear trend
            ax = axes[0, 0]
            BSMAX = 12_000
            _bidx = (
                np.random.default_rng(42).choice(len(_pw2), BSMAX, replace=False)
                if len(_pw2) > BSMAX else np.arange(len(_pw2))
            )
            ax.scatter(_pw2[_bidx], _fw[_bidx], s=3, alpha=0.25,
                       c='seagreen', rasterized=True)
            if _lfit_ok:
                ax.plot(_px2, _fy, lw=2, color='crimson',
                        label=f'Linear fit  slope={_b_coeffs[0]:.4f}')
                ax.legend(fontsize=8)
            ax.set_xlabel(f'Power ({POWER}) [kW]')
            ax.set_ylabel(f'Fuel flow ({FUEL})')
            ax.set_title('B1  Scatter + linear fit', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25)

            # B2: rolling average overlay
            ax = axes[0, 1]
            if ts_utc is not None:
                _ts_b = ts_utc[_mask_b].reset_index(drop=True)
                _df_b = pd.DataFrame({'ts': _ts_b, 'pwr': _pw2, 'fuel': _fw}).sort_values('ts')
                _df_b['roll_fuel'] = _df_b['fuel'].rolling(window=200, min_periods=20,
                                                            center=True).mean()
                _df_b['roll_pwr']  = _df_b['pwr'].rolling(window=200,  min_periods=20,
                                                           center=True).mean()
                _rn = min(4_000, len(_df_b))
                _ri = np.random.default_rng(1).choice(len(_df_b), _rn, replace=False)
                ax.scatter(_df_b['pwr'].values[_ri], _df_b['fuel'].values[_ri],
                           s=3, alpha=0.2, c='steelblue', rasterized=True)
                ax.scatter(_df_b['roll_pwr'], _df_b['roll_fuel'],
                           s=6, alpha=0.6, c='crimson', rasterized=True, label='200-pt rolling mean')
                ax.legend(fontsize=8, markerscale=2)
            else:
                ax.scatter(_pw2[_bidx], _fw[_bidx], s=3, alpha=0.25,
                           c='steelblue', rasterized=True)
                ax.text(0.5, 0.5, 'Timestamp needed for rolling avg',
                        ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_xlabel('Power [kW]'); ax.set_ylabel('Fuel flow')
            ax.set_title('B2  Rolling average overlay', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25)

            # B3: monthly slope comparison
            ax = axes[1, 0]
            if _monthly_slopes:
                _ms_vals = np.array(list(_monthly_slopes.values()))
                _ms_labels = list(_monthly_slopes.keys())
                _bar_colors = ['tomato' if v > np.median(_ms_vals) * 1.15 else
                               'royalblue' for v in _ms_vals]
                ax.bar(_ms_labels, _ms_vals, color=_bar_colors, edgecolor='white')
                ax.axhline(np.median(_ms_vals), ls='--', color='black', lw=1.2,
                           label=f'Median = {np.median(_ms_vals):.4f}')
                ax.set_xlabel('Month'); ax.set_ylabel('Fuel/Power slope (kg/h per kW)')
                ax.legend(fontsize=8)
                ax.tick_params(axis='x', rotation=40)
            else:
                ax.text(0.5, 0.5, 'Not enough monthly data',
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('B3  Monthly efficiency slope', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25, axis='y')

            # B4: residual plot
            ax = axes[1, 1]
            if _lfit_ok:
                _resid_p99 = np.percentile(np.abs(_resid), 99)
                _r_clipped = np.clip(_resid, -_resid_p99, _resid_p99)
                ax.scatter(_pw2[_bidx], _r_clipped[_bidx],
                           s=3, alpha=0.3, c='goldenrod', rasterized=True)
                ax.axhline(0, color='black', lw=1.2, ls='--')
                ax.axhline(np.percentile(_resid,  95), color='tomato', lw=1, ls=':',
                           label='p5 / p95 residual')
                ax.axhline(np.percentile(_resid,   5), color='tomato', lw=1, ls=':')
                ax.legend(fontsize=8)
                ax.set_xlabel('Power [kW]'); ax.set_ylabel('Fuel residual (actual − predicted)')
            else:
                ax.text(0.5, 0.5, 'Linear fit failed — residuals unavailable',
                        ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title('B4  Residuals after linear fit', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25)

            fig.text(0.5, 0.0,
                     'Expected: tight positive linear relationship.  '
                     'Anomalies: high residuals, step-change in monthly slope, clusters off trend.',
                     ha='center', fontsize=8, color='dimgray', style='italic')
            fig.tight_layout(rect=[0, 0.025, 1, 1])
            _fpath = biv_dir / 'B_fuel_vs_power.png'
            fig.savefig(_fpath, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"  [EDA/Bivariate] B — Fuel vs Power          → {_fpath.name}")

            report_rows.append({
                'section': 'B_fuel_vs_power',
                'n_valid': int(_mask_b.sum()),
                'linear_slope': round(float(_b_coeffs[0]), 6) if _lfit_ok else None,
                'residual_p95': round(float(np.percentile(_resid, 95)), 4) if _lfit_ok else None,
                'n_monthly_slopes': len(_monthly_slopes),
                'notes': 'slope = kg/(h·kW) or native fuel unit / kW',
            })

        # ══════════════════════════════════════════════════════════════════════
        # C. Draft vs Power / Fuel / Speed-conditioned power
        # ══════════════════════════════════════════════════════════════════════
        if DRAFT and POWER:
            _mask_c = dft.notna() & pwr.notna() & (dft > 0)
            _dc  = dft[_mask_c].values
            _pwc = pwr[_mask_c].values

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(
                f'Bivariate EDA  C — Draft vs Power / Fuel\n{vessel_name}',
                fontsize=13, fontweight='bold',
            )
            CSMAX = 10_000

            # C1: Draft vs Power
            ax = axes[0, 0]
            _ci = (np.random.default_rng(3).choice(len(_dc), min(CSMAX, len(_dc)),
                                                    replace=False))
            ax.scatter(_dc[_ci], _pwc[_ci], s=4, alpha=0.35, c='steelblue', rasterized=True)
            # Draft quartile bands
            for _q, _qc in [(0.25, 'gold'), (0.50, 'orange'), (0.75, 'tomato')]:
                _qv = np.quantile(_dc, _q)
                ax.axvline(_qv, ls='--', lw=1, color=_qc, label=f'Q{int(_q*100)}={_qv:.2f}m')
            ax.legend(fontsize=8)
            ax.set_xlabel(f'{DRAFT} [m]'); ax.set_ylabel(f'Power [kW]')
            ax.set_title('C1  Draft vs Power', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25)

            # C2: Draft vs Fuel
            ax = axes[0, 1]
            if FUEL and fuel is not None:
                _mask_cf = dft.notna() & fuel.notna() & (dft > 0) & (fuel >= 0)
                _dc2 = dft[_mask_cf].values
                _fc2 = fuel[_mask_cf].values
                _ci2 = np.random.default_rng(4).choice(
                    len(_dc2), min(CSMAX, len(_dc2)), replace=False)
                ax.scatter(_dc2[_ci2], _fc2[_ci2], s=4, alpha=0.35,
                           c='seagreen', rasterized=True)
                ax.set_ylabel(f'Fuel flow ({FUEL})')
            else:
                ax.text(0.5, 0.5, 'Fuel column not available',
                        ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_xlabel(f'{DRAFT} [m]')
            ax.set_title('C2  Draft vs Fuel flow', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25)

            # C3: Speed-conditioned power by draft quartile
            ax = axes[1, 0]
            if spd is not None:
                _mask_cs = dft.notna() & pwr.notna() & spd.notna() & (pwr >= 0) & (spd >= 0)
                _dq_bins = np.quantile(dft[_mask_cs].dropna(),
                                       [0, 0.25, 0.50, 0.75, 1.0])
                _dq_labels = [
                    f'Q1 draft [{_dq_bins[0]:.1f}–{_dq_bins[1]:.1f}m]',
                    f'Q2 draft [{_dq_bins[1]:.1f}–{_dq_bins[2]:.1f}m]',
                    f'Q3 draft [{_dq_bins[2]:.1f}–{_dq_bins[3]:.1f}m]',
                    f'Q4 draft [{_dq_bins[3]:.1f}–{_dq_bins[4]:.1f}m]',
                ]
                _dq_colors = ['royalblue', 'seagreen', 'darkorange', 'crimson']
                for _qi in range(4):
                    _qmask = (
                        _mask_cs &
                        (dft >= _dq_bins[_qi]) &
                        (dft <  (_dq_bins[_qi + 1] + 0.01))
                    )
                    _n = int(_qmask.sum())
                    if _n < 20:
                        continue
                    _qsp = spd[_qmask].values
                    _qpw = pwr[_qmask].values
                    _qi_idx = np.random.default_rng(_qi).choice(
                        _n, min(3_000, _n), replace=False)
                    ax.scatter(_qsp[_qi_idx], _qpw[_qi_idx], s=3, alpha=0.3,
                               c=_dq_colors[_qi], label=_dq_labels[_qi], rasterized=True)
                ax.set_xlabel('Speed [kn]'); ax.set_ylabel('Power [kW]')
                ax.legend(fontsize=7, markerscale=3)
            else:
                ax.text(0.5, 0.5, 'Speed column not available',
                        ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title('C3  Speed vs Power by draft quartile', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25)

            # C4: Trim vs Power
            ax = axes[1, 1]
            if trim is not None:
                _mask_ct = trim.notna() & pwr.notna()
                _tc = trim[_mask_ct].values
                _pct = pwr[_mask_ct].values
                _cti = np.random.default_rng(5).choice(
                    len(_tc), min(CSMAX, len(_tc)), replace=False)
                sc = ax.scatter(
                    _tc[_cti], _pct[_cti],
                    c=_tc[_cti], cmap='RdBu_r',
                    vmin=np.percentile(_tc, 2), vmax=np.percentile(_tc, 98),
                    s=4, alpha=0.4, rasterized=True,
                )
                plt.colorbar(sc, ax=ax, label=f'{TRIM} [m]', fraction=0.04)
                ax.axvline(0, ls='--', lw=1, color='black', alpha=0.5)
                ax.set_xlabel(f'Trim ({TRIM}) [m]')
                ax.set_ylabel('Power [kW]')
            else:
                ax.text(0.5, 0.5, 'Trim column not available',
                        ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title('C4  Trim vs Power (pos = stern-heavy)', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25)

            fig.text(0.5, 0.0,
                     'Expected: higher draft → higher power at same speed.  '
                     'Anomalies: no draft effect (sensor suspect), '
                     'extreme trim driving fuel penalty.',
                     ha='center', fontsize=8, color='dimgray', style='italic')
            fig.tight_layout(rect=[0, 0.025, 1, 1])
            _fpath = biv_dir / 'C_draft_vs_power.png'
            fig.savefig(_fpath, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"  [EDA/Bivariate] C — Draft vs Power/Fuel    → {_fpath.name}")

            report_rows.append({
                'section': 'C_draft_vs_power',
                'n_valid': int(_mask_c.sum()),
                'draft_p25': round(float(np.quantile(_dc, 0.25)), 3),
                'draft_p75': round(float(np.quantile(_dc, 0.75)), 3),
                'notes': 'Q4 draft band expected to show higher power curve',
            })

        # ══════════════════════════════════════════════════════════════════════
        # D. Wind vs Power / Fuel
        # ══════════════════════════════════════════════════════════════════════
        if AWS and POWER:
            _mask_d = aws.notna() & pwr.notna() & (aws >= 0) & (pwr >= 0)
            # Also require speed > 1 kn to filter port conditions
            if spd is not None:
                _mask_d = _mask_d & (spd > 1.0)
            _aw = aws[_mask_d].values
            _pwd = pwr[_mask_d].values

            # Residual from cubic speed-power trend (remove speed effect)
            _pwr_resid: Optional[np.ndarray] = None
            if spd is not None:
                _mask_dr = _mask_d & spd.notna() & (spd >= 0)
                _sdr = spd[_mask_dr].values
                _pdr = pwr[_mask_dr].values
                _adr = aws[_mask_dr].values
                try:
                    _dr_coeffs = np.polyfit(_sdr, _pdr, 3)
                    _pwr_resid = _pdr - np.polyval(_dr_coeffs, _sdr)
                    _aw_for_resid = _adr
                except Exception:
                    pass

            # Fuel residual
            _fuel_resid: Optional[np.ndarray] = None
            _aw_for_fuel_resid: Optional[np.ndarray] = None
            if FUEL and fuel is not None and spd is not None:
                _mask_dfr = (fuel.notna() & spd.notna() & aws.notna() &
                             (fuel >= 0) & (spd > 1.0))
                _sdfr = spd[_mask_dfr].values
                _fdfr = fuel[_mask_dfr].values
                _adfr = aws[_mask_dfr].values
                if len(_sdfr) > 50:
                    try:
                        _dfr_coeffs = np.polyfit(_sdfr, _fdfr, 3)
                        _fuel_resid = _fdfr - np.polyval(_dfr_coeffs, _sdfr)
                        _aw_for_fuel_resid = _adfr
                    except Exception:
                        pass

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(
                f'Bivariate EDA  D — Wind vs Power / Fuel\n{vessel_name}',
                fontsize=13, fontweight='bold',
            )
            DSMAX = 10_000

            # D1: AWS vs Power residual (speed removed)
            ax = axes[0, 0]
            if _pwr_resid is not None:
                _ri = np.random.default_rng(6).choice(
                    len(_pwr_resid), min(DSMAX, len(_pwr_resid)), replace=False)
                _pr99 = np.percentile(np.abs(_pwr_resid), 99)
                _pr_clipped = np.clip(_pwr_resid, -_pr99, _pr99)
                ax.scatter(_aw_for_resid[_ri], _pr_clipped[_ri],
                           s=3, alpha=0.3, c='steelblue', rasterized=True)
                ax.axhline(0, lw=1.2, ls='--', color='black')
                # rolling median
                _df_d = (pd.DataFrame({'aws': _aw_for_resid, 'resid': _pr_clipped})
                         .sort_values('aws'))
                _rm = _df_d['resid'].rolling(400, min_periods=30, center=True).median()
                ax.plot(_df_d['aws'].values, _rm.values, lw=2, color='crimson',
                        label='Rolling median')
                ax.legend(fontsize=8)
            else:
                _ri = np.random.default_rng(6).choice(
                    len(_aw), min(DSMAX, len(_aw)), replace=False)
                ax.scatter(_aw[_ri], _pwd[_ri], s=3, alpha=0.3, c='steelblue', rasterized=True)
                ax.text(0.5, 0.98, 'Speed col unavailable — showing raw power',
                        ha='center', va='top', transform=ax.transAxes, fontsize=9)
            ax.set_xlabel(f'Apparent wind speed ({AWS}) [kn]')
            ax.set_ylabel('Power residual [kW]  (speed trend removed)')
            ax.set_title('D1  AWS vs speed-detrended Power residual', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25)

            # D2: AWS vs Fuel residual (speed removed)
            ax = axes[0, 1]
            if _fuel_resid is not None:
                _fri = np.random.default_rng(7).choice(
                    len(_fuel_resid), min(DSMAX, len(_fuel_resid)), replace=False)
                _fr99 = np.percentile(np.abs(_fuel_resid), 99)
                _fr_clipped = np.clip(_fuel_resid, -_fr99, _fr99)
                ax.scatter(_aw_for_fuel_resid[_fri], _fr_clipped[_fri],
                           s=3, alpha=0.3, c='seagreen', rasterized=True)
                ax.axhline(0, lw=1.2, ls='--', color='black')
                _df_d2 = (pd.DataFrame({'aws': _aw_for_fuel_resid, 'resid': _fr_clipped})
                          .sort_values('aws'))
                _rm2 = _df_d2['resid'].rolling(400, min_periods=30, center=True).median()
                ax.plot(_df_d2['aws'].values, _rm2.values, lw=2, color='crimson',
                        label='Rolling median')
                ax.legend(fontsize=8)
                ax.set_ylabel('Fuel residual  (speed trend removed)')
            else:
                ax.text(0.5, 0.5, 'Fuel or speed column unavailable',
                        ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_xlabel(f'Apparent wind speed [kn]')
            ax.set_title('D2  AWS vs speed-detrended Fuel residual', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25)

            # D3: AWA sector boxplot of fuel / power residual
            ax = axes[1, 0]
            if AWA and awa is not None and _pwr_resid is not None:
                _awa_d = awa[_mask_dr].values if 'awa' in dir() else awa[_mask_d].values
                # normalise to [0, 360)
                _awa_norm = np.mod(_awa_d, 360)
                _sector_edges = np.arange(0, 361, 45)
                _sector_labels = [f'{int(a)}–{int(b)}°'
                                  for a, b in zip(_sector_edges[:-1], _sector_edges[1:])]
                _sector_data = []
                for _j in range(len(_sector_labels)):
                    _sm = (_awa_norm >= _sector_edges[_j]) & (_awa_norm < _sector_edges[_j + 1])
                    _sector_data.append(_pwr_resid[_sm] if _sm.sum() > 0 else np.array([]))
                _bx = ax.boxplot(
                    [d for d in _sector_data],
                    labels=_sector_labels,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color='crimson', lw=2),
                )
                _colors = plt.cm.hsv(np.linspace(0, 0.9, len(_sector_labels)))
                for _patch, _clr in zip(_bx['boxes'], _colors):
                    _patch.set_facecolor(_clr)
                    _patch.set_alpha(0.6)
                ax.axhline(0, lw=1.2, ls='--', color='black')
                ax.tick_params(axis='x', rotation=35)
                ax.set_ylabel('Power residual [kW]')
            else:
                ax.text(0.5, 0.5, 'AWA column not available',
                        ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_xlabel('AWA sector')
            ax.set_title('D3  AWA sector vs Power residual (45° bins)', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25, axis='y')

            # D4: Wind rose (AWS by AWA direction) — polar
            ax_d4 = fig.add_subplot(2, 2, 4, polar=True)
            axes[1, 1].set_visible(False)   # hide the flat axis we won't use
            if AWA and awa is not None:
                _mask_wr = aws.notna() & awa.notna() & (aws >= 0)
                _awa_wr = np.deg2rad(np.mod(awa[_mask_wr].values, 360))
                _aws_wr = aws[_mask_wr].values
                if len(_awa_wr) > 50:
                    _n_bins_wr = 36
                    _edges_wr = np.linspace(0, 2 * np.pi, _n_bins_wr + 1)
                    _bin_idx = np.digitize(_awa_wr, _edges_wr) - 1
                    _bin_idx = np.clip(_bin_idx, 0, _n_bins_wr - 1)
                    _mean_aws = np.array([
                        _aws_wr[_bin_idx == _b].mean() if (_bin_idx == _b).any() else 0
                        for _b in range(_n_bins_wr)
                    ])
                    _theta = (_edges_wr[:-1] + _edges_wr[1:]) / 2
                    _width = (2 * np.pi) / _n_bins_wr
                    _bars = ax_d4.bar(_theta, _mean_aws, width=_width * 0.9,
                                      bottom=0, alpha=0.7, edgecolor='white',
                                      color=plt.cm.YlOrRd(_mean_aws / (_mean_aws.max() + 1e-9)))
                    ax_d4.set_theta_zero_location('N')
                    ax_d4.set_theta_direction(-1)
                    ax_d4.set_title('D4  Wind rose\n(mean AWS by AWA sector)',
                                    fontsize=10, fontweight='bold', pad=12)
            else:
                ax_d4.text(0, 0, 'AWA/AWS not available', ha='center', fontsize=9)

            fig.text(0.5, 0.0,
                     'Expected: positive power/fuel residual with stronger headwinds (AWA ≈ 0–45°, 315–360°).  '
                     'Anomalies: no wind effect (sensor suspect), huge penalties in weak wind sectors.',
                     ha='center', fontsize=8, color='dimgray', style='italic')
            fig.tight_layout(rect=[0, 0.025, 1, 1])
            _fpath = biv_dir / 'D_wind_vs_power.png'
            fig.savefig(_fpath, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"  [EDA/Bivariate] D — Wind vs Power/Fuel     → {_fpath.name}")

            report_rows.append({
                'section': 'D_wind_vs_power',
                'n_valid': int(_mask_d.sum()),
                'aws_p50': round(float(np.median(_aw)), 3) if len(_aw) > 0 else None,
                'aws_p95': round(float(np.percentile(_aw, 95)), 3) if len(_aw) > 0 else None,
                'notes': 'residuals detrended against cubic speed-power fit',
            })

        # ── Save summary CSV ──────────────────────────────────────────────────
        if report_rows:
            _rdir     = vessel_dir if vessel_dir is not None else plots_dir.parent
            _rep_path = _rdir / 'bivariate_report.csv'
            pd.DataFrame(report_rows).to_csv(_rep_path, index=False)
            print(f"  [EDA/Bivariate] Summary → {_rep_path}")

        return str(biv_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 7 — Variable Metadata (data dictionary + EDA findings)
    # ══════════════════════════════════════════════════════════════════════════

    def _variable_metadata(
        self,
        df: pd.DataFrame,
        vessel_dir: Path,
    ) -> Path:
        """
        Phase 7 — Variable metadata: one row per signal combining domain
        knowledge and EDA observations.

        Sections per row
        ----------------
        1. Variable definition  — name, label, unit, raw_source_col,
                                   signal_type, is_derived
        2. Expected behaviour   — physical_lo/hi, normal_op_lo/hi,
                                   dynamics (stable/dynamic), negative_ok,
                                   domain_notes
        3. EDA findings         — min/max, pct_missing, n_duplicates,
                                   n_frozen_runs (≥5 repeats), n_spikes (RoC>p99),
                                   bimodal_flag, seasonal_flag,
                                   suspicious_unit_flag
        4. Data-quality rules   — hard_filter, soft_warning,
                                   interpolation_allowed,
                                   remove_repeated_values,
                                   zero_means_idle

        Output
        ------
        {vessel_dir}/{vessel_id}_metadata.csv
        """
        cfg      = getattr(self, '_vessel_cfg', None)
        date_col = self._date_col(df)

        # Reverse map: std_col → raw_col  (from vessel_config column_map)
        raw_col_map: dict = {}
        if cfg and cfg.column_map:
            raw_col_map = {v: k for k, v in cfg.column_map.items()}

        # Derived column names
        derived_names: set = set(cfg.derived_cols.keys()) if cfg else set()

        # Columns to document — ALL numeric columns in dataset column order.
        _EXCLUDE = {'source_file', 'Time Sec Midnight (sec)'}
        if date_col:
            _EXCLUDE.add(date_col)

        target_cols = [
            c for c in df.select_dtypes(include='number').columns
            if c not in _EXCLUDE
        ]

        # Timestamp series (UTC) for temporal checks
        ts_utc: Optional[pd.Series] = None
        if date_col:
            ts_utc = pd.to_datetime(df[date_col], errors='coerce', utc=True)

        # ── Helpers ───────────────────────────────────────────────────────────
        FROZEN_MIN_SECONDS = 1200  # runs of identical values spanning ≥ 20 min are frozen
        SPIKE_RoC_MULT    = 3.0    # RoC > 3×p99 of RoC distribution = spike

        def _rle_frozen(s: pd.Series) -> int:
            """Count data points in runs of identical values spanning ≥ 20 minutes."""
            if len(s) < 2:
                return 0
            _rounded = np.round(s.values, 4)
            _change  = np.concatenate([[1], (_rounded[1:] != _rounded[:-1]).astype(int)])
            _run_ids = np.cumsum(_change) - 1
            if ts_utc is not None:
                _ts = ts_utc.reindex(s.index)
                total = 0
                for run_id in np.unique(_run_ids):
                    mask = _run_ids == run_id
                    run_ts = _ts.iloc[np.where(mask)[0]].dropna()
                    if len(run_ts) >= 2:
                        duration = (run_ts.iloc[-1] - run_ts.iloc[0]).total_seconds()
                        if duration >= FROZEN_MIN_SECONDS:
                            total += int(mask.sum())
                return total
            else:
                # Fallback when no timestamps: flag runs of ≥ 12 identical readings
                _, _counts = np.unique(_run_ids, return_counts=True)
                return int(_counts[_counts >= 12].sum())

        def _n_spikes(valid: pd.Series, roc_p99: float) -> int:
            """Count RoC spikes exceeding SPIKE_RoC_MULT × roc_p99."""
            if roc_p99 <= 0:
                return 0
            threshold = SPIKE_RoC_MULT * roc_p99
            if ts_utc is not None:
                _ts = ts_utc.reindex(valid.index)
                _dt = _ts.diff().dt.total_seconds()
                _dv = valid.diff()
                _ok = (_dt > 0) & (_dt <= 300)
                roc = (_dv[_ok] / _dt[_ok]).abs().dropna()
            else:
                roc = valid.diff().abs().dropna()
            return int((roc > threshold).sum())

        def _is_bimodal(valid: pd.Series) -> bool:
            """Simple bimodality check: fraction of values near zero AND near median."""
            if len(valid) < 100:
                return False
            med = float(valid.median())
            if med < 0.5:
                return False
            frac_zero = float((valid < med * 0.05).mean())
            frac_high = float((valid > med * 0.5).mean())
            return frac_zero > 0.10 and frac_high > 0.20

        def _has_seasonal(valid: pd.Series) -> bool:
            """Flag if monthly means differ by > 20% of overall mean (seasonal effect)."""
            if ts_utc is None or len(valid) < 500:
                return False
            _months = ts_utc.reindex(valid.index).dt.to_period('M')
            _gp = valid.groupby(_months).mean().dropna()
            if len(_gp) < 2:
                return False
            overall = float(valid.mean())
            if overall == 0:
                return False
            return float((_gp.max() - _gp.min()) / abs(overall)) > 0.20

        # ── Signal roles: per-column description, vessel relation, port notes ──
        _SIGNAL_ROLES: dict = {
            # AE emergency fuel circuit
            'AE Emerg DO Mode ()': {
                'description': 'Operating mode or status of the auxiliary engine emergency diesel-oil system.',
                'vessel_relation': 'Indicates which fuel mode the emergency AE circuit is currently running in; used for fault diagnostics.',
                'port_notes': 'Mode value may change during port operations when emergency generator is tested.',
            },
            'AE Emerg Mass In (kg/hr)': {
                'description': 'Fuel mass flow into the emergency auxiliary-engine diesel-oil circuit; how much fuel is being supplied into that line/system.',
                'vessel_relation': 'Measures fuel supply rate to the emergency AE circuit; used to detect fuel leaks or abnormal consumption.',
                'port_notes': 'Near-zero when emergency AE is not running.',
            },
            'AE Emerg Mass Net (kg/hr)': {
                'description': 'Net fuel consumption in the emergency auxiliary-engine diesel-oil circuit.',
                'vessel_relation': 'Difference between fuel in and fuel returned; represents actual fuel burned by the emergency AE.',
                'port_notes': 'Near-zero when emergency AE is not running.',
            },
            'AE Emerg Mass Ret (kg/hr)': {
                'description': 'Fuel mass flow returning from the emergency auxiliary-engine diesel-oil circuit.',
                'vessel_relation': 'Return line flow; together with mass-in, used to calculate net consumption and detect circuit issues.',
                'port_notes': 'Near-zero when emergency AE is not running.',
            },
            'GPSSpeed_kn': {
                'description': 'Vessel speed over ground measured by GPS, in knots.',
                'vessel_relation': 'Primary indicator of vessel motion state; drives power demand and efficiency assessment.',
                'port_notes': 'Values < 0.5 kn indicate port or anchorage. Bimodal distribution: at-rest and cruising peaks.',
            },
            'SpeedLog_kn': {
                'description': 'Vessel speed through water measured by hull logs, in knots.',
                'vessel_relation': 'Used with GPS speed to estimate ocean current; relevant for speed-power efficiency.',
                'port_notes': 'Near-zero at berth; may differ from GPS speed in strong currents.',
            },
            'GPS_LAT': {
                'description': 'GPS latitude of vessel position, in decimal degrees.',
                'vessel_relation': 'Route tracking; distinguishes sea passages, port approaches, and berth positions.',
                'port_notes': 'Stable (frozen) when vessel is at berth.',
            },
            'GPS_LON': {
                'description': 'GPS longitude of vessel position, in decimal degrees.',
                'vessel_relation': 'Route tracking; correlated with latitude to reconstruct voyage legs.',
                'port_notes': 'Stable (frozen) when vessel is at berth.',
            },
            'Ship_Course_deg': {
                'description': 'True course of the vessel, in degrees (0–360).',
                'vessel_relation': 'Determines relative wind angle; used for wind resistance correction.',
                'port_notes': 'Arbitrary at berth — vessel may swing on mooring lines.',
            },
            'Main_Engine_Power_kW': {
                'description': 'Main engine shaft power output in kilowatts.',
                'vessel_relation': 'Core propulsion KPI; directly determines fuel consumption and vessel speed.',
                'port_notes': 'Should be near zero at berth. Negative values are sensor anomalies.',
            },
            'Speed_rpm': {
                'description': 'Main engine / propeller shaft rotational speed in RPM.',
                'vessel_relation': 'Tightly coupled to shaft power and vessel speed; used for slip calculations.',
                'port_notes': 'Near-zero at berth; low non-zero values indicate maneuvering.',
            },
            'ME_Shaft_Torque_kNm': {
                'description': 'Shaft torque at main engine output flange in kNm.',
                'vessel_relation': 'Together with RPM determines shaft power (P = τ × ω); used for load monitoring.',
                'port_notes': 'Should be near zero at berth.',
            },
            'ME_Shaft_Thrust_kN': {
                'description': 'Propeller thrust force in kilonewtons.',
                'vessel_relation': 'Direct measure of propulsive force; useful for bollard pull and resistance analysis.',
                'port_notes': 'Near-zero at berth.',
            },
            'Fuel_Consumption_rate': {
                'description': 'Main engine net fuel consumption rate in kg/hr.',
                'vessel_relation': 'Key efficiency KPI; used with shaft power to calculate specific fuel oil consumption (SFOC).',
                'port_notes': 'Low but non-zero at berth (hotel/auxiliary load). True zero possible at cold stop.',
            },
            'Fuel_Consumption_t_per_day': {
                'description': 'Total vessel fuel consumption rate in metric tonnes per day.',
                'vessel_relation': 'Voyage-level efficiency metric; used for CII and EEOI regulatory calculations.',
                'port_notes': 'Low at berth (auxiliary engines only).',
            },
            'DRAFTAFT': {
                'description': 'Aft (stern) draft depth in metres; measures how deep the stern sits in water.',
                'vessel_relation': 'Determines vessel displacement and trim; affects resistance and propeller immersion depth.',
                'port_notes': 'Changes slowly during cargo loading/unloading. Stable during voyages.',
            },
            'DRAFTFWD': {
                'description': 'Forward (bow) draft depth in metres; measures how deep the bow sits in water.',
                'vessel_relation': 'Paired with aft draft to calculate trim; influences bow wave and hull resistance.',
                'port_notes': 'Changes slowly during cargo operations. Sentinel value −9999 indicates sensor fault.',
            },
            'Draft_Mid_Port_m': {
                'description': 'Mid-ship port-side draft in metres.',
                'vessel_relation': 'Used with other draft readings for accurate displacement calculation.',
                'port_notes': 'Stable at sea; changes during cargo operations in port.',
            },
            'Draft_Mid_Stbd_m': {
                'description': 'Mid-ship starboard-side draft in metres.',
                'vessel_relation': 'Used with port draft to detect list (athwartship tilt).',
                'port_notes': 'Stable at sea; changes during cargo operations in port.',
            },
            'Avg_draft_m': {
                'description': 'Average of aft and forward draft in metres; proxy for vessel displacement.',
                'vessel_relation': 'Derived loading indicator; used to cluster voyages by loading condition.',
                'port_notes': 'Non-positive values indicate draft sensor fault (sentinel −9999 inherited from DRAFTFWD).',
            },
            'Trim_m': {
                'description': 'Vessel trim = aft draft minus forward draft, in metres. Positive = stern-heavy.',
                'vessel_relation': 'Affects hull resistance; optimum trim can reduce fuel consumption by up to 2%.',
                'port_notes': 'Values outside ±5 m are physical anomalies.',
            },
            'Depth_of_Water_m': {
                'description': 'Water depth below keel in metres.',
                'vessel_relation': 'Shallow-water squat effect increases resistance when depth < 3× draft.',
                'port_notes': 'Low values expected during port approach and harbour manoeuvring.',
            },
            'RelWindSpeed_kn': {
                'description': 'Apparent (relative) wind speed measured on board in knots.',
                'vessel_relation': 'Used to estimate added wind resistance; component of power prediction models.',
                'port_notes': 'Can be non-zero at berth (= true wind when stationary). Very high values may indicate sensor issues.',
            },
            'RelWindAngle_deg': {
                'description': 'Apparent wind direction relative to vessel bow, in degrees (0–360).',
                'vessel_relation': 'Determines direction component of wind force on hull and superstructure.',
                'port_notes': 'Equals true wind angle when vessel is stationary at berth.',
            },
            'TrueWindSpeed_kn': {
                'description': 'True (meteorological) wind speed derived from apparent wind and vessel motion, in knots.',
                'vessel_relation': 'Used for weather routing and ISO 19030 resistance correction.',
                'port_notes': 'Should equal apparent wind speed when vessel is stationary at berth.',
            },
        }

        def _auto_describe(col_name: str, prof: dict) -> str:
            """Generate a fallback description for columns not in _SIGNAL_ROLES."""
            n = col_name
            if n.startswith('AE '):
                return f'Auxiliary emergency engine signal: {n}.'
            if n.startswith('Blr ') or n.startswith('Cmp Blr '):
                return f'Boiler fuel/flow signal: {n}.'
            if n.startswith('GE') and '_Power_kW' in n:
                idx = n.replace('GE', '').replace('_Power_kW', '')
                return f'Generator engine {idx} electrical output in kW.'
            if n.startswith('Fuel ') and any(k in n for k in ('Grav', 'LHV', 'Sulphur')):
                return f'Fuel quality property (density/LHV/sulphur content): {n}.'
            if 'Temp' in n or 'Â°C' in n:
                return f'Temperature measurement: {n}.'
            if 'Rudder' in n:
                return 'Rudder angle in degrees; indicates steering activity and course-keeping effort.'
            if 'Cool' in n and 'SW' in n:
                return f'Sea water cooling system temperature: {n}.'
            label = prof.get('label', col_name)
            expect = prof.get('expect', '')
            return label + ('.' if not label.endswith('.') else '') + (' ' + expect if expect else '')

        rows: list = []

        for col in target_cols:
            if col not in df.columns:
                continue

            raw    = pd.to_numeric(
                df[col].replace(r'^\s*$', np.nan, regex=True),
                errors='coerce',
            )
            valid  = raw.dropna()
            n_tot  = len(raw)
            n_val  = len(valid)
            n_miss = n_tot - n_val

            if n_val < 10:
                continue

            # ── 1. Signal profile lookup ──────────────────────────────────────
            profile = UNIVARIATE_SIGNAL_PROFILES.get(col)
            if profile is None:
                for _std, _p in UNIVARIATE_SIGNAL_PROFILES.items():
                    if col in _p.get('alias', []):
                        profile = _p
                        break
            if profile is None:
                profile = {
                    'label': col, 'unit': '',
                    'phys_lo': None, 'phys_hi': None,
                    'negative_is_anomaly': False, 'alias': [], 'expect': ''
                }

            phys_lo = profile.get('phys_lo')
            phys_hi = profile.get('phys_hi')
            neg_anomaly = bool(profile.get('negative_is_anomaly', False))

            # ── 2. Normal operating range (Q10–Q90 when moving, else overall) ─
            _speed_col = next((c for c in ('GPSSpeed_kn', 'SOG') if c in df.columns), None)
            if _speed_col and col not in ('GPSSpeed_kn', 'SOG'):
                _moving = pd.to_numeric(df[_speed_col], errors='coerce') > 0.5
                _valid_moving = raw[_moving].dropna()
                _ref = _valid_moving if len(_valid_moving) > 100 else valid
            else:
                _ref = valid
            normal_lo = round(float(_ref.quantile(0.10)), 4)
            normal_hi = round(float(_ref.quantile(0.90)), 4)

            # ── 3. Dynamics — stable if RoC median very small ─────────────────
            if ts_utc is not None and n_val > 1:
                _ts_v   = ts_utc.reindex(valid.index)
                _dt_s   = _ts_v.diff().dt.total_seconds()
                _dv     = valid.diff()
                _ok     = (_dt_s > 0) & (_dt_s <= 300)
                _roc_s  = (_dv[_ok] / _dt_s[_ok]).abs().dropna()
                roc_p50 = float(_roc_s.median()) if len(_roc_s) > 0 else 0.0
                roc_p99 = float(_roc_s.quantile(0.99)) if len(_roc_s) > 0 else 0.0
            else:
                _roc_s  = valid.diff().abs().dropna()
                roc_p50 = float(_roc_s.median())
                roc_p99 = float(_roc_s.quantile(0.99))

            # Dynamics label
            _value_range = float(valid.max() - valid.min())
            _roc_relative = roc_p50 / (_value_range + 1e-12)
            dynamics = 'stable' if _roc_relative < 1e-4 else 'dynamic'

            # ── 4. EDA findings ───────────────────────────────────────────────
            # Duplicates (identical consecutive values)
            n_dup_consec = int((valid.diff() == 0).sum())

            # Frozen periods
            n_frozen = _rle_frozen(valid)

            # Spikes
            n_spikes = _n_spikes(valid, roc_p99)

            # Physical violations
            n_phys_viol = 0
            if phys_lo is not None:
                n_phys_viol += int((valid < phys_lo).sum())
            if phys_hi is not None:
                n_phys_viol += int((valid > phys_hi).sum())

            # Suspicious unit flag: e.g. if unit says 'm' but values suggest mm
            susp_unit = False
            if profile.get('unit') == 'm' and float(valid.max()) > 100:
                susp_unit = True
            if profile.get('unit') == 'kn' and float(valid.max()) > 80:
                susp_unit = True

            bimodal   = _is_bimodal(valid)
            seasonal  = _has_seasonal(valid)

            # ── 5. Data-quality rules ─────────────────────────────────────────
            #  Hard filter: definite removal
            hard_parts = []
            if neg_anomaly:
                hard_parts.append('remove_negative')
            if phys_hi is not None:
                hard_parts.append(f'remove_gt_{phys_hi}')
            if phys_lo is not None and phys_lo != 0:
                hard_parts.append(f'remove_lt_{phys_lo}')
            hard_filter = '; '.join(hard_parts) if hard_parts else 'none'

            #  Soft warning: flag but keep
            soft_parts = []
            if n_phys_viol > 0:
                soft_parts.append(f'{n_phys_viol}_phys_violations')
            if n_spikes > 0:
                soft_parts.append(f'{n_spikes}_rate_spikes')
            if n_frozen > 0:
                soft_parts.append(f'frozen_periods_detected')
            soft_warning = '; '.join(soft_parts) if soft_parts else 'none'

            #  Interpolation: OK for slow-varying signals with no long gaps
            # Also check longest consecutive gap in seconds
            if ts_utc is not None and n_miss > 0:
                _ts_all = ts_utc.reindex(raw.index)
                _gap_s  = _ts_all.diff().dt.total_seconds()
                _missing_mask = raw.isna()
                # Longest run of consecutive missing rows, converted to seconds
                _gap_runs = _gap_s[_missing_mask]
                max_gap_seconds = float(_gap_runs.sum()) if len(_gap_runs) > 0 else 0.0
                # More precisely: find max continuous gap span
                _ts_valid = _ts_all.dropna()
                if len(_ts_valid) > 1:
                    _all_gaps = _ts_all.diff().dt.total_seconds().dropna()
                    max_gap_seconds = float(_all_gaps.max())
                else:
                    max_gap_seconds = 0.0
            else:
                max_gap_seconds = 0.0
            interp_ok = (
                (dynamics == 'stable')
                and (n_miss / n_tot < 0.05)
                and (max_gap_seconds <= 600)    # no single gap > 10 min
            )

            #  Remove repeated values: any freeze ≥ 20 min is actionable
            remove_repeats = n_frozen > 0

            #  Zero = idle: for power/fuel/speed
            zero_idle = col in ('Main_Engine_Power_kW', 'Fuel_Consumption_rate',
                                 'GPSSpeed_kn', 'Speed_rpm',
                                 'Fuel_Consumption_t_per_day')

            # ── 6. Assemble row ───────────────────────────────────────────────
            role        = _SIGNAL_ROLES.get(col, {})
            description = role.get('description') or _auto_describe(col, profile)
            port_notes  = role.get('port_notes', '')

            if phys_lo is not None and phys_hi is not None:
                phys_limits_str = f"{phys_lo} – {phys_hi}"
            elif phys_lo is not None:
                phys_limits_str = f">= {phys_lo}"
            elif phys_hi is not None:
                phys_limits_str = f"<= {phys_hi}"
            else:
                phys_limits_str = ''

            norm_range_str = f"Q10={normal_lo},  Q90={normal_hi}"

            rows.append({
                'variable_name':                col,
                'feature_group':                _group_for_col(col, cfg),
                'raw_column_name':              raw_col_map.get(col, col),
                'unit':                         profile.get('unit', ''),
                'column_description':           description,
                'physical_limits':              phys_limits_str,
                'normal_operating_range':       norm_range_str,
                'stable_or_highly_dynamic':     dynamics,
                'observed_min':                 round(float(valid.min()), 4),
                'observed_max':                 round(float(valid.max()), 4),
                'pct_missing':                  round(n_miss / n_tot * 100, 3),
                'n_missing':                    n_miss,
                'n_consecutive_duplicates':     n_dup_consec,
                'n_spikes':                     n_spikes,
                'n_frozen_periods':             n_frozen,
                'suspicious_unit_flag':         susp_unit,
                'seasonal_effects':             seasonal,
                'hard_filter':                  hard_filter,
                'soft_warning':                 soft_warning,
                'interpolation_allowed':        interp_ok,
                'remove_repeated_values':       remove_repeats,
                'treat_low_values_as_idle':     zero_idle,
                'notes':                        port_notes,
            })

        out_path = vessel_dir / f'{cfg.vessel_id}_metadata.csv' if cfg else vessel_dir / 'variable_metadata.csv'
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  [EDA/Metadata] {len(rows)} variables → {out_path}")
        return out_path

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 0 — Data Audit methods
    # ══════════════════════════════════════════════════════════════════════════

    def _time_axis_audit(self, df: pd.DataFrame, out_dir: Path) -> dict:
        """
        A. Time-axis checks  (expected sampling period: 14 s).

        Six checks are performed
        ────────────────────────
        1. Sort order           — are records in chronological order?
        2. Duplicate timestamps — same timestamp appearing for ≥ 2 rows.
        3. Missing timestamps   — 14 s time slots with no data (skipped beats).
        4. Irregular gaps       — intervals that are NOT clean multiples of 14 s.
        5. Long gaps            — continuous silence  > 5 min.
        6. Repeated blocks      — data after a long gap mirrors the pre-gap block,
                                  suggesting a logger cache replay.

        CSV outputs (written to *out_dir*)
        ───────────────────────────────────
        time_axis_report.csv          scalar summary of every check
        time_axis_duplicates.csv      every timestamp appearing > once
        time_axis_gaps.csv            catalogue of long gaps (> 5 min)
        time_axis_missing_by_day.csv  daily record count, missed beats, coverage %

        PNG outputs
        ───────────
        time_axis_interval_hist.png      histogram of Δt between records
        time_axis_gap_timeline.png       monthly bar chart: n gaps + gap minutes
                                         + daily coverage trend
        time_axis_coverage_heatmap.png   calendar heatmap of daily coverage %
        """
        # Vessel-specific sampling period (default 14 s for Stena vessels, 120 s for Grimaldi)
        _vcfg      = getattr(self, '_vessel_cfg', None)
        EXPECTED   = getattr(_vcfg, 'expected_interval_s', 14)
        GAP_THRESH = 10 * EXPECTED    # long gap = 10 × expected interval

        date_col = self._date_col(df)
        if date_col is None:
            print("  [Audit/Time] WARNING: no timestamp column — skipping")
            return {'skipped': True, 'reason': 'no timestamp column'}

        # ── 0. Parse timestamps ───────────────────────────────────────────────
        ts_raw        = pd.to_datetime(df[date_col], errors='coerce', utc=True)
        n_unparseable = int(ts_raw.isna().sum())
        ts_valid      = ts_raw.dropna()
        n_total       = len(ts_valid)

        if n_total < 2:
            return {'skipped': True, 'reason': f'only {n_total} valid timestamps'}

        # ── 1. Sort check ─────────────────────────────────────────────────────
        # Are the rows already in chronological order?
        # Out-of-order rows indicate files were concatenated in the wrong order,
        # or the logger had a clock jump / rollback.
        raw_diff       = ts_valid.sort_index().diff().dt.total_seconds().dropna()
        n_out_of_order = int((raw_diff < 0).sum())
        is_sorted      = (n_out_of_order == 0)

        # Work on a time-sorted, reset-index copy from here.
        ts = ts_valid.sort_values().reset_index(drop=True)

        # ── 2. Duplicate timestamps ───────────────────────────────────────────
        # A duplicate = the exact same timestamp on two or more rows.
        # Root causes: logger double-write, overlapping file export, merge error.
        vc         = ts.value_counts()
        dup_vc     = vc[vc > 1]
        n_dup_ts   = int(len(dup_vc))           # unique timestamps with duplicates
        n_dup_rows = int((dup_vc - 1).sum())    # extra rows beyond first occurrence
        pct_dup    = round(n_dup_rows / n_total * 100, 3)

        if n_dup_rows > 0:
            dup_table          = dup_vc.rename('n_occurrences').reset_index()
            dup_table.columns  = ['timestamp', 'n_occurrences']
            dup_table['month'] = pd.DatetimeIndex(dup_table['timestamp']).strftime('%Y-%m')
            (dup_table
             .sort_values('n_occurrences', ascending=False)
             .to_csv(out_dir / 'time_axis_duplicates.csv', index=False))

        # De-duplicate: keep first occurrence only.
        ts_u = ts.drop_duplicates().sort_values().reset_index(drop=True)
        n_u  = len(ts_u)

        # ── 3. Inter-sample intervals ─────────────────────────────────────────
        # deltas has integer index 1 … n_u-1 (matching positions in ts_u).
        deltas          = ts_u.diff().dt.total_seconds().dropna()
        median_delta    = float(deltas.median())
        _tol = max(2, int(EXPECTED * 0.05))   # ±5% of expected interval, min ±2 s
        pct_at_expected = float(
            deltas.between(EXPECTED - _tol, EXPECTED + _tol).mean() * 100
        )

        # ── 4. Missing timestamps (skipped beats) ─────────────────────────────
        # A perfectly regular 14 s logger produces Δt = 14 s every step.
        # Δt = 28 s → 1 beat missed; Δt = 42 s → 2 missed; etc.
        # n_missed_for_pair = max(0, round(Δt / EXPECTED) − 1)
        beats_arr      = np.maximum(
            0, np.round(deltas.values / EXPECTED).astype(int) - 1
        )
        n_missed       = int(beats_arr.sum())
        n_expected_tot = n_u + n_missed
        pct_coverage   = round(n_u / n_expected_tot * 100, 2) if n_expected_tot else 100.0

        # Per-day breakdown: records seen, beats missed, coverage %.
        start_pos  = np.array(deltas.index, dtype=int) - 1      # positions of pair-start
        pair_days  = ts_u.iloc[start_pos].dt.strftime('%Y-%m-%d').values
        missed_by_day = (
            pd.Series(beats_arr, index=pair_days)
            .groupby(level=0).sum()
            .rename('missed_beats')
        )
        rec_by_day = (
            ts_u.dt.strftime('%Y-%m-%d').value_counts().sort_index().rename('n_records')
        )
        daily = (
            pd.concat([rec_by_day, missed_by_day], axis=1)
            .fillna(0)
            .astype({'n_records': int, 'missed_beats': int})
        )
        daily['n_expected']   = daily['n_records'] + daily['missed_beats']
        daily['pct_coverage'] = (
            daily['n_records'] / daily['n_expected'].clip(lower=1) * 100
        ).round(2)
        daily.index.name = 'date'
        daily.to_csv(out_dir / 'time_axis_missing_by_day.csv')

        # ── 5. Irregular gaps (short, non-multiples of EXPECTED) ─────────────
        # Expected: Δt ∈ {EXPECTED, 2×EXPECTED, …} (clean multiples, within ±_tol).
        # Irregular: Δt is ≥ 2 s AND < GAP_THRESH, but NOT a clean multiple.
        # Possible causes: sub-second jitter, logger clock drift, mixed-period
        # sensor merges, or time-zone shift during logging.
        short_d     = deltas[(deltas >= 2) & (deltas < GAP_THRESH)]
        remainder   = short_d % EXPECTED
        min_rem     = np.minimum(remainder, EXPECTED - remainder)
        n_irregular = int((min_rem > _tol).sum())

        # ── 6. Long-gap catalogue (> 5 min) ───────────────────────────────────
        # A "long gap" is an interval > GAP_THRESH seconds.
        # These represent sensor outages, file export windows, port stays, etc.
        long_idxs = deltas.index[deltas > GAP_THRESH]
        gap_rows  = []
        for i_end in long_idxs:
            t0  = ts_u.iloc[i_end - 1]
            t1  = ts_u.iloc[i_end]
            dur = (t1 - t0).total_seconds()
            gap_rows.append({
                'gap_start':    str(t0),
                'gap_end':      str(t1),
                'duration_min': round(dur / 60,   2),
                'duration_h':   round(dur / 3600, 3),
                'missed_beats': max(0, round(dur / EXPECTED) - 1),
                'month':        t0.strftime('%Y-%m'),
                'day':          t0.strftime('%Y-%m-%d'),
            })
        gap_df      = pd.DataFrame(gap_rows)
        gap_df.to_csv(out_dir / 'time_axis_gaps.csv', index=False)
        n_long_gaps = len(gap_df)
        total_gap_h = float(gap_df['duration_h'].sum()) if not gap_df.empty else 0.0

        # ── 7. Repeated-block detection ───────────────────────────────────────
        # After a long gap, compare the N rows just before and just after.
        # If any numeric column is constant across both windows AND the constant
        # value is the same → the logger replayed cached data ("stuck" sensor).
        n_check           = 10
        n_repeated_blocks = 0
        num_cols          = self._numeric_cols(df)[:5]
        if num_cols and not gap_df.empty:
            df_s        = df.copy()
            df_s['_ts'] = ts_raw
            df_s        = (df_s
                           .dropna(subset=['_ts'])
                           .sort_values('_ts')
                           .reset_index(drop=True))
            for _, grow in gap_df.iterrows():
                t_end = pd.to_datetime(grow['gap_end'], utc=True)
                pos   = df_s['_ts'].searchsorted(t_end)
                pre   = df_s[num_cols].iloc[max(0, pos - n_check): pos]
                post  = df_s[num_cols].iloc[pos: pos + n_check]
                if len(pre) < 3 or len(post) < 3:
                    continue
                for col in num_cols:
                    if (pre[col].nunique() == 1
                            and post[col].nunique() == 1
                            and pre[col].iloc[0] == post[col].iloc[0]):
                        n_repeated_blocks += 1
                        break

        # ── Scalar summary ────────────────────────────────────────────────────
        summary = {
            'n_total_timestamps':     n_total,
            'n_unparseable':          n_unparseable,
            'n_unique_timestamps':    n_u,
            'n_duplicate_timestamps': n_dup_ts,
            'n_duplicate_rows':       n_dup_rows,
            'pct_duplicates':         pct_dup,
            'is_sorted':              is_sorted,
            'n_out_of_order':         n_out_of_order,
            'median_interval_sec':    round(median_delta, 1),
            'pct_at_expected':        round(pct_at_expected, 2),
            'n_missed_beats':         n_missed,
            'pct_coverage':           pct_coverage,
            'n_irregular_gaps':       n_irregular,
            'n_long_gaps':            n_long_gaps,
            'total_gap_hours':        round(total_gap_h, 2),
            'n_repeated_blocks':      n_repeated_blocks,
        }
        pd.DataFrame([summary]).T.rename(columns={0: 'value'}).to_csv(
            out_dir / 'time_axis_report.csv'
        )

        # ══════════════════════════════════════════════════════════════════════
        # PLOT 1 — Inter-sample interval histogram
        # ══════════════════════════════════════════════════════════════════════
        #
        # HOW TO READ
        # ───────────
        # X-axis = time in seconds between consecutive records.
        # Y-axis = how many consecutive pairs had that gap.
        #
        # IDEAL DATA
        #   Left panel  → one very tall spike at 14 s, almost nothing else.
        #   Right panel → green-shaded zone sits right on the spike.
        #
        # ALARM signals
        #   Bars near 0 s          → duplicate/near-duplicate timestamps.
        #   Bars at 28, 42, 56 s   → missed beats (1, 2, 3 skipped samples).
        #   Long right tail        → extended periods with no data at all.
        #   Flat / spread          → irregular logger clock or merge issue.
        _fig1, _ax1s = plt.subplots(1, 2, figsize=(15, 5))
        _fig1.suptitle(
            f'Time-axis ― inter-sample interval distribution\n'
            f'{n_u:,} unique records  |  '
            f'median Δt = {median_delta:.1f} s  |  '
            f'expected = {EXPECTED} s  |  '
            f'overall coverage = {pct_coverage:.1f}%',
            fontsize=12, fontweight='bold',
        )

        # Left panel: full range, log-scale y
        _ax = _ax1s[0]
        _ax.hist(deltas.clip(upper=600), bins=200,
                 color='steelblue', edgecolor='none', alpha=0.85)
        _ax.axvline(EXPECTED, color='red', lw=2, ls='--',
                    label=f'Expected {EXPECTED} s')
        _ax.axvspan(0,   2,   alpha=0.15, color='crimson',
                    label='< 2 s  (likely duplicates)')
        _ax.axvspan(60,  600, alpha=0.08, color='darkorange',
                    label='> 1 min  (long gap)')
        _ax.set_xlabel('Δt between records (s)  [clipped at 600 s]', fontsize=9)
        _ax.set_ylabel('Count  (log scale)', fontsize=9)
        _ax.set_yscale('log')
        _ax.set_title('All intervals — full range', fontsize=10, fontweight='bold')
        _ax.legend(fontsize=8, loc='upper right')
        _ax.grid(True, alpha=0.3)
        _ax.text(
            0.98, 0.97,
            f'Duplicate rows        : {n_dup_rows:,}\n'
            f'At expected ± 2 s     : {pct_at_expected:.1f}%\n'
            f'Irregular gaps (short): {n_irregular:,}\n'
            f'Long gaps  (> 5 min)  : {n_long_gaps:,}\n'
            f'Missed beats          : {n_missed:,}',
            transform=_ax.transAxes, fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9),
        )

        # Right panel: zoom 0–120 s; tick marks at every EXPECTED-multiple
        _ax2 = _ax1s[1]
        _zoom = deltas[deltas <= 120]
        _ax2.hist(_zoom, bins=120, color='steelblue', edgecolor='none', alpha=0.85)
        _ax2.axvline(EXPECTED, color='red', lw=2, ls='--',
                     label=f'Expected {EXPECTED} s')
        # green band = expected ± 2 s
        _ax2.axvspan(EXPECTED - 2, EXPECTED + 2, alpha=0.18, color='limegreen',
                     label=f'Expected ± 2 s  ({pct_at_expected:.0f}% of intervals)')
        _ax2.axvspan(0, 2, alpha=0.15, color='crimson')
        # dotted lines at 2×, 3×, … EXPECTED to mark "missed beats"
        for _m in range(2, 9):
            _ax2.axvline(_m * EXPECTED, color='gray', lw=0.5, ls=':',
                         label=f'{_m}× {EXPECTED} s' if _m == 2 else '_')
        _ax2.set_xlabel('Δt (s)   [zoomed: 0 – 120 s]', fontsize=9)
        _ax2.set_ylabel('Count', fontsize=9)
        _ax2.set_title(
            'Zoomed view (0 – 120 s)\n'
            'Dotted lines mark 28, 42, 56 … s  (1, 2, 3 missed beats)',
            fontsize=9, fontweight='bold',
        )
        _ax2.legend(fontsize=7, loc='upper right')
        _ax2.grid(True, alpha=0.3)
        if len(_zoom) > 0:
            _peak = int(_zoom.round().value_counts().idxmax())
            _ax2.text(
                0.01, 0.97,
                f'Most common interval: {_peak} s',
                transform=_ax2.transAxes, fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.9),
            )

        plt.tight_layout()
        _fig1.savefig(out_dir / 'time_axis_interval_hist.png', dpi=150,
                      bbox_inches='tight')
        plt.close(_fig1)

        # ══════════════════════════════════════════════════════════════════════
        # PLOT 2 — Monthly gap + coverage timeline
        # ══════════════════════════════════════════════════════════════════════
        #
        # HOW TO READ
        # ───────────
        # Every calendar month between the first and last record is shown.
        # TOP panel    → number of separate gaps > 5 min that month.
        # MIDDLE panel → total minutes of data missing that month.
        # BOTTOM panel → average daily coverage % (100% = no missed beats).
        #
        # Months with zero bars in the top two panels had uninterrupted logging.
        # A low bottom-panel line indicates systematic missed beats (logger
        # clock drift, network drops, or file export windows).

        # Build a complete monthly range so all months appear even with 0 gaps.
        _all_months = [
            str(m) for m in pd.period_range(
                ts_u.iloc[0].to_period('M'),
                ts_u.iloc[-1].to_period('M'),
                freq='M',
            )
        ]
        _n_mon = len(_all_months)

        if not gap_df.empty:
            _gap_mon = (
                gap_df.groupby('month')['duration_min']
                .agg(n_gaps='count', total_gap_min='sum')
                .reindex(_all_months, fill_value=0)
            )
        else:
            _gap_mon = pd.DataFrame(
                {'n_gaps': 0, 'total_gap_min': 0}, index=_all_months
            )

        _daily_m              = daily.copy()
        _daily_m.index        = pd.to_datetime(_daily_m.index)
        _daily_m['month']     = _daily_m.index.strftime('%Y-%m')
        _cov_mon = (
            _daily_m.groupby('month')['pct_coverage']
            .mean().round(2)
            .reindex(_all_months, fill_value=0)
        )

        _fig2, _axm = plt.subplots(
            3, 1,
            figsize=(max(10, _n_mon * 1.1 + 2), 11),
            sharex=True,
        )
        _fig2.suptitle(
            f'Monthly time-axis quality summary\n'
            f'Dataset: {_all_months[0]}  →  {_all_months[-1]}  '
            f'({_n_mon} months shown)',
            fontsize=12, fontweight='bold',
        )

        _x = range(_n_mon)

        # Top: n long gaps per month
        _axm[0].bar(_x, _gap_mon['n_gaps'], color='tomato', width=0.7,
                    edgecolor='white', lw=0.5)
        for _xi, _v in zip(_x, _gap_mon['n_gaps']):
            if _v > 0:
                _axm[0].text(_xi, _v + 0.05, str(int(_v)),
                             ha='center', va='bottom', fontsize=8)
        _axm[0].set_ylabel('No. of gaps\n(> 5 min)', fontsize=9)
        _axm[0].set_title(
            'Long gaps per month   '
            '(0 bars = recording was continuous all month)',
            fontsize=9, color='dimgray',
        )
        _axm[0].grid(True, alpha=0.3, axis='y')
        _axm[0].set_ylim(bottom=0)

        # Middle: total gap minutes
        _axm[1].bar(_x, _gap_mon['total_gap_min'], color='darkorange', width=0.7,
                    edgecolor='white', lw=0.5)
        for _xi, _v in zip(_x, _gap_mon['total_gap_min']):
            if _v > 0:
                _axm[1].text(_xi, _v + 1, f'{int(_v)}',
                             ha='center', va='bottom', fontsize=8)
        _axm[1].set_ylabel('Total gap\n(minutes)', fontsize=9)
        _axm[1].set_title(
            'Total data loss per month  '
            '(tall bar = one or more long sensor outages)',
            fontsize=9, color='dimgray',
        )
        _axm[1].grid(True, alpha=0.3, axis='y')
        _axm[1].set_ylim(bottom=0)

        # Bottom: mean daily coverage % as a line chart
        _cov_vals = _cov_mon.values.astype(float)
        _axm[2].plot(_x, _cov_vals, color='steelblue', lw=2,
                     marker='o', markersize=5)
        _axm[2].fill_between(_x, _cov_vals, alpha=0.15, color='steelblue')
        _axm[2].axhline(100, color='green', lw=1, ls='--', alpha=0.5)
        for _xi, _v in zip(_x, _cov_vals):
            if _v < 99:
                _axm[2].text(_xi, _v - 2.5, f'{_v:.0f}%',
                             ha='center', va='top', fontsize=7, color='navy')
        _axm[2].set_ylabel('Avg daily\ncoverage (%)', fontsize=9)
        _axm[2].set_ylim(max(0, _cov_vals.min() - 8) if _cov_vals.size else 0, 106)
        _axm[2].set_title(
            'Average daily data coverage  '
            '(100% = every 14 s slot has a record;  '
            '< 100% = missed beats that day)',
            fontsize=9, color='dimgray',
        )
        _axm[2].grid(True, alpha=0.3)
        _axm[2].set_xticks(_x)
        _axm[2].set_xticklabels(_all_months, rotation=35, ha='right', fontsize=8)

        _fig2.text(
            0.01, 0.003,
            f'Total long gaps: {n_long_gaps}  |  '
            f'Total data loss: {total_gap_h:.1f} h  |  '
            f'Overall coverage: {pct_coverage:.1f}%  |  '
            f'Total missed beats: {n_missed:,}',
            fontsize=8, color='dimgray',
        )
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        _fig2.savefig(out_dir / 'time_axis_gap_timeline.png', dpi=150,
                      bbox_inches='tight')
        plt.close(_fig2)

        # ══════════════════════════════════════════════════════════════════════
        # PLOT 3 — Daily coverage calendar heatmap
        # ══════════════════════════════════════════════════════════════════════
        #
        # HOW TO READ
        # ───────────
        # Each cell = one calendar day.
        # Rows = months (oldest at top), columns = day of month (1–31).
        # Colour = % of expected 14 s slots that had a record that day.
        #   GREEN  (100%)   → perfect, no missed beats.
        #   YELLOW (80–99%) → minor data loss.
        #   RED    (< 50%)  → serious data loss that day.
        #   Grey            → that day does not exist in the month (e.g. Feb 30).
        if len(daily) > 1:
            _cal          = daily.copy()
            _cal.index    = pd.to_datetime(_cal.index)
            _cal['month'] = _cal.index.strftime('%Y-%m')
            _cal['day']   = _cal.index.day
            _pivot = _cal.pivot_table(
                index='month', columns='day', values='pct_coverage'
            ).reindex(columns=range(1, 32))

            _fig3, _ax3 = plt.subplots(
                figsize=(max(12, 32 * 0.38),
                         max(4, len(_pivot) * 0.6 + 1.8))
            )
            sns.heatmap(
                _pivot, ax=_ax3,
                cmap='RdYlGn',
                vmin=0, vmax=100,
                linewidths=0.4, linecolor='white',
                cbar_kws={'label': '% coverage (100 = no missed beats)'},
                annot=(len(_pivot) <= 8),
                fmt='.0f',
            )
            _ax3.set_title(
                'Daily data coverage calendar\n'
                'GREEN = complete (100%)   YELLOW = some gaps   '
                'RED = heavy data loss   Grey = day does not exist',
                fontsize=11, fontweight='bold',
            )
            _ax3.set_xlabel('Day of month', fontsize=9)
            _ax3.set_ylabel('Month', fontsize=9)
            _ax3.tick_params(axis='x', labelsize=8)
            _ax3.tick_params(axis='y', labelsize=9, rotation=0)
            plt.tight_layout()
            _fig3.savefig(out_dir / 'time_axis_coverage_heatmap.png', dpi=150,
                          bbox_inches='tight')
            plt.close(_fig3)

        return summary

    # ──────────────────────────────────────────────────────────────────────────

    def _missingness_audit(self, df: pd.DataFrame, out_dir: Path) -> dict:
        """
        B. Missingness and data quality map.

        Checks performed
        ----------------
        1.  Per-column % missing (overall)  → missingness_overall.csv
        2a. % missing by day                → missingness_by_day.csv
        2b. % missing by week               → missingness_by_week.csv
        2c. % missing by month              → missingness_by_month.csv
        3.  Missingness matrix plot         → missingness_matrix.png
            Binary heatmap: x = daily bins, y = sensor.
            Tells you at a glance when each sensor was silent.
        4.  Daily % missing line chart      → missingness_daily_lines.png
            Shows continuous vs step-change patterns per sensor.
        5.  Co-missingness (Jaccard)        → missingness_comissing.png
            Which sensors fail together?
        6.  Simultaneous-failure table      → missingness_simultaneous.csv
            Rows where ≥ 3 priority sensors are all NaN at the same time
            (signature of a logger / system outage, not individual sensor fault).
        7.  Block-outage catalogue          → missingness_blocks.csv
            Every contiguous NaN run ≥ 2 rows, classified as:
              isolated  : 2–3 rows  (≈ 28–42 s) – random packet loss
              short     : 4–200 rows (< ~47 min) – intermittent fault
              long      : > 200 rows              – sensor outage
        8.  Block summary                   → missingness_block_summary.csv
        """
        date_col = self._date_col(df)

        # ── Priority sensors (raw names used in Stenaline + standard names) ──
        PRIORITY = [
            'RelWindAngle_deg', 'RelWindSpeed_kn', 'TrueWindSpeed_kn',
            'DRAFTAFT', 'DRAFTFWD', 'Avg_draft_m',
            'Fuel_Consumption_rate', 'Fuel_Consumption_t_per_day',
            'Main_Engine_Power_kW', 'GPSSpeed_kn',
            # raw names in case normalisation hasn't been applied
            'AWA', 'AWS', 'DraftAftDynamic', 'DraftFwdDynamic',
            'FuelMassFlowMETotal', 'PropulsionPowerTotal', 'SOG',
        ]
        priority_present = [c for c in PRIORITY if c in df.columns]

        # ── 1. Overall % missing ──────────────────────────────────────────────
        total = len(df)
        # Treat empty strings and whitespace-only strings as missing in addition to NaN
        _df_miss = df.copy()
        _str_cols = _df_miss.select_dtypes(include='object').columns
        if len(_str_cols):
            _df_miss[_str_cols] = _df_miss[_str_cols].apply(
                lambda s: s.str.strip()
            ).replace('', np.nan)
        miss_overall = (
            _df_miss.isnull().sum()
            .rename('n_missing')
            .to_frame()
        )
        miss_overall['pct_missing'] = (
            miss_overall['n_missing'] / total * 100
        ).round(3)
        miss_overall = miss_overall.sort_values('pct_missing', ascending=False)
        miss_overall.to_csv(out_dir / 'missingness_overall.csv')

        miss_cols = miss_overall[miss_overall['n_missing'] > 0].index.tolist()

        # interest_cols: priority sensors first, then any other col with missingness
        interest_cols = list(dict.fromkeys(priority_present + miss_cols))[:24]

        # ── 2. Time-resolved missingness (day / week / month) ─────────────────
        by_day   = pd.DataFrame()
        by_week  = pd.DataFrame()
        by_month = pd.DataFrame()

        if date_col and interest_cols:
            ts_utc = pd.to_datetime(df[date_col], errors='coerce', utc=True)
            work = df[interest_cols].copy()
            work['_day']   = ts_utc.dt.floor('D').dt.strftime('%Y-%m-%d')
            work['_week']  = ts_utc.dt.to_period('W').astype(str)
            work['_month'] = ts_utc.dt.to_period('M').astype(str)

            def _pct_miss(grp_col):
                return (
                    work.groupby(grp_col)[interest_cols]
                    .apply(lambda g: g.isnull().mean() * 100)
                    .round(2)
                )

            by_day   = _pct_miss('_day')
            by_week  = _pct_miss('_week')
            by_month = _pct_miss('_month')

            by_day.to_csv(out_dir   / 'missingness_by_day.csv')
            by_week.to_csv(out_dir  / 'missingness_by_week.csv')
            by_month.to_csv(out_dir / 'missingness_by_month.csv')

        # ── 3. Missingness matrix plot (sensors × days) ───────────────────────
        # Each cell = % missing that day for that sensor.
        # Black = fully present, white/yellow = partially missing, red = 100% missing.
        if not by_day.empty and by_day.shape[0] > 1:
            mat = by_day[interest_cols].T    # sensors as rows, days as columns

            # Downsample columns for legibility if many days
            max_cols_shown = 120
            if mat.shape[1] > max_cols_shown:
                step = max(1, mat.shape[1] // max_cols_shown)
                mat = mat.iloc[:, ::step]

            fig_w = max(12, min(mat.shape[1] * 0.15, 28))
            fig_h = max(4,  len(interest_cols) * 0.42)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            sns.heatmap(
                mat, ax=ax,
                cmap='YlOrRd', vmin=0, vmax=100,
                linewidths=0,
                cbar_kws={'label': '% missing'},
                xticklabels=max(1, mat.shape[1] // 30),
            )
            ax.set_title(
                'Missingness matrix — sensors × days\n'
                '(yellow/red = data absent; white = fully present)',
                fontsize=12, fontweight='bold',
            )
            ax.set_xlabel('Date (daily bins)', fontsize=9)
            ax.set_ylabel('Sensor', fontsize=9)
            ax.tick_params(axis='x', labelsize=7, rotation=45)
            ax.tick_params(axis='y', labelsize=8)
            plt.tight_layout()
            fig.savefig(out_dir / 'missingness_matrix.png', dpi=150,
                        bbox_inches='tight')
            plt.close(fig)

        # ── 4. Daily % missing line chart (priority sensors only) ─────────────
        if not by_day.empty and priority_present:
            plot_cols = [c for c in priority_present if c in by_day.columns]
            if plot_cols:
                fig, ax = plt.subplots(
                    figsize=(max(14, by_day.shape[0] * 0.12), 5)
                )
                for col in plot_cols:
                    ax.plot(
                        range(len(by_day)), by_day[col].values,
                        lw=1.2, alpha=0.85, label=col,
                    )
                n_ticks = min(20, len(by_day))
                tick_idx = np.linspace(0, len(by_day) - 1, n_ticks, dtype=int)
                ax.set_xticks(tick_idx)
                ax.set_xticklabels(
                    [by_day.index[i] for i in tick_idx],
                    rotation=45, ha='right', fontsize=7,
                )
                ax.set_ylabel('% missing', fontsize=10)
                ax.set_title('Daily % missing — priority sensors',
                             fontsize=12, fontweight='bold')
                ax.legend(fontsize=8, ncol=2, loc='upper right')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                fig.savefig(out_dir / 'missingness_daily_lines.png', dpi=150,
                            bbox_inches='tight')
                plt.close(fig)

        # ── 5. Co-missingness matrix (Jaccard) ────────────────────────────────
        jaccard_df = pd.DataFrame()
        if len(interest_cols) >= 2:
            null_mat  = df[interest_cols].isnull().astype(np.int8).values
            n_c       = len(interest_cols)
            jaccard   = np.zeros((n_c, n_c))
            for i in range(n_c):
                for j in range(n_c):
                    a = null_mat[:, i]
                    b = null_mat[:, j]
                    inter = int((a & b).sum())
                    union = int((a | b).sum())
                    jaccard[i, j] = inter / union if union > 0 else 0.0
            jaccard_df = pd.DataFrame(jaccard, index=interest_cols,
                                       columns=interest_cols)

            fig, ax = plt.subplots(
                figsize=(max(6, n_c * 0.58), max(5, n_c * 0.52) + 2)
            )
            sns.heatmap(
                jaccard_df, ax=ax, cmap='Blues', vmin=0, vmax=1,
                annot=(n_c <= 15), fmt='.2f', linewidths=0.3,
                cbar_kws={'label': 'Jaccard similarity (0 = independent, 1 = always fail together)'},
            )
            ax.set_title(
                'Co-missingness between sensors — Jaccard similarity\n'
                'Each cell shows how often two sensors are BOTH missing at the same time.',
                fontsize=11, fontweight='bold',
            )
            # interpretation guide as a text box below the heatmap
            guide = (
                'HOW TO READ THIS PLOT\n'
                '─────────────────────────────────────────────────────────\n'
                '  Value = 0.0  (white)  → sensors miss data independently'
                                         ' — likely separate hardware faults\n'
                '  Value ≈ 0.5  (mid-blue) → sensors sometimes fail together'
                                         ' — may share a bus or logger\n'
                '  Value = 1.0  (dark blue) → sensors ALWAYS fail together'
                                         ' — almost certainly the same logger / system outage\n'
                '\n'
                '  Diagonal is always 1.0 (every sensor is perfectly co-missing with itself).\n'
                '  Focus on OFF-DIAGONAL dark cells — those pairs warrant investigation.'
            )
            fig.text(
                0.01, -0.01, guide,
                fontsize=7.5, color='dimgray', va='top',
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9',
                          edgecolor='lightgray', alpha=1.0),
            )
            plt.tight_layout()
            fig.savefig(out_dir / 'missingness_comissing.png', dpi=150,
                        bbox_inches='tight')
            plt.close(fig)

        # ── 6. Simultaneous failure (logger / system outage) ──────────────────
        # Rows where ≥ 3 priority sensors are all NaN at once.
        sim_thresh = 3
        if len(priority_present) >= sim_thresh:
            null_counts = df[priority_present].isnull().sum(axis=1)
            sim_rows    = df[null_counts >= sim_thresh].copy()
            sim_rows['n_sensors_missing'] = null_counts[sim_rows.index]
            if date_col:
                sim_rows['timestamp'] = pd.to_datetime(
                    df.loc[sim_rows.index, date_col], errors='coerce', utc=True
                )
            sim_rows = sim_rows[
                (['timestamp'] if date_col else [])
                + ['n_sensors_missing']
                + [c for c in priority_present if c in sim_rows.columns]
            ]
            sim_rows.to_csv(out_dir / 'missingness_simultaneous.csv', index=False)
            n_simultaneous = len(sim_rows)
        else:
            n_simultaneous = 0

        # ── 7 & 8. Block-outage catalogue with classification ─────────────────
        #   isolated  : 2–3 rows   (~28–42 s at 14 s sampling) — packet loss
        #   short     : 4–200 rows (< ~47 min)                 — intermittent fault
        #   long      : > 200 rows                              — sensor outage
        block_rows = []
        for col in interest_cols:
            s   = df[col].isnull().values
            idx = 0
            while idx < len(s):
                if s[idx]:
                    jdx = idx
                    while jdx < len(s) and s[jdx]:
                        jdx += 1
                    run_len = jdx - idx
                    if run_len >= 2:
                        start_ts = end_ts = duration_min = ''
                        if date_col:
                            t0 = pd.to_datetime(
                                df[date_col].iloc[idx], errors='coerce', utc=True)
                            t1 = pd.to_datetime(
                                df[date_col].iloc[jdx - 1], errors='coerce', utc=True)
                            start_ts    = str(t0)
                            end_ts      = str(t1)
                            try:
                                duration_min = round(
                                    (t1 - t0).total_seconds() / 60, 2)
                            except Exception:
                                duration_min = ''

                        if run_len <= 3:
                            gap_type = 'isolated'
                        elif run_len <= 200:
                            gap_type = 'short_outage'
                        else:
                            gap_type = 'long_outage'

                        block_rows.append({
                            'column':       col,
                            'gap_type':     gap_type,
                            'row_start':    idx,
                            'row_end':      jdx - 1,
                            'n_rows':       run_len,
                            'duration_min': duration_min,
                            'start_ts':     start_ts,
                            'end_ts':       end_ts,
                        })
                    idx = jdx
                else:
                    idx += 1

        block_df = pd.DataFrame(block_rows)
        block_df.to_csv(out_dir / 'missingness_blocks.csv', index=False)

        # Block summary: per-column × gap_type count
        if not block_df.empty:
            block_summary = (
                block_df.groupby(['column', 'gap_type'])
                .agg(n_events=('n_rows', 'count'),
                     total_rows=('n_rows', 'sum'))
                .reset_index()
            )
        else:
            block_summary = pd.DataFrame(
                columns=['column', 'gap_type', 'n_events', 'total_rows']
            )
        block_summary.to_csv(out_dir / 'missingness_block_summary.csv', index=False)

        # ── Console summary ───────────────────────────────────────────────────
        print(f"  [Audit/Miss] Columns with any missing : {len(miss_cols)}")
        for col in interest_cols[:12]:
            pct = float(miss_overall.loc[col, 'pct_missing']) \
                  if col in miss_overall.index else 0.0
            if pct > 0:
                sub = block_df[block_df['column'] == col] if not block_df.empty \
                      else pd.DataFrame()
                btype = ''
                if not sub.empty:
                    counts = sub['gap_type'].value_counts().to_dict()
                    btype = '  [' + ', '.join(
                        f"{v}× {k}" for k, v in counts.items()) + ']'
                print(f"  [Audit/Miss]   {col:<36s} {pct:6.2f}% missing{btype}")
        if n_simultaneous > 0:
            print(f"  [Audit/Miss] Simultaneous-failure rows  : {n_simultaneous:,} "
                  f"(≥{sim_thresh} priority sensors NaN at once "
                  f"→ likely logger/system outage)")

        result = {
            'n_cols_with_missing':    len(miss_cols),
            'most_missing_col':       miss_overall.index[0] if len(miss_overall) else 'none',
            'most_missing_pct':       float(miss_overall['pct_missing'].iloc[0])
                                      if len(miss_overall) else 0.0,
            'n_block_outages_total':  len(block_df),
            'n_isolated_gaps':        int((block_df['gap_type'] == 'isolated').sum())
                                      if not block_df.empty else 0,
            'n_short_outages':        int((block_df['gap_type'] == 'short_outage').sum())
                                      if not block_df.empty else 0,
            'n_long_outages':         int((block_df['gap_type'] == 'long_outage').sum())
                                      if not block_df.empty else 0,
            'n_simultaneous_failure_rows': n_simultaneous,
        }
        return result

    # ──────────────────────────────────────────────────────────────────────────

    def _unit_sanity_check(self, df: pd.DataFrame, out_dir: Path) -> list:
        """
        C. Physical / unit sanity checks.

        For each signal of interest the method checks whether the observed
        range is consistent with the expected physical range and flags
        likely unit mismatches.

        Checks made
        -----------
        LAT / GPS_LAT
          Expected range: 30–75 °N for North-Sea / Baltic routes.
          If |values| ≤ 2 → likely radians (convert: multiply by 180/π).

        LONG / GPS_LON
          Expected range: -10–30 °E for North-Sea / Baltic.
          If |values| ≤ 0.6 → likely radians.

        GPSSpeed_kn / SOG
          A North-Sea ferry has a service speed of ~21 kn.
          If 95th-pct ≤ 6 m/s (≈ 11.7 kn) → may still be knots at low speed,
          but if median ≈ 4.5, suspect m/s → multiply by 1.944.

        Main_Engine_Power_kW / PropulsionPowerTotal
          A large RoPax main engine is typically 20–40 MW.
          If 95th-pct < 100 → likely kW already at low load, acceptable.
          If median > 5000 and 99th-pct > 50 000 → suspect W (divide by 1000).

        Fuel_Consumption_rate
          At sea: ~2 000–8 000 kg/h for a large ferry.
          If 95th-pct < 10 → suspect t/h or t/day (multiply accordingly).
          If 95th-pct > 200 000 → suspect kg/s (multiply by 3600).

        Avg_draft_m / DraftAftDynamic
          Typical ferry draft: 4–8 m.
          If values are > 100 → likely mm (divide by 1000).

        Outputs
        -------
        - unit_sanity.csv — one row per checked signal with
          observed_min, p5, median, p95, observed_max,
          status (ok / WARNING / CRITICAL), and a human note.
        """
        checks = [
            # (std_col_name, label, lo_ok, hi_ok, warn_if_below, warn_if_above, unit_note)
            dict(
                col='GPS_LAT', label='Latitude',
                ok_lo=30.0, ok_hi=75.0,
                radian_hint=True,
                note_ok='Degrees — plausible North-Sea range',
                note_warn='Values outside expected 30–75 °N — check route or units',
                note_radian='Values look like radians; multiply by 180/π',
            ),
            dict(
                col='GPS_LON', label='Longitude',
                ok_lo=-15.0, ok_hi=35.0,
                radian_hint=True,
                note_ok='Degrees — plausible North-Sea / Baltic range',
                note_warn='Values outside expected -15–35 °E',
                note_radian='Values look like radians; multiply by 180/π',
            ),
            dict(
                col='GPSSpeed_kn', label='Speed over Ground',
                ok_lo=0.0, ok_hi=25.0,
                ms_hint=True,
                note_ok='Knots — plausible (0–25 kn)',
                note_warn='95th-pct > 25 kn is suspicious for a ferry',
                note_ms='Median < 6 and max < 15 — may be m/s; multiply by 1.944',
            ),
            dict(
                col='Main_Engine_Power_kW', label='Propulsion Power',
                ok_lo=0.0, ok_hi=45000.0,
                watt_hint=True,
                note_ok='kW — plausible (0–45 MW)',
                note_warn='Values outside 0–45 000 kW',
                note_watt='Values > 100 000 → likely W; divide by 1 000 for kW',
            ),
            dict(
                col='Fuel_Consumption_rate', label='Fuel Mass Flow',
                ok_lo=0.0, ok_hi=12000.0,
                note_ok='kg/h — plausible (0–12 000 kg/h at full power)',
                note_warn=(
                    'Check units: <10 → t/h? >200 000 → kg/s? '
                    'Expected range 0–12 000 kg/h'
                ),
            ),
            dict(
                col='Fuel_Consumption_t_per_day', label='Fuel Consumption (t/day)',
                ok_lo=0.0, ok_hi=100.0,
                note_ok='t/day — plausible (0–100 t/day)',
                note_warn='Values outside 0–100 t/day — check units',
            ),
            dict(
                col='Avg_draft_m', label='Avg Draft',
                ok_lo=3.0, ok_hi=10.0,
                mm_hint=True,
                note_ok='Metres — plausible (3–10 m)',
                note_warn='Values outside 3–10 m ferry range',
                note_mm='Values > 100 → likely mm; divide by 1 000',
            ),
            dict(
                col='DRAFTAFT', label='Draft Aft',
                ok_lo=3.0, ok_hi=10.0,
                mm_hint=True,
                note_ok='Metres — plausible',
                note_warn='Values outside 3–10 m',
                note_mm='Values > 100 → likely mm; divide by 1 000',
            ),
            dict(
                col='DRAFTFWD', label='Draft Fwd',
                ok_lo=3.0, ok_hi=10.0,
                mm_hint=True,
                note_ok='Metres — plausible',
                note_warn='Values outside 3–10 m',
                note_mm='Values > 100 → likely mm; divide by 1 000',
            ),
        ]

        rows = []
        for c in checks:
            col = c['col']
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(s) == 0:
                rows.append({
                    'column': col, 'label': c['label'],
                    'n_valid': 0, 'observed_min': None, 'p5': None,
                    'median': None, 'p95': None, 'observed_max': None,
                    'status': 'NO DATA', 'note': 'No valid numeric values',
                })
                continue

            obs_min = float(s.min())
            obs_max = float(s.max())
            p5      = float(s.quantile(0.05))
            med     = float(s.median())
            p95     = float(s.quantile(0.95))

            status = 'ok'
            note   = c.get('note_ok', '')

            # Radian hint for lat/lon
            if c.get('radian_hint') and (abs(p95) < 2.0 and abs(p5) < 2.0):
                status = 'CRITICAL'
                note   = c.get('note_radian', 'May be radians')
            # m/s hint for speed
            elif c.get('ms_hint') and (p95 < 13.0 and med > 0.5):
                # speeds < 13 might still be knots if vessel is slow;
                # distinguish by absolute max
                if obs_max < 15.0 and med < 6.0:
                    status = 'WARNING'
                    note   = c.get('note_ms', 'May be m/s')
                else:
                    note = c.get('note_ok', '')
            # Watt hint for power
            elif c.get('watt_hint') and p95 > 100_000:
                status = 'CRITICAL'
                note   = c.get('note_watt', 'May be W, not kW')
            # mm hint for draft
            elif c.get('mm_hint') and p95 > 100:
                status = 'CRITICAL'
                note   = c.get('note_mm', 'May be mm, not m')
            # Generic out-of-range
            elif obs_min < c['ok_lo'] or p95 > c['ok_hi']:
                if status == 'ok':
                    status = 'WARNING'
                    note   = c.get('note_warn', f'Values outside [{c["ok_lo"]}, {c["ok_hi"]}]')

            rows.append({
                'column':       col,
                'label':        c['label'],
                'n_valid':      len(s),
                'observed_min': round(obs_min, 4),
                'p5':           round(p5, 4),
                'median':       round(med, 4),
                'p95':          round(p95, 4),
                'observed_max': round(obs_max, 4),
                'status':       status,
                'note':         note,
            })

        unit_df = pd.DataFrame(rows)
        unit_df.to_csv(out_dir / 'unit_sanity.csv', index=False)
        return rows


# ── Module-level print helpers ─────────────────────────────────────────────────

def _summarise_time_audit(audit: dict) -> None:
    if audit.get('skipped'):
        print(f"  [Audit/Time] Skipped: {audit.get('reason')}")
        return
    ok = ('✓' if audit['is_sorted']
          else f"✗ NOT SORTED  ({audit['n_out_of_order']} out-of-order rows)")
    print(f"  [Audit/Time] Sort order          : {ok}")
    print(f"  [Audit/Time] Unique timestamps   : {audit['n_unique_timestamps']:,}  "
          f"(total rows: {audit['n_total_timestamps']:,})")
    print(f"  [Audit/Time] Duplicate rows      : {audit['n_duplicate_rows']:,}  "
          f"({audit['pct_duplicates']:.2f}%)")
    print(f"  [Audit/Time] Median interval     : {audit['median_interval_sec']} s  "
          f"(expected 14 s)")
    print(f"  [Audit/Time] At expected ± 2 s   : {audit['pct_at_expected']:.1f}%")
    print(f"  [Audit/Time] Missed beats        : {audit['n_missed_beats']:,}  "
          f"(coverage: {audit['pct_coverage']:.1f}%)")
    print(f"  [Audit/Time] Irregular gaps      : {audit['n_irregular_gaps']:,}")
    print(f"  [Audit/Time] Long gaps > 5 min   : {audit['n_long_gaps']:,}  "
          f"({audit['total_gap_hours']:.1f} h total)")
    print(f"  [Audit/Time] Repeated blocks     : {audit['n_repeated_blocks']}")


def _summarise_unit_audit(rows: list) -> None:
    for r in rows:
        icon = {'ok': '✓', 'WARNING': '⚠', 'CRITICAL': '✗', 'NO DATA': '–'}.get(
            r['status'], '?'
        )
        print(f"  [Audit/Units] {icon} {r['column']:<30s}  "
              f"median={r['median']}  p95={r['p95']}  → {r['status']}: {r['note']}")
