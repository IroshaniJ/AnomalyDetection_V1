"""
Data loading and preprocessing for maritime anomaly detection.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional

# Supported file extensions for data discovery
_SUPPORTED_EXTENSIONS = ('.csv', '.xlsx', '.xls')


def discover_files(path: str) -> List[Path]:
    """
    Return all supported data files for the given path.

    If *path* points to a file, returns ``[path]``.
    If *path* points to a directory, recursively finds all CSV / XLSX files
    in that directory and its sub-folders, sorted by their full path.

    Args:
        path: File path or directory path.

    Returns:
        Sorted list of Path objects for each discovered file.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If a file path has an unsupported extension.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data path not found: {path}")

    if p.is_file():
        if p.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension '{p.suffix}'. "
                f"Supported: {_SUPPORTED_EXTENSIONS}"
            )
        return [p]

    files = sorted(
        f for f in p.rglob('*')
        if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No CSV or XLSX files found under directory: {path}"
        )
    return files


def _read_single_file(filepath: Path) -> pd.DataFrame:
    """
    Read a single CSV or Excel file into a DataFrame.

    Attempts to parse a ``Date`` column if present.
    """
    ext = filepath.suffix.lower()
    if ext == '.csv':
        try:
            df = pd.read_csv(filepath, parse_dates=['Date'])
        except ValueError:
            df = pd.read_csv(filepath)
    elif ext in ('.xlsx', '.xls'):
        try:
            df = pd.read_excel(filepath, parse_dates=['Date'])
        except ValueError:
            df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    return df


def load_raw(path: str, vessel_config=None) -> pd.DataFrame:
    """
    Load raw data from a file or a directory of files.

    The vessel is auto-detected from the path (Stenaline / Stenateknik /
    default) and its column map is applied so the returned DataFrame always
    uses the standard pipeline column names.

    When *path* is a directory all CSV / XLSX files found recursively are
    concatenated into a single DataFrame.  A ``source_file`` column records
    the originating filename for every row.

    Args:
        path:          Path to a CSV / XLSX file **or** a directory.
        vessel_config: Optional pre-resolved VesselConfig. When None the
                       vessel is detected automatically from *path*.

    Returns:
        Concatenated, index-reset DataFrame with standard column names and
        a ``source_file`` column.
    """
    from src.vessel_config import detect_vessel, normalize_dataframe

    if vessel_config is None:
        vessel_config = detect_vessel(path)
    print(f"  [Loader] Vessel : {vessel_config.vessel_name}")

    files = discover_files(path)
    frames: List[pd.DataFrame] = []
    skipped: List[str] = []
    for f in files:
        try:
            df_part = _read_single_file(f)
            df_part = normalize_dataframe(df_part, vessel_config)
            df_part['source_file'] = f.name
            frames.append(df_part)
            print(f"  [Loader] Read {len(df_part):,} rows from {f}")
        except Exception as exc:
            print(f"  [Loader] WARNING — skipping {f}: {exc}")
            skipped.append(str(f))

    if not frames:
        raise RuntimeError(
            f"No files could be loaded from '{path}'. "
            f"All {len(skipped)} file(s) failed to read."
        )

    if skipped:
        print(f"  [Loader] {len(skipped)} file(s) skipped due to read errors:")
        for s in skipped:
            print(f"             {s}")

    df = pd.concat(frames, ignore_index=True)
    print(
        f"  [Loader] Total rows after concat: {len(df):,} "
        f"(from {len(frames)} of {len(files)} file(s))"
    )
    return df


# Default feature columns for anomaly detection (data1.csv / legacy schema).
# Vessel-specific feature columns are defined in src/vessel_config.py.
FEATURE_COLS = [
    'GPSSpeed_kn',
    'Main_Engine_Power_kW',
    'Speed_rpm',
    'Fuel_Consumption_t_per_day',
    'Avg_draft_m',
    'Trim_m',
    'TrueWindSpeed_kn',
    'RelWindSpeed_kn',
]


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load and clean vessel data from a CSV / XLSX file or a directory.

    The vessel schema is auto-detected from the path. Column renaming,
    sentinel replacement, and derived columns (Avg_draft_m, Trim_m) are
    applied before any quality flags are added.

    When *filepath* is a directory, all CSV / XLSX files found recursively
    are concatenated before cleaning.

    Args:
        filepath: Path to a CSV / XLSX file **or** a directory.

    Returns:
        Cleaned DataFrame with standard column names, sentinel values
        replaced, and anomaly flag columns added where applicable.
    """
    from src.vessel_config import detect_vessel

    config = detect_vessel(filepath)
    # load_raw applies normalize_dataframe (rename + sentinel + derived cols)
    df = load_raw(filepath, vessel_config=config)

    # Nullify Avg_draft_m where either draft measurement is NaN
    if {'DRAFTAFT', 'DRAFTFWD', 'Avg_draft_m'}.issubset(df.columns):
        bad = df['DRAFTAFT'].isna() | df['DRAFTFWD'].isna()
        df.loc[bad, 'Avg_draft_m'] = np.nan

    # ── Constraint violation flags (only for columns that exist) ─────────────
    flag_cols: List[str] = []

    if 'Main_Engine_Power_kW' in df.columns:
        df['flag_negative_power'] = df['Main_Engine_Power_kW'] < 0
        flag_cols.append('flag_negative_power')

    if 'Trim_m' in df.columns:
        df['flag_extreme_trim'] = df['Trim_m'].abs() > 5
        flag_cols.append('flag_extreme_trim')

    if 'Avg_draft_m' in df.columns:
        df['flag_invalid_draft'] = df['Avg_draft_m'] <= 0
        flag_cols.append('flag_invalid_draft')

    if flag_cols:
        df['flag_any_constraint'] = df[flag_cols].any(axis=1)
    else:
        df['flag_any_constraint'] = False

    # Operational state
    if 'GPSSpeed_kn' in df.columns:
        df['is_moving'] = df['GPSSpeed_kn'] > 0.5

    # Sort by timestamp
    ts_col = config.timestamp_col
    if ts_col in df.columns:
        df = df.sort_values(ts_col).reset_index(drop=True)

    return df


def preprocess_features(
    df: pd.DataFrame, 
    feature_cols: Optional[List[str]] = None,
    scaler: Optional[MinMaxScaler] = None
) -> Tuple[np.ndarray, MinMaxScaler, np.ndarray]:
    """
    Preprocess features with min-max scaling.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature columns to use (default: FEATURE_COLS)
        scaler: Pre-fitted scaler for inference (None for training)
        
    Returns:
        Tuple of (scaled features, fitted scaler, valid row mask)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    
    # Extract features
    X = df[feature_cols].copy()
    
    # Track valid rows (no NaN values)
    valid_mask = ~X.isna().any(axis=1)
    X_valid = X[valid_mask].values
    
    # Fit or apply scaler
    if scaler is None:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_valid)
    else:
        X_scaled = scaler.transform(X_valid)
    
    return X_scaled, scaler, valid_mask.values


def get_vessel_config(path: str):
    """Return the VesselConfig detected from *path*."""
    from src.vessel_config import detect_vessel
    return detect_vessel(path)


def get_feature_columns(path: Optional[str] = None) -> List[str]:
    """
    Return feature columns for anomaly detection.

    When *path* is provided the vessel is auto-detected and its specific
    feature column list is returned (if non-empty).  Falls back to the
    default FEATURE_COLS for unknown vessels or when no path is given.

    Args:
        path: Optional data file or directory path.

    Returns:
        List of feature column names (standard pipeline names).
    """
    if path is not None:
        from src.vessel_config import detect_vessel
        cfg = detect_vessel(path)
        if cfg.feature_cols:
            return cfg.feature_cols.copy()
    return FEATURE_COLS.copy()
