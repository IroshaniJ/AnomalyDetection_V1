"""
Data loading and preprocessing for maritime anomaly detection.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional


# Feature columns for anomaly detection
FEATURE_COLS = [
    'GPSSpeed_kn', 
    'Main_Engine_Power_kW', 
    'Speed_rpm',
    'Fuel_Consumption_t_per_day', 
    'Avg_draft_m', 
    'Trim_m',
    'TrueWindSpeed_kn', 
    'RelWindSpeed_kn'
]

# Sentinel values to replace
SENTINEL_VALUES = {
    'DRAFTFWD': -9999,
    'DRAFTAFT': -9999,
}


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV data and perform initial cleaning.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Cleaned DataFrame with sentinel values handled and anomaly flags
    """
    # Load data
    df = pd.read_csv(filepath, parse_dates=['Date'])
    
    # Handle sentinel values
    for col, sentinel in SENTINEL_VALUES.items():
        if col in df.columns:
            df.loc[df[col] == sentinel, col] = np.nan
    
    # Recalculate Avg_draft_m if drafts were cleaned
    mask_bad_draft = df['DRAFTFWD'].isna() | df['DRAFTAFT'].isna()
    df.loc[mask_bad_draft, 'Avg_draft_m'] = np.nan
    
    # Flag known constraint violations (these are definitely anomalies)
    df['flag_negative_power'] = df['Main_Engine_Power_kW'] < 0
    df['flag_extreme_trim'] = df['Trim_m'].abs() > 5
    df['flag_invalid_draft'] = df['Avg_draft_m'] <= 0
    df['flag_any_constraint'] = (
        df['flag_negative_power'] | 
        df['flag_extreme_trim'] | 
        df['flag_invalid_draft']
    )
    
    # Derive operational state
    df['is_moving'] = df['GPSSpeed_kn'] > 0.5
    
    # Sort by timestamp
    df = df.sort_values('Date').reset_index(drop=True)
    
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


def get_feature_columns() -> List[str]:
    """Return the default feature columns."""
    return FEATURE_COLS.copy()
