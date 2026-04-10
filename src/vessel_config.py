"""
Vessel-specific data schema configurations.

Each VesselConfig defines:
  - How to rename raw CSV columns to the standard pipeline column names
  - Derived columns to compute after renaming (e.g. Avg_draft_m)
  - Feature columns used for anomaly detection
  - Sentinel values to replace with NaN
  - Timestamp column name and parse format

Detection is based on the folder name in the data path:
  - Path contains 'stenaline'   → STENALINE config  (MMSI 9235517)
  - Path contains 'stenateknik' → STENATEKNIK config (MMSI 9685475)
  - Path contains 'grimaldi'    → GRIMALDI config    (EUROCARGO GENOVA)
  - Otherwise                   → DEFAULT config     (data1.csv schema)

Standard column names used throughout the pipeline
───────────────────────────────────────────────────
  Date                    – timestamp
  GPSSpeed_kn             – vessel speed over ground (kn)
  GPS_LAT / GPS_LON       – position
  Main_Engine_Power_kW    – propulsion power (kW)
  Speed_rpm               – shaft / engine RPM
  Fuel_Consumption_t_per_day  – fuel rate (t/day)  [DEFAULT only]
  Fuel_Consumption_rate   – fuel rate in vessel-native units [Stenaline: kg/h]
  DRAFTAFT / DRAFTFWD     – aft / fwd hull draft (m)
  Avg_draft_m             – mean hull draft (m)  [derived if absent]
  Trim_m                  – hull trim (m)  [derived if absent]
  TrueWindSpeed_kn        – true wind speed
  RelWindSpeed_kn         – apparent / relative wind speed
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class VesselConfig:
    """Schema configuration for a single vessel."""

    vessel_id: str    # short id, e.g. 'stenaline'
    vessel_name: str  # human-readable label
    imo: str          # IMO / MMSI identifier
    folder_hint: str  # case-insensitive substring matched against the data path

    # Raw CSV column name → standard pipeline column name
    column_map: Dict[str, str] = field(default_factory=dict)

    # Columns to compute after renaming: {std_name: callable(df) -> Series}
    derived_cols: Dict[str, Callable[[pd.DataFrame], pd.Series]] = field(
        default_factory=dict
    )

    # Feature columns used for anomaly detection (standard names)
    feature_cols: List[str] = field(default_factory=list)

    # Feature groups for EDA categorisation (standard names)
    # Keys: 'engine_propulsion' | 'navigation' | 'draft' | 'weather' | 'trip'
    feature_groups: Dict[str, List[str]] = field(default_factory=dict)

    # Sentinel values to replace with NaN: {std_col_name: sentinel_value}
    sentinel_values: Dict[str, float] = field(default_factory=dict)

    # Name of the timestamp column *after* renaming
    timestamp_col: str = 'Date'

    # strptime format string for the timestamp (None = let pandas infer)
    date_format: Optional[str] = None

    # Expected sampling interval in seconds (used by time-axis audit)
    expected_interval_s: int = 14

    # Per-vessel physical limits: {std_col_name: (lo, hi)}
    # Overrides the global UNIVARIATE_SIGNAL_PROFILES phys_lo/phys_hi in EDA.
    # Use None for an open-ended bound, e.g. (0.0, None).
    physical_limits: Dict[str, tuple] = field(default_factory=dict)

    # Fuel-power envelope filter (used in EDA soft-warning check):
    #   expected_power = fuel_rate × fuel_power_slope
    #   flag if power < expected + offset_low  OR  power > expected + offset_high
    fuel_power_slope: float = 4.7
    fuel_power_offset_low: float = -4500.0
    fuel_power_offset_high: float = 3500.0


# ── Stenaline (MMSI 9235517) ───────────────────────────────────────────────────
# Raw CSV columns (from 9235517_YYYYMM.csv):
#   sensor_time, AWA, AWS, DraftAftDynamic, DraftFwdDynamic,
#   FuelMassFlowMETotal, LAT, LONG, PropulsionPowerTotal, SOG
#
# Note: FuelMassFlowMETotal is in kg/h (not t/day), renamed accordingly.
# Note: LAT/LONG are stored in RADIANS — convert to degrees before use
#       (see fix_stenaline_latlon.py). No RPM, STW, wave or current in raw data.

STENALINE = VesselConfig(
    vessel_id='stenaline',
    vessel_name='Stenaline (MMSI 9235517)',
    imo='9235517',
    folder_hint='stenaline',
    column_map={
        'sensor_time':          'Date',
        'SOG':                  'GPSSpeed_kn',
        'LAT':                  'GPS_LAT',
        'LONG':                 'GPS_LON',
        'PropulsionPowerTotal': 'Main_Engine_Power_kW',
        'FuelMassFlowMETotal':  'Fuel_Consumption_rate',   # kg/h
        'DraftAftDynamic':      'DRAFTAFT',
        'DraftFwdDynamic':      'DRAFTFWD',
        'AWS':                  'RelWindSpeed_kn',
        'AWA':                  'RelWindAngle_deg',
    },
    derived_cols={
        'Avg_draft_m': lambda df: (df['DRAFTAFT'] + df['DRAFTFWD']) / 2,
        'Trim_m':      lambda df: df['DRAFTAFT'] - df['DRAFTFWD'],
    },
    feature_cols=[
        # Speed
        'GPSSpeed_kn',          # speed over ground (SOG)
        # Power & fuel
        'Main_Engine_Power_kW', # shaft power
        'Fuel_Consumption_rate',# fuel consumption (kg/h)
        # Draft
        'DRAFTAFT',             # draft aft
        'DRAFTFWD',             # draft fwd
        # Metocean
        'RelWindSpeed_kn',      # apparent wind speed
        # Position  ⚠ raw files store radians — convert before modelling
        'GPS_LAT',
        'GPS_LON',
        # Note: Speed_rpm, SpeedLog_kn (STW), Wave_Height_m, Current_Speed_kn
        #       are NOT available in Stenaline raw data
    ],
    feature_groups={
        'engine_propulsion': ['Main_Engine_Power_kW', 'Fuel_Consumption_rate'],
        'navigation':        ['GPSSpeed_kn', 'GPS_LAT', 'GPS_LON'],
        'draft':             ['DRAFTAFT', 'DRAFTFWD'],
        'weather':           ['RelWindSpeed_kn'],
        'trip':              [],   # not available in Stenaline raw data
    },
    sentinel_values={},  # no known sentinel values in raw Stenaline CSVs
    timestamp_col='Date',
    date_format=None,   # files use mixed formats: dd/mm/yyyy HH:MM (Jul) and
                        # yyyy-mm-dd HH:MM:SS (Aug+); let pandas auto-detect
    physical_limits={
        'GPSSpeed_kn':           (0.0, 30.0),
        'Main_Engine_Power_kW':  (0.0, 30_000.0),
        'Fuel_Consumption_rate': (0.0, 6_000.0),
        'DRAFTAFT':              (0.0, 15.0),
        'DRAFTFWD':              (0.0, 15.0),
    },
    fuel_power_slope=4.3,
    fuel_power_offset_low=-4500.0,
    fuel_power_offset_high=4000.0,
)


# ── Stenateknik (MMSI 9685475) ─────────────────────────────────────────────────
# Raw CSV columns (from 9685475_YYYYMM.csv — verified against 202407 sample):
#   sensor_time, Latitude (deg), Longitude (deg),
#   Ship Speed GPS (knot), Ship Speed Log (knot), Ship Course (deg),
#   ME Shaft Power (kW), ME Shaft Speed (rpm), ME Shaft Torque (kNm),
#   ME Shaft Thrust (kN), ME Fuel Mass Net (kg/hr),
#   Draft Aft (m), Draft Fwd (m), Draft Mid Port (m), Draft Mid Stbd (m),
#   Depth of Water (m), Wind Speed Rel. (knot), Wind Dir. Rel. (deg),
#   GE #1–4 El. Output (kW)
#
# Timestamp format: ISO  '2024-07-01 00:00:13.440'

STENATEKNIK = VesselConfig(
    vessel_id='stenateknik',
    vessel_name='Stenateknik (MMSI 9685475)',
    imo='9685475',
    folder_hint='stenateknik',
    column_map={
        'sensor_time':              'Date',
        'Ship Speed GPS (knot)':    'GPSSpeed_kn',
        'Ship Speed Log (knot)':    'SpeedLog_kn',
        'Latitude (deg)':           'GPS_LAT',
        'Longitude (deg)':          'GPS_LON',
        'ME Shaft Power (kW)':      'Main_Engine_Power_kW',
        'ME Shaft Speed (rpm)':     'Speed_rpm',
        'ME Shaft Torque (kNm)':    'ME_Shaft_Torque_kNm',
        'ME Shaft Thrust (kN)':     'ME_Shaft_Thrust_kN',
        'ME Fuel Mass Net (kg/hr)': 'Fuel_Consumption_rate',   # kg/h
        'Draft Aft (m)':            'DRAFTAFT',
        'Draft Fwd (m)':            'DRAFTFWD',
        'Draft Mid Port (m)':       'Draft_Mid_Port_m',
        'Draft Mid Stbd (m)':       'Draft_Mid_Stbd_m',
        'Depth of Water (m)':       'Depth_of_Water_m',
        'Wind Speed Rel. (knot)':   'RelWindSpeed_kn',
        'Wind Dir. Rel. (deg)':     'RelWindAngle_deg',
        'Ship Course (deg)':        'Ship_Course_deg',
        'GE #1 El. Output (kW)':    'GE1_Power_kW',
        'GE #2 El. Output (kW)':    'GE2_Power_kW',
        'GE #3 El. Output (kW)':    'GE3_Power_kW',
        'GE #4 El. Output (kW)':    'GE4_Power_kW',
    },
    derived_cols={
        'Avg_draft_m': lambda df: (df['DRAFTAFT'] + df['DRAFTFWD']) / 2,
        'Trim_m':      lambda df: df['DRAFTAFT'] - df['DRAFTFWD'],
    },
    feature_cols=[
        # Speed
        'GPSSpeed_kn',           # speed over ground (SOG)
        'SpeedLog_kn',           # speed through water (STW)
        # Power & fuel
        'Main_Engine_Power_kW',  # shaft power
        'Speed_rpm',             # shaft RPM
        'Fuel_Consumption_rate', # fuel consumption (kg/h)
        # Draft
        'DRAFTAFT',              # draft aft
        'DRAFTFWD',              # draft fwd
        # Metocean
        'RelWindSpeed_kn',       # apparent wind speed
        # Position
        'GPS_LAT',
        'GPS_LON',
        # Note: Wave_Height_m, Current_Speed_kn are NOT available
        #       in Stenateknik raw data
    ],
    feature_groups={
        'engine_propulsion': ['Main_Engine_Power_kW', 'Speed_rpm', 'Fuel_Consumption_rate'],
        'navigation':        ['GPSSpeed_kn', 'SpeedLog_kn', 'GPS_LAT', 'GPS_LON'],
        'draft':             ['DRAFTAFT', 'DRAFTFWD'],
        'weather':           ['RelWindSpeed_kn'],
        'trip':              [],   # not available in Stenateknik raw data
    },
    sentinel_values={},
    timestamp_col='Date',
    date_format=None,   # ISO format: yyyy-mm-dd HH:MM:SS.sss
    physical_limits={
        'GPSSpeed_kn':           (0.0, 25.0),
        'Main_Engine_Power_kW':  (0.0, 10_000.0),
        'Fuel_Consumption_rate': (0.0, 1_500.0),
        'DRAFTAFT':              (0.0, 20.0),
        'DRAFTFWD':              (0.0, 20.0),
        'ME_Shaft_Torque_kNm':   (-1_000.0, 1_000.0),
    },
    fuel_power_slope=5.3,
    fuel_power_offset_low=-3500.0,
    fuel_power_offset_high=3500.0,
)


# ── Default / legacy schema (data1.csv) ───────────────────────────────────────
# Columns are already in the standard pipeline format; no renaming needed.

DEFAULT = VesselConfig(
    vessel_id='default',
    vessel_name='Unknown / Legacy (data1.csv schema)',
    imo='unknown',
    folder_hint='',   # never matched by path — used as fallback only
    column_map={},
    derived_cols={},
    feature_cols=[
        'GPSSpeed_kn',
        'Main_Engine_Power_kW',
        'Speed_rpm',
        'Fuel_Consumption_t_per_day',
        'Avg_draft_m',
        'Trim_m',
        'TrueWindSpeed_kn',
        'RelWindSpeed_kn',
    ],
    feature_groups={
        'engine_propulsion': ['Main_Engine_Power_kW', 'Speed_rpm',
                              'Fuel_Consumption_t_per_day'],
        'navigation':        ['GPSSpeed_kn', 'GPS_LAT', 'GPS_LON'],
        'draft':             ['DRAFTAFT', 'DRAFTFWD', 'Avg_draft_m', 'Trim_m'],
        'weather':           ['TrueWindSpeed_kn', 'RelWindSpeed_kn'],
        'trip':              [],
    },
    sentinel_values={
        'DRAFTFWD': -9999,
        'DRAFTAFT': -9999,
    },
    timestamp_col='Date',
    date_format=None,
)


# ── Grimaldi (EUROCARGO GENOVA) ───────────────────────────────────────────────
# Source: EUROCARGO GENOVA dati.xlsx — 294,299 rows, 56 columns
# Timestamp: 'Time stamp' column, already parsed as datetime by pandas (ISO)
# Fuel rate: 'ME FLOWMETER FLOW RATE [mt/h]' — metric tonnes per hour
# Generator sets: DG1–DG4 (diesel generators, kW each)
# Metocean fields: wind, wave, swell, current, pressure, humidity (no renaming)
# Note: LATITUDE/LONGITUDE stored in RADIANS — convert to degrees before use
#       (see fix_grimaldi_latlon.py). Full metocean suite available (wave, current).

GRIMALDI = VesselConfig(
    vessel_id='grimaldi',
    vessel_name='Grimaldi – EUROCARGO GENOVA',
    imo='unknown',
    folder_hint='grimaldi',
    column_map={
        'Time stamp':                       'Date',
        'LATITUDE':                         'GPS_LAT',
        'LONGITUDE':                        'GPS_LON',
        'SPEED OVER GROUND [kn]':           'GPSSpeed_kn',
        'SPEED THROUGH WATER [kn]':         'SpeedLog_kn',
        'COURSE OVER GROUND [ ° ]':         'Ship_Course_deg',
        'HEADING  [ ° ]':                   'Heading_deg',
        'DISPLACEMENT  [m]':                'Displacement_m',
        'DRAFTAFT[m]':                      'DRAFTAFT',
        'DRAFTFOR[m]':                      'DRAFTFWD',
        'SHAFT POWER[kW]':                  'Main_Engine_Power_kW',
        'PROPELLER PITCH [%]':              'Propeller_Pitch_pct',
        'SHAFT TORQUE [kNm]':               'ME_Shaft_Torque_kNm',
        'SHAFT SPEED [rpm]':                'Speed_rpm',
        'ME FLOWMETER FLOW RATE [mt/h]':    'Fuel_Consumption_rate',   # mt/h
        'ME FLOWMETER DENSITY [kg/m3]':     'Fuel_Density_kg_m3',
        'Flow Temp [°C]':                   'Fuel_Flow_Temp_C',
        'ME FLOW TOTAL [mt]':               'Fuel_Flow_Total_mt',
        'DG1_POWER [kW]':                   'DG1_Power_kW',
        'DG2_POWER [kW]':                   'DG2_Power_kW',
        'DG3_POWER [kW]':                   'DG3_Power_kW',
        'DG4_POWER [kW]':                   'DG4_Power_kW',
        'WIND SPEED_1':                     'RelWindSpeed_kn',
        'WIND DIR_1':                       'RelWindAngle_deg',
        'SEA FORCE':                        'Sea_Force',
        'SEA DIRECTION':                    'Sea_Direction_deg',
        'AIR TEMP FORECAST [°C]':           'Air_Temp_C',
        'SWELL HEIGHT [m]':                 'Swell_Height_m',
        'SWELL PERIOD [sec]':               'Swell_Period_s',
        'SWELL DIRECTION [°]':              'Swell_Direction_deg',
        'WAVE HEIGHT [m]':                  'Wave_Height_m',
        'WAVE PERIOD [sec]':                'Wave_Period_s',
        'WAVE DIRECTION [°]':               'Wave_Direction_deg',
        'CURRENT SPEED [kn]':               'Current_Speed_kn',
        'CURRENT DIRECTION [°]':            'Current_Direction_deg',
        'PRESSURE [mb]':                    'Pressure_mb',
        'VISIBILITY [m]':                   'Visibility_m',
        'HUMIDITY [%]':                     'Humidity_pct',
        'NAUTICAL MILES':                   'Nautical_Miles',
    },
    derived_cols={
        'Avg_draft_m': lambda df: (df['DRAFTAFT'] + df['DRAFTFWD']) / 2,
        'Trim_m':      lambda df: df['DRAFTAFT'] - df['DRAFTFWD'],
    },
    feature_cols=[
        # Speed
        'GPSSpeed_kn',           # speed over ground (SOG)
        'SpeedLog_kn',           # speed through water (STW)
        # Power & fuel
        'Main_Engine_Power_kW',  # shaft power
        'Speed_rpm',             # shaft RPM
        'Fuel_Consumption_rate', # fuel consumption (mt/h)
        # Draft
        'DRAFTAFT',              # draft aft
        'DRAFTFWD',              # draft fwd
        # Metocean
        'RelWindSpeed_kn',       # apparent wind speed
        'Wave_Height_m',         # significant wave height
        'Current_Speed_kn',      # sea current speed
        # Position  ⚠ raw file stores radians — convert before modelling
        'GPS_LAT',
        'GPS_LON',
    ],
    feature_groups={
        'engine_propulsion': ['Main_Engine_Power_kW', 'Speed_rpm',
                              'Fuel_Consumption_rate'],
        'navigation':        ['GPSSpeed_kn', 'SpeedLog_kn', 'GPS_LAT', 'GPS_LON'],
        'draft':             ['DRAFTAFT', 'DRAFTFWD'],
        'weather':           ['RelWindSpeed_kn', 'Wave_Height_m', 'Current_Speed_kn'],
        'trip':              ['Nautical_Miles'],
    },
    sentinel_values={},
    timestamp_col='Date',
    date_format=None,   # timestamps pre-parsed as datetime by openpyxl/pandas
    expected_interval_s=120,   # 2-minute sampling
)


# ── Registry (order matters: more-specific hints first) ───────────────────────
_CONFIGS: List[VesselConfig] = [STENALINE, STENATEKNIK, GRIMALDI]


# ── Public helpers ─────────────────────────────────────────────────────────────

def detect_vessel(path: str) -> VesselConfig:
    """
    Return the VesselConfig that matches the data path.

    Checks each registered config's ``folder_hint`` against the path string
    (case-insensitive).  Falls back to ``DEFAULT`` when nothing matches.

    Args:
        path: File or directory path as a string or Path.

    Returns:
        Matching VesselConfig, or DEFAULT if no config matched.
    """
    path_lower = str(path).lower()
    for cfg in _CONFIGS:
        if cfg.folder_hint and cfg.folder_hint.lower() in path_lower:
            return cfg
    return DEFAULT


def _parse_timestamps_auto(series: pd.Series) -> pd.Series:
    """
    Parse a Series of timestamp strings, auto-detecting year-first (ISO)
    vs day-first (European dd/mm/yyyy) format from the first non-null value.
    """
    nonnull = series.dropna()
    if nonnull.empty:
        return pd.to_datetime(series, errors='coerce')
    sample = str(nonnull.iloc[0]).strip()
    # ISO / year-first: '2024-08-01 ...' or '2024/08/01 ...' or '2024T...'
    if len(sample) >= 5 and sample[:4].isdigit() and sample[4] in ('-', '/', 'T'):
        return pd.to_datetime(series, errors='coerce')
    # European / day-first: '01/07/2024 ...' or '01-07-2024 ...'
    return pd.to_datetime(series, dayfirst=True, errors='coerce')


def normalize_dataframe(df: pd.DataFrame, config: VesselConfig) -> pd.DataFrame:
    """
    Apply column renaming, timestamp parsing, sentinel replacement, and
    derived-column computation for a vessel.

    Args:
        df:     Raw DataFrame as loaded from CSV / Excel.
        config: VesselConfig for the vessel that produced this data.

    Returns:
        New DataFrame with standard column names and derived columns added.
    """
    df = df.copy()

    # 1. Rename raw → standard column names
    rename_map = {k: v for k, v in config.column_map.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    # 1b. Treat empty strings and whitespace-only strings as NaN so that all
    #     downstream isnull()/isna() calls count them correctly as missing.
    str_cols = df.select_dtypes(include='object').columns
    if len(str_cols):
        df[str_cols] = (
            df[str_cols]
            .apply(lambda s: s.str.strip())   # strip surrounding whitespace
            .replace('', np.nan)               # blank string → NaN
        )

    # 2. Parse timestamp column
    ts = config.timestamp_col
    if ts in df.columns and not pd.api.types.is_datetime64_any_dtype(df[ts]):
        if config.date_format:
            parsed = pd.to_datetime(df[ts], format=config.date_format, errors='coerce')
            # If the strict format produced > 50% NaT, fall back to auto-detection
            if parsed.isna().mean() > 0.5:
                parsed = _parse_timestamps_auto(df[ts])
            df[ts] = parsed
        else:
            df[ts] = _parse_timestamps_auto(df[ts])

    # 3. Replace sentinel values with NaN
    for col, sentinel in config.sentinel_values.items():
        if col in df.columns:
            df.loc[df[col] == sentinel, col] = np.nan

    # 4. Compute derived columns (only if not already present)
    for col_name, compute_fn in config.derived_cols.items():
        if col_name not in df.columns:
            try:
                df[col_name] = compute_fn(df)
            except Exception as exc:
                print(
                    f"  [VesselConfig] WARNING — could not derive '{col_name}': {exc}"
                )

    return df
