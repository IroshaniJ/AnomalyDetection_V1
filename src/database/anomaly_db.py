"""
SQLite database for storing and querying detected anomalies.
"""
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path


class AnomalyDatabase:
    """
    SQLite database for anomaly storage, tracking, and analysis.
    """
    
    def __init__(self, db_path: str = 'anomalies.db'):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_timestamp TEXT NOT NULL,
                data_timestamp TEXT,
                anomaly_score REAL NOT NULL,
                threshold REAL NOT NULL,
                model_version TEXT,
                GPSSpeed_kn REAL,
                GPS_LAT REAL,
                GPS_LON REAL,
                Main_Engine_Power_kW REAL,
                Speed_rpm REAL,
                Fuel_Consumption_t_per_day REAL,
                Avg_draft_m REAL,
                Trim_m REAL,
                TrueWindSpeed_kn REAL,
                RelWindSpeed_kn REAL,
                anomaly_type TEXT,
                reviewed INTEGER DEFAULT 0,
                notes TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS detection_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp TEXT NOT NULL,
                model_version TEXT,
                data_file TEXT,
                total_records INTEGER,
                anomalies_detected INTEGER,
                threshold REAL,
                n_components INTEGER,
                threshold_percentile REAL
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS missing_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_timestamp TEXT NOT NULL,
                data_timestamp TEXT,
                reason TEXT NOT NULL,
                missing_columns TEXT,
                source_file TEXT,
                GPSSpeed_kn REAL,
                GPS_LAT REAL,
                GPS_LON REAL,
                Main_Engine_Power_kW REAL,
                Speed_rpm REAL,
                Fuel_Consumption_t_per_day REAL,
                Avg_draft_m REAL,
                Trim_m REAL,
                TrueWindSpeed_kn REAL,
                RelWindSpeed_kn REAL,
                DRAFTFWD REAL,
                DRAFTAFT REAL
            )
        ''')

        conn.commit()
        conn.close()
    
    def log_detection_run(
        self,
        model_version: str,
        data_file: str,
        total_records: int,
        anomalies_detected: int,
        threshold: float,
        n_components: int,
        threshold_percentile: float
    ) -> int:
        """
        Log a detection run.
        
        Returns:
            Run ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            INSERT INTO detection_runs 
            (run_timestamp, model_version, data_file, total_records, 
             anomalies_detected, threshold, n_components, threshold_percentile)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow().isoformat(),
            model_version,
            data_file,
            total_records,
            anomalies_detected,
            threshold,
            n_components,
            threshold_percentile
        ))
        run_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return run_id
    
    def log_anomalies(
        self,
        df_anomalies: pd.DataFrame,
        scores: List[float],
        threshold: float,
        model_version: str = 'svd_v1'
    ) -> int:
        """
        Log detected anomalies to database.
        
        Args:
            df_anomalies: DataFrame containing anomaly records
            scores: Anomaly scores for each record
            threshold: Detection threshold used
            model_version: Model version identifier
            
        Returns:
            Number of records inserted
        """
        if len(df_anomalies) == 0:
            return 0
        
        conn = sqlite3.connect(self.db_path)
        detection_time = datetime.utcnow().isoformat()
        
        records = []
        for i, (idx, row) in enumerate(df_anomalies.iterrows()):
            score = scores[i] if i < len(scores) else None
            records.append({
                'detection_timestamp': detection_time,
                'data_timestamp': str(row.get('Date', '')) if pd.notna(row.get('Date')) else None,
                'anomaly_score': float(score) if score is not None else None,
                'threshold': float(threshold),
                'model_version': model_version,
                'GPSSpeed_kn': self._safe_float(row.get('GPSSpeed_kn')),
                'GPS_LAT': self._safe_float(row.get('GPS_LAT')),
                'GPS_LON': self._safe_float(row.get('GPS_LON')),
                'Main_Engine_Power_kW': self._safe_float(row.get('Main_Engine_Power_kW')),
                'Speed_rpm': self._safe_float(row.get('Speed_rpm')),
                'Fuel_Consumption_t_per_day': self._safe_float(row.get('Fuel_Consumption_t_per_day')),
                'Avg_draft_m': self._safe_float(row.get('Avg_draft_m')),
                'Trim_m': self._safe_float(row.get('Trim_m')),
                'TrueWindSpeed_kn': self._safe_float(row.get('TrueWindSpeed_kn')),
                'RelWindSpeed_kn': self._safe_float(row.get('RelWindSpeed_kn')),
                'anomaly_type': self._classify_anomaly(row),
                'reviewed': 0,
                'notes': None
            })
        
        pd.DataFrame(records).to_sql('anomalies', conn, if_exists='append', index=False)
        conn.close()
        
        return len(records)
    
    def log_missing_records(
        self,
        df_missing: pd.DataFrame,
        source_file: str = '',
        feature_cols: Optional[List[str]] = None,
    ) -> int:
        """
        Log records excluded during cleaning (NaN in feature columns) to the
        ``missing_records`` table.

        Args:
            df_missing:   DataFrame of rows that were removed during cleaning.
            source_file:  Original data file name for provenance.
            feature_cols: Feature columns used to detect missing values.

        Returns:
            Number of rows inserted.
        """
        if len(df_missing) == 0:
            return 0

        if feature_cols is None:
            feature_cols = [
                'GPSSpeed_kn', 'Main_Engine_Power_kW', 'Speed_rpm',
                'Fuel_Consumption_t_per_day', 'Avg_draft_m', 'Trim_m',
                'TrueWindSpeed_kn', 'RelWindSpeed_kn',
            ]

        logged_ts = datetime.utcnow().isoformat()
        records = []
        for _, row in df_missing.iterrows():
            # Determine which feature columns are missing
            missing_cols = [
                c for c in feature_cols
                if c in row.index and (row[c] is None or pd.isna(row[c]))
            ]
            # Classify reason
            if missing_cols:
                reason = 'missing_feature_data'
            elif pd.isna(row.get('Date')):
                reason = 'missing_timestamp'
            else:
                reason = 'invalid_record'

            records.append({
                'logged_timestamp':          logged_ts,
                'data_timestamp':            str(row['Date']) if 'Date' in row.index and pd.notna(row.get('Date')) else None,
                'reason':                    reason,
                'missing_columns':           ','.join(missing_cols) if missing_cols else None,
                'source_file':               source_file,
                'GPSSpeed_kn':               self._safe_float(row.get('GPSSpeed_kn')),
                'GPS_LAT':                   self._safe_float(row.get('GPS_LAT')),
                'GPS_LON':                   self._safe_float(row.get('GPS_LON')),
                'Main_Engine_Power_kW':      self._safe_float(row.get('Main_Engine_Power_kW')),
                'Speed_rpm':                 self._safe_float(row.get('Speed_rpm')),
                'Fuel_Consumption_t_per_day': self._safe_float(row.get('Fuel_Consumption_t_per_day')),
                'Avg_draft_m':               self._safe_float(row.get('Avg_draft_m')),
                'Trim_m':                    self._safe_float(row.get('Trim_m')),
                'TrueWindSpeed_kn':          self._safe_float(row.get('TrueWindSpeed_kn')),
                'RelWindSpeed_kn':           self._safe_float(row.get('RelWindSpeed_kn')),
                'DRAFTFWD':                  self._safe_float(row.get('DRAFTFWD')),
                'DRAFTAFT':                  self._safe_float(row.get('DRAFTAFT')),
            })

        conn = sqlite3.connect(self.db_path)
        pd.DataFrame(records).to_sql('missing_records', conn, if_exists='append', index=False)
        conn.close()
        return len(records)

    def get_missing_records(self) -> pd.DataFrame:
        """Retrieve all records excluded during cleaning."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(
            'SELECT * FROM missing_records ORDER BY data_timestamp ASC', conn
        )
        conn.close()
        return df

    def get_missing_records_summary(self) -> pd.DataFrame:
        """Summary of excluded records grouped by reason and missing column."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql('''
            SELECT reason, missing_columns, COUNT(*) as count,
                   MIN(data_timestamp) as first_seen,
                   MAX(data_timestamp) as last_seen
            FROM missing_records
            GROUP BY reason, missing_columns
            ORDER BY count DESC
        ''', conn)
        conn.close()
        return df

    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float, handling NaN."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _classify_anomaly(self, row: pd.Series) -> str:
        """Classify anomaly type based on constraint violations."""
        types = []
        
        power = row.get('Main_Engine_Power_kW')
        if power is not None and pd.notna(power) and power < 0:
            types.append('negative_power')
        
        trim = row.get('Trim_m')
        if trim is not None and pd.notna(trim) and abs(trim) > 5:
            types.append('extreme_trim')
        
        draft = row.get('Avg_draft_m')
        if draft is not None and pd.notna(draft) and draft <= 0:
            types.append('invalid_draft')
        
        return ','.join(types) if types else 'multivariate_outlier'
    
    def get_anomaly_summary(self) -> pd.DataFrame:
        """Get summary statistics by anomaly type."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql('''
            SELECT 
                anomaly_type, 
                COUNT(*) as count, 
                AVG(anomaly_score) as avg_score,
                MIN(anomaly_score) as min_score,
                MAX(anomaly_score) as max_score,
                MIN(data_timestamp) as first_seen,
                MAX(data_timestamp) as last_seen,
                SUM(reviewed) as reviewed_count
            FROM anomalies
            GROUP BY anomaly_type
            ORDER BY count DESC
        ''', conn)
        conn.close()
        return df
    
    def get_all_anomalies(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Get all anomalies, optionally limited."""
        conn = sqlite3.connect(self.db_path)
        query = 'SELECT * FROM anomalies ORDER BY anomaly_score DESC'
        if limit:
            query += f' LIMIT {limit}'
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    
    def get_unreviewed_anomalies(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Get unreviewed anomalies for manual inspection."""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT * FROM anomalies 
            WHERE reviewed = 0 
            ORDER BY anomaly_score DESC
        '''
        if limit:
            query += f' LIMIT {limit}'
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    
    def mark_reviewed(self, anomaly_ids: List[int], notes: Optional[str] = None) -> None:
        """Mark anomalies as reviewed."""
        conn = sqlite3.connect(self.db_path)
        placeholders = ','.join('?' * len(anomaly_ids))
        if notes:
            conn.execute(
                f'UPDATE anomalies SET reviewed = 1, notes = ? WHERE id IN ({placeholders})',
                [notes] + anomaly_ids
            )
        else:
            conn.execute(
                f'UPDATE anomalies SET reviewed = 1 WHERE id IN ({placeholders})',
                anomaly_ids
            )
        conn.commit()
        conn.close()
    
    def get_detection_runs(self) -> pd.DataFrame:
        """Get history of detection runs."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql('SELECT * FROM detection_runs ORDER BY run_timestamp DESC', conn)
        conn.close()
        return df
    
    def clear_anomalies(self) -> None:
        """Clear all anomalies (use with caution)."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('DELETE FROM anomalies')
        conn.commit()
        conn.close()
