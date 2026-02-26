"""
Clustered SVD anomaly detection - applies SVD per operational mode.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Optional, Tuple
import joblib

from .clustering import OperationalModeClusterer
from .svd_detector import SVDAnomalyDetector


class ClusteredSVDAnomalyDetector:
    """
    Anomaly detector that first clusters data into operational modes,
    then applies SVD-based detection within each cluster.
    
    This approach captures mode-specific normal behavior:
    - At-rest: Low speed, no power
    - Maneuvering: Variable speed/power
    - Cruising: High speed, steady power
    
    Anomalies are points that don't fit their cluster's normal pattern.
    """
    
    def __init__(
        self,
        n_clusters: int = 4,
        n_components: int = 3,
        threshold_percentile: float = 95.0,
        random_state: int = 42
    ):
        """
        Initialize clustered SVD detector.
        
        Args:
            n_clusters: Number of operational modes
            n_components: SVD components per cluster
            threshold_percentile: Anomaly threshold percentile per cluster
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        
        self.clusterer: Optional[OperationalModeClusterer] = None
        self.scalers: Dict[int, MinMaxScaler] = {}
        self.detectors: Dict[int, SVDAnomalyDetector] = {}
        self.cluster_thresholds: Dict[int, float] = {}
        self.feature_names: Optional[List[str]] = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, feature_names: List[str] = None) -> 'ClusteredSVDAnomalyDetector':
        """
        Fit clustering and per-cluster SVD detectors.
        
        Args:
            X: Feature matrix (unscaled)
            feature_names: Names of features
            
        Returns:
            self
        """
        self.feature_names = feature_names
        
        # Step 1: Cluster data into operational modes
        print(f"      [Clustering] Identifying {self.n_clusters} operational modes...")
        self.clusterer = OperationalModeClusterer(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )
        cluster_labels = self.clusterer.fit_predict(X, feature_names)
        
        # Print cluster summary
        summary = self.clusterer.get_cluster_summary(feature_names)
        for _, row in summary.iterrows():
            print(f"        - {row['name']}: {row['count']:,} samples ({row['percentage']:.1f}%)")
        
        # Step 2: Fit SVD detector per cluster
        print(f"      [SVD] Fitting detector per cluster (components={self.n_components})...")
        
        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            X_cluster = X[mask]
            
            if len(X_cluster) < self.n_components + 1:
                print(f"        - Cluster {cluster_id}: Too few samples, skipping SVD")
                continue
            
            # Scale within cluster
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_cluster)
            self.scalers[cluster_id] = scaler
            
            # Fit SVD detector
            detector = SVDAnomalyDetector(
                n_components=min(self.n_components, X_scaled.shape[1] - 1),
                threshold_percentile=self.threshold_percentile,
                random_state=self.random_state
            )
            detector.fit(X_scaled)
            self.detectors[cluster_id] = detector
            self.cluster_thresholds[cluster_id] = detector.threshold
            
            cluster_name = self.clusterer.cluster_names.get(cluster_id, f"Cluster_{cluster_id}")
            print(f"        - {cluster_name}: threshold={detector.threshold:.6f}, "
                  f"explained_var={detector.get_explained_variance_ratio().sum():.1%}")
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using cluster-specific SVD.
        
        Args:
            X: Feature matrix (unscaled)
            
        Returns:
            Array of predictions: 1 = normal, -1 = anomaly
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        scores = self.score_samples(X)
        
        # Get per-sample threshold based on cluster assignment
        cluster_labels = self.clusterer.predict(X)
        thresholds = np.array([
            self.cluster_thresholds.get(c, np.inf) for c in cluster_labels
        ])
        
        predictions = np.where(scores > thresholds, -1, 1)
        return predictions
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores using cluster-specific SVD.
        
        Args:
            X: Feature matrix (unscaled)
            
        Returns:
            Array of anomaly scores (reconstruction errors)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Assign to clusters
        cluster_labels = self.clusterer.predict(X)
        
        # Compute scores per cluster
        scores = np.zeros(len(X))
        
        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            if not mask.any():
                continue
            
            if cluster_id not in self.detectors:
                # No detector for this cluster, assign high score
                scores[mask] = np.inf
                continue
            
            X_cluster = X[mask]
            X_scaled = self.scalers[cluster_id].transform(X_cluster)
            scores[mask] = self.detectors[cluster_id].score_samples(X_scaled)
        
        return scores
    
    def fit_predict(self, X: np.ndarray, feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit and predict in one step.
        
        Returns:
            Tuple of (predictions, scores, cluster_labels)
        """
        self.fit(X, feature_names)
        cluster_labels = self.clusterer.cluster_labels_
        scores = self.score_samples(X)
        predictions = self.predict(X)
        return predictions, scores, cluster_labels
    
    def get_cluster_labels(self, X: np.ndarray) -> np.ndarray:
        """Get cluster assignments for data."""
        return self.clusterer.predict(X)
    
    def get_cluster_names(self) -> Dict[int, str]:
        """Get cluster ID to name mapping."""
        return self.clusterer.cluster_names
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """Get cluster summary with thresholds."""
        summary = self.clusterer.get_cluster_summary(self.feature_names)
        summary['svd_threshold'] = summary['cluster_id'].map(self.cluster_thresholds)
        return summary
    
    def save(self, filepath: str) -> None:
        """Save complete model to disk."""
        joblib.dump({
            'clusterer': self.clusterer,
            'scalers': self.scalers,
            'detectors': self.detectors,
            'cluster_thresholds': self.cluster_thresholds,
            'n_clusters': self.n_clusters,
            'n_components': self.n_components,
            'threshold_percentile': self.threshold_percentile,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'ClusteredSVDAnomalyDetector':
        """Load model from disk."""
        data = joblib.load(filepath)
        detector = cls(
            n_clusters=data['n_clusters'],
            n_components=data['n_components'],
            threshold_percentile=data['threshold_percentile'],
            random_state=data['random_state']
        )
        detector.clusterer = data['clusterer']
        detector.scalers = data['scalers']
        detector.detectors = data['detectors']
        detector.cluster_thresholds = data['cluster_thresholds']
        detector.feature_names = data['feature_names']
        detector._is_fitted = True
        return detector
