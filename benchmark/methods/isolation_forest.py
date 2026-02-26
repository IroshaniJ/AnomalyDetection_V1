"""
Clustered Isolation Forest Anomaly Detector.

Applies K-Means clustering first, then fits Isolation Forest per cluster.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Optional

from .base_detector import BaseAnomalyDetector


class ClusteredIsolationForest(BaseAnomalyDetector):
    """
    Isolation Forest with per-cluster fitting.
    
    Approach:
    1. Cluster data into operational modes using K-Means
    2. Fit separate Isolation Forest for each cluster
    3. Anomaly scores are based on isolation path length
    """
    
    def __init__(
        self,
        n_clusters: int = 4,
        threshold_percentile: float = 95.0,
        n_estimators: int = 100,
        max_samples: str = 'auto',
        contamination: float = 0.05,
        random_state: int = 42
    ):
        """
        Initialize Clustered Isolation Forest.
        
        Args:
            n_clusters: Number of operational mode clusters
            threshold_percentile: Percentile for anomaly threshold
            n_estimators: Number of trees in the forest
            max_samples: Number of samples to draw for each tree
            contamination: Expected proportion of outliers
            random_state: Random seed for reproducibility
        """
        super().__init__(n_clusters, threshold_percentile)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        
        # Per-cluster components
        self.kmeans_ = None
        self.cluster_scalers_ = {}
        self.cluster_detectors_ = {}
        
    @property
    def name(self) -> str:
        return "Isolation Forest"
    
    def _name_clusters(self, X: np.ndarray, labels: np.ndarray) -> Dict[int, str]:
        """Assign descriptive names to clusters based on characteristics."""
        names = {}
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            cluster_data = X[mask]
            
            if len(cluster_data) == 0:
                names[cluster_id] = f"Cluster_{cluster_id}"
                continue
            
            # Use speed (col 0), power (col 1), and draft (col 4) to name clusters
            avg_speed = cluster_data[:, 0].mean()
            avg_power = cluster_data[:, 1].mean()
            avg_draft = cluster_data[:, 4].mean() if cluster_data.shape[1] > 4 else None
            
            if avg_speed < 0.5 and avg_power < 100:
                # Distinguish loaded vs unloaded at-rest states by draft
                if avg_draft is not None and avg_draft > 6.5:
                    names[cluster_id] = f"Mode_{cluster_id}_AtRest_Loaded"
                elif avg_draft is not None:
                    names[cluster_id] = f"Mode_{cluster_id}_AtRest_Unloaded"
                else:
                    names[cluster_id] = f"Mode_{cluster_id}_AtRest"
            elif avg_speed < 5:
                names[cluster_id] = f"Mode_{cluster_id}_SlowSpeed"
            elif avg_speed < 10:
                names[cluster_id] = f"Mode_{cluster_id}_MediumSpeed"
            else:
                names[cluster_id] = f"Mode_{cluster_id}_HighSpeed"
        
        return names
    
    def fit(self, X: np.ndarray) -> 'ClusteredIsolationForest':
        """
        Fit the clustered Isolation Forest.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            self
        """
        print(f"      [Clustering] Identifying {self.n_clusters} operational modes...")
        
        # Step 1: Cluster into operational modes
        cluster_scaler = MinMaxScaler()
        X_scaled_cluster = cluster_scaler.fit_transform(X)
        
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.cluster_labels_ = self.kmeans_.fit_predict(X_scaled_cluster)
        self.cluster_names_ = self._name_clusters(X, self.cluster_labels_)
        
        # Print cluster distribution
        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels_ == cluster_id
            count = mask.sum()
            pct = count / len(X) * 100
            print(f"        - {self.cluster_names_[cluster_id]}: {count:,} samples ({pct:.1f}%)")
        
        # Step 2: Fit Isolation Forest per cluster
        print(f"      [IForest] Fitting detector per cluster (n_estimators={self.n_estimators})...")
        
        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels_ == cluster_id
            X_cluster = X[mask]
            
            if len(X_cluster) < 10:
                print(f"        - {self.cluster_names_[cluster_id]}: Too few samples, skipping")
                continue
            
            # Scale within cluster
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_cluster)
            self.cluster_scalers_[cluster_id] = scaler
            
            # Fit Isolation Forest
            detector = IsolationForest(
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1
            )
            detector.fit(X_scaled)
            self.cluster_detectors_[cluster_id] = detector
            
            # Calculate threshold based on anomaly scores
            scores = -detector.score_samples(X_scaled)  # Negate so higher = more anomalous
            threshold = np.percentile(scores, self.threshold_percentile)
            self.cluster_thresholds_[cluster_id] = threshold
            
            print(f"        - {self.cluster_names_[cluster_id]}: threshold={threshold:.4f}")
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (-1 = anomaly, 1 = normal)."""
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted before prediction")
        
        scores = self.score_samples(X)
        cluster_labels = self.get_cluster_labels(X)
        
        predictions = np.ones(len(X), dtype=int)
        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            if cluster_id in self.cluster_thresholds_:
                threshold = self.cluster_thresholds_[cluster_id]
                predictions[mask] = np.where(scores[mask] > threshold, -1, 1)
        
        return predictions
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted before scoring")
        
        # Get cluster assignments
        cluster_scaler = MinMaxScaler()
        cluster_scaler.fit(X)  # Refit for consistency
        X_scaled_cluster = cluster_scaler.transform(X)
        cluster_labels = self.kmeans_.predict(X_scaled_cluster)
        self.cluster_labels_ = cluster_labels
        
        scores = np.zeros(len(X))
        
        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            if mask.sum() == 0 or cluster_id not in self.cluster_detectors_:
                continue
            
            X_cluster = X[mask]
            scaler = self.cluster_scalers_[cluster_id]
            X_scaled = scaler.transform(X_cluster)
            
            detector = self.cluster_detectors_[cluster_id]
            # Negate scores so higher = more anomalous
            scores[mask] = -detector.score_samples(X_scaled)
        
        return scores
    
    def get_params(self) -> Dict:
        """Get detector parameters."""
        params = super().get_params()
        params.update({
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'contamination': self.contamination,
        })
        return params
