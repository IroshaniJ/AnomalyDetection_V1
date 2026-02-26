"""
Base class for all anomaly detection methods.
All benchmark detectors should inherit from this class.
"""

from abc import ABC, abstractmethod
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional


class BaseAnomalyDetector(ABC):
    """Abstract base class for anomaly detectors."""
    
    def __init__(self, n_clusters: int = 4, threshold_percentile: float = 95.0):
        """
        Initialize detector.
        
        Args:
            n_clusters: Number of operational mode clusters
            threshold_percentile: Percentile for anomaly threshold (95 = top 5% flagged)
        """
        self.n_clusters = n_clusters
        self.threshold_percentile = threshold_percentile
        self.cluster_labels_ = None
        self.cluster_names_ = {}
        self.cluster_thresholds_ = {}
        self.is_fitted_ = False
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the detector."""
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseAnomalyDetector':
        """
        Fit the detector on training data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Data of shape (n_samples, n_features)
            
        Returns:
            Array of predictions: -1 for anomalies, 1 for normal
        """
        pass
    
    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores for samples.
        Higher scores indicate more anomalous samples.
        
        Args:
            X: Data of shape (n_samples, n_features)
            
        Returns:
            Array of anomaly scores
        """
        pass
    
    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit detector and return predictions.
        
        Args:
            X: Training data
            
        Returns:
            Tuple of (predictions, scores, cluster_labels)
        """
        self.fit(X)
        predictions = self.predict(X)
        scores = self.score_samples(X)
        cluster_labels = self.get_cluster_labels(X)
        return predictions, scores, cluster_labels
    
    def get_cluster_labels(self, X: np.ndarray) -> np.ndarray:
        """Get cluster labels for samples."""
        if self.cluster_labels_ is not None and len(self.cluster_labels_) == len(X):
            return self.cluster_labels_
        return np.zeros(len(X), dtype=int)
    
    def get_cluster_names(self) -> Dict[int, str]:
        """Get mapping of cluster IDs to names."""
        return self.cluster_names_
    
    def get_cluster_thresholds(self) -> Dict[int, float]:
        """Get per-cluster anomaly thresholds."""
        return self.cluster_thresholds_
    
    def get_params(self) -> Dict:
        """Get detector parameters."""
        return {
            'name': self.name,
            'n_clusters': self.n_clusters,
            'threshold_percentile': self.threshold_percentile,
        }
    
    def save(self, path: str) -> None:
        """Save detector to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"      Model saved: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BaseAnomalyDetector':
        """Load detector from file."""
        return joblib.load(path)
    
    def get_summary(self) -> Dict:
        """Get summary statistics for the fitted detector."""
        return {
            'name': self.name,
            'n_clusters': self.n_clusters,
            'threshold_percentile': self.threshold_percentile,
            'cluster_thresholds': self.cluster_thresholds_,
            'is_fitted': self.is_fitted_,
        }
