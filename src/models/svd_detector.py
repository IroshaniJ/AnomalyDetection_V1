"""
SVD-based anomaly detection using reconstruction error.
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD
from typing import Tuple, Optional
import joblib


class SVDAnomalyDetector:
    """
    Anomaly detector using SVD reconstruction error.
    
    Anomalies are detected by:
    1. Projecting data to lower-dimensional space via SVD
    2. Reconstructing back to original space
    3. Computing reconstruction error (MSE per sample)
    4. Flagging points with error above threshold
    """
    
    def __init__(
        self, 
        n_components: int = 5, 
        threshold_percentile: float = 95.0,
        random_state: int = 42
    ):
        """
        Initialize SVD anomaly detector.
        
        Args:
            n_components: Number of SVD components to retain
            threshold_percentile: Percentile for anomaly threshold (e.g., 95 = top 5% are anomalies)
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        
        self.svd: Optional[TruncatedSVD] = None
        self.threshold: Optional[float] = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'SVDAnomalyDetector':
        """
        Fit the SVD model and compute anomaly threshold.
        
        Args:
            X: Training data (already scaled), shape (n_samples, n_features)
            
        Returns:
            self
        """
        # Fit SVD
        self.svd = TruncatedSVD(
            n_components=self.n_components, 
            random_state=self.random_state
        )
        X_reduced = self.svd.fit_transform(X)
        
        # Compute reconstruction error on training data
        X_reconstructed = self.svd.inverse_transform(X_reduced)
        reconstruction_errors = np.sum((X - X_reconstructed) ** 2, axis=1)
        
        # Set threshold based on percentile
        self.threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
        self._is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            X: Data to predict (already scaled), shape (n_samples, n_features)
            
        Returns:
            Array of predictions: 1 = normal, -1 = anomaly
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        scores = self.score_samples(X)
        predictions = np.where(scores > self.threshold, -1, 1)
        return predictions
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores (reconstruction error).
        
        Args:
            X: Data to score (already scaled), shape (n_samples, n_features)
            
        Returns:
            Array of reconstruction errors (higher = more anomalous)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_reduced = self.svd.transform(X)
        X_reconstructed = self.svd.inverse_transform(X_reduced)
        reconstruction_errors = np.sum((X - X_reconstructed) ** 2, axis=1)
        
        return reconstruction_errors
    
    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit model and predict anomalies in one step.
        
        Args:
            X: Data to fit and predict (already scaled)
            
        Returns:
            Tuple of (predictions, scores)
        """
        self.fit(X)
        scores = self.score_samples(X)
        predictions = np.where(scores > self.threshold, -1, 1)
        return predictions, scores
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each component."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.svd.explained_variance_ratio_
    
    def save(self, filepath: str) -> None:
        """Save model to disk."""
        joblib.dump({
            'svd': self.svd,
            'threshold': self.threshold,
            'n_components': self.n_components,
            'threshold_percentile': self.threshold_percentile,
            'random_state': self.random_state,
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'SVDAnomalyDetector':
        """Load model from disk."""
        data = joblib.load(filepath)
        detector = cls(
            n_components=data['n_components'],
            threshold_percentile=data['threshold_percentile'],
            random_state=data['random_state']
        )
        detector.svd = data['svd']
        detector.threshold = data['threshold']
        detector._is_fitted = True
        return detector
