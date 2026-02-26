"""
Clustering for identifying operational modes in maritime data.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, List
import joblib


class OperationalModeClusterer:
    """
    Cluster maritime data into operational modes before anomaly detection.
    
    Different vessel states (at-rest, maneuvering, cruising) have different
    normal patterns. Clustering identifies these modes so SVD can be applied
    per-mode for better anomaly detection.
    """
    
    def __init__(
        self, 
        n_clusters: int = 4,
        random_state: int = 42
    ):
        """
        Initialize clusterer.
        
        Args:
            n_clusters: Number of operational modes to identify
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        self.kmeans: Optional[KMeans] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.cluster_centers_original: Optional[np.ndarray] = None
        self.cluster_labels_: Optional[np.ndarray] = None
        self.cluster_names: Optional[dict] = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, feature_names: List[str] = None) -> 'OperationalModeClusterer':
        """
        Fit clustering model to identify operational modes.
        
        Args:
            X: Feature matrix (unscaled)
            feature_names: Names of features for interpretation
            
        Returns:
            self
        """
        # Scale features for clustering
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.cluster_labels_ = self.kmeans.fit_predict(X_scaled)
        
        # Store cluster centers in original scale for interpretation
        self.cluster_centers_original = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        
        # Auto-name clusters based on characteristics
        self.cluster_names = self._name_clusters(feature_names)
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster/mode for new data.
        
        Args:
            X: Feature matrix (unscaled)
            
        Returns:
            Cluster labels
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)
    
    def fit_predict(self, X: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X, feature_names)
        return self.cluster_labels_
    
    def _name_clusters(self, feature_names: List[str] = None) -> dict:
        """
        Auto-generate descriptive names for clusters based on their characteristics.
        Uses speed, power, and draft to distinguish operational modes.
        """
        names = {}
        centers = self.cluster_centers_original
        
        # Try to identify relevant feature indices
        speed_idx = None
        power_idx = None
        draft_idx = None
        
        if feature_names:
            for i, name in enumerate(feature_names):
                name_lower = name.lower()
                if 'speed' in name_lower and 'gps' in name_lower:
                    speed_idx = i
                elif 'power' in name_lower and 'engine' in name_lower:
                    power_idx = i
                elif 'draft' in name_lower and 'avg' in name_lower:
                    draft_idx = i
        
        for cluster_id in range(self.n_clusters):
            center = centers[cluster_id]
            
            if speed_idx is not None and power_idx is not None:
                speed = center[speed_idx]
                power = center[power_idx]
                draft = center[draft_idx] if draft_idx is not None else None
                
                # Classify based on speed, power, and draft
                if speed < 0.5 and power < 100:
                    # At rest - distinguish by draft (loading state)
                    if draft is not None:
                        if draft > 6.5:
                            names[cluster_id] = f"Mode_{cluster_id}_AtRest_Loaded"
                        else:
                            names[cluster_id] = f"Mode_{cluster_id}_AtRest_Unloaded"
                    else:
                        names[cluster_id] = f"Mode_{cluster_id}_AtRest"
                elif speed < 3 and power < 500:
                    names[cluster_id] = f"Mode_{cluster_id}_SlowSpeed"
                elif speed < 10:
                    names[cluster_id] = f"Mode_{cluster_id}_MediumSpeed"
                else:
                    names[cluster_id] = f"Mode_{cluster_id}_HighSpeed"
            else:
                names[cluster_id] = f"Mode_{cluster_id}"
        
        return names
    
    def get_cluster_summary(self, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Get summary statistics for each cluster.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        summary_data = []
        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels_ == cluster_id
            count = mask.sum()
            pct = count / len(self.cluster_labels_) * 100
            
            row = {
                'cluster_id': cluster_id,
                'name': self.cluster_names.get(cluster_id, f"Mode_{cluster_id}"),
                'count': count,
                'percentage': pct,
            }
            
            # Add center values
            for i, val in enumerate(self.cluster_centers_original[cluster_id]):
                col_name = feature_names[i] if feature_names and i < len(feature_names) else f'feature_{i}'
                row[f'center_{col_name}'] = val
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def save(self, filepath: str) -> None:
        """Save clusterer to disk."""
        joblib.dump({
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'cluster_centers_original': self.cluster_centers_original,
            'cluster_names': self.cluster_names,
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'OperationalModeClusterer':
        """Load clusterer from disk."""
        data = joblib.load(filepath)
        clusterer = cls(
            n_clusters=data['n_clusters'],
            random_state=data['random_state']
        )
        clusterer.kmeans = data['kmeans']
        clusterer.scaler = data['scaler']
        clusterer.cluster_centers_original = data['cluster_centers_original']
        clusterer.cluster_names = data['cluster_names']
        clusterer._is_fitted = True
        return clusterer


class GMMOperationalModeClusterer:
    """
    Gaussian Mixture Model (GMM) clustering for operational modes.
    
    GMM advantages over K-Means:
    - Soft clustering: Provides probability of belonging to each cluster
    - Elliptical clusters: Can capture correlated features (K-Means assumes spherical)
    - Better uncertainty quantification: Points near boundaries get lower confidence
    
    Good for maritime data where operational modes may overlap (e.g., transition states).
    """
    
    def __init__(
        self, 
        n_clusters: int = 4,
        covariance_type: str = 'full',
        random_state: int = 42
    ):
        """
        Initialize GMM clusterer.
        
        Args:
            n_clusters: Number of operational modes (Gaussian components)
            covariance_type: 'full', 'tied', 'diag', or 'spherical'
                - 'full': Each component has its own covariance matrix (most flexible)
                - 'diag': Diagonal covariance (assumes feature independence)
                - 'spherical': Like K-Means (spherical clusters)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.covariance_type = covariance_type
        self.random_state = random_state
        
        self.gmm: Optional[GaussianMixture] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.cluster_centers_original: Optional[np.ndarray] = None
        self.cluster_labels_: Optional[np.ndarray] = None
        self.cluster_probabilities_: Optional[np.ndarray] = None
        self.cluster_names: Optional[dict] = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, feature_names: List[str] = None) -> 'GMMOperationalModeClusterer':
        """
        Fit GMM to identify operational modes.
        
        Args:
            X: Feature matrix (unscaled)
            feature_names: Names of features for interpretation
            
        Returns:
            self
        """
        # Scale features for clustering
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=5,
            max_iter=200
        )
        self.cluster_labels_ = self.gmm.fit_predict(X_scaled)
        self.cluster_probabilities_ = self.gmm.predict_proba(X_scaled)
        
        # Store cluster centers (means) in original scale
        self.cluster_centers_original = self.scaler.inverse_transform(self.gmm.means_)
        
        # Auto-name clusters based on characteristics
        self.cluster_names = self._name_clusters(feature_names)
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster/mode for new data (hard assignment).
        
        Args:
            X: Feature matrix (unscaled)
            
        Returns:
            Cluster labels
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.gmm.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of each cluster for new data (soft assignment).
        
        Args:
            X: Feature matrix (unscaled)
            
        Returns:
            Probability matrix (n_samples, n_clusters)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.gmm.predict_proba(X_scaled)
    
    def fit_predict(self, X: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X, feature_names)
        return self.cluster_labels_
    
    def _name_clusters(self, feature_names: List[str] = None) -> dict:
        """
        Auto-generate descriptive names for clusters based on their characteristics.
        Uses speed, power, and draft to distinguish operational modes.
        """
        names = {}
        centers = self.cluster_centers_original
        
        # Try to identify relevant feature indices
        speed_idx = None
        power_idx = None
        draft_idx = None
        
        if feature_names:
            for i, name in enumerate(feature_names):
                name_lower = name.lower()
                if 'speed' in name_lower and 'gps' in name_lower:
                    speed_idx = i
                elif 'power' in name_lower and 'engine' in name_lower:
                    power_idx = i
                elif 'draft' in name_lower and 'avg' in name_lower:
                    draft_idx = i
        
        for cluster_id in range(self.n_clusters):
            center = centers[cluster_id]
            
            if speed_idx is not None and power_idx is not None:
                speed = center[speed_idx]
                power = center[power_idx]
                draft = center[draft_idx] if draft_idx is not None else None
                
                # Classify based on speed, power, and draft
                if speed < 0.5 and power < 100:
                    if draft is not None:
                        if draft > 6.5:
                            names[cluster_id] = f"GMM_{cluster_id}_AtRest_Loaded"
                        else:
                            names[cluster_id] = f"GMM_{cluster_id}_AtRest_Unloaded"
                    else:
                        names[cluster_id] = f"GMM_{cluster_id}_AtRest"
                elif speed < 3 and power < 500:
                    names[cluster_id] = f"GMM_{cluster_id}_SlowSpeed"
                elif speed < 10:
                    names[cluster_id] = f"GMM_{cluster_id}_MediumSpeed"
                else:
                    names[cluster_id] = f"GMM_{cluster_id}_HighSpeed"
            else:
                names[cluster_id] = f"GMM_{cluster_id}"
        
        return names
    
    def get_cluster_summary(self, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Get summary statistics for each cluster.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        summary_data = []
        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels_ == cluster_id
            count = mask.sum()
            pct = count / len(self.cluster_labels_) * 100
            
            # Average probability for points assigned to this cluster
            avg_prob = self.cluster_probabilities_[mask, cluster_id].mean() if count > 0 else 0
            
            row = {
                'cluster_id': cluster_id,
                'name': self.cluster_names.get(cluster_id, f"GMM_{cluster_id}"),
                'count': count,
                'percentage': pct,
                'avg_probability': avg_prob,
            }
            
            # Add center values
            for i, val in enumerate(self.cluster_centers_original[cluster_id]):
                col_name = feature_names[i] if feature_names and i < len(feature_names) else f'feature_{i}'
                row[f'center_{col_name}'] = val
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def get_model_scores(self) -> dict:
        """Get GMM model quality metrics."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return {
            'bic': self.gmm.bic(self.scaler.transform(
                self.scaler.inverse_transform(self.gmm.means_)  # Just use means for quick check
            )) if hasattr(self.gmm, 'bic') else None,
            'aic': self.gmm.aic(self.scaler.transform(
                self.scaler.inverse_transform(self.gmm.means_)
            )) if hasattr(self.gmm, 'aic') else None,
            'converged': self.gmm.converged_,
            'n_iter': self.gmm.n_iter_,
        }
    
    def save(self, filepath: str) -> None:
        """Save clusterer to disk."""
        joblib.dump({
            'gmm': self.gmm,
            'scaler': self.scaler,
            'n_clusters': self.n_clusters,
            'covariance_type': self.covariance_type,
            'random_state': self.random_state,
            'cluster_centers_original': self.cluster_centers_original,
            'cluster_names': self.cluster_names,
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'GMMOperationalModeClusterer':
        """Load clusterer from disk."""
        data = joblib.load(filepath)
        clusterer = cls(
            n_clusters=data['n_clusters'],
            covariance_type=data['covariance_type'],
            random_state=data['random_state']
        )
        clusterer.gmm = data['gmm']
        clusterer.scaler = data['scaler']
        clusterer.cluster_centers_original = data['cluster_centers_original']
        clusterer.cluster_names = data['cluster_names']
        clusterer._is_fitted = True
        return clusterer
