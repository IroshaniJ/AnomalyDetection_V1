"""
Clustered Autoencoder Anomaly Detector.

Applies K-Means clustering first, then fits Autoencoder per cluster.
Uses reconstruction error as anomaly score.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Optional, Tuple
import warnings

from .base_detector import BaseAnomalyDetector

# Try to import PyTorch, fall back to sklearn MLPRegressor
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    
from sklearn.neural_network import MLPRegressor


class PyTorchAutoencoder(nn.Module):
    """PyTorch Autoencoder model."""
    
    def __init__(self, input_dim: int, encoding_dim: int, hidden_layers: Tuple[int, ...]):
        super().__init__()
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for units in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, units))
            encoder_layers.append(nn.ReLU())
            prev_dim = units
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = encoding_dim
        for units in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, units))
            decoder_layers.append(nn.ReLU())
            prev_dim = units
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ClusteredAutoencoder(BaseAnomalyDetector):
    """
    Autoencoder with per-cluster fitting.
    
    Approach:
    1. Cluster data into operational modes using K-Means
    2. Fit separate Autoencoder for each cluster
    3. Anomaly scores are based on reconstruction error
    """
    
    def __init__(
        self,
        n_clusters: int = 4,
        threshold_percentile: float = 95.0,
        encoding_dim: int = 3,
        hidden_layers: Tuple[int, ...] = (16, 8),
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        random_state: int = 42
    ):
        """
        Initialize Clustered Autoencoder.
        
        Args:
            n_clusters: Number of operational mode clusters
            threshold_percentile: Percentile for anomaly threshold
            encoding_dim: Dimension of the encoding layer (bottleneck)
            hidden_layers: Tuple of hidden layer sizes for encoder
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            random_state: Random seed for reproducibility
        """
        super().__init__(n_clusters, threshold_percentile)
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Per-cluster components
        self.kmeans_ = None
        self.cluster_scalers_ = {}
        self.cluster_autoencoders_ = {}
        self.input_dim_ = None
        
        # Set random seeds
        np.random.seed(random_state)
        if HAS_PYTORCH:
            torch.manual_seed(random_state)
        
    @property
    def name(self) -> str:
        return "Autoencoder"
    
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
            avg_power = cluster_data[:, 1].mean() if cluster_data.shape[1] > 1 else 0
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
    
    def _build_autoencoder_pytorch(self, input_dim: int) -> PyTorchAutoencoder:
        """Build PyTorch autoencoder model."""
        return PyTorchAutoencoder(input_dim, self.encoding_dim, self.hidden_layers)
    
    def _train_pytorch_autoencoder(self, model: PyTorchAutoencoder, X: np.ndarray) -> PyTorchAutoencoder:
        """Train PyTorch autoencoder."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        model.train()
        for epoch in range(self.epochs):
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
        
        model.eval()
        return model
    
    def _predict_pytorch(self, model: PyTorchAutoencoder, X: np.ndarray) -> np.ndarray:
        """Get reconstruction from PyTorch model."""
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            output = model(X_tensor)
            return output.cpu().numpy()
    
    def _build_autoencoder_sklearn(self, input_dim: int) -> MLPRegressor:
        """Build sklearn MLP as fallback autoencoder."""
        # Symmetric architecture: input -> hidden -> encoding -> hidden -> output
        hidden_sizes = list(self.hidden_layers) + [self.encoding_dim] + list(reversed(self.hidden_layers))
        
        return MLPRegressor(
            hidden_layer_sizes=tuple(hidden_sizes),
            activation='relu',
            solver='adam',
            max_iter=self.epochs * 10,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
    
    def fit(self, X: np.ndarray) -> 'ClusteredAutoencoder':
        """
        Fit the clustered Autoencoder.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            self
        """
        self.input_dim_ = X.shape[1]
        
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
        
        # Step 2: Fit Autoencoder per cluster
        backend = "PyTorch" if HAS_PYTORCH else "sklearn"
        print(f"      [Autoencoder] Fitting detector per cluster (encoding_dim={self.encoding_dim}, backend={backend})...")
        
        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels_ == cluster_id
            X_cluster = X[mask]
            
            if len(X_cluster) < 50:
                print(f"        - {self.cluster_names_[cluster_id]}: Too few samples, skipping")
                continue
            
            # Scale within cluster
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_cluster)
            self.cluster_scalers_[cluster_id] = scaler
            
            # Build and train autoencoder
            if HAS_PYTORCH:
                autoencoder = self._build_autoencoder_pytorch(self.input_dim_)
                autoencoder = self._train_pytorch_autoencoder(autoencoder, X_scaled)
                X_reconstructed = self._predict_pytorch(autoencoder, X_scaled)
            else:
                autoencoder = self._build_autoencoder_sklearn(self.input_dim_)
                autoencoder.fit(X_scaled, X_scaled)
                X_reconstructed = autoencoder.predict(X_scaled)
            
            self.cluster_autoencoders_[cluster_id] = autoencoder
            
            # Calculate reconstruction error and threshold
            errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
            threshold = np.percentile(errors, self.threshold_percentile)
            self.cluster_thresholds_[cluster_id] = threshold
            
            print(f"        - {self.cluster_names_[cluster_id]}: threshold={threshold:.6f}")
        
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
        """Return anomaly scores (reconstruction error, higher = more anomalous)."""
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted before scoring")
        
        # Get cluster assignments
        cluster_scaler = MinMaxScaler()
        cluster_scaler.fit(X)
        X_scaled_cluster = cluster_scaler.transform(X)
        cluster_labels = self.kmeans_.predict(X_scaled_cluster)
        self.cluster_labels_ = cluster_labels
        
        scores = np.zeros(len(X))
        
        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            if mask.sum() == 0 or cluster_id not in self.cluster_autoencoders_:
                continue
            
            X_cluster = X[mask]
            scaler = self.cluster_scalers_[cluster_id]
            X_scaled = scaler.transform(X_cluster)
            
            autoencoder = self.cluster_autoencoders_[cluster_id]
            
            if HAS_PYTORCH:
                X_reconstructed = self._predict_pytorch(autoencoder, X_scaled)
            else:
                X_reconstructed = autoencoder.predict(X_scaled)
            
            # Reconstruction error (MSE per sample)
            errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
            scores[mask] = errors
        
        return scores
    
    def get_params(self) -> Dict:
        """Get detector parameters."""
        params = super().get_params()
        params.update({
            'encoding_dim': self.encoding_dim,
            'hidden_layers': self.hidden_layers,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'backend': 'pytorch' if HAS_PYTORCH else 'sklearn',
        })
        return params
