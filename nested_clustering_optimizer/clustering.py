"""
Nested Clustering Module

Implements hierarchical clustering at multiple levels for investment portfolios.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings


class NestedClusteringOptimizer:
    """
    Performs nested clustering on investment portfolios.
    
    Clusters assets at multiple levels (e.g., by sector, then by risk level,
    then by return characteristics) to create a hierarchical structure.
    """
    
    def __init__(
        self,
        n_clusters_level1: int = 3,
        n_clusters_level2: int = 2,
        clustering_method: str = "kmeans",
        random_state: Optional[int] = None
    ):
        """
        Initialize the nested clustering optimizer.
        
        Args:
            n_clusters_level1: Number of clusters at the first level
            n_clusters_level2: Number of clusters at the second level (within each level1 cluster)
            clustering_method: 'kmeans' or 'hierarchical'
            random_state: Random seed for reproducibility
        """
        self.n_clusters_level1 = n_clusters_level1
        self.n_clusters_level2 = n_clusters_level2
        self.clustering_method = clustering_method
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.level1_clusterer = None
        self.level2_clusterers = {}
        self.cluster_labels_level1 = None
        self.cluster_labels_level2 = {}
        self.feature_names = None
        
    def _get_clusterer(self, n_clusters: int):
        """Get a clusterer instance based on the method."""
        if self.clustering_method == "kmeans":
            return KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            )
        elif self.clustering_method == "hierarchical":
            return AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
    
    def fit(self, features: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit the nested clustering model.
        
        Args:
            features: Array of shape (n_assets, n_features) containing asset features
            feature_names: Optional list of feature names
        """
        if features.shape[0] < self.n_clusters_level1:
            raise ValueError(
                f"Not enough assets ({features.shape[0]}) for {self.n_clusters_level1} clusters"
            )
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(features.shape[1])]
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Level 1 clustering: Cluster all assets
        self.level1_clusterer = self._get_clusterer(self.n_clusters_level1)
        self.cluster_labels_level1 = self.level1_clusterer.fit_predict(features_scaled)
        
        # Level 2 clustering: Cluster within each level1 cluster
        self.cluster_labels_level2 = {}
        for cluster_id in range(self.n_clusters_level1):
            cluster_mask = self.cluster_labels_level1 == cluster_id
            cluster_features = features_scaled[cluster_mask]
            
            if len(cluster_features) < self.n_clusters_level2:
                # Not enough assets for level2 clustering
                self.cluster_labels_level2[cluster_id] = np.zeros(len(cluster_features), dtype=int)
                continue
            
            level2_clusterer = self._get_clusterer(self.n_clusters_level2)
            level2_labels = level2_clusterer.fit_predict(cluster_features)
            self.cluster_labels_level2[cluster_id] = level2_labels
            self.level2_clusterers[cluster_id] = level2_clusterer
        
        return self
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Predict cluster assignments for new assets.
        
        Args:
            features: Array of shape (n_assets, n_features)
            
        Returns:
            Tuple of (level1_labels, level2_labels_dict)
        """
        if self.level1_clusterer is None:
            raise ValueError("Model must be fitted before prediction")
        
        features_scaled = self.scaler.transform(features)
        
        # Level 1 prediction
        level1_labels = self.level1_clusterer.predict(features_scaled)
        
        # Level 2 prediction
        level2_labels = {}
        for cluster_id in range(self.n_clusters_level1):
            cluster_mask = level1_labels == cluster_id
            if not np.any(cluster_mask):
                continue
            
            cluster_features = features_scaled[cluster_mask]
            if cluster_id in self.level2_clusterers:
                level2_labels[cluster_id] = self.level2_clusterers[cluster_id].predict(cluster_features)
            else:
                level2_labels[cluster_id] = np.zeros(len(cluster_features), dtype=int)
        
        return level1_labels, level2_labels
    
    def get_cluster_hierarchy(self) -> Dict:
        """
        Get the complete cluster hierarchy structure.
        
        Returns:
            Dictionary mapping cluster IDs to asset indices
        """
        hierarchy = {}
        
        for level1_id in range(self.n_clusters_level1):
            level1_mask = self.cluster_labels_level1 == level1_id
            level1_indices = np.where(level1_mask)[0].tolist()
            
            hierarchy[level1_id] = {
                "assets": level1_indices,
                "subclusters": {}
            }
            
            if level1_id in self.cluster_labels_level2:
                level2_labels = self.cluster_labels_level2[level1_id]
                for level2_id in range(self.n_clusters_level2):
                    level2_mask = level2_labels == level2_id
                    level2_indices = np.where(level2_mask)[0].tolist()
                    # Map back to original indices
                    original_indices = [level1_indices[i] for i in level2_indices]
                    hierarchy[level1_id]["subclusters"][level2_id] = original_indices
        
        return hierarchy
    
    def evaluate_clustering(self, features: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the quality of the clustering.
        
        Args:
            features: Original feature array
            
        Returns:
            Dictionary with evaluation metrics
        """
        features_scaled = self.scaler.transform(features)
        
        metrics = {}
        
        # Level 1 silhouette score
        if len(np.unique(self.cluster_labels_level1)) > 1:
            metrics["level1_silhouette"] = silhouette_score(
                features_scaled,
                self.cluster_labels_level1
            )
        
        # Level 2 silhouette scores
        level2_scores = []
        for cluster_id in range(self.n_clusters_level1):
            cluster_mask = self.cluster_labels_level1 == cluster_id
            cluster_features = features_scaled[cluster_mask]
            
            if cluster_id in self.cluster_labels_level2:
                level2_labels = self.cluster_labels_level2[cluster_id]
                if len(np.unique(level2_labels)) > 1 and len(cluster_features) > 1:
                    score = silhouette_score(cluster_features, level2_labels)
                    level2_scores.append(score)
        
        if level2_scores:
            metrics["level2_silhouette_mean"] = np.mean(level2_scores)
            metrics["level2_silhouette_std"] = np.std(level2_scores)
        
        return metrics
