"""
Example script demonstrating usage of the nested clustering optimizer.
"""

import numpy as np
from nested_clustering_optimizer.main import run_optimization

if __name__ == "__main__":
    # Run with sample data
    print("Running nested clustering optimizer with sample data...")
    print()
    
    results = run_optimization(
        n_clusters_level1=3,
        n_clusters_level2=2,
        clustering_method="kmeans",
        allocation_strategy="equal_weight_clusters",
        use_sample_data=True,
        random_state=42
    )
    
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    print(f"Number of assets: {len(results['optimal_weights'])}")
    print(f"Number of clusters (Level 1): {len(results['cluster_hierarchy'])}")
    print(f"Portfolio Sharpe Ratio: {results['optimization_info']['sharpe_ratio']:.4f}")
    print(f"Clustering Quality (Level 1 Silhouette): {results['clustering_metrics'].get('level1_silhouette', 'N/A')}")
