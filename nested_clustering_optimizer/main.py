"""
Main Application Entry Point

Example usage of the nested clustering optimizer for investment portfolios.
"""

import numpy as np
from typing import Optional
import argparse
import sys

from .clustering import NestedClusteringOptimizer
from .portfolio_optimizer import PortfolioOptimizer
from .data_handler import PortfolioDataHandler


def generate_sample_data(
    n_assets: int = 50,
    n_periods: int = 252,
    n_clusters: int = 3,
    random_state: Optional[int] = 42
) -> tuple:
    """
    Generate sample portfolio data for demonstration.
    
    Args:
        n_assets: Number of assets
        n_periods: Number of time periods
        n_clusters: Number of underlying clusters (for generating correlated data)
        random_state: Random seed
        
    Returns:
        Tuple of (prices, asset_names, feature_names)
    """
    np.random.seed(random_state)
    
    # Generate correlated returns based on clusters
    returns = np.zeros((n_periods, n_assets))
    assets_per_cluster = n_assets // n_clusters
    
    for i in range(n_clusters):
        start_idx = i * assets_per_cluster
        end_idx = start_idx + assets_per_cluster if i < n_clusters - 1 else n_assets
        
        # Generate cluster-specific factor
        cluster_factor = np.random.randn(n_periods, 1)
        
        # Generate returns with cluster correlation
        cluster_returns = 0.01 * cluster_factor + 0.02 * np.random.randn(n_periods, end_idx - start_idx)
        returns[:, start_idx:end_idx] = cluster_returns
    
    # Convert returns to prices
    initial_prices = 100 * np.ones(n_assets)
    prices = np.zeros((n_periods + 1, n_assets))
    prices[0] = initial_prices
    
    for t in range(1, n_periods + 1):
        prices[t] = prices[t-1] * (1 + returns[t-1])
    
    asset_names = [f"Asset_{i:03d}" for i in range(n_assets)]
    feature_names = ["mean_return", "volatility", "sharpe_ratio", "skewness", "kurtosis", "max_drawdown"]
    
    return prices, asset_names, feature_names


def run_optimization(
    prices: Optional[np.ndarray] = None,
    asset_names: Optional[list] = None,
    n_clusters_level1: int = 3,
    n_clusters_level2: int = 2,
    clustering_method: str = "kmeans",
    allocation_strategy: str = "equal_weight_clusters",
    use_sample_data: bool = False,
    random_state: Optional[int] = 42
):
    """
    Run the nested clustering optimization pipeline.
    
    Args:
        prices: Optional price array (n_periods, n_assets)
        asset_names: Optional list of asset names
        n_clusters_level1: Number of clusters at level 1
        n_clusters_level2: Number of clusters at level 2
        clustering_method: 'kmeans' or 'hierarchical'
        allocation_strategy: 'equal_weight_clusters' or 'optimize_clusters'
        use_sample_data: If True, generate sample data
        random_state: Random seed
    """
    print("=" * 70)
    print("Nested Clustering Optimizer for Investment Portfolios")
    print("=" * 70)
    print()
    
    # Load data
    data_handler = PortfolioDataHandler()
    
    if use_sample_data or prices is None:
        print("Generating sample portfolio data...")
        prices, asset_names, feature_names = generate_sample_data(
            n_assets=50,
            n_periods=252,
            random_state=random_state
        )
        data_handler.load_from_prices(prices, asset_names, feature_names)
        print(f"✓ Generated data: {len(asset_names)} assets, {prices.shape[0]} periods")
    else:
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(prices.shape[1])]
        data_handler.load_from_prices(prices, asset_names)
        print(f"✓ Loaded data: {len(asset_names)} assets, {prices.shape[0]} periods")
    
    print()
    
    # Get features and returns
    features = data_handler.get_features()
    expected_returns = data_handler.get_expected_returns()
    covariance_matrix = data_handler.get_covariance_matrix()
    
    print(f"Features shape: {features.shape}")
    print(f"Expected returns shape: {expected_returns.shape}")
    print(f"Covariance matrix shape: {covariance_matrix.shape}")
    print()
    
    # Step 1: Nested Clustering
    print("-" * 70)
    print("Step 1: Performing Nested Clustering")
    print("-" * 70)
    
    clusterer = NestedClusteringOptimizer(
        n_clusters_level1=n_clusters_level1,
        n_clusters_level2=n_clusters_level2,
        clustering_method=clustering_method,
        random_state=random_state
    )
    
    clusterer.fit(features, data_handler.feature_names)
    cluster_hierarchy = clusterer.get_cluster_hierarchy()
    
    print(f"✓ Level 1 clusters: {n_clusters_level1}")
    print(f"✓ Level 2 clusters per level1: {n_clusters_level2}")
    
    # Print cluster structure
    for level1_id, level1_data in cluster_hierarchy.items():
        n_assets_level1 = len(level1_data["assets"])
        print(f"  Level 1 Cluster {level1_id}: {n_assets_level1} assets")
        
        if level1_data["subclusters"]:
            for level2_id, level2_assets in level1_data["subclusters"].items():
                n_assets_level2 = len(level2_assets)
                print(f"    Level 2 Cluster {level2_id}: {n_assets_level2} assets")
    
    # Evaluate clustering quality
    metrics = clusterer.evaluate_clustering(features)
    print()
    print("Clustering Quality Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print()
    
    # Step 2: Portfolio Optimization
    print("-" * 70)
    print("Step 2: Optimizing Portfolio Allocations")
    print("-" * 70)
    
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    # Optimize within clusters
    print("Optimizing allocations within clusters...")
    optimal_weights, optimization_info = optimizer.nested_optimization(
        cluster_hierarchy,
        expected_returns,
        covariance_matrix,
        allocation_strategy=allocation_strategy
    )
    
    print(f"✓ Optimization completed")
    print()
    print("Portfolio Metrics:")
    print(f"  Expected Return: {optimization_info['portfolio_return']:.4f} ({optimization_info['portfolio_return']*252:.2%} annualized)")
    print(f"  Volatility: {optimization_info['portfolio_std']:.4f} ({optimization_info['portfolio_std']*np.sqrt(252):.2%} annualized)")
    print(f"  Sharpe Ratio: {optimization_info['sharpe_ratio']:.4f}")
    print()
    
    # Show top holdings
    top_n = 10
    top_indices = np.argsort(optimal_weights)[::-1][:top_n]
    print(f"Top {top_n} Holdings:")
    for i, idx in enumerate(top_indices, 1):
        weight = optimal_weights[idx]
        asset_name = asset_names[idx]
        print(f"  {i:2d}. {asset_name}: {weight:.4f} ({weight*100:.2f}%)")
    
    print()
    print("=" * 70)
    print("Optimization Complete!")
    print("=" * 70)
    
    return {
        "optimal_weights": optimal_weights,
        "cluster_hierarchy": cluster_hierarchy,
        "optimization_info": optimization_info,
        "clustering_metrics": metrics
    }


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Nested Clustering Optimizer for Investment Portfolios"
    )
    parser.add_argument(
        "--n-clusters-level1",
        type=int,
        default=3,
        help="Number of clusters at level 1 (default: 3)"
    )
    parser.add_argument(
        "--n-clusters-level2",
        type=int,
        default=2,
        help="Number of clusters at level 2 (default: 2)"
    )
    parser.add_argument(
        "--clustering-method",
        type=str,
        default="kmeans",
        choices=["kmeans", "hierarchical"],
        help="Clustering method (default: kmeans)"
    )
    parser.add_argument(
        "--allocation-strategy",
        type=str,
        default="equal_weight_clusters",
        choices=["equal_weight_clusters", "optimize_clusters"],
        help="Allocation strategy across clusters (default: equal_weight_clusters)"
    )
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="Use sample generated data instead of loading from file"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    try:
        run_optimization(
            n_clusters_level1=args.n_clusters_level1,
            n_clusters_level2=args.n_clusters_level2,
            clustering_method=args.clustering_method,
            allocation_strategy=args.allocation_strategy,
            use_sample_data=args.use_sample_data,
            random_state=args.random_state
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
