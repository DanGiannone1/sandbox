# Nested Clustering Optimizer for Investment Portfolios

A comprehensive Python application for optimizing investment portfolios using nested clustering techniques. This tool clusters assets at multiple hierarchical levels and optimizes allocations within each cluster to create well-diversified, risk-adjusted portfolios.

## Features

- **Nested Clustering**: Performs hierarchical clustering at multiple levels (e.g., by sector, then by risk level)
- **Portfolio Optimization**: Implements mean-variance optimization with Sharpe ratio maximization
- **Flexible Data Input**: Supports loading data from prices, returns, or pre-computed features
- **Multiple Clustering Methods**: Supports K-means and hierarchical clustering algorithms
- **Comprehensive Metrics**: Provides clustering quality metrics and portfolio performance metrics

## Installation

Install the required dependencies:

```bash
pip install numpy scipy scikit-learn pandas
```

## Quick Start

### Using Sample Data

Run the optimizer with generated sample data:

```bash
python -m nested_clustering_optimizer.main --use-sample-data
```

### Command-Line Options

```bash
python -m nested_clustering_optimizer.main \
    --n-clusters-level1 3 \
    --n-clusters-level2 2 \
    --clustering-method kmeans \
    --allocation-strategy equal_weight_clusters \
    --use-sample-data \
    --random-state 42
```

**Options:**
- `--n-clusters-level1`: Number of clusters at the first level (default: 3)
- `--n-clusters-level2`: Number of clusters at the second level (default: 2)
- `--clustering-method`: Clustering algorithm - `kmeans` or `hierarchical` (default: kmeans)
- `--allocation-strategy`: Strategy for allocating across clusters - `equal_weight_clusters` or `optimize_clusters` (default: equal_weight_clusters)
- `--use-sample-data`: Use generated sample data instead of loading from file
- `--random-state`: Random seed for reproducibility (default: 42)

### Programmatic Usage

```python
import numpy as np
from nested_clustering_optimizer import (
    NestedClusteringOptimizer,
    PortfolioOptimizer,
    PortfolioDataHandler
)

# Load or generate your portfolio data
data_handler = PortfolioDataHandler()
prices = np.array(...)  # Shape: (n_periods, n_assets)
data_handler.load_from_prices(prices, asset_names=["AAPL", "GOOGL", ...])

# Get features and returns
features = data_handler.get_features()
expected_returns = data_handler.get_expected_returns()
covariance_matrix = data_handler.get_covariance_matrix()

# Perform nested clustering
clusterer = NestedClusteringOptimizer(
    n_clusters_level1=3,
    n_clusters_level2=2,
    clustering_method="kmeans"
)
clusterer.fit(features)
cluster_hierarchy = clusterer.get_cluster_hierarchy()

# Optimize portfolio
optimizer = PortfolioOptimizer(risk_free_rate=0.02)
optimal_weights, info = optimizer.nested_optimization(
    cluster_hierarchy,
    expected_returns,
    covariance_matrix,
    allocation_strategy="equal_weight_clusters"
)

print(f"Optimal weights: {optimal_weights}")
print(f"Portfolio Sharpe Ratio: {info['sharpe_ratio']:.4f}")
```

## Architecture

### Modules

1. **`clustering.py`**: Implements nested clustering algorithms
   - `NestedClusteringOptimizer`: Main clustering class

2. **`portfolio_optimizer.py`**: Portfolio optimization logic
   - `PortfolioOptimizer`: Mean-variance optimization with Sharpe ratio maximization

3. **`data_handler.py`**: Data loading and preprocessing
   - `PortfolioDataHandler`: Handles data loading, feature extraction, and statistics

4. **`main.py`**: Application entry point and example usage

## How It Works

1. **Data Preparation**: Load price or return data and extract features (mean return, volatility, Sharpe ratio, skewness, kurtosis, max drawdown)

2. **Level 1 Clustering**: Cluster all assets into `n_clusters_level1` groups based on their features

3. **Level 2 Clustering**: Within each Level 1 cluster, further subdivide into `n_clusters_level2` subclusters

4. **Cluster Optimization**: Optimize portfolio weights within each cluster using mean-variance optimization

5. **Nested Allocation**: Allocate capital across clusters using either equal weighting or cluster-level optimization

6. **Final Portfolio**: Combine cluster-level optimizations to produce the final optimal portfolio weights

## Output

The optimizer provides:

- **Cluster Hierarchy**: Structure showing which assets belong to which clusters at each level
- **Optimal Weights**: Recommended portfolio allocation for each asset
- **Portfolio Metrics**: Expected return, volatility, and Sharpe ratio
- **Clustering Quality**: Silhouette scores for evaluating cluster quality
- **Top Holdings**: List of assets with highest allocations

## Example Output

```
======================================================================
Nested Clustering Optimizer for Investment Portfolios
======================================================================

✓ Generated data: 50 assets, 253 periods

Features shape: (50, 6)
Expected returns shape: (50,)
Covariance matrix shape: (50, 50)

----------------------------------------------------------------------
Step 1: Performing Nested Clustering
----------------------------------------------------------------------
✓ Level 1 clusters: 3
✓ Level 2 clusters per level1: 2
  Level 1 Cluster 0: 16 assets
    Level 2 Cluster 0: 8 assets
    Level 2 Cluster 1: 8 assets
  Level 1 Cluster 1: 17 assets
    Level 2 Cluster 0: 8 assets
    Level 2 Cluster 1: 9 assets
  Level 1 Cluster 2: 17 assets
    Level 2 Cluster 0: 8 assets
    Level 2 Cluster 1: 9 assets

Clustering Quality Metrics:
  level1_silhouette: 0.4523
  level2_silhouette_mean: 0.3845
  level2_silhouette_std: 0.0234

----------------------------------------------------------------------
Step 2: Optimizing Portfolio Allocations
----------------------------------------------------------------------
✓ Optimization completed

Portfolio Metrics:
  Expected Return: 0.0123 (3.10% annualized)
  Volatility: 0.0234 (37.10% annualized)
  Sharpe Ratio: 0.4521

Top 10 Holdings:
   1. Asset_015: 0.0456 (4.56%)
   2. Asset_032: 0.0432 (4.32%)
   ...
```

## Requirements

- Python 3.7+
- numpy >= 1.19.0
- scipy >= 1.5.0
- scikit-learn >= 0.23.0
- pandas >= 1.1.0 (optional, for data handling)

## License

See LICENSE file in the project root.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
