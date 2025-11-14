"""
Portfolio Optimization Module

Implements mean-variance optimization and other portfolio optimization strategies.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.optimize import minimize
import warnings


class PortfolioOptimizer:
    """
    Optimizes portfolio allocations using mean-variance optimization.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        risk_aversion: float = 1.0
    ):
        """
        Initialize the portfolio optimizer.
        
        Args:
            risk_free_rate: Risk-free rate (annualized)
            risk_aversion: Risk aversion parameter (higher = more risk averse)
        """
        self.risk_free_rate = risk_free_rate
        self.risk_aversion = risk_aversion
    
    def calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate returns from price data.
        
        Args:
            prices: Array of shape (n_periods, n_assets) with price data
            
        Returns:
            Array of shape (n_periods-1, n_assets) with returns
        """
        return np.diff(prices, axis=0) / prices[:-1]
    
    def mean_variance_optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform mean-variance optimization.
        
        Args:
            expected_returns: Array of expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            constraints: Optional dictionary with constraints:
                - 'min_weight': Minimum weight per asset (default: 0)
                - 'max_weight': Maximum weight per asset (default: 1)
                - 'target_return': Target portfolio return (optional)
                - 'long_only': If True, only allow long positions (default: True)
        
        Returns:
            Tuple of (optimal_weights, optimization_info)
        """
        n_assets = len(expected_returns)
        
        # Default constraints
        if constraints is None:
            constraints = {}
        
        min_weight = constraints.get("min_weight", 0.0)
        max_weight = constraints.get("max_weight", 1.0)
        long_only = constraints.get("long_only", True)
        target_return = constraints.get("target_return", None)
        
        # Objective function: maximize Sharpe ratio (or minimize negative Sharpe)
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            # Sharpe ratio
            sharpe = (portfolio_return - self.risk_free_rate) / (portfolio_std + 1e-8)
            return -sharpe  # Minimize negative Sharpe
        
        # Constraints
        constraint_list = []
        
        # Weights sum to 1
        constraint_list.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # Target return constraint (if specified)
        if target_return is not None:
            constraint_list.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, expected_returns) - target_return
            })
        
        # Bounds
        if long_only:
            bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        else:
            bounds = [(-max_weight, max_weight) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        optimal_weights = result.x
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_variance = np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights))
        portfolio_std = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / (portfolio_std + 1e-8)
        
        info = {
            "portfolio_return": portfolio_return,
            "portfolio_std": portfolio_std,
            "sharpe_ratio": sharpe_ratio,
            "optimization_success": result.success,
            "optimization_message": result.message
        }
        
        return optimal_weights, info
    
    def optimize_cluster_portfolio(
        self,
        cluster_indices: List[int],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Optimize portfolio for a specific cluster of assets.
        
        Args:
            cluster_indices: Indices of assets in the cluster
            expected_returns: Expected returns for all assets
            covariance_matrix: Covariance matrix for all assets
            constraints: Optional constraints dictionary
            
        Returns:
            Tuple of (weights_for_cluster_assets, info)
        """
        # Extract cluster-specific data
        cluster_returns = expected_returns[cluster_indices]
        cluster_cov = covariance_matrix[np.ix_(cluster_indices, cluster_indices)]
        
        # Optimize
        weights, info = self.mean_variance_optimize(
            cluster_returns,
            cluster_cov,
            constraints
        )
        
        # Create full weight vector (zeros for assets not in cluster)
        full_weights = np.zeros(len(expected_returns))
        full_weights[cluster_indices] = weights
        
        return full_weights, info
    
    def nested_optimization(
        self,
        cluster_hierarchy: Dict,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        allocation_strategy: str = "equal_weight_clusters"
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform nested optimization across cluster hierarchy.
        
        Args:
            cluster_hierarchy: Cluster hierarchy from NestedClusteringOptimizer
            expected_returns: Expected returns for all assets
            covariance_matrix: Covariance matrix for all assets
            allocation_strategy: 'equal_weight_clusters' or 'optimize_clusters'
            
        Returns:
            Tuple of (final_weights, optimization_info)
        """
        n_assets = len(expected_returns)
        final_weights = np.zeros(n_assets)
        
        cluster_weights = {}
        cluster_info = {}
        
        # Step 1: Optimize within each cluster
        for level1_id, level1_data in cluster_hierarchy.items():
            level1_assets = level1_data["assets"]
            
            if not level1_assets:
                continue
            
            # Optimize level1 cluster
            level1_weights, level1_info = self.optimize_cluster_portfolio(
                level1_assets,
                expected_returns,
                covariance_matrix
            )
            
            cluster_weights[level1_id] = level1_weights
            cluster_info[level1_id] = level1_info
            
            # Step 2: Optimize subclusters if they exist
            if level1_data["subclusters"]:
                for level2_id, level2_assets in level1_data["subclusters"].items():
                    if not level2_assets:
                        continue
                    
                    level2_weights, level2_info = self.optimize_cluster_portfolio(
                        level2_assets,
                        expected_returns,
                        covariance_matrix
                    )
                    
                    cluster_weights[(level1_id, level2_id)] = level2_weights
                    cluster_info[(level1_id, level2_id)] = level2_info
        
        # Step 3: Allocate across clusters
        if allocation_strategy == "equal_weight_clusters":
            # Equal weight across level1 clusters
            n_level1_clusters = len([k for k in cluster_hierarchy.keys() if isinstance(k, int)])
            weight_per_cluster = 1.0 / n_level1_clusters if n_level1_clusters > 0 else 0
            
            for level1_id in cluster_hierarchy.keys():
                if level1_id in cluster_weights:
                    final_weights += weight_per_cluster * cluster_weights[level1_id]
        
        elif allocation_strategy == "optimize_clusters":
            # Optimize allocation across clusters
            # Create cluster-level returns and covariance
            cluster_returns = []
            cluster_cov_entries = []
            
            for level1_id in sorted(cluster_hierarchy.keys()):
                if level1_id in cluster_weights:
                    cluster_return = np.dot(
                        cluster_weights[level1_id],
                        expected_returns
                    )
                    cluster_returns.append(cluster_return)
            
            cluster_returns = np.array(cluster_returns)
            
            # Approximate cluster covariance (simplified)
            cluster_cov = np.eye(len(cluster_returns)) * 0.1  # Placeholder
            
            # Optimize cluster allocation
            cluster_allocation, _ = self.mean_variance_optimize(
                cluster_returns,
                cluster_cov
            )
            
            # Combine
            for idx, level1_id in enumerate(sorted(cluster_hierarchy.keys())):
                if level1_id in cluster_weights:
                    final_weights += cluster_allocation[idx] * cluster_weights[level1_id]
        else:
            raise ValueError(f"Unknown allocation strategy: {allocation_strategy}")
        
        # Normalize weights
        final_weights = final_weights / (np.sum(final_weights) + 1e-8)
        
        # Calculate final portfolio metrics
        portfolio_return = np.dot(final_weights, expected_returns)
        portfolio_variance = np.dot(final_weights, np.dot(covariance_matrix, final_weights))
        portfolio_std = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / (portfolio_std + 1e-8)
        
        info = {
            "portfolio_return": portfolio_return,
            "portfolio_std": portfolio_std,
            "sharpe_ratio": sharpe_ratio,
            "cluster_weights": cluster_weights,
            "cluster_info": cluster_info,
            "allocation_strategy": allocation_strategy
        }
        
        return final_weights, info
