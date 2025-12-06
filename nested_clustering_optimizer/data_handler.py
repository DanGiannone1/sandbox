"""
Data Handling Module

Utilities for loading, processing, and preparing portfolio data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import warnings


class PortfolioDataHandler:
    """
    Handles loading and preprocessing of portfolio data.
    """
    
    def __init__(self):
        """Initialize the data handler."""
        self.asset_names = None
        self.feature_names = None
        self.prices = None
        self.returns = None
        self.features = None
    
    def load_from_prices(
        self,
        prices: np.ndarray,
        asset_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Load portfolio data from price array.
        
        Args:
            prices: Array of shape (n_periods, n_assets) with price data
            asset_names: Optional list of asset names/identifiers
            feature_names: Optional list of feature names
        """
        self.prices = np.array(prices)
        n_assets = self.prices.shape[1]
        
        # Calculate returns
        self.returns = np.diff(self.prices, axis=0) / self.prices[:-1]
        
        # Set asset names
        if asset_names is None:
            self.asset_names = [f"Asset_{i}" for i in range(n_assets)]
        else:
            if len(asset_names) != n_assets:
                raise ValueError(
                    f"Number of asset names ({len(asset_names)}) "
                    f"does not match number of assets ({n_assets})"
                )
            self.asset_names = asset_names
        
        # Generate features from returns
        self._generate_features_from_returns()
        
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.features.shape[1])]
        else:
            if len(feature_names) != self.features.shape[1]:
                raise ValueError(
                    f"Number of feature names ({len(feature_names)}) "
                    f"does not match number of features ({self.features.shape[1]})"
                )
            self.feature_names = feature_names
    
    def load_from_returns(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Load portfolio data from returns array.
        
        Args:
            returns: Array of shape (n_periods, n_assets) with return data
            asset_names: Optional list of asset names/identifiers
            feature_names: Optional list of feature names
        """
        self.returns = np.array(returns)
        n_assets = self.returns.shape[1]
        
        # Set asset names
        if asset_names is None:
            self.asset_names = [f"Asset_{i}" for i in range(n_assets)]
        else:
            if len(asset_names) != n_assets:
                raise ValueError(
                    f"Number of asset names ({len(asset_names)}) "
                    f"does not match number of assets ({n_assets})"
                )
            self.asset_names = asset_names
        
        # Generate features from returns
        self._generate_features_from_returns()
        
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.features.shape[1])]
        else:
            if len(feature_names) != self.features.shape[1]:
                raise ValueError(
                    f"Number of feature names ({len(feature_names)}) "
                    f"does not match number of features ({self.features.shape[1]})"
                )
            self.feature_names = feature_names
    
    def load_from_features(
        self,
        features: np.ndarray,
        asset_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        returns: Optional[np.ndarray] = None
    ):
        """
        Load portfolio data directly from feature array.
        
        Args:
            features: Array of shape (n_assets, n_features) with feature data
            asset_names: Optional list of asset names/identifiers
            feature_names: Optional list of feature names
            returns: Optional returns array for optimization
        """
        self.features = np.array(features)
        n_assets = self.features.shape[0]
        
        # Set asset names
        if asset_names is None:
            self.asset_names = [f"Asset_{i}" for i in range(n_assets)]
        else:
            if len(asset_names) != n_assets:
                raise ValueError(
                    f"Number of asset names ({len(asset_names)}) "
                    f"does not match number of assets ({n_assets})"
                )
            self.asset_names = asset_names
        
        # Set feature names
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.features.shape[1])]
        else:
            if len(feature_names) != self.features.shape[1]:
                raise ValueError(
                    f"Number of feature names ({len(feature_names)}) "
                    f"does not match number of features ({self.features.shape[1]})"
                )
            self.feature_names = feature_names
        
        # Set returns if provided
        if returns is not None:
            self.returns = np.array(returns)
            if self.returns.shape[1] != n_assets:
                raise ValueError(
                    f"Returns shape ({self.returns.shape}) "
                    f"does not match number of assets ({n_assets})"
                )
    
    def _generate_features_from_returns(self):
        """Generate features from returns data."""
        if self.returns is None:
            raise ValueError("Returns data must be loaded first")
        
        n_periods, n_assets = self.returns.shape
        
        # Calculate various features
        features_list = []
        
        # Mean return
        mean_return = np.mean(self.returns, axis=0)
        features_list.append(mean_return)
        
        # Volatility (std)
        volatility = np.std(self.returns, axis=0)
        features_list.append(volatility)
        
        # Sharpe-like ratio (mean / std)
        sharpe_like = mean_return / (volatility + 1e-8)
        features_list.append(sharpe_like)
        
        # Skewness
        if n_periods > 2:
            skewness = self._calculate_skewness(self.returns)
            features_list.append(skewness)
        
        # Kurtosis
        if n_periods > 3:
            kurtosis = self._calculate_kurtosis(self.returns)
            features_list.append(kurtosis)
        
        # Maximum drawdown
        if self.prices is not None:
            max_drawdown = self._calculate_max_drawdown(self.prices)
            features_list.append(max_drawdown)
        
        # Combine features
        self.features = np.column_stack(features_list)
    
    def _calculate_skewness(self, returns: np.ndarray) -> np.ndarray:
        """Calculate skewness for each asset."""
        mean = np.mean(returns, axis=0)
        std = np.std(returns, axis=0)
        n = returns.shape[0]
        
        skew = np.mean(((returns - mean) / (std + 1e-8)) ** 3, axis=0)
        return skew
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> np.ndarray:
        """Calculate kurtosis for each asset."""
        mean = np.mean(returns, axis=0)
        std = np.std(returns, axis=0)
        
        kurt = np.mean(((returns - mean) / (std + 1e-8)) ** 4, axis=0) - 3
        return kurt
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> np.ndarray:
        """Calculate maximum drawdown for each asset."""
        n_assets = prices.shape[1]
        max_dd = np.zeros(n_assets)
        
        for i in range(n_assets):
            price_series = prices[:, i]
            cumulative = np.maximum.accumulate(price_series)
            drawdown = (price_series - cumulative) / cumulative
            max_dd[i] = np.min(drawdown)
        
        return max_dd
    
    def get_expected_returns(self) -> np.ndarray:
        """Get expected returns (mean of historical returns)."""
        if self.returns is None:
            raise ValueError("Returns data not available")
        return np.mean(self.returns, axis=0)
    
    def get_covariance_matrix(self) -> np.ndarray:
        """Get covariance matrix of returns."""
        if self.returns is None:
            raise ValueError("Returns data not available")
        return np.cov(self.returns.T)
    
    def get_features(self) -> np.ndarray:
        """Get feature matrix."""
        if self.features is None:
            raise ValueError("Features not available")
        return self.features
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of the portfolio data."""
        stats = {
            "n_assets": len(self.asset_names),
            "asset_names": self.asset_names
        }
        
        if self.returns is not None:
            stats["n_periods"] = self.returns.shape[0]
            stats["mean_returns"] = self.get_expected_returns().tolist()
            stats["volatilities"] = np.std(self.returns, axis=0).tolist()
        
        if self.features is not None:
            stats["n_features"] = self.features.shape[1]
            stats["feature_names"] = self.feature_names
        
        return stats
