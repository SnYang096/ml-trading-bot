"""
Live trading integration module.

This module provides live trading functionality using Nautilus Trader,
including strategy execution with feature engineering integration.
"""

from .nautilus_strategy_with_features import NautilusStrategyWithFeatures

__all__ = ["NautilusStrategyWithFeatures"]
