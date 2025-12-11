"""Tests for the ML trading pipeline."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
import pandas as pd
import numpy as np

# Note: MultiTimeframePipeline has been removed (replaced by config-driven training)
# from time_series_model.pipeline.multi_tf_pipeline import MultiTimeframePipeline
from time_series_model.pipeline.risk_management import RiskManager


# Note: TestMultiTimeframePipeline removed - MultiTimeframePipeline is deprecated
# All training now uses config-driven pipeline (scripts/train_strategy_pipeline.py)


class TestRiskManager(unittest.TestCase):
    """Test cases for the risk manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.risk_manager = RiskManager()

    def test_risk_manager_initialization(self):
        """Test risk manager initialization."""
        self.assertEqual(len(self.risk_manager.position_history), 0)
        self.assertEqual(self.risk_manager.consecutive_losses, 0)
        self.assertEqual(len(self.risk_manager.rolling_returns), 0)

    def test_calculate_dynamic_levels(self):
        """Test dynamic level calculation."""
        # Create sample price data
        prices = pd.Series([100, 101, 102, 101, 103, 104, 103, 105])

        stop_loss, take_profit = self.risk_manager.calculate_dynamic_levels(prices)

        # Check that levels are positive floats
        self.assertIsInstance(stop_loss, float)
        self.assertIsInstance(take_profit, float)
        self.assertGreaterEqual(stop_loss, 0)
        self.assertGreaterEqual(take_profit, 0)

    def test_adjust_position_size(self):
        """Test position size adjustment."""
        position = self.risk_manager.adjust_position_size(
            signal=0.8, expected_return=0.02, current_price=100.0
        )

        # Check that position is a positive float
        self.assertIsInstance(position, float)
        self.assertGreater(position, 0)


if __name__ == "__main__":
    unittest.main()
