"""Tests for the ML trading pipeline."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
import pandas as pd
import numpy as np
from time_series_model.pipeline.multi_tf_pipeline import MultiTimeframePipeline
from time_series_model.pipeline.risk_management import RiskManager


class TestMultiTimeframePipeline(unittest.TestCase):
    """Test cases for the multi-timeframe pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = MultiTimeframePipeline()

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertFalse(self.pipeline.is_trained)
        self.assertEqual(len(self.pipeline.stage1_models), 0)
        self.assertEqual(len(self.pipeline.stage2_models), 0)

    def test_prepare_targets(self):
        """Test target preparation."""
        # Create sample data
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 101.5, 103],
                "open": [99, 100, 101, 100.5, 102],
                "high": [101, 102, 103, 102, 104],
                "low": [99, 100, 101, 100, 101.5],
                "volume": [1000, 1100, 1200, 1050, 1300],
            }
        )

        stage1_target, stage2_target = self.pipeline.prepare_targets(data)

        # Check that targets have correct length
        self.assertEqual(len(stage1_target), len(data))
        self.assertEqual(len(stage2_target), len(data))

        # Check that stage1 target contains expected values (-1, 0, 1)
        unique_values = set(stage1_target.unique())
        expected_values = {-1, 0, 1}
        self.assertTrue(unique_values.issubset(expected_values))


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
