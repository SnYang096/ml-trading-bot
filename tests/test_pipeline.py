"""Tests for the ML trading pipeline."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
import pandas as pd
import numpy as np

# Note: MultiTimeframePipeline has been removed (replaced by config-driven training)
# from time_series_model.pipeline.multi_tf_pipeline import MultiTimeframePipeline

# Note: TestMultiTimeframePipeline removed - MultiTimeframePipeline is deprecated
# All training now uses config-driven pipeline (scripts/train_strategy_pipeline.py)

# Note: RiskManager has been removed - not used in actual code


if __name__ == "__main__":
    unittest.main()
