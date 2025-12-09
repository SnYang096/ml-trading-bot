"""Integration tests for dimensionality comparison (config-driven mode).

These tests require:
- Full data pipeline
- Feature engineering
- Strategy configurations
- Complete environment setup

Run with: pytest tests/integration/test_dimensionality_comparison_integration.py -v
"""

import sys
from pathlib import Path

import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.pipeline.dimensionality.dimensionality_comparison import (
    run_dim_compare,
)


class TestDimensionalityComparisonIntegration:
    """Integration tests for config-driven dimensionality comparison."""

    def test_run_dim_compare_basic(self, integration_env):
        """Test run_dim_compare with complete integration environment."""
        results, top_factors_path = run_dim_compare(
            config_dir=integration_env["config_dir"],
            symbol=integration_env["symbol"],
            data_path=integration_env["data_dir"],
            timeframe=integration_env["timeframe"],
            train_start=None,
            train_end=None,
        )

        # Verify results structure
        assert isinstance(results, dict), "Results should be a dictionary"
        assert "strategy" in results, "Results should contain strategy name"
        assert "symbol" in results, "Results should contain symbol"
        assert "data_info" in results, "Results should contain data_info"
        assert "performance" in results, "Results should contain performance"
        assert "top_factors_path" in results, "Results should contain top_factors_path"

        # Verify data_info
        data_info = results["data_info"]
        assert "original_features_count" in data_info
        assert "stage1_all_features" in data_info
        assert "stage2_ic_filtered" in data_info
        assert "stage3_representatives" in data_info
        assert "compression_ratio" in data_info

        # Verify performance
        performance = results["performance"]
        assert "before_reduction" in performance
        assert "after_reduction" in performance
        assert "performance_change" in performance

        # Verify top_factors.json exists
        top_factors_file = Path(top_factors_path)
        assert (
            top_factors_file.exists()
        ), f"Top factors file should exist: {top_factors_path}"

        # Verify top_factors.json structure
        import json

        with open(top_factors_file, "r") as f:
            top_factors_data = json.load(f)

        assert "top_factors" in top_factors_data
        assert "count" in top_factors_data
        assert isinstance(top_factors_data["top_factors"], list)
        assert top_factors_data["count"] >= 0

        print(f"\n✅ Integration test passed!")
        print(f"   Strategy: {results['strategy']}")
        print(f"   Symbol: {results['symbol']}")
        print(f"   Original features: {data_info['original_features_count']}")
        print(f"   Final features: {data_info['stage3_representatives']}")
        print(f"   Compression ratio: {data_info['compression_ratio']:.2f}x")
        print(f"   Performance change: {performance['performance_change']:.4f}")

    def test_run_dim_compare_with_date_range(self, integration_env):
        """Test run_dim_compare with date range specified."""
        results, top_factors_path = run_dim_compare(
            config_dir=integration_env["config_dir"],
            symbol=integration_env["symbol"],
            data_path=integration_env["data_dir"],
            timeframe=integration_env["timeframe"],
            train_start="2024-01-01",
            train_end="2024-12-31",
        )

        assert isinstance(results, dict)
        assert results["symbol"] == integration_env["symbol"]
        assert Path(top_factors_path).exists()

    def test_run_dim_compare_feature_selection_stages(self, integration_env):
        """Test that all three stages of feature selection are executed."""
        results, top_factors_path = run_dim_compare(
            config_dir=integration_env["config_dir"],
            symbol=integration_env["symbol"],
            data_path=integration_env["data_dir"],
            timeframe=integration_env["timeframe"],
        )

        data_info = results["data_info"]

        # Verify all stages are present
        assert "stage1_all_features" in data_info
        assert "stage2_ic_filtered" in data_info
        assert "stage3_representatives" in data_info

        # Verify stage progression (each stage should reduce features)
        assert data_info["stage1_all_features"] >= data_info["stage2_ic_filtered"]
        assert data_info["stage2_ic_filtered"] >= data_info["stage3_representatives"]

        # Verify compression ratio
        compression_ratio = data_info["compression_ratio"]
        assert compression_ratio >= 1.0, "Compression ratio should be >= 1.0"

        print(f"\n✅ Feature selection stages verified:")
        print(
            f"   Stage 1 (after filtering): {data_info['stage1_all_features']} features"
        )
        print(
            f"   Stage 2 (after IC ranking): {data_info['stage2_ic_filtered']} features"
        )
        print(
            f"   Stage 3 (representatives): {data_info['stage3_representatives']} features"
        )
        print(f"   Compression: {compression_ratio:.2f}x")

    def test_run_dim_compare_output_files(self, integration_env):
        """Test that all expected output files are created."""
        results, top_factors_path = run_dim_compare(
            config_dir=integration_env["config_dir"],
            symbol=integration_env["symbol"],
            data_path=integration_env["data_dir"],
            timeframe=integration_env["timeframe"],
        )

        results_dir = Path(top_factors_path).parent

        # Verify top_factors.json exists
        assert Path(top_factors_path).exists()

        # Verify results.json exists
        results_file = results_dir / "results.json"
        assert results_file.exists(), "results.json should exist"

        # Verify results.json content
        import json

        with open(results_file, "r") as f:
            saved_results = json.load(f)

        assert saved_results["strategy"] == results["strategy"]
        assert saved_results["symbol"] == results["symbol"]

        print(f"\n✅ Output files verified:")
        print(f"   Results directory: {results_dir}")
        print(f"   Top factors: {Path(top_factors_path).name}")
        print(f"   Results JSON: {results_file.name}")


@pytest.mark.slow
class TestDimensionalityComparisonWithRealConfig:
    """Integration tests using actual strategy configurations.

    These tests require the actual config/strategies/ directory to exist.
    Marked as 'slow' because they may take longer to run.
    """

    @pytest.fixture
    def real_config_dir(self):
        """Use actual sr_reversal strategy config."""
        config_dir = PROJECT_ROOT / "config" / "strategies" / "sr_reversal"
        if not config_dir.exists():
            pytest.skip("Real strategy config not available")
        return config_dir

    @pytest.fixture
    def real_data_dir(self):
        """Use actual data directory if available."""
        data_dir = PROJECT_ROOT / "data" / "parquet_data"
        if not data_dir.exists():
            pytest.skip("Real data directory not available")
        return str(data_dir)

    def test_run_dim_compare_with_real_config(
        self, real_config_dir, real_data_dir, integration_env
    ):
        """Test with actual sr_reversal strategy configuration.

        Note: This test requires actual market data.
        """
        # Use integration test data if real data is not available
        data_dir = (
            real_data_dir
            if Path(real_data_dir).exists()
            else integration_env["data_dir"]
        )

        results, top_factors_path = run_dim_compare(
            config_dir=real_config_dir,
            symbol="BTCUSDT",
            data_path=data_dir,
            timeframe="15T",
            train_start="2024-01-01",
            train_end="2024-12-31",
        )

        assert isinstance(results, dict)
        assert results["strategy"] == "sr_reversal"
        assert Path(top_factors_path).exists()

        print(f"\n✅ Real config test passed!")
        print(f"   Strategy: {results['strategy']}")
        print(f"   Top factors: {Path(top_factors_path).name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
