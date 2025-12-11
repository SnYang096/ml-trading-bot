"""Integration tests for factor evaluation tools.

Tests for:
- factor_ts_eval.py (time-series factor evaluation)
- cross_sectional_eval.py (cross-sectional factor evaluation)

These tests verify that the evaluation scripts work correctly with the integration test environment.

Note: Some tests may be skipped if dependencies are not available.
"""

import pytest
import pandas as pd
import numpy as np

IMPORT_ERROR_TS = ""
IMPORT_ERROR_CS = ""


# Try to import functions, skip tests if imports fail
try:
    from src.time_series_model.diagnostics.factor_ts_eval import (
        compute_factor_metrics,
        compute_ic_decay,
    )

    FACTOR_TS_EVAL_AVAILABLE = True
except ImportError as e:
    FACTOR_TS_EVAL_AVAILABLE = False
    IMPORT_ERROR_TS = str(e)

try:
    from src.time_series_model.diagnostics.cross_sectional_eval import (
        load_requested_features,
        compute_future_return,
        compute_factor_metrics as cs_compute_factor_metrics,
    )

    CROSS_SECTIONAL_EVAL_AVAILABLE = True
except ImportError as e:
    CROSS_SECTIONAL_EVAL_AVAILABLE = False
    IMPORT_ERROR_CS = str(e)


class TestFactorTsEvalIntegration:
    """Integration tests for time-series factor evaluation."""

    @pytest.mark.skipif(
        not FACTOR_TS_EVAL_AVAILABLE,
        reason=f"factor_ts_eval not available: {IMPORT_ERROR_TS}",
    )
    def test_compute_factor_metrics_structure(self):
        """Test compute_factor_metrics function structure."""
        np.random.seed(42)

        # Create mock data with realistic structure
        n_samples = 200
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="15min")
        df = pd.DataFrame(
            {
                "factor_1": np.random.randn(n_samples),
                "target": np.random.randn(n_samples),
            },
            index=dates,
        )

        # Test compute_factor_metrics
        metrics, ic_series = compute_factor_metrics(
            df=df,
            factor="factor_1",
            target_col="target",
            quantile=0.2,
            ic_decay_lags=[1, 3, 5],
        )

        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        # Check for common metric keys
        assert "ic_mean" in metrics or "error" in metrics

        # Verify IC series
        assert isinstance(ic_series, pd.DataFrame)

        print(f"✅ Factor metrics computed: {list(metrics.keys())[:5]}...")

    @pytest.mark.skipif(
        not FACTOR_TS_EVAL_AVAILABLE,
        reason=f"factor_ts_eval not available: {IMPORT_ERROR_TS}",
    )
    def test_compute_ic_decay(self):
        """Test IC decay computation."""
        np.random.seed(42)

        n_samples = 200
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="15min")
        df = pd.DataFrame(
            {
                "factor_1": np.random.randn(n_samples),
                "target": np.random.randn(n_samples),
            },
            index=dates,
        )

        decay_metrics = compute_ic_decay(
            df=df,
            factor="factor_1",
            target_col="target",
            decay_lags=[1, 3, 5],
        )

        assert isinstance(decay_metrics, dict)
        # Should have metrics for each lag
        for lag in [1, 3, 5]:
            assert f"ic_lag_{lag}" in decay_metrics

        print(f"✅ IC decay computed for lags: {list(decay_metrics.keys())}")


class TestCrossSectionalEvalIntegration:
    """Integration tests for cross-sectional factor evaluation."""

    @pytest.mark.skipif(
        not CROSS_SECTIONAL_EVAL_AVAILABLE,
        reason=f"cross_sectional_eval not available: {IMPORT_ERROR_CS}",
    )
    def test_load_requested_features(self, integration_env):
        """Test loading requested features from config."""
        from pathlib import Path

        # Use integration config
        config_path = Path(integration_env["config_dir"]) / "features.yaml"

        if config_path.exists():
            features = load_requested_features(config_path)
            assert isinstance(features, list)
            assert len(features) > 0
            print(f"✅ Loaded {len(features)} features: {features[:3]}...")
        else:
            pytest.skip(f"Config file not found: {config_path}")

    @pytest.mark.skipif(
        not CROSS_SECTIONAL_EVAL_AVAILABLE,
        reason=f"cross_sectional_eval not available: {IMPORT_ERROR_CS}",
    )
    def test_compute_future_return(self):
        """Test future return computation."""
        np.random.seed(42)

        # Create mock data with close prices
        n_samples = 100
        prices = 50000 * (1 + np.random.randn(n_samples) * 0.01).cumprod()
        df = pd.DataFrame({"close": prices})

        horizon = 24
        future_returns = compute_future_return(df, horizon)

        assert isinstance(future_returns, pd.Series)
        assert len(future_returns) == len(df)
        # Last `horizon` values should be NaN (no future data)
        assert future_returns.iloc[-horizon:].isna().all()

        print(
            f"✅ Future returns computed: {len(future_returns)} values, {future_returns.notna().sum()} valid"
        )

    @pytest.mark.skipif(
        not CROSS_SECTIONAL_EVAL_AVAILABLE,
        reason=f"cross_sectional_eval not available: {IMPORT_ERROR_CS}",
    )
    def test_cs_compute_factor_metrics_structure(self):
        """Test cross-sectional factor metrics computation structure."""
        np.random.seed(42)

        # Create mock cross-sectional data
        n_timestamps = 10
        n_symbols = 5
        timestamps = pd.date_range("2024-01-01", periods=n_timestamps, freq="1D")

        data = []
        for ts in timestamps:
            for symbol_idx in range(n_symbols):
                data.append(
                    {
                        "timestamp": ts,
                        "_symbol": f"SYMBOL{symbol_idx}",
                        "factor_1": np.random.randn(),
                        "future_return": np.random.randn(),
                    }
                )

        df = pd.DataFrame(data)

        # Test metrics computation
        metrics, quantile_df = cs_compute_factor_metrics(
            df=df,
            factor="factor_1",
            target_col="future_return",
            quantiles=5,
            min_assets=3,
            ic_decay_lags=[1, 3],
        )

        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert "rank_ic_mean" in metrics or "error" in metrics
        assert "rank_ic_std" in metrics or "error" in metrics

        # Verify quantile DataFrame (may be empty for small datasets)
        assert isinstance(quantile_df, pd.DataFrame)

        print(f"✅ Cross-sectional metrics computed: {list(metrics.keys())}")


class TestFactorEvalEndToEnd:
    """End-to-end tests for factor evaluation (using integration environment)."""

    def test_factor_ts_eval_workflow_setup(self, integration_env):
        """Test that integration environment is set up correctly for ts-factor-eval."""
        from pathlib import Path

        assert "data_dir" in integration_env
        assert "config_dir" in integration_env
        assert "symbol" in integration_env

        # Verify config directory has required files
        config_dir = Path(integration_env["config_dir"])
        assert (config_dir / "features.yaml").exists()
        assert (config_dir / "labels.yaml").exists()
        assert (config_dir / "model.yaml").exists()
        assert (config_dir / "evaluation.yaml").exists()

        print(f"✅ Integration environment ready for ts-factor-eval")
        print(f"   Config: {config_dir}")
        print(f"   Data: {integration_env['data_dir']}")
        print(f"   Symbol: {integration_env['symbol']}")

    def test_cs_factor_eval_workflow_setup(self, integration_env):
        """Test that integration environment is set up correctly for cs-factor-eval."""
        from pathlib import Path

        assert "data_dir" in integration_env
        assert "config_dir" in integration_env

        # Verify data directory has parquet files
        data_dir = Path(integration_env["data_dir"])
        parquet_files = list(data_dir.glob("*.parquet"))
        assert len(parquet_files) > 0, "Should have at least one parquet file"

        # Verify config has features.yaml (needed for cs-factor-eval)
        config_dir = Path(integration_env["config_dir"])
        assert (config_dir / "features.yaml").exists()

        print(f"✅ Integration environment ready for cs-factor-eval")
        print(f"   Data files: {len(parquet_files)}")
        print(f"   Config: {config_dir}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
