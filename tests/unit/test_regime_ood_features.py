"""Tests for P5 non-stationarity features: regime_state and ood_score.

Bug/Feature background:
  - P5 identifies three non-stationarity gaps: Regime detection, OOD detection, Alpha Decay.
  - `compute_regime_state_from_df` replicates RegimeDetector logic as a feature for Gate/Evidence.
  - `compute_ood_score_from_df` computes fraction of features outside training [q05, q95].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# regime_state tests
# ============================================================================


class TestRegimeState:
    """Test compute_regime_state_from_df."""

    @staticmethod
    def _make_df(
        atr_percentile=None,
        oi_zscore=None,
        funding_rate_abs_zscore_50=None,
        n: int = 5,
    ) -> pd.DataFrame:
        data = {}
        if atr_percentile is not None:
            data["atr_percentile"] = (
                atr_percentile
                if hasattr(atr_percentile, "__len__")
                else [atr_percentile] * n
            )
        if oi_zscore is not None:
            data["oi_zscore"] = (
                oi_zscore if hasattr(oi_zscore, "__len__") else [oi_zscore] * n
            )
        if funding_rate_abs_zscore_50 is not None:
            data["funding_rate_abs_zscore_50"] = (
                funding_rate_abs_zscore_50
                if hasattr(funding_rate_abs_zscore_50, "__len__")
                else [funding_rate_abs_zscore_50] * n
            )
        return pd.DataFrame(data)

    def test_normal_regime(self):
        """All normal conditions → regime_state = 0."""
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        df = self._make_df(
            atr_percentile=0.3, oi_zscore=0.5, funding_rate_abs_zscore_50=0.5
        )
        result = compute_regime_state_from_df(df)
        assert "regime_state" in result.columns
        assert (result["regime_state"] == 0).all()

    def test_high_vol_regime(self):
        """atr_percentile > 0.7 → regime_state = 1 (HIGH_VOL)."""
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        df = self._make_df(
            atr_percentile=0.85, oi_zscore=0.5, funding_rate_abs_zscore_50=0.5
        )
        result = compute_regime_state_from_df(df)
        assert (result["regime_state"] == 1).all()

    def test_high_leverage_regime(self):
        """oi_zscore > 1.5 AND funding_abs_zscore > 2.0 → regime_state = 2 (HIGH_LEVERAGE)."""
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        df = self._make_df(
            atr_percentile=0.85, oi_zscore=2.0, funding_rate_abs_zscore_50=3.0
        )
        result = compute_regime_state_from_df(df)
        # HIGH_LEVERAGE (2) overrides HIGH_VOL (1)
        assert (result["regime_state"] == 2).all()

    def test_high_leverage_overrides_high_vol(self):
        """HIGH_LEVERAGE takes precedence even when HIGH_VOL also true."""
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        df = self._make_df(
            atr_percentile=0.9, oi_zscore=2.5, funding_rate_abs_zscore_50=4.0
        )
        result = compute_regime_state_from_df(df)
        assert (result["regime_state"] == 2).all()

    def test_missing_columns_defaults_to_normal(self):
        """If all required columns are missing, defaults to NORMAL (0)."""
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        df = pd.DataFrame({"close": [100, 101, 102]})
        result = compute_regime_state_from_df(df)
        assert (result["regime_state"] == 0).all()

    def test_partial_columns(self):
        """Only atr_percentile present → can detect HIGH_VOL but not HIGH_LEVERAGE."""
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        df = self._make_df(atr_percentile=0.9)
        result = compute_regime_state_from_df(df)
        assert (result["regime_state"] == 1).all()

    def test_mixed_rows(self):
        """Different rows have different regimes."""
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        df = self._make_df(
            atr_percentile=[0.3, 0.8, 0.9, 0.5, 0.4],
            oi_zscore=[0.5, 0.5, 2.0, 0.5, 0.5],
            funding_rate_abs_zscore_50=[0.5, 0.5, 3.0, 0.5, 0.5],
        )
        result = compute_regime_state_from_df(df)
        expected = [0, 1, 2, 0, 0]
        np.testing.assert_array_equal(result["regime_state"].values, expected)

    def test_output_shape(self):
        """Output DataFrame has same index as input."""
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        idx = pd.date_range("2024-01-01", periods=10, freq="4h")
        df = pd.DataFrame({"atr_percentile": np.random.rand(10)}, index=idx)
        result = compute_regime_state_from_df(df)
        assert len(result) == 10
        assert result.index.equals(idx)


# ============================================================================
# ood_score tests
# ============================================================================


class TestOodScore:
    """Test compute_ood_score_from_df."""

    @staticmethod
    def _baseline():
        return {
            "feat_a": {"q05": 0.0, "q95": 1.0},
            "feat_b": {"q05": -1.0, "q95": 1.0},
            "feat_c": {"q05": 10.0, "q95": 20.0},
        }

    def test_all_in_distribution(self):
        """All features within [q05, q95] → ood_score ≈ 0."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        df = pd.DataFrame({"feat_a": [0.5], "feat_b": [0.0], "feat_c": [15.0]})
        result = compute_ood_score_from_df(df, baseline=self._baseline())
        assert result["ood_score"].iloc[0] == pytest.approx(0.0)

    def test_all_out_of_distribution(self):
        """All features outside [q05, q95] → ood_score = 1.0."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        df = pd.DataFrame({"feat_a": [5.0], "feat_b": [5.0], "feat_c": [50.0]})
        result = compute_ood_score_from_df(df, baseline=self._baseline())
        assert result["ood_score"].iloc[0] == pytest.approx(1.0)

    def test_partial_ood(self):
        """1 of 3 features OOD → ood_score ≈ 0.333."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        df = pd.DataFrame({"feat_a": [5.0], "feat_b": [0.0], "feat_c": [15.0]})
        result = compute_ood_score_from_df(df, baseline=self._baseline())
        assert result["ood_score"].iloc[0] == pytest.approx(1 / 3, abs=0.01)

    def test_no_baseline_returns_zero(self):
        """If no baseline provided, ood_score defaults to 0."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        df = pd.DataFrame({"feat_a": [5.0]})
        result = compute_ood_score_from_df(df, baseline=None)
        assert (result["ood_score"] == 0.0).all()

    def test_missing_features_ignored(self):
        """Features not in df are ignored in OOD count."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        baseline = {
            "feat_a": {"q05": 0.0, "q95": 1.0},
            "feat_missing": {"q05": 0.0, "q95": 1.0},
        }
        df = pd.DataFrame({"feat_a": [0.5]})
        result = compute_ood_score_from_df(df, baseline=baseline)
        # Only feat_a checked, it's in-distribution → 0/1 = 0
        assert result["ood_score"].iloc[0] == pytest.approx(0.0)

    def test_nan_values_not_counted_as_ood(self):
        """NaN values should not be counted as OOD."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        df = pd.DataFrame({"feat_a": [np.nan], "feat_b": [0.0], "feat_c": [15.0]})
        result = compute_ood_score_from_df(df, baseline=self._baseline())
        # feat_a=NaN → not OOD, feat_b=in, feat_c=in → 0/3 = 0
        assert result["ood_score"].iloc[0] == pytest.approx(0.0)

    def test_multiple_rows(self):
        """Each row gets its own ood_score."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        df = pd.DataFrame(
            {
                "feat_a": [0.5, 5.0],  # in, out
                "feat_b": [0.0, 5.0],  # in, out
                "feat_c": [15.0, 15.0],  # in, in
            }
        )
        result = compute_ood_score_from_df(df, baseline=self._baseline())
        assert result["ood_score"].iloc[0] == pytest.approx(0.0)
        assert result["ood_score"].iloc[1] == pytest.approx(2 / 3, abs=0.01)

    def test_ood_score_bounded(self):
        """ood_score should always be in [0, 1]."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        df = pd.DataFrame({"feat_a": [100.0], "feat_b": [100.0], "feat_c": [100.0]})
        result = compute_ood_score_from_df(df, baseline=self._baseline())
        assert 0.0 <= result["ood_score"].iloc[0] <= 1.0

    def test_p5_p95_key_fallback(self):
        """Baseline can use p5/p95 keys (from training_baseline.json format)."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        baseline = {"feat_a": {"p5": 0.0, "p95": 1.0, "mean": 0.5, "std": 0.3}}
        df = pd.DataFrame({"feat_a": [5.0]})
        result = compute_ood_score_from_df(df, baseline=baseline)
        assert result["ood_score"].iloc[0] == pytest.approx(1.0)
