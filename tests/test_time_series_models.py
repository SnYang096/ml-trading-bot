import numpy as np
import pandas as pd
import pytest

from regime_detection.detector import RegimeLabel
from time_series_model.pipeline.training import regime_gating
from time_series_model.pipeline.training.regime_gating import (
    RegimeGatedTimeSeriesModel,
)
from time_series_model.pipeline.risk_management import RiskManager


class DummyModel:
    """Lightweight stand-in for LightGBMModel used in unit tests."""

    counter = 0

    def __init__(self, model_type: str = "regression", **kwargs):
        self.model_type = model_type
        DummyModel.counter += 1
        self.pred_value = float(DummyModel.counter)
        self.is_trained = False
        self.last_X = None
        self.last_y = None

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        self.is_trained = True
        self.last_X = X.copy()
        self.last_y = y.copy()
        return {"n_samples": int(len(y))}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("DummyModel used before training.")
        return np.full(len(X), self.pred_value, dtype=float)


@pytest.fixture(autouse=True)
def _reset_dummy_model_counter():
    DummyModel.counter = 0


def _make_time_series(length: int, freq: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=length, freq=freq)
    close = np.linspace(100, 110, num=length) + rng.normal(0, 0.2, size=length)
    df = pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.05, size=length),
            "high": close + rng.normal(0.1, 0.05, size=length),
            "low": close - rng.normal(0.1, 0.05, size=length),
            "close": close,
            "volume": rng.uniform(800, 1200, size=length),
        },
        index=idx,
    )
    return df


def test_regime_gated_model_trains_and_predicts(monkeypatch):
    # Prepare synthetic multi-timeframe data
    df_60 = _make_time_series(220, "60min")
    df_60["trend_ma_signal"] = np.linspace(0, 1, len(df_60))
    df_60["compression_metric"] = np.linspace(1, 0, len(df_60))

    df_15 = _make_time_series(220, "15min")
    df_15["bb_width_value"] = np.linspace(0.5, 1.5, len(df_15))
    df_15["rsi_value"] = np.linspace(30, 70, len(df_15))

    engineered = {"60T": df_60, "15T": df_15}

    # Patch LightGBMModel with dummy implementation
    monkeypatch.setattr(regime_gating, "LightGBMModel", DummyModel)

    # Patch regime detection to return deterministic labels
    def fake_detect(self, data_by_tf):
        return {
            "60T": pd.Series(RegimeLabel.TRENDING, index=engineered["60T"].index),
            "15T": pd.Series(RegimeLabel.RANGE, index=engineered["15T"].index),
        }

    monkeypatch.setattr(
        RegimeGatedTimeSeriesModel, "_detect_regimes", fake_detect, raising=False
    )

    model = RegimeGatedTimeSeriesModel(forward_bars=6, include_regime_features=True)
    metrics = model.train(engineered)

    # Momentum expert should train on 60T timeframe
    assert "momentum" in metrics and "60T" in metrics["momentum"]
    momentum_model = model.experts["momentum"]["60T"]
    assert isinstance(momentum_model, DummyModel)
    # Regime features should have been appended during training
    assert "regime_trending" in momentum_model.last_X.columns

    # Mean reversion expert should train on 15T timeframe
    assert "mean_reversion" in metrics and "15T" in metrics["mean_reversion"]
    mean_rev_model = model.experts["mean_reversion"]["15T"]
    assert isinstance(mean_rev_model, DummyModel)
    assert "regime_range" in mean_rev_model.last_X.columns

    # Prepare explicit regime probabilities for prediction
    probs_60 = pd.DataFrame(
        {
            RegimeLabel.RANGE.value: 0.0,
            RegimeLabel.PRE_BREAKOUT.value: 0.0,
            RegimeLabel.TRENDING.value: 1.0,
            RegimeLabel.COLLAPSE.value: 0.0,
            RegimeLabel.TRANSITION.value: 0.0,
        },
        index=engineered["60T"].index,
    )
    probs_15 = pd.DataFrame(
        {
            RegimeLabel.RANGE.value: 1.0,
            RegimeLabel.PRE_BREAKOUT.value: 0.0,
            RegimeLabel.TRENDING.value: 0.0,
            RegimeLabel.COLLAPSE.value: 0.0,
            RegimeLabel.TRANSITION.value: 0.0,
        },
        index=engineered["15T"].index,
    )

    preds = model.predict(engineered, regime_probs={"60T": probs_60, "15T": probs_15})
    # Weighted predictions should align to indices
    assert preds["60T"].index.equals(engineered["60T"].index)
    assert preds["15T"].index.equals(engineered["15T"].index)
    # Dummy predictions should be constant per expert
    assert np.allclose(preds["60T"].values, momentum_model.pred_value)
    assert np.allclose(preds["15T"].values, mean_rev_model.pred_value)


def test_risk_manager_regime_sensitive_positioning():
    rm = RiskManager()
    idx = pd.date_range("2024-01-01", periods=8, freq="h")
    ensemble_df = pd.DataFrame(
        {
            "ensemble_return": [0.02, 0.015, 0.01, 0.005, -0.003, 0.012, 0.01, 0.008],
            "discrete_signal": [1, 1, 1, 1, 0, 1, 1, 1],
        },
        index=idx,
    )
    price_data = pd.DataFrame({"close": np.linspace(100, 101.4, len(idx))}, index=idx)
    regime_probs = pd.DataFrame(
        {
            RegimeLabel.RANGE.value: 0.1,
            RegimeLabel.PRE_BREAKOUT.value: 0.0,
            RegimeLabel.TRENDING.value: 0.8,
            RegimeLabel.COLLAPSE.value: 0.0,
            RegimeLabel.TRANSITION.value: 0.1,
        },
        index=idx,
    )

    result = rm.apply_risk_management(
        ensemble_df,
        price_data,
        regime_probs=regime_probs,
        account_value=250_000.0,
    )

    assert "position" in result.columns
    assert "risk_mode" in result.columns
    # Trending bias should place the manager in Aggressive mode
    assert result["risk_mode"].iloc[-1] == "Aggressive"
    assert not np.allclose(result["position"].values, 0.0)

    # Collapse-heavy probabilities should drive Defensive mode
    rm_collapse = RiskManager()
    collapse_probs = regime_probs.copy()
    collapse_probs[RegimeLabel.TRENDING.value] = 0.0
    collapse_probs[RegimeLabel.COLLAPSE.value] = 0.8
    collapse_result = rm_collapse.apply_risk_management(
        ensemble_df,
        price_data,
        regime_probs=collapse_probs,
        account_value=250_000.0,
    )
    assert collapse_result["risk_mode"].iloc[-1] == "Defensive"


def test_risk_manager_cooldown_and_antimartingale():
    rm = RiskManager()
    idx = pd.date_range("2024-02-01", periods=6, freq="h")
    # Strong positive signals to trigger anti-martingale escalation, then a loss
    ensemble_df = pd.DataFrame(
        {
            "ensemble_return": [0.03, 0.025, 0.02, -0.05, 0.01, 0.012],
            "discrete_signal": [1, 1, 1, 1, 1, 1],
        },
        index=idx,
    )
    # Prices rise then drop sharply to induce a loss
    price_data = pd.DataFrame({"close": [100, 102, 104, 99, 100, 101]}, index=idx)
    regime_probs = pd.DataFrame(
        {
            RegimeLabel.RANGE.value: 0.0,
            RegimeLabel.PRE_BREAKOUT.value: 0.0,
            RegimeLabel.TRENDING.value: 0.9,
            RegimeLabel.COLLAPSE.value: 0.0,
            RegimeLabel.TRANSITION.value: 0.1,
        },
        index=idx,
    )

    result = rm.apply_risk_management(
        ensemble_df,
        price_data,
        regime_probs=regime_probs,
        account_value=200_000.0,
    )

    # After two profitable bars the anti-martingale ladder should step up
    assert rm.add_count >= 1
    # The loss on the fourth bar should trigger cooldown
    assert rm.cooldown_remaining >= 5
    # Positions during cooldown should shrink compared to earlier bars
    assert abs(result["position"].iloc[4]) < abs(result["position"].iloc[1])

