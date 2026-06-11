"""Unit tests for regime-aware monitor helpers."""

from __future__ import annotations

import pandas as pd

from src.monitoring.regime_health import (
    evaluate_regime_share_drift,
    has_labeled_regime_schema,
    regime_shares_from_window,
)

TPC_LABELED_REGIME = {
    "allowed_regimes": {
        "bull": {
            "match": "all",
            "rules": [
                {"feature": "adx_50", "operator": ">=", "value": 25},
                {"feature": "ema_1200_position", "operator": ">=", "value": 0.1},
            ],
        },
        "bear": {
            "match": "any",
            "rules": [
                {"feature": "adx_50", "operator": "<=", "value": 20},
                {"feature": "ema_1200_position", "operator": "<=", "value": -0.1},
            ],
        },
        "neutral": {
            "match": "any",
            "rules": [{"feature": "adx_50", "operator": "<=", "value": 25}],
        },
    }
}


def test_has_labeled_regime_schema_true_for_tpc_style():
    assert has_labeled_regime_schema(TPC_LABELED_REGIME) is True
    assert has_labeled_regime_schema({"allowed_regimes": ["bull", "bear"]}) is False


def test_regime_shares_classify_rows():
    df = pd.DataFrame(
        {
            "adx_50": [30.0, 15.0, 10.0],
            "ema_1200_position": [0.2, -0.2, 0.0],
        }
    )
    shares = regime_shares_from_window(df, TPC_LABELED_REGIME)
    assert shares["bull"] == 1 / 3
    assert shares["bear"] == 2 / 3
    assert shares["neutral"] == 0.0


def test_regime_share_drift_alerts_on_delta():
    df = pd.DataFrame(
        {
            "adx_50": [30.0] * 100,
            "ema_1200_position": [0.2] * 100,
        }
    )
    r = evaluate_regime_share_drift(
        strategy="tpc",
        regime_yaml=TPC_LABELED_REGIME,
        window_df=df,
        baseline_entry={"regime_shares": {"bull": 0.0, "bear": 0.5, "neutral": 0.5}},
        share_tol=0.10,
    )
    assert r["status"] == "ALERT"
    assert any("REGIME_SHARE_DRIFT" in a for a in r.get("alerts") or [])


def test_regime_share_drift_baseline_missing_not_alert():
    df = pd.DataFrame({"adx_50": [10.0], "ema_1200_position": [0.0]})
    r = evaluate_regime_share_drift(
        strategy="tpc",
        regime_yaml=TPC_LABELED_REGIME,
        window_df=df,
        baseline_entry=None,
    )
    assert r["any_alert"] is False
    assert r["status"] == "BASELINE_MISSING"
