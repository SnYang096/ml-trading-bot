import pandas as pd
import pytest

from src.time_series_model.portfolio.portfolio_assets_artifacts import (
    build_portfolio_assets_v1_artifacts_from_modes,
)


def _write_cfg(tmp_path):
    p = tmp_path / "pa.yaml"
    p.write_text(
        """
name: pa_test
assets:
  GLOBAL_TREND: {min_weight: 0.0, max_weight: 0.4}
  GLOBAL_MEAN: {min_weight: 0.2, max_weight: 0.35}
  GLOBAL_CASH: {min_weight: 0.1, max_weight: 1.0}
  HIGH_BETA_OVERLAY: {min_weight: 0.0, max_weight: 0.1}
  DEFENSIVE_MEAN: {min_weight: 0.0, max_weight: 0.25}
router_to_weights:
  global_trend: {p_trend_min: 0.6, regime_entropy_max: 0.9, max_weight: 0.4, crowding_penalty: false}
  global_mean: {base_floor: 0.2, max_weight: 0.35}
  global_cash: {min_weight: 0.1}
  defensive_mean: {regime_entropy_min: 0.5, max_weight: 0.25}
  high_beta_overlay: {p_trend_min: 0.75, crowding_max: 0.3, confidence_min: 0.7, max_weight: 0.1}
trend_zero_law:
  rules:
    - {name: dd, portfolio_drawdown_gt: 0.5}
""",
        encoding="utf-8",
    )
    return str(p)


@pytest.mark.unit
def test_build_portfolio_assets_v1_artifacts_from_modes_smoke(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    df = pd.DataFrame(
        [
            {"timestamp": "2025-01-01 00:00:00", "symbol": "BTCUSDT", "mode": "TREND"},
            {"timestamp": "2025-01-01 00:00:00", "symbol": "ETHUSDT", "mode": "TREND"},
            {"timestamp": "2025-01-01 04:00:00", "symbol": "BTCUSDT", "mode": "MEAN"},
            {
                "timestamp": "2025-01-01 04:00:00",
                "symbol": "ETHUSDT",
                "mode": "NO_TRADE",
            },
        ]
    )
    art = build_portfolio_assets_v1_artifacts_from_modes(
        df,
        portfolio_assets_yaml=cfg_path,
        tail_points=10,
        key_symbols=("BTCUSDT", "ETHUSDT"),
        portfolio_drawdown=0.0,
    )
    assert isinstance(art.summary, dict)
    assert art.summary.get("n_timestamps", 0) == 2
    assert "avg_weights" in art.summary
    assert isinstance(art.timeseries_tail, pd.DataFrame)
    assert len(art.timeseries_tail) == 2
    # weights are present in tail
    assert any(c.startswith("w__") for c in art.timeseries_tail.columns)
