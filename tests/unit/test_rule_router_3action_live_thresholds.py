import pandas as pd
import pytest

from src.time_series_model.rule.router_3action import (
    Rule3ActionConfig,
    compute_mode_3action,
)


@pytest.mark.unit
def test_rule_router_3action_mean_vs_trend_boundaries():
    cfg = Rule3ActionConfig(
        mfe_min=0.4,
        eff_min=1.05,
        dir_conf_trend_min=0.25,
        mfe_trend_min=0.8,
        ttm_trend_min=8.0,
        eff_mean_min=1.15,
        ttm_mean_max=12.0,
    )
    # preds_in_log1p=False => values are already in ATR units
    df = pd.DataFrame(
        [
            # mean: tradable, not trend, eff high, ttm low
            {
                "pred_dir_prob": 0.51,
                "pred_mfe_atr": 0.6,
                "pred_mae_atr": 0.4,
                "pred_t_to_mfe": 5.0,
            },
            # trend: strong dir conf, mfe high, ttm high
            {
                "pred_dir_prob": 0.95,
                "pred_mfe_atr": 1.0,
                "pred_mae_atr": 0.3,
                "pred_t_to_mfe": 10.0,
            },
            # no_trade: mfe below mfe_min
            {
                "pred_dir_prob": 0.9,
                "pred_mfe_atr": 0.1,
                "pred_mae_atr": 0.1,
                "pred_t_to_mfe": 3.0,
            },
        ]
    )
    out = compute_mode_3action(df, cfg=cfg, preds_in_log1p=False)
    assert out["mode"].tolist() == ["MEAN", "TREND", "NO_TRADE"]
