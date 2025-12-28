import numpy as np
import pandas as pd

from src.time_series_model.rl.counterfactual_eval_3action import (
    CounterfactualEvalConfig,
    train_and_counterfactual_eval_bc3,
)
from src.time_series_model.rl.sim_env_3action import SimEnvConfig
from src.time_series_model.rl.walk_forward import WalkForwardSplitConfig


def test_counterfactual_eval_smoke(tmp_path) -> None:
    rng = np.random.default_rng(0)
    n = 600
    ts = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC").astype(str)
    symbols = np.where(np.arange(n) % 2 == 0, "BTC", "ETH")

    # Build a learnable mapping and consistent returns:
    dir_score = rng.normal(0, 1, size=n)
    mfe = np.abs(rng.normal(1.0, 0.3, size=n))
    mae = np.abs(rng.normal(0.8, 0.3, size=n))
    ttm = np.abs(rng.normal(1.0, 0.2, size=n))

    mode = np.where(mfe < 0.6, "NO_TRADE", np.where(dir_score > 0, "TREND", "MEAN"))

    # Returns: MEAN pays +0.5% on average, TREND pays +0.8% on average; add small noise.
    ret_mean = 0.005 + rng.normal(0, 0.001, size=n)
    ret_trend = 0.008 + rng.normal(0, 0.001, size=n)

    df = pd.DataFrame(
        {
            "symbol": symbols,
            "timestamp": ts,
            "mode": mode,
            "head_dir_score": dir_score,
            "head_mfe_atr": mfe,
            "head_mae_atr": mae,
            "head_t_to_mfe": ttm,
            "drawdown": np.zeros(n),
            "ret_mean": ret_mean,
            "ret_trend": ret_trend,
        }
    )

    cfg = CounterfactualEvalConfig(
        state_keys=(
            "head_dir_score",
            "head_mfe_atr",
            "head_mae_atr",
            "head_t_to_mfe",
            "drawdown",
        ),
        split_cfg=WalkForwardSplitConfig(train_ratio=0.7),
        sim_cfg=SimEnvConfig(
            entry_delay=0, cost_per_turnover=0.0, slippage_bps=0.0, initial_equity=1.0
        ),
    )
    _, metrics, per_symbol = train_and_counterfactual_eval_bc3(
        df, cfg=cfg, out_dir=str(tmp_path / "cf")
    )

    assert metrics["test_symbols"] == 2.0
    assert len(per_symbol) == 2
    assert (tmp_path / "cf" / "report.html").exists()
    assert (tmp_path / "cf" / "metrics.json").exists()
    # Newly added production-grade metrics
    for k in [
        "rule_sharpe_mean",
        "pred_sharpe_mean",
        "rule_sortino_mean",
        "pred_sortino_mean",
        "rule_ann_return_mean",
        "pred_ann_return_mean",
        "rule_ann_vol_mean",
        "pred_ann_vol_mean",
        "rule_score",
        "pred_score",
    ]:
        assert k in metrics
        assert metrics[k] == metrics[k]  # not NaN
    assert isinstance(metrics.get("score_formula"), str)
