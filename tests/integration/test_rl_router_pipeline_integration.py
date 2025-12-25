import numpy as np
import pandas as pd
import pytest

from src.time_series_model.rl.counterfactual_eval_3action import (
    CounterfactualEvalConfig,
    train_and_counterfactual_eval_bc3,
)
from src.time_series_model.rl.fallback_fsm import (
    FallbackFSM,
    GateConfig,
    GateInputs,
    RouterControlState,
)
from src.time_series_model.rl.shadow_eval_3action import (
    ShadowEvalConfig,
    train_and_shadow_eval_bc3_from_logs,
)
from src.time_series_model.rl.sim_env_3action import SimEnvConfig
from src.time_series_model.rl.walk_forward import WalkForwardSplitConfig


@pytest.mark.integration
def test_rl_pipeline_shadow_counterfactual_fsm(tmp_path) -> None:
    """
    End-to-end integration smoke:
    logs(df) -> shadow eval -> counterfactual eval -> FSM decision.

    Uses synthetic data with positive mode returns to verify "can be profitable" in principle
    and that all artifacts are produced.
    """
    rng = np.random.default_rng(0)
    n = 1200
    ts = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC").astype(str)
    symbols = np.where(
        np.arange(n) % 3 == 0, "BTC", np.where(np.arange(n) % 3 == 1, "ETH", "SOL")
    )

    # heads/state
    dir_score = rng.normal(0, 1, size=n)
    mfe = np.abs(rng.normal(1.0, 0.3, size=n))
    mae = np.abs(rng.normal(0.8, 0.3, size=n))
    ttm = np.abs(rng.normal(1.0, 0.2, size=n))
    drawdown = np.zeros(n)

    # rule mode
    mode = np.where(mfe < 0.6, "NO_TRADE", np.where(dir_score > 0.0, "TREND", "MEAN"))

    # execution-derived returns (positive expectancy, with noise)
    ret_mean = 0.003 + rng.normal(0, 0.002, size=n)
    ret_trend = 0.005 + rng.normal(0, 0.002, size=n)

    df = pd.DataFrame(
        {
            "symbol": symbols,
            "timestamp": ts,
            "mode": mode,
            "head_dir_score": dir_score,
            "head_mfe_atr": mfe,
            "head_mae_atr": mae,
            "head_t_to_mfe": ttm,
            "drawdown": drawdown,
            "ret_mean": ret_mean,
            "ret_trend": ret_trend,
        }
    )

    split_cfg = WalkForwardSplitConfig(train_ratio=0.7)

    # 1) shadow eval (behavior gate)
    shadow_dir = tmp_path / "shadow"
    _, _, shadow_metrics = train_and_shadow_eval_bc3_from_logs(
        df,
        cfg=ShadowEvalConfig(split_cfg=split_cfg),
        out_dir=str(shadow_dir),
    )
    assert (shadow_dir / "shadow_report.html").exists()
    assert shadow_metrics["acc_vs_rule_mode"] > 0.7

    # 2) counterfactual eval (PnL gate) using mode-returns
    cf_dir = tmp_path / "counterfactual"
    _, cf_metrics, per_symbol = train_and_counterfactual_eval_bc3(
        df,
        cfg=CounterfactualEvalConfig(
            split_cfg=split_cfg,
            sim_cfg=SimEnvConfig(
                entry_delay=0, cost_per_turnover=0.0, slippage_bps=0.0
            ),
        ),
        out_dir=str(cf_dir),
    )
    assert (cf_dir / "report.html").exists()
    assert len(per_symbol) >= 2
    # "can be profitable" sanity on synthetic data
    assert cf_metrics["pred_avg_total_return"] > 0.0

    # 3) FSM: candidate -> active given good gates
    # Build a single "window" gate input from aggregated metrics
    dd_rule = float(cf_metrics["rule_avg_max_dd"])
    dd_rl = float(cf_metrics["pred_avg_max_dd"])
    # Avoid div-by-zero; drift gate is relaxed in this integration test anyway.
    pnl_dd_rule = (
        None if dd_rule <= 0 else float(cf_metrics["rule_avg_total_return"] / dd_rule)
    )
    pnl_dd_rl = (
        None if dd_rl <= 0 else float(cf_metrics["pred_avg_total_return"] / dd_rl)
    )
    gate_inp = GateInputs(
        max_dd_rule=dd_rule,
        max_dd_rl=dd_rl,
        switch_rate_rule=float(cf_metrics["rule_avg_switch_rate"]),
        switch_rate_rl=float(cf_metrics["pred_avg_switch_rate"]),
        pnl_dd_rule=pnl_dd_rule,
        pnl_dd_rl=pnl_dd_rl,
    )

    # Integration test purpose: validate end-to-end wiring. Use a relaxed drift gate
    # so that minor random differences don't block promotion.
    fsm = FallbackFSM(
        cfg=GateConfig(
            promote_min_days=2, cooldown_days=3, pnl_dd_margin=1.0, dd_ratio_max=100.0
        )
    )
    fsm.state = RouterControlState.RL_CANDIDATE
    fsm.step(gate_inp)
    out = fsm.step(gate_inp)
    assert out["state"] == RouterControlState.RL_ACTIVE.value
