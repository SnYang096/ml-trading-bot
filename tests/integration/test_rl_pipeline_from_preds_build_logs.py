import numpy as np
import pandas as pd
import pytest

from src.time_series_model.rl.build_execution_logs import (
    BuildExecutionLogsConfig,
    build_execution_logs,
)
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
def test_rl_pipeline_from_preds_build_logs(tmp_path) -> None:
    """
    End-to-end integration smoke:
      preds + raw(close) -> build_execution_logs -> shadow eval -> counterfactual eval -> FSM decision.

    Uses synthetic but structured price series to ensure:
      - multi-symbol joins work
      - build_logs creates required columns
      - artifacts are produced
    """
    n = 600
    ts = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")

    symbols = ["AAA", "BBB", "CCC"]
    parts_raw = []
    parts_pred = []
    rng = np.random.default_rng(7)

    for i, sym in enumerate(symbols):
        # Create gentle trend for each symbol with small noise
        base = 100.0 + 50.0 * i
        drift = 0.0008 if sym in {"AAA", "CCC"} else -0.0006  # BBB downtrend
        rets = drift + rng.normal(0.0, 0.0005, size=n)
        close = base * np.cumprod(1.0 + rets)

        parts_raw.append(pd.DataFrame({"symbol": sym, "timestamp": ts, "close": close}))

        # Make preds that strongly prefer TREND in rule router:
        # - dir_prob far from 0.5, mfe high, ttm large
        dir_prob = 0.9 if sym in {"AAA", "CCC"} else 0.1
        parts_pred.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "timestamp": ts,
                    "pred_dir_prob": [dir_prob] * n,
                    "pred_mfe_atr": np.log1p(np.full(n, 1.6)),  # > mfe_trend_min
                    "pred_mae_atr": np.log1p(np.full(n, 0.7)),
                    "pred_t_to_mfe": np.log1p(np.full(n, 18.0)),  # >= ttm_trend_min
                }
            )
        )

    raw = pd.concat(parts_raw, axis=0, ignore_index=True)
    preds = pd.concat(parts_pred, axis=0, ignore_index=True)

    cfg_logs = BuildExecutionLogsConfig(momentum_lookback=5, preds_in_log1p=True)
    logs = build_execution_logs(preds, raw_df=raw, cfg=cfg_logs)

    # Ensure required columns exist
    required_cols = {
        "symbol",
        "timestamp",
        "head_dir_score",
        "head_mfe_atr",
        "head_mae_atr",
        "head_t_to_mfe",
        "drawdown",
        "ret_mean",
        "ret_trend",
    }
    assert required_cols.issubset(set(logs.columns))
    assert len(logs) > 0

    split_cfg = WalkForwardSplitConfig(train_ratio=0.7)

    # 1) shadow eval
    shadow_dir = tmp_path / "shadow"
    _, _, shadow_metrics = train_and_shadow_eval_bc3_from_logs(
        logs, cfg=ShadowEvalConfig(split_cfg=split_cfg), out_dir=str(shadow_dir)
    )
    assert (shadow_dir / "shadow_report.html").exists()
    # On synthetic deterministic-ish labels, BC should reach high accuracy quickly.
    assert shadow_metrics["acc_vs_rule_mode"] > 0.7

    # 2) counterfactual eval
    cf_dir = tmp_path / "counterfactual"
    _, cf_metrics, _ = train_and_counterfactual_eval_bc3(
        logs,
        cfg=CounterfactualEvalConfig(
            split_cfg=split_cfg,
            sim_cfg=SimEnvConfig(
                entry_delay=0, cost_per_turnover=0.0, slippage_bps=0.0
            ),
        ),
        out_dir=str(cf_dir),
    )
    assert (cf_dir / "report.html").exists()
    assert (cf_dir / "metrics.json").exists()
    assert float(cf_metrics["pred_avg_total_return"]) != float("nan")

    # 3) FSM decision (relaxed drift gate to avoid random blocking)
    dd_rule = float(cf_metrics["rule_avg_max_dd"])
    dd_rl = float(cf_metrics["pred_avg_max_dd"])
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
    fsm = FallbackFSM(
        cfg=GateConfig(
            promote_min_days=1, cooldown_days=1, pnl_dd_margin=1.0, dd_ratio_max=100.0
        )
    )
    fsm.state = RouterControlState.RL_CANDIDATE
    out = fsm.step(gate_inp)
    assert "state" in out
