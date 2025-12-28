from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .bc_dataset import Router3Action
from .sim_env_3action import SimEnvConfig, simulate_3action_episode


@dataclass(frozen=True)
class ExecControlConfig:
    """
    Execution control invariants / early-warning checks.

    This is NOT an alpha model. It is a production safety wrapper:
    - detect abnormal turnover/cost/drawdown behavior
    - detect data pathologies (NaNs, infinities)
    - provide a kill-switch decision (suspend -> NO_TRADE)
    """

    symbol_col: str = "symbol"
    timestamp_col: str = "timestamp"
    mode_col: str = "mode"

    # Required returns
    ret_mean_col: str = "ret_mean"
    ret_trend_col: str = "ret_trend"

    # Thresholds (invariants)
    max_nan_ratio: float = 0.001
    max_abs_return: float = 0.5  # per-step sanity cap (50% is extreme)
    max_dd: float = 0.35  # peak-to-trough drawdown limit
    max_turnover_mean: float = 0.35
    max_turnover_p95: float = 1.0
    max_cost_mean: float = 0.002  # 20 bps per step mean cost
    max_cost_p95: float = 0.01  # 100 bps p95 cost

    # Kill-switch policy
    kill_on_any_hard_violation: bool = True

    # Simulation assumptions (must match counterfactual_eval_3action if you want parity)
    sim_cfg: SimEnvConfig = SimEnvConfig()


def _mode_to_action(mode: Any) -> int:
    m = "" if mode is None else str(mode).upper()
    if m in {"NO_TRADE", "NOTRADE", "OFF", "OBSERVE", "PAUSE"}:
        return int(Router3Action.NO_TRADE)
    if m in {"MEAN", "MEAN_REVERT", "MEANREVERT"}:
        return int(Router3Action.MEAN)
    if m in {"TREND", "TREND_FOLLOW", "TRENDFOLLOW"}:
        return int(Router3Action.TREND)
    return int(Router3Action.NO_TRADE)


def _nan_ratio(x: pd.Series) -> float:
    if x is None or len(x) == 0:
        return 0.0
    a = pd.to_numeric(x, errors="coerce")
    return float(a.isna().mean())


def control_check_from_logs(
    df_logs: pd.DataFrame, *, cfg: ExecControlConfig = ExecControlConfig()
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Returns (metrics, per_symbol_df).
    """
    for c in [
        cfg.symbol_col,
        cfg.timestamp_col,
        cfg.mode_col,
        cfg.ret_mean_col,
        cfg.ret_trend_col,
    ]:
        if c not in df_logs.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df_logs.copy()
    df[cfg.timestamp_col] = pd.to_datetime(
        df[cfg.timestamp_col], utc=True, errors="coerce"
    )
    df = df.dropna(subset=[cfg.timestamp_col]).sort_values(
        [cfg.symbol_col, cfg.timestamp_col]
    )

    # Global data sanity
    nan_r = max(_nan_ratio(df[cfg.ret_mean_col]), _nan_ratio(df[cfg.ret_trend_col]))
    ret_abs_max = float(
        max(
            pd.to_numeric(df[cfg.ret_mean_col], errors="coerce").abs().max(skipna=True)
            or 0.0,
            pd.to_numeric(df[cfg.ret_trend_col], errors="coerce").abs().max(skipna=True)
            or 0.0,
        )
    )

    rows = []
    hard_any = False
    for sym, g in df.groupby(cfg.symbol_col, sort=False):
        g = g.reset_index(drop=True)
        actions = [_mode_to_action(m) for m in g[cfg.mode_col].tolist()]
        ep = simulate_3action_episode(g, actions=actions, cfg=cfg.sim_cfg)
        if ep.empty:
            continue
        dd_max = float(pd.to_numeric(ep["drawdown"], errors="coerce").fillna(0.0).max())
        turnover = (
            pd.to_numeric(ep["turnover"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        cost = (
            pd.to_numeric(ep["cost"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )

        t_mean = float(np.mean(turnover)) if len(turnover) else 0.0
        t_p95 = float(np.quantile(turnover, 0.95)) if len(turnover) else 0.0
        c_mean = float(np.mean(cost)) if len(cost) else 0.0
        c_p95 = float(np.quantile(cost, 0.95)) if len(cost) else 0.0

        hard = {
            "hard_dd": dd_max > float(cfg.max_dd),
            "hard_turnover_mean": t_mean > float(cfg.max_turnover_mean),
            "hard_turnover_p95": t_p95 > float(cfg.max_turnover_p95),
            "hard_cost_mean": c_mean > float(cfg.max_cost_mean),
            "hard_cost_p95": c_p95 > float(cfg.max_cost_p95),
        }
        hard_violation = bool(any(hard.values()))
        hard_any = hard_any or hard_violation

        rows.append(
            {
                "symbol": str(sym),
                "n": int(len(g)),
                "dd_max": dd_max,
                "turnover_mean": t_mean,
                "turnover_p95": t_p95,
                "cost_mean": c_mean,
                "cost_p95": c_p95,
                "hard_violation": hard_violation,
                **hard,
            }
        )

    per_symbol = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)

    data_bad = bool(
        nan_r > float(cfg.max_nan_ratio) or ret_abs_max > float(cfg.max_abs_return)
    )
    kill = bool((cfg.kill_on_any_hard_violation and hard_any) or data_bad)

    metrics: Dict[str, Any] = {
        "symbols": int(per_symbol.shape[0]),
        "steps": int(df.shape[0]),
        "nan_ratio": float(nan_r),
        "ret_abs_max": float(ret_abs_max),
        "hard_any": bool(hard_any),
        "data_bad": bool(data_bad),
        "kill_switch": bool(kill),
        "cfg": asdict(cfg),
    }
    return metrics, per_symbol


def write_exec_control_artifacts(
    *, out_dir: str, metrics: Dict[str, Any], per_symbol: pd.DataFrame
) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    per_symbol.to_csv(p / "per_symbol.csv", index=False)

    html = []
    html.append("<!doctype html><html><head><meta charset='utf-8'>")
    html.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    html.append("<title>Execution control check</title>")
    html.append(
        "<style>body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;margin:24px;color:#111}"
        "code{background:#f4f4f5;padding:2px 6px;border-radius:6px}"
        "table{border-collapse:collapse;width:100%;font-size:12px}"
        "th,td{border-bottom:1px solid #eee;text-align:left;padding:6px 8px;vertical-align:top}"
        "th{background:#fafafa}</style></head><body>"
    )
    html.append("<h1>Execution control check</h1>")
    html.append("<h2>Summary</h2>")
    html.append("<table><thead><tr><th>metric</th><th>value</th></tr></thead><tbody>")
    for k in [
        "kill_switch",
        "data_bad",
        "hard_any",
        "nan_ratio",
        "ret_abs_max",
        "symbols",
        "steps",
    ]:
        html.append(f"<tr><td><code>{k}</code></td><td>{metrics.get(k)}</td></tr>")
    html.append("</tbody></table>")
    html.append("<h2>Per symbol</h2>")
    html.append(per_symbol.to_html(index=False, escape=True))
    html.append("</body></html>")
    (p / "report.html").write_text("\n".join(html), encoding="utf-8")
