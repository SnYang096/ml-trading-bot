from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RouterDiagnosticsConfig:
    symbol_col: str = "symbol"
    timestamp_col: str = "timestamp"
    mode_col: str = "mode"
    ret_mean_col: str = "ret_mean"
    ret_trend_col: str = "ret_trend"

    # Rolling drift
    rolling_window: int = 300
    rolling_min_periods: int = 60

    # Numerical stability
    eps: float = 1e-12


def _mode_to_action(mode: Any) -> int:
    m = "" if mode is None else str(mode).upper()
    if m in {"NO_TRADE", "NOTRADE", "OFF", "OBSERVE", "PAUSE"}:
        return 0
    if m in {"MEAN", "MEAN_REVERT", "MEANREVERT"}:
        return 1
    if m in {"TREND", "TREND_FOLLOW", "TRENDFOLLOW"}:
        return 2
    return 0


def _js_divergence(p: np.ndarray, q: np.ndarray, *, eps: float) -> float:
    """
    Jensen-Shannon divergence (base-e). Returns [0, ln(2)].
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p.clip(min=0.0)
    q = q.clip(min=0.0)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)

    def _kl(a, b):
        a = (a + eps) / (a.sum() + eps * a.size)
        b = (b + eps) / (b.sum() + eps * b.size)
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _action_probs(actions: np.ndarray) -> np.ndarray:
    counts = np.bincount(actions.astype(int), minlength=3).astype(float)
    s = float(counts.sum())
    return counts / s if s > 0 else np.array([1.0, 0.0, 0.0], dtype=float)


def _expected_step_reward(
    df: pd.DataFrame, *, cfg: RouterDiagnosticsConfig
) -> np.ndarray:
    """
    Reward implied by *taking the chosen mode* at each step:
      NO -> 0
      MEAN -> ret_mean
      TREND -> ret_trend
    """
    a = np.asarray([_mode_to_action(x) for x in df[cfg.mode_col].tolist()], dtype=int)
    r_mean = (
        pd.to_numeric(df.get(cfg.ret_mean_col), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    r_trend = (
        pd.to_numeric(df.get(cfg.ret_trend_col), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    out = np.zeros(len(df), dtype=float)
    out[a == 1] = r_mean[a == 1]
    out[a == 2] = r_trend[a == 2]
    return out


def diagnose_router_from_logs(
    df_logs: pd.DataFrame,
    *,
    cfg: RouterDiagnosticsConfig = RouterDiagnosticsConfig(),
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Returns: (meta, metrics, per_symbol_df, rolling_df)
    """
    for c in [cfg.symbol_col, cfg.timestamp_col, cfg.mode_col]:
        if c not in df_logs.columns:
            raise ValueError(f"Missing required column: {c}")
    if (
        cfg.ret_mean_col not in df_logs.columns
        or cfg.ret_trend_col not in df_logs.columns
    ):
        raise ValueError(
            f"Missing returns columns: {cfg.ret_mean_col}/{cfg.ret_trend_col} (run build-logs-3action first)"
        )

    df = df_logs.copy()
    df[cfg.timestamp_col] = pd.to_datetime(
        df[cfg.timestamp_col], utc=True, errors="coerce"
    )
    df = (
        df.dropna(subset=[cfg.timestamp_col])
        .sort_values([cfg.symbol_col, cfg.timestamp_col])
        .reset_index(drop=True)
    )

    # actions + implied reward
    df["_action"] = np.asarray(
        [_mode_to_action(x) for x in df[cfg.mode_col].tolist()], dtype=int
    )
    df["_reward"] = _expected_step_reward(df, cfg=cfg)

    # Per-symbol summary
    rows: List[Dict[str, Any]] = []
    pooled_p = _action_probs(df["_action"].to_numpy(dtype=int))
    for sym, g in df.groupby(cfg.symbol_col, sort=False):
        a = g["_action"].to_numpy(dtype=int)
        p = _action_probs(a)
        js = _js_divergence(p, pooled_p, eps=float(cfg.eps))

        # action -> reward mapping consistency (mean reward per action)
        r_mean = (
            float(g.loc[g["_action"] == 1, "_reward"].mean())
            if (g["_action"] == 1).any()
            else 0.0
        )
        r_trend = (
            float(g.loc[g["_action"] == 2, "_reward"].mean())
            if (g["_action"] == 2).any()
            else 0.0
        )
        r_no = (
            float(g.loc[g["_action"] == 0, "_reward"].mean())
            if (g["_action"] == 0).any()
            else 0.0
        )
        rows.append(
            {
                "symbol": str(sym),
                "n": int(len(g)),
                "p_no": float(p[0]),
                "p_mean": float(p[1]),
                "p_trend": float(p[2]),
                "js_to_pooled": float(js),
                "reward_no_mean": r_no,
                "reward_mean_mean": r_mean,
                "reward_trend_mean": r_trend,
                "reward_all_mean": float(g["_reward"].mean()) if len(g) else 0.0,
                "reward_all_std": (
                    float(g["_reward"].std(ddof=1)) if len(g) > 1 else 0.0
                ),
            }
        )

    per_symbol = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)

    # Rolling drift (per symbol): JS divergence between rolling action distribution and its long-run dist
    roll_rows: List[Dict[str, Any]] = []
    for sym, g in df.groupby(cfg.symbol_col, sort=False):
        g = g.reset_index(drop=True)
        base_p = _action_probs(g["_action"].to_numpy(dtype=int))
        win = int(cfg.rolling_window)
        minp = int(cfg.rolling_min_periods)
        for i in range(len(g)):
            lo = max(0, i - win + 1)
            w = g.iloc[lo : i + 1]
            if len(w) < minp:
                continue
            p_w = _action_probs(w["_action"].to_numpy(dtype=int))
            js_w = _js_divergence(p_w, base_p, eps=float(cfg.eps))
            roll_rows.append(
                {
                    "symbol": str(sym),
                    "timestamp": pd.Timestamp(g[cfg.timestamp_col].iloc[i]).isoformat(),
                    "roll_js_to_symbol_base": float(js_w),
                    "roll_p_no": float(p_w[0]),
                    "roll_p_mean": float(p_w[1]),
                    "roll_p_trend": float(p_w[2]),
                }
            )
    rolling = pd.DataFrame(roll_rows)

    metrics: Dict[str, Any] = {
        "symbols": int(per_symbol.shape[0]),
        "steps": int(df.shape[0]),
        "pooled_p_no": float(pooled_p[0]),
        "pooled_p_mean": float(pooled_p[1]),
        "pooled_p_trend": float(pooled_p[2]),
        "js_mean_to_pooled": (
            float(per_symbol["js_to_pooled"].mean()) if len(per_symbol) else 0.0
        ),
        "js_max_to_pooled": (
            float(per_symbol["js_to_pooled"].max()) if len(per_symbol) else 0.0
        ),
    }

    meta: Dict[str, Any] = {
        "cfg": {
            "symbol_col": cfg.symbol_col,
            "timestamp_col": cfg.timestamp_col,
            "mode_col": cfg.mode_col,
            "ret_mean_col": cfg.ret_mean_col,
            "ret_trend_col": cfg.ret_trend_col,
            "rolling_window": cfg.rolling_window,
            "rolling_min_periods": cfg.rolling_min_periods,
        }
    }
    return meta, metrics, per_symbol, rolling


def write_router_diagnostics_artifacts(
    *,
    out_dir: str,
    meta: Dict[str, Any],
    metrics: Dict[str, Any],
    per_symbol: pd.DataFrame,
    rolling: pd.DataFrame,
) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (p / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    per_symbol.to_csv(p / "per_symbol.csv", index=False)
    rolling.to_csv(p / "rolling.csv", index=False)

    # Minimal HTML (dependency-free)
    html = []
    html.append("<!doctype html><html><head><meta charset='utf-8'>")
    html.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    html.append("<title>Router diagnostics</title>")
    html.append(
        "<style>body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;margin:24px;color:#111}"
        "code{background:#f4f4f5;padding:2px 6px;border-radius:6px}"
        "table{border-collapse:collapse;width:100%;font-size:12px}"
        "th,td{border-bottom:1px solid #eee;text-align:left;padding:6px 8px;vertical-align:top}"
        "th{background:#fafafa}</style></head><body>"
    )
    html.append("<h1>Router diagnostics (multi-symbol)</h1>")
    html.append(
        "<p class='muted'>JS divergence + action→reward mapping + rolling drift.</p>"
    )
    html.append("<h2>Aggregate</h2>")
    html.append("<table><thead><tr><th>metric</th><th>value</th></tr></thead><tbody>")
    for k in sorted(metrics.keys()):
        html.append(f"<tr><td><code>{k}</code></td><td>{metrics[k]}</td></tr>")
    html.append("</tbody></table>")
    html.append("<h2>Per symbol</h2>")
    html.append(per_symbol.to_html(index=False, escape=True))
    html.append("<h2>Rolling drift (tail)</h2>")
    if len(rolling) > 0:
        html.append(rolling.tail(200).to_html(index=False, escape=True))
    else:
        html.append(
            "<p><em>No rolling rows (increase data length or lower rolling_min_periods).</em></p>"
        )
    html.append("</body></html>")
    (p / "report.html").write_text("\n".join(html), encoding="utf-8")
