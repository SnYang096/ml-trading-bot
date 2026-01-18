#!/usr/bin/env python3
"""
Threshold plateau tuning protocol for Rule Router (3-action).

Goal:
  Avoid "sharp peak" overfitting by selecting thresholds that sit on a flat, robust plateau:
  - consistent across multiple OOS sub-windows (walk-forward slices)
  - stable under bootstrap resampling
  - low local sensitivity (small threshold nudges don't flip the result)

Inputs:
  - preds: preds_*.parquet (nnmultihead outputs per symbol; includes timestamp + pred_*)
  - logs:  logs_3action.parquet (per-step ret_mean/ret_trend already computed, e.g. rr_execution)
  - model: model.pt (to infer preds_in_log1p flag)

Outputs (out dir):
  - candidates.csv
  - summary.json
  - report.md
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")  # headless backend for CI/servers
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.time_series_model.rl.sim_env_3action import (
    SimEnvConfig,
    simulate_3action_episode,
)
from src.time_series_model.rule.router_3action import (
    Rule3ActionConfig,
    compute_mode_3action,
)


ROUTER_KEYS = [
    "mfe_min",
    "eff_min",
    "dir_conf_trend_min",
    "mfe_trend_min",
    "ttm_trend_min",
    "eff_mean_min",
    "ttm_mean_max",
]


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(x))))


def _compute_router_bounds(
    *,
    preds_by_sym: Dict[str, pd.DataFrame],
    preds_in_log1p: bool,
    qmin: float,
    qmax: float,
) -> Dict[str, Tuple[float, float]]:
    """
    Heuristic bounds from empirical distributions.
    Ensures thresholds stay within reasonable data-driven ranges.
    """
    rows: List[pd.DataFrame] = []
    for df in preds_by_sym.values():
        if df is None or df.empty:
            continue
        # Use default config; derived fields are independent of thresholds.
        diag = compute_mode_3action(
            df, cfg=Rule3ActionConfig(), preds_in_log1p=preds_in_log1p
        )
        rows.append(diag[["mfe_atr", "eff", "t_to_mfe", "dir_conf"]])
    if not rows:
        return {}
    all_df = pd.concat(rows, axis=0)
    bounds: Dict[str, Tuple[float, float]] = {}

    def _q(col: str) -> Tuple[float, float]:
        s = pd.to_numeric(all_df[col], errors="coerce").dropna()
        if s.empty:
            return (0.0, 0.0)
        return (float(s.quantile(qmin)), float(s.quantile(qmax)))

    mfe_lo, mfe_hi = _q("mfe_atr")
    eff_lo, eff_hi = _q("eff")
    ttm_lo, ttm_hi = _q("t_to_mfe")
    dconf_lo, dconf_hi = _q("dir_conf")
    bounds["mfe_min"] = (mfe_lo, mfe_hi)
    bounds["mfe_trend_min"] = (mfe_lo, mfe_hi)
    bounds["eff_min"] = (eff_lo, eff_hi)
    bounds["eff_mean_min"] = (eff_lo, eff_hi)
    bounds["ttm_trend_min"] = (ttm_lo, ttm_hi)
    bounds["ttm_mean_max"] = (ttm_lo, ttm_hi)
    bounds["dir_conf_trend_min"] = (dconf_lo, dconf_hi)
    return bounds


def _apply_bounds(
    cand: Dict[str, float], bounds: Dict[str, Tuple[float, float]]
) -> Dict[str, float]:
    if not bounds:
        return cand
    out = dict(cand)
    for k, (lo, hi) in bounds.items():
        if k in out and np.isfinite(lo) and np.isfinite(hi) and hi >= lo:
            out[k] = _clamp(out[k], lo, hi)
    return out


def _fig_to_data_uri(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _write_html_report(
    *,
    out_dir: Path,
    df: pd.DataFrame,
    best_thresholds: Dict[str, float],
    best_row: Dict[str, Any],
    plateau_frac: float,
    trade_rate_target: Optional[float] = None,
    trade_rate_tol: float = 0.0,
    trade_rate_min: Optional[float] = None,
    trade_rate_max: Optional[float] = None,
    trend_rate_target: Optional[float] = None,
    trend_rate_tol: float = 0.0,
    trend_rate_min: Optional[float] = None,
    trend_rate_max: Optional[float] = None,
) -> None:
    """
    Create a minimal HTML report to visually confirm “plateau” behavior.

    It embeds a few PNG plots as data URIs (no external deps beyond matplotlib).
    """
    if df is None or df.empty:
        return
    if "robust_score" not in df.columns:
        return

    dfx = df.reset_index(drop=True).copy()
    best_score = float(dfx["robust_score"].iloc[0])
    # Plateau cutoff must work for negative scores too.
    # If best_score < 0, multiplying by 0.95 makes it *less negative* (higher than best),
    # which would yield plateau_frac=0 by construction. Instead we use:
    #   cutoff = best_score - 0.05 * |best_score|
    cutoff = (
        (best_score - 0.05 * float(abs(best_score)))
        if np.isfinite(best_score)
        else float("nan")
    )

    # Plot 1: robust_score by rank
    fig1 = plt.figure(figsize=(9.0, 3.6))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(dfx["robust_score"].to_numpy(dtype=float), lw=1.2)
    ax1.axhline(
        cutoff,
        color="orange",
        lw=1.2,
        ls="--",
        label="within 5% of |best|",
    )
    ax1.set_title("robust_score by candidate rank (sorted, best at left)")
    ax1.set_xlabel("rank")
    ax1.set_ylabel("robust_score")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")
    img1 = _fig_to_data_uri(fig1)

    # Plot 2: trade_rate_mean vs robust_score (if present)
    img2 = ""
    if "trade_rate_mean" in dfx.columns:
        fig2 = plt.figure(figsize=(6.2, 4.2))
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.scatter(
            dfx["trade_rate_mean"].to_numpy(dtype=float),
            dfx["robust_score"].to_numpy(dtype=float),
            s=10,
            alpha=0.45,
        )
        ax2.scatter(
            [float(dfx["trade_rate_mean"].iloc[0])],
            [float(dfx["robust_score"].iloc[0])],
            s=60,
            color="red",
            label="best",
        )
        ax2.axhline(cutoff, color="orange", lw=1.0, ls="--")
        # Optional target band / min-max band
        try:
            if trade_rate_min is not None:
                ax2.axvline(float(trade_rate_min), color="gray", lw=1.0, ls=":")
            if trade_rate_max is not None:
                ax2.axvline(float(trade_rate_max), color="gray", lw=1.0, ls=":")
            if trade_rate_target is not None and np.isfinite(float(trade_rate_target)):
                t = float(trade_rate_target)
                tol = float(abs(trade_rate_tol))
                ax2.axvline(t, color="gray", lw=1.0, ls="--")
                if tol > 0:
                    ax2.axvspan(t - tol, t + tol, color="gray", alpha=0.10)
        except Exception:
            pass
        ax2.set_title("trade_rate_mean vs robust_score")
        ax2.set_xlabel("trade_rate_mean")
        ax2.set_ylabel("robust_score")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best")
        img2 = _fig_to_data_uri(fig2)

    # Plot 3: histogram of robust_score
    fig3 = plt.figure(figsize=(6.2, 3.6))
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.hist(dfx["robust_score"].to_numpy(dtype=float), bins=40, alpha=0.85)
    ax3.axvline(best_score, color="red", lw=1.2, label="best")
    ax3.axvline(cutoff, color="orange", lw=1.2, ls="--", label="plateau cutoff")
    ax3.set_title("robust_score distribution")
    ax3.set_xlabel("robust_score")
    ax3.set_ylabel("count")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="best")
    img3 = _fig_to_data_uri(fig3)

    def _fmt(x) -> str:
        try:
            if x is None:
                return "null"
            return f"{float(x):.6g}"
        except Exception:
            return str(x)

    rows = "\n".join(
        f"<tr><td><code>{k}</code></td><td>{_fmt(v)}</td></tr>"
        for k, v in best_thresholds.items()
    )

    best_meta_rows = []
    for k in [
        "robust_score",
        "win_score_mean",
        "win_score_p25",
        "win_sharpe_mean",
        "win_sharpe_p25",
        "win_dd_mean",
        "trade_rate_mean",
        "trade_rate_p25",
        "trade_rate_pen_mean",
        "trend_rate_mean",
        "trend_rate_p25",
        "trend_rate_pen_mean",
    ]:
        if k in best_row:
            best_meta_rows.append(f"<li><b>{k}</b>: {_fmt(best_row.get(k))}</li>")
    best_meta_html = "\n".join(best_meta_rows)

    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>Threshold plateau report</title>
    <style>
      body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin: 24px; }}
      code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
      .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
      img {{ max-width: 100%; border: 1px solid #eee; border-radius: 8px; }}
      table {{ border-collapse: collapse; }}
      td, th {{ border: 1px solid #eee; padding: 6px 10px; }}
      .muted {{ color: #666; }}
    </style>
  </head>
  <body>
    <h2>Threshold plateau report (Rule Router 3-action)</h2>
    <p class="muted">
      A “plateau” means many candidates near the best score (not a single sharp peak).
      We use <code>plateau_frac</code> = fraction of candidates with <code>robust_score</code> ≥ (best − 5%·|best|).
    </p>
    <ul>
      <li><b>n_candidates</b>: {len(dfx)}</li>
      <li><b>best_robust_score</b>: {_fmt(best_score)}</li>
      <li><b>plateau_cutoff</b>: {_fmt(cutoff)}</li>
      <li><b>plateau_frac</b>: <b>{_fmt(plateau_frac)}</b></li>
      <li><b>trade_rate_target</b>: {_fmt(trade_rate_target)}</li>
      <li><b>trade_rate_tol</b>: {_fmt(trade_rate_tol)}</li>
      <li><b>trade_rate_min</b>: {_fmt(trade_rate_min)}</li>
      <li><b>trade_rate_max</b>: {_fmt(trade_rate_max)}</li>
      <li><b>trend_rate_target</b>: {_fmt(trend_rate_target)}</li>
      <li><b>trend_rate_tol</b>: {_fmt(trend_rate_tol)}</li>
      <li><b>trend_rate_min</b>: {_fmt(trend_rate_min)}</li>
      <li><b>trend_rate_max</b>: {_fmt(trend_rate_max)}</li>
    </ul>
    <h3>Best candidate diagnostics</h3>
    <ul>
      {best_meta_html}
    </ul>
    <h3>Best thresholds (robust)</h3>
    <table>
      <thead><tr><th>key</th><th>value</th></tr></thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    <h3>Plots</h3>
    <div class="grid">
      <div><img src="{img1}" alt="robust_score_by_rank"/></div>
      {f'<div><img src="{img2}" alt="trade_rate_vs_score"/></div>' if img2 else ''}
      <div><img src="{img3}" alt="robust_score_hist"/></div>
    </div>
    <h3>Files</h3>
    <ul>
      <li><code>candidates.csv</code> (all candidates)</li>
      <li><code>summary.json</code> (machine-readable summary)</li>
      <li><code>router_thresholds_best.json</code> (best thresholds to feed back into pipeline)</li>
      <li><code>report.md</code> (text report)</li>
    </ul>
  </body>
</html>
"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def _read_parquet_or_csv(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _collect_pred_files(preds_path: Path) -> List[Path]:
    if preds_path.is_dir():
        files = sorted(preds_path.glob("preds_*.parquet"))
        if not files:
            files = sorted(preds_path.glob("*.parquet"))
        return files
    return [preds_path]


def _ensure_timestamp_col(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        if "timestamp" in getattr(df.index, "names", []):
            return df.reset_index(drop=True)
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out["timestamp"] = out.index
        return out.reset_index(drop=True)
    raise ValueError("Expected preds to have a timestamp column or DatetimeIndex.")


def _infer_steps_per_year(ts: pd.Series) -> float:
    dt = pd.to_datetime(ts, utc=True, errors="coerce").dropna()
    dt = dt.sort_values().drop_duplicates()
    if len(dt) < 3:
        return 1.0
    diffs = dt.diff().dropna()
    if len(diffs) == 0:
        return 1.0
    step_sec = float(diffs.median().total_seconds())
    if not np.isfinite(step_sec) or step_sec <= 0:
        return 1.0
    sec_per_year = 365.25 * 24.0 * 3600.0
    return float(max(1.0, min(1e6, sec_per_year / step_sec)))


def _sharpe(pnl: np.ndarray, steps_per_year: float) -> float:
    if pnl is None or len(pnl) < 2:
        return 0.0
    mu = float(np.mean(pnl))
    sd = float(np.std(pnl, ddof=1))
    if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 1e-12:
        return 0.0
    return float(mu / sd * np.sqrt(float(steps_per_year)))


def _max_drawdown(equity: np.ndarray) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    peak = float(equity[0])
    mdd = 0.0
    for x in equity:
        peak = max(peak, float(x))
        dd = (peak - float(x)) / peak if peak > 0 else 0.0
        mdd = max(mdd, dd)
    return float(mdd)


def _load_preds_by_symbol(preds_path: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for f in _collect_pred_files(preds_path):
        df = _ensure_timestamp_col(_read_parquet_or_csv(f))
        if "timestamp" in getattr(df.index, "names", []):
            df = df.reset_index(drop=True)
        sym = (
            str(df["symbol"].iloc[0])
            if "symbol" in df.columns
            else f.stem.replace("preds_", "")
        )
        df = df.copy()
        df["symbol"] = sym
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce"
        ).dt.tz_localize(None)
        df = (
            df.dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        out[sym] = df
    return out


def _load_logs_by_symbol(logs_path: Path) -> Dict[str, pd.DataFrame]:
    df = _read_parquet_or_csv(logs_path)
    if "symbol" not in df.columns or "timestamp" not in df.columns:
        raise ValueError("logs must contain columns: symbol,timestamp")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(
        None
    )
    df = df.dropna(subset=["timestamp"])
    out: Dict[str, pd.DataFrame] = {}
    for sym, g in df.groupby("symbol", sort=False):
        out[str(sym)] = g.sort_values("timestamp").reset_index(drop=True)
    return out


def _load_preds_in_log1p_from_model(model_path: Path) -> bool:
    payload = torch.load(str(model_path), map_location="cpu")
    meta = payload.get("meta") or {}
    ds_cfg = meta.get("dataset_cfg") or {}
    return bool(ds_cfg.get("log1p_targets", True))


def _slice_by_time(
    df: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    m = (ts >= start) & (ts <= end)
    return df.loc[m].reset_index(drop=True)


def _make_time_windows(
    logs_by_sym: Dict[str, pd.DataFrame],
    *,
    n_windows: int,
    min_days_per_window: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    # Use intersection across symbols to avoid one symbol dominating window boundaries.
    starts = []
    ends = []
    for _, df in logs_by_sym.items():
        ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
        if ts.empty:
            continue
        starts.append(ts.min())
        ends.append(ts.max())
    if not starts or not ends:
        return []
    global_start = max(starts)
    global_end = min(ends)
    if global_end <= global_start:
        return []
    # build equal-length windows
    total_days = int((global_end - global_start).total_seconds() // (24 * 3600))
    if total_days < max(1, min_days_per_window):
        return [(global_start, global_end)]
    n = max(1, int(n_windows))
    step_days = max(min_days_per_window, total_days // n)
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = global_start
    while cur < global_end and len(windows) < n:
        nxt = min(global_end, cur + pd.Timedelta(days=step_days))
        windows.append((cur, nxt))
        cur = nxt
    if windows and windows[-1][1] < global_end:
        windows[-1] = (windows[-1][0], global_end)
    return windows


def _eval_thresholds_on_logs(
    *,
    preds_by_sym: Dict[str, pd.DataFrame],
    logs_by_sym: Dict[str, pd.DataFrame],
    cfg: Rule3ActionConfig,
    preds_in_log1p: bool,
    sim_cfg: SimEnvConfig,
    trend_correct_horizon: int,
) -> Dict[str, float]:
    sharpe_by_sym: List[float] = []
    dd_by_sym: List[float] = []
    trade_rate_by_sym: List[float] = []
    trend_rate_by_sym: List[float] = []
    mean_rate_by_sym: List[float] = []
    no_trade_rate_by_sym: List[float] = []
    switch_rate_by_sym: List[float] = []
    entropy_by_sym: List[float] = []
    trend_correct_by_sym: List[float] = []

    for sym, logs in logs_by_sym.items():
        preds = preds_by_sym.get(sym)
        if preds is None or preds.empty or logs is None or logs.empty:
            continue

        mode_df = compute_mode_3action(preds, cfg=cfg, preds_in_log1p=preds_in_log1p)
        s = mode_df["mode_action"].astype(int)
        s.index = pd.to_datetime(preds["timestamp"], errors="coerce").dt.tz_localize(
            None
        )
        actions = (
            s.reindex(
                pd.to_datetime(logs["timestamp"], errors="coerce").dt.tz_localize(None)
            )
            .fillna(0)
            .astype(int)
            .to_numpy()
        )

        sim = simulate_3action_episode(logs, actions=actions.tolist(), cfg=sim_cfg)
        pnl = (
            sim["pnl"].to_numpy(dtype=float)
            if "pnl" in sim.columns
            else np.zeros(len(sim))
        )
        equity = (
            sim["equity"].to_numpy(dtype=float)
            if "equity" in sim.columns
            else np.ones(len(sim))
        )

        steps_per_year = _infer_steps_per_year(logs["timestamp"])
        sharpe_by_sym.append(_sharpe(pnl, steps_per_year))
        dd_by_sym.append(_max_drawdown(equity))
        trade_rate_by_sym.append(float(np.mean(actions != 0)))
        trend_rate_by_sym.append(float(np.mean(actions == 2)))
        mean_rate_by_sym.append(float(np.mean(actions == 1)))
        no_trade_rate_by_sym.append(float(np.mean(actions == 0)))
        if len(actions) > 1:
            switch_rate_by_sym.append(float(np.mean(actions[1:] != actions[:-1])))
        else:
            switch_rate_by_sym.append(0.0)
        # entropy over {NO_TRADE, MEAN, TREND}
        counts = np.bincount(actions, minlength=3).astype(float)
        probs = counts / max(1.0, counts.sum())
        entropy = float(-np.sum([p * np.log(p) for p in probs if p > 0.0]))
        entropy_by_sym.append(entropy)
        # conditional correctness: realized MFE over next horizon (if OHLC available)
        if {"high", "low", "close"}.issubset(set(logs.columns)):
            high = pd.to_numeric(logs["high"], errors="coerce").astype(float)
            low = pd.to_numeric(logs["low"], errors="coerce").astype(float)
            close = pd.to_numeric(logs["close"], errors="coerce").astype(float)
            # ATR (use provided atr if available, else TR rolling mean)
            if "atr" in logs.columns:
                atr = pd.to_numeric(logs["atr"], errors="coerce").astype(float)
            else:
                tr = pd.concat(
                    [
                        high - low,
                        (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs(),
                    ],
                    axis=1,
                ).max(axis=1)
                atr = tr.rolling(window=14, min_periods=1).mean().astype(float)
            horizon = max(1, int(trend_correct_horizon))
            # forward window max/min (exclude current bar)
            f_high = (
                high[::-1].rolling(window=horizon, min_periods=1).max()[::-1].shift(-1)
            )
            f_low = (
                low[::-1].rolling(window=horizon, min_periods=1).min()[::-1].shift(-1)
            )
            dir_score = pd.to_numeric(
                logs.get("head_dir_score", 0.0), errors="coerce"
            ).fillna(0.0)
            dir_sign = np.sign(dir_score.to_numpy(dtype=float))
            mfe_atr = np.where(
                dir_sign >= 0,
                (f_high.to_numpy(dtype=float) - close.to_numpy(dtype=float))
                / np.maximum(1e-9, atr.to_numpy(dtype=float)),
                (close.to_numpy(dtype=float) - f_low.to_numpy(dtype=float))
                / np.maximum(1e-9, atr.to_numpy(dtype=float)),
            )
            mfe_thr = float(cfg.mfe_trend_min)
            mask = (actions == 2) & np.isfinite(mfe_atr)
            if mask.any():
                trend_correct_by_sym.append(float(np.mean(mfe_atr[mask] >= mfe_thr)))
        else:
            raise ValueError(
                "conditional_correctness requires logs with high/low/close (and atr or inferable ATR). "
                "Regenerate logs with returns_source=rr_execution or vectorbt_execution."
            )

    if not sharpe_by_sym:
        return {
            "rule_sharpe_mean": 0.0,
            "rule_sharpe_std": 0.0,
            "rule_dd_mean": 0.0,
            "trade_rate_mean": 0.0,
            "n_symbols": 0.0,
            "trend_rate_mean": 0.0,
            "mean_rate_mean": 0.0,
            "no_trade_rate_mean": 0.0,
            "switch_rate_mean": 0.0,
            "entropy_mean": 0.0,
            "trend_correctness_mean": 0.0,
        }

    return {
        "rule_sharpe_mean": float(np.mean(sharpe_by_sym)),
        "rule_sharpe_std": (
            float(np.std(sharpe_by_sym, ddof=1)) if len(sharpe_by_sym) > 1 else 0.0
        ),
        "rule_dd_mean": float(np.mean(dd_by_sym)),
        "trade_rate_mean": float(np.mean(trade_rate_by_sym)),
        "n_symbols": float(len(sharpe_by_sym)),
        "trend_rate_mean": (
            float(np.mean(trend_rate_by_sym)) if trend_rate_by_sym else 0.0
        ),
        "mean_rate_mean": float(np.mean(mean_rate_by_sym)) if mean_rate_by_sym else 0.0,
        "no_trade_rate_mean": (
            float(np.mean(no_trade_rate_by_sym)) if no_trade_rate_by_sym else 0.0
        ),
        "switch_rate_mean": (
            float(np.mean(switch_rate_by_sym)) if switch_rate_by_sym else 0.0
        ),
        "entropy_mean": float(np.mean(entropy_by_sym)) if entropy_by_sym else 0.0,
        "trend_correctness_mean": (
            float(np.mean(trend_correct_by_sym)) if trend_correct_by_sym else 0.0
        ),
    }


def _robust_score(m: Dict[str, float], *, lam: float, mu: float) -> float:
    return (
        float(m.get("rule_sharpe_mean", 0.0))
        - lam * float(m.get("rule_sharpe_std", 0.0))
        - mu * float(m.get("rule_dd_mean", 0.0))
    )


def _trade_rate_penalty(
    trade_rate: float,
    *,
    target: Optional[float],
    tol: float,
    w: float,
    min_v: Optional[float],
    max_v: Optional[float],
) -> float:
    """
    Penalize candidates whose trade_rate is outside desired operating band.

    Two modes:
    - min/max band: hinge penalty on violations
    - target+tol: quadratic penalty when |trade_rate-target| exceeds tol
    """
    tr = float(trade_rate)
    if not np.isfinite(tr):
        return 0.0
    if (min_v is not None) or (max_v is not None):
        lo = float(min_v) if min_v is not None else -1e9
        hi = float(max_v) if max_v is not None else 1e9
        viol = max(0.0, lo - tr) + max(0.0, tr - hi)
        return float(w * viol)
    if target is None:
        return 0.0
    t = float(target)
    if not np.isfinite(t):
        return 0.0
    tol = float(max(1e-6, abs(tol)))
    d = abs(tr - t) - tol
    if d <= 0:
        return 0.0
    # normalized squared deviation beyond tolerance band
    return float(w * (d / tol) ** 2)


def _range_penalty(value: float, *, lo: float, hi: float, weight: float = 1.0) -> float:
    if not np.isfinite(value):
        return 0.0
    viol = max(0.0, lo - value) + max(0.0, value - hi)
    return float(weight * viol)


def _bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n, endpoint=False)


def _candidate_from_baseline(
    base: Dict[str, float],
    *,
    rng: np.random.Generator,
    rel_sigma: float,
    abs_sigma: float,
) -> Dict[str, float]:
    out = dict(base)
    for k in ROUTER_KEYS:
        v = float(base[k])
        noise = rng.normal(0.0, 1.0)
        dv = abs_sigma + rel_sigma * abs(v)
        out[k] = float(v + noise * dv)
    # keep some obvious constraints
    out["mfe_min"] = max(0.0, out["mfe_min"])
    out["eff_min"] = max(0.0, out["eff_min"])
    out["eff_mean_min"] = max(0.0, out["eff_mean_min"])
    out["mfe_trend_min"] = max(0.0, out["mfe_trend_min"])
    out["ttm_trend_min"] = max(0.0, out["ttm_trend_min"])
    out["ttm_mean_max"] = max(0.0, out["ttm_mean_max"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="preds file/dir (preds_*.parquet)")
    ap.add_argument(
        "--logs",
        required=True,
        help="logs_3action.parquet (contains ret_mean/ret_trend)",
    )
    ap.add_argument("--model", required=True, help="model.pt (infer preds_in_log1p)")
    ap.add_argument("--out", required=True, help="output directory")

    ap.add_argument(
        "--baseline-json", required=True, help="baseline thresholds JSON (7 keys)"
    )
    ap.add_argument("--n-candidates", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--heuristic-bounds",
        action="store_true",
        help="Clamp candidate thresholds to empirical quantile bounds.",
    )
    ap.add_argument("--heuristic-qmin", type=float, default=0.05)
    ap.add_argument("--heuristic-qmax", type=float, default=0.95)

    ap.add_argument("--n-windows", type=int, default=6)
    ap.add_argument("--min-days-per-window", type=int, default=25)
    ap.add_argument("--n-bootstrap", type=int, default=30)

    ap.add_argument(
        "--rel-sigma",
        type=float,
        default=0.05,
        help="relative perturbation scale per param",
    )
    ap.add_argument(
        "--abs-sigma",
        type=float,
        default=0.01,
        help="absolute perturbation floor per param",
    )

    ap.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=1.0,
        help="penalty on sharpe std across symbols",
    )
    ap.add_argument(
        "--mu", dest="mu", type=float, default=0.5, help="penalty on max drawdown mean"
    )

    ap.add_argument("--entry-delay", type=int, default=0)
    ap.add_argument("--cost-per-turnover", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument(
        "--trade-rate-target",
        type=float,
        default=None,
        help="Optional target trade rate (fraction of non-zero actions). If set, applies a penalty when outside tolerance.",
    )
    ap.add_argument(
        "--trade-rate-tol",
        type=float,
        default=0.06,
        help="Tolerance band around trade-rate-target before penalty applies.",
    )
    ap.add_argument(
        "--trade-rate-min",
        type=float,
        default=None,
        help="Optional minimum trade rate (hinge penalty if below). Overrides target-mode when set.",
    )
    ap.add_argument(
        "--trade-rate-max",
        type=float,
        default=None,
        help="Optional maximum trade rate (hinge penalty if above). Overrides target-mode when set.",
    )
    ap.add_argument(
        "--trade-rate-penalty",
        type=float,
        default=1.5,
        help="Penalty weight for trade-rate deviation (higher => force stable operating density).",
    )
    ap.add_argument(
        "--trend-rate-target",
        type=float,
        default=None,
        help="Optional target trend rate (fraction of TREND actions).",
    )
    ap.add_argument(
        "--trend-rate-tol",
        type=float,
        default=0.04,
        help="Tolerance band around trend-rate-target before penalty applies.",
    )
    ap.add_argument(
        "--trend-rate-min",
        type=float,
        default=0.10,
        help="Optional minimum trend rate (hinge penalty if below). Overrides target-mode when set.",
    )
    ap.add_argument(
        "--trend-rate-max",
        type=float,
        default=0.60,
        help="Optional maximum trend rate (hinge penalty if above). Overrides target-mode when set.",
    )
    ap.add_argument(
        "--trend-rate-penalty",
        type=float,
        default=1.0,
        help="Penalty weight for trend-rate deviation (higher => force non-zero TREND activation).",
    )
    ap.add_argument("--mean-rate-min", type=float, default=0.05)
    ap.add_argument("--mean-rate-max", type=float, default=0.40)
    ap.add_argument("--no-trade-rate-min", type=float, default=0.10)
    ap.add_argument("--no-trade-rate-max", type=float, default=0.70)
    ap.add_argument(
        "--disable-dist-rate-constraints",
        action="store_true",
        help="Disable mean/no_trade distribution range constraints.",
    )
    ap.add_argument("--trend-correct-horizon", type=int, default=24)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_obj = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
    base: Dict[str, float] = {k: float(base_obj[k]) for k in ROUTER_KEYS}

    preds_in_log1p = _load_preds_in_log1p_from_model(Path(args.model))
    preds_by_sym = _load_preds_by_symbol(Path(args.preds))
    logs_by_sym = _load_logs_by_symbol(Path(args.logs))
    keep = sorted(set(preds_by_sym.keys()) & set(logs_by_sym.keys()))
    preds_by_sym = {k: preds_by_sym[k] for k in keep}
    logs_by_sym = {k: logs_by_sym[k] for k in keep}

    bounds = {}
    if args.heuristic_bounds:
        bounds = _compute_router_bounds(
            preds_by_sym=preds_by_sym,
            preds_in_log1p=preds_in_log1p,
            qmin=float(args.heuristic_qmin),
            qmax=float(args.heuristic_qmax),
        )

    windows = _make_time_windows(
        logs_by_sym,
        n_windows=int(args.n_windows),
        min_days_per_window=int(args.min_days_per_window),
    )
    if not windows:
        raise RuntimeError("No valid windows (check logs timestamps / symbol overlap).")

    sim_cfg = SimEnvConfig(
        entry_delay=int(args.entry_delay),
        cost_per_turnover=float(args.cost_per_turnover),
        slippage_bps=float(args.slippage_bps),
    )

    rng = np.random.default_rng(int(args.seed))

    # baseline ranges for distribution-style KPIs
    dist_keys = [
        "trade_rate_mean",
        "trend_rate_mean",
        "mean_rate_mean",
        "no_trade_rate_mean",
        "switch_rate_mean",
        "entropy_mean",
        "trend_correctness_mean",
    ]
    baseline_window_metrics: List[Dict[str, float]] = []
    base_cfg = Rule3ActionConfig(**{k: float(base[k]) for k in ROUTER_KEYS})
    for ws, we in windows:
        preds_w = {
            s: _slice_by_time(df, start=ws, end=we) for s, df in preds_by_sym.items()
        }
        logs_w = {
            s: _slice_by_time(df, start=ws, end=we) for s, df in logs_by_sym.items()
        }
        m = _eval_thresholds_on_logs(
            preds_by_sym=preds_w,
            logs_by_sym=logs_w,
            cfg=base_cfg,
            preds_in_log1p=preds_in_log1p,
            sim_cfg=sim_cfg,
            trend_correct_horizon=int(args.trend_correct_horizon),
        )
        baseline_window_metrics.append(m)

    baseline_ranges: Dict[str, Tuple[float, float]] = {}
    for k in dist_keys:
        vals = [float(x.get(k, 0.0)) for x in baseline_window_metrics]
        vmean = float(np.mean(vals)) if vals else 0.0
        vstd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        lo = vmean - vstd
        hi = vmean + vstd
        if k in {
            "trade_rate_mean",
            "trend_rate_mean",
            "mean_rate_mean",
            "no_trade_rate_mean",
            "switch_rate_mean",
            "trend_correctness_mean",
        }:
            lo = max(0.0, lo)
            hi = min(1.0, hi)
        if k == "entropy_mean":
            lo = max(0.0, lo)
            hi = max(lo, hi)
        baseline_ranges[k] = (lo, hi)

    # candidates include baseline
    cands: List[Dict[str, float]] = [_apply_bounds(dict(base), bounds)]
    for _ in range(int(args.n_candidates)):
        cand = _candidate_from_baseline(
            base,
            rng=rng,
            rel_sigma=float(args.rel_sigma),
            abs_sigma=float(args.abs_sigma),
        )
        cands.append(_apply_bounds(cand, bounds))

    rows = []
    for i, cand in enumerate(cands):
        cfg = Rule3ActionConfig(**{k: float(cand[k]) for k in ROUTER_KEYS})

        # window metrics
        win_scores: List[float] = []
        win_sharpes: List[float] = []
        win_dd: List[float] = []
        win_trade_rate: List[float] = []
        win_trend_rate: List[float] = []
        win_mean_rate: List[float] = []
        win_no_trade_rate: List[float] = []
        win_switch_rate: List[float] = []
        win_entropy: List[float] = []
        win_trend_correct: List[float] = []
        win_trade_pen: List[float] = []
        win_trend_pen: List[float] = []
        win_dist_pen: List[float] = []
        for ws, we in windows:
            preds_w = {
                s: _slice_by_time(df, start=ws, end=we)
                for s, df in preds_by_sym.items()
            }
            logs_w = {
                s: _slice_by_time(df, start=ws, end=we) for s, df in logs_by_sym.items()
            }
            m = _eval_thresholds_on_logs(
                preds_by_sym=preds_w,
                logs_by_sym=logs_w,
                cfg=cfg,
                preds_in_log1p=preds_in_log1p,
                sim_cfg=sim_cfg,
                trend_correct_horizon=int(args.trend_correct_horizon),
            )
            tr = float(m.get("trade_rate_mean", 0.0))
            tr_trend = float(m.get("trend_rate_mean", 0.0))
            tr_mean = float(m.get("mean_rate_mean", 0.0))
            tr_no = float(m.get("no_trade_rate_mean", 0.0))
            tr_switch = float(m.get("switch_rate_mean", 0.0))
            tr_entropy = float(m.get("entropy_mean", 0.0))
            tr_correct = float(m.get("trend_correctness_mean", 0.0))

            # explicit penalties (optional)
            pen = _trade_rate_penalty(
                tr,
                target=args.trade_rate_target,
                tol=float(args.trade_rate_tol),
                w=float(args.trade_rate_penalty),
                min_v=args.trade_rate_min,
                max_v=args.trade_rate_max,
            )
            pen_trend = _trade_rate_penalty(
                tr_trend,
                target=args.trend_rate_target,
                tol=float(args.trend_rate_tol),
                w=float(args.trend_rate_penalty),
                min_v=args.trend_rate_min,
                max_v=args.trend_rate_max,
            )

            # baseline-driven distribution penalties
            dist_pen = 0.0
            # baseline-driven penalties for non-explicit keys
            explicit_keys = {
                "trade_rate_mean",
                "trend_rate_mean",
                "mean_rate_mean",
                "no_trade_rate_mean",
            }
            for key, val in [
                ("trade_rate_mean", tr),
                ("trend_rate_mean", tr_trend),
                ("mean_rate_mean", tr_mean),
                ("no_trade_rate_mean", tr_no),
                ("switch_rate_mean", tr_switch),
                ("entropy_mean", tr_entropy),
                ("trend_correctness_mean", tr_correct),
            ]:
                if key in explicit_keys:
                    continue
                lo, hi = baseline_ranges.get(key, (0.0, 1.0))
                dist_pen += _range_penalty(val, lo=lo, hi=hi, weight=1.0)

            if not args.disable_dist_rate_constraints:
                dist_pen += _range_penalty(
                    tr_mean, lo=float(args.mean_rate_min), hi=float(args.mean_rate_max)
                )
                dist_pen += _range_penalty(
                    tr_no,
                    lo=float(args.no_trade_rate_min),
                    hi=float(args.no_trade_rate_max),
                )

            win_sharpes.append(float(m["rule_sharpe_mean"]))
            win_dd.append(float(m["rule_dd_mean"]))
            win_trade_rate.append(tr)
            win_trend_rate.append(tr_trend)
            win_mean_rate.append(tr_mean)
            win_no_trade_rate.append(tr_no)
            win_switch_rate.append(tr_switch)
            win_entropy.append(tr_entropy)
            win_trend_correct.append(tr_correct)
            win_trade_pen.append(float(pen))
            win_trend_pen.append(float(pen_trend))
            win_dist_pen.append(float(dist_pen))
            win_scores.append(-float(pen) - float(pen_trend) - float(dist_pen))

        # bootstrap metrics (resample windows)
        bs_scores: List[float] = []
        if int(args.n_bootstrap) > 0:
            for _ in range(int(args.n_bootstrap)):
                idx = _bootstrap_indices(len(windows), rng)
                bs_scores.append(float(np.mean([win_scores[j] for j in idx])))

        row = {
            "cand_id": int(i),
            **{k: float(cand[k]) for k in ROUTER_KEYS},
            "win_score_mean": float(np.mean(win_scores)),
            "win_score_p25": float(np.quantile(win_scores, 0.25)),
            "win_sharpe_mean": float(np.mean(win_sharpes)),
            "win_sharpe_p25": float(np.quantile(win_sharpes, 0.25)),
            "win_dd_mean": float(np.mean(win_dd)),
            "trade_rate_mean": (
                float(np.mean(win_trade_rate)) if win_trade_rate else 0.0
            ),
            "trade_rate_p25": (
                float(np.quantile(win_trade_rate, 0.25)) if win_trade_rate else 0.0
            ),
            "trend_rate_mean": (
                float(np.mean(win_trend_rate)) if win_trend_rate else 0.0
            ),
            "trend_rate_p25": (
                float(np.quantile(win_trend_rate, 0.25)) if win_trend_rate else 0.0
            ),
            "mean_rate_mean": float(np.mean(win_mean_rate)) if win_mean_rate else 0.0,
            "no_trade_rate_mean": (
                float(np.mean(win_no_trade_rate)) if win_no_trade_rate else 0.0
            ),
            "switch_rate_mean": (
                float(np.mean(win_switch_rate)) if win_switch_rate else 0.0
            ),
            "entropy_mean": float(np.mean(win_entropy)) if win_entropy else 0.0,
            "trend_correctness_mean": (
                float(np.mean(win_trend_correct)) if win_trend_correct else 0.0
            ),
            "trade_rate_pen_mean": (
                float(np.mean(win_trade_pen)) if win_trade_pen else 0.0
            ),
            "trend_rate_pen_mean": (
                float(np.mean(win_trend_pen)) if win_trend_pen else 0.0
            ),
            "dist_pen_mean": float(np.mean(win_dist_pen)) if win_dist_pen else 0.0,
            "bs_score_mean": float(np.mean(bs_scores)) if bs_scores else float("nan"),
            "bs_score_p10": (
                float(np.quantile(bs_scores, 0.10)) if bs_scores else float("nan")
            ),
            "robust_score": (
                float(np.mean(win_scores)) - float(np.std(win_scores, ddof=1))
                if len(win_scores) > 1
                else float(np.mean(win_scores))
            ),
        }
        rows.append(row)

    df = (
        pd.DataFrame(rows)
        .sort_values(["robust_score", "win_score_p25"], ascending=False)
        .reset_index(drop=True)
    )
    df.to_csv(out_dir / "candidates.csv", index=False)

    best = df.iloc[0].to_dict()
    # plateau width: fraction of candidates within (best - 5%*|best|).
    # This definition works for both positive and negative best_score.
    best_score = float(best["robust_score"])
    thr = (
        best_score - 0.05 * float(abs(best_score))
        if np.isfinite(best_score)
        else float("nan")
    )
    plateau_frac = (
        float(np.mean(df["robust_score"].to_numpy(dtype=float) >= float(thr)))
        if np.isfinite(best_score) and np.isfinite(thr)
        else 0.0
    )

    summary = {
        "best": {
            k: best[k]
            for k in [
                "cand_id",
                "robust_score",
                "win_score_mean",
                "win_score_p25",
                "trade_rate_mean",
                "trade_rate_p25",
                "trade_rate_pen_mean",
                "trend_rate_mean",
                "trend_rate_p25",
                "trend_rate_pen_mean",
                "mean_rate_mean",
                "no_trade_rate_mean",
                "switch_rate_mean",
                "entropy_mean",
                "trend_correctness_mean",
                "dist_pen_mean",
            ]
            + ROUTER_KEYS
        },
        "plateau_frac_ge_95pct": plateau_frac,
        "n_candidates": int(len(df)),
        "n_windows": int(len(windows)),
        "windows": [{"start": str(a), "end": str(b)} for a, b in windows],
        "sim_cfg": asdict(sim_cfg),
        "preds_in_log1p": bool(preds_in_log1p),
        "score_formula": "window_score = - (trade_rate_penalty + trend_rate_penalty + dist_penalty) ; robust_score = mean(window_score) - std(window_score)",
        "lambda": float(args.lam),
        "mu": float(args.mu),
        "trade_rate_target": args.trade_rate_target,
        "trade_rate_tol": float(args.trade_rate_tol),
        "trade_rate_min": args.trade_rate_min,
        "trade_rate_max": args.trade_rate_max,
        "trade_rate_penalty": float(args.trade_rate_penalty),
        "trend_rate_target": args.trend_rate_target,
        "trend_rate_tol": float(args.trend_rate_tol),
        "trend_rate_min": args.trend_rate_min,
        "trend_rate_max": args.trend_rate_max,
        "trend_rate_penalty": float(args.trend_rate_penalty),
        "mean_rate_min": float(args.mean_rate_min),
        "mean_rate_max": float(args.mean_rate_max),
        "no_trade_rate_min": float(args.no_trade_rate_min),
        "no_trade_rate_max": float(args.no_trade_rate_max),
        "disable_dist_rate_constraints": bool(args.disable_dist_rate_constraints),
        "trend_correct_horizon": int(args.trend_correct_horizon),
        "heuristic_bounds": {
            "enabled": bool(args.heuristic_bounds),
            "qmin": float(args.heuristic_qmin),
            "qmax": float(args.heuristic_qmax),
            "bounds": {k: [float(v[0]), float(v[1])] for k, v in bounds.items()},
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # human-friendly report
    md = []
    md.append("## Threshold Plateau Tuning Report (Rule Router 3-action)\n")
    md.append(f"- preds_in_log1p: **{preds_in_log1p}**\n")
    md.append(f"- n_candidates: **{len(df)}** (includes baseline)\n")
    md.append(
        f"- n_windows: **{len(windows)}**, n_bootstrap: **{int(args.n_bootstrap)}**\n"
    )
    md.append(f"- plateau_frac (>= best - 5%*|best|): **{plateau_frac:.3f}**\n")
    md.append("\n### Best thresholds (robust)\n")
    md.append("```json\n")
    md.append(json.dumps({k: float(best[k]) for k in ROUTER_KEYS}, indent=2))
    md.append("\n```\n")
    # Machine-friendly export for one-click reuse
    (out_dir / "router_thresholds_best.json").write_text(
        json.dumps(
            {k: float(best[k]) for k in ROUTER_KEYS}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    md.append("\n### Best scores\n")
    md.append(
        f"- robust_score: **{float(best['robust_score']):.4f}**\n"
        f"- win_score_mean: {float(best['win_score_mean']):.4f}\n"
        f"- win_score_p25: {float(best['win_score_p25']):.4f}\n"
        f"- win_sharpe_mean: {float(best['win_sharpe_mean']):.4f}\n"
        f"- win_sharpe_p25: {float(best['win_sharpe_p25']):.4f}\n"
        f"- win_dd_mean: {float(best['win_dd_mean']):.4f}\n"
        f"- trade_rate_mean: {float(best.get('trade_rate_mean', 0.0)):.4f}\n"
        f"- trade_rate_p25: {float(best.get('trade_rate_p25', 0.0)):.4f}\n"
        f"- trade_rate_pen_mean: {float(best.get('trade_rate_pen_mean', 0.0)):.4f}\n"
        f"- trend_rate_mean: {float(best.get('trend_rate_mean', 0.0)):.4f}\n"
        f"- trend_rate_p25: {float(best.get('trend_rate_p25', 0.0)):.4f}\n"
        f"- trend_rate_pen_mean: {float(best.get('trend_rate_pen_mean', 0.0)):.4f}\n"
        f"- mean_rate_mean: {float(best.get('mean_rate_mean', 0.0)):.4f}\n"
        f"- no_trade_rate_mean: {float(best.get('no_trade_rate_mean', 0.0)):.4f}\n"
        f"- switch_rate_mean: {float(best.get('switch_rate_mean', 0.0)):.4f}\n"
        f"- entropy_mean: {float(best.get('entropy_mean', 0.0)):.4f}\n"
        f"- trend_correctness_mean: {float(best.get('trend_correctness_mean', 0.0)):.4f}\n"
        f"- dist_pen_mean: {float(best.get('dist_pen_mean', 0.0)):.4f}\n"
    )
    md.append("\n### How to interpret\n")
    md.append(
        "- Prefer **high win_score_p25** over high win_score_mean: it means fewer bad sub-windows.\n"
    )
    md.append(
        "- Prefer **higher plateau_frac**: means less sensitivity / more controllability.\n"
    )
    md.append(
        "- Keep an eye on trade_rate_mean in candidates.csv to avoid unrealistic overtrading.\n"
    )
    md.append(
        "- Note: win_sharpe_* are diagnostics only; Router KPI does not optimize Sharpe.\n"
    )
    if args.heuristic_bounds and bounds:
        md.append(
            f"- heuristic_bounds: qmin={float(args.heuristic_qmin):.2f}, qmax={float(args.heuristic_qmax):.2f}\n"
        )
    (out_dir / "report.md").write_text("".join(md), encoding="utf-8")

    # Optional: HTML report with plots (helps visually confirm plateau vs sharp peak)
    try:
        _write_html_report(
            out_dir=out_dir,
            df=df,
            best_thresholds={k: float(best[k]) for k in ROUTER_KEYS},
            best_row=best,
            plateau_frac=float(plateau_frac),
            trade_rate_target=args.trade_rate_target,
            trade_rate_tol=float(args.trade_rate_tol),
            trade_rate_min=args.trade_rate_min,
            trade_rate_max=args.trade_rate_max,
            trend_rate_target=args.trend_rate_target,
            trend_rate_tol=float(args.trend_rate_tol),
            trend_rate_min=args.trend_rate_min,
            trend_rate_max=args.trend_rate_max,
        )
    except Exception:
        pass

    print("✅ Wrote:", (out_dir / "candidates.csv").as_posix())
    print("✅ Wrote:", (out_dir / "summary.json").as_posix())
    print("✅ Wrote:", (out_dir / "router_thresholds_best.json").as_posix())
    print("✅ Wrote:", (out_dir / "report.md").as_posix())
    if (out_dir / "report.html").exists():
        print("✅ Wrote:", (out_dir / "report.html").as_posix())

    # KPI journal (append-only): if this out_dir is under a run root, append plateau KPI status there.
    try:
        from src.time_series_model.diagnostics.kpi_journal import (
            find_run_root,
            write_kpi_journal,
        )

        rr = find_run_root(out_dir)
        if rr is not None:
            write_kpi_journal(
                run_dir=str(rr),
                stage="threshold_plateau",
                extra={"plateau_out_dir": str(out_dir)},
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
