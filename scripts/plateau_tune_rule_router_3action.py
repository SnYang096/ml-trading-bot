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
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

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
) -> Dict[str, float]:
    sharpe_by_sym: List[float] = []
    dd_by_sym: List[float] = []
    trade_rate_by_sym: List[float] = []

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

    if not sharpe_by_sym:
        return {
            "rule_sharpe_mean": 0.0,
            "rule_sharpe_std": 0.0,
            "rule_dd_mean": 0.0,
            "trade_rate_mean": 0.0,
            "n_symbols": 0.0,
        }

    return {
        "rule_sharpe_mean": float(np.mean(sharpe_by_sym)),
        "rule_sharpe_std": (
            float(np.std(sharpe_by_sym, ddof=1)) if len(sharpe_by_sym) > 1 else 0.0
        ),
        "rule_dd_mean": float(np.mean(dd_by_sym)),
        "trade_rate_mean": float(np.mean(trade_rate_by_sym)),
        "n_symbols": float(len(sharpe_by_sym)),
    }


def _robust_score(m: Dict[str, float], *, lam: float, mu: float) -> float:
    return (
        float(m.get("rule_sharpe_mean", 0.0))
        - lam * float(m.get("rule_sharpe_std", 0.0))
        - mu * float(m.get("rule_dd_mean", 0.0))
    )


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

    # candidates include baseline
    cands: List[Dict[str, float]] = [dict(base)]
    for _ in range(int(args.n_candidates)):
        cands.append(
            _candidate_from_baseline(
                base,
                rng=rng,
                rel_sigma=float(args.rel_sigma),
                abs_sigma=float(args.abs_sigma),
            )
        )

    rows = []
    for i, cand in enumerate(cands):
        cfg = Rule3ActionConfig(**{k: float(cand[k]) for k in ROUTER_KEYS})

        # window metrics
        win_scores: List[float] = []
        win_sharpes: List[float] = []
        win_dd: List[float] = []
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
            )
            win_sharpes.append(float(m["rule_sharpe_mean"]))
            win_dd.append(float(m["rule_dd_mean"]))
            win_scores.append(_robust_score(m, lam=float(args.lam), mu=float(args.mu)))

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
            "bs_score_mean": float(np.mean(bs_scores)) if bs_scores else float("nan"),
            "bs_score_p10": (
                float(np.quantile(bs_scores, 0.10)) if bs_scores else float("nan")
            ),
            "robust_score": float(np.mean(win_scores))
            + (float(np.mean(bs_scores)) if bs_scores else 0.0),
        }
        rows.append(row)

    df = (
        pd.DataFrame(rows)
        .sort_values(["robust_score", "win_score_p25"], ascending=False)
        .reset_index(drop=True)
    )
    df.to_csv(out_dir / "candidates.csv", index=False)

    best = df.iloc[0].to_dict()
    # plateau width: fraction of candidates within 95% of best robust_score
    best_score = float(best["robust_score"])
    thr = 0.95 * best_score
    plateau_frac = (
        float(np.mean(df["robust_score"].to_numpy(dtype=float) >= thr))
        if np.isfinite(best_score)
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
                "win_sharpe_mean",
                "win_sharpe_p25",
                "win_dd_mean",
            ]
            + ROUTER_KEYS
        },
        "plateau_frac_ge_95pct": plateau_frac,
        "n_candidates": int(len(df)),
        "n_windows": int(len(windows)),
        "windows": [{"start": str(a), "end": str(b)} for a, b in windows],
        "sim_cfg": asdict(sim_cfg),
        "preds_in_log1p": bool(preds_in_log1p),
        "score_formula": "robust_score = mean(window_score) + mean(bootstrap(window_score)) ; window_score = sharpe_mean - lambda*sharpe_std - mu*dd_mean",
        "lambda": float(args.lam),
        "mu": float(args.mu),
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
    md.append(f"- plateau_frac (>=95% best robust_score): **{plateau_frac:.3f}**\n")
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
    (out_dir / "report.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
