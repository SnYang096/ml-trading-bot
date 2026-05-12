"""
Offline re-simulator: replay baseline trades under a new execution.yaml.

用途：
  用已有 rolling_sim 的 baseline trades（entry_time / side / entry_price / ATR 等），
  在 2H bar 上重放 position_logic 的 exit 过程，使用不同 execution.yaml
  （trail / breakeven / time_stop / initial_r），得到 alt_pnl_r。

比如：
  baseline (target_r=2.0 封顶) vs trail (take_profit=false, trail@1.5R)

精度限制：用 2H bar 粒度（event_backtest 原始是 1min），方向性结论可信，
绝对值会有偏差（同 bar 内的 SL/TP 顺序取决于 open→high→low→close 的假设）。

用法：
  python scripts/simulate_exec_trail.py \
    --trades results/fbf/research_roll.features_on-exp-fatter-tp/_rolling_sim/20260416_153251/fast_month_*/fbf/event_trades_fbf.csv \
    --feature-store feature_store/features_fbf_120T_06702ab6f8 \
    --exec-config config/strategies/fbf_exp_trail/archetypes/execution.yaml \
    --out reports/fbf_trail_resim.csv

  # 结构减仓实验：在对面 OLS 轨（与 fer_ols_pos 同源）先平一半，剩余仍走 trail / SL
  python scripts/simulate_exec_trail.py ... \
    --scale-out-target opposite_ols --scale-out-fraction 0.5

  或在 execution.yaml 增加:

  scale_out:
    enabled: true
    fraction: 0.5
    target: opposite_ols   # 或 opposite_range（同 fer_range_pos_20 包络）
    ols_window: 96
    range_window: 20
    move_sl_to_be_after_scale: true
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.features.time_series.fer_features import _rolling_ols_channel  # noqa: E402


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]


def load_bars(store: str, symbol: str, tf: str = "120T") -> pd.DataFrame:
    files = sorted(glob.glob(f"{store}/{symbol}/{tf}/*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    keep = [c for c in ["open", "high", "low", "close", "atr"] if c in df.columns]
    return df[keep]


def attach_opposite_rails(
    bars: pd.DataFrame,
    *,
    ols_window: int = 96,
    range_window: int = 20,
) -> pd.DataFrame:
    """
    Add structural rails aligned with FER semantics:
    - OLS: same _rolling_ols_channel as fer_features (fer_ols_pos rails).
    - Range: rolling high/low over N bars (fer_range_pos_20 envelope).
    """
    out = bars.copy()
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    mid, width = _rolling_ols_channel(close, int(ols_window))
    half = (width / 2.0).replace(0.0, np.nan)
    out["ols_mid"] = mid
    out["ols_upper"] = (mid + half).astype(float)
    out["ols_lower"] = (mid - half).astype(float)

    high = pd.to_numeric(out["high"], errors="coerce").astype(float)
    low = pd.to_numeric(out["low"], errors="coerce").astype(float)
    rw = max(2, int(range_window))
    out["range_upper"] = high.rolling(rw, min_periods=1).max()
    out["range_lower"] = low.rolling(rw, min_periods=1).min()
    return out


def load_exec(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _leg_pnl_r(
    entry_price: float,
    exit_price: float,
    *,
    is_long: bool,
    atr_at_entry: float,
    initial_r: float,
) -> float:
    if is_long:
        return (exit_price - entry_price) / atr_at_entry / initial_r
    return (entry_price - exit_price) / atr_at_entry / initial_r


def simulate_trade(
    bars: pd.DataFrame,
    *,
    entry_time: pd.Timestamp,
    side: str,
    entry_price: float,
    atr_at_entry: float,
    exec_cfg: dict,
    scale_out: Optional[dict] = None,
) -> dict:
    """
    Replay a single trade from entry_time + 1 bar onwards until close.

    Optional scale_out (execution.yaml key ``scale_out`` or CLI):
      - enabled, fraction (default 0.5), target: opposite_ols | opposite_range
      - ols_window (default 96), range_window (default 20)
      - move_sl_to_be_after_scale (default true)

    PnL is reported in **full-position R units** (same as baseline): a 50% exit
    contributes half of that leg's move in R, plus the runner's PnL on the
    remaining fraction.

    Returns dict with: exit_time, exit_price, pnl_r, exit_reason, bars_held,
    mfe_r, mae_r, scale_out_done (bool), scale_out_fill (float|None)
    """
    sl_cfg = exec_cfg.get("stop_loss", {}) or {}
    tp_cfg = exec_cfg.get("take_profit", {}) or {}
    be_cfg = exec_cfg.get("breakeven", {}) or {}
    trail_cfg = sl_cfg.get("trailing", {}) or {}
    holding = exec_cfg.get("holding", {}) or {}
    so = scale_out or {}

    initial_r = float(sl_cfg.get("initial_r", 1.0))
    tp_enabled = bool(tp_cfg.get("enabled", False))
    target_r = float(tp_cfg.get("target_r", 0.0))
    be_enabled = bool(be_cfg.get("enabled", False))
    be_trigger_r = float(be_cfg.get("trigger_r", 1.0))
    trail_enabled = bool(trail_cfg.get("enabled", False))
    trail_act_r = float(trail_cfg.get("activation_r", 1.5))
    trail_r = float(trail_cfg.get("trail_r", 0.8))
    time_stop = int(
        holding.get("time_stop_bars") or holding.get("max_holding_bars") or 36
    )

    so_enabled = bool(so.get("enabled"))
    so_frac = float(so.get("fraction", 0.5))
    so_frac = min(max(so_frac, 1e-6), 1.0 - 1e-6)
    so_target = str(so.get("target", "opposite_ols")).lower().strip()
    so_move_be = bool(so.get("move_sl_to_be_after_scale", True))

    is_long = side.upper() in ("LONG", "BUY")
    R = atr_at_entry * initial_r  # Risk distance in price units (1R = initial_r ATR)

    if is_long:
        sl_price = entry_price - R
        tp_price = entry_price + target_r * atr_at_entry if tp_enabled else None
    else:
        sl_price = entry_price + R
        tp_price = entry_price - target_r * atr_at_entry if tp_enabled else None

    future = bars.loc[bars.index > entry_time]
    if future.empty:
        return {
            "exit_time": None,
            "exit_price": entry_price,
            "pnl_r": 0.0,
            "exit_reason": "no_bars",
            "bars_held": 0,
            "mfe_r": 0.0,
            "mae_r": 0.0,
            "scale_out_done": False,
            "scale_out_fill": None,
        }
    future = future.head(time_stop)

    hwm = entry_price if is_long else None
    lwm = entry_price if not is_long else None
    breakeven_locked = False
    trailing_activated = False
    mfe_r = 0.0
    mae_r = 0.0

    partial_realized_r = 0.0
    rem = 1.0
    scaled = False
    scale_out_fill: Optional[float] = None

    def rail_for_row(row: pd.Series) -> float:
        if so_target in ("opposite_range", "range", "rolling_range"):
            return float(row["range_upper"] if is_long else row["range_lower"])
        return float(row["ols_upper"] if is_long else row["ols_lower"])

    for bar_idx, (ts, bar) in enumerate(future.iterrows(), start=1):
        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
        # MFE / MAE update
        if is_long:
            mfe_r = max(mfe_r, (h - entry_price) / atr_at_entry)
            mae_r = min(mae_r, (l - entry_price) / atr_at_entry)
        else:
            mfe_r = max(mfe_r, (entry_price - l) / atr_at_entry)
            mae_r = min(mae_r, (entry_price - h) / atr_at_entry)

        # Breakeven lock check
        if be_enabled and not breakeven_locked:
            check_price = h if is_long else l
            if is_long:
                profit_r = (check_price - entry_price) / atr_at_entry
            else:
                profit_r = (entry_price - check_price) / atr_at_entry
            if profit_r >= be_trigger_r:
                breakeven_locked = True
                if is_long and entry_price > sl_price:
                    sl_price = entry_price
                elif not is_long and entry_price < sl_price:
                    sl_price = entry_price

        # Update HWM/LWM
        if is_long:
            if hwm is None or h > hwm:
                hwm = h
        else:
            if lwm is None or l < lwm:
                lwm = l

        # Activation trailing
        if trail_enabled:
            check_price = h if is_long else l
            if is_long:
                profit_r = (check_price - entry_price) / atr_at_entry
            else:
                profit_r = (entry_price - check_price) / atr_at_entry
            first_activation_bar = False
            if profit_r >= trail_act_r:
                if not trailing_activated:
                    trailing_activated = True
                    first_activation_bar = True
                if is_long:
                    new_sl = hwm - trail_r * atr_at_entry
                    if new_sl > sl_price:
                        sl_price = new_sl
                else:
                    new_sl = lwm + trail_r * atr_at_entry
                    if new_sl < sl_price:
                        sl_price = new_sl
            # Skip SL / TP / structural scale on first activation bar (mimics position_logic)
            if first_activation_bar:
                continue

        # SL hit before structural scale (same bar: stop-out wins — conservative)
        if is_long and l <= sl_price:
            leg = _leg_pnl_r(
                entry_price,
                sl_price,
                is_long=is_long,
                atr_at_entry=atr_at_entry,
                initial_r=initial_r,
            )
            pnl_total = partial_realized_r + rem * leg
            reason = (
                "trailing_sl"
                if trailing_activated
                else ("breakeven_sl" if breakeven_locked else "sl")
            )
            if scaled:
                reason = f"scale_out+{reason}"
            return {
                "exit_time": ts,
                "exit_price": sl_price,
                "pnl_r": pnl_total,
                "exit_reason": reason,
                "bars_held": bar_idx,
                "mfe_r": mfe_r,
                "mae_r": mae_r,
                "scale_out_done": scaled,
                "scale_out_fill": scale_out_fill,
            }
        if not is_long and h >= sl_price:
            leg = _leg_pnl_r(
                entry_price,
                sl_price,
                is_long=is_long,
                atr_at_entry=atr_at_entry,
                initial_r=initial_r,
            )
            pnl_total = partial_realized_r + rem * leg
            reason = (
                "trailing_sl"
                if trailing_activated
                else ("breakeven_sl" if breakeven_locked else "sl")
            )
            if scaled:
                reason = f"scale_out+{reason}"
            return {
                "exit_time": ts,
                "exit_price": sl_price,
                "pnl_r": pnl_total,
                "exit_reason": reason,
                "bars_held": bar_idx,
                "mfe_r": mfe_r,
                "mae_r": mae_r,
                "scale_out_done": scaled,
                "scale_out_fill": scale_out_fill,
            }

        # Structural scale-out at opposite rail (experimental)
        if so_enabled and not scaled and rem >= 0.999:
            try:
                tgt = rail_for_row(bar)
            except (KeyError, TypeError, ValueError):
                tgt = float("nan")
            if not np.isnan(tgt) and tgt > 0:
                hit = (is_long and h >= tgt) or ((not is_long) and l <= tgt)
                if hit:
                    fill_px = float(tgt)
                    leg = _leg_pnl_r(
                        entry_price,
                        fill_px,
                        is_long=is_long,
                        atr_at_entry=atr_at_entry,
                        initial_r=initial_r,
                    )
                    partial_realized_r += so_frac * leg
                    rem = 1.0 - so_frac
                    scaled = True
                    scale_out_fill = fill_px
                    if so_move_be:
                        if is_long and entry_price > sl_price:
                            sl_price = entry_price
                        elif (not is_long) and entry_price < sl_price:
                            sl_price = entry_price
                        breakeven_locked = True

        # TP hit (applies to remaining fraction)
        if tp_price is not None:
            if is_long and h >= tp_price:
                leg = _leg_pnl_r(
                    entry_price,
                    tp_price,
                    is_long=is_long,
                    atr_at_entry=atr_at_entry,
                    initial_r=initial_r,
                )
                pnl_total = partial_realized_r + rem * leg
                reason = "tp" + ("_after_scale" if scaled else "")
                return {
                    "exit_time": ts,
                    "exit_price": tp_price,
                    "pnl_r": pnl_total,
                    "exit_reason": reason,
                    "bars_held": bar_idx,
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "scale_out_done": scaled,
                    "scale_out_fill": scale_out_fill,
                }
            if not is_long and l <= tp_price:
                leg = _leg_pnl_r(
                    entry_price,
                    tp_price,
                    is_long=is_long,
                    atr_at_entry=atr_at_entry,
                    initial_r=initial_r,
                )
                pnl_total = partial_realized_r + rem * leg
                reason = "tp" + ("_after_scale" if scaled else "")
                return {
                    "exit_time": ts,
                    "exit_price": tp_price,
                    "pnl_r": pnl_total,
                    "exit_reason": reason,
                    "bars_held": bar_idx,
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "scale_out_done": scaled,
                    "scale_out_fill": scale_out_fill,
                }

    # Time stop: exit at last bar close
    last_ts, last_bar = future.index[-1], future.iloc[-1]
    c = last_bar["close"]
    leg = _leg_pnl_r(
        entry_price,
        float(c),
        is_long=is_long,
        atr_at_entry=atr_at_entry,
        initial_r=initial_r,
    )
    pnl_total = partial_realized_r + rem * leg
    reason = "time_stop" + ("_after_scale" if scaled else "")
    return {
        "exit_time": last_ts,
        "exit_price": float(c),
        "pnl_r": pnl_total,
        "exit_reason": reason,
        "bars_held": len(future),
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "scale_out_done": scaled,
        "scale_out_fill": scale_out_fill,
    }


def run(
    trades_glob: str,
    store: str,
    exec_path: str,
    out_path: str,
    *,
    scale_out_cli: Optional[Dict[str, object]] = None,
) -> None:
    exec_cfg = load_exec(exec_path)
    so_cfg: Dict[str, object] = dict(exec_cfg.get("scale_out") or {})
    if scale_out_cli:
        so_cfg.update(scale_out_cli)
    so_enabled = bool(so_cfg.get("enabled"))

    print(f"execution config: {exec_path}")
    print(f"  sl.initial_r = {exec_cfg.get('stop_loss',{}).get('initial_r')}")
    print(
        f"  tp.enabled   = {exec_cfg.get('take_profit',{}).get('enabled')}   target_r = {exec_cfg.get('take_profit',{}).get('target_r')}"
    )
    print(
        f"  trail.enabled= {exec_cfg.get('stop_loss',{}).get('trailing',{}).get('enabled')}   act={exec_cfg.get('stop_loss',{}).get('trailing',{}).get('activation_r')}  trail_r={exec_cfg.get('stop_loss',{}).get('trailing',{}).get('trail_r')}"
    )
    print(
        f"  be.enabled   = {exec_cfg.get('breakeven',{}).get('enabled')}   trigger_r={exec_cfg.get('breakeven',{}).get('trigger_r')}"
    )
    print(f"  time_stop    = {exec_cfg.get('holding',{}).get('time_stop_bars')}")
    if so_enabled:
        print(
            f"  scale_out      enabled  target={so_cfg.get('target')}  fraction={so_cfg.get('fraction', 0.5)}"
        )
    print()

    files = sorted(glob.glob(trades_glob))
    dfs = [pd.read_csv(p) for p in files if os.path.getsize(p) > 10]
    dfs = [d for d in dfs if len(d)]
    trades = pd.concat(dfs, ignore_index=True)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert(
        None
    )
    print(f"loaded baseline trades: {len(trades)} from {len(files)} files")

    ols_w = int(so_cfg.get("ols_window", 96) or 96)
    range_w = int(so_cfg.get("range_window", 20) or 20)

    bars_cache: Dict[str, pd.DataFrame] = {}
    for sym in trades["symbol"].unique():
        b = load_bars(store, sym)
        if b.empty:
            print(f"WARN: no bars for {sym}")
            continue
        if so_enabled:
            b = attach_opposite_rails(b, ols_window=ols_w, range_window=range_w)
        bars_cache[sym] = b
    print(f"loaded bars for {len(bars_cache)} symbols")

    results = []
    for i, t in trades.iterrows():
        sym = t["symbol"]
        if sym not in bars_cache:
            continue
        atr = float(t.get("atr", 0) or 0)
        if atr <= 0:
            continue
        r = simulate_trade(
            bars_cache[sym],
            entry_time=t["entry_time"],
            side=t["side"],
            entry_price=float(t["entry_price"]),
            atr_at_entry=atr,
            exec_cfg=exec_cfg,
            scale_out=so_cfg if so_enabled else None,
        )
        r["symbol"] = sym
        r["side"] = t["side"]
        r["entry_time"] = t["entry_time"]
        r["entry_price"] = float(t["entry_price"])
        r["atr_at_entry"] = atr
        r["baseline_pnl_r"] = float(t.get("pnl_r", 0) or 0)
        r["baseline_exit_reason"] = t.get("exit_reason", "")
        r["baseline_bars_held"] = t.get("bars_held", None)
        results.append(r)

    out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}  ({len(out)} rows)")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"{'metric':<20} {'BASELINE':>10}   {'ALT (resim)':>12}   {'Δ':>8}")
    print("-" * 70)

    def stats(s, key):
        s = s.dropna()
        n = len(s)
        total = s.sum()
        mean = s.mean() if n else 0
        win = (s > 0).mean() * 100 if n else 0
        return n, total, mean, win

    b_n, b_t, b_m, b_w = stats(out["baseline_pnl_r"], "b")
    a_n, a_t, a_m, a_w = stats(out["pnl_r"], "a")
    print(f"{'n':<20} {b_n:>10}   {a_n:>12}")
    print(f"{'totalR':<20} {b_t:>+10.2f}   {a_t:>+12.2f}   {a_t-b_t:>+8.2f}")
    print(f"{'meanR':<20} {b_m:>+10.3f}   {a_m:>+12.3f}   {a_m-b_m:>+8.3f}")
    print(f"{'win %':<20} {b_w:>10.1f}   {a_w:>12.1f}")

    # Winner tail distribution
    print(f"\n{'Winners by R bucket':<30} {'baseline':>10}   {'alt':>10}")
    for lo, hi in [(0, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 5), (5, 100)]:
        b_c = ((out["baseline_pnl_r"] > lo) & (out["baseline_pnl_r"] <= hi)).sum()
        a_c = ((out["pnl_r"] > lo) & (out["pnl_r"] <= hi)).sum()
        print(f"  ({lo:>4.1f}, {hi:>5.1f}]{'':<14} {b_c:>10}   {a_c:>10}")

    # Exit reason breakdown (alt)
    print(f"\nAlt exit reasons:")
    for r, n in out["exit_reason"].value_counts().items():
        g = out[out["exit_reason"] == r]
        print(
            f"  {r:<20} n={n:4d} ({n/len(out)*100:5.1f}%)  meanR={g['pnl_r'].mean():+.3f}  totalR={g['pnl_r'].sum():+.2f}"
        )

    # Sharpe
    out["m"] = pd.to_datetime(out["entry_time"]).dt.to_period("M").astype(str)
    mon_b = out.groupby("m")["baseline_pnl_r"].sum()
    mon_a = out.groupby("m")["pnl_r"].sum()
    all_months = sorted(set(mon_b.index) | set(mon_a.index))
    mon_b = mon_b.reindex(all_months, fill_value=0)
    mon_a = mon_a.reindex(all_months, fill_value=0)
    sh_b = mon_b.mean() / max(mon_b.std(), 1e-9) * np.sqrt(12)
    sh_a = mon_a.mean() / max(mon_a.std(), 1e-9) * np.sqrt(12)
    print(f"\n{'Sharpe(ann)':<20} {sh_b:>10.2f}   {sh_a:>12.2f}")

    # Max DD
    cum_b = out.sort_values("entry_time")["baseline_pnl_r"].cumsum()
    cum_a = out.sort_values("entry_time")["pnl_r"].cumsum()
    dd_b = (cum_b - cum_b.cummax()).min()
    dd_a = (cum_a - cum_a.cummax()).min()
    print(f"{'maxDD(R)':<20} {dd_b:>10.2f}   {dd_a:>12.2f}")

    # Max winner
    print(
        f"{'max winner R':<20} {out['baseline_pnl_r'].max():>10.2f}   {out['pnl_r'].max():>12.2f}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True, help="glob for event_trades_*.csv")
    ap.add_argument(
        "--feature-store",
        required=True,
        help="feature_store/features_<strat>_120T_<hash>/",
    )
    ap.add_argument("--exec-config", required=True, help="path to execution.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--scale-out-target",
        default="none",
        choices=("none", "opposite_ols", "opposite_range"),
        help="Structural partial exit at opposite rail (merged into execution.yaml scale_out)",
    )
    ap.add_argument("--scale-out-fraction", type=float, default=0.5)
    ap.add_argument("--ols-window", type=int, default=96)
    ap.add_argument("--range-window", type=int, default=20)
    args = ap.parse_args()
    so_cli: Optional[Dict[str, object]] = None
    if args.scale_out_target != "none":
        so_cli = {
            "enabled": True,
            "target": args.scale_out_target,
            "fraction": float(args.scale_out_fraction),
            "ols_window": int(args.ols_window),
            "range_window": int(args.range_window),
        }
    run(
        args.trades,
        args.feature_store,
        args.exec_config,
        args.out,
        scale_out_cli=so_cli,
    )
