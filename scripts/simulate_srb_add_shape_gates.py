"""
SRB add-leg shape gate sweep (Phase D.3)
========================================

对每笔 SRB 母仓首单，模拟 "开仓 → trailing → 加仓 → 母+子全部共享 SL" 的全过程，
扫描 `post_hoc_shape_gate` 4 个子门（retrace / momentum / r2 / wide_expansion）在
单独 enable 时对 total R / add_count 的边际贡献。

与 simulate_srb_l3_trailing.py 的区别：
  - 主 simulator 只跑 "首单 + 动态 trailing"，不加 adds；
  - 本脚本以 Phase A 结论（activation_r=1, trail_r_far=7, trail_r_near=5, thr=2）
    + Phase B（mother_breakeven_r=3）作为 baseline，然后启用 adds，测每个 gate
    单独 enable 的 delta_totalR。

baseline：
  mother_params = SimParams(m_far=7, m_near=5, thr_l3_atr=2, activation_r=1,
                            breakeven_lock_r=0, max_hold_bars=360)
  mother_breakeven_r = 3.0
  add_ladder = [0.5, 1.0, 1.5]  (current_r 阈值)
  add_size_multipliers = [0.8, 0.5, 0.3]
  inherit_parent_stop = true

用法：
  python scripts/simulate_srb_add_shape_gates.py \
      --trades reports/srb_break_level_attribution_v2_alltrades_trades.parquet \
      --feature-store feature_store/features_srb_120T_5643a66b47 \
      --out reports/srb_add_shape_gate_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scripts.simulate_srb_l3_trailing import load_symbol_bars, SYMBOLS


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


@dataclass
class MotherParams:
    m_far: float = 7.0
    m_near: float = 5.0
    thr_l3_atr: float = 2.0
    activation_r: float = 1.0
    mother_breakeven_r: float = 3.0
    max_hold_bars: int = 360


@dataclass
class AddLadder:
    min_current_r: Tuple[float, ...] = (0.5, 1.0, 1.5)
    size_mult: Tuple[float, ...] = (0.8, 0.5, 0.3)


@dataclass
class ShapeGateCfg:
    retrace_guard_enabled: bool = False
    retrace_min_captured_pct: float = 0.7
    momentum_enabled: bool = False
    momentum_lookback: int = 6
    momentum_min_move_atr: float = 1.5
    r2_gate_enabled: bool = False
    r2_min: float = 0.4
    wide_expansion_enabled: bool = False
    wide_expansion_min_atr: float = 1.0


@dataclass
class AddLeg:
    entry_idx: int
    entry_price: float
    size_mult: float


@dataclass
class MotherTradeResult:
    total_r: float  # Σ (leg pnl × size) / mother sl_dist
    mother_pnl_r: float
    n_adds_triggered: int
    n_adds_accepted: int
    exit_reason: str


# ---------------------------------------------------------------------------
# Shape gate evaluation
# ---------------------------------------------------------------------------


def _evaluate_gate(
    side: str,
    current_r: float,
    mfe_r: float,
    row: pd.Series,
    bars_before: pd.DataFrame,
    entry_wide_dist: float,
    cfg: ShapeGateCfg,
) -> Tuple[bool, str]:
    """True = reject this add."""
    if cfg.retrace_guard_enabled and mfe_r > 0:
        if current_r < cfg.retrace_min_captured_pct * mfe_r:
            return True, "retrace"

    if cfg.momentum_enabled:
        # recent_net_move_atr: 最近 lookback 根 close 净变化 / atr，符号 = 价格方向
        if len(bars_before) >= 2:
            tail = bars_before.tail(cfg.momentum_lookback + 1)
            net = float(tail["close"].iloc[-1] - tail["close"].iloc[0])
            atr_now = float(row.get("atr", 0.0)) or float(tail["atr"].iloc[-1])
            if atr_now > 0:
                move = net / atr_now  # 带符号
                if side.upper() in ("LONG", "BUY") and move < cfg.momentum_min_move_atr:
                    return True, "momentum"
                if (
                    side.upper() in ("SHORT", "SELL")
                    and move > -cfg.momentum_min_move_atr
                ):
                    return True, "momentum"

    if cfg.r2_gate_enabled:
        r2 = float(row.get("trend_r2_20", np.nan))
        if not np.isfinite(r2):
            # 回退到 close 的滚动线性回归 R²（近 20 bar）
            tail = bars_before.tail(20)
            if len(tail) >= 5:
                y = tail["close"].values.astype(float)
                x = np.arange(len(y), dtype=float)
                var = y.var()
                if var > 0:
                    slope = np.polyfit(x, y, 1)[0]
                    yhat = np.polyval([slope, y.mean() - slope * x.mean()], x)
                    ss_res = ((y - yhat) ** 2).sum()
                    ss_tot = ((y - y.mean()) ** 2).sum()
                    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                else:
                    r2 = 0.0
        if np.isfinite(r2) and r2 < cfg.r2_min:
            return True, "r2"

    if cfg.wide_expansion_enabled:
        cur_wide = float(row.get("wide_sr_dist_atr", np.nan))
        if np.isfinite(cur_wide) and np.isfinite(entry_wide_dist):
            if (cur_wide - entry_wide_dist) < cfg.wide_expansion_min_atr:
                return True, "wide_expansion"

    return False, ""


# ---------------------------------------------------------------------------
# Mother + add leg simulator
# ---------------------------------------------------------------------------


def simulate_mother_with_adds(
    bars: pd.DataFrame,
    side: str,
    entry_time: pd.Timestamp,
    entry_price: float,
    effective_stop_pct: float,
    mother: MotherParams,
    ladder: AddLadder,
    gate: ShapeGateCfg,
) -> Optional[MotherTradeResult]:
    if bars is None or bars.empty:
        return None
    if not effective_stop_pct or effective_stop_pct <= 0:
        return None
    is_long = side.upper() in ("LONG", "BUY")
    sl_px = (
        entry_price * (1 - effective_stop_pct)
        if is_long
        else entry_price * (1 + effective_stop_pct)
    )
    sl_dist = abs(entry_price - sl_px)
    if sl_dist <= 0:
        return None

    before_all = bars.loc[bars.index <= entry_time]
    after = bars.loc[bars.index > entry_time].iloc[: mother.max_hold_bars]
    if after.empty:
        return None

    # entry snapshot for wide expansion gate
    if not before_all.empty and "wide_sr_dist_atr" in before_all.columns:
        entry_wide_dist = float(before_all["wide_sr_dist_atr"].iloc[-1])
    else:
        entry_wide_dist = (
            float(after["wide_sr_dist_atr"].iloc[0])
            if "wide_sr_dist_atr" in after.columns
            else np.nan
        )

    active_sl = sl_px
    best_px = entry_price
    trailing_active = False
    mother_breakeven_locked = False
    add_legs: List[AddLeg] = []
    next_add_idx = 0
    n_triggered = 0
    exit_reason = "time_stop"
    exit_bar_idx = len(after) - 1

    for i, (ts, row) in enumerate(after.iterrows()):
        hi = float(row.get("high", np.nan))
        lo = float(row.get("low", np.nan))
        cl = float(row.get("close", np.nan))
        atr = float(row.get("atr", np.nan))
        if not (np.isfinite(hi) and np.isfinite(lo) and np.isfinite(atr) and atr > 0):
            continue

        # 1. SL hit (tighten-only)
        if is_long and lo <= active_sl:
            exit_reason = (
                "trailing_sl"
                if trailing_active
                else ("mother_breakeven" if mother_breakeven_locked else "sl")
            )
            exit_bar_idx = i
            break
        if (not is_long) and hi >= active_sl:
            exit_reason = (
                "trailing_sl"
                if trailing_active
                else ("mother_breakeven" if mother_breakeven_locked else "sl")
            )
            exit_bar_idx = i
            break

        # 2. update HWM/LWM
        if is_long:
            best_px = max(best_px, hi)
            running_r = (best_px - entry_price) / sl_dist
            current_r = (cl - entry_price) / sl_dist
        else:
            best_px = min(best_px, lo)
            running_r = (entry_price - best_px) / sl_dist
            current_r = (entry_price - cl) / sl_dist

        # 3. mother breakeven lock
        if (
            mother.mother_breakeven_r > 0
            and not mother_breakeven_locked
            and running_r >= mother.mother_breakeven_r
        ):
            if is_long:
                active_sl = max(active_sl, entry_price)
            else:
                active_sl = min(active_sl, entry_price)
            mother_breakeven_locked = True

        # 4. trailing activation + L3 dynamic trail_r
        if not trailing_active and running_r >= mother.activation_r:
            trailing_active = True
        if trailing_active:
            if is_long:
                w_upper = row.get("wide_sr_upper_px", np.nan)
                if np.isfinite(w_upper):
                    rev_dist = max((w_upper - cl) / atr, 0.0)
                    mult = (
                        mother.m_near if rev_dist < mother.thr_l3_atr else mother.m_far
                    )
                else:
                    mult = mother.m_far
                new_trail = best_px - mult * atr
                if new_trail > active_sl:
                    active_sl = new_trail
            else:
                w_lower = row.get("wide_sr_lower_px", np.nan)
                if np.isfinite(w_lower):
                    rev_dist = max((cl - w_lower) / atr, 0.0)
                    mult = (
                        mother.m_near if rev_dist < mother.thr_l3_atr else mother.m_far
                    )
                else:
                    mult = mother.m_far
                new_trail = best_px + mult * atr
                if new_trail < active_sl:
                    active_sl = new_trail

        # 5. add triggers: 逐层 trigger，shape gate 决定 accept
        while (
            next_add_idx < len(ladder.min_current_r)
            and current_r >= ladder.min_current_r[next_add_idx]
        ):
            n_triggered += 1
            # shape gate
            bars_before_add = bars.loc[bars.index <= ts]
            reject, _ = _evaluate_gate(
                side, current_r, running_r, row, bars_before_add, entry_wide_dist, gate
            )
            if not reject:
                add_legs.append(
                    AddLeg(
                        entry_idx=i,
                        entry_price=cl,
                        size_mult=ladder.size_mult[next_add_idx],
                    )
                )
            next_add_idx += 1

    # 如果 loop 结束无 SL/time_stop，结束位取 exit_bar_idx + close
    if exit_reason == "time_stop":
        exit_bar_idx = len(after) - 1
        exit_cl = float(after.iloc[exit_bar_idx]["close"])
        if is_long:
            mother_pnl = (exit_cl - entry_price) / sl_dist
        else:
            mother_pnl = (entry_price - exit_cl) / sl_dist
        add_legs_pnl = []
        for leg in add_legs:
            if is_long:
                add_legs_pnl.append(
                    ((exit_cl - leg.entry_price) / sl_dist) * leg.size_mult
                )
            else:
                add_legs_pnl.append(
                    ((leg.entry_price - exit_cl) / sl_dist) * leg.size_mult
                )
    else:
        # SL hit at active_sl
        if is_long:
            mother_pnl = (active_sl - entry_price) / sl_dist
        else:
            mother_pnl = (entry_price - active_sl) / sl_dist
        add_legs_pnl = []
        for leg in add_legs:
            if leg.entry_idx > exit_bar_idx:
                continue  # 这个 add leg 还没发生
            if is_long:
                add_legs_pnl.append(
                    ((active_sl - leg.entry_price) / sl_dist) * leg.size_mult
                )
            else:
                add_legs_pnl.append(
                    ((leg.entry_price - active_sl) / sl_dist) * leg.size_mult
                )

    return MotherTradeResult(
        total_r=float(mother_pnl + sum(add_legs_pnl)),
        mother_pnl_r=float(mother_pnl),
        n_adds_triggered=n_triggered,
        n_adds_accepted=len(add_legs),
        exit_reason=exit_reason,
    )


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def sweep(
    trades: pd.DataFrame,
    bars_by_sym: Dict[str, pd.DataFrame],
    mother: MotherParams,
    ladder: AddLadder,
) -> Dict:
    gate_configs = {
        "all_off": ShapeGateCfg(),
        "retrace_only_0.5": ShapeGateCfg(
            retrace_guard_enabled=True, retrace_min_captured_pct=0.5
        ),
        "retrace_only_0.7": ShapeGateCfg(
            retrace_guard_enabled=True, retrace_min_captured_pct=0.7
        ),
        "retrace_only_0.8": ShapeGateCfg(
            retrace_guard_enabled=True, retrace_min_captured_pct=0.8
        ),
        "momentum_only_1.0": ShapeGateCfg(
            momentum_enabled=True, momentum_min_move_atr=1.0
        ),
        "momentum_only_1.5": ShapeGateCfg(
            momentum_enabled=True, momentum_min_move_atr=1.5
        ),
        "momentum_only_2.0": ShapeGateCfg(
            momentum_enabled=True, momentum_min_move_atr=2.0
        ),
        "r2_only_0.3": ShapeGateCfg(r2_gate_enabled=True, r2_min=0.3),
        "r2_only_0.4": ShapeGateCfg(r2_gate_enabled=True, r2_min=0.4),
        "r2_only_0.5": ShapeGateCfg(r2_gate_enabled=True, r2_min=0.5),
        "wide_exp_0.5": ShapeGateCfg(
            wide_expansion_enabled=True, wide_expansion_min_atr=0.5
        ),
        "wide_exp_1.0": ShapeGateCfg(
            wide_expansion_enabled=True, wide_expansion_min_atr=1.0
        ),
        "wide_exp_2.0": ShapeGateCfg(
            wide_expansion_enabled=True, wide_expansion_min_atr=2.0
        ),
    }
    results = {}
    for name, cfg in gate_configs.items():
        rows = []
        for _, t in trades.iterrows():
            b = bars_by_sym.get(t["symbol"])
            if b is None:
                continue
            res = simulate_mother_with_adds(
                b,
                t["side"],
                t["entry_time"],
                float(t["entry_price"]),
                float(t.get("effective_stop_pct", np.nan)),
                mother,
                ladder,
                cfg,
            )
            if res is None:
                continue
            rows.append(
                {
                    "symbol": t["symbol"],
                    "entry_time": t["entry_time"],
                    "side": t["side"],
                    "total_r": res.total_r,
                    "mother_pnl_r": res.mother_pnl_r,
                    "n_adds_triggered": res.n_adds_triggered,
                    "n_adds_accepted": res.n_adds_accepted,
                    "exit_reason": res.exit_reason,
                }
            )
        df = pd.DataFrame(rows)
        results[name] = {
            "n": int(len(df)),
            "total_r": float(df.total_r.sum()) if len(df) else 0.0,
            "mean_r": float(df.total_r.mean()) if len(df) else 0.0,
            "mother_total_r": float(df.mother_pnl_r.sum()) if len(df) else 0.0,
            "adds_triggered": int(df.n_adds_triggered.sum()) if len(df) else 0,
            "adds_accepted": int(df.n_adds_accepted.sum()) if len(df) else 0,
            "accept_rate": (
                float(df.n_adds_accepted.sum() / df.n_adds_triggered.sum())
                if len(df) and df.n_adds_triggered.sum() > 0
                else float("nan")
            ),
            "win_rate": float((df.total_r > 0).mean()) if len(df) else 0.0,
        }
    return results


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trades",
        default="reports/srb_break_level_attribution_v2_alltrades_trades.parquet",
    )
    ap.add_argument(
        "--feature-store", default="feature_store/features_srb_120T_5643a66b47"
    )
    ap.add_argument("--out", default="reports/srb_add_shape_gate_sweep.json")
    args = ap.parse_args()

    trades = pd.read_parquet(args.trades)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    for c in ("is_add_position", "is_reverse"):
        if c in trades.columns:
            trades[c] = trades[c].fillna(False).astype(bool)
    trades = trades[~trades.is_add_position & ~trades.is_reverse].copy()
    print(f"[sweep] loaded {len(trades)} first-entry mother trades")

    bars_by_sym: Dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        b = load_symbol_bars(args.feature_store, sym)
        if b.empty:
            continue
        bars_by_sym[sym] = b
    print(f"[sweep] loaded bars for {len(bars_by_sym)} symbols")

    mother = MotherParams()
    ladder = AddLadder()
    results = sweep(trades, bars_by_sym, mother, ladder)

    # pretty print
    print("\n" + "=" * 84)
    print(
        f"{'gate_config':>22s}  {'n':>4s}  {'total_r':>9s}  {'mother_r':>9s}  {'adds_acc/trig':>14s}  {'win':>5s}"
    )
    print("-" * 84)
    baseline = results.get("all_off", {})
    for name, r in results.items():
        delta = (
            r["total_r"] - baseline.get("total_r", 0.0) if name != "all_off" else 0.0
        )
        print(
            f"{name:>22s}  {r['n']:>4d}  {r['total_r']:>+9.2f}  {r['mother_total_r']:>+9.2f}  "
            f"{r['adds_accepted']}/{r['adds_triggered']:>4d}  {r['win_rate']:>5.2f}  (Δ {delta:+.2f})"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "config": {
                    "trades": args.trades,
                    "feature_store": args.feature_store,
                    "mother_params": mother.__dict__,
                    "ladder": ladder.__dict__,
                },
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n[sweep] wrote {args.out}")


if __name__ == "__main__":
    main()
