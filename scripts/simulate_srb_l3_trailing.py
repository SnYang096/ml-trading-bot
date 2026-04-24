"""
SRB 逐 bar 复盘 + L3 dynamic trailing 模拟
=========================================

用途：
  1) 诊断："被切掉多少趋势"—— per-trade 计算 MFE / MAE / captured_pct，以及
     hold 期间与反向 L3 的最近距离；按 regime / symbol / month 分层。
  2) 模拟：对每笔首单，用 feature store bar 回放 entry_price/initial_SL，
     套用"**按 wide_sr_dist_atr 动态收放 trailing**"策略，计算 alt_pnl_r，扫二维
     参数 (M_far, M_near, L3_threshold)，和原始 pnl_r 对比。

策略语义（L3 dynamic trailing）：
  - 价距反向 L3 >= thr_l3_atr → trailing 使用 M_far × ATR（宽容）
  - 价距反向 L3 <  thr_l3_atr → trailing 使用 M_near × ATR（收紧）
  - "反向 L3":
        LONG  -> wide_sr_upper_px (上沿)
        SHORT -> wide_sr_lower_px (下沿)
  - trailing 锚点：入场后 MFE（LONG max high；SHORT min low）
  - 只对 reward 侧生效；初始 SL 按 effective_stop_pct 原样保留
  - 入场 bar 的下一根开始走；最大 hold = max_hold_bars

用法：
    python scripts/simulate_srb_l3_trailing.py \
        --trades reports/srb_break_level_attribution_v2_alltrades_trades.parquet \
        --feature-store feature_store/features_srb_120T_5643a66b47 \
        --out reports/srb_l3_dynamic_trailing.json

输出：
  - per-trade 诊断 parquet: `reports/srb_trade_excursions.parquet`
  - per-month XRP 专项表（stdout）
  - 参数扫描 JSON: reports/srb_l3_dynamic_trailing.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
TIMEFRAME_BARS_PER_HOUR = 0.5  # 2H bars


# ---------------------------------------------------------------------------
# Feature store 加载
# ---------------------------------------------------------------------------


def load_symbol_bars(store: str, symbol: str, tf: str = "120T") -> pd.DataFrame:
    files = sorted(glob.glob(f"{store}/{symbol}/{tf}/*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    keep = [
        c
        for c in [
            "open",
            "high",
            "low",
            "close",
            "atr",
            "wide_sr_upper_px",
            "wide_sr_lower_px",
            "wide_sr_dist_atr",
            "wide_sr_side",
            "wide_sr_range_width_atr",
        ]
        if c in df.columns
    ]
    return df[keep]


# ---------------------------------------------------------------------------
# Per-trade bar walk（MFE/MAE + L3 接近度）
# ---------------------------------------------------------------------------


@dataclass
class ExcursionResult:
    mfe_px: float
    mae_px: float
    mfe_r: float
    mae_r: float
    captured_pct: float  # exit_pnl_r / mfe_r, 只对盈利 trade 有意义
    min_reverse_l3_atr: float  # hold 期间距反向 L3 的最小 ATR（越小越危险）
    max_wide_dist_same: float  # 同向 (wide_sr_side 对齐) 时 wide_dist 的最大值
    bars_walked: int
    exit_bar_idx: int  # bar index 0 = entry 后第一根
    exit_reason_bar: str  # "hit_sl" / "time_stop" / "reached_original_exit"


def walk_trade(
    bars: pd.DataFrame,
    side: str,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    entry_price: float,
    effective_stop_pct: float,
    max_hold_bars: int = 360,  # 30 天 @ 2h
) -> Optional[ExcursionResult]:
    """从 entry_time 之后第一根 bar 起走到 exit_time 或 SL，统计 MFE/MAE 与 wide_sr 接近度。"""
    if (
        bars is None
        or bars.empty
        or effective_stop_pct is None
        or effective_stop_pct <= 0
    ):
        return None
    is_long = side.upper() == "LONG"
    sl_px = (
        entry_price * (1 - effective_stop_pct)
        if is_long
        else entry_price * (1 + effective_stop_pct)
    )
    sl_dist = abs(entry_price - sl_px)
    if sl_dist <= 0:
        return None

    after = bars.loc[bars.index > entry_time]
    if after.empty:
        return None
    after = after.iloc[:max_hold_bars]

    mfe_px = 0.0
    mae_px = 0.0
    min_rev_atr = np.inf
    max_same_dist = -np.inf
    exit_idx = -1
    exit_reason = "reached_original_exit"

    for i, (ts, row) in enumerate(after.iterrows()):
        hi = float(row.get("high", np.nan))
        lo = float(row.get("low", np.nan))
        if not np.isfinite(hi) or not np.isfinite(lo):
            continue
        if is_long:
            mfe_px = max(mfe_px, hi - entry_price)
            mae_px = min(mae_px, lo - entry_price)
            if lo <= sl_px:
                exit_idx = i
                exit_reason = "hit_sl"
                break
        else:
            mfe_px = max(mfe_px, entry_price - lo)
            mae_px = min(mae_px, entry_price - hi)
            if hi >= sl_px:
                exit_idx = i
                exit_reason = "hit_sl"
                break

        w_upper = row.get("wide_sr_upper_px", np.nan)
        w_lower = row.get("wide_sr_lower_px", np.nan)
        atr = row.get("atr", np.nan)
        if np.isfinite(atr) and atr > 0:
            if is_long and np.isfinite(w_upper):
                rev_atr = max(
                    (w_upper - float(row.get("close", entry_price))) / atr, 0.0
                )
                min_rev_atr = min(min_rev_atr, rev_atr)
            if (not is_long) and np.isfinite(w_lower):
                rev_atr = max(
                    (float(row.get("close", entry_price)) - w_lower) / atr, 0.0
                )
                min_rev_atr = min(min_rev_atr, rev_atr)
            ws_dist = row.get("wide_sr_dist_atr", np.nan)
            ws_side = row.get("wide_sr_side", np.nan)
            if np.isfinite(ws_dist) and np.isfinite(ws_side):
                if (is_long and ws_side > 0) or ((not is_long) and ws_side < 0):
                    max_same_dist = max(max_same_dist, float(ws_dist))

        if ts >= exit_time:
            exit_idx = i
            exit_reason = "reached_original_exit"
            break

    if exit_idx < 0:
        exit_idx = len(after) - 1
        exit_reason = "time_stop"

    mfe_r = mfe_px / sl_dist
    mae_r = mae_px / sl_dist
    # captured_pct：只对 mfe>0 的 trade 有意义
    return ExcursionResult(
        mfe_px=mfe_px,
        mae_px=mae_px,
        mfe_r=mfe_r,
        mae_r=mae_r,
        captured_pct=float("nan"),  # 外层用实际 pnl_r 填
        min_reverse_l3_atr=(
            float(min_rev_atr) if np.isfinite(min_rev_atr) else float("nan")
        ),
        max_wide_dist_same=(
            float(max_same_dist) if np.isfinite(max_same_dist) else float("nan")
        ),
        bars_walked=exit_idx + 1,
        exit_bar_idx=exit_idx,
        exit_reason_bar=exit_reason,
    )


# ---------------------------------------------------------------------------
# L3 dynamic trailing 模拟
# ---------------------------------------------------------------------------


@dataclass
class SimParams:
    """生产端默认：activation_r=6.0, trail_r=5.0, breakeven disabled, initial_r=6.0。
    使用动态 L3：接近反向 L3 时把 trail_r 收紧到 m_near，远离时维持 m_far。
    """

    m_far: float = 5.0
    m_near: float = 2.5
    thr_l3_atr: float = 2.0
    activation_r: float = 6.0  # trailing 激活门槛（R 单位）
    breakeven_lock_r: float = 0.0  # 0 = off；>0 表示 MFE ≥ N R 时把 SL 抬到 entry
    max_hold_bars: int = 360


def simulate_trailing(
    bars: pd.DataFrame,
    side: str,
    entry_time: pd.Timestamp,
    entry_price: float,
    effective_stop_pct: float,
    p: SimParams,
) -> Tuple[float, str, int]:
    """返回 (alt_pnl_r, reason, bars_held)"""
    if (
        bars is None
        or bars.empty
        or effective_stop_pct is None
        or effective_stop_pct <= 0
    ):
        return (float("nan"), "nodata", 0)
    is_long = side.upper() == "LONG"
    sl_px = (
        entry_price * (1 - effective_stop_pct)
        if is_long
        else entry_price * (1 + effective_stop_pct)
    )
    sl_dist = abs(entry_price - sl_px)
    if sl_dist <= 0:
        return (float("nan"), "nodata", 0)
    after = bars.loc[bars.index > entry_time].iloc[: p.max_hold_bars]
    if after.empty:
        return (float("nan"), "nodata", 0)

    trailing_active = False
    breakeven_locked = False
    active_sl = sl_px
    best_px = entry_price

    for i, (ts, row) in enumerate(after.iterrows()):
        hi = float(row.get("high", np.nan))
        lo = float(row.get("low", np.nan))
        cl = float(row.get("close", np.nan))
        atr = float(row.get("atr", np.nan))
        if (
            not np.isfinite(hi)
            or not np.isfinite(lo)
            or not np.isfinite(atr)
            or atr <= 0
        ):
            continue

        if is_long:
            if lo <= active_sl:
                reason = (
                    "trailing_sl"
                    if trailing_active
                    else ("breakeven" if breakeven_locked else "sl")
                )
                return ((active_sl - entry_price) / sl_dist, reason, i + 1)
            best_px = max(best_px, hi)
            running_r = (best_px - entry_price) / sl_dist
            # breakeven lock
            if (
                (p.breakeven_lock_r > 0)
                and (not breakeven_locked)
                and running_r >= p.breakeven_lock_r
            ):
                active_sl = max(active_sl, entry_price)
                breakeven_locked = True
            # trailing activation
            if not trailing_active and running_r >= p.activation_r:
                trailing_active = True
            if trailing_active:
                w_upper = row.get("wide_sr_upper_px", np.nan)
                if np.isfinite(w_upper):
                    rev_dist_atr = max((w_upper - cl) / atr, 0.0)
                    mult = p.m_near if rev_dist_atr < p.thr_l3_atr else p.m_far
                else:
                    mult = p.m_far
                new_trail = best_px - mult * atr
                if new_trail > active_sl:
                    active_sl = new_trail
        else:  # SHORT
            if hi >= active_sl:
                reason = (
                    "trailing_sl"
                    if trailing_active
                    else ("breakeven" if breakeven_locked else "sl")
                )
                return ((entry_price - active_sl) / sl_dist, reason, i + 1)
            best_px = min(best_px, lo)
            running_r = (entry_price - best_px) / sl_dist
            if (
                (p.breakeven_lock_r > 0)
                and (not breakeven_locked)
                and running_r >= p.breakeven_lock_r
            ):
                active_sl = min(active_sl, entry_price)
                breakeven_locked = True
            if not trailing_active and running_r >= p.activation_r:
                trailing_active = True
            if trailing_active:
                w_lower = row.get("wide_sr_lower_px", np.nan)
                if np.isfinite(w_lower):
                    rev_dist_atr = max((cl - w_lower) / atr, 0.0)
                    mult = p.m_near if rev_dist_atr < p.thr_l3_atr else p.m_far
                else:
                    mult = p.m_far
                new_trail = best_px + mult * atr
                if new_trail < active_sl:
                    active_sl = new_trail

    last = after.iloc[-1]
    cl = float(last.get("close", entry_price))
    alt_r = (cl - entry_price) / sl_dist if is_long else (entry_price - cl) / sl_dist
    return (alt_r, "time_stop", len(after))


# ---------------------------------------------------------------------------
# MAIN
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
    ap.add_argument("--out", default="reports/srb_l3_dynamic_trailing.json")
    ap.add_argument("--first-entry-only", action="store_true", default=True)
    ap.add_argument("--max-hold-bars", type=int, default=360)
    args = ap.parse_args()

    print("=" * 72)
    trades = pd.read_parquet(args.trades)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    for c in ("is_add_position", "is_reverse"):
        if c in trades.columns:
            trades[c] = trades[c].fillna(False).astype(bool)
    print(f"loaded : {len(trades)} trades")

    if args.first_entry_only:
        trades = trades[~trades.is_add_position & ~trades.is_reverse].copy()
        print(f"filter first-entry: {len(trades)} rows")

    print("加载 bars...")
    bars_by_sym: Dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        b = load_symbol_bars(args.feature_store, sym)
        if b.empty:
            continue
        bars_by_sym[sym] = b
        print(f"  {sym}: {len(b)} bars [{b.index.min()} .. {b.index.max()}]")

    # ------------------------------------------------------------
    # Part 1: excursion 诊断
    # ------------------------------------------------------------
    print("\n=== Excursion 诊断 ===")
    exc_rows = []
    for _, t in trades.iterrows():
        b = bars_by_sym.get(t["symbol"])
        r = (
            walk_trade(
                b,
                t["side"],
                t["entry_time"],
                t["exit_time"],
                float(t["entry_price"]),
                float(t.get("effective_stop_pct", np.nan)),
                max_hold_bars=args.max_hold_bars,
            )
            if b is not None
            else None
        )
        row = {
            "symbol": t["symbol"],
            "side": t["side"],
            "entry_time": t["entry_time"],
            "exit_time": t["exit_time"],
            "pnl_r": float(t["pnl_r"]) if pd.notna(t.get("pnl_r", np.nan)) else np.nan,
            "exit_reason": t.get("exit_reason", ""),
            "bars_held": t.get("bars_held", np.nan),
        }
        if r is None:
            row.update(
                {
                    "mfe_r": np.nan,
                    "mae_r": np.nan,
                    "captured_pct": np.nan,
                    "min_rev_l3_atr": np.nan,
                    "max_same_dist_atr": np.nan,
                    "exit_reason_sim": "nodata",
                }
            )
        else:
            captured = (
                (row["pnl_r"] / r.mfe_r)
                if (r.mfe_r > 0 and np.isfinite(row["pnl_r"]))
                else float("nan")
            )
            row.update(
                {
                    "mfe_r": r.mfe_r,
                    "mae_r": r.mae_r,
                    "captured_pct": captured,
                    "min_rev_l3_atr": r.min_reverse_l3_atr,
                    "max_same_dist_atr": r.max_wide_dist_same,
                    "exit_reason_sim": r.exit_reason_bar,
                }
            )
        exc_rows.append(row)
    exc = pd.DataFrame(exc_rows)
    exc.to_parquet("reports/srb_trade_excursions.parquet", index=False)

    valid = exc[exc["mfe_r"].notna()]
    print(f"  first-entry excursion 成功: {len(valid)}/{len(exc)}")
    print("\n  按 symbol:")
    for sym, g in valid.groupby("symbol"):
        w = g[g.pnl_r > 0]
        l = g[g.pnl_r <= 0]
        print(
            f"    {sym}: n={len(g):3d}  mfe~{g.mfe_r.mean():+.2f}  mae~{g.mae_r.mean():+.2f}  "
            f"captured~{w.captured_pct.dropna().mean():.2f}  "
            f"min_rev_l3~{g.min_rev_l3_atr.mean():.2f}  pnl={g.pnl_r.sum():+6.2f}"
        )

    print("\n  XRP per-month 专项（含 add_pos, 从原 parquet）:")
    trades_all = pd.read_parquet(args.trades)
    trades_all["entry_time"] = pd.to_datetime(trades_all["entry_time"])
    xrp_all = trades_all[trades_all.symbol == "XRPUSDT"].copy()
    xrp_all["ym"] = xrp_all.entry_time.dt.to_period("M")
    for ym, g in xrp_all.groupby("ym"):
        first = g[~g.is_add_position.fillna(False)]
        adds = g[g.is_add_position.fillna(False)]
        print(
            f"    {ym}: first={len(first):2d} add={len(adds):2d} totalR={g.pnl_r.sum():+7.2f} "
            f"win={(g.pnl_r>0).mean():.2f} exits={g.exit_reason.value_counts().to_dict()}"
        )

    print("\n=== 按 exit_reason 统计 leftover profit (MFE - 已实现 R) ===")
    for er, g in valid.groupby("exit_reason"):
        leftover = g["mfe_r"] - g["pnl_r"]
        print(
            f"  {er:14s}: n={len(g):3d}  avg_pnl={g.pnl_r.mean():+.2f}  avg_mfe={g.mfe_r.mean():+.2f}  avg_leftover={leftover.mean():+.2f}"
        )

    # ------------------------------------------------------------
    # Part 2: L3 dynamic trailing 扫参
    # ------------------------------------------------------------
    print("\n=== L3 dynamic trailing 扫参 ===")
    grid = []
    orig_total = float(exc["pnl_r"].sum())
    orig_mean = float(exc["pnl_r"].mean())
    orig_win = float((exc["pnl_r"] > 0).mean())
    print(
        f"  ORIGINAL:  n={len(exc)}  totalR={orig_total:+.2f}  meanR={orig_mean:+.3f}  win={orig_win:.3f}"
    )

    sweep_m_far = [3.0, 5.0, 7.0]
    sweep_m_near = [1.5, 2.5, 3.5, 5.0, 7.0]  # m_near==m_far 表示 "no L3 switching"
    sweep_thr = [2.0]  # thr 扫描已确认不敏感，固定 2.0
    sweep_activation = [1.0, 2.0, 3.0, 6.0]
    sweep_breakeven = [0.0, 1.0, 2.0]  # 0.5 已确认灾难，移除

    for m_far in sweep_m_far:
        for m_near in sweep_m_near:
            if m_near > m_far:  # 允许 == 以表示"no L3 switching"
                continue
            for thr in sweep_thr:
                for act in sweep_activation:
                    for be in sweep_breakeven:
                        p = SimParams(
                            m_far=m_far,
                            m_near=m_near,
                            thr_l3_atr=thr,
                            activation_r=act,
                            breakeven_lock_r=be,
                            max_hold_bars=args.max_hold_bars,
                        )
                        alt_pnls = []
                        for _, t in trades.iterrows():
                            b = bars_by_sym.get(t["symbol"])
                            alt, _, _ = (
                                simulate_trailing(
                                    b,
                                    t["side"],
                                    t["entry_time"],
                                    float(t["entry_price"]),
                                    float(t.get("effective_stop_pct", np.nan)),
                                    p,
                                )
                                if b is not None
                                else (float("nan"), "nodata", 0)
                            )
                            alt_pnls.append(alt)
                        alt = pd.Series(alt_pnls).replace([np.inf, -np.inf], np.nan)
                        valid_mask = alt.notna() & exc["pnl_r"].notna()
                        n_valid = int(valid_mask.sum())
                        if n_valid == 0:
                            continue
                        a = alt[valid_mask]
                        o = exc.loc[valid_mask, "pnl_r"]
                        grid.append(
                            {
                                "m_far": m_far,
                                "m_near": m_near,
                                "thr_l3_atr": thr,
                                "activation_r": act,
                                "breakeven_r": be,
                                "n": n_valid,
                                "orig_totalR": float(o.sum()),
                                "alt_totalR": float(a.sum()),
                                "delta_totalR": float(a.sum() - o.sum()),
                                "orig_meanR": float(o.mean()),
                                "alt_meanR": float(a.mean()),
                                "alt_win": float((a > 0).mean()),
                                "n_improved": int((a > o).sum()),
                                "n_worsened": int((a < o).sum()),
                            }
                        )
    grid_df = pd.DataFrame(grid).sort_values("delta_totalR", ascending=False)
    grid_df["l3_dynamic"] = (grid_df["m_near"] < grid_df["m_far"]).astype(int)
    print("\nTop 15 配置 by delta_totalR:")
    print(grid_df.head(15).to_string(index=False))
    print("\nBottom 5 配置:")
    print(grid_df.tail(5).to_string(index=False))

    # 对比分析：L3 dynamic 是否比 fixed trailing 有额外增量？
    print("\n=== L3 dynamic vs fixed trailing 增量分析 ===")
    print(
        "    按 (activation_r, breakeven_r) 分组, 取同组最优 fixed 与最优 dynamic 对比"
    )
    groups = grid_df.groupby(["activation_r", "breakeven_r"])
    rows = []
    for (act, be), g in groups:
        fixed = g[g.l3_dynamic == 0]
        dyn = g[g.l3_dynamic == 1]
        if len(fixed) == 0 or len(dyn) == 0:
            continue
        best_fixed = fixed.sort_values("delta_totalR", ascending=False).iloc[0]
        best_dyn = dyn.sort_values("delta_totalR", ascending=False).iloc[0]
        rows.append(
            {
                "activation_r": act,
                "breakeven_r": be,
                "fixed_best_delta": float(best_fixed.delta_totalR),
                "fixed_best_(m,)": f"m={best_fixed.m_far}",
                "dyn_best_delta": float(best_dyn.delta_totalR),
                "dyn_best_(m_far,m_near)": f"({best_dyn.m_far}/{best_dyn.m_near})",
                "dyn_gain_over_fixed": float(
                    best_dyn.delta_totalR - best_fixed.delta_totalR
                ),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    # ------------------------------------------------------------
    # 写报告
    # ------------------------------------------------------------
    report = {
        "config": {
            "trades": args.trades,
            "feature_store": args.feature_store,
            "first_entry_only": args.first_entry_only,
            "max_hold_bars": args.max_hold_bars,
        },
        "original": {
            "n": int(len(exc)),
            "totalR": orig_total,
            "meanR": orig_mean,
            "winrate": orig_win,
        },
        "excursion_by_symbol": [
            {
                "symbol": s,
                "n": int(len(g)),
                "mean_mfe_r": float(g.mfe_r.mean()),
                "mean_mae_r": float(g.mae_r.mean()),
                "mean_captured_pct": float(g[g.pnl_r > 0]["captured_pct"].mean()),
                "mean_min_rev_l3_atr": float(g.min_rev_l3_atr.mean()),
                "totalR": float(g.pnl_r.sum()),
            }
            for s, g in valid.groupby("symbol")
        ],
        "leftover_by_exit_reason": [
            {
                "exit_reason": er,
                "n": int(len(g)),
                "mean_pnl_r": float(g.pnl_r.mean()),
                "mean_mfe_r": float(g.mfe_r.mean()),
                "mean_leftover_r": float((g.mfe_r - g.pnl_r).mean()),
            }
            for er, g in valid.groupby("exit_reason")
        ],
        "sweep": grid_df.to_dict(orient="records"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n报告: {args.out}")
    print("Per-trade excursions: reports/srb_trade_excursions.parquet")


if __name__ == "__main__":
    main()
