"""
离线诊断脚本（一次性研究用）
=========================
目的：
  1. 对 FBF / RMR 的完整 rolling trades，测算"大级别 SR"距离（wide_sr_dist_atr）
     是否为盈利锚点。
  2. 原型 Hub-Rebound 策略并做离线回测。
  3. 输出结构化结果到 stdout 与 JSON，供上层决策文档引用。

运行：
    python scripts/analyze_wide_sr_and_hub_rebound.py \
        --fbf-trades /tmp/fbf_trades.parquet \
        --rmr-trades /tmp/rmr_trades_v2.parquet \
        --feature-store feature_store/features_rmr_120T_e4cc44a22b \
        --out /tmp/wide_sr_hub_rebound_report.json
"""

from __future__ import annotations
import argparse
import glob
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]

# ---------------------------------------------------------------------------
# 1. 加载完整 bar 特征（拼接所有月份）
# ---------------------------------------------------------------------------


def load_symbol_features(store: str, symbol: str) -> pd.DataFrame:
    files = sorted(glob.glob(f"{store}/{symbol}/120T/*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


# ---------------------------------------------------------------------------
# 2. 构造 Wide SR 特征
# ---------------------------------------------------------------------------


def add_wide_sr(
    df: pd.DataFrame, wide_window: int = 240, anchor_shift: int = 12
) -> pd.DataFrame:
    """
    wide_window bars (~20 日 @2h) 为"大级别"尺度；anchor_shift 避免当前 bar 自我包含。

    产出：
      wide_sr_high: 过去 wide_window bar 的最高点（排除最近 anchor_shift 根）
      wide_sr_low : 同上的最低点
      wide_sr_dist_atr: min(|close - wide_sr_high|, |close - wide_sr_low|) / ATR
      wide_sr_side: +1 = 靠近上沿; -1 = 靠近下沿
      wide_range_width_atr: (wide_sr_high - wide_sr_low) / ATR
    """
    out = df.copy()
    high = (
        df["high"]
        .rolling(wide_window, min_periods=wide_window // 2)
        .max()
        .shift(anchor_shift)
    )
    low = (
        df["low"]
        .rolling(wide_window, min_periods=wide_window // 2)
        .min()
        .shift(anchor_shift)
    )
    atr = df["atr"].replace(0.0, np.nan)
    close = df["close"]
    d_high = (close - high).abs() / atr
    d_low = (close - low).abs() / atr
    dist = np.minimum(d_high, d_low)
    side = np.where(d_high <= d_low, +1.0, -1.0)  # +1 upper, -1 lower
    out["wide_sr_high"] = high
    out["wide_sr_low"] = low
    out["wide_sr_dist_atr"] = dist
    out["wide_sr_side"] = side
    out["wide_range_width_atr"] = (high - low) / atr
    return out


# ---------------------------------------------------------------------------
# 3. Trade 入场时刻特征 join
# ---------------------------------------------------------------------------


def enrich_trades(
    trades: pd.DataFrame, feat_by_sym: Dict[str, pd.DataFrame], cols: List[str]
) -> pd.DataFrame:
    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"]).dt.tz_localize(None)
    pieces = []
    for sym, g in trades.groupby("symbol"):
        fdf = feat_by_sym.get(sym)
        if fdf is None or fdf.empty:
            g = g.copy()
            for c in cols:
                g[c] = np.nan
            pieces.append(g)
            continue
        fdf = fdf.copy()
        fdf.index = (
            pd.to_datetime(fdf.index).tz_localize(None)
            if getattr(fdf.index, "tz", None) is not None
            else pd.to_datetime(fdf.index)
        )
        fdf = fdf.sort_index()
        avail = [c for c in cols if c in fdf.columns]
        missing = [c for c in cols if c not in fdf.columns]
        merged = pd.merge_asof(
            g.sort_values("entry_time"),
            fdf[avail]
            .reset_index()
            .rename(columns={fdf.index.name or "timestamp": "entry_time"}),
            on="entry_time",
            direction="backward",
            tolerance=pd.Timedelta("2H"),
        )
        for c in missing:
            merged[c] = np.nan
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True).sort_values("entry_time")


# ---------------------------------------------------------------------------
# 4. 按阈值切分报告
# ---------------------------------------------------------------------------


def threshold_report(
    trades: pd.DataFrame, col: str, thresholds: List[float], tag: str
) -> List[dict]:
    rows = []
    valid = trades[trades[col].notna()].copy()
    if valid.empty:
        return rows
    for t in thresholds:
        near = valid[valid[col] <= t]
        far = valid[valid[col] > t]
        rows.append(
            {
                "strategy": tag,
                "threshold_atr": t,
                "near_count": int(len(near)),
                "near_totalR": float(near["pnl_r"].sum()),
                "near_meanR": (
                    float(near["pnl_r"].mean()) if len(near) else float("nan")
                ),
                "near_win": (
                    float((near["pnl_r"] > 0).mean()) if len(near) else float("nan")
                ),
                "far_count": int(len(far)),
                "far_totalR": float(far["pnl_r"].sum()),
                "far_meanR": float(far["pnl_r"].mean()) if len(far) else float("nan"),
                "far_win": (
                    float((far["pnl_r"] > 0).mean()) if len(far) else float("nan")
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 5. Hub-Rebound 状态机（离线纯事件版）
# ---------------------------------------------------------------------------


@dataclass
class HubRebParams:
    hub_min_bars: int = 20  # ~1.7 天 @ 2h（放宽自 36）
    hub_compress_frac: float = 0.75  # 窗口内需 >= 75% 的 bar 是压缩 bar
    bb_width_pct_max: float = 0.40
    trend_r2_max: float = 0.40
    break_buffer_atr: float = 0.30
    break_min_magnitude_atr: float = 0.5
    break_max_bars_after_hub: int = 6
    rebound_buffer_atr: float = 0.10
    rebound_window_bars: int = 10
    sl_buffer_atr: float = 0.30
    target_r: float = 2.0
    max_holding_bars: int = 120


def simulate_hub_rebound(df: pd.DataFrame, p: HubRebParams) -> pd.DataFrame:
    """
    在单品种单时间序列上模拟 hub-rebound long 策略。
    返回 trade 清单（entry_time, entry_price, exit_price, exit_time, pnl_r, reason）。
    """
    need = [
        "open",
        "high",
        "low",
        "close",
        "atr",
        "bb_width_normalized_pct",
        "trend_r2_20",
    ]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan
    idx = df.index
    O = df["open"].to_numpy()
    H = df["high"].to_numpy()
    L = df["low"].to_numpy()
    C = df["close"].to_numpy()
    A = df["atr"].to_numpy()
    BBp = df["bb_width_normalized_pct"].to_numpy()
    R2 = df["trend_r2_20"].to_numpy()
    n = len(df)

    # 中枢条件: bb_width 压缩 + 非趋势
    compressed_bar = (BBp <= p.bb_width_pct_max) & (R2 <= p.trend_r2_max)
    # 滑动窗口内压缩占比（允许少量非压缩噪声）
    rolling_frac = (
        pd.Series(compressed_bar.astype(float))
        .rolling(p.hub_min_bars, min_periods=p.hub_min_bars)
        .mean()
        .to_numpy()
    )
    hub_window_ok = rolling_frac >= p.hub_compress_frac

    trades: List[dict] = []
    state = "IDLE"
    hub_start = -1
    hub_end = -1
    hub_high = np.nan
    hub_low = np.nan
    break_bar = -1
    break_low = np.nan

    # 锁 in trade state：禁止重叠
    in_trade = False
    entry_i = -1
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    target_price = 0.0

    i = 0
    while i < n:
        if in_trade:
            # 检查 stop / target / time stop
            hit_sl = L[i] <= stop_price
            hit_tp = H[i] >= target_price
            over_time = (i - entry_i) >= p.max_holding_bars
            if hit_sl and hit_tp:
                # 保守：视为同bar先SL再TP -> 按 SL 计
                pnl_r = -1.0
                reason = "sl_and_tp_samebar_sl"
                exit_i = i
                exit_price = stop_price
            elif hit_sl:
                pnl_r = -1.0
                reason = "sl"
                exit_i = i
                exit_price = stop_price
            elif hit_tp:
                pnl_r = (target_price - entry_price) / max(
                    entry_price - stop_price, 1e-9
                )
                reason = "tp"
                exit_i = i
                exit_price = target_price
            elif over_time:
                pnl_r = (C[i] - entry_price) / max(entry_price - stop_price, 1e-9)
                reason = "time_stop"
                exit_i = i
                exit_price = C[i]
            else:
                i += 1
                continue
            trades.append(
                {
                    "entry_time": idx[entry_i],
                    "exit_time": idx[exit_i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_r": pnl_r,
                    "exit_reason": reason,
                    "bars_held": exit_i - entry_i,
                }
            )
            in_trade = False
            state = "IDLE"
            i = exit_i + 1
            continue

        # 状态机推进
        if state == "IDLE":
            if hub_window_ok[i]:
                hub_start = i - p.hub_min_bars + 1
                hub_end = i
                hub_high = np.nanmax(H[hub_start : hub_end + 1])
                hub_low = np.nanmin(L[hub_start : hub_end + 1])
                state = "HUB_READY"
        elif state == "HUB_READY":
            # 延续：中枢窗口仍然成立 -> 刷新边界
            if hub_window_ok[i]:
                hub_end = i
                hub_high = max(hub_high, H[i])
                hub_low = min(hub_low, L[i])
            # 检查破位
            atr_i = A[i] if A[i] > 0 else np.nan
            if (
                np.isfinite(atr_i)
                and (C[i] < hub_low - p.break_buffer_atr * atr_i)
                and ((hub_low - C[i]) >= p.break_min_magnitude_atr * atr_i)
            ):
                state = "BROKEN"
                break_bar = i
                break_low = L[i]
        elif state == "BROKEN":
            break_low = min(break_low, L[i])
            if (i - break_bar) > p.rebound_window_bars:
                state = "IDLE"
                i += 1
                continue
            atr_i = A[i] if A[i] > 0 else np.nan
            if (
                np.isfinite(atr_i)
                and C[i] > hub_low + p.rebound_buffer_atr * atr_i
                and L[i] > break_low
            ):
                entry_i = i
                entry_price = C[i]
                entry_atr = atr_i
                stop_price = break_low - p.sl_buffer_atr * atr_i
                min_stop_dist = 0.3 * atr_i
                if entry_price - stop_price < min_stop_dist:
                    stop_price = entry_price - 0.8 * atr_i
                target_price = max(
                    hub_high, entry_price + p.target_r * (entry_price - stop_price)
                )
                in_trade = True
                state = "IN_TRADE"
                i += 1
                continue
            if np.isfinite(atr_i) and C[i] < break_low - 0.5 * atr_i:
                state = "IDLE"
        i += 1

    if trades:
        tdf = pd.DataFrame(trades)
    else:
        tdf = pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "pnl_r",
                "exit_reason",
                "bars_held",
            ]
        )
    return tdf


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbf-trades", default="/tmp/fbf_trades.parquet")
    ap.add_argument("--rmr-trades", default="/tmp/rmr_trades_v2.parquet")
    ap.add_argument(
        "--feature-store", default="feature_store/features_rmr_120T_e4cc44a22b"
    )
    ap.add_argument("--out", default="/tmp/wide_sr_hub_rebound_report.json")
    ap.add_argument("--wide-window", type=int, default=240)
    args = ap.parse_args()

    print("=" * 70)
    print("加载各品种 bar 特征 ...")
    feat_by_sym = {}
    for sym in SYMBOLS:
        df = load_symbol_features(args.feature_store, sym)
        if df.empty:
            print(f"  [skip] {sym}: 无数据")
            continue
        df = add_wide_sr(df, wide_window=args.wide_window, anchor_shift=12)
        feat_by_sym[sym] = df
        print(f"  {sym}: {df.shape[0]} bars, {df.index.min()} -> {df.index.max()}")

    enrich_cols = [
        "wide_sr_dist_atr",
        "wide_sr_side",
        "wide_range_width_atr",
        "wide_sr_high",
        "wide_sr_low",
        "dist_to_nearest_sr",
        "sr_strength_max",
        "bb_width_normalized_pct",
        "trend_r2_20",
    ]

    report: Dict[str, object] = {
        "config": {
            "wide_window_bars": args.wide_window,
            "anchor_shift_bars": 12,
            "scale_approx_days": args.wide_window * 2 / 24.0,
        },
    }

    # ---- FBF ----
    print("\n" + "=" * 70)
    print("FBF 分析")
    fbf = pd.read_parquet(args.fbf_trades)
    fbf = enrich_trades(fbf, feat_by_sym, enrich_cols)
    print(f"FBF 成功 enrich: {fbf['wide_sr_dist_atr'].notna().sum()}/{len(fbf)}")
    fbf_rep = {
        "total_trades": int(len(fbf)),
        "total_R": float(fbf["pnl_r"].sum()),
        "mean_R": float(fbf["pnl_r"].mean()),
        "winrate": float((fbf["pnl_r"] > 0).mean()),
        "wide_sr_threshold_sweep": threshold_report(
            fbf, "wide_sr_dist_atr", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], "FBF"
        ),
        "narrow_sr_threshold_sweep": threshold_report(
            # dist_to_nearest_sr 是相对百分比；换算 ATR 倍数 = (dist_pct * price) / atr
            (
                lambda df: df.assign(
                    narrow_dist_atr=(df["dist_to_nearest_sr"].abs() * df["entry_price"])
                    / df["atr"].replace(0, np.nan)
                )
            )(fbf),
            "narrow_dist_atr",
            [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
            "FBF",
        ),
    }
    report["fbf"] = fbf_rep
    print(json.dumps(fbf_rep["wide_sr_threshold_sweep"], indent=2, default=str))

    # ---- RMR ----
    print("\n" + "=" * 70)
    print("RMR 分析")
    rmr = pd.read_parquet(args.rmr_trades)
    rmr = enrich_trades(rmr, feat_by_sym, enrich_cols)
    print(f"RMR 成功 enrich: {rmr['wide_sr_dist_atr'].notna().sum()}/{len(rmr)}")
    rmr_rep = {
        "total_trades": int(len(rmr)),
        "total_R": float(rmr["pnl_r"].sum()),
        "mean_R": float(rmr["pnl_r"].mean()),
        "winrate": float((rmr["pnl_r"] > 0).mean()),
        "wide_sr_threshold_sweep": threshold_report(
            rmr, "wide_sr_dist_atr", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], "RMR"
        ),
        "narrow_sr_threshold_sweep": threshold_report(
            (
                lambda df: df.assign(
                    narrow_dist_atr=(df["dist_to_nearest_sr"].abs() * df["entry_price"])
                    / df["atr"].replace(0, np.nan)
                )
            )(rmr),
            "narrow_dist_atr",
            [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
            "RMR",
        ),
    }
    report["rmr"] = rmr_rep
    print(json.dumps(rmr_rep["wide_sr_threshold_sweep"], indent=2, default=str))

    # ---- Hub-Rebound ----
    print("\n" + "=" * 70)
    print("Hub-Rebound 策略模拟（多参数扫描）")

    def run_hr(label: str, p: HubRebParams) -> dict:
        print(f"\n--- {label} --- 参数:", asdict(p))
        all_trades = []
        for sym, fdf in feat_by_sym.items():
            sub = fdf.loc[
                (fdf.index >= pd.Timestamp("2023-09-01"))
                & (fdf.index <= pd.Timestamp("2024-12-31 23:59"))
            ]
            if sub.empty or "bb_width_normalized_pct" not in sub.columns:
                continue
            tdf = simulate_hub_rebound(sub, p)
            if len(tdf):
                tdf["symbol"] = sym
                all_trades.append(tdf)
                print(
                    f"  {sym}: {len(tdf):3d} trades, totalR={tdf['pnl_r'].sum():+6.2f}, "
                    f"meanR={tdf['pnl_r'].mean():+.3f}, win={(tdf['pnl_r']>0).mean():.3f}"
                )
            else:
                print(f"  {sym}:   0 trades")
        if all_trades:
            hr = pd.concat(all_trades, ignore_index=True)
        else:
            hr = pd.DataFrame()
        rep = {
            "label": label,
            "params": asdict(p),
            "total_trades": int(len(hr)),
            "total_R": float(hr["pnl_r"].sum()) if len(hr) else 0.0,
            "mean_R": float(hr["pnl_r"].mean()) if len(hr) else float("nan"),
            "winrate": float((hr["pnl_r"] > 0).mean()) if len(hr) else float("nan"),
            "by_symbol": {},
            "exit_reason_counts": {},
        }
        if len(hr):
            for sym, g in hr.groupby("symbol"):
                rep["by_symbol"][sym] = {
                    "n": int(len(g)),
                    "totalR": float(g["pnl_r"].sum()),
                    "meanR": float(g["pnl_r"].mean()),
                    "win": float((g["pnl_r"] > 0).mean()),
                }
            rep["exit_reason_counts"] = hr["exit_reason"].value_counts().to_dict()
        print(
            f"  => TOTAL: {rep['total_trades']} trades, R={rep['total_R']:+.2f}, "
            f"meanR={rep['mean_R']:+.3f}, win={rep['winrate']:.3f}"
        )
        return rep, hr

    strict_rep, hr_strict = run_hr("STRICT (贴 spec)", HubRebParams())
    # 宽松: 放松中枢长度 + 破位幅度 + 目标 R
    relaxed = HubRebParams(
        hub_min_bars=14,
        hub_compress_frac=0.7,
        bb_width_pct_max=0.50,
        trend_r2_max=0.50,
        break_min_magnitude_atr=0.35,
        rebound_window_bars=12,
        target_r=1.5,
    )
    relaxed_rep, hr_relax = run_hr("RELAXED", relaxed)
    # 激进: 几乎只要价格砸破区间短暂反弹就入场
    aggr = HubRebParams(
        hub_min_bars=10,
        hub_compress_frac=0.6,
        bb_width_pct_max=0.60,
        trend_r2_max=0.60,
        break_min_magnitude_atr=0.25,
        break_buffer_atr=0.10,
        rebound_buffer_atr=0.05,
        rebound_window_bars=14,
        target_r=1.2,
    )
    aggr_rep, hr_aggr = run_hr("AGGRESSIVE", aggr)

    report["hub_rebound_variants"] = {
        "strict": strict_rep,
        "relaxed": relaxed_rep,
        "aggressive": aggr_rep,
    }
    if len(hr_relax):
        hr_relax.to_parquet("/tmp/hub_rebound_trades_relaxed.parquet")
    if len(hr_aggr):
        hr_aggr.to_parquet("/tmp/hub_rebound_trades_aggressive.parquet")

    # 保存 RMR / FBF enriched
    fbf.to_parquet("/tmp/fbf_trades_wideSR.parquet")
    rmr.to_parquet("/tmp/rmr_trades_wideSR.parquet")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n报告已写入 {args.out}")


if __name__ == "__main__":
    main()
