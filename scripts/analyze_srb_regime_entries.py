"""
SRB regime × 入场类型诊断
=========================
目的：验证两个观察/假设
  #2 "很多 trade 开仓在波动区（ranging），而 SRB 语义是突破→追势，应该避免"
  #3 "波动区开仓后反而加仓多；趋势区加仓少、退出快"

读取 `reports/srb_break_level_attribution_v2_alltrades_trades.parquet`（内含
每笔 trade 在 entry_time 的 bar 特征）。给每笔 trade 打 regime 标签，分层统计：

Regime 定义（入场 bar）：
    RANGING  : trend_r2_20 <= r2_low  AND bb_width_normalized_pct <= bb_low
    TRENDING : trend_r2_20 >= r2_high
    MIXED    : 其他
默认 r2_low=0.30, r2_high=0.50, bb_low=0.50

输出：
  - Regime × (first_entry, add_position) 计数矩阵
  - 每个 regime 下首单的 n / meanR / winrate / mean_bars_held / exit_reasons
  - "每笔首单平均 follow-on add_position 数量" 按 regime 拆解
  - trend_r2_20 / bb_width_normalized_pct 阈值扫描（首单 meanR）

用法：
    python scripts/analyze_srb_regime_entries.py \
        --trades reports/srb_break_level_attribution_v2_alltrades_trades.parquet \
        --out reports/srb_regime_entry_analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Regime 标注
# ---------------------------------------------------------------------------


def tag_regime(
    df: pd.DataFrame,
    r2_low: float,
    r2_high: float,
    bb_low: float,
) -> pd.DataFrame:
    out = df.copy()
    r2 = out["f_trend_r2_20"]
    bb = out["f_bb_width_normalized_pct"]
    ranging = (r2 <= r2_low) & (bb <= bb_low) & r2.notna() & bb.notna()
    trending = (r2 >= r2_high) & r2.notna()
    regime = np.where(ranging, "RANGING", np.where(trending, "TRENDING", "MIXED"))
    out["regime"] = regime
    out.loc[r2.isna(), "regime"] = "UNKNOWN"
    return out


# ---------------------------------------------------------------------------
# 计数 / 统计
# ---------------------------------------------------------------------------


def regime_entry_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["entry_type"] = np.where(
        df["is_add_position"].astype(bool), "add_pos", "first_entry"
    )
    pivot = df.groupby(["regime", "entry_type"]).size().unstack(fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    if "add_pos" in pivot.columns and "first_entry" in pivot.columns:
        pivot["add_pct"] = pivot["add_pos"] / pivot["total"]
        pivot["adds_per_first_entry"] = pivot["add_pos"] / pivot["first_entry"].replace(
            0, np.nan
        )
    return pivot


def summarize_first_entries(df: pd.DataFrame) -> Dict[str, dict]:
    df = df[~df["is_add_position"].astype(bool) & ~df["is_reverse"].astype(bool)].copy()
    stats = {}
    for regime, g in df.groupby("regime"):
        pnl = pd.to_numeric(g["pnl_r"], errors="coerce").dropna().to_numpy()
        if len(pnl) == 0:
            stats[regime] = {"n": 0}
            continue
        stats[regime] = {
            "n": int(len(g)),
            "totalR": float(np.nansum(pnl)),
            "meanR": float(np.nanmean(pnl)),
            "medianR": float(np.nanmedian(pnl)),
            "winrate": float((pnl > 0).mean()),
            "mean_bars_held": float(
                pd.to_numeric(g["bars_held"], errors="coerce").mean()
            ),
            "median_bars_held": float(
                pd.to_numeric(g["bars_held"], errors="coerce").median()
            ),
            "exit_reasons": g["exit_reason"].value_counts().to_dict(),
        }
    return stats


def follow_on_adds_per_first_entry(
    df: pd.DataFrame, window_bars_minutes: int = 14400
) -> Dict[str, dict]:
    """对每笔首单，数其后在同 symbol 同 side 内的 add_position 笔数。
    默认 window = 14400 分钟 (~10 天) 兜底；实际通常受 exit_time 前止损。"""
    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    first = df[~df["is_add_position"].astype(bool)].copy().reset_index(drop=True)
    adds = df[df["is_add_position"].astype(bool)].copy()

    counts_per_regime: Dict[str, List[int]] = {}
    for _, row in first.iterrows():
        sym = row["symbol"]
        side = row["side"]
        et = row["entry_time"]
        xt = (
            row["exit_time"]
            if pd.notna(row["exit_time"])
            else et + pd.Timedelta(minutes=window_bars_minutes)
        )
        mask = (
            (adds["symbol"] == sym)
            & (adds["side"] == side)
            & (adds["entry_time"] >= et)
            & (adds["entry_time"] <= xt)
        )
        n_adds = int(mask.sum())
        counts_per_regime.setdefault(row["regime"], []).append(n_adds)

    summary = {}
    for regime, lst in counts_per_regime.items():
        arr = np.asarray(lst)
        summary[regime] = {
            "n_first_entries": int(len(arr)),
            "mean_adds": float(arr.mean()) if len(arr) else float("nan"),
            "median_adds": float(np.median(arr)) if len(arr) else float("nan"),
            "pct_with_any_add": float((arr > 0).mean()) if len(arr) else float("nan"),
            "max_adds": int(arr.max()) if len(arr) else 0,
        }
    return summary


# ---------------------------------------------------------------------------
# 阈值扫描（提议 prefilter）
# ---------------------------------------------------------------------------


def threshold_sweep_r2(df: pd.DataFrame) -> List[dict]:
    df = df[~df["is_add_position"].astype(bool)].copy()
    pnl = pd.to_numeric(df["pnl_r"], errors="coerce")
    r2 = df["f_trend_r2_20"]
    rows = []
    for t in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        low = df[r2 <= t]
        high = df[r2 > t]
        rows.append(
            {
                "r2_threshold": t,
                "low_r2_n": int(len(low)),
                "low_r2_meanR": (
                    float(pd.to_numeric(low["pnl_r"], errors="coerce").mean())
                    if len(low)
                    else float("nan")
                ),
                "low_r2_win": (
                    float((pd.to_numeric(low["pnl_r"], errors="coerce") > 0).mean())
                    if len(low)
                    else float("nan")
                ),
                "high_r2_n": int(len(high)),
                "high_r2_meanR": (
                    float(pd.to_numeric(high["pnl_r"], errors="coerce").mean())
                    if len(high)
                    else float("nan")
                ),
                "high_r2_win": (
                    float((pd.to_numeric(high["pnl_r"], errors="coerce") > 0).mean())
                    if len(high)
                    else float("nan")
                ),
            }
        )
    return rows


def threshold_sweep_bb(df: pd.DataFrame) -> List[dict]:
    df = df[~df["is_add_position"].astype(bool)].copy()
    bb = df["f_bb_width_normalized_pct"]
    rows = []
    for t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        narrow = df[bb <= t]
        wide = df[bb > t]
        rows.append(
            {
                "bb_threshold": t,
                "narrow_n": int(len(narrow)),
                "narrow_meanR": (
                    float(pd.to_numeric(narrow["pnl_r"], errors="coerce").mean())
                    if len(narrow)
                    else float("nan")
                ),
                "narrow_win": (
                    float((pd.to_numeric(narrow["pnl_r"], errors="coerce") > 0).mean())
                    if len(narrow)
                    else float("nan")
                ),
                "wide_n": int(len(wide)),
                "wide_meanR": (
                    float(pd.to_numeric(wide["pnl_r"], errors="coerce").mean())
                    if len(wide)
                    else float("nan")
                ),
                "wide_win": (
                    float((pd.to_numeric(wide["pnl_r"], errors="coerce") > 0).mean())
                    if len(wide)
                    else float("nan")
                ),
            }
        )
    return rows


def joint_filter_sweep(df: pd.DataFrame) -> List[dict]:
    """(trend_r2_20 >= r2_thr) AND (bb >= bb_thr) 作为 prefilter，观察留存与过滤的 meanR。"""
    df = df[~df["is_add_position"].astype(bool)].copy()
    rows = []
    for r2_thr in [0.20, 0.30, 0.40]:
        for bb_thr in [0.30, 0.40, 0.50]:
            keep = df[
                (df["f_trend_r2_20"] >= r2_thr)
                & (df["f_bb_width_normalized_pct"] >= bb_thr)
            ]
            drop = df[
                ~(
                    (df["f_trend_r2_20"] >= r2_thr)
                    & (df["f_bb_width_normalized_pct"] >= bb_thr)
                )
            ]
            rows.append(
                {
                    "r2_thr": r2_thr,
                    "bb_thr": bb_thr,
                    "kept_n": int(len(keep)),
                    "kept_meanR": (
                        float(pd.to_numeric(keep["pnl_r"], errors="coerce").mean())
                        if len(keep)
                        else float("nan")
                    ),
                    "kept_totalR": (
                        float(pd.to_numeric(keep["pnl_r"], errors="coerce").sum())
                        if len(keep)
                        else 0.0
                    ),
                    "kept_win": (
                        float(
                            (pd.to_numeric(keep["pnl_r"], errors="coerce") > 0).mean()
                        )
                        if len(keep)
                        else float("nan")
                    ),
                    "dropped_n": int(len(drop)),
                    "dropped_meanR": (
                        float(pd.to_numeric(drop["pnl_r"], errors="coerce").mean())
                        if len(drop)
                        else float("nan")
                    ),
                    "dropped_totalR": (
                        float(pd.to_numeric(drop["pnl_r"], errors="coerce").sum())
                        if len(drop)
                        else 0.0
                    ),
                    "dropped_win": (
                        float(
                            (pd.to_numeric(drop["pnl_r"], errors="coerce") > 0).mean()
                        )
                        if len(drop)
                        else float("nan")
                    ),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trades",
        default="reports/srb_break_level_attribution_v2_alltrades_trades.parquet",
    )
    ap.add_argument("--out", default="reports/srb_regime_entry_analysis.json")
    ap.add_argument("--r2-low", type=float, default=0.30)
    ap.add_argument("--r2-high", type=float, default=0.50)
    ap.add_argument("--bb-low", type=float, default=0.50)
    args = ap.parse_args()

    print("=" * 72)
    print(f"trades : {args.trades}")
    df = pd.read_parquet(args.trades)
    print(f"loaded : {len(df)} trades")
    for c in ("is_add_position", "is_reverse"):
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    tagged = tag_regime(df, args.r2_low, args.r2_high, args.bb_low)
    print(
        "Regime distribution (all trades):", tagged["regime"].value_counts().to_dict()
    )

    # 1) 入场矩阵
    matrix = regime_entry_matrix(tagged)
    print("\n=== regime × entry_type matrix ===")
    print(matrix.to_string())

    # 2) 首单统计
    first_stats = summarize_first_entries(tagged)
    print("\n=== first-entry stats by regime ===")
    for regime in sorted(first_stats.keys()):
        s = first_stats[regime]
        if s.get("n", 0) == 0:
            print(f"  {regime:10s}: n=0")
            continue
        print(
            f"  {regime:10s}: n={s['n']:3d}  totalR={s['totalR']:+7.2f}  meanR={s['meanR']:+.3f}  "
            f"medianR={s['medianR']:+.3f}  win={s['winrate']:.3f}  barsH~{s['mean_bars_held']:.0f}  "
            f"exits={s['exit_reasons']}"
        )

    # 3) 每笔首单平均后续 add 数
    adds_summary = follow_on_adds_per_first_entry(tagged)
    print("\n=== follow-on adds per first entry by regime ===")
    for regime in sorted(adds_summary.keys()):
        s = adds_summary[regime]
        print(
            f"  {regime:10s}: n={s['n_first_entries']:3d}  mean_adds={s['mean_adds']:.2f}  "
            f"median_adds={s['median_adds']:.1f}  pct_with_add={s['pct_with_any_add']:.2f}  "
            f"max_adds={s['max_adds']}"
        )

    # 4) 阈值扫描
    r2_sweep = threshold_sweep_r2(tagged)
    print("\n=== trend_r2_20 阈值扫描（首单） ===")
    for r in r2_sweep:
        print(
            f"  r2≤{r['r2_threshold']:.2f}: n={r['low_r2_n']:3d} meanR={r['low_r2_meanR']:+.3f} win={r['low_r2_win']:.3f}"
            f"  |  r2>{r['r2_threshold']:.2f}: n={r['high_r2_n']:3d} meanR={r['high_r2_meanR']:+.3f} win={r['high_r2_win']:.3f}"
        )

    bb_sweep = threshold_sweep_bb(tagged)
    print("\n=== bb_width_normalized_pct 阈值扫描（首单） ===")
    for r in bb_sweep:
        print(
            f"  bb≤{r['bb_threshold']:.2f}: n={r['narrow_n']:3d} meanR={r['narrow_meanR']:+.3f} win={r['narrow_win']:.3f}"
            f"  |  bb>{r['bb_threshold']:.2f}: n={r['wide_n']:3d} meanR={r['wide_meanR']:+.3f} win={r['wide_win']:.3f}"
        )

    joint = joint_filter_sweep(tagged)
    print("\n=== joint filter (keep if r2≥r2_thr AND bb≥bb_thr) 首单 ===")
    for r in joint:
        print(
            f"  r2≥{r['r2_thr']:.2f} & bb≥{r['bb_thr']:.2f}: kept n={r['kept_n']:3d} meanR={r['kept_meanR']:+.3f} totalR={r['kept_totalR']:+6.2f} win={r['kept_win']:.3f}"
            f"  |  dropped n={r['dropped_n']:3d} meanR={r['dropped_meanR']:+.3f} totalR={r['dropped_totalR']:+6.2f}"
        )

    report = {
        "config": {
            "trades": args.trades,
            "r2_low": args.r2_low,
            "r2_high": args.r2_high,
            "bb_low": args.bb_low,
            "total_trades": int(len(df)),
        },
        "regime_entry_matrix": matrix.reset_index().to_dict(orient="records"),
        "first_entry_stats_by_regime": first_stats,
        "follow_on_adds_by_regime": adds_summary,
        "threshold_sweep_r2": r2_sweep,
        "threshold_sweep_bb": bb_sweep,
        "joint_filter_sweep": joint,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n报告已写入 {args.out}")


if __name__ == "__main__":
    main()
