#!/usr/bin/env python3
"""
ME 漏斗诊断：在指定时间窗内统计 prefilter / VWAP 位置 / bypass 信号占比。

用于定位「末段上冲完全没点」是死在 prefilter、带通上沿，还是 bypass 过弱。

示例:
  python scripts/diagnose_me_funnel_nov2024.py \\
    --parquet results/train_final_20260409_214938_rr_extreme/me/features_labeled.parquet \\
    --start 2024-11-01 --end 2024-12-31 --symbol BTCUSDT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parquet", type=Path, required=True, help="features_labeled.parquet 路径"
    )
    p.add_argument("--start", type=str, default="2024-11-01", help="含该日")
    p.add_argument("--end", type=str, default="2024-12-31", help="含该日")
    p.add_argument(
        "--symbol", type=str, default="", help="可选，如 BTCUSDT；空则全品种"
    )
    p.add_argument(
        "--accel-abs-min", type=float, default=1.0, help="重算 bypass 时的 accel 阈值"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cols = [
        "datetime",
        "symbol",
        "me_atr_pct",
        "me_cvd_alignment",
        "macro_tp_vwap_1200_position",
        "me_accel_5k",
    ]
    full = pd.read_parquet(args.parquet)
    miss = [c for c in cols if c not in full.columns]
    if miss:
        raise SystemExit(f"parquet 缺列: {miss}")
    dt = pd.to_datetime(full["datetime"], utc=True, errors="coerce")
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)
    m = (dt >= start) & (dt < end)
    if args.symbol:
        m &= full["symbol"].astype(str) == args.symbol
    d = full.loc[m, cols].copy()
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True, errors="coerce")
    if len(d) == 0:
        print("无样本：检查 parquet / 时间窗 / symbol")
        return
    d = d.reset_index(drop=True)

    atr = pd.to_numeric(d["me_atr_pct"], errors="coerce")
    cvd = pd.to_numeric(d["me_cvd_alignment"], errors="coerce")
    pv = pd.to_numeric(d["macro_tp_vwap_1200_position"], errors="coerce")
    accel = pd.to_numeric(d["me_accel_5k"], errors="coerce")

    n = len(d)
    ok = atr.notna() & cvd.notna()

    def rate(mask: pd.Series) -> float:
        return float(mask.sum()) / float(ok.sum()) if ok.sum() else 0.0

    p_lo = ok & (atr >= 0.65)
    p_hi_old = ok & (atr <= 0.99)
    p_hi_new = ok & (atr <= 1.0)
    cvd_old = ok & (cvd >= 0.55)
    cvd_new = ok & (cvd >= 0.51)
    pf_old = p_lo & p_hi_old & cvd_old
    pf_new = p_lo & p_hi_new & cvd_new

    if "me_vwap_structure_bypass_direction" in full.columns:
        # 用与 d 相同的布尔掩码对齐，避免 duplicate index 时 loc[d.index] 行数膨胀
        bypass = pd.to_numeric(
            full.loc[m, "me_vwap_structure_bypass_direction"], errors="coerce"
        ).reset_index(drop=True)
    else:
        from src.features.time_series.momentum_expansion_features import (
            compute_me_vwap_structure_bypass_direction_from_series,
        )

        bypass_df = compute_me_vwap_structure_bypass_direction_from_series(
            me_accel_5k=accel,
            macro_tp_vwap_1200_position=pv,
            accel_abs_min=float(args.accel_abs_min),
        )
        bypass = bypass_df["me_vwap_structure_bypass_direction"].reset_index(drop=True)

    bypass = pd.to_numeric(bypass, errors="coerce")
    if len(bypass) != len(d):
        raise SystemExit(
            f"bypass 长度 {len(bypass)} 与 窗口行数 {len(d)} 不一致（检查 parquet 索引）"
        )
    bypass = pd.Series(bypass.values, index=d.index)
    nz_bypass = ok & bypass.notna() & (bypass != 0)

    pv_ok = ok & pv.notna()
    p_ge_1 = pv_ok & (pv >= 1.0)
    p_le_0_long_dead = pv_ok & (pv <= 0)

    print("=== ME funnel diagnose ===")
    print(f"parquet: {args.parquet}")
    print(
        f"window(UTC): {start.date()} .. {args.end}  rows={n}  symbol={args.symbol or '*'}"
    )
    print()
    print("--- Prefilter (AND) pass rate among finite atr+cvd ---")
    print(f"  atr>=0.65           : {rate(p_lo):.1%}")
    print(f"  atr<=0.99 (old cap) : {rate(p_hi_old):.1%}")
    print(f"  atr<=1.00 (new cap) : {rate(p_hi_new):.1%}")
    print(f"  cvd>=0.55 (old)     : {rate(cvd_old):.1%}")
    print(f"  cvd>=0.51 (new)     : {rate(cvd_new):.1%}")
    print(f"  ALL old (0.99/0.55) : {rate(pf_old):.1%}")
    print(f"  ALL new (1.0/0.51)  : {rate(pf_new):.1%}")
    print()
    print("--- macro_tp_vwap_1200_position (finite rows) ---")
    print(f"  p >= 1.0 (sat/over): {rate(p_ge_1):.1%}")
    print(f"  p <= 0   (long dead): {rate(p_le_0_long_dead):.1%}")
    print()
    print("--- Bypass (accel_abs_min=%.2f) ---" % float(args.accel_abs_min))
    print(f"  bypass != 0        : {rate(nz_bypass):.1%}")


if __name__ == "__main__":
    main()
