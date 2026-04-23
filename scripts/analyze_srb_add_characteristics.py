#!/usr/bin/env python3
"""
分析 SRB 16 个月历史加仓的特征：
- 将 is_add_position=True 的 trade 与其母仓匹配（same symbol+side，entry_time 紧邻之前的 first-entry）
- 计算 bars_since_mother_entry、mother 的退出结果
- 按 "赢家加仓 vs 输家加仓" 分组统计 bars_since_mother_entry / exit_reason / 持仓时长

输出：
- 总体分布表
- 按 symbol / side 的输赢 add 对比
- 建议的过滤阈值（bars_since 分位点）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BAR_MINUTES = 120  # primary 2H


def _load_trades(root: Path) -> pd.DataFrame:
    frames = []
    for d in sorted(root.glob("*")):
        f = d / "trades.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["month"] = d.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True)
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True)
    return out


def _match_mothers(df: pd.DataFrame) -> pd.DataFrame:
    """为每条 add_position 找到对应母仓 entry_time。

    规则：按 (symbol, side) 分组，时间升序。每个 first-entry (is_add=False) 开启一个母仓；
    后续 add 直到下一个 first-entry 或与 first-entry 持仓重叠结束之后，都挂在它下面。
    """
    out_rows = []
    for (sym, side), grp in df.groupby(["symbol", "side"]):
        g = grp.sort_values("entry_time").reset_index(drop=True)
        current_mother_entry = None
        current_mother_exit = None
        current_mother_pnl = None
        current_mother_reason = None
        for _, row in g.iterrows():
            is_add = bool(row.get("is_add_position", False))
            if not is_add:
                current_mother_entry = row["entry_time"]
                current_mother_exit = row["exit_time"]
                current_mother_pnl = row["pnl_r"]
                current_mother_reason = row["exit_reason"]
            else:
                if current_mother_entry is None:
                    bars_since = float("nan")
                    mother_pnl = float("nan")
                    mother_reason = ""
                else:
                    bars_since = (
                        (row["entry_time"] - current_mother_entry).total_seconds()
                        / 60.0
                        / BAR_MINUTES
                    )
                    mother_pnl = current_mother_pnl
                    mother_reason = current_mother_reason
                out_rows.append(
                    {
                        "month": row["month"],
                        "symbol": sym,
                        "side": side,
                        "add_entry": row["entry_time"],
                        "add_exit": row["exit_time"],
                        "add_pnl_r": row["pnl_r"],
                        "add_exit_reason": row["exit_reason"],
                        "add_bars_held": row["bars_held"],
                        "bars_since_mother_entry": bars_since,
                        "mother_pnl_r": mother_pnl,
                        "mother_exit_reason": mother_reason,
                    }
                )
    return pd.DataFrame(out_rows)


def _summary(adds: pd.DataFrame) -> None:
    print(f"\n=== 全部 add_position 统计（n={len(adds)}）===")
    print(f"总 add_R: {adds['add_pnl_r'].sum():.2f}")
    print(f"平均 add_R: {adds['add_pnl_r'].mean():.3f}")
    print(f"胜率: {(adds['add_pnl_r'] > 0).mean():.2%}")

    winners = adds[adds["add_pnl_r"] > 0]
    losers = adds[adds["add_pnl_r"] <= 0]

    print("\n=== 赢家 add vs 输家 add ===")
    print(f"{'col':30} {'winner':>10} {'loser':>10}")
    for col in [
        "bars_since_mother_entry",
        "add_bars_held",
        "mother_pnl_r",
    ]:
        w = winners[col].median() if len(winners) else float("nan")
        l = losers[col].median() if len(losers) else float("nan")
        print(f"{col:30} {w:>10.2f} {l:>10.2f}")

    print(f"{'count':30} {len(winners):>10d} {len(losers):>10d}")
    print(
        f"{'sum_R':30} {winners['add_pnl_r'].sum():>10.2f} {losers['add_pnl_r'].sum():>10.2f}"
    )
    print(
        f"{'mean_R':30} {winners['add_pnl_r'].mean():>10.2f} {losers['add_pnl_r'].mean():>10.2f}"
    )

    print("\n=== bars_since_mother_entry 分位点（输家）===")
    if len(losers):
        for q in [0.25, 0.5, 0.75, 0.9]:
            v = losers["bars_since_mother_entry"].quantile(q)
            print(f"  q{int(q*100):02d}: {v:.1f} bars")

    print("\n=== 按 bars_since 分桶：add_R 分布 ===")
    bins = [-0.01, 3, 6, 12, 24, 48, 96, 240, 1e9]
    labels = ["0-3", "3-6", "6-12", "12-24", "24-48", "48-96", "96-240", ">240"]
    adds["bucket"] = pd.cut(adds["bars_since_mother_entry"], bins=bins, labels=labels)
    grouped = (
        adds.groupby("bucket", observed=True)
        .agg(
            n=("add_pnl_r", "size"),
            sum_R=("add_pnl_r", "sum"),
            mean_R=("add_pnl_r", "mean"),
        )
        .round(2)
    )
    print(grouped.to_string())

    print("\n=== 按 symbol 汇总（输家 add）===")
    if len(losers):
        per = (
            losers.groupby("symbol")
            .agg(
                n=("add_pnl_r", "size"),
                sum_R=("add_pnl_r", "sum"),
                mean_R=("add_pnl_r", "mean"),
            )
            .round(2)
            .sort_values("sum_R")
        )
        print(per.to_string())

    # losing adds with short bars_since: smoking gun for ranging-area stacked adds
    print("\n=== 短 bars_since（<12 bar ≈ 24h） + 输家：嫌疑犯 ===")
    suspect = losers[losers["bars_since_mother_entry"] < 12]
    if len(suspect):
        agg = (
            suspect.groupby(["month", "symbol"])
            .agg(
                n=("add_pnl_r", "size"),
                sum_R=("add_pnl_r", "sum"),
                median_bars=("bars_since_mother_entry", "median"),
            )
            .round(2)
        )
        print(agg.to_string())
        print(f"\n总计：{len(suspect)} 笔 / {suspect['add_pnl_r'].sum():.2f} R")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="reports/srb_fast_ab_e1_e2_full16m/baseline",
    )
    args = p.parse_args()
    df = _load_trades(Path(args.root))
    if df.empty:
        print("no trades")
        return
    print(f"total trades: {len(df)}, adds: {df['is_add_position'].sum()}")
    adds = _match_mothers(df)
    _summary(adds)


if __name__ == "__main__":
    main()
