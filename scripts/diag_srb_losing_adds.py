#!/usr/bin/env python3
"""
深挖 SRB 16 个月输家 adds 的具体特征。

对 XRP / SOL / BTC / ETH 的每一笔输家 add：
  - add 当时的 mother 持仓已有 bars
  - mother 的最终 exit_reason / pnl_r
  - add 入场后的持仓 bars
  - add 的 exit_reason
  - 推断母仓 "MFE→current 回撤比"（估计：取 |add.entry_price - mother.entry_price| / risk 作为代理）

目标：找出"如果当时加了某个 gate，这 N 笔 loser 是否能被拦掉、同时不误伤 winner"。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BAR_MINUTES = 120


def _load(root: Path) -> pd.DataFrame:
    frames = []
    for d in sorted(root.glob("*")):
        f = d / "trades.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["month"] = d.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True)
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True)
    return out


def _pair_mothers(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sym, side), grp in df.groupby(["symbol", "side"]):
        g = grp.sort_values("entry_time").reset_index(drop=True)
        current = None
        for _, row in g.iterrows():
            if not bool(row.get("is_add_position", False)):
                current = row.to_dict()
            else:
                if current is None:
                    continue
                bars_since = (
                    (row["entry_time"] - current["entry_time"]).total_seconds()
                    / 60.0
                    / BAR_MINUTES
                )
                # Price progression at add time relative to mother entry
                is_long = side in ("LONG", "BUY")
                price_delta_r = (
                    (row["entry_price"] - current["entry_price"])
                    if is_long
                    else (current["entry_price"] - row["entry_price"])
                )
                # use mother's atr as proxy for risk unit
                risk_proxy = (
                    max(current.get("atr", 1.0) or 1.0, 1e-9) * 6.0
                )  # initial_r=6
                add_current_r = price_delta_r / risk_proxy
                rows.append(
                    {
                        "month": row["month"],
                        "symbol": sym,
                        "side": side,
                        "mother_entry": current["entry_time"],
                        "mother_exit": current["exit_time"],
                        "mother_exit_reason": current["exit_reason"],
                        "mother_pnl_r": current["pnl_r"],
                        "mother_bars_held": current["bars_held"],
                        "add_entry": row["entry_time"],
                        "add_exit": row["exit_time"],
                        "add_pnl_r": row["pnl_r"],
                        "add_exit_reason": row["exit_reason"],
                        "add_bars_held": row["bars_held"],
                        "bars_since_mother": bars_since,
                        "add_est_current_r_at_entry": add_current_r,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="reports/srb_fast_ab_e1_e2_full16m/baseline")
    args = p.parse_args()
    df = _load(Path(args.root))
    adds = _pair_mothers(df)
    losers = adds[adds["add_pnl_r"] < 0].copy()
    winners = adds[adds["add_pnl_r"] > 0].copy()

    print(f"total adds: {len(adds)}, losers: {len(losers)}, winners: {len(winners)}")

    print("\n=== 全部输家 adds 明细（按 add_pnl_r 升序）===")
    cols = [
        "month",
        "symbol",
        "side",
        "mother_pnl_r",
        "mother_exit_reason",
        "bars_since_mother",
        "add_est_current_r_at_entry",
        "add_pnl_r",
        "add_exit_reason",
    ]
    print(losers.sort_values("add_pnl_r")[cols].to_string(index=False))

    print("\n=== 输家 add 的 mother_exit_reason 分布 ===")
    print(losers["mother_exit_reason"].value_counts().to_string())

    print("\n=== 赢家 add 的 mother_exit_reason 分布 ===")
    print(winners["mother_exit_reason"].value_counts().to_string())

    # 构造"假设 filter"：比如 add_est_current_r_at_entry < 0.5 拒绝
    print("\n=== 假设 filter：add_est_current_r_at_entry < X 拒绝 ===")
    for thr in [0.2, 0.3, 0.5, 0.7]:
        would_reject = adds[adds["add_est_current_r_at_entry"] < thr]
        rej_w = would_reject[would_reject["add_pnl_r"] > 0]
        rej_l = would_reject[would_reject["add_pnl_r"] < 0]
        print(
            f"  thr={thr:.2f}: reject n={len(would_reject)}, "
            f"winner_lost={len(rej_w)}({rej_w['add_pnl_r'].sum():.1f}R), "
            f"loser_saved={len(rej_l)}({rej_l['add_pnl_r'].sum():.1f}R), "
            f"net Δ={-(rej_w['add_pnl_r'].sum() + rej_l['add_pnl_r'].sum()):+.2f}R"
        )

    print("\n=== 假设 filter：bars_since_mother > X 拒绝 ===")
    for thr in [24, 36, 48, 72]:
        would_reject = adds[adds["bars_since_mother"] > thr]
        rej_w = would_reject[would_reject["add_pnl_r"] > 0]
        rej_l = would_reject[would_reject["add_pnl_r"] < 0]
        print(
            f"  thr={thr}: reject n={len(would_reject)}, "
            f"winner_lost={len(rej_w)}({rej_w['add_pnl_r'].sum():.1f}R), "
            f"loser_saved={len(rej_l)}({rej_l['add_pnl_r'].sum():.1f}R), "
            f"net Δ={-(rej_w['add_pnl_r'].sum() + rej_l['add_pnl_r'].sum()):+.2f}R"
        )


if __name__ == "__main__":
    main()
