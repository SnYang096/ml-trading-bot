#!/usr/bin/env python3
"""
SRB entry L3-breach 分组诊断（响应 2026-04-23 "SRB prefilter 应不应该加 '站在 L3 同向外' 硬门"）。

将 results/reports/srb_fast_ab_e1_e2_full16m/baseline 的 16 个月所有首仓 trade
（is_add_position=False）按以下 3 维重新归组：

  - breach   : above / below / inside
      * above  : close > wide_sr_upper_px
      * below  : close < wide_sr_lower_px
      * inside : 既不 above 也不 below
  - direction: LONG / SHORT
  - 语义对齐 : "breach-aligned" = (LONG & above) 或 (SHORT & below)
              "breach-wrong"   = (LONG & below) 或 (SHORT & above)
              "inside"         = breach==inside（不论 direction）

输出：
  - 总体 3 x 2 组 meanR / sum R / n
  - "站在同向外" vs "站在同向内" vs "站在反向外" 的 R 分布
  - 附加：fer_ols_pos 的极端区间 (<0.10 / >0.90 / 0.10-0.90) 做 FBF 对照
  - 诊断 "硬门" 每个候选阈值下会留下多少笔 + 这些 trade 的 meanR 是否显著优于全集

对比的是"原诊断 SRB_break_level_attribution_20260422.md"里的 L1+L3 confluence
定义（wide_sr_dist_atr ≤ 1.5 且 wide_sr_side 对齐方向），以确认两者不等价。

数据需求：
  - results/reports/srb_fast_ab_e1_e2_full16m/baseline/<month>/trades.csv
  - feature_store/features_srb_120T_5643a66b47/<SYM>/120T/<month>.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_STORE = "feature_store/features_srb_120T_5643a66b47"


def _load_trades(root: Path) -> pd.DataFrame:
    frames = []
    for d in sorted(root.glob("*")):
        f = d / "trades.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True)
    return out


def _load_features(symbol: str) -> pd.DataFrame:
    store = Path(FEATURE_STORE) / symbol / "120T"
    parts = sorted(store.glob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in parts]
    df = pd.concat(frames, axis=0).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    keep = [
        c
        for c in [
            "wide_sr_upper_px",
            "wide_sr_lower_px",
            "wide_sr_dist_atr",
            "wide_sr_side",
            "fer_ols_pos",
            "close",
        ]
        if c in df.columns
    ]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[keep]


def _annotate(trades: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for sym, grp in trades.groupby("symbol"):
        feats = _load_features(sym)
        if feats.empty:
            continue
        # asof match：取 entry_time 对应 bar
        g = grp.sort_values("entry_time")
        joined = pd.merge_asof(
            g, feats, left_on="entry_time", right_index=True, direction="backward"
        )
        out_rows.append(joined)
    if not out_rows:
        return pd.DataFrame()
    df = pd.concat(out_rows, ignore_index=True)
    # compute breach
    close = df["entry_price"].astype(float)  # 入场价作为 close 代理
    up = df["wide_sr_upper_px"].astype(float)
    lo = df["wide_sr_lower_px"].astype(float)
    breach = np.where(close > up, "above", np.where(close < lo, "below", "inside"))
    df["breach"] = breach
    df["is_long"] = df["side"].isin(["LONG", "BUY"])
    df["breach_aligned"] = ((df["breach"] == "above") & df["is_long"]) | (
        (df["breach"] == "below") & ~df["is_long"]
    )
    df["breach_wrong"] = ((df["breach"] == "above") & ~df["is_long"]) | (
        (df["breach"] == "below") & df["is_long"]
    )
    return df


def _summarize(df: pd.DataFrame) -> None:
    firsts = df[~df["is_add_position"].astype(bool)].copy()
    print(f"\nfirst-entry trades: n={len(firsts)}, totR={firsts['pnl_r'].sum():.2f}")

    print("\n=== by breach × direction ===")
    grp = (
        firsts.groupby(["breach", "is_long"])
        .agg(
            n=("pnl_r", "size"),
            sumR=("pnl_r", "sum"),
            meanR=("pnl_r", "mean"),
            win=("pnl_r", lambda s: (s > 0).mean()),
        )
        .round(3)
    )
    print(grp.to_string())

    print("\n=== aligned / wrong / inside ===")
    cats = []
    for label, mask in [
        ("breach_aligned", firsts["breach_aligned"]),
        ("breach_wrong", firsts["breach_wrong"]),
        ("inside", firsts["breach"] == "inside"),
    ]:
        sub = firsts[mask]
        cats.append(
            {
                "group": label,
                "n": len(sub),
                "sumR": round(sub["pnl_r"].sum(), 2),
                "meanR": round(sub["pnl_r"].mean(), 3) if len(sub) else np.nan,
                "win": round((sub["pnl_r"] > 0).mean(), 3) if len(sub) else np.nan,
            }
        )
    print(pd.DataFrame(cats).to_string(index=False))

    print("\n=== fer_ols_pos 极端区间 (对照 FBF strict) ===")
    firsts["ols_bucket"] = pd.cut(
        firsts["fer_ols_pos"].astype(float),
        bins=[-0.01, 0.10, 0.25, 0.75, 0.90, 1.01],
        labels=["<0.10", "0.10-0.25", "0.25-0.75", "0.75-0.90", ">0.90"],
    )
    grp2 = (
        firsts.groupby("ols_bucket", observed=True)
        .agg(n=("pnl_r", "size"), sumR=("pnl_r", "sum"), meanR=("pnl_r", "mean"))
        .round(3)
    )
    print(grp2.to_string())

    print("\n=== 假设硬门：只保留 breach_aligned（LONG & above / SHORT & below）===")
    kept = firsts[firsts["breach_aligned"]]
    dropped = firsts[~firsts["breach_aligned"]]
    print(
        f"kept: n={len(kept)}, sumR={kept['pnl_r'].sum():.2f}, "
        f"meanR={kept['pnl_r'].mean() if len(kept) else float('nan'):.3f}, "
        f"win={ (kept['pnl_r']>0).mean() if len(kept) else float('nan'):.3f}"
    )
    print(
        f"dropped: n={len(dropped)}, sumR={dropped['pnl_r'].sum():.2f}, "
        f"meanR={dropped['pnl_r'].mean() if len(dropped) else float('nan'):.3f}"
    )
    # 影响加仓：要不要看整个 trade family
    adds = df[df["is_add_position"].astype(bool)]
    add_on_kept = adds[adds["symbol"].isin(kept["symbol"].unique())]  # rough
    print(
        f"adds on symbols that passed filter: n={len(add_on_kept)}, sumR={add_on_kept['pnl_r'].sum():.2f}"
    )

    # 样本分月: 确认不是某个月集中的
    print("\n=== breach_aligned trades 分月分布 ===")
    kept["month"] = pd.to_datetime(kept["entry_time"]).dt.strftime("%Y-%m")
    print(
        kept.groupby("month")
        .agg(n=("pnl_r", "size"), sumR=("pnl_r", "sum"))
        .round(2)
        .to_string()
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root", default="results/reports/srb_fast_ab_e1_e2_full16m/baseline"
    )
    global FEATURE_STORE
    p.add_argument("--feature-store", default=FEATURE_STORE)
    args = p.parse_args()
    FEATURE_STORE = args.feature_store
    trades = _load_trades(Path(args.root))
    print(
        f"loaded trades: {len(trades)} rows, symbols: {sorted(trades['symbol'].unique())}"
    )
    df = _annotate(trades)
    if df.empty:
        print("no annotated data")
        return 1
    _summarize(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
