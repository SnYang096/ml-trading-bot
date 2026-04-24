"""
SRB 破位级别归因诊断（一次性研究用）
===================================
目的：回答核心疑问——"L3 破位的 SRB 真的不同吗？"

每一笔 SRB trade 默认已经通过 L1 (~20-bar swing) 突破确认。
本脚本在 entry_time 时刻 join 入场 bar 特征，给每笔 trade 打三级 confluence 标签：

    L1-only           : 未同时触发 L2/L3 confluence（纯窄窗破位）
    L1+L2             : 入场时刻 L2 (~160-bar POC) 也处于破位距离内，且方向对齐
    L1+L3             : 入场时刻 L3 (~240-bar wide SR) 也处于破位距离内，且方向对齐
    L1+L2+L3          : 三级同时 confluence

输出：
    - 每组 n / totalR / meanR / medianR / winrate / 平均 bars_held / exit_reason 分布 / 平均 size_multiplier
    - Bootstrap 95% CI（meanR）
    - 单尾 bootstrap p：triple-confluence meanR 是否显著 > L1-only
    - 阈值扫描 wide_sr_dist_atr / L2 narrow_dist_atr

用法：
    python scripts/analyze_srb_break_levels.py \
        --run-dir results/srb/slow-rolling-sim/_rolling_sim/20260421_222624 \
        --feature-store feature_store/features_srb_120T_5643a66b47 \
        --out reports/srb_break_level_attribution.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]

# L1 narrow swing 窄窗，与 srb_regime.maybe_inject_srb_experiment_features 保持一致
L1_SWING_LOOKBACK = 20
L1_ANCHOR_SHIFT = 1


# ---------------------------------------------------------------------------
# 1. Trade 加载
# ---------------------------------------------------------------------------


def load_trades(run_dir: str) -> pd.DataFrame:
    pattern = os.path.join(run_dir, "fast_month_*", "srb", "event_trades_srb.csv")
    files = sorted(glob.glob(pattern))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as exc:  # pragma: no cover
            print(f"  [skip] {f}: {exc}")
            continue
        df["_src"] = os.path.basename(os.path.dirname(os.path.dirname(f)))
        dfs.append(df)
    if not dfs:
        raise SystemExit(f"No trade csv under {pattern}")
    trades = pd.concat(dfs, ignore_index=True)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert(
        None
    )
    trades["exit_time"] = pd.to_datetime(
        trades["exit_time"], utc=True, errors="coerce"
    ).dt.tz_convert(None)
    trades["pnl_r"] = pd.to_numeric(trades["pnl_r"], errors="coerce")
    return trades


# ---------------------------------------------------------------------------
# 2. Feature store 加载
# ---------------------------------------------------------------------------


def load_symbol_features(store: str, symbol: str, tf: str = "120T") -> pd.DataFrame:
    files = sorted(glob.glob(f"{store}/{symbol}/{tf}/*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    return df


def add_l1_narrow_sr(df: pd.DataFrame) -> pd.DataFrame:
    """重建 L1 窄窗 swing SR（与 srb_regime 保持一致：20-bar，shift=1）。"""
    if df.empty:
        return df
    out = df.copy()
    high_roll = (
        df["high"]
        .rolling(L1_SWING_LOOKBACK, min_periods=L1_SWING_LOOKBACK // 2)
        .max()
        .shift(L1_ANCHOR_SHIFT)
    )
    low_roll = (
        df["low"]
        .rolling(L1_SWING_LOOKBACK, min_periods=L1_SWING_LOOKBACK // 2)
        .min()
        .shift(L1_ANCHOR_SHIFT)
    )
    atr = df["atr"].replace(0.0, np.nan)
    out["l1_sr_upper_px"] = high_roll
    out["l1_sr_lower_px"] = low_roll
    out["l1_range_width_atr"] = (high_roll - low_roll) / atr
    return out


# ---------------------------------------------------------------------------
# 3. Enrich trades
# ---------------------------------------------------------------------------

ENRICH_COLS = [
    "atr",
    "close",
    "dist_to_nearest_sr",
    "sr_strength_max",
    "wide_sr_upper_px",
    "wide_sr_lower_px",
    "wide_sr_dist_atr",
    "wide_sr_side",
    "wide_sr_range_width_atr",
    "l1_sr_upper_px",
    "l1_sr_lower_px",
    "l1_range_width_atr",
    "bb_width_normalized_pct",
    "trend_r2_20",
    "compression_duration",
    "dual_compression_score",
    "sma_200_slope",
    "fer_range_pos_20",
    "vpin_trend",
]


def enrich_trades(
    trades: pd.DataFrame, feat_by_sym: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    pieces = []
    for sym, g in trades.groupby("symbol"):
        fdf = feat_by_sym.get(sym)
        g = g.sort_values("entry_time").copy()
        if fdf is None or fdf.empty:
            for c in ENRICH_COLS:
                g[f"f_{c}"] = np.nan
            pieces.append(g)
            continue
        avail = [c for c in ENRICH_COLS if c in fdf.columns]
        sub = fdf[avail].copy()
        sub.columns = [f"f_{c}" for c in sub.columns]
        sub = sub.reset_index().rename(columns={sub.index.name or "index": "bar_time"})
        if "bar_time" not in sub.columns:
            sub = sub.rename(columns={sub.columns[0]: "bar_time"})
        sub["bar_time"] = pd.to_datetime(sub["bar_time"])
        sub = sub.sort_values("bar_time")
        merged = pd.merge_asof(
            g.sort_values("entry_time"),
            sub,
            left_on="entry_time",
            right_on="bar_time",
            direction="backward",
            tolerance=pd.Timedelta("2H"),
        )
        for c in [c for c in ENRICH_COLS if c not in avail]:
            merged[f"f_{c}"] = np.nan
        pieces.append(merged)
    out = pd.concat(pieces, ignore_index=True).sort_values("entry_time")
    return out


# ---------------------------------------------------------------------------
# 4. Confluence 打标签
# ---------------------------------------------------------------------------


def tag_confluence(
    trades: pd.DataFrame,
    l2_atr_threshold: float = 1.0,
    l3_atr_threshold: float = 1.5,
) -> pd.DataFrame:
    out = trades.copy()
    entry_px = out["entry_price"].astype(float)
    atr = out["f_atr"].replace(0.0, np.nan)
    side_long = out["side"].str.upper().eq("LONG")

    # L2: dist_to_nearest_sr 为 signed 相对百分比（相对价格）。取绝对值 × 价格 / ATR -> ATR 倍数。
    dist_pct = out["f_dist_to_nearest_sr"].abs()
    out["narrow_dist_atr"] = (dist_pct * entry_px) / atr

    # L3: wide_sr_dist_atr 是 ATR 距离；side=+1 上沿，-1 下沿。
    wide_dist = out["f_wide_sr_dist_atr"]
    wide_side = out["f_wide_sr_side"]

    # 方向对齐：LONG 突破 = 靠近/刚越过上沿 -> wide_sr_side == +1
    wide_side_aligned = np.where(side_long, wide_side > 0, wide_side < 0)

    out["l2_confluence"] = (out["narrow_dist_atr"] <= l2_atr_threshold) & out[
        "narrow_dist_atr"
    ].notna()
    out["l3_confluence"] = (
        (wide_dist <= l3_atr_threshold) & wide_dist.notna() & wide_side_aligned
    )

    def _grp(r):
        l2 = bool(r.l2_confluence)
        l3 = bool(r.l3_confluence)
        if l2 and l3:
            return "L1+L2+L3"
        if l3:
            return "L1+L3"
        if l2:
            return "L1+L2"
        return "L1_only"

    out["break_group"] = out.apply(_grp, axis=1)
    return out


# ---------------------------------------------------------------------------
# 5. 统计 & bootstrap
# ---------------------------------------------------------------------------


def summarize_group(g: pd.DataFrame) -> dict:
    n = len(g)
    if n == 0:
        return {"n": 0}
    pnl = g["pnl_r"].dropna().to_numpy()
    wins = (pnl > 0).mean() if len(pnl) else float("nan")
    exit_reason = (
        g["exit_reason"].value_counts().to_dict() if "exit_reason" in g.columns else {}
    )
    return {
        "n": int(n),
        "n_valid_pnl": int(len(pnl)),
        "totalR": float(np.nansum(pnl)),
        "meanR": float(np.nanmean(pnl)) if len(pnl) else float("nan"),
        "medianR": float(np.nanmedian(pnl)) if len(pnl) else float("nan"),
        "stdR": float(np.nanstd(pnl, ddof=1)) if len(pnl) > 1 else float("nan"),
        "winrate": float(wins),
        "mean_bars_held": (
            float(pd.to_numeric(g.get("bars_held"), errors="coerce").mean())
            if "bars_held" in g.columns
            else float("nan")
        ),
        "median_bars_held": (
            float(pd.to_numeric(g.get("bars_held"), errors="coerce").median())
            if "bars_held" in g.columns
            else float("nan")
        ),
        "mean_size_multiplier": (
            float(pd.to_numeric(g.get("size_multiplier"), errors="coerce").mean())
            if "size_multiplier" in g.columns
            else float("nan")
        ),
        "exit_reasons": exit_reason,
        "add_position_pct": (
            float(
                (
                    pd.to_numeric(g.get("is_add_position"), errors="coerce").fillna(0)
                    > 0
                ).mean()
            )
            if "is_add_position" in g.columns
            else float("nan")
        ),
        "reverse_pct": (
            float(
                (
                    pd.to_numeric(g.get("is_reverse"), errors="coerce").fillna(0) > 0
                ).mean()
            )
            if "is_reverse" in g.columns
            else float("nan")
        ),
    }


def bootstrap_mean_ci(
    x: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 42
) -> Tuple[float, float]:
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    draws = x[idx].mean(axis=1)
    lo = float(np.quantile(draws, (1 - ci) / 2))
    hi = float(np.quantile(draws, 1 - (1 - ci) / 2))
    return (lo, hi)


def bootstrap_diff_pvalue(
    a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 42
) -> float:
    """单尾 p：P(mean(a) <= mean(b)) 的 bootstrap 估计（越小越支持 a>b）。"""
    a = np.asarray([v for v in a if np.isfinite(v)])
    b = np.asarray([v for v in b if np.isfinite(v)])
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    return float((da <= db).mean())


# ---------------------------------------------------------------------------
# 6. 阈值扫描
# ---------------------------------------------------------------------------


def threshold_sweep(df: pd.DataFrame, col: str, thresholds: List[float]) -> List[dict]:
    rows = []
    valid = df[df[col].notna()].copy()
    if valid.empty:
        return rows
    for t in thresholds:
        near = valid[valid[col] <= t]
        far = valid[valid[col] > t]
        rows.append(
            {
                "threshold_atr": t,
                "near_n": int(len(near)),
                "near_totalR": float(near["pnl_r"].sum()),
                "near_meanR": (
                    float(near["pnl_r"].mean()) if len(near) else float("nan")
                ),
                "near_win": (
                    float((near["pnl_r"] > 0).mean()) if len(near) else float("nan")
                ),
                "far_n": int(len(far)),
                "far_totalR": float(far["pnl_r"].sum()),
                "far_meanR": float(far["pnl_r"].mean()) if len(far) else float("nan"),
                "far_win": (
                    float((far["pnl_r"] > 0).mean()) if len(far) else float("nan")
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
        "--run-dir",
        required=True,
        help="rolling_sim run dir, e.g. results/srb/slow-rolling-sim/_rolling_sim/20260421_222624",
    )
    ap.add_argument(
        "--feature-store",
        required=True,
        help="SRB feature store dir containing wide_sr_upper_px",
    )
    ap.add_argument("--out", default="reports/srb_break_level_attribution.json")
    ap.add_argument("--l2-atr-threshold", type=float, default=1.0)
    ap.add_argument("--l3-atr-threshold", type=float, default=1.5)
    ap.add_argument(
        "--filter-add-reverse",
        action="store_true",
        help="排除 add_position / reverse 的 trade，只留开仓首单",
    )
    args = ap.parse_args()

    print("=" * 72)
    print(f"run_dir       : {args.run_dir}")
    print(f"feature_store : {args.feature_store}")
    print(f"L2 threshold  : {args.l2_atr_threshold} ATR")
    print(f"L3 threshold  : {args.l3_atr_threshold} ATR")

    # 1. 加载 trades
    trades = load_trades(args.run_dir)
    print(
        f"\n原始 trades: {len(trades)} rows  time=[{trades.entry_time.min()} -> {trades.entry_time.max()}]"
    )
    if args.filter_add_reverse:
        mask = (
            pd.to_numeric(trades.get("is_add_position", 0), errors="coerce").fillna(0)
            == 0
        ) & (pd.to_numeric(trades.get("is_reverse", 0), errors="coerce").fillna(0) == 0)
        trades = trades[mask].copy()
        print(f"过滤 add_pos/reverse 后: {len(trades)} rows")

    # 2. 加载 features & 重建 L1
    print("\n加载 feature store ...")
    feat_by_sym: Dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        fdf = load_symbol_features(args.feature_store, sym)
        if fdf.empty:
            print(f"  [skip] {sym}")
            continue
        fdf = add_l1_narrow_sr(fdf)
        feat_by_sym[sym] = fdf
        print(
            f"  {sym}: {fdf.shape[0]} bars, {fdf.index.min()} -> {fdf.index.max()}, "
            f"wide_sr_upper_px present={'wide_sr_upper_px' in fdf.columns}"
        )

    # 3. Enrich + tag
    enriched = enrich_trades(trades, feat_by_sym)
    n_merged = enriched["f_atr"].notna().sum()
    print(f"\nenrich 成功: {n_merged}/{len(enriched)}")
    tagged = tag_confluence(enriched, args.l2_atr_threshold, args.l3_atr_threshold)

    # 4. 全样本汇总
    overall = summarize_group(tagged)

    # 5. 分组
    group_order = ["L1_only", "L1+L2", "L1+L3", "L1+L2+L3"]
    group_stats: Dict[str, dict] = {}
    for grp in group_order:
        g = tagged[tagged["break_group"] == grp]
        stat = summarize_group(g)
        if stat.get("n_valid_pnl", 0) >= 2:
            lo, hi = bootstrap_mean_ci(g["pnl_r"].dropna().to_numpy())
            stat["meanR_ci95"] = [lo, hi]
        group_stats[grp] = stat

    # 6. triple vs L1_only 显著性
    l1_only_R = (
        tagged.loc[tagged["break_group"] == "L1_only", "pnl_r"].dropna().to_numpy()
    )
    triple_R = (
        tagged.loc[tagged["break_group"] == "L1+L2+L3", "pnl_r"].dropna().to_numpy()
    )
    p_triple_gt_l1 = bootstrap_diff_pvalue(triple_R, l1_only_R)
    # L1+L2 vs L1_only
    l12_R = tagged.loc[tagged["break_group"] == "L1+L2", "pnl_r"].dropna().to_numpy()
    p_l12_gt_l1 = bootstrap_diff_pvalue(l12_R, l1_only_R)
    # L1+L3 vs L1_only
    l13_R = tagged.loc[tagged["break_group"] == "L1+L3", "pnl_r"].dropna().to_numpy()
    p_l13_gt_l1 = bootstrap_diff_pvalue(l13_R, l1_only_R)

    # 7. 阈值扫描
    sweep_wide = threshold_sweep(
        tagged, "f_wide_sr_dist_atr", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    )
    sweep_narrow = threshold_sweep(
        tagged, "narrow_dist_atr", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    )

    # 8. 按 side 分解 triple confluence（方向是否对称）
    by_side_triple = {}
    for s, g in tagged[tagged["break_group"] == "L1+L2+L3"].groupby("side"):
        by_side_triple[s] = summarize_group(g)

    # 9. 持仓时长分布
    holds_by_group = {
        grp: pd.to_numeric(
            tagged.loc[tagged.break_group == grp, "bars_held"], errors="coerce"
        )
        .dropna()
        .quantile([0.25, 0.5, 0.75, 0.9])
        .to_dict()
        for grp in group_order
    }

    report = {
        "config": {
            "run_dir": args.run_dir,
            "feature_store": args.feature_store,
            "l2_atr_threshold": args.l2_atr_threshold,
            "l3_atr_threshold": args.l3_atr_threshold,
            "filter_add_reverse": args.filter_add_reverse,
            "l1_swing_lookback": L1_SWING_LOOKBACK,
            "l1_anchor_shift": L1_ANCHOR_SHIFT,
        },
        "overall": overall,
        "group_stats": group_stats,
        "significance": {
            "p_L1+L2+L3 > L1_only": p_triple_gt_l1,
            "p_L1+L2 > L1_only": p_l12_gt_l1,
            "p_L1+L3 > L1_only": p_l13_gt_l1,
        },
        "threshold_sweep_wide_sr_dist_atr": sweep_wide,
        "threshold_sweep_narrow_dist_atr": sweep_narrow,
        "triple_by_side": by_side_triple,
        "holds_quantiles_by_group": holds_by_group,
    }

    # ---- 打印 ----
    print("\n" + "=" * 72)
    print("OVERALL")
    print(json.dumps(overall, indent=2, default=str))
    print("\nGROUP STATS")
    for grp in group_order:
        s = group_stats[grp]
        n = s.get("n", 0)
        if n == 0:
            print(f"  {grp:10s}: n=0")
            continue
        ci = s.get("meanR_ci95", [float("nan"), float("nan")])
        print(
            f"  {grp:10s}: n={n:4d}  totalR={s['totalR']:+7.2f}  meanR={s['meanR']:+.3f}  "
            f"medianR={s['medianR']:+.3f}  win={s['winrate']:.3f}  "
            f"barsH~{s['mean_bars_held']:.0f}  sizex={s['mean_size_multiplier']:.2f}  "
            f"addpct={s.get('add_position_pct', float('nan')):.2f}  rev={s.get('reverse_pct', float('nan')):.2f}  "
            f"CI95=[{ci[0]:+.3f},{ci[1]:+.3f}]"
        )
    print("\nSIGNIFICANCE (bootstrap p, 单尾)")
    for k, v in report["significance"].items():
        print(f"  {k:30s}: p={v}")

    print("\nWIDE SR DIST ATR 阈值扫描")
    for r in sweep_wide:
        print(
            f"  t={r['threshold_atr']:.2f}  near n={r['near_n']:3d} meanR={r['near_meanR']:+.3f} win={r['near_win']:.3f} | "
            f"far n={r['far_n']:3d} meanR={r['far_meanR']:+.3f} win={r['far_win']:.3f}"
        )
    print("\nL2 NARROW DIST ATR 阈值扫描")
    for r in sweep_narrow:
        print(
            f"  t={r['threshold_atr']:.2f}  near n={r['near_n']:3d} meanR={r['near_meanR']:+.3f} win={r['near_win']:.3f} | "
            f"far n={r['far_n']:3d} meanR={r['far_meanR']:+.3f} win={r['far_win']:.3f}"
        )

    # ---- 写文件 ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n报告已写入 {args.out}")

    # 附带一份 enriched + tagged trades
    parquet_out = args.out.replace(".json", "_trades.parquet")
    tagged.to_parquet(parquet_out, index=False)
    print(f"Tagged trades 已写入 {parquet_out}")


if __name__ == "__main__":
    main()
