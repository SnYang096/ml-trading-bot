#!/usr/bin/env python3
"""
Evidence Demo: 证明 evidence 候选特征对 trade quality 有分层作用
=================================================================
方法:
  1. 读取多个策略的 logs_gated.parquet（含全量特征 + gate 决策 + 交易结果）
  2. 只保留 gate 放行 + 有入场信号的样本
  3. 对每个 evidence 候选特征:
     a) 按 quantile 分 5 组，计算每组 winrate、avg_R、tail_R（≥2R 占比）
     b) Spearman 相关 → 单调性
     c) 高分组 vs 低分组的 expectancy 差异 t-test
  4. 输出汇总表 + 每个特征的分组明细

运行:
  python scripts/evidence_demo.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT)

# ─────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────

# 三大类 evidence 候选特征 (volatility / liquidity / leverage)
EVIDENCE_CANDIDATES = {
    # ── Volatility State ──
    "bpc_bb_compression": "volatility",
    "bpc_vol_compression": "volatility",
    "bpc_vol_compression_state": "volatility",
    "bb_width_normalized_pct": "volatility",
    "atr_percentile": "volatility",
    "dual_compression_score": "volatility",
    "wpt_compression_score": "volatility",
    "vp_compression_score": "volatility",
    "bpc_garch_compression": "volatility",
    "vol_zscore": "volatility",
    "vol_percentile_approx": "volatility",
    "garch_volatility": "volatility",
    "bpc_pre_breakout_score": "volatility",
    # ── Liquidity / Participation ──
    "volume_ratio_pct": "liquidity",
    "volume_activity_pct": "liquidity",
    "volume_participation_score": "liquidity",
    "ofci_pct": "liquidity",
    "vpin": "liquidity",
    "vpin_zscore_20": "liquidity",
    "vpin_zscore_50": "liquidity",
    "liquidity_void_speed": "liquidity",
    "liquidity_void_price_impact": "liquidity",
    # ── Leverage / Positioning ──
    "funding_rate_zscore_50": "leverage",
    "funding_rate_abs_zscore_50": "leverage",
    "oi_change_pct": "leverage",
    "oi_zscore": "leverage",
    "oi_flow_zscore": "leverage",
    "funding_oi_crowding_score": "leverage",
    "dual_ignition_score": "leverage",
    "dual_exhaustion_score": "leverage",
    # ── BPC 结构 ──
    "bpc_pullback_depth_pct": "structure",
    "bpc_pullback_quality": "structure",
    "bpc_structure_health": "structure",
    "bpc_phase_confidence": "structure",
    "sr_strength_max": "structure",
}

# 用于识别 outcome 的候选列名 (按优先级)
OUTCOME_CANDIDATES = [
    "forward_rr",
    "rr",
    "bpc_impulse_return_atr",
    "return_atr",
    "realized_rr",
]

N_BINS = 5
TAIL_THRESHOLD = 2.0  # ≥ 2R 算 tail trade


def find_latest_logs_gated(strategy: str) -> Path | None:
    """找到某策略最新的 logs_gated.parquet."""
    results = PROJECT / "results"
    candidates = sorted(
        results.glob(f"train_final_*_rr_extreme/{strategy}/logs_gated.parquet"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def detect_outcome_col(df: pd.DataFrame) -> str | None:
    """自动检测 outcome 列."""
    for col in OUTCOME_CANDIDATES:
        if col in df.columns:
            return col
    # fallback: 搜索任何含 rr 或 return 的数值列
    for col in df.columns:
        if ("rr" in col.lower() or "return" in col.lower()) and df[col].dtype in [
            np.float64,
            np.float32,
            np.int64,
        ]:
            return col
    return None


def filter_gated_entries(df: pd.DataFrame) -> pd.DataFrame:
    """只保留 gate 放行 + 有入场信号的样本."""
    mask = pd.Series(True, index=df.index)

    # gate 过滤
    if "gate_decision" in df.columns:
        mask &= df["gate_decision"] == "allow"
    elif "gate_passed" in df.columns:
        mask &= df["gate_passed"] == True  # noqa: E712

    # entry 过滤
    if "entry_direction" in df.columns:
        mask &= df["entry_direction"] != 0

    return df[mask].copy()


def analyze_feature(
    df: pd.DataFrame, feature: str, outcome_col: str, n_bins: int = N_BINS
) -> dict | None:
    """对单个特征做分组分析."""
    if feature not in df.columns:
        return None

    vals = df[feature].dropna()
    if len(vals) < 50:
        return None

    # 取非 NaN 的行
    sub = df[[feature, outcome_col]].dropna()
    if len(sub) < 50:
        return None

    # 分位数分组
    try:
        sub["bin"] = pd.qcut(sub[feature], n_bins, labels=False, duplicates="drop")
    except ValueError:
        return None

    actual_bins = sub["bin"].nunique()
    if actual_bins < 3:
        return None

    # 每组统计
    groups = []
    for b in sorted(sub["bin"].unique()):
        g = sub[sub["bin"] == b]
        rr = g[outcome_col]
        groups.append(
            {
                "bin": int(b),
                "n": len(g),
                "feature_mean": g[feature].mean(),
                "feature_median": g[feature].median(),
                "winrate": (rr > 0).mean(),
                "avg_R": rr.mean(),
                "median_R": rr.median(),
                "tail_pct": (rr >= TAIL_THRESHOLD).mean(),
                "avg_tail_R": (
                    rr[rr >= TAIL_THRESHOLD].mean()
                    if (rr >= TAIL_THRESHOLD).any()
                    else np.nan
                ),
                "worst_R": rr.min(),
            }
        )
    groups_df = pd.DataFrame(groups)

    # Spearman: feature vs outcome (用原始值, 非分组)
    sp_r, sp_p = stats.spearmanr(sub[feature], sub[outcome_col])

    # 高分组 vs 低分组 t-test
    low_bin = groups_df["bin"].min()
    high_bin = groups_df["bin"].max()
    low_rr = sub[sub["bin"] == low_bin][outcome_col]
    high_rr = sub[sub["bin"] == high_bin][outcome_col]
    if len(low_rr) > 5 and len(high_rr) > 5:
        t_stat, t_p = stats.ttest_ind(high_rr, low_rr, equal_var=False)
    else:
        t_stat, t_p = np.nan, np.nan

    # 单调性: winrate 或 tail_pct 是否随 bin 递增
    wr_trend = stats.spearmanr(groups_df["bin"], groups_df["winrate"])
    tail_trend = stats.spearmanr(groups_df["bin"], groups_df["tail_pct"])
    avgR_trend = stats.spearmanr(groups_df["bin"], groups_df["avg_R"])

    return {
        "n_samples": len(sub),
        "spearman_r": sp_r,
        "spearman_p": sp_p,
        "ttest_t": t_stat,
        "ttest_p": t_p,
        "wr_low": groups_df.iloc[0]["winrate"],
        "wr_high": groups_df.iloc[-1]["winrate"],
        "wr_delta": groups_df.iloc[-1]["winrate"] - groups_df.iloc[0]["winrate"],
        "wr_monotonic_r": wr_trend.statistic,
        "avgR_low": groups_df.iloc[0]["avg_R"],
        "avgR_high": groups_df.iloc[-1]["avg_R"],
        "avgR_delta": groups_df.iloc[-1]["avg_R"] - groups_df.iloc[0]["avg_R"],
        "avgR_monotonic_r": avgR_trend.statistic,
        "tail_low": groups_df.iloc[0]["tail_pct"],
        "tail_high": groups_df.iloc[-1]["tail_pct"],
        "tail_delta": groups_df.iloc[-1]["tail_pct"] - groups_df.iloc[0]["tail_pct"],
        "tail_monotonic_r": tail_trend.statistic,
        "groups": groups_df,
    }


def print_summary_table(results: dict[str, dict], categories: dict[str, str]):
    """输出汇总表."""
    rows = []
    for feat, res in results.items():
        if res is None:
            continue
        cat = categories.get(feat, "?")
        rows.append(
            {
                "feature": feat,
                "category": cat,
                "n": res["n_samples"],
                "spearman_r": res["spearman_r"],
                "spearman_p": res["spearman_p"],
                "wr_delta": res["wr_delta"],
                "wr_mono": res["wr_monotonic_r"],
                "avgR_delta": res["avgR_delta"],
                "avgR_mono": res["avgR_monotonic_r"],
                "tail_delta": res["tail_delta"],
                "tail_mono": res["tail_monotonic_r"],
                "ttest_p": res["ttest_p"],
            }
        )

    if not rows:
        print("❌ No features had enough data for analysis")
        return

    summary = pd.DataFrame(rows).sort_values("spearman_p")

    print("\n" + "=" * 120)
    print("EVIDENCE FEATURE SUMMARY")
    print("=" * 120)
    print(
        f"{'feature':<40} {'cat':>9} {'n':>6} {'sp_r':>6} {'sp_p':>8} "
        f"{'wr_Δ':>7} {'wr_↗':>5} {'avgR_Δ':>8} {'avgR_↗':>6} "
        f"{'tail_Δ':>7} {'tail_↗':>6} {'t_p':>8}"
    )
    print("-" * 120)

    for _, r in summary.iterrows():
        # Mark significant features
        sig = ""
        if r["spearman_p"] < 0.05:
            sig = "★"
        if r["spearman_p"] < 0.01:
            sig = "★★"
        if r["spearman_p"] < 0.001:
            sig = "★★★"

        print(
            f"{r['feature']:<40} {r['category']:>9} {r['n']:>6} "
            f"{r['spearman_r']:>+6.3f} {r['spearman_p']:>8.4f} "
            f"{r['wr_delta']:>+7.3f} {r['wr_mono']:>+5.2f} "
            f"{r['avgR_delta']:>+8.3f} {r['avgR_mono']:>+6.2f} "
            f"{r['tail_delta']:>+7.3f} {r['tail_mono']:>+6.2f} "
            f"{r['ttest_p']:>8.4f} {sig}"
        )

    # 按类别汇总
    print("\n" + "=" * 80)
    print("PER-CATEGORY SUMMARY")
    print("=" * 80)
    for cat in ["volatility", "liquidity", "leverage", "structure"]:
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        sig_rows = [r for r in cat_rows if r["spearman_p"] < 0.05]
        print(
            f"\n  {cat.upper():>12}: {len(cat_rows)} features tested, "
            f"{len(sig_rows)} significant (p<0.05)"
        )
        for r in sorted(cat_rows, key=lambda x: x["spearman_p"]):
            mark = "✓" if r["spearman_p"] < 0.05 else " "
            print(
                f"    {mark} {r['feature']:<35} sp_r={r['spearman_r']:+.3f}  "
                f"p={r['spearman_p']:.4f}  wr_Δ={r['wr_delta']:+.3f}  "
                f"tail_Δ={r['tail_delta']:+.3f}"
            )

    return summary


def print_feature_detail(feat: str, res: dict, cat: str):
    """输出单个特征的分组明细."""
    g = res["groups"]
    print(f"\n  ── {feat} [{cat}] ──")
    print(
        f"  Spearman r={res['spearman_r']:.3f} (p={res['spearman_p']:.4f})  "
        f"t-test p={res['ttest_p']:.4f}  n={res['n_samples']}"
    )
    print(
        f"  {'bin':>4} {'n':>6} {'feat_mean':>10} {'winrate':>8} "
        f"{'avg_R':>8} {'med_R':>8} {'tail%':>7} {'avg_tail':>9}"
    )
    for _, r in g.iterrows():
        print(
            f"  {int(r['bin']):>4} {int(r['n']):>6} {r['feature_mean']:>10.4f} "
            f"{r['winrate']:>8.3f} {r['avg_R']:>+8.3f} {r['median_R']:>+8.3f} "
            f"{r['tail_pct']:>7.3f} {r['avg_tail_R']:>9.3f}"
        )


def run_strategy(strategy: str) -> dict[str, dict] | None:
    """分析单个策略."""
    path = find_latest_logs_gated(strategy)
    if path is None:
        print(f"\n⚠️  {strategy}: No logs_gated.parquet found")
        return None

    print(f"\n{'='*80}")
    print(f"Strategy: {strategy.upper()}")
    print(f"File: {path}")
    print(f"{'='*80}")

    df = pd.read_parquet(path)
    print(f"Raw rows: {len(df)}, columns: {len(df.columns)}")

    # 检测 outcome 列
    outcome_col = detect_outcome_col(df)
    if outcome_col is None:
        print(f"❌ No outcome column found. Available: {list(df.columns)[:30]}...")
        return None
    print(f"Outcome column: {outcome_col}")

    # 过滤
    df_filtered = filter_gated_entries(df)
    print(
        f"After gate+entry filter: {len(df_filtered)} trades "
        f"(pass rate: {len(df_filtered)/len(df)*100:.1f}%)"
    )

    if len(df_filtered) < 50:
        print(f"⚠️  Too few trades ({len(df_filtered)}), skipping")
        return None

    # 基线统计
    rr = df_filtered[outcome_col]
    print(f"\nBaseline (all gated trades):")
    print(
        f"  Trades: {len(rr)}, WinRate: {(rr>0).mean():.3f}, "
        f"AvgR: {rr.mean():.3f}, MedianR: {rr.median():.3f}, "
        f"Tail(≥{TAIL_THRESHOLD}R): {(rr>=TAIL_THRESHOLD).mean():.3f}"
    )

    # 找到在这个 DataFrame 中存在的候选特征
    available = {f: c for f, c in EVIDENCE_CANDIDATES.items() if f in df.columns}
    print(f"\nEvidence candidates found: {len(available)}/{len(EVIDENCE_CANDIDATES)}")
    missing = set(EVIDENCE_CANDIDATES) - set(available)
    if missing:
        print(f"  Missing: {sorted(missing)}")

    # 分析每个特征
    results = {}
    for feat in sorted(available.keys()):
        res = analyze_feature(df_filtered, feat, outcome_col)
        results[feat] = res

    # 输出汇总表
    summary = print_summary_table(results, EVIDENCE_CANDIDATES)

    # 输出 top 特征的分组明细
    top_feats = sorted(
        [(f, r) for f, r in results.items() if r is not None],
        key=lambda x: x[1]["spearman_p"],
    )[:10]

    if top_feats:
        print(f"\n{'='*80}")
        print(f"TOP 10 FEATURES - DETAILED BREAKDOWN")
        print(f"{'='*80}")
        for feat, res in top_feats:
            print_feature_detail(feat, res, EVIDENCE_CANDIDATES.get(feat, "?"))

    return results


def main():
    OUT = PROJECT / "_evidence_demo_results.txt"

    # 重定向 stdout 到文件 + 终端
    class Tee:
        def __init__(self, *fps):
            self.fps = fps

        def write(self, s):
            for fp in self.fps:
                fp.write(s)
                fp.flush()

        def flush(self):
            for fp in self.fps:
                fp.flush()

    f_out = open(OUT, "w", encoding="utf-8")
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, f_out)

    print("=" * 80)
    print("Evidence Feature Demo")
    print("验证 volatility / liquidity / leverage 三类特征对 trade quality 的分层作用")
    print("=" * 80)

    all_results = {}
    for strategy in ["bpc", "me", "fer"]:
        res = run_strategy(strategy)
        if res:
            all_results[strategy] = res

    # ── 跨策略汇总 ──
    if len(all_results) > 1:
        print(f"\n\n{'#'*80}")
        print("CROSS-STRATEGY CONSENSUS")
        print(f"{'#'*80}")

        # 找在所有策略中都显著的特征
        all_feats = set()
        for res in all_results.values():
            all_feats.update(f for f, r in res.items() if r is not None)

        consensus = []
        for feat in sorted(all_feats):
            sig_count = 0
            total_count = 0
            avg_sp_r = []
            for strat, res in all_results.items():
                if feat in res and res[feat] is not None:
                    total_count += 1
                    if res[feat]["spearman_p"] < 0.05:
                        sig_count += 1
                    avg_sp_r.append(res[feat]["spearman_r"])
            if total_count > 0:
                consensus.append(
                    {
                        "feature": feat,
                        "category": EVIDENCE_CANDIDATES.get(feat, "?"),
                        "sig_in": sig_count,
                        "tested_in": total_count,
                        "avg_sp_r": np.mean(avg_sp_r),
                    }
                )

        consensus_df = pd.DataFrame(consensus).sort_values(
            ["sig_in", "avg_sp_r"], ascending=[False, True]
        )
        print(f"\n{'feature':<40} {'cat':>9} {'sig/tested':>10} {'avg_sp_r':>9}")
        print("-" * 70)
        for _, r in consensus_df.head(20).iterrows():
            mark = "★" * int(r["sig_in"])
            print(
                f"{r['feature']:<40} {r['category']:>9} "
                f"{int(r['sig_in'])}/{int(r['tested_in']):>9} "
                f"{r['avg_sp_r']:>+9.3f}  {mark}"
            )

    # ── 结论 ──
    print(f"\n\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    print("如果上述特征在 spearman_p < 0.05 且 wr_Δ > 0 或 tail_Δ > 0,")
    print("则说明该特征对 trade quality 有分层作用, 可用作 evidence。")
    print("注意: 即使 wr_Δ < 0, 如果 tail_Δ > 0 且 avgR_Δ > 0,")
    print("说明特征改变了收益分布形状 (distribution shift), 同样有用。")
    print(f"\nResults saved to: {OUT}")

    sys.stdout = old_stdout
    f_out.close()
    print(f"\n✅ Done! Results saved to {OUT}")


if __name__ == "__main__":
    main()
