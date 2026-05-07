#!/usr/bin/env python3
"""Evidence Demo - runs silently, writes all output to file."""
import os, sys, warnings, traceback
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT = Path("/home/yin/trading/ml_trading_bot")
os.chdir(PROJECT)
OUT = PROJECT / "_evidence_demo_results.txt"

EVIDENCE_CANDIDATES = {
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
    "volume_ratio_pct": "liquidity",
    "volume_activity_pct": "liquidity",
    "volume_participation_score": "liquidity",
    "ofci_pct": "liquidity",
    "vpin": "liquidity",
    "vpin_zscore_20": "liquidity",
    "vpin_zscore_50": "liquidity",
    "liquidity_void_speed": "liquidity",
    "liquidity_void_price_impact": "liquidity",
    "funding_rate_zscore_50": "leverage",
    "funding_rate_abs_zscore_50": "leverage",
    "oi_change_pct": "leverage",
    "oi_zscore": "leverage",
    "oi_flow_zscore": "leverage",
    "funding_oi_crowding_score": "leverage",
    "dual_ignition_score": "leverage",
    "dual_exhaustion_score": "leverage",
    "bpc_pullback_depth_pct": "structure",
    "bpc_pullback_quality": "structure",
    "bpc_structure_health": "structure",
    "bpc_phase_confidence": "structure",
    "sr_strength_max": "structure",
}

OUTCOME_COLS = [
    "forward_rr",
    "rr",
    "bpc_impulse_return_atr",
    "return_atr",
    "realized_rr",
]
N_BINS = 5
TAIL_R = 2.0


def find_logs(strat):
    cands = sorted(
        list(
            PROJECT.glob(f"results/train_final_*_rr_extreme/{strat}/logs_gated.parquet")
        )
        + list(
            PROJECT.glob(
                f"results/{strat}/train_final_*_rr_extreme/{strat}/logs_gated.parquet"
            )
        )
        + list(
            PROJECT.glob(
                f"results/train_final/{strat}/train_final_*_rr_extreme/{strat}/logs_gated.parquet"
            )
        ),
        key=lambda p: p.stat().st_mtime,
    )
    return cands[-1] if cands else None


def detect_outcome(df):
    for c in OUTCOME_COLS:
        if c in df.columns:
            return c
    for c in df.columns:
        if ("rr" in c.lower() or "return" in c.lower()) and df[c].dtype in [
            np.float64,
            np.float32,
            np.int64,
        ]:
            return c
    return None


def analyze(df, feat, oc):
    sub = df[[feat, oc]].dropna()
    if len(sub) < 50:
        return None
    try:
        sub["bin"] = pd.qcut(sub[feat], N_BINS, labels=False, duplicates="drop")
    except:
        return None
    if sub["bin"].nunique() < 3:
        return None

    groups = []
    for b in sorted(sub["bin"].unique()):
        g = sub[sub["bin"] == b]
        rr = g[oc]
        groups.append(
            {
                "bin": int(b),
                "n": len(g),
                "feat_mean": g[feat].mean(),
                "winrate": (rr > 0).mean(),
                "avg_R": rr.mean(),
                "med_R": rr.median(),
                "tail_pct": (rr >= TAIL_R).mean(),
                "avg_tail": rr[rr >= TAIL_R].mean() if (rr >= TAIL_R).any() else np.nan,
            }
        )
    gdf = pd.DataFrame(groups)
    sp_r, sp_p = stats.spearmanr(sub[feat], sub[oc])
    low_rr = sub[sub["bin"] == gdf["bin"].min()][oc]
    hi_rr = sub[sub["bin"] == gdf["bin"].max()][oc]
    tt, tp = (
        stats.ttest_ind(hi_rr, low_rr, equal_var=False)
        if len(low_rr) > 5 and len(hi_rr) > 5
        else (np.nan, np.nan)
    )
    wr_t = stats.spearmanr(gdf["bin"], gdf["winrate"])
    tail_t = stats.spearmanr(gdf["bin"], gdf["tail_pct"])
    ar_t = stats.spearmanr(gdf["bin"], gdf["avg_R"])
    return {
        "n": len(sub),
        "sp_r": sp_r,
        "sp_p": sp_p,
        "tt": tt,
        "tp": tp,
        "wr_lo": gdf.iloc[0]["winrate"],
        "wr_hi": gdf.iloc[-1]["winrate"],
        "wr_d": gdf.iloc[-1]["winrate"] - gdf.iloc[0]["winrate"],
        "wr_m": wr_t.statistic,
        "ar_lo": gdf.iloc[0]["avg_R"],
        "ar_hi": gdf.iloc[-1]["avg_R"],
        "ar_d": gdf.iloc[-1]["avg_R"] - gdf.iloc[0]["avg_R"],
        "ar_m": ar_t.statistic,
        "tl_lo": gdf.iloc[0]["tail_pct"],
        "tl_hi": gdf.iloc[-1]["tail_pct"],
        "tl_d": gdf.iloc[-1]["tail_pct"] - gdf.iloc[0]["tail_pct"],
        "tl_m": tail_t.statistic,
        "groups": gdf,
    }


def main():
    lines = []

    def p(s=""):
        lines.append(str(s))

    try:
        p("=" * 100)
        p(
            "Evidence Feature Demo: volatility / liquidity / leverage 对 trade quality 的分层作用"
        )
        p("=" * 100)

        all_results = {}
        for strat in ["bpc", "me-long", "fer"]:
            path = find_logs(strat)
            if not path:
                p(f"\n⚠️  {strat}: No logs_gated.parquet")
                continue
            p(f"\n{'='*80}")
            p(f"Strategy: {strat.upper()}  |  File: {path}")
            p(f"{'='*80}")
            df = pd.read_parquet(path)
            p(f"Raw: {len(df)} rows, {len(df.columns)} cols")
            oc = detect_outcome(df)
            if not oc:
                p(f"  ❌ No outcome col. Cols: {list(df.columns)[:20]}...")
                continue
            p(f"Outcome: {oc}")

            # ── 诊断: 打印 gate/entry 列的值分布 ──
            for col in [
                "gate_decision",
                "gate_passed",
                "gate_ok",
                "entry_direction",
                "direction",
            ]:
                if col in df.columns:
                    p(f"  {col} distribution: {dict(df[col].value_counts())}")

            mask = pd.Series(True, index=df.index)
            gate_col_used = None
            if "gate_decision" in df.columns:
                mask &= df["gate_decision"] == "allow"
                gate_col_used = "gate_decision"
            elif "gate_passed" in df.columns:
                mask &= df["gate_passed"] == True
                gate_col_used = "gate_passed"
            entry_col_used = None
            if "entry_direction" in df.columns:
                mask &= df["entry_direction"] != 0
                entry_col_used = "entry_direction"
            dff = df[mask].copy()
            p(
                f"Filtered (gate={gate_col_used}, entry={entry_col_used}): {len(dff)} trades ({len(dff)/len(df)*100:.1f}%)"
            )

            # ── 如果 gate+entry 过滤后为 0, 尝试 fallback ──
            if len(dff) < 50:
                # 尝试只用 gate, 不过滤 entry
                mask2 = pd.Series(True, index=df.index)
                if "gate_decision" in df.columns:
                    mask2 &= df["gate_decision"] == "allow"
                elif "gate_passed" in df.columns:
                    mask2 &= df["gate_passed"] == True
                n_gate_only = mask2.sum()
                # 尝试只用 entry, 不过滤 gate
                mask3 = pd.Series(True, index=df.index)
                if "entry_direction" in df.columns:
                    mask3 &= df["entry_direction"] != 0
                n_entry_only = mask3.sum()
                p(
                    f"  Diagnostic: gate_only={n_gate_only}, entry_only={n_entry_only}, all={len(df)}"
                )

                # fallback: 使用全量数据（logs_gated 本身就是 gate 输出）
                if n_gate_only >= 50:
                    dff = df[mask2].copy()
                    p(f"  → Fallback: using gate-only filter ({len(dff)} rows)")
                elif n_entry_only >= 50:
                    dff = df[mask3].copy()
                    p(f"  → Fallback: using entry-only filter ({len(dff)} rows)")
                elif len(df) >= 50:
                    dff = df.copy()
                    p(
                        f"  → Fallback: using ALL rows ({len(dff)} rows, logs_gated = already gated)"
                    )
                else:
                    p(f"  ⚠️  Too few trades even with fallback")
                    continue

            rr = dff[oc]
            p(
                f"Baseline: n={len(rr)} WR={((rr>0).mean()):.3f} avgR={rr.mean():.3f} medR={rr.median():.3f} tail≥2R={((rr>=TAIL_R).mean()):.3f}"
            )

            avail = {f: c for f, c in EVIDENCE_CANDIDATES.items() if f in df.columns}
            p(f"Evidence features found: {len(avail)}/{len(EVIDENCE_CANDIDATES)}")

            results = {}
            for feat in sorted(avail):
                results[feat] = analyze(dff, feat, oc)
            all_results[strat] = results

            # Summary table
            p(
                f"\n{'feature':<40} {'cat':>9} {'n':>6} {'sp_r':>6} {'sp_p':>8} {'wr_Δ':>7} {'wr↗':>5} {'avgR_Δ':>8} {'tail_Δ':>7} {'t_p':>8}"
            )
            p("-" * 110)
            sorted_feats = sorted(
                [(f, r) for f, r in results.items() if r], key=lambda x: x[1]["sp_p"]
            )
            for feat, r in sorted_feats:
                sig = (
                    "★★★"
                    if r["sp_p"] < 0.001
                    else (
                        "★★" if r["sp_p"] < 0.01 else ("★" if r["sp_p"] < 0.05 else "")
                    )
                )
                p(
                    f"{feat:<40} {avail[feat]:>9} {r['n']:>6} {r['sp_r']:>+6.3f} {r['sp_p']:>8.4f} {r['wr_d']:>+7.3f} {r['wr_m']:>+5.2f} {r['ar_d']:>+8.3f} {r['tl_d']:>+7.3f} {r['tp']:>8.4f} {sig}"
                )

            # Top 5 detail
            p(f"\nTOP 5 FEATURES - DETAILED BREAKDOWN:")
            for feat, r in sorted_feats[:5]:
                g = r["groups"]
                p(
                    f"\n  ── {feat} [{avail[feat]}] sp_r={r['sp_r']:.3f} p={r['sp_p']:.4f} n={r['n']} ──"
                )
                p(
                    f"  {'bin':>4} {'n':>6} {'feat_mean':>10} {'winrate':>8} {'avg_R':>8} {'tail%':>7}"
                )
                for _, gr in g.iterrows():
                    p(
                        f"  {int(gr['bin']):>4} {int(gr['n']):>6} {gr['feat_mean']:>10.4f} {gr['winrate']:>8.3f} {gr['avg_R']:>+8.3f} {gr['tail_pct']:>7.3f}"
                    )

            # Per-category
            p(f"\nPER-CATEGORY SUMMARY:")
            for cat in ["volatility", "liquidity", "leverage", "structure"]:
                cat_r = [
                    (f, r) for f, r in results.items() if r and avail.get(f) == cat
                ]
                if not cat_r:
                    continue
                sig_n = sum(1 for _, r in cat_r if r["sp_p"] < 0.05)
                p(
                    f"  {cat.upper():>12}: {len(cat_r)} tested, {sig_n} significant (p<0.05)"
                )
                for f, r in sorted(cat_r, key=lambda x: x[1]["sp_p"]):
                    mk = "✓" if r["sp_p"] < 0.05 else " "
                    p(
                        f"    {mk} {f:<35} sp_r={r['sp_r']:+.3f} p={r['sp_p']:.4f} wr_Δ={r['wr_d']:+.3f} tail_Δ={r['tl_d']:+.3f}"
                    )

        # Cross-strategy
        if len(all_results) > 1:
            p(f"\n\n{'#'*80}")
            p("CROSS-STRATEGY CONSENSUS")
            p(f"{'#'*80}")
            all_feats = set()
            for res in all_results.values():
                all_feats.update(f for f, r in res.items() if r)
            p(f"\n{'feature':<40} {'cat':>9} {'sig/total':>10} {'avg_sp_r':>9}")
            p("-" * 70)
            rows = []
            for feat in sorted(all_feats):
                sig_c, tot_c, sp_rs = 0, 0, []
                for s, res in all_results.items():
                    if feat in res and res[feat]:
                        tot_c += 1
                        if res[feat]["sp_p"] < 0.05:
                            sig_c += 1
                        sp_rs.append(res[feat]["sp_r"])
                if tot_c > 0:
                    rows.append((feat, sig_c, tot_c, np.mean(sp_rs)))
            for f, sc, tc, asr in sorted(rows, key=lambda x: -x[1]):
                p(
                    f"{f:<40} {EVIDENCE_CANDIDATES.get(f,'?'):>9} {sc}/{tc:>9} {asr:>+9.3f} {'★'*sc}"
                )

        p(f"\n\n{'='*80}")
        p("CONCLUSION")
        p(f"{'='*80}")
        p("spearman_p < 0.05 且 wr_Δ > 0 或 tail_Δ > 0 → 有分层作用")
        p("即使 wr_Δ < 0, 若 tail_Δ > 0 且 avgR_Δ > 0 → distribution shift, 同样有用")

    except Exception as e:
        p(f"\n\nERROR: {e}")
        p(traceback.format_exc())

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
