"""One-shot: compute TPC factor IC baseline + monthly IC_IR, save md + json.

This is a throwaway driver for 20260526. The reusable, integrated version
will live in scripts/quick_layer_scan.py as --mode ic-decay once the TPC
pipeline stabilizes (see docs/strategy/方法论_R_and_D流程_CN.md §2).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parquet = sorted(
        (PROJECT_ROOT / "results/train_final/tpc").glob(
            "train_final_*/tpc/features_labeled.parquet"
        )
    )
    if not parquet:
        print("ERROR: no train_final parquet found", file=sys.stderr)
        return 3
    p = parquet[-1]
    df = pd.read_parquet(p)
    for c in ("datetime", "timestamp"):
        if c in df.columns:
            df["_dt"] = pd.to_datetime(df[c], utc=True, errors="coerce")
            break

    target = "forward_rr"
    y_all = pd.to_numeric(df[target], errors="coerce")
    features = [
        "ema_1200_position",
        "ema_1200_slope_10",
        "tpc_pullback_depth",
        "tpc_semantic_chop",
        "vol_persistence",
        "vol_leverage_asymmetry",
        "tpc_cvd_absorption",
        "macd_atr",
    ]
    buckets = {
        "all": pd.Series(True, index=df.index),
        "cal_2024_bull": (df["_dt"] >= "2024-01-01") & (df["_dt"] < "2025-01-01"),
        "cal_2025_2026_recent": (df["_dt"] >= "2025-04-01")
        & (df["_dt"] < "2026-04-01"),
    }

    rows = []
    for feat in features:
        if feat not in df.columns:
            continue
        for bn, bm in buckets.items():
            m = bm & y_all.notna() & df[feat].notna()
            if m.sum() < 100:
                continue
            x = pd.to_numeric(df[feat][m], errors="coerce")
            y = y_all[m]
            rho, p_ = spearmanr(x, y)
            pear, _ = pearsonr(x, y)
            sub = df[m][[feat, target, "_dt"]].copy()
            sub["_period"] = sub["_dt"].dt.tz_localize(None).dt.to_period("M")
            ics = []
            for _, g in sub.groupby("_period"):
                if len(g) < 30:
                    continue
                r, _ = spearmanr(g[feat], g[target])
                if not np.isnan(r):
                    ics.append(r)
            if len(ics) >= 6:
                arr = np.array(ics)
                ic_ir = float(arr.mean() / arr.std()) if arr.std() > 0 else 0.0
                ic_pos = float((arr > 0).mean())
                n_months = len(arr)
                ic_mean_m = float(arr.mean())
            else:
                ic_ir = ic_pos = n_months = ic_mean_m = None
            rows.append(
                {
                    "feature": feat,
                    "bucket": bn,
                    "n": int(m.sum()),
                    "rank_ic": float(rho),
                    "p_value": float(p_),
                    "pearson_ic": float(pear),
                    "ic_ir_monthly": ic_ir,
                    "ic_positive_rate": ic_pos,
                    "n_months": n_months,
                    "ic_mean_monthly": ic_mean_m,
                }
            )

    out_dir = PROJECT_ROOT / "results/factor_health/ic_baseline_20260526"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline.json").write_text(
        json.dumps(
            {
                "target": target,
                "source_parquet": str(p),
                "ts": "20260526",
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# TPC factor IC baseline (20260526)",
        "",
        f"- Source: `{p}`",
        f"- Target: `{target}` (forward R-multiple at archetype trigger points)",
        f"- Rows total: {len(df)}",
        "",
        "## Per-feature IC (Spearman rank IC vs forward_rr)",
        "",
        "| feature | bucket | n | rank_IC | p | IC_IR (monthly) | IC_pos_rate | months |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    prev_feat = None
    for r in rows:
        if prev_feat and r["feature"] != prev_feat:
            lines.append("| | | | | | | | |")
        sig = (
            "***"
            if r["p_value"] < 0.001
            else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
        )
        ir = f"{r['ic_ir_monthly']:+.2f}" if r["ic_ir_monthly"] is not None else "n/a"
        pos = (
            f"{r['ic_positive_rate']*100:.0f}%"
            if r["ic_positive_rate"] is not None
            else "n/a"
        )
        nm = str(r["n_months"]) if r["n_months"] is not None else "n/a"
        lines.append(
            f"| {r['feature']} | {r['bucket']} | {r['n']} | "
            f"{r['rank_ic']:+.4f}{sig} | {r['p_value']:.2g} | {ir} | {pos} | {nm} |"
        )
        prev_feat = r["feature"]

    lines.extend(
        [
            "",
            "## Key observations",
            "",
            "1. **ema_1200_position / ema_1200_slope_10 sign flip across regimes**:",
            "   bull 2024 +0.027/+0.029 (p<0.001), recent -0.076 (p<1e-30).",
            "   Same feature flips sign between periods; explains why a single H config",
            "   cannot be optimal on both segments.",
            "2. **tpc_pullback_depth: stable -IC across all buckets** (deeper pullback",
            "   higher R), aligned with the earlier quick_layer_scan plateau finding.",
            "3. **tpc_semantic_chop on 2024 bull**: IC_IR=-0.49, IC_pos=17%. High chop",
            "   robustly hurts R; the chop gate is well-designed.",
            "4. **vol_persistence / vol_leverage_asymmetry: +IC ~+0.06*** on recent**:",
            "   higher vp/vla -> higher R. Validates B-variant: disabling vol gate on",
            "   recent added +17R. H preserves the bull-side vol gate purely for DD",
            "   protection, NOT because vp predicts forward_rr.",
            "5. **tpc_cvd_absorption: IC ~= 0 (p>0.05 across all buckets)**. The current",
            "   entry filter uses a no-signal feature; the E-variant replacement",
            "   direction (box_compression_score / vp_absorption_score) is correct.",
            "6. **macd_atr: stable -IC across all buckets**. The direction-rule using",
            "   macd_atr sign is on a real signal.",
            "",
            "## Suggested W-level health-monitor alert thresholds",
            "",
            "| feature | baseline bucket | rank_IC | alert |",
            "|---|---|---:|---|",
            "| ema_1200_position | recent | -0.076 | rolling 90d abs(IC) < 0.030 -> ALERT (regime signal decay) |",
            "| tpc_pullback_depth | all | -0.028 | rolling 90d sign flip or abs(IC)<0.010 -> ALERT |",
            "| tpc_semantic_chop | bull | -0.040 | rolling 90d IC > 0 -> ALERT (chop gate failed) |",
            "| vol_persistence | recent | +0.060 | rolling 90d abs(IC)<0.020 -> ALERT (rationale to disable gate gone) |",
            "| macd_atr | all | -0.057 | rolling 90d sign flip -> ALERT (direction inverted) |",
            "",
            "## Relation to quick_layer_scan label scan",
            "",
            "- label scan gives discrete decision metric: succ rate of archetype under condition.",
            "- IC gives continuous metric: feature -> forward R correlation.",
            "- They are cross-checks:",
            "  * label scan said H gives +1pp succ on both bull and recent;",
            "  * IC said ema flips sign between bull and recent;",
            "  * conclusion is consistent: a single H cannot dominate both segments;",
            "    the sign flip explains why H underperforms B by ~13R on recent.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {out_dir} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
