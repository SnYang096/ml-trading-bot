"""Decompose FBF-strict gates and show which is the most binding.

Counts how many 2H bars (across all 6 symbols, 2023-09..2024-12) pass each
prefilter / entry_filter condition independently and cumulatively, so we know
which knob to loosen to increase trade frequency.

Usage:
    python scripts/diag_fbf_strict_gate_funnel.py \
        --feature-store feature_store/features_fbf_120T_06702ab6f8 \
        --start 2023-09-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]

FEATURES_NEEDED = [
    "fer_ols_pos",
    "fer_sr_failed_breakout_score",
    "fer_sr_failed_breakout_direction_signed",
    "fer_efficiency_flip_strength",
    "fer_aggressor_absorption",
    "fer_range_pos_20",
    "sr_strength_max",
    "fer_ols_width_norm",
    "trend_r2_20",
]


def load_symbol(
    store: str, symbol: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    files = sorted(glob.glob(f"{store}/{symbol}/120T/*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        ym = Path(f).stem  # e.g. 2023-09
        try:
            t = pd.Timestamp(ym + "-01", tz="UTC")
        except Exception:
            continue
        if t < start - pd.Timedelta(days=35) or t > end + pd.Timedelta(days=5):
            continue
        dfs.append(pd.read_parquet(f))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[(df.index >= start) & (df.index < end)]
    missing = [c for c in FEATURES_NEEDED if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    return df[FEATURES_NEEDED]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--feature-store",
        default="feature_store/features_fbf_120T_06702ab6f8",
    )
    ap.add_argument("--start", default="2023-09-01")
    ap.add_argument("--end", default="2025-01-01")
    args = ap.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    frames = []
    for sym in SYMBOLS:
        d = load_symbol(args.feature_store, sym, start, end)
        if not d.empty:
            d["symbol"] = sym
            frames.append(d)
    df = pd.concat(frames)
    n = len(df)
    print(
        f"[info] total 2H bars: {n}  (symbols={len(frames)}, period {args.start} .. {args.end})"
    )
    print()

    # === prefilter conditions (strict current) ===
    pf = pd.DataFrame(index=df.index)
    pf["pf1_ols_boundary"] = (df["fer_ols_pos"] >= 0.90) | (df["fer_ols_pos"] <= 0.10)
    pf["pf2_fbs_ge_038"] = df["fer_sr_failed_breakout_score"] >= 0.38
    pf["pf3_sr_strength_ge_052"] = df["sr_strength_max"] >= 0.52
    pf["pf4_ols_width_ge_022"] = df["fer_ols_width_norm"] >= 0.22
    pf["pf5_trend_r2_le_050"] = df["trend_r2_20"] <= 0.50

    # === entry_filter conditions (strict) ===
    dirn = df["fer_sr_failed_breakout_direction_signed"]
    ef = pd.DataFrame(index=df.index)
    ef["ef_dir_long"] = dirn >= 1
    ef["ef_dir_short"] = dirn <= -1
    ef["ef_flip_080"] = df["fer_efficiency_flip_strength"] >= 0.80
    ef["ef_absorp_076"] = df["fer_aggressor_absorption"] >= 0.76
    ef["ef_exhaustion_any"] = ef["ef_flip_080"] | ef["ef_absorp_076"]
    ef["ef_ols_low"] = df["fer_ols_pos"] <= 0.10
    ef["ef_ols_high"] = df["fer_ols_pos"] >= 0.90

    # long entry filter combo (any of flip | absorp)
    ef["ef_long"] = (
        ef["ef_ols_low"]
        & ef["ef_dir_long"]
        & pf["pf2_fbs_ge_038"]
        & ef["ef_exhaustion_any"]
    )
    ef["ef_short"] = (
        ef["ef_ols_high"]
        & ef["ef_dir_short"]
        & pf["pf2_fbs_ge_038"]
        & ef["ef_exhaustion_any"]
    )
    ef["ef_any"] = ef["ef_long"] | ef["ef_short"]

    def rate(s: pd.Series) -> str:
        return f"{100*s.mean():6.2f}%  ({int(s.sum()):>6d})"

    print("=== Prefilter gates (independent pass rate) ===")
    for c in [
        "pf1_ols_boundary",
        "pf2_fbs_ge_038",
        "pf3_sr_strength_ge_052",
        "pf4_ols_width_ge_022",
        "pf5_trend_r2_le_050",
    ]:
        print(f"  {c:<30s} {rate(pf[c])}")
    print()

    print("=== Prefilter cumulative funnel (AND stack in order) ===")
    stack = pd.Series(True, index=df.index)
    for c in [
        "pf1_ols_boundary",
        "pf2_fbs_ge_038",
        "pf3_sr_strength_ge_052",
        "pf4_ols_width_ge_022",
        "pf5_trend_r2_le_050",
    ]:
        stack = stack & pf[c]
        print(f"  + {c:<30s} → {rate(stack)}")
    pf_pass = stack
    print()

    print("=== Entry-filter conditions (independent, on ALL bars) ===")
    for c in [
        "ef_dir_long",
        "ef_dir_short",
        "ef_flip_080",
        "ef_absorp_076",
        "ef_exhaustion_any",
        "ef_ols_low",
        "ef_ols_high",
    ]:
        print(f"  {c:<22s} {rate(ef[c])}")
    print()

    print("=== Entry-filter combined (requires all sub-conditions) ===")
    print(f"  ef_long                {rate(ef['ef_long'])}")
    print(f"  ef_short               {rate(ef['ef_short'])}")
    print(f"  ef_any                 {rate(ef['ef_any'])}")
    print()

    print("=== Final: prefilter AND entry_filter (strict current) ===")
    final = pf_pass & ef["ef_any"]
    print(f"  strict_firing          {rate(final)}")
    print(f"  (long)                 {rate(pf_pass & ef['ef_long'])}")
    print(f"  (short)                {rate(pf_pass & ef['ef_short'])}")
    print()

    # === Sweeps ===
    print("=== Sweep: fer_ols_pos window (Plan A) while keeping other gates ===")
    print("  LONG requires fer_ols_pos in [lo, hi]; SHORT in [1-hi, 1-lo]")
    for lo, hi in [
        (0.00, 0.10),
        (0.05, 0.25),
        (0.05, 0.30),
        (0.00, 0.25),
        (0.10, 0.25),
        (0.00, 0.30),
    ]:
        long_ok = (
            (df["fer_ols_pos"] >= lo) & (df["fer_ols_pos"] <= hi) & ef["ef_dir_long"]
        )
        short_ok = (
            (df["fer_ols_pos"] >= (1 - hi))
            & (df["fer_ols_pos"] <= (1 - lo))
            & ef["ef_dir_short"]
        )
        pf_alt = (
            pf["pf2_fbs_ge_038"]
            & pf["pf3_sr_strength_ge_052"]
            & pf["pf4_ols_width_ge_022"]
            & pf["pf5_trend_r2_le_050"]
        )
        final_alt = pf_alt & ef["ef_exhaustion_any"] & (long_ok | short_ok)
        print(
            f"  OLS∈[{lo:.2f},{hi:.2f}]  → bars {rate(final_alt)}  | long-only {rate(pf_alt & ef['ef_exhaustion_any'] & long_ok)}"
            f"  | short-only {rate(pf_alt & ef['ef_exhaustion_any'] & short_ok)}"
        )
    print()

    print("=== Sweep: add range_pos_20 short-term exhaustion (Plan B) ===")
    print("  LONG also requires range_pos_20 <= R; SHORT range_pos_20 >= 1-R")
    for R in [0.30, 0.25, 0.20, 0.15]:
        rp = df["fer_range_pos_20"]
        long_rp = rp <= R
        short_rp = rp >= (1 - R)
        base_long = pf_pass & ef["ef_long"] & long_rp
        base_short = pf_pass & ef["ef_short"] & short_rp
        final_alt = base_long | base_short
        print(
            f"  range_pos≤{R:.2f} / ≥{1-R:.2f}  → {rate(final_alt)}  "
            f"| long {rate(base_long)} | short {rate(base_short)}"
        )
    print()

    print("=== Sweep: Plan A (OLS window) + Plan B (range_pos_20) combined ===")
    for lo, hi in [(0.05, 0.25), (0.00, 0.25), (0.05, 0.30)]:
        for R in [0.30, 0.25]:
            rp = df["fer_range_pos_20"]
            long_ok = (
                (df["fer_ols_pos"] >= lo)
                & (df["fer_ols_pos"] <= hi)
                & ef["ef_dir_long"]
                & (rp <= R)
            )
            short_ok = (
                (df["fer_ols_pos"] >= (1 - hi))
                & (df["fer_ols_pos"] <= (1 - lo))
                & ef["ef_dir_short"]
                & (rp >= (1 - R))
            )
            pf_alt = (
                pf["pf2_fbs_ge_038"]
                & pf["pf3_sr_strength_ge_052"]
                & pf["pf4_ols_width_ge_022"]
                & pf["pf5_trend_r2_le_050"]
            )
            final_alt = pf_alt & ef["ef_exhaustion_any"] & (long_ok | short_ok)
            print(f"  OLS∈[{lo:.2f},{hi:.2f}] × range_pos≤{R:.2f}  → {rate(final_alt)}")
    print()

    # === Extra diagnostics on the real bottlenecks ===
    print("=== fer_sr_failed_breakout_score distribution ===")
    s = df["fer_sr_failed_breakout_score"].dropna()
    qs = [0.50, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99]
    print("  n=", len(s))
    for q in qs:
        print(f"   q{int(q*1000)/10:<5}  {s.quantile(q):.3f}")
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.38]:
        print(
            f"  score≥{thr:.2f}  pass_rate={100*(s>=thr).mean():.2f}%  ({int((s>=thr).sum())})"
        )
    print()

    print("=== fer_sr_failed_breakout_direction_signed value counts ===")
    print(
        df["fer_sr_failed_breakout_direction_signed"]
        .value_counts(dropna=False)
        .head(10)
    )
    print()

    print("=== Sweep: lower FBS score threshold (keep everything else strict) ===")
    for fbs_thr in [0.38, 0.30, 0.25, 0.20, 0.15]:
        pf2_alt = df["fer_sr_failed_breakout_score"] >= fbs_thr
        pf_alt = (
            pf["pf1_ols_boundary"]
            & pf2_alt
            & pf["pf3_sr_strength_ge_052"]
            & pf["pf4_ols_width_ge_022"]
            & pf["pf5_trend_r2_le_050"]
        )
        ef_long_alt = (
            ef["ef_ols_low"] & ef["ef_dir_long"] & pf2_alt & ef["ef_exhaustion_any"]
        )
        ef_short_alt = (
            ef["ef_ols_high"] & ef["ef_dir_short"] & pf2_alt & ef["ef_exhaustion_any"]
        )
        final_alt = pf_alt & (ef_long_alt | ef_short_alt)
        print(f"  FBS≥{fbs_thr:.2f}  → strict_firing {rate(final_alt)}")
    print()

    print(
        "=== Sweep: drop direction_signed anchor, use OLS side to imply direction ==="
    )
    print("  (keep FBS score + exhaustion + ols_boundary; drop direction_signed)")
    for fbs_thr in [0.38, 0.30, 0.25]:
        pf2_alt = df["fer_sr_failed_breakout_score"] >= fbs_thr
        pf_alt = (
            pf["pf1_ols_boundary"]
            & pf2_alt
            & pf["pf3_sr_strength_ge_052"]
            & pf["pf4_ols_width_ge_022"]
            & pf["pf5_trend_r2_le_050"]
        )
        # no direction_signed anchor
        ef_long_alt = ef["ef_ols_low"] & pf2_alt & ef["ef_exhaustion_any"]
        ef_short_alt = ef["ef_ols_high"] & pf2_alt & ef["ef_exhaustion_any"]
        final_alt = pf_alt & (ef_long_alt | ef_short_alt)
        print(
            f"  FBS≥{fbs_thr:.2f}, no-dir  →  {rate(final_alt)}  "
            f"| long {rate(pf_alt & ef_long_alt)}  | short {rate(pf_alt & ef_short_alt)}"
        )
    print()

    print("=== Sweep: Plan A (OLS widen) + drop direction + lower FBS ===")
    for fbs_thr in [0.38, 0.30, 0.25]:
        for lo, hi in [(0.00, 0.10), (0.00, 0.20), (0.00, 0.25)]:
            pf2_alt = df["fer_sr_failed_breakout_score"] >= fbs_thr
            pf_alt = (
                pf2_alt
                & pf["pf3_sr_strength_ge_052"]
                & pf["pf4_ols_width_ge_022"]
                & pf["pf5_trend_r2_le_050"]
            )
            ols_long = (df["fer_ols_pos"] >= lo) & (df["fer_ols_pos"] <= hi)
            ols_short = (df["fer_ols_pos"] >= (1 - hi)) & (
                df["fer_ols_pos"] <= (1 - lo)
            )
            final_alt = pf_alt & ef["ef_exhaustion_any"] & (ols_long | ols_short)
            print(f"  FBS≥{fbs_thr:.2f}, OLS∈[{lo:.2f},{hi:.2f}]  →  {rate(final_alt)}")


if __name__ == "__main__":
    main()
