#!/usr/bin/env python3
"""Offline lift: strong order-flow vs structure-only on FBF features_labeled.parquet.

Replays production prefilter + entry_filters (OR of branches), then adds absorption /
CVD divergence cuts. Includes shuffle-null for absorption to detect spurious lift.

Usage:
  python scripts/analyze_fbf_orderflow_lift.py \\
    results/train_final_20260411_061030_rr_extreme/fbf/features_labeled.parquet

  # 2024–2025 prepare 完成后（见 mlbot train final --prepare-only ...）:
  python scripts/analyze_fbf_orderflow_lift.py results/fbf_lift_2024_2025/fbf/features_labeled.parquet \\
    --datetime-min 2024-01-01 --datetime-max 2025-12-31 --out-json results/fbf_orderflow_lift_2024_2025.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

_OPS = {
    ">=": lambda s, v: s >= v,
    "<=": lambda s, v: s <= v,
    ">": lambda s, v: s > v,
    "<": lambda s, v: s < v,
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _mask_from_conditions(
    df: pd.DataFrame, conditions: List[Dict[str, Any]]
) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for cond in conditions:
        feat = cond["feature"]
        op = cond["operator"]
        val = float(cond["value"])
        if feat not in df.columns:
            return pd.Series(False, index=df.index)
        s = pd.to_numeric(df[feat], errors="coerce").astype(float)
        fn = _OPS.get(str(op))
        if fn is None:
            return pd.Series(False, index=df.index)
        m &= fn(s, val)
    return m


def _prefilter_mask(df: pd.DataFrame, rules: List[Dict[str, Any]]) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for r in rules:
        feat = r["feature"]
        op = r["operator"]
        val = float(r["value"])
        if feat not in df.columns:
            return pd.Series(False, index=df.index)
        s = pd.to_numeric(df[feat], errors="coerce").astype(float)
        m &= _OPS[str(op)](s, val)
    return m


def _semantic_core_mask(df: pd.DataFrame) -> pd.Series:
    """Prefilter caller must AND this. SR-failure semantics without OLS/dist/RSI (wider n for lift)."""
    short_m = _mask_from_conditions(
        df,
        [
            {
                "feature": "fer_sr_failed_breakout_direction_signed",
                "operator": "<=",
                "value": -1,
            },
            {"feature": "fer_range_pos_20", "operator": ">=", "value": 0.56},
            {"feature": "bars_since_local_high", "operator": "<=", "value": 0.30},
            {
                "feature": "fer_sr_failed_breakout_score",
                "operator": ">=",
                "value": 0.38,
            },
        ],
    )
    long_m = _mask_from_conditions(
        df,
        [
            {
                "feature": "fer_sr_failed_breakout_direction_signed",
                "operator": ">=",
                "value": 1,
            },
            {"feature": "fer_range_pos_20", "operator": "<=", "value": 0.44},
            {"feature": "bars_since_local_low", "operator": "<=", "value": 0.30},
            {
                "feature": "fer_sr_failed_breakout_score",
                "operator": ">=",
                "value": 0.38,
            },
        ],
    )
    return short_m | long_m


def _entry_or_mask(df: pd.DataFrame, entry_cfg: Dict[str, Any]) -> pd.Series:
    masks: List[pd.Series] = []
    for f in entry_cfg.get("filters", []):
        if not f.get("enabled", True):
            continue
        conds = f.get("conditions") or []
        if not conds:
            continue
        masks.append(_mask_from_conditions(df, conds))
    if not masks:
        return pd.Series(False, index=df.index)
    out = masks[0]
    for m in masks[1:]:
        out = out | m
    return out


def _summarize(rr: pd.Series) -> Tuple[int, float, float, float]:
    rr = pd.to_numeric(rr, errors="coerce").dropna()
    n = len(rr)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    pos = float((rr > 0).mean())
    return n, float(rr.mean()), float(rr.median()), pos


def main() -> None:
    ap = argparse.ArgumentParser(
        description="FBF order-flow lift on features_labeled.parquet"
    )
    ap.add_argument(
        "parquet",
        nargs="?",
        default="results/train_final_20260411_061030_rr_extreme/fbf/features_labeled.parquet",
    )
    ap.add_argument(
        "--prefilter-yaml",
        type=Path,
        default=Path("config/strategies/fbf/archetypes/prefilter.yaml"),
    )
    ap.add_argument(
        "--entry-yaml",
        type=Path,
        default=Path("config/strategies/fbf/archetypes/entry_filters.yaml"),
    )
    ap.add_argument(
        "--shuffle-repeats",
        type=int,
        default=80,
        help="null draws for absorption permute",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--baseline",
        choices=("entry", "semantic_core", "prefilter"),
        default="semantic_core",
        help="entry=full YAML OR; semantic_core=SR failure+range+bars (no OLS/dist/RSI); prefilter=prefilter only",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional JSON summary path (e.g. results/fbf_orderflow_lift.json)",
    )
    ap.add_argument(
        "--datetime-min",
        default=None,
        help="Inclusive lower bound on datetime/timestamp column (YYYY-MM-DD)",
    )
    ap.add_argument(
        "--datetime-max",
        default=None,
        help="Inclusive upper bound on datetime/timestamp column (YYYY-MM-DD)",
    )
    args = ap.parse_args()
    pq_path = Path(args.parquet)
    if not pq_path.is_file():
        print(f"Missing parquet: {pq_path}", file=sys.stderr)
        sys.exit(1)

    need = [
        "forward_rr",
        "success_no_rr_extreme",
        "fer_aggressor_absorption",
        "fer_absorption_streak",
        "cvd_divergence_score",
        "cvd_divergence_score_pct",
    ]
    df = pd.read_parquet(pq_path)
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    if args.datetime_min or args.datetime_max:
        dcol = next((c for c in ("datetime", "timestamp") if c in df.columns), None)
        if dcol is None:
            print(
                "⚠️  No datetime/timestamp column; ignoring --datetime-min/max",
                file=sys.stderr,
            )
        else:
            n0 = len(df)
            if args.datetime_min:
                lo = pd.Timestamp(args.datetime_min, tz="UTC")
                dt = pd.to_datetime(df[dcol], utc=True, errors="coerce")
                df = df.loc[dt >= lo]
            if args.datetime_max:
                hi_excl = pd.Timestamp(args.datetime_max, tz="UTC") + pd.Timedelta(
                    days=1
                )
                dt = pd.to_datetime(df[dcol], utc=True, errors="coerce")
                df = df.loc[dt < hi_excl]
            dt2 = pd.to_datetime(df[dcol], utc=True, errors="coerce")
            print(
                f"datetime filter ({dcol}): {n0} -> {len(df)} rows, range {dt2.min()} .. {dt2.max()}"
            )

    pf_cfg = _load_yaml(args.prefilter_yaml)
    entry_cfg = _load_yaml(args.entry_yaml)

    pf_mask = _prefilter_mask(df, pf_cfg.get("rules") or [])
    if args.baseline == "entry":
        extra = _entry_or_mask(df, entry_cfg)
    elif args.baseline == "semantic_core":
        extra = _semantic_core_mask(df)
    else:
        extra = pd.Series(True, index=df.index)
    baseline = pf_mask & extra

    rr = pd.to_numeric(df["forward_rr"], errors="coerce")
    n_pf = int(pf_mask.sum())
    n_base = int(baseline.sum())
    print(f"parquet={pq_path} rows={len(df)}")
    print(f"prefilter_pass={n_pf} ({100 * n_pf / len(df):.2f}%)")
    print(
        f"baseline({args.baseline})=prefilter AND ... => n={n_base} ({100 * n_base / len(df):.2f}%)"
    )

    base_n, base_mean, base_med, base_pos = _summarize(rr.loc[baseline])
    print(
        f"\nBaseline forward_rr: n={base_n} mean={base_mean:.5f} median={base_med:.5f} pos_rate={base_pos:.4f}"
    )

    abs_col = pd.to_numeric(df["fer_aggressor_absorption"], errors="coerce")
    streak_col = pd.to_numeric(df["fer_absorption_streak"], errors="coerce")
    cvd_pct = pd.to_numeric(df["cvd_divergence_score_pct"], errors="coerce")

    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    print("\n== +fer_aggressor_absorption >= t (within baseline population) ==")
    print("t\tn_sub\tfrac_of_base\tmean_rr\tlift_vs_base\tpos_rate")
    for t in thresholds:
        sub = baseline & (abs_col >= t)
        sn, sm, _, sp = _summarize(rr.loc[sub])
        frac = sn / base_n if base_n else 0.0
        lift = (
            sm - base_mean
            if np.isfinite(sm) and np.isfinite(base_mean)
            else float("nan")
        )
        print(f"{t}\t{sn}\t{frac:.4f}\t{sm:.5f}\t{lift:+.5f}\t{sp:.4f}")

    print("\n== +fer_absorption_streak >= t ==")
    print("t\tn_sub\tfrac_of_base\tmean_rr\tlift_vs_base\tpos_rate")
    for t in thresholds:
        sub = baseline & (streak_col >= t)
        sn, sm, _, sp = _summarize(rr.loc[sub])
        frac = sn / base_n if base_n else 0.0
        lift = (
            sm - base_mean
            if np.isfinite(sm) and np.isfinite(base_mean)
            else float("nan")
        )
        print(f"{t}\t{sn}\t{frac:.4f}\t{sm:.5f}\t{lift:+.5f}\t{sp:.4f}")

    print("\n== +cvd_divergence_score_pct >= t ==")
    print("t\tn_sub\tfrac_of_base\tmean_rr\tlift_vs_base\tpos_rate")
    for t in thresholds:
        sub = baseline & (cvd_pct >= t)
        sn, sm, _, sp = _summarize(rr.loc[sub])
        frac = sn / base_n if base_n else 0.0
        lift = (
            sm - base_mean
            if np.isfinite(sm) and np.isfinite(base_mean)
            else float("nan")
        )
        print(f"{t}\t{sn}\t{frac:.4f}\t{sm:.5f}\t{lift:+.5f}\t{sp:.4f}")

    print(
        "\n== cvd_divergence_score_pct tail: n_sub on baseline (variance vs sample) =="
    )
    for t in (0.50, 0.52, 0.54, 0.55, 0.56, 0.58):
        subn = int((baseline & (cvd_pct >= float(t))).sum())
        print(f"  >= {t}: n_sub={subn}")

    print("\n== combined: absorption>=0.5 AND streak>=0.45 AND cvd_div_pct>=0.4 ==")
    comb = baseline & (abs_col >= 0.5) & (streak_col >= 0.45) & (cvd_pct >= 0.4)
    sn, sm, _, sp = _summarize(rr.loc[comb])
    frac = sn / base_n if base_n else 0.0
    lift = (
        sm - base_mean if np.isfinite(sm) and np.isfinite(base_mean) else float("nan")
    )
    print(
        f"n={sn} frac_of_base={frac:.4f} mean_rr={sm:.5f} lift={lift:+.5f} pos_rate={sp:.4f}"
    )

    # Shuffle null: permute absorption only on baseline rows, fixed t=0.5
    rng = np.random.default_rng(args.seed)
    pos = np.flatnonzero(baseline.fillna(False).to_numpy())
    abs_arr = pd.to_numeric(
        df["fer_aggressor_absorption"].iloc[pos], errors="coerce"
    ).to_numpy(dtype=float)
    rr_arr = pd.to_numeric(df["forward_rr"].iloc[pos], errors="coerce").to_numpy(
        dtype=float
    )
    ok = np.isfinite(abs_arr) & np.isfinite(rr_arr)
    abs_b = abs_arr[ok]
    rr_b = rr_arr[ok]
    n0 = int(abs_b.size)
    if n0 < 50:
        print(
            "\n(shuffle null skipped: too few baseline rows with non-null absorption)"
        )
    else:
        t_fix = 0.5
        real_pass = abs_b >= t_fix
        real_lift = (
            float(rr_b[real_pass].mean() - rr_b.mean())
            if real_pass.any()
            else float("nan")
        )
        lifts = []
        for _ in range(args.shuffle_repeats):
            perm = rng.permutation(abs_b)
            pass_perm = perm >= t_fix
            if not pass_perm.any():
                continue
            lifts.append(float(rr_b[pass_perm].mean() - rr_b.mean()))
        lifts_arr = np.array(lifts, dtype=float)
        q = float(np.mean(lifts_arr >= real_lift)) if len(lifts_arr) else float("nan")
        print(
            f"\n== Shuffle null (absorption permuted within baseline, t={t_fix}, n={n0}) =="
        )
        print(
            f"observed_lift={real_lift:+.5f}  null_mean_lift={lifts_arr.mean():+.5f}  p_empirical(one-sided)={q:.4f}"
        )

    # Label lift (success_no_rr_extreme)
    y = pd.to_numeric(df["success_no_rr_extreme"], errors="coerce")
    if y.notna().sum() > 100:
        print(
            "\n== success_no_rr_extreme rate: baseline vs absorption>=0.5 on baseline =="
        )
        br = float(y.loc[baseline].dropna().mean())
        sub = baseline & (abs_col >= 0.5)
        sr = float(y.loc[sub].dropna().mean()) if sub.any() else float("nan")
        print(
            f"baseline_pos_rate={br:.4f}  with_abs>=0.5_pos_rate={sr:.4f}  delta={sr - br:+.4f}"
        )

    # Bar-level A/B (proxy for event_backtest when n is small)
    t_cvd = 0.45
    b_mask = baseline & (cvd_pct >= t_cvd)
    bn, bm, bmed, bp = _summarize(rr.loc[b_mask])
    print(
        f"\n== Bar-level A/B proxy: B = baseline & cvd_divergence_score_pct>={t_cvd} =="
    )
    print(
        f"B n={bn} mean_rr={bm:.5f} median={bmed:.5f} pos_rate={bp:.4f} lift_vs_baseline={(bm - base_mean):+.5f}"
    )

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        import json

        row = {
            "parquet": str(pq_path),
            "baseline_mode": args.baseline,
            "n_prefilter": n_pf,
            "n_baseline": base_n,
            "baseline_mean_rr": base_mean,
            "bar_level_b_cvd_pct_ge": t_cvd,
            "n_b": bn,
            "mean_rr_b": bm,
            "lift_b": (
                float(bm - base_mean)
                if np.isfinite(bm) and np.isfinite(base_mean)
                else None
            ),
        }
        out.write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
