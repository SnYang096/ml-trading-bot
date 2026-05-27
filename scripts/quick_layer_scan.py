#!/usr/bin/env python3
"""Quick layer scan — R&D 第一步：离线扫各层候选特征 + 阈值 plateau。

设计哲学（见 docs/strategy/WORKFLOW_整体架构与管线改进计划_CN.md §4 + §5）:
    - 这是 R&D / Q-级假设生成的入口，不修改任何 yaml；
    - 任何"看上去显著"的特征，下一步必须 event_backtest 看 R-multiple；
    - 输出 markdown 报告 + 同时打印一个紧凑摘要，便于在终端快速判断。

支持四种扫描模式:
    --feature-plateau  : 对单个特征在某 label 上做阈值 plateau 扫描
    --condition-set    : 比较若干 regime 条件（all_of 形式）对 label 的效应
    --pair-scan        : 两两特征 (feature_a, feature_b) 的联合 deny 效应（粗糙双变量）
    ic-decay           : 特征 × horizon 的 Spearman IC（可选对比 baseline JSON）

CLI 示例:
    # pullback 在 bull 子样本上的最优阈值
    python scripts/quick_layer_scan.py feature-plateau \
        --features-parquet results/<...>/features_labeled.parquet \
        --label success_no_rr_extreme \
        --feature tpc_pullback_depth --operator le \
        --grid 0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95 \
        --filter "tpc_semantic_chop<=0.4" "ema_1200_position>=0.10" \
        --calendar-window 2024-01-01,2025-01-01 \
        --out results/tpc/quick_scan/pullback_in_bull.md

    # regime 候选：H / F / F' / F'' 对比
    python scripts/quick_layer_scan.py condition-set \
        --features-parquet ... --label success_no_rr_extreme \
        --filter "tpc_semantic_chop<=0.4" \
        --condition "H: abs(ema_1200_position)>0.10" \
        --condition "F: abs(ema_1200_position)>0.12" \
        --condition "Fp: abs(ema_1200_position)>0.10 AND abs(ema_1200_slope_10)>0.002" \
        --out results/tpc/quick_scan/regime_candidates.md

Exit codes:
    0  正常
    3  输入错误
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.research.expr import OPS as _OPS
from src.research.expr import build_calendar_mask as _build_calendar_mask
from src.research.expr import parse_clause as _parse_clause
from src.research.stat_kernels.ic import ic_decay_rows
from src.research.stat_kernels.z_test import two_proportion_z as _binary_proportions_z

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def feature_plateau_payload(
    args: argparse.Namespace, df: pd.DataFrame, label: pd.Series, base_mask: pd.Series
) -> dict:
    """Scan threshold grid; return JSON-serializable payload for calibrate chain."""
    feature = args.feature
    op = args.operator
    grid = [float(x) for x in args.grid.split(",")]
    if feature not in df.columns:
        raise KeyError(f"Feature missing: {feature}")
    s = pd.to_numeric(df[feature], errors="coerce")
    valid = s.notna() & base_mask
    s = s[valid]
    y = label[valid]

    row_dicts: List[dict] = []
    base_succ = float(y.mean()) if len(y) else float("nan")
    for thr in grid:
        m = _OPS[op](s, thr)
        n_hit = int(m.sum())
        if n_hit == 0:
            row_dicts.append(
                {
                    "threshold": float(thr),
                    "n_hit": 0,
                    "succ_hit": None,
                    "succ_other": None,
                    "z": 0.0,
                }
            )
            continue
        n_oth = int((~m).sum())
        p_hit = float(y[m].mean())
        p_oth = float(y[~m].mean()) if n_oth else float("nan")
        z = _binary_proportions_z(p_hit, n_hit, p_oth, n_oth) if n_oth else 0.0
        row_dicts.append(
            {
                "threshold": float(thr),
                "n_hit": n_hit,
                "succ_hit": p_hit,
                "succ_other": None if pd.isna(p_oth) else float(p_oth),
                "z": float(z),
            }
        )

    eligible = [r for r in row_dicts if r["n_hit"] > 0 and r["succ_hit"] is not None]
    best = max(eligible, key=lambda r: abs(r["z"]), default=None) if eligible else None
    rec = float(best["threshold"]) if best else None

    plateau_start: float | None = None
    plateau_end: float | None = None
    if len(eligible) >= 2:
        sorted_rows = sorted(eligible, key=lambda r: r["threshold"])
        anchor_succ = sorted_rows[0]["succ_hit"]
        run_start = sorted_rows[0]["threshold"]
        run_end = run_start
        best_width = 0.0
        best_mid: float | None = None
        for row in sorted_rows:
            if anchor_succ is None or row["succ_hit"] is None:
                continue
            if abs(row["succ_hit"] - anchor_succ) <= 0.10:
                run_end = row["threshold"]
            else:
                width = run_end - run_start
                if width > best_width:
                    best_width = width
                    plateau_start = run_start
                    plateau_end = run_end
                    best_mid = (run_start + run_end) / 2
                anchor_succ = row["succ_hit"]
                run_start = row["threshold"]
                run_end = run_start
        width = run_end - run_start
        if width >= best_width and width > 0:
            plateau_start = run_start
            plateau_end = run_end
            best_mid = (run_start + run_end) / 2
        if best_mid is not None and rec is None:
            rec = float(best_mid)

    return {
        "feature": feature,
        "operator": op,
        "base_n": int(len(y)),
        "base_success": base_succ,
        "recommended": rec,
        "mid": rec,
        "plateau_start": plateau_start,
        "plateau_end": plateau_end,
        "rows": row_dicts,
    }


def mode_feature_plateau(
    args: argparse.Namespace, df: pd.DataFrame, label: pd.Series, base_mask: pd.Series
) -> str:
    payload = feature_plateau_payload(args, df, label, base_mask)
    feature = payload["feature"]
    op = payload["operator"]
    base_succ = payload["base_success"]
    base_n = payload["base_n"]

    md = [f"# feature_plateau · {feature} {op} ?", ""]
    md.append(f"- base n = {base_n}, base_success = {base_succ:.3%}")
    md.append("")
    md.append("| threshold | n_hit | succ_hit | succ_other | |z| |")
    md.append("|---:|---:|---:|---:|---:|")
    for row in payload["rows"]:
        p_hit = row["succ_hit"]
        p_oth = row["succ_other"]
        z = row["z"]
        ph = "nan" if p_hit is None or pd.isna(p_hit) else f"{p_hit:.3%}"
        po = "nan" if p_oth is None or pd.isna(p_oth) else f"{p_oth:.3%}"
        md.append(
            f"| {row['threshold']:.3g} | {row['n_hit']} | {ph} | {po} | {z:.2f} |"
        )
    md.append("")
    md.append(
        "**Plateau interpretation**: consecutive thresholds with similar succ_hit and |z|<2.0 → effect is noise; if all thresholds give ~same succ_hit, **feature is not the bottleneck on this subset**."
    )
    return "\n".join(md)


def mode_condition_set(
    args: argparse.Namespace, df: pd.DataFrame, label: pd.Series, base_mask: pd.Series
) -> str:
    base_y = label[base_mask]
    base_n = int(base_mask.sum())
    base_succ = float(base_y.mean()) if base_n else float("nan")

    rows: List[Tuple[str, int, float, float, float]] = []
    for raw in args.condition:
        if ":" in raw:
            name, expr = raw.split(":", 1)
            name = name.strip()
            expr = expr.strip()
        else:
            name, expr = raw, raw
        cond_fn = _parse_clause(expr)
        cond = cond_fn(df) & base_mask
        n = int(cond.sum())
        if n == 0:
            rows.append((name, 0, float("nan"), float("nan"), 0.0))
            continue
        succ = float(label[cond].mean())
        other = base_mask & ~cond
        n_oth = int(other.sum())
        succ_oth = float(label[other].mean()) if n_oth else float("nan")
        z = _binary_proportions_z(succ, n, succ_oth, n_oth) if n_oth else 0.0
        rows.append((name, n, succ, succ_oth, z))

    md = ["# condition_set scan", ""]
    md.append(f"- base mask n = {base_n}, base_success = {base_succ:.3%}")
    md.append("")
    md.append("| condition | n | succ_in | succ_out | Δpp vs base | |z| |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for name, n, s_in, s_out, z in rows:
        si = "nan" if pd.isna(s_in) else f"{s_in:.3%}"
        so = "nan" if pd.isna(s_out) else f"{s_out:.3%}"
        delta = "nan" if pd.isna(s_in) else f"{(s_in - base_succ) * 100:+.2f}"
        md.append(f"| {name} | {n} | {si} | {so} | {delta} | {z:.2f} |")
    md.append("")
    md.append(
        "**Reading**: positive Δpp + |z|>2 ⇒ condition改善 label，可考虑做 backtest 验证 R-multiple；|z|<2 即使方向对也属 noise。"
    )
    return "\n".join(md)


def mode_pair_scan(
    args: argparse.Namespace, df: pd.DataFrame, label: pd.Series, base_mask: pd.Series
) -> str:
    """Coarse two-feature deny scan: for each pair (fa op va, fb op vb), measure succ in joint mask."""
    a_feat, a_op, a_grid = args.pair_a.split(":")
    b_feat, b_op, b_grid = args.pair_b.split(":")
    a_grid = [float(x) for x in a_grid.split(",")]
    b_grid = [float(x) for x in b_grid.split(",")]

    base_n = int(base_mask.sum())
    base_succ = float(label[base_mask].mean()) if base_n else float("nan")

    md = [f"# pair_scan · {a_feat} {a_op} grid × {b_feat} {b_op} grid", ""]
    md.append(f"- base n = {base_n}, base_success = {base_succ:.3%}")
    md.append("")
    header = ["thr_a \\ thr_b"] + [f"{b:g}" for b in b_grid]
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join(["---:"] * len(header)) + "|")
    for ta in a_grid:
        cells = [f"**{ta:g}**"]
        for tb in b_grid:
            ma = _OPS[a_op](pd.to_numeric(df[a_feat], errors="coerce"), ta).fillna(
                False
            )
            mb = _OPS[b_op](pd.to_numeric(df[b_feat], errors="coerce"), tb).fillna(
                False
            )
            m = ma & mb & base_mask
            n = int(m.sum())
            if n == 0:
                cells.append("—")
                continue
            succ = float(label[m].mean())
            cells.append(f"{succ:.1%}<br/>n={n}")
        md.append("| " + " | ".join(cells) + " |")
    return "\n".join(md)


# ---------------------------------------------------------------------------
# ic-decay
# ---------------------------------------------------------------------------


def mode_ic_decay(
    args: argparse.Namespace,
    df: pd.DataFrame,
    base_mask: pd.Series,
) -> str:
    features = [f.strip() for f in args.features.split(",") if f.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    target = args.target

    baseline_map: Dict[Tuple[str, str, int], float] = {}
    if args.baseline_json:
        bp = Path(args.baseline_json)
        if not bp.is_absolute():
            bp = (PROJECT_ROOT / bp).resolve()
        if bp.exists():
            blob = json.loads(bp.read_text(encoding="utf-8"))
            for row in blob.get("rows", []):
                if row.get("bucket") != "all":
                    continue
                feat = str(row.get("feature", ""))
                baseline_map[(feat, "all", 0)] = float(row.get("rank_ic", 0))

    md = ["# IC decay scan", ""]
    md.append(f"- target base: `{target}`")
    md.append(f"- horizons: {horizons}")
    md.append(f"- n_base (after filters): {int(base_mask.sum())}")
    md.append("")
    md.append(
        "| feature | horizon | target_col | n | rank_ic | p_value | "
        "baseline_ic | delta | sign_flip |"
    )
    md.append("|---:|---:|---|---:|---:|---:|---:|---:|:---:|")

    rows = ic_decay_rows(df, features, horizons, target, mask=base_mask, min_n=100)
    for row in rows:
        feat = row["feature"]
        h = row["horizon"]
        if h is None:
            md.append(f"| {feat} | — | — | 0 | — | — | — | — | — |")
            continue
        tcol = row["target_col"]
        n = row["n"]
        if tcol in (None, "missing") or n == 0:
            md.append(f"| {feat} | {h} | missing | 0 | — | — | — | — | — |")
            continue
        rho = row["rank_ic"]
        p = row["p_value"]
        base_ic = baseline_map.get((feat, "all", 0))
        delta_s = ""
        flip = ""
        if base_ic is not None and not np.isnan(rho):
            delta = rho - base_ic
            delta_s = f"{delta:+.4f}"
            if base_ic * rho < 0 and abs(rho) > 0.02 and abs(base_ic) > 0.02:
                flip = "yes"
        base_s = f"{base_ic:.4f}" if base_ic is not None else "—"
        rho_s = f"{rho:.4f}" if not np.isnan(rho) else "—"
        p_s = f"{p:.2e}" if not np.isnan(p) else "—"
        md.append(
            f"| {feat} | {h} | `{tcol}` | {n} | {rho_s} | {p_s} | "
            f"{base_s} | {delta_s} | {flip} |"
        )
    md.append("")
    md.append(
        "**Reading**: sign_flip=yes with |IC|>0.02 suggests regime drift; "
        "confirm with event_backtest before yaml changes."
    )
    return "\n".join(md)


# ---------------------------------------------------------------------------
# bucket-by (ema / calendar / feature_quantile)
# ---------------------------------------------------------------------------


def _parse_bucket_by(
    spec: str,
) -> List[Tuple[str, Callable[[pd.DataFrame], pd.Series]]]:
    """Parse ``--bucket-by`` into named bucket masks.

    Formats:
        ema:ema_1200_position@0.10
            → ``|ema|>=0.10`` vs ``|ema|<0.10``
        calendar:2024-01-01,2025-01-01;2025-01-01,2026-04-01
            → one bucket per semicolon-separated window
        feature_quantile:vol_persistence@4
            → quartile buckets Q1..Q4 on ``vol_persistence``
    """
    if ":" not in spec:
        raise ValueError(f"bucket-by must be kind:args, got {spec!r}")
    kind, rest = spec.split(":", 1)
    kind = kind.strip().lower()
    if kind == "ema":
        if "@" not in rest:
            raise ValueError(
                "ema bucket needs feature@threshold, e.g. ema_1200_position@0.10"
            )
        feat, thr_s = rest.rsplit("@", 1)
        feat = feat.strip()
        thr = float(thr_s.strip())

        def _ge(df: pd.DataFrame) -> pd.Series:
            s = pd.to_numeric(df[feat], errors="coerce").abs()
            return (s >= thr).fillna(False)

        def _lt(df: pd.DataFrame) -> pd.Series:
            s = pd.to_numeric(df[feat], errors="coerce").abs()
            return (s < thr).fillna(False)

        return [
            (f"|{feat}|>={thr:g}", _ge),
            (f"|{feat}|<{thr:g}", _lt),
        ]
    if kind == "calendar":
        windows = [w.strip() for w in rest.split(";") if w.strip()]
        out: List[Tuple[str, Callable[[pd.DataFrame], pd.Series]]] = []
        for w in windows:
            if "," not in w:
                raise ValueError(f"calendar window needs start,end: {w!r}")
            start_s, end_s = [x.strip() for x in w.split(",", 1)]

            def _cal(
                df: pd.DataFrame,
                *,
                _w: str = w,
                _start: str = start_s,
                _end: str = end_s,
            ) -> pd.Series:
                return _build_calendar_mask(df, f"{_start},{_end}")

            out.append((w.replace(",", " → "), _cal))
        return out
    if kind == "feature_quantile":
        if "@" not in rest:
            raise ValueError(
                "feature_quantile needs feature@n_bins, e.g. vol_persistence@4"
            )
        feat, n_s = rest.rsplit("@", 1)
        feat = feat.strip()
        n_bins = int(n_s.strip())
        if n_bins < 2:
            raise ValueError("feature_quantile n_bins must be >= 2")

        def _q_labels(df: pd.DataFrame) -> pd.Series:
            s = pd.to_numeric(df[feat], errors="coerce")
            try:
                return pd.qcut(s, q=n_bins, duplicates="drop")
            except ValueError:
                return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

        # Build per-interval masks lazily via unique categories on first df pass
        return [("__quantile__", lambda d, f=feat, nb=n_bins: _q_labels(d))]  # type: ignore

    raise ValueError(
        f"unknown bucket-by kind {kind!r}; use ema, calendar, or feature_quantile"
    )


def _expand_quantile_buckets(
    df: pd.DataFrame, feat: str, n_bins: int
) -> List[Tuple[str, Callable[[pd.DataFrame], pd.Series]]]:
    s = pd.to_numeric(df[feat], errors="coerce")
    try:
        cats = pd.qcut(s, q=n_bins, duplicates="drop")
    except ValueError:
        return []
    buckets: List[Tuple[str, Callable[[pd.DataFrame], pd.Series]]] = []
    for cat in cats.cat.categories:
        label = f"{feat} in {cat}"

        def _mask(d: pd.DataFrame, *, _cat=cat, _feat=feat, _nb=n_bins) -> pd.Series:
            qs = pd.to_numeric(d[_feat], errors="coerce")
            qc = pd.qcut(qs, q=_nb, duplicates="drop")
            return qc == _cat

        buckets.append((label, _mask))
    return buckets


def _resolve_bucket_masks(
    df: pd.DataFrame, spec: str
) -> List[Tuple[str, Callable[[pd.DataFrame], pd.Series]]]:
    parsed = _parse_bucket_by(spec)
    if len(parsed) == 1 and parsed[0][0] == "__quantile__":
        feat_part = spec.split(":", 1)[1]
        feat, n_s = feat_part.rsplit("@", 1)
        return _expand_quantile_buckets(df, feat.strip(), int(n_s.strip()))
    return parsed


def _run_mode_report(
    args: argparse.Namespace,
    df: pd.DataFrame,
    label: pd.Series,
    base_mask: pd.Series,
) -> str:
    if args.mode == "feature-plateau":
        return mode_feature_plateau(args, df, label, base_mask)
    if args.mode == "condition-set":
        return mode_condition_set(args, df, label, base_mask)
    if args.mode == "pair-scan":
        return mode_pair_scan(args, df, label, base_mask)
    if args.mode == "ic-decay":
        return mode_ic_decay(args, df, base_mask)
    raise ValueError(f"unknown mode {args.mode}")


def _bucketed_report(
    args: argparse.Namespace,
    df: pd.DataFrame,
    label: pd.Series,
    base_mask: pd.Series,
    bucket_spec: str,
) -> str:
    buckets = _resolve_bucket_masks(df, bucket_spec)
    sections: List[str] = [
        f"# {args.mode} scan (bucket-by: `{bucket_spec}`)",
        "",
    ]
    for bname, bfn in buckets:
        bmask = base_mask & bfn(df)
        n = int(bmask.sum())
        sections.append(f"## Bucket: {bname}")
        sections.append("")
        sections.append(f"- rows in bucket (after filters): {n}")
        sections.append("")
        if n == 0:
            sections.append("_(empty bucket — skip)_")
            sections.append("")
            continue
        body = _run_mode_report(args, df, label, bmask)
        # Drop duplicate top-level title from sub-report
        lines = body.splitlines()
        if lines and lines[0].startswith("# "):
            lines = lines[1:]
            while lines and not lines[0].strip():
                lines = lines[1:]
        sections.extend(lines)
        sections.append("")
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quick layer scan (R&D entry tool)")
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--features-parquet", required=True)
    common.add_argument("--label", default="success_no_rr_extreme")
    common.add_argument(
        "--filter",
        nargs="*",
        default=[],
        help="Pre-conditions joined with AND (e.g. 'tpc_semantic_chop<=0.4').",
    )
    common.add_argument(
        "--calendar-window", default=None, help="ISO datetimes 'start,end' (UTC)."
    )
    common.add_argument(
        "--out", default=None, help="Markdown output path; if absent, print to stdout."
    )
    common.add_argument(
        "--bucket-by",
        default=None,
        help=(
            "Repeat scan per bucket: "
            "ema:FEATURE@THR | "
            "calendar:start,end;start,end | "
            "feature_quantile:FEATURE@N"
        ),
    )

    fp = sub.add_parser("feature-plateau", parents=[common])
    fp.add_argument("--feature", required=True)
    fp.add_argument("--operator", default="<=", choices=list(_OPS.keys()))
    fp.add_argument("--grid", required=True, help="Comma-separated thresholds.")

    cs = sub.add_parser("condition-set", parents=[common])
    cs.add_argument(
        "--condition",
        action="append",
        required=True,
        help="May be repeated; format 'name: expr' or just 'expr'.",
    )

    ps = sub.add_parser("pair-scan", parents=[common])
    ps.add_argument(
        "--pair-a",
        required=True,
        help="'feature:op:grid' e.g. 'vol_persistence:>:0.003,0.01,0.03'",
    )
    ps.add_argument("--pair-b", required=True)

    ic = sub.add_parser("ic-decay", parents=[common])
    ic.add_argument(
        "--features",
        required=True,
        help="Comma-separated feature columns.",
    )
    ic.add_argument(
        "--horizons",
        default="1",
        help="Comma-separated horizon labels (uses target or target_H column).",
    )
    ic.add_argument("--target", default="forward_rr")
    ic.add_argument(
        "--baseline-json",
        default=None,
        help="Optional IC baseline JSON (rows with bucket=all).",
    )
    return p


def main() -> int:
    import sys as _sys

    print(
        "DEPRECATED: use 'mlbot research scan|ic|plateau' instead of "
        "scripts/quick_layer_scan.py (forwards to same kernels).",
        file=_sys.stderr,
    )
    args = _make_parser().parse_args()

    pq = Path(args.features_parquet)
    if not pq.is_absolute():
        pq = (PROJECT_ROOT / pq).resolve()
    if not pq.exists():
        print(f"ERROR: parquet not found: {pq}", file=sys.stderr)
        return 3

    df = pd.read_parquet(pq)
    label: pd.Series
    if args.mode == "ic-decay":
        label = pd.Series(True, index=df.index)
    else:
        if args.label not in df.columns:
            print(f"ERROR: label '{args.label}' missing", file=sys.stderr)
            return 3
        label = pd.to_numeric(df[args.label], errors="coerce").fillna(0).astype(bool)

    base_mask = pd.Series(True, index=df.index)
    for f in args.filter or []:
        base_mask = base_mask & _parse_clause(f)(df)
    base_mask = base_mask & _build_calendar_mask(df, args.calendar_window)

    try:
        if getattr(args, "bucket_by", None):
            report = _bucketed_report(args, df, label, base_mask, args.bucket_by)
        elif args.mode == "feature-plateau":
            report = mode_feature_plateau(args, df, label, base_mask)
        elif args.mode == "condition-set":
            report = mode_condition_set(args, df, label, base_mask)
        elif args.mode == "pair-scan":
            report = mode_pair_scan(args, df, label, base_mask)
        elif args.mode == "ic-decay":
            report = mode_ic_decay(args, df, base_mask)
        else:
            print(f"unknown mode {args.mode}", file=sys.stderr)
            return 3
    except (ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = (PROJECT_ROOT / out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
