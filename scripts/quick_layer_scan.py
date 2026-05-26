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
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# 表达式解析（极简：只支持 'feature OP value'、'abs(feature) OP value'，和 AND 链接）
# ---------------------------------------------------------------------------

_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "le": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "ge": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}

_TOKEN_RE = re.compile(
    r"^\s*(abs\()?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)?\s*(<=|>=|<|>|==|!=)\s*([-+0-9eE\.]+)\s*$"
)


def _parse_atom(expr: str) -> Tuple[str, bool, str, float]:
    m = _TOKEN_RE.match(expr)
    if not m:
        raise ValueError(f"Cannot parse atom: {expr!r}")
    abs_, feature, op, value = m.groups()
    return feature, bool(abs_), op, float(value)


def _eval_atom(
    df: pd.DataFrame, feature: str, take_abs: bool, op: str, value: float
) -> pd.Series:
    if feature not in df.columns:
        raise KeyError(f"Feature missing: {feature}")
    s = pd.to_numeric(df[feature], errors="coerce")
    if take_abs:
        s = s.abs()
    return _OPS[op](s, value).fillna(False)


def _parse_clause(expr: str) -> Callable[[pd.DataFrame], pd.Series]:
    parts = [
        p.strip()
        for p in re.split(r"\s+AND\s+", expr, flags=re.IGNORECASE)
        if p.strip()
    ]
    atoms = [_parse_atom(p) for p in parts]

    def fn(df: pd.DataFrame) -> pd.Series:
        masks = [_eval_atom(df, *a) for a in atoms]
        if not masks:
            return pd.Series(True, index=df.index)
        out = masks[0]
        for m in masks[1:]:
            out = out & m
        return out

    return fn


def _build_calendar_mask(df: pd.DataFrame, window: Optional[str]) -> pd.Series:
    if not window:
        return pd.Series(True, index=df.index)
    dt_col = None
    for c in ("datetime", "timestamp", "ts"):
        if c in df.columns:
            dt_col = c
            break
    if dt_col is None:
        raise KeyError("No datetime/timestamp column for --calendar-window")
    dt = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    start_s, end_s = [x.strip() for x in window.split(",")]
    start = pd.to_datetime(start_s, utc=True)
    end = pd.to_datetime(end_s, utc=True)
    return ((dt >= start) & (dt < end)).fillna(False)


# ---------------------------------------------------------------------------
# Stat helpers
# ---------------------------------------------------------------------------


def _binary_proportions_z(p_hit: float, n_hit: int, p_oth: float, n_oth: int) -> float:
    """Two-proportion z-test; returns absolute z (proxy for p-value)."""
    if n_hit < 5 or n_oth < 5:
        return 0.0
    pool = (p_hit * n_hit + p_oth * n_oth) / max(n_hit + n_oth, 1)
    var = pool * (1 - pool) * (1 / n_hit + 1 / n_oth)
    if var <= 0:
        return 0.0
    return float(abs(p_hit - p_oth) / np.sqrt(var))


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def mode_feature_plateau(
    args: argparse.Namespace, df: pd.DataFrame, label: pd.Series, base_mask: pd.Series
) -> str:
    feature = args.feature
    op = args.operator
    grid = [float(x) for x in args.grid.split(",")]
    if feature not in df.columns:
        raise KeyError(f"Feature missing: {feature}")
    s = pd.to_numeric(df[feature], errors="coerce")
    valid = s.notna() & base_mask
    s = s[valid]
    y = label[valid]

    rows = []
    base_succ = float(y.mean()) if len(y) else float("nan")
    for thr in grid:
        m = _OPS[op](s, thr)
        n_hit = int(m.sum())
        if n_hit == 0:
            rows.append((thr, n_hit, float("nan"), float("nan"), 0.0))
            continue
        n_oth = int((~m).sum())
        p_hit = float(y[m].mean())
        p_oth = float(y[~m].mean()) if n_oth else float("nan")
        z = _binary_proportions_z(p_hit, n_hit, p_oth, n_oth) if n_oth else 0.0
        rows.append((thr, n_hit, p_hit, p_oth, z))

    md = [f"# feature_plateau · {feature} {op} ?", ""]
    md.append(f"- base n = {len(y)}, base_success = {base_succ:.3%}")
    md.append("")
    md.append("| threshold | n_hit | succ_hit | succ_other | |z| |")
    md.append("|---:|---:|---:|---:|---:|")
    for thr, n_hit, p_hit, p_oth, z in rows:
        ph = "nan" if pd.isna(p_hit) else f"{p_hit:.3%}"
        po = "nan" if pd.isna(p_oth) else f"{p_oth:.3%}"
        md.append(f"| {thr:.3g} | {n_hit} | {ph} | {po} | {z:.2f} |")
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


def _resolve_target_col(df: pd.DataFrame, target: str, horizon: int) -> Optional[str]:
    if horizon <= 1:
        return target if target in df.columns else None
    for cand in (f"{target}_{horizon}", f"{target}{horizon}"):
        if cand in df.columns:
            return cand
    return target if target in df.columns else None


def _spearman_ic(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 100:
        return float("nan"), float("nan"), n
    rho, p = spearmanr(x[m], y[m])
    return float(rho), float(p), n


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

    sub = df.loc[base_mask]
    for feat in features:
        if feat not in sub.columns:
            md.append(f"| {feat} | — | — | 0 | — | — | — | — | — |")
            continue
        x = pd.to_numeric(sub[feat], errors="coerce")
        for h in horizons:
            tcol = _resolve_target_col(sub, target, h) or target
            if tcol not in sub.columns:
                md.append(f"| {feat} | {h} | missing | 0 | — | — | — | — | — |")
                continue
            y = pd.to_numeric(sub[tcol], errors="coerce")
            rho, p, n = _spearman_ic(x, y)
            base_ic = baseline_map.get((feat, "all", 0))
            delta_s = ""
            flip = ""
            if base_ic is not None and not np.isnan(rho):
                delta = rho - base_ic
                delta_s = f"{delta:+.4f}"
                if base_ic * rho < 0 and abs(rho) > 0.02 and abs(base_ic) > 0.02:
                    flip = "yes"
            base_s = f"{base_ic:.4f}" if base_ic is not None else "—"
            md.append(
                f"| {feat} | {h} | `{tcol}` | {n} | {rho:.4f} | {p:.2e} | "
                f"{base_s} | {delta_s} | {flip} |"
            )
    md.append("")
    md.append(
        "**Reading**: sign_flip=yes with |IC|>0.02 suggests regime drift; "
        "confirm with event_backtest before yaml changes."
    )
    return "\n".join(md)


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

    if args.mode == "feature-plateau":
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
