#!/usr/bin/env python3
"""
Scan per-strategy suggested feature YAMLs and export richer RuleFit rules.

Why:
- The existing best4 export uses ONE "best" config per strategy.
- Many configs do not include VPIN / trade_cluster / footprint nodes in requested_features,
  so top rules look dominated by CVD-like columns.
- This script scans multiple `features_suggested*.yaml` per strategy and exports rules
  with more permissive pruning (more rules / more conditions), then writes a report.

Outputs:
- results/rules_export/tree_scan/<strategy>/<yaml_stem>/rules_regression.{md,json,py}
- results/rules_export/tree_scan/REPORT.md
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RULE_LINE_RE = re.compile(
    r"^\|\s*(?P<rank>\d+)\s*\|\s*(?P<coef>[-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*\|\s*(?P<support>[\d.]*(?:[eE][-+]?\d+)?)\s*\|\s*`(?P<rule>.*)`\s*\|\s*$"
)


@dataclass
class ExportResult:
    strategy: str
    yaml_path: Path
    out_dir: Path
    n_rules: int
    counts: Dict[str, int]
    top_non_cvd: List[Tuple[float, float, str]]  # (abscoef, support, rule)
    top_orderflow: List[Tuple[float, float, str]]


def _find_suggested_yamls(strategy: str) -> List[Path]:
    d = ROOT / "config" / "strategies" / strategy
    if not d.exists():
        return []
    pats = ["features_suggested*.yaml"]
    out: List[Path] = []
    for pat in pats:
        out += list(d.glob(pat))
    # newest first
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def _parse_rules_md(md_path: Path) -> List[Tuple[float, float, str]]:
    if not md_path.exists():
        return []
    rules: List[Tuple[float, float, str]] = []
    for line in md_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RULE_LINE_RE.match(line.strip())
        if not m:
            continue
        try:
            coef = float(m.group("coef"))
        except Exception:
            continue
        sup_raw = m.group("support")
        try:
            support = float(sup_raw) if sup_raw else float("nan")
        except Exception:
            support = float("nan")
        rule = (m.group("rule") or "").strip()
        rules.append((abs(coef), support, rule))
    rules.sort(key=lambda t: (t[0], (t[1] if t[1] == t[1] else -1.0)), reverse=True)
    return rules


def _count_by_keywords(rules: List[Tuple[float, float, str]]) -> Dict[str, int]:
    keys = {
        "cvd": r"\bcvd_",
        "vpin": r"\bvpin_",
        "trade_cluster": r"\btrade_cluster_",
        "footprint_fp": r"\bfp_|\bfootprint_",
        "order_flow": r"\border_flow_",
        "funding": r"\bfunding_",
        "liquidity_void": r"\bliquidity_void_",
        "dtw": r"\bdtw_",
        "wpt": r"\bwpt_",
    }
    out = {k: 0 for k in keys}
    for _, __, rule in rules:
        for k, pat in keys.items():
            if re.search(pat, rule):
                out[k] += 1
    return out


def _pick_top_filtered(
    rules: List[Tuple[float, float, str]],
    *,
    include_pat: str | None = None,
    exclude_pat: str | None = None,
    n: int = 12,
) -> List[Tuple[float, float, str]]:
    out: List[Tuple[float, float, str]] = []
    for abscoef, sup, rule in rules:
        if include_pat and not re.search(include_pat, rule):
            continue
        if exclude_pat and re.search(exclude_pat, rule):
            continue
        out.append((abscoef, sup, rule))
        if len(out) >= n:
            break
    return out


def _run_export_one(
    *,
    strategy: str,
    yaml_path: Path,
    out_dir: Path,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    test_size: float,
    max_rules: int,
    max_conditions: int,
    min_support: float,
    max_rule_len: int,
    random_state: int,
    fast_mode: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/export_tree_rules_imodels.py",
        "--strategy-config",
        str(ROOT / "config" / "strategies" / strategy),
        "--features-yaml",
        str(yaml_path),
        "--symbol",
        str(symbol),
        "--timeframe",
        str(timeframe),
        "--start-date",
        str(start_date),
        "--end-date",
        str(end_date),
        "--test-size",
        str(test_size),
        "--output-dir",
        str(out_dir),
        "--max-rules",
        str(max_rules),
        "--max-conditions",
        str(max_conditions),
        "--min-support",
        str(min_support),
        "--max-rule-len",
        str(max_rule_len),
        "--random-state",
        str(random_state),
    ]
    env = os.environ.copy()
    if fast_mode:
        env["FEATURE_FAST_MODE"] = "1"
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=env)


def _write_report(out_root: Path, results: List[ExportResult]) -> None:
    lines: List[str] = []
    lines.append("# Tree rules scan report (imodels RuleFit)")
    lines.append("")
    lines.append(
        "This report scans multiple `features_suggested*.yaml` per strategy and exports richer rules."
    )
    lines.append("")
    lines.append(
        "Key idea: if VPIN/TradeCluster/FP rarely appear, it's usually because the corresponding YAMLs"
    )
    lines.append(
        "did not request those feature nodes (or they are weaker vs CVD under the label)."
    )
    lines.append("")

    by_strat: Dict[str, List[ExportResult]] = {}
    for r in results:
        by_strat.setdefault(r.strategy, []).append(r)

    for strat in sorted(by_strat):
        lines.append(f"## {strat}")
        lines.append("")
        items = by_strat[strat]
        lines.append(f"- scanned: **{len(items)}** yamls")
        lines.append("")

        # rank configs by having more orderflow rules, then abscoef of top non-cvd
        def score(er: ExportResult) -> Tuple[int, float]:
            of = (
                er.counts.get("vpin", 0)
                + er.counts.get("trade_cluster", 0)
                + er.counts.get("footprint_fp", 0)
            )
            top_non_cvd = er.top_non_cvd[0][0] if er.top_non_cvd else 0.0
            return (of, top_non_cvd)

        items2 = sorted(items, key=score, reverse=True)
        lines.append("### Best candidates (diversity / orderflow)")
        lines.append("")
        for er in items2[:5]:
            rel = er.yaml_path.relative_to(ROOT).as_posix()
            out_rel = er.out_dir.relative_to(ROOT).as_posix()
            of = (
                er.counts.get("vpin", 0)
                + er.counts.get("trade_cluster", 0)
                + er.counts.get("footprint_fp", 0)
            )
            lines.append(
                f"- **{rel}** → `{out_rel}` | rules={er.n_rules}, orderflow_hits={of}, cvd_hits={er.counts.get('cvd',0)}"
            )
        lines.append("")

        lines.append("### Top non-CVD rules (per best candidate)")
        lines.append("")
        best = items2[0]
        rel = best.yaml_path.relative_to(ROOT).as_posix()
        lines.append(f"Source: `{rel}`")
        for abscoef, sup, rule in best.top_non_cvd:
            sup_s = "" if (sup != sup) else f"{sup:.3f}"
            lines.append(f"- |coef|={abscoef:.6g}, support={sup_s}: `{rule}`")
        lines.append("")

        lines.append("### Top orderflow-ish rules (VPIN / trade_cluster / fp)")
        lines.append("")
        for abscoef, sup, rule in best.top_orderflow:
            sup_s = "" if (sup != sup) else f"{sup:.3f}"
            lines.append(f"- |coef|={abscoef:.6g}, support={sup_s}: `{rule}`")
        if not best.top_orderflow:
            lines.append(
                "- (none in top rules; either YAML didn't include these nodes, or signal is weak under this label)"
            )
        lines.append("")

    (out_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strategies",
        default="sr_reversal_rr_reg_long,sr_breakout,compression_breakout,trend_following",
    )
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--timeframe", default="240T")
    ap.add_argument("--start-date", default="2024-01-01")
    ap.add_argument("--end-date", default="2025-12-31")
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument(
        "--max-per-strategy",
        type=int,
        default=8,
        help="Scan newest N suggested yamls per strategy",
    )
    ap.add_argument("--out-root", default="results/rules_export/tree_scan")
    ap.add_argument("--max-rules", type=int, default=200)
    ap.add_argument("--max-conditions", type=int, default=5)
    ap.add_argument("--min-support", type=float, default=0.005)
    ap.add_argument("--max-rule-len", type=int, default=220)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument(
        "--fast-mode", action="store_true", help="Set FEATURE_FAST_MODE=1 during export"
    )
    args = ap.parse_args()

    out_root = ROOT / str(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    strategies = [s.strip() for s in str(args.strategies).split(",") if s.strip()]
    results: List[ExportResult] = []
    failures: List[Tuple[str, Path, str]] = []  # (strategy, yaml_path, error)

    for strat in strategies:
        yamls = _find_suggested_yamls(strat)[: int(args.max_per_strategy)]
        if not yamls:
            print(f"⚠️  no suggested yamls for {strat}")
            continue
        for y in yamls:
            stem = y.stem
            out_dir = out_root / strat / stem
            print(f"▶️ export {strat}: {y} -> {out_dir}")
            try:
                _run_export_one(
                    strategy=strat,
                    yaml_path=y,
                    out_dir=out_dir,
                    symbol=str(args.symbol),
                    timeframe=str(args.timeframe),
                    start_date=str(args.start_date),
                    end_date=str(args.end_date),
                    test_size=float(args.test_size),
                    max_rules=int(args.max_rules),
                    max_conditions=int(args.max_conditions),
                    min_support=float(args.min_support),
                    max_rule_len=int(args.max_rule_len),
                    random_state=int(args.random_state),
                    fast_mode=bool(args.fast_mode),
                )
            except subprocess.CalledProcessError as e:
                failures.append((strat, y, f"CalledProcessError: {e}"))
                print(f"⚠️  export failed for {strat}: {y} ({e})")
                continue
            except Exception as e:
                failures.append((strat, y, f"{type(e).__name__}: {e}"))
                print(f"⚠️  export failed for {strat}: {y} ({type(e).__name__}: {e})")
                continue

            md_path = out_dir / f"{strat}__imodels_rules" / "rules_regression.md"
            rules = _parse_rules_md(md_path)
            counts = _count_by_keywords(rules)
            top_non_cvd = _pick_top_filtered(rules, exclude_pat=r"\bcvd_", n=12)
            top_orderflow = _pick_top_filtered(
                rules,
                include_pat=r"\bvpin_|\btrade_cluster_|\bfp_|\bfootprint_|\border_flow_",
                n=12,
            )
            results.append(
                ExportResult(
                    strategy=strat,
                    yaml_path=y,
                    out_dir=out_dir,
                    n_rules=len(rules),
                    counts=counts,
                    top_non_cvd=top_non_cvd,
                    top_orderflow=top_orderflow,
                )
            )

    _write_report(out_root, results)
    if failures:
        p = out_root / "FAILURES.txt"
        lines = []
        for strat, y, err in failures:
            try:
                rel = y.relative_to(ROOT).as_posix()
            except Exception:
                rel = str(y)
            lines.append(f"{strat}\t{rel}\t{err}")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"⚠️  Failures written to: {p}")
    print(f"✅ Done. Report: {out_root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
