#!/usr/bin/env python3
"""Classify IC-selected / model features by family and summarize overfit signals."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATH_PATTERNS = (
    r"hilbert",
    r"wpt_",
    r"spectrum_",
    r"evt_",
    r"alpha101",
    r"hurst",
    r"sqs_",
    r"dtw_",
    r"dl_seq",
    r"zigzag",
    r"hilbert",
)
STRUCTURAL_PATTERNS = (
    r"box_",
    r"bpc_",
    r"tpc_",
    r"me_",
    r"srb_",
    r"cvd",
    r"vpin",
    r"roc_",
    r"macd",
    r"rsi",
    r"shd",
    r"vol_",
    r"hour_",
    r"macro_",
    r"liquidity_void",
    r"footprint",
    r"trade_cluster",
    r"volume_profile",
    r"bb_",
    r"impulse",
)


def classify_feature(name: str) -> str:
    low = name.lower()
    for pat in MATH_PATTERNS:
        if re.search(pat, low):
            return "math"
    for pat in STRUCTURAL_PATTERNS:
        if re.search(pat, low):
            return "structural"
    return "other"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_features(ic: dict[str, Any]) -> list[str]:
    cols = ic.get("columns") or ic.get("selected_columns") or []
    if cols and isinstance(cols[0], dict):
        return [c["feature"] for c in cols[: ic.get("n_selected_columns", 20)]]
    return list(cols)


def _importance_mass(results: dict[str, Any]) -> dict[str, float]:
    imp = results.get("feature_importance") or {}
    if not imp:
        diag = results.get("diagnostics") or {}
        imp = diag.get("feature_importance") or {}
    total = sum(float(v) for v in imp.values()) or 1.0
    by_family: dict[str, float] = {"math": 0.0, "structural": 0.0, "other": 0.0}
    for feat, val in imp.items():
        fam = classify_feature(str(feat))
        by_family[fam] += float(val) / total
    return by_family


def _parse_tau_best(tau_md: Path) -> dict[str, Any]:
    if not tau_md.is_file():
        return {}
    text = tau_md.read_text(encoding="utf-8")
    out: dict[str, Any] = {}
    for line in text.splitlines():
        if line.startswith("**Recommended q**"):
            m = re.search(r"q[=:]?\s*([\d.]+)", line)
            if m:
                out["recommended_q"] = float(m.group(1))
        if "| 0.05 |" in line or line.strip().startswith("| 0.05"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 5 and parts[0].replace(".", "").isdigit():
                try:
                    out["sharpe_q05"] = float(parts[3])
                    out["return_pct_q05"] = float(parts[4].replace("%", ""))
                    out["trades_q05"] = int(float(parts[5]))
                except (ValueError, IndexError):
                    pass
    return out


def analyze_run(run: dict[str, Any], root: Path) -> dict[str, Any]:
    ic_path = root / run["ic_prune_json"]
    res_path = root / run["train_results_json"]
    tau_path = root / run["tau_md"]
    ic = _load_json(ic_path)
    results = _load_json(res_path)
    feats = _selected_features(ic)
    fam_counts = {"math": 0, "structural": 0, "other": 0}
    for f in feats:
        fam_counts[classify_feature(f)] += 1
    n = len(feats) or 1
    eval_block = results.get("evaluation") or {}
    pearson = eval_block.get("pearson_correlation") or results.get(
        "pearson_correlation"
    )
    cv = results.get("avg_cv_metric")
    imp_mass = _importance_mass(results)
    tau = _parse_tau_best(tau_path)
    return {
        "id": run["id"],
        "label": run.get("label", run["id"]),
        "n_features": n,
        "math_pct": round(100 * fam_counts["math"] / n, 1),
        "structural_pct": round(100 * fam_counts["structural"] / n, 1),
        "other_pct": round(100 * fam_counts["other"] / n, 1),
        "importance_math_pct": round(100 * imp_mass.get("math", 0), 1),
        "pearson": pearson,
        "cv_metric": cv,
        "cv_minus_pearson": (
            (cv - pearson) if cv is not None and pearson is not None else None
        ),
        "tau": tau,
        "top_features": feats[:5],
    }


def render_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Feature-family overfit analysis",
        "",
        "| Run | n | math% | struct% | imp math% | Pearson | CV | CV−Pearson | Sharpe@q0.05 | Return% |",
        "|-----|--:|------:|--------:|----------:|--------:|---:|-----------:|---------------:|--------:|",
    ]
    for r in rows:
        tau = r.get("tau") or {}
        lines.append(
            f"| {r['label']} | {r['n_features']} | {r['math_pct']} | {r['structural_pct']} "
            f"| {r['importance_math_pct']} | {r.get('pearson', '—')} | {r.get('cv_metric', '—')} "
            f"| {r.get('cv_minus_pearson', '—')} | {tau.get('sharpe_q05', '—')} "
            f"| {tau.get('return_pct_q05', '—')} |"
        )
    lines.extend(["", "## Top features (first 5)", ""])
    for r in rows:
        lines.append(f"- **{r['label']}**: {', '.join(r.get('top_features') or [])}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        default="config/experiments/20260601_1322_tree_forward_rr_ic_small_pool/feature_family_manifest.yaml",
    )
    p.add_argument(
        "--out",
        default="results/rd_loop/tree_forward_rr_ic_small_pool/feature_family_analysis.md",
    )
    args = p.parse_args(argv)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    rows = [analyze_run(r, PROJECT_ROOT) for r in data.get("runs", [])]
    md = render_md(rows)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
