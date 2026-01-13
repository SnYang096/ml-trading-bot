#!/usr/bin/env python3
"""
Tree model wrap-up utilities:

1) Summarize best-known feature sets per tree strategy into a single Markdown doc.
2) Export "lite" strategy configs that trim Tier2/3 heavy blocks (DTW/Spectrum/WPT/Hilbert + ticks/orderflow),
   so you can re-introduce features gradually while refactoring.

This script is idempotent: re-running regenerates the doc and overwrites export dirs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]


TREE_STRATEGIES_DEFAULT = [
    "sr_reversal_rr_reg_long",
    "sr_breakout",
    "compression_breakout",
    "trend_following",
]


STAGE_PRIORITY = {"C": 3, "B": 2, "A": 1}


HEAVY_NODE_PATTERNS = [
    # Tier2 (math/signal-processing)
    r"^dtw_.*_f$",
    r"^spectrum_.*_f$",
    r"^wpt_.*_f$",
    r"^hilbert_.*_f$",
    # Tier3 (ticks/orderflow)
    r"^order_flow_.*_f$",
    r"^footprint_.*_f$",
    r"^vpin_.*_f$",
    r"^trade_cluster_.*_f$",
]


def _matches_any(name: str, patterns: List[str]) -> bool:
    for pat in patterns:
        if re.match(pat, name):
            return True
    return False


@dataclass
class PickedRun:
    strategy: str
    stage: str  # A/B/C
    tag: str
    result_json: Path
    suggested_yaml: Optional[Path]
    mtime: float


def _parse_tag_and_stage_from_dirname(
    dirname: str,
) -> Tuple[Optional[str], Optional[str]]:
    # Examples:
    # trend_following_pipeline_poolb_semantic_20260110_tf_fast_blacklist_C
    # compression_breakout_pipeline_poolb_semantic_20260108_best_abc_B
    m = re.match(r".*?_poolb_semantic_(.+?)_([ABC])$", dirname)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _find_candidate_result_jsons(strategy: str) -> List[Path]:
    root = ROOT / "results" / "feature_group_search"
    return sorted(
        root.glob(
            f"{strategy}_pipeline_poolb_semantic_*_*/feature_group_search_result.json"
        )
    )


def _pick_best_run(strategy: str) -> Optional[PickedRun]:
    candidates: List[PickedRun] = []
    for p in _find_candidate_result_jsons(strategy):
        tag, stage = _parse_tag_and_stage_from_dirname(p.parent.name)
        if not tag or not stage:
            continue
        # Suggested YAML convention
        sy = (
            ROOT
            / "config"
            / "strategies"
            / strategy
            / f"features_suggested_pipeline_poolb_semantic_{tag}_{stage}.yaml"
        )
        candidates.append(
            PickedRun(
                strategy=strategy,
                stage=stage,
                tag=tag,
                result_json=p,
                suggested_yaml=sy if sy.exists() else None,
                mtime=p.stat().st_mtime,
            )
        )
    if not candidates:
        return None
    # Prefer higher stage; within same stage prefer latest mtime
    candidates.sort(
        key=lambda r: (STAGE_PRIORITY.get(r.stage, 0), r.mtime), reverse=True
    )
    return candidates[0]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(obj: dict) -> str:
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)


def _make_lite_features_yaml(full_yaml: dict) -> Tuple[dict, List[str]]:
    """
    Return (lite_yaml, removed_feature_nodes).
    """
    out = json.loads(json.dumps(full_yaml))  # deep-ish copy without extra deps
    fp = out.get("feature_pipeline") or {}
    req = fp.get("requested_features") or []
    req = [str(x).strip() for x in req if str(x).strip()]
    # stable dedup
    seen = set()
    req_dedup = []
    for f in req:
        if f in seen:
            continue
        seen.add(f)
        req_dedup.append(f)
    req = req_dedup
    kept = []
    removed = []
    for f in req:
        if _matches_any(f, HEAVY_NODE_PATTERNS):
            removed.append(f)
        else:
            kept.append(f)
    fp["requested_features"] = kept

    # Trim invert_features for lite: keep only those that are clearly not tied to removed heavy blocks.
    inv = fp.get("invert_features") or []
    inv = [str(x).strip() for x in inv if str(x).strip()]
    inv_removed_pats = [
        r"^dtw_",
        r"^wpt_",
        r"^spectrum_",
        r"^trade_cluster_",
        r"^vpin_",
        r"^footprint_",
        r"^order_flow_",
        r"^hilbert_",
        r"^fp_",  # footprint output columns often start with fp_*
    ]
    inv_kept = []
    for c in inv:
        if any(re.match(p, c) for p in inv_removed_pats):
            continue
        inv_kept.append(c)
    # stable dedup
    seen = set()
    inv2 = []
    for c in inv_kept:
        if c in seen:
            continue
        seen.add(c)
        inv2.append(c)
    inv_kept = inv2
    fp["invert_features"] = inv_kept

    out["feature_pipeline"] = fp

    # Keep provenance metadata (feature_group_search) as-is.
    return out, removed


def _export_strategy_dir(
    *,
    strategy: str,
    tag: str,
    stage: str,
    full_features_yaml_path: Path,
    export_root: Path,
) -> Path:
    src_dir = ROOT / "config" / "strategies" / strategy
    out_dir = export_root / f"{strategy}__{tag}__{stage}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(src_dir, out_dir)

    full_obj = _load_yaml(full_features_yaml_path)
    lite_obj, removed = _make_lite_features_yaml(full_obj)

    # Write files
    (out_dir / "features_full.yaml").write_text(_dump_yaml(full_obj), encoding="utf-8")
    (out_dir / "features_lite.yaml").write_text(_dump_yaml(lite_obj), encoding="utf-8")
    # Default exported config uses lite (for faster incremental testing)
    (out_dir / "features.yaml").write_text(_dump_yaml(lite_obj), encoding="utf-8")

    # Add a small readme
    lines = [
        f"# Exported tree strategy: `{strategy}`",
        "",
        f"- Tag: `{tag}`",
        f"- Stage: `{stage}`",
        "",
        "## Files",
        "",
        "- `features.yaml`: **lite** (default) — heavy blocks removed to make iteration fast",
        "- `features_lite.yaml`: same as above",
        "- `features_full.yaml`: exact suggested config from the selected stage",
        "",
        "## What was trimmed in lite?",
        "",
    ]
    if removed:
        lines += [*(f"- `{x}`" for x in removed)]
    else:
        lines += ["- (nothing)"]
    lines += [
        "",
        "## Notes",
        "",
        "- Lite removes Tier2/3 heavy nodes (DTW/Spectrum/WPT/Hilbert + ticks/orderflow).",
        "- Re-introduce blocks gradually as you refactor and benchmark.",
        "",
    ]
    (out_dir / "EXPORT_INFO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir


def _render_doc(picks: List[PickedRun], doc_path: Path, export_root: Path) -> None:
    rows = []
    for r in picks:
        obj = _load_json(r.result_json)
        sel = obj.get("selected_groups") or []
        # stable dedup for presentation
        feats0 = [str(x) for x in (obj.get("final_features") or []) if str(x).strip()]
        seen = set()
        feats = []
        for x in feats0:
            if x in seen:
                continue
            seen.add(x)
            feats.append(x)
        inv0 = [
            str(x) for x in (obj.get("final_invert_features") or []) if str(x).strip()
        ]
        seen = set()
        inv = []
        for x in inv0:
            if x in seen:
                continue
            seen.add(x)
            inv.append(x)
        rows.append(
            {
                "strategy": r.strategy,
                "tag": r.tag,
                "stage": r.stage,
                "objective": obj.get("objective"),
                "seeds": obj.get("seeds"),
                "search_algo": obj.get("search_algo"),
                "elapsed_sec": obj.get("elapsed_sec"),
                "selected_groups": sel,
                "final_features": feats,
                "final_invert_features": inv,
                "suggested_yaml": str(r.suggested_yaml) if r.suggested_yaml else None,
                "result_json": str(r.result_json),
                "export_dir": str(export_root / f"{r.strategy}__{r.tag}__{r.stage}"),
            }
        )

    lines: List[str] = []
    lines += [
        "# 树模型策略收尾：各策略最有效特征（自动生成）",
        "",
        f"此文档由脚本生成：`scripts/tree_model_finalize.py`",
        "",
        "## 总览",
        "",
        "| strategy | tag | stage | objective | seeds | selected_groups | n_final_features | n_invert | suggested_yaml | export_dir |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["strategy"]),
                    str(r["tag"]),
                    str(r["stage"]),
                    str(r.get("objective") or ""),
                    str(len(r.get("seeds") or [])),
                    str(len(r.get("selected_groups") or [])),
                    str(len(r.get("final_features") or [])),
                    str(len(r.get("final_invert_features") or [])),
                    str(r.get("suggested_yaml") or ""),
                    str(r.get("export_dir") or ""),
                ]
            )
            + " |"
        )

    lines += ["", "## 逐策略详情", ""]
    for r in rows:
        lines += [
            f"### {r['strategy']}",
            "",
            f"- **tag/stage**: `{r['tag']}` / `{r['stage']}`",
            f"- **objective**: `{r.get('objective')}`",
            f"- **search_algo**: `{r.get('search_algo')}`",
            f"- **result_json**: `{r.get('result_json')}`",
            f"- **suggested_yaml（可直接用）**: `{r.get('suggested_yaml')}`",
            f"- **export_dir（默认 lite，已剪掉重特征）**: `{r.get('export_dir')}`",
            "",
            "**selected_groups**:",
        ]
        sg = r.get("selected_groups") or []
        if sg:
            lines += [*(f"- `{x}`" for x in sg)]
        else:
            lines += ["- (none)"]
        lines += ["", "**final_features（建议传给模型的 feature nodes）**:"]
        ff = r.get("final_features") or []
        if ff:
            lines += [*(f"- `{x}`" for x in ff)]
        else:
            lines += ["- (none)"]
        lines += ["", "**final_invert_features（最终确认需要取反的输出列）**:"]
        inv = r.get("final_invert_features") or []
        if inv:
            lines += [*(f"- `{x}`" for x in inv)]
        else:
            lines += ["- (none)"]
        lines += [""]

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strategies",
        default=",".join(TREE_STRATEGIES_DEFAULT),
        help="Comma-separated strategy dir names under config/strategies/",
    )
    ap.add_argument(
        "--doc-out",
        default="docs/strategies/树模型策略结论TREE_STRATEGY_FINAL_FEATURES_CN.md",
        help="Output markdown doc path (relative to repo root).",
    )
    ap.add_argument(
        "--export-root",
        default="config/strategies_exported/tree_best",
        help="Directory to export pruned strategy configs under (relative to repo root).",
    )
    args = ap.parse_args()

    strategies = [s.strip() for s in str(args.strategies).split(",") if s.strip()]
    picks: List[PickedRun] = []
    for s in strategies:
        r = _pick_best_run(s)
        if r is None:
            continue
        picks.append(r)

    export_root = (ROOT / str(args.export_root)).resolve()
    export_root.mkdir(parents=True, exist_ok=True)

    # Export strategy dirs (lite + full)
    for r in picks:
        if r.suggested_yaml is None:
            continue
        _export_strategy_dir(
            strategy=r.strategy,
            tag=r.tag,
            stage=r.stage,
            full_features_yaml_path=r.suggested_yaml,
            export_root=export_root,
        )

    # Write doc
    doc_path = (ROOT / str(args.doc_out)).resolve()
    _render_doc(picks, doc_path, export_root)
    print(f"✅ Wrote doc: {doc_path}")
    print(f"✅ Exported configs under: {export_root}")


if __name__ == "__main__":
    main()
