#!/usr/bin/env python3
"""Regime ablation report — 验证 regime.yaml 的边际贡献。

诊断目标：评估 ``regime.yaml`` 中每条 locked 规则与 ``allowed_sides`` 掩码
对 ``success_no_rr_extreme`` 的提升。与 ``posthoc_layer_effectiveness.py`` 分工：
  - posthoc 侧重 prefilter/gate/entry 三层的 hit-rate / 效应；
  - 本脚本只看 regime 层（chop / box / EMA1200）+ allowed_sides 拆解，
    并按 bull/bear/neutral × long/short 做交叉桶。

输出：JSON + Markdown，路径 ``results/regime_ablation/<ts>/``。

调用示例：
    python scripts/regime_ablation_report.py --strategies bpc,tpc,me,srb
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.posthoc_layer_effectiveness import (
    _build_stat,
    _collect_prefilter_features,
    _eval_prefilter_predicate,
    _find_latest_predictions,
    _load_yaml,
)


def _regime_scopes(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """构造 bull/bear/neutral × long/short 6 个桶 + 总集合。"""
    out: Dict[str, pd.Series] = {"all": pd.Series(True, index=df.index)}

    if "ema_1200_position" in df.columns:
        ema = pd.to_numeric(df["ema_1200_position"], errors="coerce")
        bull = (ema >= 0.005).fillna(False)
        bear = (ema <= -0.005).fillna(False)
        neutral = ((ema > -0.005) & (ema < 0.005)).fillna(False)
    else:
        bull = pd.Series(False, index=df.index)
        bear = pd.Series(False, index=df.index)
        neutral = pd.Series(False, index=df.index)

    if "entry_direction" in df.columns:
        direction = pd.to_numeric(df["entry_direction"], errors="coerce").fillna(0)
        is_long = direction > 0
        is_short = direction < 0
    else:
        is_long = pd.Series(False, index=df.index)
        is_short = pd.Series(False, index=df.index)

    out["bull"] = bull
    out["bear"] = bear
    out["neutral"] = neutral
    out["bull_long"] = bull & is_long
    out["bull_short"] = bull & is_short
    out["bear_long"] = bear & is_long
    out["bear_short"] = bear & is_short
    out["neutral_long"] = neutral & is_long
    out["neutral_short"] = neutral & is_short
    return out


def _allowed_sides_mask(allowed_sides: List[str], direction: pd.Series) -> pd.Series:
    """计算 allowed_sides 掩码 (long/short → +1/-1)。"""
    long_ok = "long" in allowed_sides
    short_ok = "short" in allowed_sides
    is_long = direction > 0
    is_short = direction < 0
    mask = (is_long & long_ok) | (is_short & short_ok)
    if not long_ok and not short_ok:
        mask = pd.Series(False, index=direction.index)
    return mask


def analyze_regime(
    strategy: str, results_root: Path, config_root: Path
) -> Dict[str, Any]:
    pred = _find_latest_predictions(results_root, strategy)
    df = pd.read_parquet(pred)
    if "success_no_rr_extreme" not in df.columns:
        raise KeyError(f"{strategy}: missing label `success_no_rr_extreme` in {pred}")
    success = (
        pd.to_numeric(df["success_no_rr_extreme"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    regime_path = config_root / strategy / "archetypes" / "regime.yaml"
    regime_cfg = _load_yaml(regime_path) if regime_path.exists() else {}

    scopes = _regime_scopes(df)
    stats: List[Dict[str, Any]] = []
    missing: Dict[str, List[str]] = {}

    # 1) 每条 regime rule 的边际效应
    rules = regime_cfg.get("rules", []) or []
    rule_flags: List[pd.Series] = []
    for i, rule in enumerate(rules):
        name = str(rule.get("id") or rule.get("feature") or f"regime_rule_{i+1}")
        feats = _collect_prefilter_features(rule)
        miss = sorted({f for f in feats if f not in df.columns})
        if miss:
            missing[name] = miss
        flag = _eval_prefilter_predicate(rule, df)
        rule_flags.append(flag)
        for scope_name, scope_mask in scopes.items():
            stats.append(
                {
                    **_build_stat(
                        strategy,
                        "regime",
                        scope_name,
                        name,
                        "allow",
                        flag[scope_mask],
                        success[scope_mask],
                    ).__dict__,
                }
            )

    # 2) 层合并（所有 regime 规则同时成立）
    layer_flag = pd.Series(True, index=df.index)
    for f in rule_flags:
        layer_flag &= f
    for scope_name, scope_mask in scopes.items():
        stats.append(
            {
                **_build_stat(
                    strategy,
                    "regime",
                    scope_name,
                    "__layer_all_rules__",
                    "allow",
                    layer_flag[scope_mask],
                    success[scope_mask],
                ).__dict__,
            }
        )

    # 3) allowed_sides 掩码
    allowed_sides = list(regime_cfg.get("allowed_sides", ["long", "short"]))
    if "entry_direction" in df.columns:
        direction = pd.to_numeric(df["entry_direction"], errors="coerce").fillna(0)
        sides_mask = _allowed_sides_mask(allowed_sides, direction)
        for scope_name, scope_mask in scopes.items():
            stats.append(
                {
                    **_build_stat(
                        strategy,
                        "regime",
                        scope_name,
                        "__allowed_sides__",
                        "allow",
                        sides_mask[scope_mask],
                        success[scope_mask],
                    ).__dict__,
                }
            )

    return {
        "strategy": strategy,
        "predictions_path": str(pred),
        "n_rows": int(len(df)),
        "allowed_regimes": list(regime_cfg.get("allowed_regimes", [])),
        "allowed_sides": allowed_sides,
        "missing_features": missing,
        "stats": stats,
    }


def _fmt_pct(v: float) -> str:
    return "n/a" if not np.isfinite(v) else f"{v * 100:.2f}%"


def write_markdown(report: List[Dict[str, Any]], out_md: Path) -> None:
    lines: List[str] = ["# Regime Ablation Report", ""]
    lines.append(
        "Buckets: bull/bear/neutral × long/short via `ema_1200_position` + "
        "`entry_direction`. Evaluates regime.yaml rules + `allowed_sides` "
        "mask on `success_no_rr_extreme`."
    )
    lines.append("")
    for strat in report:
        lines.append(f"## {strat['strategy']}")
        lines.append(f"- predictions: `{strat['predictions_path']}`")
        lines.append(f"- rows: `{strat['n_rows']}`")
        lines.append(f"- allowed_regimes: `{strat['allowed_regimes']}`")
        lines.append(f"- allowed_sides: `{strat['allowed_sides']}`")
        if strat.get("missing_features"):
            lines.append("- missing regime features:")
            for name, feats in strat["missing_features"].items():
                lines.append(f"  - {name}: {', '.join(feats)}")
        for scope in (
            "all",
            "bull",
            "bear",
            "neutral",
            "bull_long",
            "bull_short",
            "bear_long",
            "bear_short",
        ):
            row = next(
                (
                    s
                    for s in strat["stats"]
                    if s["scope"] == scope and s["name"] == "__layer_all_rules__"
                ),
                None,
            )
            if row is None:
                continue
            lines.append(
                f"- {scope}: regime_pass effect={_fmt_pct(row['effect'])}, "
                f"p={row['p_value']:.4g}, pass_rate={_fmt_pct(row['true_rate'])}"
            )
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Regime layer ablation report (bull/bear/neutral × long/short buckets)."
    )
    p.add_argument("--strategies", default="bpc,me,tpc,srb")
    p.add_argument("--results-root", default="results")
    p.add_argument("--config-root", default="config/strategies")
    p.add_argument("--out-dir", default="")
    args = p.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    results_root = Path(args.results_root).resolve()
    config_root = Path(args.config_root).resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir.strip()
        else results_root / "regime_ablation" / ts
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    report = [analyze_regime(s, results_root, config_root) for s in strategies]

    out_json = out_dir / "report.json"
    out_md = out_dir / "report.md"
    out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(report, out_md)
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
