#!/usr/bin/env python3
"""
统一方向研究 — Direction Determination 对比分析

对比三个 archetype (BPC / ME / FER) 的方向确定逻辑:
  1. 旧逻辑 (原始 _detect_direction_col)
  2. 新逻辑 (direction.yaml 规则驱动)

输出:
  - 每个策略的方向分布对比
  - 方向一致性检查
  - 方向变更影响的交易统计

用法:
    python docs/z实验_005_统一研究/direction_research.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = Path(__file__).resolve().parent


# ================================================================
# Direction Config 加载与应用
# ================================================================


def load_direction_config(
    strategy: str, strategies_root: str = "config/strategies"
) -> Dict[str, Any]:
    """加载 archetypes/direction.yaml"""
    path = Path(strategies_root) / strategy / "archetypes" / "direction.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_direction_rules_series(
    df: pd.DataFrame, direction_cfg: Dict[str, Any]
) -> pd.Series:
    """对 DataFrame 应用 direction.yaml 规则，返回方向 Series"""
    rules = direction_cfg.get("direction_rules", [])
    result = pd.Series(0.0, index=df.index)
    assigned = pd.Series(False, index=df.index)
    used_rule = "none"

    for rule in rules:
        feature = rule.get("feature", "")
        transform = rule.get("transform", "raw")

        if feature not in df.columns:
            continue

        series = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
        unassigned = ~assigned

        if transform == "raw":
            result[unassigned] = series[unassigned]
        elif transform == "sign":
            result[unassigned] = np.sign(series[unassigned])
        elif transform == "negate_sign":
            result[unassigned] = -np.sign(series[unassigned])
        elif transform == "center_sign":
            result[unassigned] = np.sign(series[unassigned] - 0.5)
        else:
            result[unassigned] = series[unassigned]

        newly_assigned = unassigned & (result != 0)
        assigned = assigned | newly_assigned
        used_rule = f"{feature} ({transform})"

        if assigned.all():
            break

    return result, used_rule


def detect_direction_col_legacy(df: pd.DataFrame, archetype: str) -> Optional[str]:
    """旧逻辑: 检测列名"""
    candidates = [
        f"{archetype}_breakout_direction",
        "breakout_direction",
        "entry_direction",
        "direction",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


# ================================================================
# 数据路径
# ================================================================

PREDICTIONS = {
    "bpc": "results/train_final_20260208_220616_return_tree/bpc/predictions.parquet",
    "me": "results/train_final_20260215_234211_return_tree/me/predictions_fixed.parquet",
    "fer": "results/train_final_20260216_184525_return_tree/fer/predictions_fixed.parquet",
}


# ================================================================
# 分析函数
# ================================================================


def analyze_archetype(
    arch_name: str, df: pd.DataFrame, strategies_root: str = "config/strategies"
) -> Dict[str, Any]:
    """分析单个 archetype 的方向确定"""

    result = {"archetype": arch_name, "rows": len(df)}

    # ── 1. 旧逻辑 ──
    legacy_col = detect_direction_col_legacy(df, arch_name)
    if legacy_col:
        legacy_dir = df[legacy_col].astype(float)
    else:
        legacy_dir = pd.Series(0.0, index=df.index)
        legacy_col = "none"

    result["legacy_col"] = legacy_col
    result["legacy_long"] = int((legacy_dir > 0).sum())
    result["legacy_short"] = int((legacy_dir < 0).sum())
    result["legacy_zero"] = int((legacy_dir == 0).sum())

    # ── 2. 新逻辑 (direction.yaml) ──
    dir_cfg = load_direction_config(arch_name, strategies_root)
    if dir_cfg:
        new_dir, used_rule = apply_direction_rules_series(df, dir_cfg)
        result["new_source"] = dir_cfg.get("causal_source", "unknown")
        result["new_rule"] = used_rule
    else:
        new_dir = legacy_dir.copy()
        result["new_source"] = "fallback_legacy"
        result["new_rule"] = legacy_col

    result["new_long"] = int((new_dir > 0).sum())
    result["new_short"] = int((new_dir < 0).sum())
    result["new_zero"] = int((new_dir == 0).sum())

    # ── 3. 对比 ──
    legacy_sign = np.sign(legacy_dir)
    new_sign = np.sign(new_dir)
    match = (legacy_sign == new_sign).sum()
    total = len(df)
    result["direction_match_pct"] = round(match / total * 100, 2) if total > 0 else 0
    result["direction_changed"] = int((legacy_sign != new_sign).sum())

    # 变更明细
    changed_mask = legacy_sign != new_sign
    if changed_mask.any():
        # long→short, short→long, zero→nonzero, etc.
        changes = pd.DataFrame({
            "legacy": legacy_sign[changed_mask],
            "new": new_sign[changed_mask],
        })
        change_types = (
            changes.groupby(["legacy", "new"]).size().reset_index(name="count")
        )
        result["change_details"] = change_types.to_dict("records")
    else:
        result["change_details"] = []

    # ── 4. 方向分布可视化数据 ──
    result["legacy_dist"] = {
        "long_pct": round(result["legacy_long"] / total * 100, 1),
        "short_pct": round(result["legacy_short"] / total * 100, 1),
        "zero_pct": round(result["legacy_zero"] / total * 100, 1),
    }
    result["new_dist"] = {
        "long_pct": round(result["new_long"] / total * 100, 1),
        "short_pct": round(result["new_short"] / total * 100, 1),
        "zero_pct": round(result["new_zero"] / total * 100, 1),
    }

    return result


def print_report(results: List[Dict[str, Any]]) -> str:
    """生成文本报告"""
    lines = []
    lines.append("=" * 80)
    lines.append("Direction Determination 统一研究报告")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 80)

    # 文档原则
    lines.append("")
    lines.append("## 设计原则 (来自 方向问题.md)")
    lines.append("")
    lines.append("| Archetype | 方向来自 | 因果来源 | 原则 |")
    lines.append("|-----------|---------|---------|------|")
    lines.append("| BPC       | 结构方向 | 突破    | 跟结构走 |")
    lines.append("| ME        | 动能方向 | 加速    | 跟钱走   |")
    lines.append("| FER       | 失败方向 | 反转    | 跟失败走 |")
    lines.append("")
    lines.append("方向是 deterministic 的，不让模型学。")
    lines.append("三者绝对不能共用 direction head。")
    lines.append("")

    for r in results:
        arch = r["archetype"].upper()
        lines.append("-" * 80)
        lines.append(f"### {arch}")
        lines.append(f"数据行数: {r['rows']}")
        lines.append("")

        # 旧逻辑
        lines.append(f"**旧逻辑 (legacy)**")
        lines.append(f"  使用列: {r['legacy_col']}")
        lines.append(
            f"  分布: Long={r['legacy_long']} ({r['legacy_dist']['long_pct']}%) | "
            f"Short={r['legacy_short']} ({r['legacy_dist']['short_pct']}%) | "
            f"Zero={r['legacy_zero']} ({r['legacy_dist']['zero_pct']}%)"
        )
        lines.append("")

        # 新逻辑
        lines.append(f"**新逻辑 (direction.yaml)**")
        lines.append(f"  因果来源: {r['new_source']}")
        lines.append(f"  命中规则: {r['new_rule']}")
        lines.append(
            f"  分布: Long={r['new_long']} ({r['new_dist']['long_pct']}%) | "
            f"Short={r['new_short']} ({r['new_dist']['short_pct']}%) | "
            f"Zero={r['new_zero']} ({r['new_dist']['zero_pct']}%)"
        )
        lines.append("")

        # 对比
        lines.append(f"**方向对比**")
        lines.append(f"  一致率: {r['direction_match_pct']}%")
        lines.append(f"  变更数: {r['direction_changed']}")

        if r["change_details"]:
            lines.append("  变更明细:")
            for c in r["change_details"]:
                legacy_label = (
                    "Long" if c["legacy"] > 0
                    else ("Short" if c["legacy"] < 0 else "Zero")
                )
                new_label = (
                    "Long" if c["new"] > 0
                    else ("Short" if c["new"] < 0 else "Zero")
                )
                lines.append(f"    {legacy_label} → {new_label}: {c['count']} bars")
        lines.append("")

    # 汇总表
    lines.append("=" * 80)
    lines.append("## 汇总")
    lines.append("")
    lines.append(
        "| Archetype | 旧方向来源 | 新因果来源 | 命中规则 | 一致率 | 变更数 |"
    )
    lines.append(
        "|-----------|-----------|-----------|---------|-------|-------|"
    )
    for r in results:
        lines.append(
            f"| {r['archetype'].upper():9s} | {r['legacy_col'][:20]:20s} | "
            f"{r['new_source']:14s} | {r['new_rule'][:25]:25s} | "
            f"{r['direction_match_pct']:5.1f}% | {r['direction_changed']:5d} |"
        )
    lines.append("")

    # direction.yaml 配置摘要
    lines.append("## direction.yaml 配置摘要")
    lines.append("")
    for arch in ["bpc", "me", "fer"]:
        cfg = load_direction_config(arch)
        if cfg:
            lines.append(f"### {arch.upper()}")
            lines.append(f"  causal_source: {cfg.get('causal_source', 'N/A')}")
            for i, rule in enumerate(cfg.get("direction_rules", [])):
                lines.append(
                    f"  规则 {i+1}: {rule.get('feature')} "
                    f"(transform={rule.get('transform')}) — {rule.get('description', '')}"
                )
            lines.append("")

    return "\n".join(lines)


# ================================================================
# Main
# ================================================================


def main():
    print("=" * 80)
    print("Direction Determination 统一研究")
    print("=" * 80)

    strategies_root = str(PROJECT_ROOT / "config" / "strategies")
    results = []

    for arch_name, pred_path in PREDICTIONS.items():
        full_path = PROJECT_ROOT / pred_path
        if not full_path.exists():
            print(f"⚠️  {arch_name}: {full_path} 不存在，跳过")
            continue

        print(f"\n📂 {arch_name}: 加载 {pred_path}")
        df = pd.read_parquet(full_path)
        if "_symbol" in df.columns and "symbol" not in df.columns:
            df["symbol"] = df["_symbol"]
        print(f"   行数: {len(df)}")

        result = analyze_archetype(arch_name, df, strategies_root)
        results.append(result)

        print(f"   旧: {result['legacy_col']} → Long={result['legacy_long']}, Short={result['legacy_short']}")
        print(f"   新: {result['new_rule']} → Long={result['new_long']}, Short={result['new_short']}")
        print(f"   一致率: {result['direction_match_pct']}%, 变更: {result['direction_changed']}")

    # 生成报告
    report = print_report(results)

    output_path = OUTPUT_DIR / "direction_research_report.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n📝 报告已保存: {output_path}")

    # 也打印到终端
    print("\n")
    print(report)


if __name__ == "__main__":
    main()
