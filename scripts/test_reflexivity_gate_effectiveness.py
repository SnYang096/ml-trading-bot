#!/usr/bin/env python3
"""
反身性特征Gate规则有效性验证脚本

验证OFCI和SHD特征是否正确触发gate规则：
- 当 ofci_pct > 0.9 时，soft veto（仓位减少60%）触发率
- 当 shd_pct > 0.9 时，hard veto（拒绝交易）触发率
- 当 ofci_pct > 0.95 时，hard veto触发率

使用方法:
    python scripts/test_reflexivity_gate_effectiveness.py \
        --logs results/live_logs \
        --output results/reflexivity_gate_analysis.json
"""

from __future__ import annotations

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

from src.time_series_model.diagnostics.execution_log_aggregate import (
    aggregate_stage_logs,
)


def extract_reflexivity_features(
    features: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """从features字典中提取反身性特征"""
    if not features:
        return {"ofci_pct": 0.0, "shd_pct": 0.0, "lfi_pct": 0.0}

    return {
        "ofci_pct": float(features.get("ofci_pct", 0.0)),
        "shd_pct": float(features.get("shd_pct", 0.0)),
        "lfi_pct": float(features.get("lfi_pct", 0.0)),
    }


def extract_gate_decision(gate: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """从gate字典中提取决策信息"""
    if not gate:
        return {
            "blocked": False,
            "decisions": [],
            "reasons": {},
            "has_reflexivity_veto": False,
            "veto_type": None,  # "hard" or "soft" or None
        }

    blocked = bool(gate.get("blocked", False))
    decisions = gate.get("decisions", [])
    reasons = gate.get("reasons", {})

    # 检查是否有反身性相关的veto
    has_reflexivity_veto = False
    veto_type = None

    # 检查hard veto (SHD)
    if any(
        "reflexivity" in str(d).lower()
        or "shd" in str(d).lower()
        or "strategy_homogeneity" in str(d).lower()
        for d in decisions
    ):
        has_reflexivity_veto = True
        veto_type = "hard"
    # 检查soft veto (OFCI)
    elif any(
        "ofci" in str(d).lower() or "high_consensus" in str(d).lower()
        for d in decisions
    ):
        has_reflexivity_veto = True
        veto_type = "soft"
    # 检查execution_rules中的反身性规则
    elif isinstance(reasons, dict):
        exec_rules = reasons.get("execution_rules", {})
        if isinstance(exec_rules, dict):
            for rule_name, rule_value in exec_rules.items():
                if "reflexivity" in str(rule_name).lower():
                    has_reflexivity_veto = True
                    if "shd" in str(rule_name).lower() and rule_value:
                        veto_type = "hard"
                    elif "ofci" in str(rule_name).lower() and rule_value:
                        veto_type = "soft"

    return {
        "blocked": blocked,
        "decisions": decisions,
        "reasons": reasons,
        "has_reflexivity_veto": has_reflexivity_veto,
        "veto_type": veto_type,
    }


def analyze_gate_trigger_rates(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    分析gate规则触发率

    Args:
        records: 聚合后的执行日志记录列表

    Returns:
        包含触发率统计的字典
    """
    stats = {
        "total_decisions": 0,
        "ofci_high_count": 0,  # ofci_pct > 0.9
        "ofci_extreme_count": 0,  # ofci_pct > 0.95
        "shd_high_count": 0,  # shd_pct > 0.9
        "ofci_soft_veto_triggered": 0,
        "ofci_soft_veto_expected": 0,
        "shd_hard_veto_triggered": 0,
        "shd_hard_veto_expected": 0,
        "ofci_extreme_hard_veto_triggered": 0,
        "ofci_extreme_hard_veto_expected": 0,
        "false_positives": [],  # 应该触发但未触发的情况
        "false_negatives": [],  # 不应该触发但触发了的情况
    }

    for record in records:
        features = record.get("features")
        gate = record.get("gate")
        execution = record.get("execution")

        if not features:
            continue

        stats["total_decisions"] += 1

        # 提取反身性特征
        reflexivity = extract_reflexivity_features(features)
        ofci_pct = reflexivity["ofci_pct"]
        shd_pct = reflexivity["shd_pct"]

        # 提取gate决策
        gate_decision = extract_gate_decision(gate)
        blocked = gate_decision["blocked"]
        has_reflexivity_veto = gate_decision["has_reflexivity_veto"]
        veto_type = gate_decision["veto_type"]

        # 统计高风险场景
        if ofci_pct > 0.9:
            stats["ofci_high_count"] += 1
            stats["ofci_soft_veto_expected"] += 1
            if has_reflexivity_veto and veto_type == "soft":
                stats["ofci_soft_veto_triggered"] += 1
            elif not has_reflexivity_veto:
                # False negative: 应该触发soft veto但未触发
                stats["false_negatives"].append(
                    {
                        "timestamp": record.get("timestamp"),
                        "symbol": record.get("symbol"),
                        "ofci_pct": ofci_pct,
                        "shd_pct": shd_pct,
                        "expected": "soft_veto",
                        "actual": "no_veto",
                    }
                )

        if ofci_pct > 0.95:
            stats["ofci_extreme_count"] += 1
            stats["ofci_extreme_hard_veto_expected"] += 1
            if blocked and has_reflexivity_veto and veto_type == "hard":
                stats["ofci_extreme_hard_veto_triggered"] += 1
            elif not blocked:
                # False negative: 应该触发hard veto但未触发
                stats["false_negatives"].append(
                    {
                        "timestamp": record.get("timestamp"),
                        "symbol": record.get("symbol"),
                        "ofci_pct": ofci_pct,
                        "shd_pct": shd_pct,
                        "expected": "hard_veto",
                        "actual": "not_blocked",
                    }
                )

        if shd_pct > 0.9:
            stats["shd_high_count"] += 1
            stats["shd_hard_veto_expected"] += 1
            if blocked and has_reflexivity_veto and veto_type == "hard":
                stats["shd_hard_veto_triggered"] += 1
            elif not blocked:
                # False negative: 应该触发hard veto但未触发
                stats["false_negatives"].append(
                    {
                        "timestamp": record.get("timestamp"),
                        "symbol": record.get("symbol"),
                        "ofci_pct": ofci_pct,
                        "shd_pct": shd_pct,
                        "expected": "hard_veto",
                        "actual": "not_blocked",
                    }
                )

        # 检查false positives: 不应该触发但触发了
        if has_reflexivity_veto:
            if veto_type == "soft" and ofci_pct <= 0.9:
                stats["false_positives"].append(
                    {
                        "timestamp": record.get("timestamp"),
                        "symbol": record.get("symbol"),
                        "ofci_pct": ofci_pct,
                        "shd_pct": shd_pct,
                        "expected": "no_veto",
                        "actual": "soft_veto",
                    }
                )
            elif veto_type == "hard" and shd_pct <= 0.9 and ofci_pct <= 0.95:
                stats["false_positives"].append(
                    {
                        "timestamp": record.get("timestamp"),
                        "symbol": record.get("symbol"),
                        "ofci_pct": ofci_pct,
                        "shd_pct": shd_pct,
                        "expected": "no_veto",
                        "actual": "hard_veto",
                    }
                )

    # 计算触发率
    ofci_soft_veto_rate = (
        stats["ofci_soft_veto_triggered"] / stats["ofci_soft_veto_expected"]
        if stats["ofci_soft_veto_expected"] > 0
        else 0.0
    )

    shd_hard_veto_rate = (
        stats["shd_hard_veto_triggered"] / stats["shd_hard_veto_expected"]
        if stats["shd_hard_veto_expected"] > 0
        else 0.0
    )

    ofci_extreme_hard_veto_rate = (
        stats["ofci_extreme_hard_veto_triggered"]
        / stats["ofci_extreme_hard_veto_expected"]
        if stats["ofci_extreme_hard_veto_expected"] > 0
        else 0.0
    )

    return {
        "summary": {
            "total_decisions": stats["total_decisions"],
            "ofci_high_scenarios": stats["ofci_high_count"],
            "ofci_extreme_scenarios": stats["ofci_extreme_count"],
            "shd_high_scenarios": stats["shd_high_count"],
        },
        "trigger_rates": {
            "ofci_soft_veto_rate": ofci_soft_veto_rate,
            "ofci_soft_veto_triggered": stats["ofci_soft_veto_triggered"],
            "ofci_soft_veto_expected": stats["ofci_soft_veto_expected"],
            "shd_hard_veto_rate": shd_hard_veto_rate,
            "shd_hard_veto_triggered": stats["shd_hard_veto_triggered"],
            "shd_hard_veto_expected": stats["shd_hard_veto_expected"],
            "ofci_extreme_hard_veto_rate": ofci_extreme_hard_veto_rate,
            "ofci_extreme_hard_veto_triggered": stats[
                "ofci_extreme_hard_veto_triggered"
            ],
            "ofci_extreme_hard_veto_expected": stats["ofci_extreme_hard_veto_expected"],
        },
        "accuracy": {
            "false_positives_count": len(stats["false_positives"]),
            "false_negatives_count": len(stats["false_negatives"]),
            "false_positives": stats["false_positives"][:10],  # 只保留前10个示例
            "false_negatives": stats["false_negatives"][:10],  # 只保留前10个示例
        },
    }


def analyze_position_scaling(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    分析仓位缩放是否正确应用

    验证当ofci_pct > 0.9时，仓位是否被正确缩放（减少60%）
    """
    scaling_stats = {
        "ofci_high_with_execution": 0,
        "position_scaled_correctly": 0,
        "position_not_scaled": 0,
        "examples": [],
    }

    for record in records:
        features = record.get("features")
        execution = record.get("execution")

        if not features or not execution:
            continue

        reflexivity = extract_reflexivity_features(features)
        ofci_pct = reflexivity["ofci_pct"]

        if ofci_pct <= 0.9:
            continue

        # 检查是否有执行（说明通过了gate）
        if not execution.get("submit_order", False):
            continue

        scaling_stats["ofci_high_with_execution"] += 1

        # 检查仓位大小（如果有记录）
        qty = execution.get("qty")
        if qty is not None:
            # 这里需要对比基础仓位大小来判断是否被缩放
            # 由于日志中可能没有记录基础仓位，我们只能记录观察到的仓位
            scaling_stats["examples"].append(
                {
                    "timestamp": record.get("timestamp"),
                    "symbol": record.get("symbol"),
                    "ofci_pct": ofci_pct,
                    "qty": qty,
                }
            )

    return scaling_stats


def main():
    parser = argparse.ArgumentParser(
        description="Test reflexivity gate rule effectiveness"
    )
    parser.add_argument(
        "--logs",
        type=str,
        required=True,
        help="Path to execution logs directory (stage logs) or canonical log file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/reflexivity_gate_analysis.json",
        help="Output path for analysis results",
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="If set, treat --logs as a canonical log file (JSONL) instead of stage directory",
    )

    args = parser.parse_args()

    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"Error: Logs path does not exist: {logs_path}")
        return 1

    # 加载日志
    print(f"Loading logs from: {logs_path}")
    if args.canonical:
        # 加载canonical log文件
        records = []
        with logs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    else:
        # 加载stage logs并聚合
        records = aggregate_stage_logs(logs_path)

    print(f"Loaded {len(records)} records")

    # 分析gate触发率
    print("Analyzing gate trigger rates...")
    trigger_analysis = analyze_gate_trigger_rates(records)

    # 分析仓位缩放
    print("Analyzing position scaling...")
    scaling_analysis = analyze_position_scaling(records)

    # 汇总结果
    results = {
        "trigger_analysis": trigger_analysis,
        "scaling_analysis": scaling_analysis,
    }

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis saved to: {output_path}")
    print("\n=== Reflexivity Gate Effectiveness Summary ===")
    print(f"Total decisions: {trigger_analysis['summary']['total_decisions']}")
    print(
        f"OFCI high scenarios (>0.9): {trigger_analysis['summary']['ofci_high_scenarios']}"
    )
    print(
        f"OFCI extreme scenarios (>0.95): {trigger_analysis['summary']['ofci_extreme_scenarios']}"
    )
    print(
        f"SHD high scenarios (>0.9): {trigger_analysis['summary']['shd_high_scenarios']}"
    )
    print("\n=== Trigger Rates ===")
    print(
        f"OFCI soft veto rate: {trigger_analysis['trigger_rates']['ofci_soft_veto_rate']:.2%}"
    )
    print(
        f"  Triggered: {trigger_analysis['trigger_rates']['ofci_soft_veto_triggered']}/{trigger_analysis['trigger_rates']['ofci_soft_veto_expected']}"
    )
    print(
        f"SHD hard veto rate: {trigger_analysis['trigger_rates']['shd_hard_veto_rate']:.2%}"
    )
    print(
        f"  Triggered: {trigger_analysis['trigger_rates']['shd_hard_veto_triggered']}/{trigger_analysis['trigger_rates']['shd_hard_veto_expected']}"
    )
    print(
        f"OFCI extreme hard veto rate: {trigger_analysis['trigger_rates']['ofci_extreme_hard_veto_rate']:.2%}"
    )
    print(
        f"  Triggered: {trigger_analysis['trigger_rates']['ofci_extreme_hard_veto_triggered']}/{trigger_analysis['trigger_rates']['ofci_extreme_hard_veto_expected']}"
    )
    print("\n=== Accuracy ===")
    print(f"False positives: {trigger_analysis['accuracy']['false_positives_count']}")
    print(f"False negatives: {trigger_analysis['accuracy']['false_negatives_count']}")

    return 0


if __name__ == "__main__":
    exit(main())
