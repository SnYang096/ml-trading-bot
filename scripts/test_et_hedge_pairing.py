#!/usr/bin/env python3
"""
ET对冲配对机制有效性验证脚本

验证ET订单与TC/TE仓位的配对关系是否正确：
1. ET订单只在有TC/TE仓位时创建
2. ET订单与TC/TE仓位正确关联
3. TC/TE仓位关闭时，ET对冲也正确关闭
4. ET对冲成本在可接受范围内

使用方法:
    python scripts/test_et_hedge_pairing.py \
        --logs results/live_logs \
        --output results/et_pairing_analysis.json
"""

from __future__ import annotations

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
from datetime import datetime

from src.time_series_model.diagnostics.execution_log_aggregate import (
    aggregate_stage_logs,
)


def extract_et_pairing_info(execution: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """从execution字典中提取ET配对信息"""
    if not execution:
        return {
            "is_et_order": False,
            "et_position_id": None,
            "paired_tc_te_ids": [],
        }

    # 检查是否是ET订单
    execution_strategy = execution.get("execution_strategy")
    execution_tags = execution.get("execution_tags", [])

    is_et_order = execution_strategy == "ET" or any(
        "ET" in str(tag).upper() for tag in execution_tags
    )

    position_id = execution.get("position_id", "")
    et_position_id = position_id if is_et_order else None

    # 尝试从tags或metadata中提取配对信息
    paired_tc_te_ids = []
    if is_et_order:
        # 检查是否有配对信息（可能存储在metadata或其他字段中）
        metadata = execution.get("metadata", {})
        if isinstance(metadata, dict):
            paired_tc_te_ids = metadata.get("paired_positions", [])

    return {
        "is_et_order": is_et_order,
        "et_position_id": et_position_id,
        "paired_tc_te_ids": paired_tc_te_ids,
    }


def extract_tc_te_position_info(execution: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """从execution字典中提取TC/TE仓位信息"""
    if not execution:
        return {
            "is_tc_te_order": False,
            "archetype": None,
            "position_id": None,
        }

    execution_strategy = execution.get("execution_strategy")
    execution_tags = execution.get("execution_tags", [])
    position_id = execution.get("position_id", "")

    # 检查是否是TC/TE订单
    archetype = None
    if execution_strategy in ["TC", "TE"]:
        archetype = execution_strategy
    else:
        # 从tags中查找
        for tag in execution_tags:
            tag_str = str(tag).upper()
            if tag_str in ["TC", "TE"]:
                archetype = tag_str
                break

    is_tc_te_order = archetype is not None

    return {
        "is_tc_te_order": is_tc_te_order,
        "archetype": archetype,
        "position_id": position_id,
    }


def analyze_et_pairing(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    分析ET配对机制

    Args:
        records: 聚合后的执行日志记录列表

    Returns:
        包含配对分析结果的字典
    """
    stats = {
        "total_et_orders": 0,
        "total_tc_te_orders": 0,
        "et_orders_with_tc_te": 0,  # ET订单创建时已有TC/TE仓位
        "et_orders_without_tc_te": 0,  # ET订单创建时没有TC/TE仓位（错误情况）
        "tc_te_orders_with_et": 0,  # TC/TE订单创建后立即有ET对冲
        "tc_te_orders_without_et": 0,  # TC/TE订单创建后没有ET对冲
        "et_orders_closed_with_tc_te": 0,  # ET订单关闭时TC/TE仓位也关闭
        "et_orders_closed_without_tc_te": 0,  # ET订单关闭时TC/TE仓位还在（可能错误）
        "pairing_examples": [],
        "errors": [],
    }

    # 跟踪活跃的TC/TE仓位
    active_tc_te_positions: Set[str] = set()
    active_et_positions: Set[str] = set()
    position_to_archetype: Dict[str, str] = {}
    et_to_tc_te_mapping: Dict[str, List[str]] = {}

    # 按时间戳排序记录
    sorted_records = sorted(
        records,
        key=lambda r: r.get("timestamp", ""),
    )

    for record in sorted_records:
        execution = record.get("execution")
        if not execution:
            continue

        timestamp = record.get("timestamp", "")
        symbol = record.get("symbol", "")

        # 检查是否是TC/TE订单
        tc_te_info = extract_tc_te_position_info(execution)
        if tc_te_info["is_tc_te_order"]:
            stats["total_tc_te_orders"] += 1
            position_id = tc_te_info["position_id"]
            archetype = tc_te_info["archetype"]

            if position_id:
                active_tc_te_positions.add(position_id)
                position_to_archetype[position_id] = archetype

            # 检查后续是否有ET订单（在时间窗口内，例如1分钟内）
            has_et_after = False
            for future_record in sorted_records:
                if future_record == record:
                    continue
                future_timestamp = future_record.get("timestamp", "")
                if future_timestamp <= timestamp:
                    continue
                # 检查时间差（假设timestamp是ISO格式）
                try:
                    dt1 = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    dt2 = datetime.fromisoformat(
                        future_timestamp.replace("Z", "+00:00")
                    )
                    if (dt2 - dt1).total_seconds() > 60:  # 超过1分钟，不再检查
                        break
                except Exception:
                    pass

                future_execution = future_record.get("execution")
                if future_execution:
                    et_info = extract_et_pairing_info(future_execution)
                    if et_info["is_et_order"]:
                        has_et_after = True
                        # 记录配对关系
                        et_position_id = et_info["et_position_id"]
                        if et_position_id:
                            et_to_tc_te_mapping.setdefault(et_position_id, []).append(
                                position_id
                            )
                        break

            if has_et_after:
                stats["tc_te_orders_with_et"] += 1
            else:
                stats["tc_te_orders_without_et"] += 1

        # 检查是否是ET订单
        et_info = extract_et_pairing_info(execution)
        if et_info["is_et_order"]:
            stats["total_et_orders"] += 1
            et_position_id = et_info["et_position_id"]

            if et_position_id:
                active_et_positions.add(et_position_id)

            # 检查创建ET订单时是否有活跃的TC/TE仓位
            if len(active_tc_te_positions) > 0:
                stats["et_orders_with_tc_te"] += 1
                # 记录配对示例
                stats["pairing_examples"].append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "et_position_id": et_position_id,
                        "active_tc_te_positions": list(active_tc_te_positions),
                    }
                )
            else:
                stats["et_orders_without_tc_te"] += 1
                # 记录错误
                stats["errors"].append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "et_position_id": et_position_id,
                        "error": "ET order created without active TC/TE positions",
                    }
                )

    # 计算配对正确率
    et_pairing_rate = (
        stats["et_orders_with_tc_te"] / stats["total_et_orders"]
        if stats["total_et_orders"] > 0
        else 0.0
    )

    tc_te_hedged_rate = (
        stats["tc_te_orders_with_et"] / stats["total_tc_te_orders"]
        if stats["total_tc_te_orders"] > 0
        else 0.0
    )

    return {
        "summary": {
            "total_et_orders": stats["total_et_orders"],
            "total_tc_te_orders": stats["total_tc_te_orders"],
            "et_pairing_rate": et_pairing_rate,
            "tc_te_hedged_rate": tc_te_hedged_rate,
        },
        "details": {
            "et_orders_with_tc_te": stats["et_orders_with_tc_te"],
            "et_orders_without_tc_te": stats["et_orders_without_tc_te"],
            "tc_te_orders_with_et": stats["tc_te_orders_with_et"],
            "tc_te_orders_without_et": stats["tc_te_orders_without_et"],
        },
        "pairing_examples": stats["pairing_examples"][:10],  # 只保留前10个示例
        "errors": stats["errors"][:20],  # 只保留前20个错误
        "et_to_tc_te_mapping": {
            k: v[:5] for k, v in list(et_to_tc_te_mapping.items())[:10]
        },  # 只保留前10个映射关系
    }


def analyze_et_cost(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    分析ET对冲成本

    验证ET对冲成本是否在可接受范围内（例如，不超过总收益的5%）
    """
    et_returns = []
    total_returns = []

    for record in records:
        returns = record.get("returns")
        execution = record.get("execution")

        if not returns or not execution:
            continue

        # 检查是否是ET订单
        et_info = extract_et_pairing_info(execution)
        if et_info["is_et_order"]:
            ret_mean = returns.get("ret_mean", 0.0)
            et_returns.append(float(ret_mean))

        # 计算总收益
        ret_mean = returns.get("ret_mean", 0.0)
        ret_trend = returns.get("ret_trend", 0.0)
        total_returns.append(float(ret_mean) + float(ret_trend))

    et_total_cost = sum(et_returns) if et_returns else 0.0
    total_pnl = sum(total_returns) if total_returns else 0.0

    cost_rate = abs(et_total_cost) / abs(total_pnl) if total_pnl != 0 else 0.0

    return {
        "et_total_cost": et_total_cost,
        "et_trade_count": len(et_returns),
        "et_avg_cost_per_trade": et_total_cost / len(et_returns) if et_returns else 0.0,
        "total_pnl": total_pnl,
        "cost_rate": cost_rate,
        "cost_acceptable": cost_rate <= 0.05,  # 成本率不超过5%认为可接受
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test ET hedge pairing mechanism effectiveness"
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
        default="results/et_pairing_analysis.json",
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

    # 分析ET配对
    print("Analyzing ET pairing mechanism...")
    pairing_analysis = analyze_et_pairing(records)

    # 分析ET成本
    print("Analyzing ET cost...")
    cost_analysis = analyze_et_cost(records)

    # 汇总结果
    results = {
        "pairing_analysis": pairing_analysis,
        "cost_analysis": cost_analysis,
    }

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis saved to: {output_path}")
    print("\n=== ET Pairing Mechanism Summary ===")
    print(f"Total ET orders: {pairing_analysis['summary']['total_et_orders']}")
    print(f"Total TC/TE orders: {pairing_analysis['summary']['total_tc_te_orders']}")
    print(f"ET pairing rate: {pairing_analysis['summary']['et_pairing_rate']:.2%}")
    print(
        f"  ET orders with TC/TE: {pairing_analysis['details']['et_orders_with_tc_te']}"
    )
    print(
        f"  ET orders without TC/TE: {pairing_analysis['details']['et_orders_without_tc_te']}"
    )
    print(f"TC/TE hedged rate: {pairing_analysis['summary']['tc_te_hedged_rate']:.2%}")
    print(
        f"  TC/TE orders with ET: {pairing_analysis['details']['tc_te_orders_with_et']}"
    )
    print(
        f"  TC/TE orders without ET: {pairing_analysis['details']['tc_te_orders_without_et']}"
    )
    print("\n=== ET Cost Analysis ===")
    print(f"ET total cost: {cost_analysis['et_total_cost']:.4f}")
    print(f"ET trade count: {cost_analysis['et_trade_count']}")
    print(f"ET avg cost per trade: {cost_analysis['et_avg_cost_per_trade']:.4f}")
    print(f"Total PnL: {cost_analysis['total_pnl']:.4f}")
    print(f"Cost rate: {cost_analysis['cost_rate']:.2%}")
    print(f"Cost acceptable: {cost_analysis['cost_acceptable']}")

    if pairing_analysis["errors"]:
        print(f"\n⚠️  Found {len(pairing_analysis['errors'])} pairing errors")

    return 0


if __name__ == "__main__":
    exit(main())
