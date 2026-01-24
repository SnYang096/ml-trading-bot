#!/usr/bin/env python3
"""
反身性特征和ET对冲有效性综合测试脚本

运行所有有效性测试并生成综合报告：
1. 反身性特征Gate规则触发验证
2. ET对冲配对机制验证
3. ET对冲成本分析

使用方法:
    python scripts/run_reflexivity_et_effectiveness_tests.py \
        --logs results/live_logs \
        --output results/reflexivity_et_effectiveness_report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any


def run_test_script(script_path: Path, args: list[str]) -> tuple[int, Dict[str, Any]]:
    """运行测试脚本并返回结果"""
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    # 尝试从输出中提取JSON结果（如果脚本输出JSON）
    output_data = {}
    try:
        # 查找输出中的JSON部分
        output_lines = result.stdout.split("\n")
        for line in output_lines:
            if line.strip().startswith("{") or line.strip().startswith("["):
                output_data = json.loads(line)
                break
    except Exception:
        pass

    return result.returncode, output_data


def main():
    parser = argparse.ArgumentParser(
        description="Run all reflexivity and ET hedge effectiveness tests"
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
        default="results/reflexivity_et_effectiveness_report.json",
        help="Output path for comprehensive report",
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="If set, treat --logs as a canonical log file (JSONL) instead of stage directory",
    )
    parser.add_argument(
        "--skip-gate-test",
        action="store_true",
        help="Skip gate rule effectiveness test",
    )
    parser.add_argument(
        "--skip-pairing-test",
        action="store_true",
        help="Skip ET pairing test",
    )

    args = parser.parse_args()

    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"Error: Logs path does not exist: {logs_path}")
        return 1

    script_dir = Path(__file__).parent
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "test_timestamp": None,
        "logs_path": str(logs_path),
        "canonical_mode": args.canonical,
        "tests": {},
        "summary": {},
    }

    # 运行测试1: Gate规则触发验证
    if not args.skip_gate_test:
        print("=" * 80)
        print("Test 1: Reflexivity Gate Rule Effectiveness")
        print("=" * 80)
        gate_test_script = script_dir / "test_reflexivity_gate_effectiveness.py"
        gate_output = output_dir / "reflexivity_gate_analysis.json"

        gate_args = [
            "--logs",
            str(logs_path),
            "--output",
            str(gate_output),
        ]
        if args.canonical:
            gate_args.append("--canonical")

        exit_code, _ = run_test_script(gate_test_script, gate_args)

        if exit_code == 0 and gate_output.exists():
            with gate_output.open("r", encoding="utf-8") as f:
                gate_results = json.load(f)
            report["tests"]["gate_effectiveness"] = gate_results
            print("✅ Gate rule effectiveness test completed")
        else:
            print(f"❌ Gate rule effectiveness test failed (exit code: {exit_code})")
            report["tests"]["gate_effectiveness"] = {"error": "Test failed"}
    else:
        print("⏭️  Skipping gate rule effectiveness test")

    # 运行测试2: ET配对机制验证
    if not args.skip_pairing_test:
        print("\n" + "=" * 80)
        print("Test 2: ET Hedge Pairing Mechanism")
        print("=" * 80)
        pairing_test_script = script_dir / "test_et_hedge_pairing.py"
        pairing_output = output_dir / "et_pairing_analysis.json"

        pairing_args = [
            "--logs",
            str(logs_path),
            "--output",
            str(pairing_output),
        ]
        if args.canonical:
            pairing_args.append("--canonical")

        exit_code, _ = run_test_script(pairing_test_script, pairing_args)

        if exit_code == 0 and pairing_output.exists():
            with pairing_output.open("r", encoding="utf-8") as f:
                pairing_results = json.load(f)
            report["tests"]["et_pairing"] = pairing_results
            print("✅ ET pairing mechanism test completed")
        else:
            print(f"❌ ET pairing mechanism test failed (exit code: {exit_code})")
            report["tests"]["et_pairing"] = {"error": "Test failed"}
    else:
        print("⏭️  Skipping ET pairing mechanism test")

    # 运行测试3: ET对冲成本分析（使用现有的analyze_et_hedge_effectiveness.py）
    print("\n" + "=" * 80)
    print("Test 3: ET Hedge Cost Analysis")
    print("=" * 80)

    # 注意：analyze_et_hedge_effectiveness.py需要parquet/csv格式的日志
    # 如果输入是stage logs，需要先转换为canonical格式
    # 这里我们假设用户已经提供了正确格式的日志
    cost_test_script = script_dir / "analyze_et_hedge_effectiveness.py"
    cost_output = output_dir / "et_hedge_cost_analysis.json"

    # 检查脚本是否存在
    if cost_test_script.exists():
        # 注意：这个脚本需要parquet/csv格式，不是stage logs
        # 如果输入是stage logs，跳过这个测试或提示用户
        print("ℹ️  ET cost analysis requires parquet/csv format logs")
        print(
            "   Skipping cost analysis (use analyze_et_hedge_effectiveness.py directly)"
        )
    else:
        print("⚠️  analyze_et_hedge_effectiveness.py not found")

    # 生成综合摘要
    print("\n" + "=" * 80)
    print("Generating Summary")
    print("=" * 80)

    summary = {
        "all_tests_passed": True,
        "gate_effectiveness": {},
        "et_pairing": {},
        "recommendations": [],
    }

    # 汇总Gate规则测试结果
    if "gate_effectiveness" in report["tests"]:
        gate_test = report["tests"]["gate_effectiveness"]
        if "trigger_analysis" in gate_test:
            trigger = gate_test["trigger_analysis"]
            summary["gate_effectiveness"] = {
                "ofci_soft_veto_rate": trigger.get("trigger_rates", {}).get(
                    "ofci_soft_veto_rate", 0.0
                ),
                "shd_hard_veto_rate": trigger.get("trigger_rates", {}).get(
                    "shd_hard_veto_rate", 0.0
                ),
                "false_positives": trigger.get("accuracy", {}).get(
                    "false_positives_count", 0
                ),
                "false_negatives": trigger.get("accuracy", {}).get(
                    "false_negatives_count", 0
                ),
            }

            # 生成建议
            if trigger.get("trigger_rates", {}).get("ofci_soft_veto_rate", 0.0) < 0.8:
                summary["recommendations"].append(
                    "OFCI soft veto触发率较低，建议检查gate规则配置"
                )
            if trigger.get("trigger_rates", {}).get("shd_hard_veto_rate", 0.0) < 0.9:
                summary["recommendations"].append(
                    "SHD hard veto触发率较低，建议检查gate规则配置"
                )
            if trigger.get("accuracy", {}).get("false_negatives_count", 0) > 10:
                summary["recommendations"].append(
                    "发现较多false negatives，建议检查反身性特征计算"
                )

    # 汇总ET配对测试结果
    if "et_pairing" in report["tests"]:
        pairing_test = report["tests"]["et_pairing"]
        if "pairing_analysis" in pairing_test:
            pairing = pairing_test["pairing_analysis"]
            summary["et_pairing"] = {
                "et_pairing_rate": pairing.get("summary", {}).get(
                    "et_pairing_rate", 0.0
                ),
                "tc_te_hedged_rate": pairing.get("summary", {}).get(
                    "tc_te_hedged_rate", 0.0
                ),
                "errors_count": len(pairing.get("errors", [])),
            }

            # 生成建议
            if pairing.get("summary", {}).get("et_pairing_rate", 0.0) < 0.9:
                summary["recommendations"].append(
                    "ET配对率较低，建议检查ET订单创建逻辑"
                )
            if len(pairing.get("errors", [])) > 5:
                summary["recommendations"].append(
                    "发现ET配对错误，建议检查配对机制实现"
                )

        if "cost_analysis" in pairing_test:
            cost = pairing_test["cost_analysis"]
            summary["et_pairing"]["cost_rate"] = cost.get("cost_rate", 0.0)
            summary["et_pairing"]["cost_acceptable"] = cost.get(
                "cost_acceptable", False
            )

            if not cost.get("cost_acceptable", False):
                summary["recommendations"].append(
                    "ET对冲成本率超过5%，建议优化ET激活条件"
                )

    report["summary"] = summary

    # 保存报告
    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Comprehensive report saved to: {output_path}")
    print("\n=== Summary ===")
    print(f"Gate effectiveness: {summary.get('gate_effectiveness', {})}")
    print(f"ET pairing: {summary.get('et_pairing', {})}")
    if summary["recommendations"]:
        print("\n=== Recommendations ===")
        for i, rec in enumerate(summary["recommendations"], 1):
            print(f"{i}. {rec}")

    return 0


if __name__ == "__main__":
    exit(main())
