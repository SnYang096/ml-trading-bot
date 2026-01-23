#!/usr/bin/env python3
"""
生成分层上线评估报告

汇总各层诊断结果，评估每一层是否满足上线标准，生成综合报告。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_diagnostic_results(diagnostics_dir: Path) -> Dict[str, Any]:
    """加载各层诊断结果"""
    results = {}

    # Layer 1: NN Path Head (if exists)
    nn_path_file = diagnostics_dir / "nn_path_head.md"
    if nn_path_file.exists():
        results["layer1_nn_path_head"] = {
            "status": "completed",
            "file": str(nn_path_file),
        }

    # Layer 2: Gate (if exists)
    gate_file = diagnostics_dir / "gate_performance.md"
    if gate_file.exists():
        results["layer2_gate"] = {
            "status": "completed",
            "file": str(gate_file),
        }

    # Layer 3: Archetype (if exists)
    archetype_file = diagnostics_dir / "archetype_stability.md"
    if archetype_file.exists():
        results["layer3_archetype"] = {
            "status": "completed",
            "file": str(archetype_file),
        }

    # Layer 4: Execution (if exists)
    execution_file = diagnostics_dir / "execution_performance.md"
    if execution_file.exists():
        results["layer4_execution"] = {
            "status": "completed",
            "file": str(execution_file),
        }

    # Layer 5: PCM
    pcm_file = diagnostics_dir / "pcm_performance.md"
    if pcm_file.exists():
        results["layer5_pcm"] = {
            "status": "completed",
            "file": str(pcm_file),
        }

    # Layer 6: Outcome/Attribution
    outcome_file = diagnostics_dir / "outcome_attribution.md"
    if outcome_file.exists():
        results["layer6_outcome"] = {
            "status": "completed",
            "file": str(outcome_file),
        }

    # Production attribution summary
    attribution_file = diagnostics_dir / "degradation_report.json"
    if attribution_file.exists():
        with open(attribution_file, "r") as f:
            results["production_attribution"] = json.load(f)

    return results


def assess_layer_readiness(
    diagnostic_results: Dict[str, Any],
    logs_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """评估每一层的上线就绪状态"""
    import pandas as pd

    readiness = {}

    # Load logs for basic checks
    try:
        df = pd.read_parquet(logs_path)
        gated_df = df[df.get("gate_ok", pd.Series([False] * len(df))) == True]
        total_trades = len(gated_df)
    except Exception:
        total_trades = 0

    # Layer 1: NN Path Head
    readiness["layer1_nn_path_head"] = {
        "name": "NN Path Head",
        "status": "unknown",
        "readiness": "unknown",
        "notes": "需要运行diagnose_nn_path_head.py进行评估",
    }
    if "layer1_nn_path_head" in diagnostic_results:
        readiness["layer1_nn_path_head"]["status"] = "diagnosed"
        readiness["layer1_nn_path_head"]["readiness"] = "pending_review"  # 需要人工审查

    # Layer 2: Gate
    readiness["layer2_gate"] = {
        "name": "Gate",
        "status": "unknown",
        "readiness": "unknown",
        "notes": "需要运行diagnose_gate_performance.py进行评估",
    }
    if "layer2_gate" in diagnostic_results:
        readiness["layer2_gate"]["status"] = "diagnosed"
        readiness["layer2_gate"]["readiness"] = "pending_review"

    # Layer 3: Archetype
    readiness["layer3_archetype"] = {
        "name": "Archetype",
        "status": "unknown",
        "readiness": "unknown",
        "notes": "需要运行diagnose_archetype_stability.py进行评估",
    }
    if "layer3_archetype" in diagnostic_results:
        readiness["layer3_archetype"]["status"] = "diagnosed"
        readiness["layer3_archetype"]["readiness"] = "pending_review"

    # Layer 4: Execution
    readiness["layer4_execution"] = {
        "name": "Execution",
        "status": "unknown",
        "readiness": "unknown",
        "notes": "需要运行diagnose_execution_performance.py进行评估",
    }
    if "layer4_execution" in diagnostic_results:
        readiness["layer4_execution"]["status"] = "diagnosed"
        readiness["layer4_execution"]["readiness"] = "pending_review"

    # Layer 5: PCM
    readiness["layer5_pcm"] = {
        "name": "PCM (Portfolio Capital Management)",
        "status": "unknown",
        "readiness": "unknown",
        "notes": "需要运行diagnose_pcm_performance.py进行评估",
    }
    if "layer5_pcm" in diagnostic_results:
        readiness["layer5_pcm"]["status"] = "diagnosed"
        readiness["layer5_pcm"]["readiness"] = "pending_review"

    # Layer 6: Outcome/Attribution
    readiness["layer6_outcome"] = {
        "name": "Outcome/Attribution",
        "status": "unknown",
        "readiness": "unknown",
        "notes": "需要运行diagnose_outcome_attribution.py进行评估",
    }
    if "layer6_outcome" in diagnostic_results:
        readiness["layer6_outcome"]["status"] = "diagnosed"
        readiness["layer6_outcome"]["readiness"] = "pending_review"

    # Overall assessment from production attribution
    if "production_attribution" in diagnostic_results:
        attr = diagnostic_results["production_attribution"]
        if attr.get("degradation_detected", False):
            # If degradation detected, mark all layers as needing review
            for layer_key in readiness:
                if readiness[layer_key]["readiness"] == "pending_review":
                    readiness[layer_key]["readiness"] = "needs_attention"
                    readiness[layer_key][
                        "notes"
                    ] = f"Degradation detected: {', '.join(attr.get('alerts', []))}"

    # Basic checks
    if total_trades > 0:
        # If we have trades, at least some layers are working
        for layer_key in readiness:
            if readiness[layer_key]["readiness"] == "unknown":
                readiness[layer_key]["readiness"] = "basic_check_passed"
                readiness[layer_key][
                    "notes"
                ] = f"Basic check: {total_trades} trades found"

    return readiness


def format_readiness_report(
    readiness: Dict[str, Dict[str, Any]],
    diagnostic_results: Dict[str, Any],
) -> str:
    """格式化上线评估报告"""
    lines = []
    lines.append("# 分层上线评估报告")
    lines.append("")
    lines.append("本报告汇总各层诊断结果，评估每一层是否满足上线标准。")
    lines.append("")

    # Summary
    lines.append("## 总体状态")
    lines.append("")

    status_counts = {}
    for layer_info in readiness.values():
        status = layer_info.get("readiness", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    lines.append("| 状态 | 数量 |")
    lines.append("|------|------|")
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")
    lines.append("")

    # Layer details
    lines.append("## 各层详细评估")
    lines.append("")

    layer_order = [
        "layer1_nn_path_head",
        "layer2_gate",
        "layer3_archetype",
        "layer4_execution",
        "layer5_pcm",
        "layer6_outcome",
    ]

    for layer_key in layer_order:
        if layer_key not in readiness:
            continue

        layer_info = readiness[layer_key]
        lines.append(f"### {layer_info['name']}")
        lines.append("")
        lines.append(f"- **状态**: {layer_info.get('status', 'unknown')}")
        lines.append(f"- **上线就绪**: {layer_info.get('readiness', 'unknown')}")
        lines.append(f"- **说明**: {layer_info.get('notes', 'N/A')}")

        # Link to diagnostic file if available
        if layer_key in diagnostic_results:
            diag_info = diagnostic_results[layer_key]
            if "file" in diag_info:
                lines.append(f"- **诊断报告**: {diag_info['file']}")
        lines.append("")

    # Recommendations
    lines.append("## 上线建议")
    lines.append("")

    needs_attention = [
        k for k, v in readiness.items() if v.get("readiness") == "needs_attention"
    ]
    pending_review = [
        k for k, v in readiness.items() if v.get("readiness") == "pending_review"
    ]
    unknown = [k for k, v in readiness.items() if v.get("readiness") == "unknown"]

    if needs_attention:
        lines.append("### ⚠️ 需要关注")
        lines.append("")
        for layer_key in needs_attention:
            lines.append(
                f"- **{readiness[layer_key]['name']}**: {readiness[layer_key].get('notes', 'N/A')}"
            )
        lines.append("")

    if pending_review:
        lines.append("### 📋 待审查")
        lines.append("")
        for layer_key in pending_review:
            lines.append(
                f"- **{readiness[layer_key]['name']}**: 诊断已完成，需要人工审查"
            )
        lines.append("")

    if unknown:
        lines.append("### ❓ 未诊断")
        lines.append("")
        for layer_key in unknown:
            lines.append(f"- **{readiness[layer_key]['name']}**: 尚未运行诊断")
        lines.append("")

    if not needs_attention and not pending_review and not unknown:
        lines.append("✅ 所有层都已通过基本检查，可以进行上线评估。")
        lines.append("")

    lines.append("## 下一步行动")
    lines.append("")
    lines.append("1. 审查所有'待审查'和'需要关注'的层")
    lines.append("2. 运行缺失的诊断脚本")
    lines.append("3. 根据诊断结果修复问题")
    lines.append("4. 重新运行评估，确认所有层都满足上线标准")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate layer readiness report")
    p.add_argument(
        "--diagnostics-dir",
        required=True,
        help="Directory containing diagnostic results",
    )
    p.add_argument("--logs", required=True, help="Logs file for basic checks (parquet)")
    p.add_argument("--output-md", required=True, help="Output markdown report path")
    p.add_argument("--output-json", required=True, help="Output JSON report path")
    args = p.parse_args()

    diagnostics_dir = Path(args.diagnostics_dir)
    if not diagnostics_dir.exists():
        print(
            f"Error: Diagnostics directory not found: {diagnostics_dir}",
            file=sys.stderr,
        )
        return 1

    # Load diagnostic results
    diagnostic_results = load_diagnostic_results(diagnostics_dir)

    # Assess readiness
    readiness = assess_layer_readiness(diagnostic_results, Path(args.logs))

    # Format and save report
    report = format_readiness_report(readiness, diagnostic_results)

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")

    output_json = Path(args.output_json)
    report_data = {
        "readiness": readiness,
        "diagnostic_results": {
            k: {"status": v.get("status")} for k, v in diagnostic_results.items()
        },
    }
    with open(output_json, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"Layer readiness report generated:")
    print(f"  - Markdown: {output_md}")
    print(f"  - JSON: {output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
