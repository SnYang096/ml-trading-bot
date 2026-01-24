#!/usr/bin/env python3
"""
应用优化后的gate参数

根据诊断结果和平坦高原优化结果，生成优化后的execution_archetypes.yaml配置。

使用方法:
    python scripts/apply_optimized_gate_params.py \
        --diagnosis results/diagnosis/gate_filtering/gate_filtering_analysis.json \
        --optimization results/experiments/gate_plateau/optimal_params.json \
        --current-config config/nnmultihead/execution_archetypes.yaml \
        --out-config config/nnmultihead/execution_archetypes_optimized.yaml
"""

from __future__ import annotations

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_optimization_results(opt_path: Path) -> Dict[str, Any]:
    """加载优化结果"""
    if not opt_path.exists():
        return {}

    with open(opt_path, "r") as f:
        return json.load(f)


def apply_optimizations(
    current_config: Dict[str, Any],
    optimization_results: Dict[str, Any],
    diagnosis_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """应用优化结果到配置"""
    optimized_config = current_config.copy()

    # 如果没有优化结果，使用诊断结果建议的放宽阈值
    if not optimization_results:
        print("⚠️  没有优化结果，使用诊断建议的放宽阈值")

        # 根据诊断报告，主要问题是TC的regime规则太严格
        # 建议放宽以下规则：
        tc_archetype = optimized_config["regimes"]["TREND"]["archetypes"][
            "TrendContinuationTC"
        ]
        rules = tc_archetype["gate_rules"]["rules"]

        optimizations = {
            "tc_not_tc_regime_path_efficiency_too_low": 0.3,  # 从0.6降到0.3（更激进）
            "tc_not_tc_regime_path_length_too_low": 0.2,  # 从0.4降到0.2（更激进）
            "tc_not_tc_regime_jump_risk_too_low": 0.15,  # 从0.3降到0.15（更激进）
            "tc_not_tc_regime_jump_risk_too_high": 0.75,  # 从0.6升到0.75（更激进）
            "tc_not_tc_regime_dir_consistency_too_low": 0.4,  # 从0.6降到0.4
        }

        for rule in rules:
            rule_name = rule.get("name", "")
            if rule_name in optimizations:
                old_threshold = rule.get("threshold")
                new_threshold = optimizations[rule_name]
                rule["threshold"] = new_threshold
                print(f"  ✅ {rule_name}: {old_threshold} → {new_threshold}")

        return optimized_config

    # 应用优化结果
    for arch_name, arch_results in optimization_results.items():
        if arch_name not in optimized_config.get("regimes", {}):
            continue

        # 找到对应的archetype
        for regime_name, regime_config in optimized_config["regimes"].items():
            if arch_name in regime_config.get("archetypes", {}):
                arch_config = regime_config["archetypes"][arch_name]
                rules = arch_config.get("gate_rules", {}).get("rules", [])

                # 应用每个规则的优化结果
                for result in arch_results:
                    rule_name = result.get("rule_name")
                    recommended_threshold = result.get("recommended_threshold")

                    if recommended_threshold is None:
                        continue

                    # 找到对应的规则
                    for rule in rules:
                        if rule.get("name") == rule_name:
                            old_threshold = rule.get("threshold") or rule.get(
                                "quantile"
                            )
                            rule["threshold"] = recommended_threshold
                            print(
                                f"  ✅ {arch_name}/{rule_name}: {old_threshold} → {recommended_threshold}"
                            )
                            break

    return optimized_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="应用优化后的gate参数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--diagnosis",
        default=None,
        help="诊断结果JSON文件（可选）",
    )
    parser.add_argument(
        "--optimization",
        default=None,
        help="优化结果JSON文件（可选）",
    )
    parser.add_argument(
        "--current-config",
        required=True,
        help="当前配置文件路径",
    )
    parser.add_argument(
        "--out-config",
        required=True,
        help="输出配置文件路径",
    )

    args = parser.parse_args()

    # 加载当前配置
    with open(args.current_config, "r") as f:
        current_config = yaml.safe_load(f)

    # 加载优化结果
    optimization_results = {}
    if args.optimization:
        optimization_results = load_optimization_results(Path(args.optimization))

    # 加载诊断结果
    diagnosis_results = None
    if args.diagnosis:
        with open(args.diagnosis, "r") as f:
            diagnosis_results = json.load(f)

    # 应用优化
    print("🔧 应用优化参数...")
    optimized_config = apply_optimizations(
        current_config,
        optimization_results,
        diagnosis_results,
    )

    # 保存优化后的配置
    out_path = Path(args.out_config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(optimized_config, f, default_flow_style=False, allow_unicode=True)

    print(f"✅ 优化后的配置已保存: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
