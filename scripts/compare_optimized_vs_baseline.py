#!/usr/bin/env python3
"""
对比优化参数与基线参数的效果

从优化结果JSON中提取推荐阈值，应用参数，生成新的gated logs，与基线对比。
"""
from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_optimization_results(
    optimization_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """加载优化结果JSON文件"""
    summary_file = optimization_dir / "all_archetypes_optimization_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            return json.load(f)

    # 如果没有summary，尝试加载各个archetype的结果
    results = {}
    for arch_file in optimization_dir.glob("*_optimization.json"):
        arch_name = arch_file.stem.replace("_optimization", "")
        with open(arch_file, "r") as f:
            results[arch_name] = json.load(f)
    return results


def update_execution_archetypes_config(
    config_path: Path,
    optimization_results: Dict[str, List[Dict[str, Any]]],
    backup: bool = True,
) -> bool:
    """更新execution_archetypes.yaml中的阈值"""
    if backup:
        backup_path = config_path.with_suffix(".yaml.backup")
        import shutil

        shutil.copy(config_path, backup_path)
        print(f"Backed up config to: {backup_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    updated = False

    # 遍历regimes和archetypes
    if "regimes" in config:
        for regime_name, regime_config in config["regimes"].items():
            if "archetypes" not in regime_config:
                continue
            for arch_name, arch_config in regime_config["archetypes"].items():
                if arch_name not in optimization_results:
                    continue

                # 更新gate rules中的阈值
                if "gate_rules" in arch_config and "rules" in arch_config["gate_rules"]:
                    deny_if = arch_config["gate_rules"]["rules"].get("deny_if", [])
                    for rule in deny_if:
                        rule_name = rule.get("name", "")
                        # 查找对应的优化结果
                        for opt_result in optimization_results[arch_name]:
                            if opt_result.get("rule_name") == rule_name:
                                recommended = opt_result.get("recommended_threshold")
                                if recommended is not None:
                                    # 更新quantile阈值
                                    if "quantile" in rule:
                                        old_value = rule["quantile"]
                                        rule["quantile"] = recommended
                                        print(
                                            f"  Updated {arch_name}/{rule_name}: {old_value} -> {recommended}"
                                        )
                                        updated = True
                                    break

    if updated:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                config, f, default_flow_style=False, allow_unicode=True, sort_keys=False
            )
        print(f"Updated config: {config_path}")
        return True
    else:
        print("No updates made to config")
        return False


def compare_performance(
    baseline_logs: Path,
    optimized_logs: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """对比基线和优化后的性能"""
    baseline_df = pd.read_parquet(baseline_logs)
    optimized_df = pd.read_parquet(optimized_logs)

    def _sharpe(returns: pd.Series) -> float:
        returns = returns.dropna()
        if len(returns) < 2:
            return 0.0
        mean = returns.mean()
        std = returns.std(ddof=1)
        return float(mean / std * np.sqrt(252)) if std > 1e-12 else 0.0

    def _archetype_return(
        row: pd.Series, ret_mean_col: str, ret_trend_col: str
    ) -> float:
        archetype = str(row.get("gate_archetype") or row.get("archetype") or "").upper()
        if not archetype:
            return 0.0
        if "TC" in archetype or "TE" in archetype:
            return float(row.get(ret_trend_col, 0.0) or 0.0)
        if "FR" in archetype or "ET" in archetype:
            return float(row.get(ret_mean_col, 0.0) or 0.0)
        return 0.0

    # 计算基线性能
    baseline_gated = baseline_df[
        baseline_df.get("gate_ok", pd.Series([False] * len(baseline_df))) == True
    ]
    baseline_gated["archetype_ret"] = baseline_gated.apply(
        lambda r: _archetype_return(r, "ret_mean", "ret_trend"), axis=1
    )
    baseline_returns = baseline_gated["archetype_ret"]

    # 计算优化后性能
    optimized_gated = optimized_df[
        optimized_df.get("gate_ok", pd.Series([False] * len(optimized_df))) == True
    ]
    optimized_gated["archetype_ret"] = optimized_gated.apply(
        lambda r: _archetype_return(r, "ret_mean", "ret_trend"), axis=1
    )
    optimized_returns = optimized_gated["archetype_ret"]

    # 按archetype对比
    comparison = {
        "overall": {
            "baseline": {
                "trades": len(baseline_gated),
                "sharpe": _sharpe(baseline_returns),
                "win_rate": (
                    float((baseline_returns > 0).mean())
                    if len(baseline_returns) > 0
                    else 0.0
                ),
                "avg_return": (
                    float(baseline_returns.mean()) if len(baseline_returns) > 0 else 0.0
                ),
            },
            "optimized": {
                "trades": len(optimized_gated),
                "sharpe": _sharpe(optimized_returns),
                "win_rate": (
                    float((optimized_returns > 0).mean())
                    if len(optimized_returns) > 0
                    else 0.0
                ),
                "avg_return": (
                    float(optimized_returns.mean())
                    if len(optimized_returns) > 0
                    else 0.0
                ),
            },
        },
        "by_archetype": {},
    }

    # 计算变化
    comparison["overall"]["change"] = {
        "trades": comparison["overall"]["optimized"]["trades"]
        - comparison["overall"]["baseline"]["trades"],
        "sharpe": comparison["overall"]["optimized"]["sharpe"]
        - comparison["overall"]["baseline"]["sharpe"],
        "win_rate": comparison["overall"]["optimized"]["win_rate"]
        - comparison["overall"]["baseline"]["win_rate"],
        "avg_return": comparison["overall"]["optimized"]["avg_return"]
        - comparison["overall"]["baseline"]["avg_return"],
    }

    # 按archetype对比
    if (
        "gate_archetype" in baseline_gated.columns
        and "gate_archetype" in optimized_gated.columns
    ):
        for arch in baseline_gated["gate_archetype"].dropna().unique():
            base_arch = baseline_gated[baseline_gated["gate_archetype"] == arch]
            opt_arch = optimized_gated[optimized_gated["gate_archetype"] == arch]

            base_ret = base_arch["archetype_ret"]
            opt_ret = opt_arch["archetype_ret"]

            comparison["by_archetype"][arch] = {
                "baseline": {
                    "trades": len(base_arch),
                    "sharpe": _sharpe(base_ret),
                    "win_rate": (
                        float((base_ret > 0).mean()) if len(base_ret) > 0 else 0.0
                    ),
                },
                "optimized": {
                    "trades": len(opt_arch),
                    "sharpe": _sharpe(opt_ret),
                    "win_rate": (
                        float((opt_ret > 0).mean()) if len(opt_ret) > 0 else 0.0
                    ),
                },
                "change": {
                    "trades": len(opt_arch) - len(base_arch),
                    "sharpe": _sharpe(opt_ret) - _sharpe(base_ret),
                    "win_rate": (
                        float((opt_ret > 0).mean()) if len(opt_ret) > 0 else 0.0
                    )
                    - (float((base_ret > 0).mean()) if len(base_ret) > 0 else 0.0),
                },
            }

    return comparison


def format_comparison_report(comparison: Dict[str, Any]) -> str:
    """格式化对比报告"""
    lines = []
    lines.append("# 优化参数 vs 基线参数对比报告")
    lines.append("")

    # Overall
    lines.append("## 整体性能对比")
    lines.append("")
    overall = comparison["overall"]
    base = overall["baseline"]
    opt = overall["optimized"]
    change = overall["change"]

    lines.append("| 指标 | 基线 | 优化后 | 变化 |")
    lines.append("|------|------|--------|------|")
    lines.append(
        f"| 交易数 | {base['trades']} | {opt['trades']} | {change['trades']:+d} |"
    )
    lines.append(
        f"| Sharpe | {base['sharpe']:.4f} | {opt['sharpe']:.4f} | {change['sharpe']:+.4f} |"
    )
    lines.append(
        f"| 胜率 | {base['win_rate']:.2%} | {opt['win_rate']:.2%} | {change['win_rate']:+.2%} |"
    )
    lines.append(
        f"| 平均收益 | {base['avg_return']:.6f} | {opt['avg_return']:.6f} | {change['avg_return']:+.6f} |"
    )
    lines.append("")

    # By archetype
    if comparison["by_archetype"]:
        lines.append("## 按Archetype对比")
        lines.append("")
        lines.append("| Archetype | 指标 | 基线 | 优化后 | 变化 |")
        lines.append("|-----------|------|------|--------|------|")

        for arch_name, arch_comp in sorted(comparison["by_archetype"].items()):
            base_arch = arch_comp["baseline"]
            opt_arch = arch_comp["optimized"]
            change_arch = arch_comp["change"]

            lines.append(
                f"| {arch_name} | 交易数 | {base_arch['trades']} | {opt_arch['trades']} | {change_arch['trades']:+d} |"
            )
            lines.append(
                f"| {arch_name} | Sharpe | {base_arch['sharpe']:.4f} | {opt_arch['sharpe']:.4f} | {change_arch['sharpe']:+.4f} |"
            )
            lines.append(
                f"| {arch_name} | 胜率 | {base_arch['win_rate']:.2%} | {opt_arch['win_rate']:.2%} | {change_arch['win_rate']:+.2%} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Compare optimized parameters vs baseline")
    p.add_argument(
        "--optimization-dir",
        required=True,
        help="Directory containing optimization results",
    )
    p.add_argument(
        "--baseline-logs", required=True, help="Baseline gated logs file (parquet)"
    )
    p.add_argument(
        "--raw-logs", required=True, help="Raw logs file for re-applying gate (parquet)"
    )
    p.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="Path to execution_archetypes.yaml",
    )
    p.add_argument(
        "--features-store-layer", required=True, help="FeatureStore layer name"
    )
    p.add_argument(
        "--features-store-root",
        default="feature_store",
        help="FeatureStore root directory",
    )
    p.add_argument(
        "--output-dir", required=True, help="Output directory for comparison results"
    )
    p.add_argument(
        "--update-config",
        action="store_true",
        help="Update execution_archetypes.yaml with optimized thresholds",
    )
    args = p.parse_args()

    optimization_dir = Path(args.optimization_dir)
    if not optimization_dir.exists():
        print(
            f"Error: Optimization directory not found: {optimization_dir}",
            file=sys.stderr,
        )
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load optimization results
    print("Loading optimization results...")
    optimization_results = load_optimization_results(optimization_dir)
    print(f"Loaded optimization results for {len(optimization_results)} archetypes")

    # Update config if requested
    config_path = Path(args.execution_archetypes)
    if args.update_config:
        print("\nUpdating execution_archetypes.yaml...")
        update_execution_archetypes_config(
            config_path, optimization_results, backup=True
        )

    # Re-apply gate with optimized parameters
    optimized_logs_path = output_dir / "logs_optimized.parquet"
    print(f"\nRe-applying gate with optimized parameters...")
    cmd = [
        sys.executable,
        "scripts/apply_archetype_gate.py",
        "--logs",
        str(args.raw_logs),
        "--out",
        str(optimized_logs_path),
        "--features-store-layer",
        args.features_store_layer,
        "--features-store-root",
        args.features_store_root,
        "--execution-archetypes",
        str(config_path),
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"Error: Gate application failed", file=sys.stderr)
        return result.returncode

    # Compare performance
    print("\nComparing performance...")
    comparison = compare_performance(
        Path(args.baseline_logs),
        optimized_logs_path,
        output_dir,
    )

    # Save comparison results
    comparison_json = output_dir / "comparison_results.json"
    with open(comparison_json, "w") as f:
        json.dump(comparison, f, indent=2)

    comparison_md = output_dir / "comparison_report.md"
    report = format_comparison_report(comparison)
    comparison_md.write_text(report, encoding="utf-8")

    print(f"\nComparison complete:")
    print(f"  - JSON: {comparison_json}")
    print(f"  - Markdown: {comparison_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
