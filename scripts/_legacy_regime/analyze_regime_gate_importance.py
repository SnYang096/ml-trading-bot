#!/usr/bin/env python3
"""
系统分析regime vs gate的重要性，计算独立贡献和影响
"""

import json
import pandas as pd
from pathlib import Path


def analyze_regime_gate_importance():
    """分析regime vs gate的重要性"""

    # 加载实验汇总
    with open("results/experiments_summary.json", "r") as f:
        summary = json.load(f)

    configs = {
        "baseline": "有Regime + 有Gate + 有Semantic",
        "only_gate_rules": "有Regime + 有Gate + 无Semantic",
        "no_regime_filter": "无Regime + 有Gate + 有Semantic",
        "no_gate_veto": "有Regime + 无Gate + 有Semantic",
        "no_semantic_veto": "有Regime + 有Gate + 无Semantic",
        "no_regime_no_veto": "无Regime + 无Gate + 有Semantic",
        "all_veto_off": "无Regime + 无Gate + 无Semantic",
    }

    print("=" * 80)
    print("Regime vs Gate重要性分析")
    print("=" * 80)

    # 提取KPI
    kpis = {}
    for name, desc in configs.items():
        if name in summary["experiments"]:
            kpi = summary["experiments"][name]["kpi"]["overall"]
            kpis[name] = {
                "sharpe": kpi.get("sharpe", 0),
                "trades": kpi.get("trade_count", 0),
                "win_rate": kpi.get("win_rate", 0),
                "description": desc,
            }

    results = {
        "configs": kpis,
        "analysis": {},
    }

    # 1. Regime的影响（固定Gate和Semantic）
    print("\n1. Regime的影响（固定Gate和Semantic）:")
    print(
        "   baseline (有regime + 有gate): Sharpe {:.3f}, 交易数 {}".format(
            kpis["baseline"]["sharpe"], kpis["baseline"]["trades"]
        )
    )
    print(
        "   no_regime_filter (无regime + 有gate): Sharpe {:.3f}, 交易数 {}".format(
            kpis["no_regime_filter"]["sharpe"], kpis["no_regime_filter"]["trades"]
        )
    )
    regime_impact = kpis["baseline"]["sharpe"] - kpis["no_regime_filter"]["sharpe"]
    regime_trade_impact = (
        kpis["no_regime_filter"]["trades"] - kpis["baseline"]["trades"]
    )
    print(
        f"   → Regime的影响: {regime_impact:.3f} Sharpe, 减少交易数 {regime_trade_impact}"
    )

    results["analysis"]["regime_impact"] = {
        "sharpe_impact": float(regime_impact),
        "trade_impact": int(regime_trade_impact),
    }

    # 2. Gate的影响（固定Regime和Semantic）
    print("\n2. Gate的影响（固定Regime和Semantic）:")
    print(
        "   baseline (有regime + 有gate): Sharpe {:.3f}, 交易数 {}".format(
            kpis["baseline"]["sharpe"], kpis["baseline"]["trades"]
        )
    )
    print(
        "   no_gate_veto (有regime + 无gate): Sharpe {:.3f}, 交易数 {}".format(
            kpis["no_gate_veto"]["sharpe"], kpis["no_gate_veto"]["trades"]
        )
    )
    gate_impact = kpis["baseline"]["sharpe"] - kpis["no_gate_veto"]["sharpe"]
    gate_trade_impact = kpis["no_gate_veto"]["trades"] - kpis["baseline"]["trades"]
    print(f"   → Gate的影响: {gate_impact:.3f} Sharpe, 减少交易数 {gate_trade_impact}")

    results["analysis"]["gate_impact"] = {
        "sharpe_impact": float(gate_impact),
        "trade_impact": int(gate_trade_impact),
    }

    # 3. Semantic的影响（固定Regime和Gate）
    print("\n3. Semantic的影响（固定Regime和Gate）:")
    print(
        "   baseline (有regime + 有gate + 有semantic): Sharpe {:.3f}, 交易数 {}".format(
            kpis["baseline"]["sharpe"], kpis["baseline"]["trades"]
        )
    )
    print(
        "   only_gate_rules (有regime + 有gate + 无semantic): Sharpe {:.3f}, 交易数 {}".format(
            kpis["only_gate_rules"]["sharpe"], kpis["only_gate_rules"]["trades"]
        )
    )
    semantic_impact = kpis["baseline"]["sharpe"] - kpis["only_gate_rules"]["sharpe"]
    semantic_trade_impact = (
        kpis["only_gate_rules"]["trades"] - kpis["baseline"]["trades"]
    )
    print(
        f"   → Semantic的影响: {semantic_impact:.3f} Sharpe, 增加交易数 {semantic_trade_impact}"
    )

    results["analysis"]["semantic_impact"] = {
        "sharpe_impact": float(semantic_impact),
        "trade_impact": int(semantic_trade_impact),
    }

    # 4. 综合分析
    print("\n4. 综合分析:")
    print(
        f"   Regime影响: {regime_impact:.3f} Sharpe (减少 {regime_trade_impact} 交易)"
    )
    print(f"   Gate影响: {gate_impact:.3f} Sharpe (减少 {gate_trade_impact} 交易)")
    print(
        f"   Semantic影响: {semantic_impact:.3f} Sharpe (增加 {semantic_trade_impact} 交易)"
    )

    # 计算相对重要性
    total_impact = abs(regime_impact) + abs(gate_impact) + abs(semantic_impact)
    if total_impact > 0:
        regime_importance = abs(regime_impact) / total_impact * 100
        gate_importance = abs(gate_impact) / total_impact * 100
        semantic_importance = abs(semantic_impact) / total_impact * 100

        print(f"\n   相对重要性（按Sharpe影响）:")
        print(f"   Regime: {regime_importance:.1f}%")
        print(f"   Gate: {gate_importance:.1f}%")
        print(f"   Semantic: {semantic_importance:.1f}%")

        results["analysis"]["relative_importance"] = {
            "regime_pct": float(regime_importance),
            "gate_pct": float(gate_importance),
            "semantic_pct": float(semantic_importance),
        }

    # 判断哪个更重要
    if abs(regime_impact) > abs(gate_impact):
        conclusion = "Regime的影响更大"
    elif abs(gate_impact) > abs(regime_impact):
        conclusion = "Gate的影响更大"
    else:
        conclusion = "Regime和Gate的影响相当"

    print(f"\n   结论: {conclusion}")
    results["analysis"]["conclusion"] = conclusion

    # 5. 分析交易数变化模式
    print("\n5. 交易数变化模式:")
    print("   从all_veto_off到baseline的过滤效果:")
    if "all_veto_off" in kpis and "baseline" in kpis:
        total_filtered = kpis["all_veto_off"]["trades"] - kpis["baseline"]["trades"]
        regime_filtered = (
            kpis["no_regime_filter"]["trades"] - kpis["baseline"]["trades"]
        )
        gate_filtered = kpis["no_gate_veto"]["trades"] - kpis["baseline"]["trades"]

        print(f"   总过滤: {total_filtered} 交易")
        print(
            f"   Regime过滤: {regime_filtered} 交易 ({regime_filtered/total_filtered*100:.1f}%)"
        )
        print(
            f"   Gate过滤: {gate_filtered} 交易 ({gate_filtered/total_filtered*100:.1f}%)"
        )
        print(f"   其他过滤: {total_filtered - regime_filtered - gate_filtered} 交易")

        results["analysis"]["trade_filtering"] = {
            "total_filtered": int(total_filtered),
            "regime_filtered": int(regime_filtered),
            "gate_filtered": int(gate_filtered),
            "regime_pct": (
                float(regime_filtered / total_filtered * 100)
                if total_filtered > 0
                else 0
            ),
            "gate_pct": (
                float(gate_filtered / total_filtered * 100) if total_filtered > 0 else 0
            ),
        }

    # 6. 按Archetype分析
    print("\n6. 按Archetype分析（baseline vs no_regime_filter）:")
    if (
        "baseline" in summary["experiments"]
        and "no_regime_filter" in summary["experiments"]
    ):
        baseline_archetypes = summary["experiments"]["baseline"]["kpi"].get(
            "by_archetype", {}
        )
        no_regime_archetypes = summary["experiments"]["no_regime_filter"]["kpi"].get(
            "by_archetype", {}
        )

        archetype_analysis = {}
        for arch in ["TC", "TE", "FR", "ET"]:
            if arch in baseline_archetypes and arch in no_regime_archetypes:
                baseline_kpi = baseline_archetypes[arch]
                no_regime_kpi = no_regime_archetypes[arch]

                baseline_sharpe = baseline_kpi.get("sharpe", 0)
                no_regime_sharpe = no_regime_kpi.get("sharpe", 0)
                baseline_trades = baseline_kpi.get("trade_count", 0)
                no_regime_trades = no_regime_kpi.get("trade_count", 0)

                sharpe_diff = baseline_sharpe - no_regime_sharpe
                trade_diff = no_regime_trades - baseline_trades

                archetype_analysis[arch] = {
                    "baseline_sharpe": float(baseline_sharpe),
                    "no_regime_sharpe": float(no_regime_sharpe),
                    "sharpe_impact": float(sharpe_diff),
                    "baseline_trades": int(baseline_trades),
                    "no_regime_trades": int(no_regime_trades),
                    "trade_impact": int(trade_diff),
                }

                print(f"   {arch}:")
                print(
                    f"     baseline: Sharpe {baseline_sharpe:.3f}, {baseline_trades} 交易"
                )
                print(
                    f"     no_regime: Sharpe {no_regime_sharpe:.3f}, {no_regime_trades} 交易"
                )
                print(f"     impact: {sharpe_diff:+.3f} Sharpe, {trade_diff:+d} 交易")

        results["analysis"]["by_archetype"] = archetype_analysis

    # 保存结果
    output_file = Path("results/regime_gate_importance_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n分析结果已保存到: {output_file}")

    return results


if __name__ == "__main__":
    analyze_regime_gate_importance()
