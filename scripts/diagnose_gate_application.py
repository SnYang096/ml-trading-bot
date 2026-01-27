#!/usr/bin/env python3
"""
诊断Gate规则应用逻辑

检查：
1. Gate规则是否正确应用到每个样本
2. 特征值分布，确认是否都在阈值范围内
3. 哪些规则实际生效（veto了交易）
4. 如果需要，建议调整gate规则的阈值

使用方法:
    python scripts/diagnose_gate_application.py \
        --logs results/pipeline_with_reflexivity_2024_full/logs_execution.parquet \
        --execution-archetypes config/nnmultihead/execution_archetypes.yaml \
        --output results/gate_diagnosis.json \
        --feature-store-layer nnmh_highcap6_240T_2024_with_reflexivity \
        --timeframe 240T
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)
from src.time_series_model.live.tree_gate import apply_gate_rules, _eval_when_clause
from src.time_series_model.core.constitution.execution_evidence import (
    load_evidence_quantiles,
)
from scripts.apply_archetype_gate import _read_feature_store_range


def _collect_feature_keys_from_when(when: Any, out: set) -> None:
    if isinstance(when, list):
        for item in when:
            _collect_feature_keys_from_when(item, out)
        return
    if not isinstance(when, dict):
        return
    if "all_of" in when:
        for item in when.get("all_of") or []:
            _collect_feature_keys_from_when(item, out)
        return
    if "any_of" in when:
        for item in when.get("any_of") or []:
            _collect_feature_keys_from_when(item, out)
        return
    if "not" in when:
        _collect_feature_keys_from_when(when.get("not"), out)
        return
    if "key" in when and "op" in when:
        key = str(when.get("key") or "").strip()
        if key:
            out.add(key)
        return
    if "any_key_contains" in when:
        return
    if len(when) == 1:
        k = next(iter(when.keys()))
        if str(k).strip():
            out.add(str(k).strip())


def _collect_when_conditions(when: Any, out: List[Dict[str, Any]]) -> None:
    if isinstance(when, list):
        for item in when:
            _collect_when_conditions(item, out)
        return
    if not isinstance(when, dict):
        return
    if "all_of" in when:
        for item in when.get("all_of") or []:
            _collect_when_conditions(item, out)
        return
    if "any_of" in when:
        for item in when.get("any_of") or []:
            _collect_when_conditions(item, out)
        return
    if "not" in when:
        _collect_when_conditions(when.get("not"), out)
        return
    if "key" in when and "op" in when:
        out.append(
            {
                "key": str(when.get("key") or ""),
                "op": str(when.get("op") or ""),
                "value": when.get("value"),
            }
        )
        return
    if "any_key_contains" in when:
        return
    if len(when) == 1:
        k = next(iter(when.keys()))
        cond = when.get(k) or {}
        if isinstance(cond, dict) and len(cond) == 1:
            op = next(iter(cond.keys()))
            out.append({"key": str(k), "op": str(op), "value": cond.get(op)})


def extract_required_features(
    execution_archetypes_path: str,
) -> List[str]:
    """从execution_archetypes.yaml提取所有gate规则使用的特征"""
    arches = load_execution_archetypes_registry(execution_archetypes_path)
    features = set()

    for arch in arches.values():
        rules = list(getattr(arch, "when_then_rules", []) or [])
        if rules:
            for rule in rules:
                _collect_feature_keys_from_when(rule.get("when"), features)
            continue
        if not arch.gate_rules:
            continue
        raw_rules = arch.gate_rules.get("rules", [])
        for rule in raw_rules:
            feature_key = rule.get("key")
            if feature_key:
                features.add(feature_key)

    return sorted(list(features))


def load_features_from_featurestore(
    logs_df: pd.DataFrame,
    feature_store_root: str,
    feature_store_layer: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """从FeatureStore加载特征并merge到logs DataFrame"""
    symbols = sorted(logs_df["symbol"].astype(str).unique().tolist())

    feats = _read_feature_store_range(
        features_store_root=feature_store_root,
        layer=feature_store_layer,
        symbols=symbols,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
    )

    if feats.empty:
        raise ValueError(
            f"FeatureStore读取失败: layer={feature_store_layer}, "
            f"symbols={symbols}, timeframe={timeframe}"
        )

    # 处理timestamp列
    feats = feats.copy()
    if getattr(feats.index, "name", None) == "timestamp":
        if "timestamp" in feats.columns:
            feats = feats.reset_index(drop=True)
        else:
            feats = feats.reset_index()

    if "timestamp" not in feats.columns:
        if getattr(feats.index, "name", None) == "timestamp":
            feats = feats.reset_index()

    feats["symbol"] = feats["symbol"].astype(str)
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], errors="coerce")
    logs_df = logs_df.copy()
    logs_df["symbol"] = logs_df["symbol"].astype(str)
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"], errors="coerce")

    # Merge特征
    merged = logs_df.merge(
        feats, on=["symbol", "timestamp"], how="left", suffixes=("", "_feat")
    )

    # 处理重复列
    feat_suffix_cols = [c for c in merged.columns if c.endswith("_feat")]
    cols_to_drop = []
    cols_to_rename = {}
    for feat_col in feat_suffix_cols:
        original_col = feat_col[:-5]
        if original_col in merged.columns:
            cols_to_drop.append(feat_col)
        else:
            cols_to_rename[feat_col] = original_col

    if cols_to_drop:
        merged = merged.drop(columns=[c for c in cols_to_drop if c in merged.columns])
    if cols_to_rename:
        merged = merged.rename(
            columns={k: v for k, v in cols_to_rename.items() if k in merged.columns}
        )

    return merged


def analyze_rule_effectiveness(
    df: pd.DataFrame,
    arches: Dict[str, Any],
    quantiles: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    分析每个gate规则的生效情况

    返回：
    - 每个规则的veto次数
    - 每个规则的特征值分布
    - 阈值与特征值的关系
    """
    rule_stats = {}

    for arch_name, arch in arches.items():
        when_then_rules = list(getattr(arch, "when_then_rules", []) or [])
        if when_then_rules:
            arch_stats = []
            for rule in when_then_rules:
                rule_id = str(rule.get("id") or rule.get("name") or "unknown")
                phase = str(rule.get("phase") or "")
                action = str(rule.get("then", {}).get("action") or "")
                when = rule.get("when")
                conditions: List[Dict[str, Any]] = []
                _collect_when_conditions(when, conditions)
                feature_keys = sorted(
                    {c.get("key") for c in conditions if c.get("key")}
                )

                matched = df.apply(
                    lambda r: _eval_when_clause(
                        when, features=r.to_dict(), quantiles=quantiles
                    ),
                    axis=1,
                )

                if action == "deny" and phase in ("safety", "exclusions"):
                    veto_count = int(matched.sum())
                elif action == "require" and phase in ("preconditions", "evidence"):
                    veto_count = int((~matched).sum())
                else:
                    veto_count = 0

                feature_stats = {}
                for key in feature_keys:
                    if key in df.columns:
                        values = df[key].dropna()
                        if len(values) > 0:
                            feature_stats[key] = {
                                "min": float(values.min()),
                                "max": float(values.max()),
                                "mean": float(values.mean()),
                                "median": float(values.median()),
                                "p5": float(values.quantile(0.05)),
                                "p95": float(values.quantile(0.95)),
                            }

                arch_stats.append(
                    {
                        "rule_name": rule_id,
                        "phase": phase,
                        "action": action,
                        "feature_keys": feature_keys,
                        "ops": sorted({c.get("op") for c in conditions if c.get("op")}),
                        "veto_count": veto_count,
                        "veto_rate": (
                            float(veto_count / len(df)) if len(df) > 0 else 0.0
                        ),
                        "match_rate": float(matched.mean()) if len(df) > 0 else 0.0,
                        "feature_stats": feature_stats,
                    }
                )
            if arch_stats:
                rule_stats[arch_name] = arch_stats
            continue

        if not arch.gate_rules:
            continue

        rules = arch.gate_rules.get("rules", [])
        arch_stats = []

        for rule in rules:
            rule_name = rule.get("name", "unknown")
            feature_key = rule.get("key")
            rule_kind = rule.get("kind", "")
            threshold = rule.get("threshold") or rule.get("quantile")

            if not feature_key or feature_key not in df.columns:
                continue

            feature_values = df[feature_key].dropna()
            if len(feature_values) == 0:
                continue

            veto_count = 0
            if rule_kind in ("value_lt", "quantile_lt", "value_lte", "quantile_lte"):
                if threshold is not None:
                    veto_count = int((feature_values < threshold).sum())
            elif rule_kind in ("value_gt", "quantile_gt", "value_gte", "quantile_gte"):
                if threshold is not None:
                    veto_count = int((feature_values > threshold).sum())

            feature_stats = {
                "min": float(feature_values.min()),
                "max": float(feature_values.max()),
                "mean": float(feature_values.mean()),
                "median": float(feature_values.median()),
                "p5": float(feature_values.quantile(0.05)),
                "p95": float(feature_values.quantile(0.95)),
            }

            threshold_position = None
            if threshold is not None:
                if rule_kind.startswith("quantile_"):
                    threshold_position = threshold
                else:
                    if len(feature_values) > 0:
                        threshold_position = float((feature_values < threshold).mean())

            arch_stats.append(
                {
                    "rule_name": rule_name,
                    "feature_key": feature_key,
                    "rule_kind": rule_kind,
                    "threshold": threshold,
                    "threshold_position": threshold_position,
                    "veto_count": veto_count,
                    "veto_rate": (
                        float(veto_count / len(feature_values))
                        if len(feature_values) > 0
                        else 0.0
                    ),
                    "feature_stats": feature_stats,
                }
            )

        if arch_stats:
            rule_stats[arch_name] = arch_stats

    return rule_stats


def diagnose_gate_application(
    df: pd.DataFrame,
    arches: Dict[str, Any],
    quantiles: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    诊断gate规则应用逻辑

    检查：
    1. 每个样本应用gate规则的结果
    2. 哪些规则实际veto了交易
    3. 特征值分布与阈值的关系
    """
    print("\n📊 诊断Gate规则应用逻辑...")

    # 应用gate规则并记录详细信息
    gate_results = []
    veto_reasons_by_rule = {}

    for idx, row in df.iterrows():
        features = row.to_dict()

        sample_result = {
            "index": int(idx),
            "symbol": str(row.get("symbol", "")),
            "timestamp": str(row.get("timestamp", "")),
            "gate_ok": False,
            "matched_archetype": None,
            "veto_rules": [],
        }

        # 尝试每个archetype
        for arch_name, arch in arches.items():
            if not arch.gate_rules:
                sample_result["gate_ok"] = True
                sample_result["matched_archetype"] = arch_name
                break

            ok, reasons = apply_gate_rules(
                gate_rules=arch.gate_rules,
                features=features,
                quantiles=quantiles,
            )

            if ok:
                sample_result["gate_ok"] = True
                sample_result["matched_archetype"] = arch_name
                break
            else:
                # 记录veto原因
                for reason in reasons:
                    sample_result["veto_rules"].append(str(reason))

        gate_results.append(sample_result)

        # 统计veto规则
        for rule_name in sample_result["veto_rules"]:
            if rule_name not in veto_reasons_by_rule:
                veto_reasons_by_rule[rule_name] = 0
            veto_reasons_by_rule[rule_name] += 1

    # 分析规则生效情况
    rule_stats = analyze_rule_effectiveness(df, arches, quantiles)

    # 统计结果
    total_samples = len(df)
    passed_samples = sum(1 for r in gate_results if r["gate_ok"])
    vetoed_samples = total_samples - passed_samples

    return {
        "total_samples": total_samples,
        "passed_samples": passed_samples,
        "vetoed_samples": vetoed_samples,
        "pass_rate": (
            float(passed_samples / total_samples) if total_samples > 0 else 0.0
        ),
        "veto_reasons_by_rule": veto_reasons_by_rule,
        "rule_stats": rule_stats,
        "archetype_distribution": {
            arch: sum(1 for r in gate_results if r["matched_archetype"] == arch)
            for arch in arches.keys()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="诊断Gate规则应用逻辑",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--logs",
        required=True,
        help="logs文件（parquet）",
    )
    parser.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="execution_archetypes.yaml路径",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出JSON文件",
    )
    parser.add_argument(
        "--feature-store-root",
        default="feature_store",
        help="FeatureStore根目录",
    )
    parser.add_argument(
        "--feature-store-layer",
        default=None,
        help="FeatureStore layer名称",
    )
    parser.add_argument(
        "--evidence-quantiles",
        default=None,
        help="evidence_quantiles.json路径（用于quantile_*规则）",
    )
    parser.add_argument(
        "--timeframe",
        default="240T",
        help="时间框架",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="开始日期",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="结束日期",
    )

    args = parser.parse_args()

    # 读取数据
    print("📊 读取数据...")
    df = pd.read_parquet(args.logs)
    print(f"✅ 读取数据: {len(df)} 行")

    # 加载特征
    if args.feature_store_layer:
        print("📊 从FeatureStore加载特征...")
        required_features = extract_required_features(args.execution_archetypes)
        available_features = set(df.columns)
        missing_features = [f for f in required_features if f not in available_features]

        if missing_features:
            print(f"⚠️  缺少 {len(missing_features)} 个特征，从FeatureStore加载...")
            df = load_features_from_featurestore(
                df,
                args.feature_store_root,
                args.feature_store_layer,
                args.timeframe,
                args.start_date,
                args.end_date,
            )
            print(f"✅ 特征加载完成，DataFrame现在有 {len(df.columns)} 列")

    # 加载quantiles（用于quantile_*规则）
    quantiles = (
        load_evidence_quantiles(args.evidence_quantiles)
        if args.evidence_quantiles
        else None
    )

    # 加载archetypes
    print("📊 加载archetypes配置...")
    arches = load_execution_archetypes_registry(args.execution_archetypes)
    arches = {
        k: v for k, v in arches.items() if k != "VolMeanCompressionExpansionReversion"
    }

    # 诊断gate规则应用
    diagnosis = diagnose_gate_application(df, arches, quantiles)

    print(f"\n✅ 诊断完成:")
    print(f"   总样本数: {diagnosis['total_samples']}")
    print(f"   通过样本数: {diagnosis['passed_samples']}")
    print(f"   被veto样本数: {diagnosis['vetoed_samples']}")
    print(f"   通过率: {diagnosis['pass_rate']:.4f}")

    # 显示最常veto的规则
    if diagnosis["veto_reasons_by_rule"]:
        print(f"\n   最常veto的规则:")
        sorted_veto = sorted(
            diagnosis["veto_reasons_by_rule"].items(), key=lambda x: x[1], reverse=True
        )[:10]
        for rule_name, count in sorted_veto:
            print(f"     {rule_name}: {count} 次")

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ 诊断结果已保存: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
