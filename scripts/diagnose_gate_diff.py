#!/usr/bin/env python3
"""诊断事件回测 vs 向量回测 gate 通过率差异

核心问题：
- 事件回测: 7751 通过 gate (29.3%)
- 向量回测: 2322 通过 gate (8.8%)
- 差异 3.3x

可能原因：
1. 输入特征不同
2. 规则加载不同
3. 评估逻辑不同
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.time_series_model.archetype.loader import (
    load_strategy_archetype,
    GateConfig,
    _evaluate_when_clause,
)


def load_gate_config_raw(strategy: str, strategies_root: str = "config/strategies"):
    """向量回测风格: 直接加载 gate.yaml"""
    gate_path = Path(strategies_root) / strategy / "archetypes" / "gate.yaml"
    if not gate_path.exists():
        return {}
    with open(gate_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _eval_when_vectorized_single(when: dict, row: pd.Series) -> bool:
    """向量回测风格: 单行评估 (复制自 backtest_execution_layer.py)"""
    if not when:
        return False

    # all_of: AND
    if "all_of" in when:
        conditions = when["all_of"]
        min_matches = int(when.get("min_matches", len(conditions)))
        match_count = sum(1 for c in conditions if _eval_when_vectorized_single(c, row))
        return match_count >= min_matches

    # any_of: OR
    if "any_of" in when:
        conditions = when["any_of"]
        min_matches = int(when.get("min_matches", 1))
        match_count = sum(1 for c in conditions if _eval_when_vectorized_single(c, row))
        return match_count >= min_matches

    # 单条件: AND 逻辑
    for feature, conditions in when.items():
        if feature in ("all_of", "any_of", "min_matches"):
            continue
        if feature not in row.index:
            return False
        if not isinstance(conditions, dict):
            continue
        value = row[feature]
        if pd.isna(value):
            return False
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False
        for op, threshold in conditions.items():
            if op == "on_missing":
                continue
            try:
                threshold = float(threshold)
            except (TypeError, ValueError):
                continue
            if op == "value_gt" and not (value > threshold):
                return False
            elif op in ("value_gte", "value_ge") and not (value >= threshold):
                return False
            elif op == "value_lt" and not (value < threshold):
                return False
            elif op in ("value_lte", "value_le") and not (value <= threshold):
                return False
    return True


def eval_gate_vector_style(row: pd.Series, gate_cfg: dict) -> tuple[bool, list[str]]:
    """向量回测风格 gate 评估"""
    hard_gates = gate_cfg.get("hard_gates", [])
    deny_reasons = []

    for rule in hard_gates:
        when = rule.get("when", {})
        action = rule.get("then", {}).get("action", "deny")
        if action != "deny":
            continue

        rule_id = rule.get("id", "unknown")
        if _eval_when_vectorized_single(when, row):
            deny_reasons.append(rule_id)
            return False, deny_reasons

    return True, []


def eval_gate_event_style(features: dict, archetype) -> tuple[bool, list[str]]:
    """事件回测风格 gate 评估"""
    passed, deny_reasons, _ = archetype.apply_gate(features, quantiles=None)
    return passed, deny_reasons


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="me-long", help="策略名")
    parser.add_argument("--logs", required=True, help="logs_gated.parquet 路径")
    parser.add_argument("--rows", type=int, default=200, help="检查行数")
    args = parser.parse_args()

    strategies_root = "config/strategies"

    # 1. 加载两套 gate 配置
    print(f"\n{'='*70}")
    print(f"📋 加载 {args.strategy} 策略 gate 配置")
    print(f"{'='*70}")

    # 事件回测风格
    archetype = load_strategy_archetype(args.strategy, strategies_root)
    event_rules = list(archetype.gate.all_rules)
    print(f"\n[事件回测] archetype.gate.all_rules: {len(event_rules)} 条规则")
    for r in event_rules:
        print(f"   - {r.id}: when={r.when}, action={r.then.get('action')}")

    # 向量回测风格
    gate_cfg = load_gate_config_raw(args.strategy, strategies_root)
    vector_rules = gate_cfg.get("hard_gates", [])
    print(f"\n[向量回测] gate_cfg['hard_gates']: {len(vector_rules)} 条规则")
    for r in vector_rules:
        print(
            f"   - {r.get('id')}: when={r.get('when')}, action={r.get('then',{}).get('action')}"
        )

    # 2. 加载数据
    print(f"\n{'='*70}")
    print(f"📂 加载数据: {args.logs}")
    print(f"{'='*70}")

    df = pd.read_parquet(args.logs)
    print(f"总行数: {len(df)}")

    # 检查需要的特征列是否存在
    needed_features = set()
    for r in event_rules:
        for k in r.when.keys():
            if k not in ("all_of", "any_of", "min_matches"):
                needed_features.add(k)
            if k == "all_of" or k == "any_of":
                for sub in r.when[k]:
                    for sk in sub.keys():
                        if sk not in ("all_of", "any_of", "min_matches"):
                            needed_features.add(sk)

    available = [f for f in needed_features if f in df.columns]
    missing = [f for f in needed_features if f not in df.columns]
    print(f"\n需要的特征: {needed_features}")
    print(f"  可用: {available}")
    print(f"  缺失: {missing}")

    # 3. 逐行对比
    print(f"\n{'='*70}")
    print(f"🔍 逐行对比 gate 评估 (前 {args.rows} 行)")
    print(f"{'='*70}")

    event_pass = 0
    vector_pass = 0
    mismatch_count = 0
    mismatch_details = []

    sample_df = df.head(args.rows).copy()

    for idx, row in sample_df.iterrows():
        # 转换为 features dict (事件回测风格)
        features = row.to_dict()

        # 事件回测评估
        event_ok, event_reasons = eval_gate_event_style(features, archetype)

        # 向量回测评估
        vector_ok, vector_reasons = eval_gate_vector_style(row, gate_cfg)

        if event_ok:
            event_pass += 1
        if vector_ok:
            vector_pass += 1

        if event_ok != vector_ok:
            mismatch_count += 1
            if len(mismatch_details) < 10:
                mismatch_details.append(
                    {
                        "idx": idx,
                        "event_ok": event_ok,
                        "event_reasons": event_reasons,
                        "vector_ok": vector_ok,
                        "vector_reasons": vector_reasons,
                        # 关键特征值
                        "features": {
                            f: features.get(f) for f in needed_features if f in features
                        },
                    }
                )

    print(f"\n统计 (前 {args.rows} 行):")
    print(f"  事件回测 PASS: {event_pass} ({100*event_pass/args.rows:.1f}%)")
    print(f"  向量回测 PASS: {vector_pass} ({100*vector_pass/args.rows:.1f}%)")
    print(f"  不匹配数: {mismatch_count} ({100*mismatch_count/args.rows:.1f}%)")

    if mismatch_details:
        print(f"\n不匹配样例 (前 10 个):")
        for m in mismatch_details:
            print(f"\n  idx={m['idx']}:")
            print(
                f"    事件回测: {'PASS' if m['event_ok'] else 'DENY'} reasons={m['event_reasons']}"
            )
            print(
                f"    向量回测: {'PASS' if m['vector_ok'] else 'DENY'} reasons={m['vector_reasons']}"
            )
            print(f"    特征值: {m['features']}")

    # 4. 检查原始 gate_decision 列
    if "gate_decision" in df.columns:
        print(f"\n{'='*70}")
        print(f"📊 原始 gate_decision 列统计")
        print(f"{'='*70}")
        print(df["gate_decision"].value_counts())

        # 对比重新评估 vs 原始
        print(f"\n对比重新评估 vs 原始 (全量数据):")
        recomputed_pass = 0
        for _, row in df.iterrows():
            features = row.to_dict()
            event_ok, _ = eval_gate_event_style(features, archetype)
            if event_ok:
                recomputed_pass += 1

        original_pass = int((df["gate_decision"] == "allow").sum())
        print(f"  原始 allow: {original_pass} ({100*original_pass/len(df):.1f}%)")
        print(
            f"  重新评估 PASS: {recomputed_pass} ({100*recomputed_pass/len(df):.1f}%)"
        )


if __name__ == "__main__":
    main()
