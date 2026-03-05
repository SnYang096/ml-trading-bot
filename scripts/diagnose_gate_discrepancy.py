#!/usr/bin/env python3
"""
诊断事件回测 vs 向量回测 gate 评估差异

用法:
    python scripts/diagnose_gate_discrepancy.py --strategy me --rows 100
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import yaml
import pandas as pd
import numpy as np

from src.time_series_model.archetype.loader import (
    load_strategy_archetype,
    _evaluate_when_clause,
)
from src.data_tools.data_handler import DataHandler


def load_gate_config_raw(strategy: str, strategies_root: str = "config/strategies"):
    """向量回测风格: 直接加载 YAML dict"""
    gate_path = Path(strategies_root) / strategy / "archetypes" / "gate.yaml"
    if not gate_path.exists():
        return {}
    with open(gate_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def eval_gate_event_style(features: dict, strategy: str, strategies_root: str):
    """事件回测风格: 使用 StrategyArchetype.apply_gate()"""
    arch = load_strategy_archetype(strategy, strategies_root)
    passed, reasons, weight = arch.apply_gate(features, quantiles=None)
    return passed, reasons


def eval_gate_vector_style(row: pd.Series, gate_cfg: dict):
    """向量回测风格: 使用 hard_gates 规则"""
    hard_gates = gate_cfg.get("hard_gates", [])

    for rule in hard_gates:
        when = rule.get("when", {})
        action = rule.get("then", {}).get("action", "deny")
        if action != "deny":
            continue

        # 简化的单条件评估
        matched = _eval_when_simple(when, row)
        if matched:
            return False, [rule.get("id", "unknown")]

    return True, []


def _eval_when_simple(when: dict, row: pd.Series) -> bool:
    """简化的 when 评估 (对标向量回测)"""
    if not when:
        return False

    # all_of
    if "all_of" in when:
        conditions = when["all_of"]
        min_matches = when.get("min_matches", len(conditions))
        matches = sum(1 for c in conditions if _eval_when_simple(c, row))
        return matches >= min_matches

    # any_of
    if "any_of" in when:
        conditions = when["any_of"]
        min_matches = when.get("min_matches", 1)
        matches = sum(1 for c in conditions if _eval_when_simple(c, row))
        return matches >= min_matches

    # 单条件
    for feature, cond in when.items():
        if feature in ("all_of", "any_of", "min_matches"):
            continue
        if not isinstance(cond, dict):
            continue

        value = row.get(feature)
        if value is None or pd.isna(value):
            return False  # 缺失特征 = 不匹配

        try:
            value = float(value)
        except:
            return False

        for op, threshold in cond.items():
            if op == "on_missing":
                continue
            try:
                threshold = float(threshold)
            except:
                continue

            if op == "value_gt":
                if not (value > threshold):
                    return False
            elif op in ("value_gte", "value_ge"):
                if not (value >= threshold):
                    return False
            elif op == "value_lt":
                if not (value < threshold):
                    return False
            elif op in ("value_lte", "value_le"):
                if not (value <= threshold):
                    return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", "-s", required=True)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--strategies-root", default="config/strategies")
    parser.add_argument("--data-path", default="data/parquet_data")
    args = parser.parse_args()

    print(f"=" * 70)
    print(f"诊断 {args.strategy} Gate 评估差异")
    print(f"=" * 70)

    # 加载数据
    dh = DataHandler(args.data_path)
    timeframe = "240T" if args.strategy in ("bpc", "fer") else "60T"

    symbols = ["BTCUSDT", "ETHUSDT"]
    all_features = []

    for sym in symbols:
        df = dh.load_ohlcv(sym, timeframe, "2025-06-01", "2025-12-31")
        if df is not None and not df.empty:
            df["symbol"] = sym
            all_features.append(df)

    if not all_features:
        print("❌ 无数据")
        return

    features_df = pd.concat(all_features, axis=0)
    print(f"加载 {len(features_df)} 行特征")

    # 加载两种风格的 gate 配置
    arch = load_strategy_archetype(args.strategy, args.strategies_root)
    gate_cfg_raw = load_gate_config_raw(args.strategy, args.strategies_root)

    print(f"\n事件回测: GateConfig.all_rules = {len(arch.gate.all_rules)} 条规则")
    for r in arch.gate.all_rules:
        print(f"  - {r.id}: {r.reason[:50]}...")

    print(f"\n向量回测: hard_gates = {len(gate_cfg_raw.get('hard_gates', []))} 条规则")
    for r in gate_cfg_raw.get("hard_gates", []):
        print(f"  - {r.get('id')}: {r.get('reason', '')[:50]}...")

    # 对比评估
    sample = features_df.head(args.rows).copy()

    event_pass = 0
    vector_pass = 0
    mismatch = 0

    for idx, row in sample.iterrows():
        features_dict = {
            str(k): float(v)
            for k, v in row.items()
            if v is not None and not pd.isna(v) and isinstance(v, (int, float))
        }

        # 事件回测风格
        e_passed, e_reasons = eval_gate_event_style(
            features_dict, args.strategy, args.strategies_root
        )

        # 向量回测风格
        v_passed, v_reasons = eval_gate_vector_style(row, gate_cfg_raw)

        if e_passed:
            event_pass += 1
        if v_passed:
            vector_pass += 1
        if e_passed != v_passed:
            mismatch += 1
            if mismatch <= 5:  # 打印前5个不匹配
                print(f"\n⚠️  行 {idx} 不匹配:")
                print(f"   事件回测: {'PASS' if e_passed else 'DENY'} {e_reasons}")
                print(f"   向量回测: {'PASS' if v_passed else 'DENY'} {v_reasons}")
                # 打印相关特征值
                for r in arch.gate.all_rules:
                    for feat in r.when.keys():
                        if feat not in ("all_of", "any_of", "min_matches"):
                            val = features_dict.get(feat, "MISSING")
                            print(f"   {feat} = {val}")

    print(f"\n" + "=" * 70)
    print(f"统计 (前 {args.rows} 行):")
    print(f"  事件回测 PASS: {event_pass} ({event_pass/args.rows:.1%})")
    print(f"  向量回测 PASS: {vector_pass} ({vector_pass/args.rows:.1%})")
    print(f"  不匹配数: {mismatch} ({mismatch/args.rows:.1%})")
    print(f"=" * 70)


if __name__ == "__main__":
    main()
