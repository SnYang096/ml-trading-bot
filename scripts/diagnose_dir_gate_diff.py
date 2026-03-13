#!/usr/bin/env python3
"""深度诊断：向量回测 vs 事件回测 direction + gate 差异

对比相同原始数据上两边的评估结果。
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.time_series_model.archetype.loader import load_strategy_archetype
from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy


def load_direction_config(strategy: str, strategies_root: str = "config/strategies"):
    """加载 direction.yaml"""
    import yaml

    dir_path = Path(strategies_root) / strategy / "archetypes" / "direction.yaml"
    if not dir_path.exists():
        return None
    with open(dir_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_direction_rules_vectorized(df: pd.DataFrame, strategy: str, dir_cfg: dict):
    """向量回测风格 direction 评估"""
    from scripts.backtest_execution_layer import apply_direction_rules

    return apply_direction_rules(df, strategy, dir_cfg)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="me-long", help="策略名")
    parser.add_argument("--rows", type=int, default=200, help="检查行数")
    parser.add_argument("--data-path", default="data/parquet_data")
    parser.add_argument("--test-start", default="2025-08-01")
    parser.add_argument("--test-end", default="2026-02-01")
    args = parser.parse_args()

    strategies_root = "config/strategies"

    # 1. 加载特征数据（用向量回测的 _load_raw_features_for_archetype）
    print(f"\n{'='*70}")
    print(f"📂 加载 {args.strategy} 策略原始数据")
    print(f"{'='*70}")

    from scripts.backtest_execution_layer import _load_raw_features_for_archetype

    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
    df = _load_raw_features_for_archetype(
        arch_name=args.strategy,
        strategies_root=strategies_root,
        symbols=symbols,
        data_path=args.data_path,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    print(f"总行数: {len(df)}")

    # 2. 加载配置
    dir_cfg = load_direction_config(args.strategy, strategies_root)
    archetype = load_strategy_archetype(args.strategy, strategies_root)

    # 初始化事件回测风格的策略
    strat = GenericLiveStrategy(
        strategy_name=args.strategy,
        strategies_root=strategies_root,
    )

    # 3. 向量回测 direction 评估
    print(f"\n{'='*70}")
    print(f"🎯 向量回测 direction 评估")
    print(f"{'='*70}")

    df_copy = df.copy()
    vec_applied = apply_direction_rules_vectorized(df_copy, args.strategy, dir_cfg)
    if vec_applied is None:
        print("❌ 向量回测 direction 未命中")
        return

    vec_dir_count = int((df_copy["entry_direction"] != 0).sum())
    print(
        f"向量回测 direction 有效: {vec_dir_count} / {len(df_copy)} ({100*vec_dir_count/len(df_copy):.1f}%)"
    )

    # 4. 事件回测 direction 评估（逐行）
    print(f"\n{'='*70}")
    print(f"🎯 事件回测 direction 评估 (前 {args.rows} 行)")
    print(f"{'='*70}")

    evt_dir_pass = 0
    evt_gate_pass = 0
    vec_gate_pass = 0

    mismatch_dir = []
    mismatch_gate = []

    sample_df = df.head(args.rows).copy()

    # 向量回测 gate
    from scripts.backtest_execution_layer import _apply_gate_from_yaml_vectorized

    _apply_gate_from_yaml_vectorized(sample_df, args.strategy, strategies_root)

    for idx, row in sample_df.iterrows():
        features = row.to_dict()

        # 事件回测 direction
        evt_dir, _ = strat.direction_evaluator.evaluate(features)
        if evt_dir != 0:
            evt_dir_pass += 1

        # 向量回测 direction (已在 df_copy 中计算)
        vec_dir = df_copy.loc[idx, "entry_direction"] if idx in df_copy.index else 0

        # direction 不匹配
        evt_has_dir = evt_dir != 0
        vec_has_dir = vec_dir != 0
        if evt_has_dir != vec_has_dir:
            if len(mismatch_dir) < 5:
                mismatch_dir.append(
                    {
                        "idx": idx,
                        "evt_dir": evt_dir,
                        "vec_dir": vec_dir,
                    }
                )

        # 事件回测 gate (仅当有 direction 时)
        if evt_dir != 0:
            gate_ok, _, _ = archetype.apply_gate(features, quantiles=None)
            if gate_ok:
                evt_gate_pass += 1

        # 向量回测 gate
        if row.get("gate_decision") == "allow":
            vec_gate_pass += 1

    vec_sample_dir = (
        int((sample_df["entry_direction"] != 0).sum())
        if "entry_direction" in sample_df.columns
        else 0
    )

    print(f"\nDirection 统计 (前 {args.rows} 行):")
    print(f"  事件回测有 direction: {evt_dir_pass} ({100*evt_dir_pass/args.rows:.1f}%)")
    print(
        f"  向量回测有 direction: {vec_sample_dir} ({100*vec_sample_dir/args.rows:.1f}%)"
    )

    if mismatch_dir:
        print(f"\nDirection 不匹配样例:")
        for m in mismatch_dir:
            print(f"  idx={m['idx']}: evt={m['evt_dir']}, vec={m['vec_dir']}")

    print(f"\nGate 统计 (前 {args.rows} 行, 仅有 direction 的行):")
    print(f"  事件回测 gate 通过: {evt_gate_pass}")
    print(f"  向量回测 gate 通过: {vec_gate_pass}")


if __name__ == "__main__":
    main()
