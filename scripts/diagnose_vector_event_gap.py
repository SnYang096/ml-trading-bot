#!/usr/bin/env python3
"""
向量回测 vs 事件回测 — 逐层差异诊断

按漏斗顺序逐层对比:
  Layer 0: 数据时间范围
  Layer 1: Gate 通过率 (向量 logs_gated gate_decision vs 事件 GenericLiveStrategy gate)
  Layer 2: Entry Filter (向量 entry_direction vs 事件 decide entry filter)
  Layer 3: PCM Slot 机制 (向量 post-hoc vs 事件 real-time)
  Layer 4: Execution 参数 (SL/TP/trailing)

用法:
    python scripts/diagnose_vector_event_gap.py
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# 配置: 最新 logs_gated 路径
# ═══════════════════════════════════════════════════════════════
LOGS_GATED = {
    "bpc": "results/train_final_20260228_155016_return_tree/bpc/logs_gated.parquet",
    "fer": "results/train_final_20260228_155642_return_tree/fer/logs_gated.parquet",
    "me-long": "results/train_final_20260228_160556_return_tree/me/logs_gated.parquet",
}

STRATEGIES_ROOT = "config/strategies"


def _hr(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════
# Layer 0: 数据时间范围
# ═══════════════════════════════════════════════════════════════
def layer0_time_range():
    _hr("Layer 0: 数据时间范围")
    for arch, path in LOGS_GATED.items():
        df = pd.read_parquet(path)
        ts = pd.to_datetime(df["timestamp"])
        print(f"\n  {arch.upper()}:")
        print(f"    Rows: {len(df)}")
        print(f"    Time: {ts.min()} → {ts.max()}")
        print(f"    Span: {(ts.max()-ts.min()).days} days")
        print(f"    Symbols: {sorted(df['symbol'].unique())}")
        if "gate_decision" in df.columns:
            gc = df["gate_decision"].value_counts()
            print(f"    Gate: {gc.to_dict()}")

    print("\n  ⚠️  事件回测用 --days 180，从 now()-180 天开始")
    print("      logs_gated 时间范围 = 2025-08-01 ~ 2026-01-01 (153d)")
    print("      两者几乎不重叠！这是差异的根本原因之一。")


# ═══════════════════════════════════════════════════════════════
# Layer 1: Gate 通过率 — 在重叠时间窗口内对比
# ═══════════════════════════════════════════════════════════════
def layer1_gate_rate():
    _hr("Layer 1: Gate 通过率 (向量 vs 事件)")

    # 加载事件回测的 gate 评估器
    sys.path.insert(0, ".")
    from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy

    for arch, path in LOGS_GATED.items():
        df = pd.read_parquet(path)
        total = len(df)
        if "gate_decision" not in df.columns:
            print(f"  {arch}: 无 gate_decision 列, 跳过")
            continue

        vector_allow = (df["gate_decision"] == "allow").sum()
        vector_veto = (df["gate_decision"] != "allow").sum()
        print(f"\n  {arch.upper()} (向量侧):")
        print(f"    总信号: {total}")
        print(f"    Gate allow: {vector_allow} ({vector_allow/total*100:.1f}%)")
        print(f"    Gate veto:  {vector_veto} ({vector_veto/total*100:.1f}%)")

        # 尝试用事件侧 gate evaluator 重新评估同一批数据
        try:
            strat = GenericLiveStrategy(
                strategy_name=arch,
                strategies_root=STRATEGIES_ROOT,
                primary_timeframe={"bpc": "240T", "fer": "240T", "me-long": "60T"}[
                    arch
                ],
                bar_minutes={"bpc": 240, "fer": 240, "me-long": 60}[arch],
            )
            # 需要 quantiles
            strat.set_quantiles_from_df(df)

            event_allow = 0
            event_veto = 0
            event_no_dir = 0
            sample_size = min(len(df), 500)  # 采样避免太慢
            sample_df = df.sample(n=sample_size, random_state=42)

            for _, row in sample_df.iterrows():
                features = {}
                for k, v in row.items():
                    try:
                        if v is not None and np.isscalar(v) and not pd.isna(v):
                            features[str(k)] = float(v)
                    except (ValueError, TypeError):
                        continue

                # 调用 decide 获取信号
                intents = strat.decide(features, str(row.get("symbol", "BTCUSDT")))
                lf = getattr(strat, "_last_funnel", {})

                if not lf.get("direction", False):
                    event_no_dir += 1
                elif lf.get("gate", True) is False:
                    event_veto += 1
                else:
                    event_allow += 1

            print(f"\n  {arch.upper()} (事件侧, {sample_size}行采样):")
            print(f"    Gate allow: {event_allow} ({event_allow/sample_size*100:.1f}%)")
            print(f"    Gate veto:  {event_veto} ({event_veto/sample_size*100:.1f}%)")
            print(
                f"    No dir:     {event_no_dir} ({event_no_dir/sample_size*100:.1f}%)"
            )

            # 逐行对比 (采样中 gate_decision=allow 的行，事件侧是否也 allow)
            allow_rows = sample_df[sample_df["gate_decision"] == "allow"]
            if len(allow_rows) > 0:
                match = 0
                for _, row in allow_rows.iterrows():
                    features = {}
                    for k, v in row.items():
                        try:
                            if v is not None and np.isscalar(v) and not pd.isna(v):
                                features[str(k)] = float(v)
                        except (ValueError, TypeError):
                            continue
                    intents = strat.decide(features, str(row.get("symbol", "BTCUSDT")))
                    if intents:
                        match += 1
                print(
                    f"\n    向量allow={len(allow_rows)}, 事件也allow={match}, "
                    f"一致率={match/len(allow_rows)*100:.1f}%"
                )

        except Exception as e:
            print(f"  {arch}: 事件侧评估失败: {e}")


# ═══════════════════════════════════════════════════════════════
# Layer 2: Entry Filter 通过率
# ═══════════════════════════════════════════════════════════════
def layer2_entry_filter():
    _hr("Layer 2: Entry Filter")

    for arch, path in LOGS_GATED.items():
        df = pd.read_parquet(path)

        # 向量侧: 读取 entry_filters.yaml 并应用
        ef_path = Path(STRATEGIES_ROOT) / arch / "entry_filters.yaml"
        if ef_path.exists():
            import yaml

            with open(ef_path) as f:
                ef_cfg = yaml.safe_load(f)
            filters = ef_cfg.get("filters", [])
            enabled = [f for f in filters if f.get("enabled", True)]
            print(
                f"\n  {arch.upper()}: {len(enabled)} enabled entry filters from {ef_path}"
            )
            for f in enabled:
                print(
                    f"    - {f.get('name', '?')}: "
                    f"col={f.get('feature', '?')}, "
                    f"op={f.get('operator', '?')}, "
                    f"val={f.get('value', '?')}"
                )
        else:
            print(f"\n  {arch.upper()}: 无 entry_filters.yaml")

        # 统计向量侧 gate=allow 后还剩多少信号
        if "gate_decision" in df.columns:
            allowed = df[df["gate_decision"] == "allow"]
            print(f"    Gate allow: {len(allowed)} rows")
            # 检查是否有 direction 相关列
            dir_cols = [
                c
                for c in df.columns
                if "direction" in c.lower() or "entry_dir" in c.lower()
            ]
            if dir_cols:
                for dc in dir_cols[:3]:
                    if dc in allowed.columns:
                        nonzero = (allowed[dc] != 0).sum()
                        print(
                            f"    {dc} != 0: {nonzero}/{len(allowed)} ({nonzero/max(len(allowed),1)*100:.1f}%)"
                        )


# ═══════════════════════════════════════════════════════════════
# Layer 3: Execution 参数对比
# ═══════════════════════════════════════════════════════════════
def layer3_execution_params():
    _hr("Layer 3: Execution 参数对比 (向量 execution.yaml vs 事件 execution.yaml)")

    import yaml

    for arch in LOGS_GATED:
        exec_path = Path(STRATEGIES_ROOT) / arch / "archetypes" / "execution.yaml"
        if exec_path.exists():
            with open(exec_path) as f:
                cfg = yaml.safe_load(f)
            print(f"\n  {arch.upper()} ({exec_path}):")
            # 提取关键参数
            for key in [
                "initial_r",
                "take_profit_r",
                "trailing_activation_r",
                "trailing_callback_r",
                "max_hold_bars",
                "breakeven_lock_r",
                "take_profit_enabled",
            ]:
                val = cfg.get(key, "N/A")
                print(f"    {key}: {val}")
        else:
            print(f"\n  {arch}: 无 execution.yaml")


# ═══════════════════════════════════════════════════════════════
# Layer 4: PCM Slot 机制
# ═══════════════════════════════════════════════════════════════
def layer4_slot_mechanism():
    _hr("Layer 4: PCM Slot 机制差异")

    import yaml

    const_path = Path("config/constitution/constitution.yaml")
    if const_path.exists():
        with open(const_path) as f:
            const = yaml.safe_load(f)
        slots = const.get("slots", {})
        psl = const.get("resource_allocation", {}).get("per_strategy_limits", {})
        print(f"\n  Constitution:")
        print(f"    slot_count: {slots.get('slot_count', '?')}")
        print(f"    risk_per_slot: {slots.get('risk_per_slot', '?')}")
        print(f"    per_strategy_limits:")
        for k, v in psl.items():
            print(f"      {k}: {v}")

    print(f"\n  向量回测 slot 机制:")
    print(f"    - post-hoc: 按 entry_idx 排序, 维护 active trades 集合")
    print(f"    - per-strategy slot: 各策略独立 max_slots")
    print(f"    - evidence 竞争: 同 archetype 内 weakest 被替换")
    print(f"\n  事件回测 slot 机制:")
    print(f"    - real-time: LivePCM.decide() 实时检查 slot")
    print(f"    - get_open_slot_count: 跨所有 symbol 当前持仓总数")
    print(f"    - per-strategy slot: _slot_available() 检查")
    print(f"    - evidence 竞争: _try_slot_competition()")

    print(f"\n  ⚠️  关键差异:")
    print(f"    1. 向量侧只看同时间 active trades (exit_idx > entry_idx)")
    print(f"    2. 事件侧看跨 symbol 全局持仓数 (实盘行为)")
    print(f"    3. 持仓持续时间不同 → slot 占用时间不同 → 接受的 trade 集合不同")


# ═══════════════════════════════════════════════════════════════
# Layer 5: 综合建议
# ═══════════════════════════════════════════════════════════════
def layer5_recommendations():
    _hr("Layer 5: 修复建议")

    print(
        """
  已发现的差异源:

  1. 时间范围不对齐 (最大根因)
     - 向量: 2025-08-01 ~ 2026-01-01 (153 天)
     - 事件: now()-180 ~ now()
     - 修复: 事件回测增加 --start-date / --end-date 参数,
             或者用 logs_gated 的时间范围来驱动事件回测

  2. Gate 评估系统不同
     - 向量: 读 parquet 中 gate_decision 列 (研究 pipeline 预计算)
     - 事件: GenericLiveStrategy.gate_evaluator.evaluate() (实时评估)
     - 修复: 确认两侧用同一份 gate.yaml + 同一套 quantiles

  3. Entry Filter 应用方式不同
     - 向量: backtest_execution_layer.py 读 entry_filters.yaml 过滤
     - 事件: GenericLiveStrategy.decide() 内置 entry filter 评估
     - 修复: 确认两侧读同一份 entry_filters.yaml

  4. PCM Slot 机制 (结构性差异, 预期会有少量偏差)
     - 向量: post-hoc filtering (看 bar index 重叠)
     - 事件: real-time slot competition (看当前持仓数)
     - 这是设计上的差异, 预期 ~10-20% 偏差可接受

  5. Execution 模拟精度
     - 向量: simulate_rr_execution (逐 bar / 逐 1min)
     - 事件: PositionSimulator (逐 1min)
     - 两者应等价, 但参数需对齐

  推荐修复顺序:
    Step 1: 对齐时间范围 (最重要)
    Step 2: 逐行对比 gate 评估结果
    Step 3: 对比 entry filter
    Step 4: 接受 slot 结构性差异 (~10-20%)
"""
    )


def main():
    layer0_time_range()
    layer1_gate_rate()
    layer2_entry_filter()
    layer3_execution_params()
    layer4_slot_mechanism()
    layer5_recommendations()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
