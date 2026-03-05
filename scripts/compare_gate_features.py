#!/usr/bin/env python3
"""快速对比向量回测 vs 事件回测的 gate 特征值"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy
from scripts.backtest_execution_layer import _load_raw_features_for_archetype


def main():
    strategy = "me"
    symbol = "BTCUSDT"
    strategies_root = "config/strategies"

    # ME gate 相关特征
    gate_features = ["atr_percentile", "me_atr_pct", "me_cvd_alignment", "vpin_max20"]

    # 1. 向量回测特征（from-raw）
    print("=" * 60)
    print("📊 向量回测特征 (from-raw)")
    print("=" * 60)

    vec_df = _load_raw_features_for_archetype(
        arch_name=strategy,
        strategies_root=strategies_root,
        symbols=[symbol],
        data_path="data/parquet_data",
        test_start="2025-08-01",
        test_end="2025-08-03",  # 只取2天加速
    )
    vec_df = vec_df[vec_df["symbol"] == symbol].copy()

    print(f"向量回测行数: {len(vec_df)}")
    for f in gate_features:
        if f in vec_df.columns:
            vals = vec_df[f]
            print(
                f"  {f}: min={vals.min():.4f}, max={vals.max():.4f}, "
                f"mean={vals.mean():.4f}, NaN={vals.isna().sum()}"
            )

    # 2. 事件回测特征（IncrementalFeatureComputer）
    print("\n" + "=" * 60)
    print("📊 事件回测特征 (IncrementalFeatureComputer)")
    print("=" * 60)

    strat = GenericLiveStrategy(
        strategy_name=strategy,
        strategies_root=strategies_root,
    )

    # 读取1min数据，模拟事件回测的特征计算
    # 数据是按月份分割的
    bars_1min = pd.read_parquet(f"data/parquet_data/{symbol}_2025-08.parquet")
    bars_1min.index = pd.to_datetime(bars_1min.index)
    bars_1min = bars_1min.loc["2025-08-01":"2025-08-03"]

    print(f"1min bars: {len(bars_1min)}")

    # 计算特征
    from src.time_series_model.live.incremental_feature_computer import (
        IncrementalFeatureComputer,
    )

    meta_path = Path(strategies_root) / strategy / "archetypes" / "meta.yaml"
    import yaml

    with open(meta_path) as f:
        meta = yaml.safe_load(f) or {}
    timeframe = meta.get("timeframe", "1h")

    computer = IncrementalFeatureComputer(
        strategy_name=strategy,
        strategies_root=strategies_root,
        timeframe=timeframe,
    )

    # 逐条计算
    evt_features_list = []
    for ts, row in bars_1min.iterrows():
        bar = {
            "timestamp": ts,
            "open": float(row.get("open", 0)),
            "high": float(row.get("high", 0)),
            "low": float(row.get("low", 0)),
            "close": float(row.get("close", 0)),
            "volume": float(row.get("volume", 0)),
        }
        result = computer.update(bar)
        if result is not None:
            evt_features_list.append(result)

    if not evt_features_list:
        print("事件回测无特征输出")
        return

    evt_df = pd.DataFrame(evt_features_list)
    print(f"事件回测行数: {len(evt_df)}")

    for f in gate_features:
        if f in evt_df.columns:
            vals = evt_df[f]
            print(
                f"  {f}: min={vals.min():.4f}, max={vals.max():.4f}, "
                f"mean={vals.mean():.4f}, NaN={vals.isna().sum()}"
            )
        else:
            print(f"  {f}: 缺失")

    # 3. 逐行对比（取前5个相同时间戳的 bar）
    print("\n" + "=" * 60)
    print("📊 逐行对比 (前5个共同时间戳)")
    print("=" * 60)

    if "timestamp" in evt_df.columns:
        evt_df = evt_df.set_index("timestamp")

    common_ts = vec_df.index.intersection(evt_df.index)[:5]
    for ts in common_ts:
        print(f"\n{ts}:")
        for f in gate_features:
            vec_val = vec_df.loc[ts, f] if f in vec_df.columns else "N/A"
            evt_val = evt_df.loc[ts, f] if f in evt_df.columns else "N/A"
            match = "✓" if vec_val == evt_val else "✗"
            print(f"  {f}: vec={vec_val:.4f}, evt={evt_val:.4f} {match}")


if __name__ == "__main__":
    main()
