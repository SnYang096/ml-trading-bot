#!/usr/bin/env python3
"""
测试不同的 VPIN 阈值对 SR Reversal 基线策略的影响

测试多个 VPIN 阈值组合，找出最优参数
"""

"""
NOTE FOR PYTEST:
This file is a *diagnostic script*, not a pytest test module.
Pytest may try to collect it because its filename starts with `test_`.
We explicitly disable collection via `__test__ = False`.
"""

__test__ = False  # pytest: ignore this module

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train_strategy_pipeline as strategy_runner
from src.data_tools.data_utils import load_raw_data
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.time_series_model.strategy_config import StrategyConfigLoader
from src.time_series_model.pipeline.training.label_utils import compute_rr_label
from src.time_series_model.strategies.labels.sr_reversal_label import (
    SRSignalConfig,
    _ensure_atr,
    _generate_sr_reversal_signals,
)


def test_vpin_threshold(
    df_features: pd.DataFrame,
    atr_series: pd.Series,
    min_vpin: float,
    max_vpin: float,
    max_holding_bars: int = 50,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    atr_window: int = 14,
    entry_offset: int = 1,
    entry_price_col: str = "open",
) -> Dict[str, Any]:
    """
    测试单个 VPIN 阈值组合

    Returns:
        Dict with metrics: signals, trades, wins, win_rate, avg_r, total_r
    """
    sr_cfg = SRSignalConfig(
        use_vpin_filter=True,
        min_vpin=min_vpin,
        max_vpin=max_vpin,
    )

    auto_signals = _generate_sr_reversal_signals(
        df_features,
        price_col="close",
        high_col="high",
        low_col="low",
        atr_series=atr_series,
        cfg=sr_cfg,
    )

    n_signals = int((auto_signals != 0).sum())

    if n_signals == 0:
        return {
            "min_vpin": min_vpin,
            "max_vpin": max_vpin,
            "signals": 0,
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "total_r": 0.0,
        }

    df_features_test = df_features.copy()
    df_features_test["signal"] = auto_signals

    labels = compute_rr_label(
        df_features_test,
        signal_col="signal",
        price_col="close",
        atr_col="atr",
        atr_window=atr_window,
        rr_ratio=take_profit_r,
        max_holding_bars=max_holding_bars,
        stop_loss_r=stop_loss_r,
        take_profit_r=take_profit_r,
        use_continuous_label=False,
        entry_price_col=entry_price_col,
        entry_offset=entry_offset,
    )

    mask_valid = (auto_signals != 0) & labels.notna()
    df_trades = pd.DataFrame(
        {
            "signal": auto_signals[mask_valid],
            "label": labels[mask_valid],
        }
    )

    n_trades = len(df_trades)
    if n_trades == 0:
        return {
            "min_vpin": min_vpin,
            "max_vpin": max_vpin,
            "signals": n_signals,
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "total_r": 0.0,
        }

    n_win = int((df_trades["label"] == 1.0).sum())
    win_rate = n_win / n_trades

    realized_r = np.where(df_trades["label"].values == 1.0, take_profit_r, -stop_loss_r)
    avg_r = float(realized_r.mean())
    total_r = float(realized_r.sum())

    return {
        "min_vpin": min_vpin,
        "max_vpin": max_vpin,
        "signals": n_signals,
        "trades": n_trades,
        "wins": n_win,
        "win_rate": win_rate,
        "avg_r": avg_r,
        "total_r": total_r,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test different VPIN thresholds for SR Reversal baseline"
    )
    parser.add_argument(
        "--strategy-config", type=str, default="config/strategies/sr_reversal_long"
    )
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--data-path", type=str, default="data/parquet_data")
    parser.add_argument("--timeframe", type=str, default="240T")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--max-holding-bars", type=int, default=50)
    parser.add_argument("--stop-loss-r", type=float, default=1.0)
    parser.add_argument("--take-profit-r", type=float, default=2.0)
    parser.add_argument("--atr-window", type=int, default=14)
    parser.add_argument("--entry-offset", type=int, default=1)
    parser.add_argument("--entry-price-col", type=str, default="open")
    parser.add_argument(
        "--output-csv", type=str, default="results/vpin_threshold_test.csv"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("VPIN 阈值测试")
    print("=" * 80)
    print()

    # 加载数据
    print(f"📈 Loading data for {args.symbol} [{args.timeframe}]...")
    df_raw = load_raw_data(
        data_path=args.data_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
    )
    print(f"   ✅ Loaded {len(df_raw)} bars")
    print()

    # 加载特征
    print("🔧 Loading features...")
    cfg_dir = Path(args.strategy_config).resolve()
    loader = StrategyConfigLoader(cfg_dir)
    strategy_cfg = loader.load()

    feature_loader = StrategyFeatureLoader()
    df_features = strategy_runner.run_feature_pipeline(
        df_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=True,
    )

    atr_series = _ensure_atr(
        df_features,
        atr_col="atr",
        price_col="close",
        high_col="high",
        low_col="low",
        atr_window=args.atr_window,
    )
    df_features["atr"] = atr_series

    # 检查 VPIN 是否存在
    if "vpin" not in df_features.columns:
        print(
            "   ❌ VPIN feature not found! Make sure vpin_features is in requested_features."
        )
        return

    vpin_stats = df_features["vpin"].describe()
    print(f"   ✅ VPIN feature found")
    print(f"      VPIN range: [{vpin_stats['min']:.3f}, {vpin_stats['max']:.3f}]")
    print(f"      VPIN mean: {vpin_stats['mean']:.3f}, median: {vpin_stats['50%']:.3f}")
    print()

    # 测试不同的阈值组合
    print("🧪 Testing VPIN thresholds...")
    print()

    # 测试方案：
    # 1. 无过滤（baseline）
    # 2. 对称阈值：0.3/0.7, 0.4/0.6, 0.45/0.55, 0.5/0.5
    # 3. 非对称阈值：更严格的多头/空头
    test_configs = [
        # (min_vpin, max_vpin, description)
        (None, None, "无 VPIN 过滤（baseline）"),
        (0.3, 0.7, "宽松：min=0.3, max=0.7"),
        (0.35, 0.65, "中等：min=0.35, max=0.65"),
        (0.4, 0.6, "当前设置：min=0.4, max=0.6"),
        (0.45, 0.55, "严格：min=0.45, max=0.55"),
        (0.5, 0.5, "非常严格：min=0.5, max=0.5"),
        # 非对称
        (0.4, 0.5, "多头宽松，空头严格：min=0.4, max=0.5"),
        (0.5, 0.6, "多头严格，空头宽松：min=0.5, max=0.6"),
        (0.45, 0.6, "中等严格：min=0.45, max=0.6"),
        (0.4, 0.55, "中等严格：min=0.4, max=0.55"),
    ]

    results = []

    for min_vpin, max_vpin, desc in test_configs:
        print(f"   测试: {desc}")
        if min_vpin is None:
            # 无过滤
            sr_cfg = SRSignalConfig(use_vpin_filter=False)
            auto_signals = _generate_sr_reversal_signals(
                df_features,
                price_col="close",
                high_col="high",
                low_col="low",
                atr_series=atr_series,
                cfg=sr_cfg,
            )
            n_signals = int((auto_signals != 0).sum())

            if n_signals > 0:
                df_features_test = df_features.copy()
                df_features_test["signal"] = auto_signals

                labels = compute_rr_label(
                    df_features_test,
                    signal_col="signal",
                    price_col="close",
                    atr_col="atr",
                    atr_window=args.atr_window,
                    rr_ratio=args.take_profit_r,
                    max_holding_bars=args.max_holding_bars,
                    stop_loss_r=args.stop_loss_r,
                    take_profit_r=args.take_profit_r,
                    use_continuous_label=False,
                    entry_price_col=args.entry_price_col,
                    entry_offset=args.entry_offset,
                )

                mask_valid = (auto_signals != 0) & labels.notna()
                df_trades = pd.DataFrame(
                    {
                        "signal": auto_signals[mask_valid],
                        "label": labels[mask_valid],
                    }
                )

                n_trades = len(df_trades)
                n_win = int((df_trades["label"] == 1.0).sum()) if n_trades > 0 else 0
                win_rate = n_win / n_trades if n_trades > 0 else 0.0

                realized_r = (
                    np.where(
                        df_trades["label"].values == 1.0,
                        args.take_profit_r,
                        -args.stop_loss_r,
                    )
                    if n_trades > 0
                    else np.array([])
                )
                avg_r = float(realized_r.mean()) if len(realized_r) > 0 else 0.0
                total_r = float(realized_r.sum()) if len(realized_r) > 0 else 0.0
            else:
                n_trades = 0
                n_win = 0
                win_rate = 0.0
                avg_r = 0.0
                total_r = 0.0

            result = {
                "min_vpin": None,
                "max_vpin": None,
                "description": desc,
                "signals": n_signals,
                "trades": n_trades,
                "wins": n_win,
                "win_rate": win_rate,
                "avg_r": avg_r,
                "total_r": total_r,
            }
        else:
            result = test_vpin_threshold(
                df_features,
                atr_series,
                min_vpin,
                max_vpin,
                max_holding_bars=args.max_holding_bars,
                stop_loss_r=args.stop_loss_r,
                take_profit_r=args.take_profit_r,
                atr_window=args.atr_window,
                entry_offset=args.entry_offset,
                entry_price_col=args.entry_price_col,
            )
            result["description"] = desc

        results.append(result)

        print(
            f"      信号: {result['signals']}, 交易: {result['trades']}, "
            f"胜率: {result['win_rate']:.2%}, 平均R: {result['avg_r']:.3f}, "
            f"总R: {result['total_r']:.3f}"
        )
        print()

    # 汇总结果
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("total_r", ascending=False)

    print("=" * 80)
    print("测试结果汇总（按总 R 排序）")
    print("=" * 80)
    print()
    print(df_results.to_string(index=False))
    print()

    # 找出最优配置
    best = df_results.iloc[0]
    print(f"🏆 最优配置: {best['description']}")
    print(f"   min_vpin={best['min_vpin']}, max_vpin={best['max_vpin']}")
    print(f"   信号: {best['signals']}, 交易: {best['trades']}")
    print(
        f"   胜率: {best['win_rate']:.2%}, 平均R: {best['avg_r']:.3f}, 总R: {best['total_r']:.3f}"
    )
    print()

    # 保存结果
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False)
    print(f"💾 结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
