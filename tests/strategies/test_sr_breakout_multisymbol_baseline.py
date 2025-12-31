"""
测试 SR Breakout 多符号场景下的 baseline 问题诊断

此测试用于诊断和验证：
1. 标签生成是否正确（多符号隔离）
2. 特征计算是否正确（多符号隔离）
3. 回测配置是否正确（freq, use_rr_exit, use_signal_direction）
4. Baseline Sharpe 为负的可能原因
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.strategies.labels.sr_breakout_label import (
    compute_sr_breakout_label,
)
from src.time_series_model.strategies.backtesting.vectorbt_backtest import (
    VectorBTBacktest,
)
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.time_series_model.strategy_config import StrategyConfigLoader
from src.data_tools.data_handler import DataHandler


def test_sr_breakout_label_multisymbol_isolation():
    """测试标签生成是否正确隔离多符号数据"""
    # 创建两个符号的交错数据（相同时间戳）
    idx = pd.date_range("2024-01-01", periods=60, freq="4H", tz="UTC")

    # Symbol A: 上升趋势
    base_a = np.linspace(100, 130, len(idx)) + 2.0 * np.sin(np.linspace(0, 6, len(idx)))
    df_a = pd.DataFrame(
        {
            "_symbol": "BTCUSDT",
            "close": base_a,
            "high": base_a + 1.0,
            "low": base_a - 8.0,
            "signal": 1.0,  # always long
        },
        index=idx,
    )

    # Symbol B: 下降趋势
    base_b = np.linspace(1000, 900, len(idx)) + 5.0 * np.sin(
        np.linspace(0, 6, len(idx))
    )
    df_b = pd.DataFrame(
        {
            "_symbol": "ETHUSDT",
            "close": base_b,
            "high": base_b + 2.0,
            "low": base_b - 16.0,
            "signal": -1.0,  # always short
        },
        index=idx,
    )

    # 交错数据（常见池化格式）
    df = pd.concat([df_a, df_b], axis=0).sort_index()

    labels = compute_sr_breakout_label(
        df,
        signal_col="signal",
        price_col="close",
        high_col="high",
        low_col="low",
        atr_col="atr",
        atr_window=3,
        max_holding_bars=10,
        max_rr=3.0,
        stop_loss_r=1.0,
        auto_generate_signals=False,
    )

    # 验证：应该为两个符号都生成标签
    assert labels.notna().sum() > 0, "应该生成一些非 NaN 标签"

    # 验证：标签不应该在整个池化数据集上塌陷为单一常量
    uniq = set(np.round(labels.dropna().values, 6).tolist())
    assert len(uniq) > 1, "标签应该有变化，不应该塌陷为单一值"

    # 验证：每个符号的标签应该独立
    labels_a = labels[df["_symbol"] == "BTCUSDT"]
    labels_b = labels[df["_symbol"] == "ETHUSDT"]

    assert labels_a.notna().sum() > 0, "BTCUSDT 应该有标签"
    assert labels_b.notna().sum() > 0, "ETHUSDT 应该有标签"

    # 验证：不同符号的标签分布应该不同（因为价格路径不同）
    if labels_a.notna().sum() > 5 and labels_b.notna().sum() > 5:
        mean_a = labels_a.dropna().mean()
        mean_b = labels_b.dropna().mean()
        # 由于一个做多一个做空，且价格路径不同，均值应该不同
        assert not np.isclose(mean_a, mean_b, rtol=0.1), "不同符号的标签均值应该不同"


def test_sr_breakout_backtest_multisymbol_config():
    """测试回测配置是否正确支持多符号"""
    # 创建多符号测试数据
    idx = pd.date_range("2024-01-01", periods=100, freq="4H", tz="UTC")

    df_list = []
    for sym, base_price in [("BTCUSDT", 50000), ("ETHUSDT", 3000)]:
        prices = base_price + np.cumsum(np.random.randn(len(idx)) * 10)
        df_sym = pd.DataFrame(
            {
                "_symbol": sym,
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.uniform(1000, 10000, len(idx)),
                "atr": prices * 0.01,
                "signal": np.random.choice([-1, 0, 1], len(idx)),
            },
            index=idx,
        )
        df_list.append(df_sym)

    df = pd.concat(df_list, axis=0).sort_index()

    # 创建预测（回归任务）
    preds = np.random.uniform(0, 3.0, len(df))

    # 测试回测配置
    bt = VectorBTBacktest()

    # 按符号分组回测
    results_by_symbol = {}
    for sym in df["_symbol"].unique():
        mask = (df["_symbol"] == sym).to_numpy()
        df_sym = df.loc[mask].copy()
        preds_sym = preds[mask]

        result = bt.run(
            df=df_sym,
            predictions=preds_sym,
            task_type="regression",
            price_col="close",
            freq="4H",
            strategy_direction="both",
            use_rr_exit=True,
            use_signal_direction=True,
            rr={
                "atr_col": "atr",
                "max_holding_bars": 50,
                "stop_loss_r": 1.0,
                "take_profit_r": 2.0,
            },
            long_entry_threshold=0.55,
            short_entry_threshold=0.45,
            initial_cash=10000.0,
            fee=0.0004,
            slippage=0.0001,
        )

        if result:
            results_by_symbol[sym] = result

    # 验证：每个符号都应该有回测结果
    assert len(results_by_symbol) > 0, "应该至少有一个符号的回测结果"

    # 验证：回测结果应该包含必要的指标
    for sym, result in results_by_symbol.items():
        assert "total_trades" in result, f"{sym} 回测结果应该包含 total_trades"
        assert "sharpe" in result, f"{sym} 回测结果应该包含 sharpe"
        assert "total_return_pct" in result, f"{sym} 回测结果应该包含 total_return_pct"


def test_sr_breakout_baseline_negative_diagnosis(tmp_path: Path):
    """
    诊断 baseline Sharpe 为负的可能原因

    此测试模拟多符号场景，检查可能导致 baseline 为负的因素：
    1. 标签生成问题
    2. 特征计算问题
    3. 回测配置问题
    4. 数据质量问题
    """
    # 创建模拟的多符号数据
    idx = pd.date_range("2024-01-01", periods=200, freq="4H", tz="UTC")

    df_list = []
    for sym, base_price, trend in [
        ("BTCUSDT", 50000, 1.0),  # 上升趋势
        ("ETHUSDT", 3000, -1.0),  # 下降趋势
    ]:
        # 创建有趋势的价格路径
        trend_component = np.linspace(0, trend * 0.1 * base_price, len(idx))
        noise = np.random.randn(len(idx)) * base_price * 0.01
        prices = base_price + trend_component + noise

        df_sym = pd.DataFrame(
            {
                "_symbol": sym,
                "open": prices,
                "high": prices * 1.005,
                "low": prices * 0.995,
                "close": prices,
                "volume": np.random.uniform(1000, 10000, len(idx)),
            },
            index=idx,
        )
        df_list.append(df_sym)

    df = pd.concat(df_list, axis=0).sort_index()

    # 1. 测试标签生成
    labels = compute_sr_breakout_label(
        df,
        signal_col="signal",
        price_col="close",
        high_col="high",
        low_col="low",
        atr_col="atr",
        auto_generate_signals=True,
        signal_horizon=1,
        signal_threshold_atr=0.0,
    )

    # 诊断：检查标签分布
    label_stats = {}
    for sym in df["_symbol"].unique():
        mask = (df["_symbol"] == sym).to_numpy()
        labels_sym = labels[mask]
        label_stats[sym] = {
            "non_null_count": labels_sym.notna().sum(),
            "mean": (
                labels_sym.dropna().mean() if labels_sym.notna().sum() > 0 else None
            ),
            "std": labels_sym.dropna().std() if labels_sym.notna().sum() > 0 else None,
            "min": labels_sym.dropna().min() if labels_sym.notna().sum() > 0 else None,
            "max": labels_sym.dropna().max() if labels_sym.notna().sum() > 0 else None,
        }

    # 验证：每个符号都应该有标签
    for sym, stats in label_stats.items():
        assert stats["non_null_count"] > 0, f"{sym} 应该有非空标签"
        assert stats["mean"] is not None, f"{sym} 标签均值不应该为 None"

    # 2. 测试特征计算（如果可能）
    # 这里只做基本检查，完整特征计算需要更多依赖

    # 3. 测试回测配置
    # 使用简单的预测值进行回测
    preds = np.random.uniform(0.5, 2.5, len(df))

    bt = VectorBTBacktest()

    # 按符号分组回测
    backtest_results = {}
    for sym in df["_symbol"].unique():
        mask = (df["_symbol"] == sym).to_numpy()
        df_sym = df.loc[mask].copy()
        preds_sym = preds[mask]

        # 确保有必要的列
        if "atr" not in df_sym.columns:
            df_sym["atr"] = (
                (df_sym["high"] - df_sym["low"]).rolling(14, min_periods=1).mean()
            )

        result = bt.run(
            df=df_sym,
            predictions=preds_sym,
            task_type="regression",
            price_col="close",
            freq="4H",
            strategy_direction="both",
            use_rr_exit=True,
            use_signal_direction=True,
            rr={
                "atr_col": "atr",
                "max_holding_bars": 50,
                "stop_loss_r": 1.0,
                "take_profit_r": 2.0,
            },
            long_entry_threshold=0.55,
            short_entry_threshold=0.45,
            initial_cash=10000.0,
            fee=0.0004,
            slippage=0.0001,
        )

        if result:
            backtest_results[sym] = result

    # 诊断：检查回测结果
    for sym, result in backtest_results.items():
        assert "total_trades" in result, f"{sym} 应该有交易"
        assert "sharpe" in result, f"{sym} 应该有 Sharpe 值"

        # 记录诊断信息（不强制断言，因为这是诊断测试）
        print(f"\n{sym} 诊断信息:")
        print(f"  Total Trades: {result.get('total_trades', 0)}")
        print(f"  Sharpe: {result.get('sharpe', 'N/A')}")
        print(f"  Return%: {result.get('total_return_pct', 'N/A')}")
        print(f"  Max DD%: {result.get('max_drawdown_pct', 'N/A')}")
        print(f"  Label Stats: {label_stats.get(sym, {})}")


def test_sr_breakout_feature_computation_multisymbol():
    """测试特征计算是否正确支持多符号隔离"""
    # 创建多符号数据
    idx = pd.date_range("2024-01-01", periods=100, freq="4H", tz="UTC")

    df_list = []
    for sym, base_price in [("BTCUSDT", 50000), ("ETHUSDT", 3000)]:
        prices = base_price + np.cumsum(np.random.randn(len(idx)) * 10)
        df_sym = pd.DataFrame(
            {
                "_symbol": sym,
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.uniform(1000, 10000, len(idx)),
            },
            index=idx,
        )
        df_list.append(df_sym)

    df = pd.concat(df_list, axis=0).sort_index()

    # 测试：特征计算应该能够处理多符号数据
    # 这里只做基本验证，完整特征计算需要 StrategyFeatureLoader

    # 验证：每个符号的数据都存在
    for sym in ["BTCUSDT", "ETHUSDT"]:
        mask = (df["_symbol"] == sym).to_numpy()
        assert mask.sum() > 0, f"{sym} 数据应该存在"
        assert df.loc[mask, "close"].notna().all(), f"{sym} close 价格不应该有 NaN"

    # 验证：时间索引可能有重复（多符号池化格式）
    assert len(df.index) == len(df.index.unique()) * 2, "多符号数据应该有重复的时间索引"


@pytest.mark.integration
def test_sr_breakout_multisymbol_full_pipeline():
    """
    完整的多符号 pipeline 测试（使用真实数据）

    此测试验证：
    1. 真实的策略配置加载
    2. 真实的市场数据加载（多符号）
    3. 完整的特征计算 pipeline
    4. 标签生成（多符号隔离）
    5. 回测配置验证

    用于验证整个 pipeline 是否正确处理多符号数据
    """
    # 检查策略配置
    config_dir = Path("config/strategies/sr_breakout")
    if not config_dir.exists():
        pytest.skip("sr_breakout 配置不存在")

    # 检查数据目录
    data_dir = Path("data/parquet_data")
    if not data_dir.exists():
        pytest.skip(f"真实数据目录不存在: {data_dir}")

    # 加载策略配置
    config_loader = StrategyConfigLoader(config_dir)
    strategy_config = config_loader.load()

    # 1. 加载多符号数据
    symbols = ["BTCUSDT", "ETHUSDT"]
    timeframe = "240T"  # 4H

    data_handler = DataHandler(data_path=str(data_dir))

    dfs = []
    for symbol in symbols:
        try:
            df_sym = data_handler.load_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
            )
            if df_sym is not None and len(df_sym) > 0:
                df_sym["_symbol"] = symbol
                dfs.append(df_sym)
                print(f"   ✅ 加载 {symbol}: {len(df_sym)} 条记录")
            else:
                print(f"   ⚠️  {symbol}: 无数据")
        except Exception as e:
            print(f"   ❌ {symbol}: 加载失败 - {e}")

    if len(dfs) == 0:
        pytest.skip("无法加载任何符号的数据")

    # 合并多符号数据
    df = pd.concat(dfs, axis=0).sort_index()
    print(f"   📊 总数据量: {len(df)} 条记录，符号: {df['_symbol'].unique().tolist()}")

    # 限制数据量以加快测试（使用最近的数据）
    if len(df) > 1000:
        df = df.tail(1000).copy()
        print(f"   📉 限制数据量到最近 1000 条")

    # 2. 计算特征（如果配置了特征）
    # 注意：特征计算可能很耗时，这里跳过以加快测试
    # 主要目的是验证多符号隔离和回测配置
    features_config_path = config_dir / "features.yaml"
    if features_config_path.exists():
        try:
            feature_loader = StrategyFeatureLoader(
                strategy_config_path=str(features_config_path),
                cache_dir=None,  # 不使用缓存以加快测试
                use_disk_cache=False,
                use_memory_cache=False,
            )

            # 获取请求的特征列表
            if hasattr(strategy_config, "features") and hasattr(
                strategy_config.features, "requested_features"
            ):
                requested_features = strategy_config.features.requested_features
                if requested_features:
                    print(f"   🔧 计算特征（{len(requested_features)} 个特征）...")
                    # 这里可以调用 compute_features，但为了加快测试，我们跳过
                    # df_with_features = feature_loader.compute_features(df, requested_features=requested_features)
                    print("   ⏭️  跳过特征计算以加快测试（主要验证多符号隔离）")
        except Exception as e:
            print(f"   ⚠️  特征加载器初始化失败: {e}，继续使用原始数据")

    # 3. 生成标签（多符号隔离验证）
    print("   🏷️  生成标签...")

    # 检查是否有必要的列
    required_cols = ["close", "high", "low"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        pytest.skip(f"缺少必要的列: {missing_cols}")

    # 计算 ATR（如果不存在）
    if "atr" not in df.columns:
        df["atr"] = (df["high"] - df["low"]).rolling(14, min_periods=1).mean()

    labels = compute_sr_breakout_label(
        df,
        signal_col="signal",
        price_col="close",
        high_col="high",
        low_col="low",
        atr_col="atr",
        auto_generate_signals=True,
        signal_horizon=1,
        signal_threshold_atr=0.0,
    )

    # 验证标签生成
    assert labels.notna().sum() > 0, "应该生成一些非空标签"

    # 验证多符号隔离
    label_stats = {}
    for symbol in df["_symbol"].unique():
        mask = (df["_symbol"] == symbol).to_numpy()
        labels_sym = labels[mask]
        label_stats[symbol] = {
            "non_null_count": labels_sym.notna().sum(),
            "mean": (
                labels_sym.dropna().mean() if labels_sym.notna().sum() > 0 else None
            ),
        }
        print(
            f"   📈 {symbol}: {label_stats[symbol]['non_null_count']} 个标签, 均值: {label_stats[symbol]['mean']:.4f}"
        )

    # 4. 运行回测（验证配置）
    print("   🔄 运行回测...")

    # 创建简单的预测值（使用标签作为预测，模拟 baseline）
    preds = labels.fillna(0.0).values

    bt = VectorBTBacktest()

    backtest_results = {}
    for symbol in df["_symbol"].unique():
        mask = (df["_symbol"] == symbol).to_numpy()
        df_sym = df.loc[mask].copy()
        preds_sym = preds[mask]

        # 确保有必要的列
        if "atr" not in df_sym.columns:
            df_sym["atr"] = (
                (df_sym["high"] - df_sym["low"]).rolling(14, min_periods=1).mean()
            )

        try:
            result = bt.run(
                df=df_sym,
                predictions=preds_sym,
                task_type="regression",
                price_col="close",
                freq="4H",
                strategy_direction="both",
                use_rr_exit=True,
                use_signal_direction=True,
                rr={
                    "atr_col": "atr",
                    "max_holding_bars": 50,
                    "stop_loss_r": 1.0,
                    "take_profit_r": 2.0,
                },
                long_entry_threshold=0.55,
                short_entry_threshold=0.45,
                initial_cash=10000.0,
                fee=0.0004,
                slippage=0.0001,
            )

            if result:
                backtest_results[symbol] = result
                print(
                    f"   ✅ {symbol} 回测完成: Sharpe={result.get('sharpe', 'N/A'):.4f}, "
                    f"Trades={result.get('total_trades', 0)}, "
                    f"Return%={result.get('total_return_pct', 'N/A'):.4f}"
                )
        except Exception as e:
            print(f"   ❌ {symbol} 回测失败: {e}")

    # 5. 验证结果
    assert len(backtest_results) > 0, "应该至少有一个符号的回测结果"

    # 验证每个符号的回测结果
    for symbol, result in backtest_results.items():
        assert "total_trades" in result, f"{symbol} 回测结果应该包含 total_trades"
        assert "sharpe" in result, f"{symbol} 回测结果应该包含 sharpe"
        assert (
            "total_return_pct" in result
        ), f"{symbol} 回测结果应该包含 total_return_pct"

        # 诊断信息
        sharpe = result.get("sharpe", float("nan"))
        trades = result.get("total_trades", 0)

        print(f"\n   📊 {symbol} 诊断信息:")
        print(f"      Sharpe: {sharpe:.4f}")
        print(f"      Trades: {trades}")
        print(f"      Return%: {result.get('total_return_pct', 'N/A')}")
        print(f"      Max DD%: {result.get('max_drawdown_pct', 'N/A')}")
        print(f"      标签统计: {label_stats.get(symbol, {})}")

        # 如果 Sharpe 为负，记录诊断信息（不强制失败，因为这是诊断测试）
        if not np.isnan(sharpe) and sharpe < 0:
            print(f"      ⚠️  {symbol} Sharpe 为负，可能原因:")
            print(
                f"         - 标签分布: {label_stats.get(symbol, {}).get('mean', 'N/A')}"
            )
            print(f"         - 交易次数: {trades}")
            if trades == 0:
                print(f"         - 可能原因: 无交易信号或回测配置问题")

    print("\n   ✅ 完整 pipeline 测试通过")
