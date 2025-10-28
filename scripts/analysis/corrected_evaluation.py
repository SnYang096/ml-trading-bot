"""修正后的评估方法 - 更真实地反映策略表现."""

import sys
import os
import pickle
import pandas as pd
import numpy as np
import zipfile
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml_trading.data_tools.data_loader import MarketDataLoader


def load_and_prepare_test_data(zip_path: str) -> pd.DataFrame:
    """加载测试数据."""
    temp_dir = os.path.join(os.path.dirname(zip_path), "temp_eval")
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
    csv_path = os.path.join(temp_dir, csv_files[0])

    df = pd.read_csv(csv_path)

    if "transact_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df.set_index("timestamp", inplace=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["price", "quantity"])

    ohlc_dict = {"price": ["first", "max", "min", "last"], "quantity": "sum"}
    resampled = df.groupby(pd.Grouper(freq="1s")).agg(ohlc_dict)
    resampled.columns = ["open", "high", "low", "close", "volume"]
    resampled = resampled.dropna().ffill()

    # Add microstructure
    try:
        agg = pd.read_csv(csv_path)
        if "transact_time" in agg.columns:
            agg["timestamp"] = pd.to_datetime(agg["transact_time"], unit="ms")
        else:
            agg["timestamp"] = pd.to_datetime(agg["timestamp"])
        agg["price"] = pd.to_numeric(agg["price"], errors="coerce")
        agg["quantity"] = pd.to_numeric(agg["quantity"], errors="coerce")
        agg = agg.dropna(subset=["price", "quantity"])

        if "is_buyer_maker" in agg.columns:
            agg["taker_buy"] = (~agg["is_buyer_maker"].astype(bool)).astype(int)
        else:
            agg["taker_buy"] = 0

        agg["buy_qty"] = np.where(agg["taker_buy"] == 1, agg["quantity"], 0.0)
        agg["sell_qty"] = np.where(agg["taker_buy"] == 1, 0.0, agg["quantity"])
        agg = agg.set_index("timestamp")

        per_sec = agg.groupby(pd.Grouper(freq="1s")).agg(
            {"buy_qty": "sum", "sell_qty": "sum"}
        )
        per_sec["taker_buy_ratio"] = per_sec["buy_qty"] / (
            per_sec["buy_qty"] + per_sec["sell_qty"]
        ).replace(0, np.nan)
        per_sec["taker_buy_ratio"] = per_sec["taker_buy_ratio"].fillna(0.5)
        per_sec["cvd"] = (per_sec["buy_qty"] - per_sec["sell_qty"]).cumsum()

        resampled = (
            resampled.join(
                per_sec[["buy_qty", "sell_qty", "taker_buy_ratio", "cvd"]], how="left"
            )
            .ffill()
            .fillna(0)
        )
    except:
        pass

    import shutil

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    return resampled


def corrected_evaluation(model_path: str = "trained_model_wavelet_may_2025.pkl"):
    """修正后的评估方法."""

    print("=" * 80)
    print("📊 修正后的评估分析")
    print("=" * 80)

    # Load model
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    strategy = model_data["strategy"]
    data_loader = model_data["data_loader"]
    feature_engineer = model_data["feature_engineer"]

    # Test on June
    zip_path = "data/aggTrades/BTCUSDT-aggTrades-2025-06.zip"

    print(f"\n测试月份: June 2025")
    print("-" * 80)

    test_data = load_and_prepare_test_data(zip_path)
    data_loader.raw_data = test_data

    multi_tf_data = data_loader.get_multi_timeframe_data()
    engineered_data = feature_engineer.engineer_features(multi_tf_data, fit=False)

    timeframe = "5T"
    data = engineered_data[timeframe]

    # Get models
    stage1_model = strategy.pipeline.stage1_models.get(timeframe)
    stage2_model = strategy.pipeline.stage2_models.get(timeframe)

    # Prepare features
    feature_columns = [
        col
        for col in data.columns
        if col not in ["open", "high", "low", "close", "volume"]
    ]
    X = data[feature_columns]

    # Calculate TRUE future returns (without threshold filtering)
    future_returns = data["close"].shift(-1) / data["close"] - 1

    # Remove NaN
    valid_indices = ~(X.isna().any(axis=1) | future_returns.isna())
    X_clean = X[valid_indices]
    future_returns_clean = future_returns[valid_indices]

    # Predict
    pred_stage1 = stage1_model.predict(X_clean)
    pred_stage1_binary = (pred_stage1 > 0.5).astype(int)

    print(f"\n预测统计:")
    print(f"  总样本数: {len(pred_stage1_binary)}")
    print(
        f"  预测做多: {(pred_stage1_binary == 1).sum()} ({(pred_stage1_binary == 1).sum()/len(pred_stage1_binary)*100:.2f}%)"
    )
    print(
        f"  预测不做: {(pred_stage1_binary == 0).sum()} ({(pred_stage1_binary == 0).sum()/len(pred_stage1_binary)*100:.2f}%)"
    )

    # 方法1: 原始计算方式（有问题的）
    print(f"\n📊 方法1: 原始计算（基于threshold过滤的target）")
    y_stage1, y_stage2 = strategy.pipeline.prepare_targets(data[valid_indices])
    signals = pred_stage1_binary
    returns = y_stage2.values
    strategy_returns = returns * signals

    win_rate_old = (strategy_returns > 0).sum() / max((signals == 1).sum(), 1)
    total_return_old = strategy_returns.sum()

    print(f"  胜率: {win_rate_old*100:.2f}%")
    print(f"  总收益: {total_return_old:.4f} ({total_return_old*100:.2f}%)")

    # 方法2: 修正后的计算（更真实）
    print(f"\n✅ 方法2: 修正后计算（所有预测信号）")

    # 只看模型预测做多的样本
    long_predictions = pred_stage1_binary == 1

    if long_predictions.sum() > 0:
        # 这些预测中，实际有多少次上涨
        actual_returns_when_pred_long = future_returns_clean[long_predictions]

        win_rate_corrected = (actual_returns_when_pred_long > 0).sum() / len(
            actual_returns_when_pred_long
        )
        total_return_corrected = actual_returns_when_pred_long.sum()
        avg_return_corrected = actual_returns_when_pred_long.mean()

        print(f"  模型预测做多次数: {long_predictions.sum()}")
        print(f"  实际上涨次数: {(actual_returns_when_pred_long > 0).sum()}")
        print(f"  实际下跌次数: {(actual_returns_when_pred_long <= 0).sum()}")
        print(f"  ")
        print(f"  **修正后胜率**: {win_rate_corrected*100:.2f}%")
        print(
            f"  平均单次收益: {avg_return_corrected:.6f} ({avg_return_corrected*100:.4f}%)"
        )
        print(
            f"  累计总收益: {total_return_corrected:.4f} ({total_return_corrected*100:.2f}%)"
        )

        # 更详细的统计
        print(f"\n  收益分布:")
        print(
            f"    > 0.2%: {(actual_returns_when_pred_long > 0.002).sum()} ({(actual_returns_when_pred_long > 0.002).sum()/len(actual_returns_when_pred_long)*100:.2f}%)"
        )
        print(
            f"    > 0.1%: {(actual_returns_when_pred_long > 0.001).sum()} ({(actual_returns_when_pred_long > 0.001).sum()/len(actual_returns_when_pred_long)*100:.2f}%)"
        )
        print(
            f"    > 0%: {(actual_returns_when_pred_long > 0).sum()} ({win_rate_corrected*100:.2f}%)"
        )
        print(
            f"    < 0%: {(actual_returns_when_pred_long < 0).sum()} ({(actual_returns_when_pred_long < 0).sum()/len(actual_returns_when_pred_long)*100:.2f}%)"
        )
        print(
            f"    < -0.1%: {(actual_returns_when_pred_long < -0.001).sum()} ({(actual_returns_when_pred_long < -0.001).sum()/len(actual_returns_when_pred_long)*100:.2f}%)"
        )

        # 盈亏比
        winning_trades = actual_returns_when_pred_long[
            actual_returns_when_pred_long > 0
        ]
        losing_trades = actual_returns_when_pred_long[actual_returns_when_pred_long < 0]

        if len(winning_trades) > 0 and len(losing_trades) > 0:
            avg_win = winning_trades.mean()
            avg_loss = abs(losing_trades.mean())
            profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")

            print(f"\n  盈亏比分析:")
            print(f"    平均盈利: {avg_win:.6f} ({avg_win*100:.4f}%)")
            print(f"    平均亏损: {avg_loss:.6f} ({avg_loss*100:.4f}%)")
            print(f"    盈亏比: {profit_factor:.2f}:1")

    # 方法3: 如果完全随机交易
    print(f"\n📊 对比: 完全随机交易")
    random_long_mask = np.random.rand(len(future_returns_clean)) > 0.5
    num_random_trades = random_long_mask.sum()

    if num_random_trades > 0:
        random_returns = future_returns_clean[random_long_mask]
        random_winrate = (random_returns > 0).sum() / len(random_returns)
        random_total = random_returns.sum()

        print(f"  随机交易次数: {num_random_trades}")
        print(f"  随机胜率: {random_winrate*100:.2f}%")
        print(f"  随机累计收益: {random_total:.4f} ({random_total*100:.2f}%)")

    print(f"\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)


if __name__ == "__main__":
    corrected_evaluation()
