"""测试增强模型在6-9月OOS数据上的表现."""

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
    """加载并准备测试数据，包含订单流特征."""
    temp_dir = os.path.join(os.path.dirname(zip_path), "temp_enhanced_test")
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
    csv_path = os.path.join(temp_dir, csv_files[0])

    # Load data
    df = pd.read_csv(csv_path)

    if "transact_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df.set_index("timestamp", inplace=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["price", "quantity"])

    # Create OHLCV
    ohlc_dict = {"price": ["first", "max", "min", "last"], "quantity": "sum"}
    resampled = df.groupby(pd.Grouper(freq="1s")).agg(ohlc_dict)
    resampled.columns = ["open", "high", "low", "close", "volume"]
    resampled = resampled.dropna().ffill()

    # Add microstructure features (CVD and taker_buy_ratio)
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
    except Exception as e:
        print(f"Warning: {e}")

    import shutil

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    return resampled


def test_enhanced_model_oos():
    """测试增强模型在OOS数据上的表现."""

    print("=" * 80)
    print("🔬 增强模型 OOS测试（6-9月）")
    print("=" * 80)

    # Load enhanced model
    model_path = "trained_model_enhanced_may_2025.pkl"
    print(f"\n加载增强模型: {model_path}")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    strategy = model_data["strategy"]
    data_loader = model_data["data_loader"]
    feature_engineer = model_data["feature_engineer"]

    print(f"✅ 模型加载成功")
    print(f"   训练日期: {model_data['training_date']}")

    # Test on each month
    months = {
        "June 2025": "data/aggTrades/BTCUSDT-aggTrades-2025-06.zip",
        "July 2025": "data/aggTrades/BTCUSDT-aggTrades-2025-07.zip",
        "August 2025": "data/aggTrades/BTCUSDT-aggTrades-2025-08.zip",
        "September 2025": "data/aggTrades/BTCUSDT-aggTrades-2025-09.zip",
    }

    all_results = {}

    for month_name, zip_path in months.items():
        if not os.path.exists(zip_path):
            print(f"\n⚠️  文件未找到: {zip_path}")
            continue

        print(f"\n{'='*80}")
        print(f"测试月份: {month_name}")
        print("=" * 80)

        try:
            # Load test data
            test_data = load_and_prepare_test_data(zip_path)
            print(f"加载数据: {len(test_data)} bars")
            print(f"时间范围: {test_data.index[0]} 到 {test_data.index[-1]}")
            print(
                f"价格范围: {test_data['close'].min():.2f} 到 {test_data['close'].max():.2f}"
            )

            # Set test data
            data_loader.raw_data = test_data

            # Get multi-timeframe data
            multi_tf_data = data_loader.get_multi_timeframe_data()

            # Engineer features (using fitted scalers)
            print("工程化特征（使用训练时的scaler）...")
            engineered_data = feature_engineer.engineer_features(
                multi_tf_data, fit=False
            )

            # Test 5T timeframe
            timeframe = "5T"
            data = engineered_data[timeframe]

            stage1_model = strategy.pipeline.stage1_models.get(timeframe)
            stage2_model = strategy.pipeline.stage2_models.get(timeframe)

            if stage1_model is None:
                print(f"⚠️  未找到{timeframe}模型")
                continue

            # Prepare features
            feature_columns = [
                col
                for col in data.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            X = data[feature_columns]

            # Prepare targets
            y_stage1, y_stage2 = strategy.pipeline.prepare_targets(data)

            # Remove NaN
            valid_indices = ~(X.isna().any(axis=1) | y_stage1.isna() | y_stage2.isna())
            X_clean = X[valid_indices]
            y_stage1_clean = y_stage1[valid_indices]
            y_stage2_clean = y_stage2[valid_indices]

            print(f"\n{timeframe} 周期:")
            print(f"  有效样本: {len(X_clean)}")

            # Predict
            pred_stage1 = stage1_model.predict(X_clean)
            pred_stage2 = stage2_model.predict(X_clean)

            # Evaluate
            from sklearn.metrics import (
                accuracy_score,
                mean_squared_error,
                mean_absolute_error,
            )

            pred_stage1_binary = (pred_stage1 > 0.5).astype(int)
            y_stage1_binary = (y_stage1_clean == 1).astype(int)

            accuracy = accuracy_score(y_stage1_binary, pred_stage1_binary)
            mse = mean_squared_error(y_stage2_clean, pred_stage2)
            rmse = np.sqrt(mse)

            # Calculate trading metrics
            future_returns = data["close"].shift(-1) / data["close"] - 1
            future_returns_clean = future_returns[valid_indices]

            long_predictions = pred_stage1_binary == 1

            if long_predictions.sum() > 0:
                actual_returns_when_pred_long = future_returns_clean[long_predictions]

                win_rate = (actual_returns_when_pred_long > 0).sum() / len(
                    actual_returns_when_pred_long
                )
                total_return = actual_returns_when_pred_long.sum()
                avg_return = actual_returns_when_pred_long.mean()

                # 盈亏比
                winning_trades = actual_returns_when_pred_long[
                    actual_returns_when_pred_long > 0
                ]
                losing_trades = actual_returns_when_pred_long[
                    actual_returns_when_pred_long < 0
                ]

                avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
                avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
                profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")

                results = {
                    "accuracy": accuracy,
                    "rmse": rmse,
                    "win_rate": win_rate,
                    "total_return": total_return,
                    "avg_return": avg_return,
                    "num_signals": long_predictions.sum(),
                    "num_samples": len(X_clean),
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "profit_factor": profit_factor,
                }

                all_results[month_name] = results

                print(f"  准确率: {accuracy*100:.2f}%")
                print(f"  胜率: {win_rate*100:.2f}%")
                print(f"  总收益: {total_return*100:.2f}%")
                print(f"  交易次数: {long_predictions.sum()}")
                print(f"  盈亏比: {profit_factor:.2f}:1")

        except Exception as e:
            print(f"❌ 测试{month_name}时出错: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("📊 增强模型 OOS 测试总结")
    print("=" * 80)

    print("\n| 月份 | 准确率 | 胜率 | 收益率 | 交易次数 | 盈亏比 |")
    print("|------|--------|------|--------|----------|--------|")

    for month, res in all_results.items():
        print(
            f"| {month} | {res['accuracy']*100:.2f}% | {res['win_rate']*100:.2f}% | {res['total_return']*100:.2f}% | {res['num_signals']} | {res['profit_factor']:.2f}:1 |"
        )

    # Calculate average
    if all_results:
        avg_accuracy = np.mean([r["accuracy"] for r in all_results.values()])
        avg_winrate = np.mean([r["win_rate"] for r in all_results.values()])
        avg_return = np.mean([r["total_return"] for r in all_results.values()])
        avg_signals = np.mean([r["num_signals"] for r in all_results.values()])
        avg_pf = np.mean([r["profit_factor"] for r in all_results.values()])

        print(
            f"\n**平均值**: {avg_accuracy*100:.2f}% 准确率, {avg_winrate*100:.2f}% 胜率, {avg_return*100:.2f}% 收益, {avg_signals:.0f}次交易, {avg_pf:.2f}:1 盈亏比"
        )

    # Save results
    with open("oos_test_results_enhanced.json", "w") as f:
        json.dump(
            {
                k: {
                    key: (
                        float(val) if isinstance(val, (np.float32, np.float64)) else val
                    )
                    for key, val in v.items()
                }
                for k, v in all_results.items()
            },
            f,
            indent=2,
        )

    print(f"\n✅ 结果已保存到: oos_test_results_enhanced.json")

    return all_results


if __name__ == "__main__":
    test_enhanced_model_oos()
