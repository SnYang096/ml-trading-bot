"""
通用滚动回测脚本 - 支持多种集成方法

支持的集成方法:
- independent: 任何时间框架强信号都可触发
- weighted: 加权投票，大周期权重更高
- hierarchical: 大周期定趋势，小周期找时机
- majority: 多数投票
- average: 原始平均方法
"""

import os
import zipfile
import pandas as pd
import numpy as np
import json
import warnings

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb

    print(f"✅ LightGBM version: {lgb.__version__}")
except ImportError:
    print("❌ LightGBM not installed. Install: pip install lightgbm")
    exit(1)

from sklearn.metrics import accuracy_score


def extract_zip(zip_path):
    """Extract zip file."""
    temp_dir = os.path.join(
        os.path.dirname(zip_path),
        f'temp_extract_{os.path.basename(zip_path).replace(".", "_")}',
    )
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(temp_dir)

    csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV in {zip_path}")

    return os.path.join(temp_dir, csv_files[0]), temp_dir


def load_and_resample(csv_path, freq="5T"):
    """Load aggTrades and create OHLCV bars."""
    df = pd.read_csv(csv_path)

    if "transact_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df.set_index("timestamp", inplace=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["price", "quantity"])

    ohlc = df.groupby(pd.Grouper(freq=freq)).agg(
        {"price": ["first", "max", "min", "last"], "quantity": "sum"}
    )
    ohlc.columns = ["open", "high", "low", "close", "volume"]
    ohlc = ohlc.dropna().ffill()

    return ohlc


def add_features(df):
    """Add technical indicators."""
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    for window in [5, 10, 20, 50]:
        df[f"sma_{window}"] = df["close"].rolling(window).mean()
        df[f"price_to_sma_{window}"] = df["close"] / df[f"sma_{window}"]

    df["volatility_20"] = df["returns"].rolling(20).std()

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    exp1 = df["close"].ewm(span=12).mean()
    exp2 = df["close"].ewm(span=26).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )

    df["hl"] = df["high"] - df["low"]
    df["hc"] = abs(df["high"] - df["close"].shift(1))
    df["lc"] = abs(df["low"] - df["close"].shift(1))
    df["tr"] = df[["hl", "hc", "lc"]].max(axis=1)
    df["atr"] = df["tr"].rolling(14).mean()

    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"]

    for period in [5, 10, 20]:
        df[f"momentum_{period}"] = df["close"] - df["close"].shift(period)
        df[f"roc_{period}"] = (df["close"] - df["close"].shift(period)) / df[
            "close"
        ].shift(period)

    df["high_low_ratio"] = df["high"] / df["low"]
    df["close_to_high"] = df["close"] / df["high"]
    df["close_to_low"] = df["close"] / df["low"]

    return df


def create_labels(df, forward_bars=3, threshold=0.005):
    """Create labels."""
    df["future_return"] = df["close"].shift(-forward_bars) / df["close"] - 1
    df["signal"] = 0
    df.loc[df["future_return"] > threshold, "signal"] = 1
    df.loc[df["future_return"] < -threshold, "signal"] = -1
    return df


def train_models_multi_tf(data_dict):
    """Train models for multiple timeframes."""
    models = {}

    for tf, df in data_dict.items():
        df = add_features(df)
        df = create_labels(df, forward_bars=3, threshold=0.005)
        df = df.dropna()

        if len(df) < 100:
            continue

        feature_cols = [
            col
            for col in df.columns
            if col
            not in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "signal",
                "future_return",
                "hl",
                "hc",
                "lc",
                "tr",
            ]
        ]

        X = df[feature_cols].values
        y_binary = (df["signal"] == 1).astype(int).values
        y_return = df["future_return"].values

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y1_train, y1_val = y_binary[:split_idx], y_binary[split_idx:]
        y2_train, y2_val = y_return[:split_idx], y_return[split_idx:]

        # Stage 1
        params1 = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "verbose": -1,
        }

        train_data1 = lgb.Dataset(X_train, label=y1_train)
        val_data1 = lgb.Dataset(X_val, label=y1_val)
        model1 = lgb.train(
            params1,
            train_data1,
            num_boost_round=300,
            valid_sets=[val_data1],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        # Stage 2
        params2 = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "verbose": -1,
        }

        train_data2 = lgb.Dataset(X_train, label=y2_train)
        val_data2 = lgb.Dataset(X_val, label=y2_val)
        model2 = lgb.train(
            params2,
            train_data2,
            num_boost_round=300,
            valid_sets=[val_data2],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        models[tf] = {
            "model_stage1": model1,
            "model_stage2": model2,
            "feature_cols": feature_cols,
        }

    return models


def ensemble_predictions(
    stage1_preds_dict, stage2_preds_dict, method="weighted", signal_threshold=0.75
):
    """
    集成预测 - 支持多种方法

    Args:
        stage1_preds_dict: {timeframe: predictions}
        stage2_preds_dict: {timeframe: predictions}
        method: 集成方法
        signal_threshold: 信号阈值 (用于independent/weighted/majority)

    Returns:
        signals: array of -1/0/1
    """
    # 使用最短时间框架作为基准
    base_tf = "5T" if "5T" in stage1_preds_dict else list(stage1_preds_dict.keys())[0]
    n_samples = len(stage1_preds_dict[base_tf])

    if method == "independent":
        # 任何时间框架的强信号都可触发
        tf_signals = {}
        for tf, preds in stage1_preds_dict.items():
            tf_signal = np.zeros(len(preds))
            tf_signal[preds > signal_threshold] = 1
            tf_signal[preds < (1 - signal_threshold)] = -1
            tf_signals[tf] = tf_signal

        signals = tf_signals[base_tf].copy()

    elif method == "weighted":
        # 加权投票：大周期权重更高
        timeframes = sorted(stage1_preds_dict.keys(), key=lambda x: int(x.rstrip("T")))
        tf_minutes = [int(tf.rstrip("T")) for tf in timeframes]

        # 使用平方根权重
        raw_weights = [np.sqrt(minutes) for minutes in tf_minutes]
        weights = [w / sum(raw_weights) for w in raw_weights]

        # 计算加权信号 - 修复：确保所有信号长度一致
        weighted_signal = np.zeros(n_samples)
        for i, tf in enumerate(timeframes):
            preds = stage1_preds_dict[tf]
            # 对齐到n_samples长度
            if len(preds) > n_samples:
                preds = preds[:n_samples]
            elif len(preds) < n_samples:
                # 如果太短，用0填充或者重复最后一个值
                preds_aligned = np.zeros(n_samples)
                preds_aligned[: len(preds)] = preds
                # 用最后一个值填充剩余部分
                if len(preds) > 0:
                    preds_aligned[len(preds) :] = preds[-1]
                preds = preds_aligned

            # 转换为[-1, 0, 1]
            tf_signal = np.zeros(n_samples)
            tf_signal[preds > 0.6] = 1
            tf_signal[preds < 0.4] = -1
            weighted_signal += tf_signal * weights[i]

        # 转换为离散信号
        signals = np.zeros(n_samples)
        signals[weighted_signal > 0.3] = 1
        signals[weighted_signal < -0.3] = -1

    elif method == "hierarchical":
        # 大周期定趋势，小周期找时机
        timeframes = sorted(stage1_preds_dict.keys(), key=lambda x: int(x.rstrip("T")))
        largest_tf = timeframes[-1]
        smallest_tf = timeframes[0]

        # 趋势方向（大周期）
        trend_preds = stage1_preds_dict[largest_tf]
        trend_signal = np.zeros(len(trend_preds))
        trend_signal[trend_preds > 0.5] = 1
        trend_signal[trend_preds < 0.5] = -1

        # 入场时机（小周期）
        entry_preds = stage1_preds_dict[smallest_tf][:n_samples]
        entry_signal = np.zeros(len(entry_preds))
        entry_signal[entry_preds > 0.5] = 1
        entry_signal[entry_preds < 0.5] = -1

        # 只有趋势和入场同向才开仓
        signals = np.zeros(n_samples)
        for i in range(min(len(trend_signal), n_samples)):
            # 如果大周期数据不够长，使用最后一个值
            trend_idx = min(i, len(trend_signal) - 1)
            if trend_signal[trend_idx] > 0 and entry_signal[i] > 0:
                signals[i] = 1
            elif trend_signal[trend_idx] < 0 and entry_signal[i] < 0:
                signals[i] = -1

    elif method == "majority":
        # 多数投票
        signals = np.zeros(n_samples)
        n_tf = len(stage1_preds_dict)

        for i in range(n_samples):
            count_long = 0
            count_short = 0

            for tf, preds in stage1_preds_dict.items():
                if i < len(preds):
                    if preds[i] > 0.6:
                        count_long += 1
                    elif preds[i] < 0.4:
                        count_short += 1

            if count_long > n_tf / 2:
                signals[i] = 1
            elif count_short > n_tf / 2:
                signals[i] = -1

    else:  # average
        # 原始平均方法
        avg_signal = np.zeros(n_samples)

        for tf, preds in stage1_preds_dict.items():
            # 转换为[-1, 0, 1]
            tf_signal = np.zeros(len(preds[:n_samples]))
            tf_signal[preds[:n_samples] > 0.6] = 1
            tf_signal[preds[:n_samples] < 0.4] = -1
            avg_signal += tf_signal

        avg_signal /= len(stage1_preds_dict)

        signals = np.zeros(n_samples)
        signals[avg_signal > 0.1] = 1
        signals[avg_signal < -0.1] = -1

    return signals


def generate_signals(
    models_dict, data_dict, ensemble_method="weighted", signal_threshold=0.75
):
    """Generate signals using trained models."""
    stage1_preds = {}
    stage2_preds = {}

    for tf, model_info in models_dict.items():
        if tf not in data_dict:
            continue

        df = data_dict[tf].copy()
        df = add_features(df)
        df = df.dropna()

        if len(df) == 0:
            continue

        X = df[model_info["feature_cols"]].values

        pred1 = model_info["model_stage1"].predict(X)
        pred2 = model_info["model_stage2"].predict(X)

        stage1_preds[tf] = pred1
        stage2_preds[tf] = pred2

    signals = ensemble_predictions(
        stage1_preds,
        stage2_preds,
        method=ensemble_method,
        signal_threshold=signal_threshold,
    )

    return signals


def simple_backtest(df, signals):
    """Simple backtest."""
    if len(df) == 0 or len(signals) == 0:
        return None

    df_bt = df.copy()
    df_bt = df_bt.iloc[: len(signals)]
    df_bt["signal"] = signals
    df_bt["returns"] = df_bt["close"].pct_change()
    df_bt["position"] = df_bt["signal"].shift(1).fillna(0)
    df_bt["strategy_returns"] = df_bt["position"] * df_bt["returns"]

    cumulative = (1 + df_bt["strategy_returns"]).cumprod()
    total_return = (cumulative.iloc[-1] - 1) * 100

    bh_cumulative = (1 + df_bt["returns"]).cumprod()
    bh_return = (bh_cumulative.iloc[-1] - 1) * 100

    position_changes = df_bt["position"].diff().fillna(0)
    total_trades = (position_changes != 0).sum()

    winning = df_bt[df_bt["strategy_returns"] > 0]["strategy_returns"].count()
    losing = df_bt[df_bt["strategy_returns"] < 0]["strategy_returns"].count()
    win_rate = winning / (winning + losing) * 100 if (winning + losing) > 0 else 0

    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    gross_profit = df_bt[df_bt["strategy_returns"] > 0]["strategy_returns"].sum()
    gross_loss = abs(df_bt[df_bt["strategy_returns"] < 0]["strategy_returns"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    return {
        "total_return": total_return,
        "bh_return": bh_return,
        "total_trades": int(total_trades),
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "long_signals": int((signals == 1).sum()),
        "short_signals": int((signals == -1).sum()),
    }


def run_backtest(ensemble_method="weighted", signal_threshold=0.75):
    """运行回测."""
    print("\n" + "=" * 80)
    print(f"🚀 滚动回测 - {ensemble_method.upper()} 集成方法")
    print("=" * 80)
    print(f"\n⚙️  配置:")
    print(f"   - 集成方法: {ensemble_method}")
    if ensemble_method in ["independent", "weighted"]:
        print(f"   - 信号阈值: {signal_threshold}")

    data_dir = r"D:\GitHub\trading\rlbot\data\agg_data"
    results_dir = f"results/rolling_{ensemble_method}"
    os.makedirs(results_dir, exist_ok=True)

    train_months = ["2024-10", "2024-11", "2024-12", "2025-01"]
    test_months = ["2025-02", "2025-03", "2025-04"]
    timeframes = ["5T", "15T", "60T"]

    temp_dirs = []
    all_results = []

    try:
        # Train
        print(f"\n{'='*80}")
        print(f"📚 训练阶段: {', '.join(train_months)}")
        print(f"{'='*80}")

        train_data_base = []
        for month in train_months:
            zip_path = os.path.join(data_dir, f"BTCUSDT-aggTrades-{month}.zip")
            if not os.path.exists(zip_path):
                continue

            csv_path, temp_dir = extract_zip(zip_path)
            temp_dirs.append(temp_dir)
            print(f"📦 {month}...")
            ohlcv = load_and_resample(csv_path, freq="5T")
            train_data_base.append(ohlcv)

        if not train_data_base:
            print("❌ 没有训练数据")
            return None

        train_base = pd.concat(train_data_base, axis=0).sort_index()
        print(f"✓ 训练数据 (5T): {len(train_base)} 条")

        # Multi-timeframe
        train_data_dict = {}
        for tf in timeframes:
            if tf == "5T":
                train_data_dict[tf] = train_base
            else:
                resampled = (
                    train_base.resample(tf)
                    .agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                        }
                    )
                    .dropna()
                )
                train_data_dict[tf] = resampled

        print(f"\n训练模型...")
        models_dict = train_models_multi_tf(train_data_dict)
        print(f"✓ 训练完成！")

        # Test
        for test_month in test_months:
            print(f"\n{'='*80}")
            print(f"📊 测试: {test_month}")
            print(f"{'='*80}")

            zip_path = os.path.join(data_dir, f"BTCUSDT-aggTrades-{test_month}.zip")
            if not os.path.exists(zip_path):
                continue

            csv_path, temp_dir = extract_zip(zip_path)
            temp_dirs.append(temp_dir)
            test_base = load_and_resample(csv_path, freq="5T")

            # Multi-timeframe test data
            test_data_dict = {}
            for tf in timeframes:
                if tf == "5T":
                    test_data_dict[tf] = test_base
                else:
                    resampled = (
                        test_base.resample(tf)
                        .agg(
                            {
                                "open": "first",
                                "high": "max",
                                "low": "min",
                                "close": "last",
                                "volume": "sum",
                            }
                        )
                        .dropna()
                    )
                    test_data_dict[tf] = resampled

            # Generate signals
            signals = generate_signals(
                models_dict,
                test_data_dict,
                ensemble_method=ensemble_method,
                signal_threshold=signal_threshold,
            )

            print(
                f"✓ 信号: 多={( signals==1).sum()} | 空={(signals==-1).sum()} | 不开仓={(signals==0).sum()}"
            )

            # Backtest
            results = simple_backtest(test_base, signals)

            if results:
                results["month"] = test_month
                results["method"] = ensemble_method
                all_results.append(results)

                print(
                    f"   收益: {results['total_return']:.2f}% | BH: {results['bh_return']:.2f}%"
                )
                print(
                    f"   交易: {results['total_trades']} | 胜率: {results['win_rate']:.2f}%"
                )

        # Summary
        if all_results:
            print(f"\n{'='*80}")
            print(f"📊 汇总")
            print(f"{'='*80}")

            df_results = pd.DataFrame(all_results)
            print(
                f"\n{df_results[['month', 'total_return', 'total_trades', 'win_rate']].to_string(index=False)}"
            )

            print(f"\n平均月收益: {df_results['total_return'].mean():.2f}%")
            print(f"平均交易: {df_results['total_trades'].mean():.0f}次")

            # Save
            summary_path = os.path.join(results_dir, "summary.csv")
            df_results.to_csv(summary_path, index=False)
            print(f"\n💾 已保存: {summary_path}")

            return df_results

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        import shutil

        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """主函数 - 测试多种集成方法"""
    print("\n" + "=" * 80)
    print("🎯 多集成方法对比测试")
    print("=" * 80)

    methods_to_test = [
        ("weighted", 0.75),
        ("hierarchical", None),
        # ('independent', 0.75),  # 已经测试过
    ]

    all_results = {}

    for method_info in methods_to_test:
        if len(method_info) == 2:
            method, threshold = method_info
        else:
            method = method_info[0]
            threshold = 0.75

        print(f"\n\n{'='*80}")
        print(f"开始测试: {method.upper()}")
        print(f"{'='*80}")

        results = run_backtest(
            ensemble_method=method, signal_threshold=threshold or 0.75
        )

        if results is not None:
            all_results[method] = results

    print(f"\n\n{'='*80}")
    print(f"✅ 所有测试完成！")
    print(f"{'='*80}")
    print(f"\n已测试方法: {', '.join(all_results.keys())}")
    print(f"\n运行对比分析:")
    print(f"python scripts/compare_all_ensemble_methods.py")


if __name__ == "__main__":
    main()
