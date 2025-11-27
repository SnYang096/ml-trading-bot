"""
分析ML+波动率模型效果差的原因
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.data_utils import load_raw_data
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.strategy_config import StrategyConfigLoader
from src.time_series_model.strategies.labels.sr_reversal_label import (
    SRSignalConfig,
    _generate_sr_reversal_signals,
    _ensure_atr,
)
from src.time_series_model.pipeline.training.label_utils import (
    compute_rr_label,
    future_volatility_label,
)
from scripts.diagnostics.compute_adaptive_rr_with_predicted_vol import (
    compute_adaptive_rr_label_with_predicted_vol,
)
import lightgbm as lgb


def analyze_ml_volatility_model():
    """分析ML+波动率模型的问题"""

    # 加载数据
    print("📊 Loading data...")
    df_raw = load_raw_data(
        data_path="data/parquet_data",
        symbol="BTCUSDT",
        timeframe="4H",
        start_date="2024-01-01",
        end_date="2025-10-31",
    )

    # 加载特征
    print("🔧 Loading features...")
    cfg_dir = Path("config/strategies/sr_reversal").resolve()
    strategy_cfg_loader = StrategyConfigLoader(cfg_dir)
    strategy_cfg = strategy_cfg_loader.load()

    feature_loader = StrategyFeatureLoader()
    from scripts import train_strategy_pipeline as strategy_runner

    df_features = strategy_runner.run_feature_pipeline(
        df_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=True,
    )

    # 分割训练/测试
    split_idx = int(len(df_features) * 0.85)
    df_train = df_features.iloc[:split_idx].copy()
    df_test = df_features.iloc[split_idx:].copy()

    # 确保ATR
    atr_series = _ensure_atr(df_features, "atr", "close", "high", "low", 14)
    atr_train = atr_series.iloc[:split_idx]
    atr_test = atr_series.iloc[split_idx:]

    # 生成信号
    rule_params = {
        "sr_strength_min": 0.3,
        "sqs_min": 0.7,
        "touch_distance_atr": 1.5,
        "stop_loss_r": 1.25,
        "take_profit_r": 3.0,
        "max_holding_bars": 72,
        "use_vpin_filter": False,
    }

    sr_cfg = SRSignalConfig(
        min_sr_strength=rule_params["sr_strength_min"],
        min_support_score=rule_params["sqs_min"],
        min_resistance_score=rule_params["sqs_min"],
        tolerance_mult=rule_params["touch_distance_atr"],
        use_vpin_filter=rule_params["use_vpin_filter"],
    )

    train_signals = _generate_sr_reversal_signals(
        df_train,
        price_col="close",
        high_col="high",
        low_col="low",
        atr_series=atr_train,
        cfg=sr_cfg,
    )
    df_train["signal"] = train_signals

    # 计算标签
    train_labels = compute_rr_label(
        df_train.copy(),
        signal_col="signal",
        price_col="close",
        atr_col="atr",
        atr_window=14,
        max_holding_bars=rule_params["max_holding_bars"],
        stop_loss_r=rule_params["stop_loss_r"],
        take_profit_r=rule_params["take_profit_r"],
        use_continuous_label=False,
        entry_price_col="open",
        entry_offset=1,
        use_breakeven_stop=False,
    )

    # 计算未来波动率标签
    train_vol_labels = future_volatility_label(
        df_train["close"],
        horizon=10,
    )

    # 准备特征 - 优先选择波动率相关特征
    feature_cols = [
        col
        for col in df_train.columns
        if col
        not in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "signal",
            "label",
            "atr",
            "_symbol",
            "symbol",
            "timestamp",
            "datetime",
            "date",
        ]
    ]
    numeric_cols = (
        df_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    )

    # 选择波动率相关特征（GARCH, 扩展波动率特征, ATR等）
    volatility_relevant_features = []

    # GARCH特征（波动聚集性和杠杆效应）- 关键特征
    garch_features = [col for col in numeric_cols if col.startswith("garch_")]
    volatility_relevant_features.extend(garch_features)

    # 注意：EVT特征不用于波动率预测，而是用于风险管理/仓位控制（离场、不加仓）

    # 扩展波动率特征（历史波动率、滞后特征、趋势特征）
    extended_vol_features = [col for col in numeric_cols if col.startswith("vol_")]
    volatility_relevant_features.extend(extended_vol_features)

    # 注意：DTW特征不用于波动率模型，而是用于SR Reversal策略（反转模板匹配）

    # ATR相关特征（历史波动率）
    atr_features = [col for col in numeric_cols if "atr" in col.lower()]
    volatility_relevant_features.extend(atr_features)

    # 波动率相关特征
    vol_features = [
        col
        for col in numeric_cols
        if "vol" in col.lower() or "volatility" in col.lower()
    ]
    volatility_relevant_features.extend(vol_features)

    # 其他可能相关的特征（波动率压缩、范围等）
    other_features = [
        col
        for col in numeric_cols
        if any(
            keyword in col.lower()
            for keyword in [
                "bb_width",
                "compression",
                "squeeze",
                "range",
                "range_ratio",
            ]
        )
    ]
    volatility_relevant_features.extend(other_features)

    # 去重并确保特征存在
    volatility_relevant_features = list(set(volatility_relevant_features))
    available_features = [
        f for f in volatility_relevant_features if f in df_train.columns
    ]

    if not available_features:
        print("   ⚠️ No volatility-specific features found, using all numeric features")
        available_features = numeric_cols
    else:
        print(f"   ✅ Selected {len(available_features)} volatility-relevant features:")
        print(
            f"      GARCH: {len([f for f in available_features if f.startswith('garch_')])}"
        )
        print(
            f"      Extended Volatility: {len([f for f in available_features if f.startswith('vol_')])}"
        )
        print(
            f"      ATR: {len([f for f in available_features if 'atr' in f.lower()])}"
        )
        print(
            f"      Other volatility-related: {len([f for f in available_features if f not in garch_features + extended_vol_features + atr_features])}"
        )
        print(
            f"      Note: EVT features excluded (used for risk management, not volatility prediction)"
        )

    X_train = df_train[available_features].fillna(0)
    y_train = train_labels.fillna(0).astype(int)
    y_vol_train = train_vol_labels.fillna(train_vol_labels.median())

    # 过滤有效样本
    valid_mask = (train_signals != 0) & train_labels.notna()
    X_train_valid = X_train[valid_mask]
    y_train_valid = y_train[valid_mask]
    y_vol_train_valid = y_vol_train[valid_mask]

    print(f"\n📊 Training data stats:")
    print(f"   Valid samples: {len(X_train_valid)}")
    print(
        f"   Positive labels: {int(y_train_valid.sum())} ({y_train_valid.mean():.2%})"
    )
    print(
        f"   Volatility labels - Mean: {y_vol_train_valid.mean():.6f}, Std: {y_vol_train_valid.std():.6f}"
    )

    # 训练波动率模型
    print("\n🔧 Training volatility model with volatility-specific features...")
    train_data = lgb.Dataset(X_train_valid.values, label=y_vol_train_valid.values)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }
    vol_model = lgb.train(params, train_data, num_boost_round=100)

    # 在测试集上预测（使用相同的特征）
    X_test = df_test[available_features].fillna(0)
    pred_vol_relative = vol_model.predict(X_test.values)
    pred_vol_relative = np.maximum(pred_vol_relative, 0.0)

    # 转换为绝对波动率
    prices_test = df_test["close"].values
    pred_vol_absolute = pred_vol_relative * prices_test

    # 获取实际ATR
    atr_test_values = atr_test.values

    # 计算实际未来波动率（用于对比）
    test_vol_labels = future_volatility_label(
        df_test["close"],
        horizon=10,
    )
    actual_vol_relative = test_vol_labels.fillna(test_vol_labels.median()).values
    actual_vol_absolute = actual_vol_relative * prices_test

    print("\n📊 Volatility Prediction Analysis:")
    print(
        f"   Predicted Vol (relative) - Mean: {np.mean(pred_vol_relative):.6f}, Std: {np.std(pred_vol_relative):.6f}"
    )
    print(
        f"   Actual Vol (relative) - Mean: {np.mean(actual_vol_relative):.6f}, Std: {np.std(actual_vol_relative):.6f}"
    )
    print(
        f"   Prediction Error (relative) - Mean: {np.mean(np.abs(pred_vol_relative - actual_vol_relative)):.6f}"
    )
    print(
        f"   Prediction Error (relative) - RMSE: {np.sqrt(np.mean((pred_vol_relative - actual_vol_relative)**2)):.6f}"
    )

    print(
        f"\n   Predicted Vol (absolute) - Mean: {np.mean(pred_vol_absolute):.2f}, Std: {np.std(pred_vol_absolute):.2f}"
    )
    print(
        f"   Actual Vol (absolute) - Mean: {np.mean(actual_vol_absolute):.2f}, Std: {np.std(actual_vol_absolute):.2f}"
    )
    print(
        f"   ATR - Mean: {np.mean(atr_test_values):.2f}, Std: {np.std(atr_test_values):.2f}"
    )

    # 分析预测波动率与ATR的关系
    pred_vol_atr_ratio = pred_vol_absolute / (atr_test_values + 1e-8)
    actual_vol_atr_ratio = actual_vol_absolute / (atr_test_values + 1e-8)

    print(
        f"\n   Predicted Vol / ATR ratio - Mean: {np.mean(pred_vol_atr_ratio):.3f}, Std: {np.std(pred_vol_atr_ratio):.3f}"
    )
    print(
        f"   Actual Vol / ATR ratio - Mean: {np.mean(actual_vol_atr_ratio):.3f}, Std: {np.std(actual_vol_atr_ratio):.3f}"
    )

    # 分析自适应R/R的影响
    print("\n📊 Adaptive R/R Impact Analysis:")

    # 固定R/R参数
    stop_loss_r = rule_params["stop_loss_r"]
    take_profit_r = rule_params["take_profit_r"]

    # 计算固定R/R的止损止盈范围
    fixed_sl_range = stop_loss_r * atr_test_values
    fixed_tp_range = take_profit_r * atr_test_values

    # 计算自适应R/R的止损止盈范围（使用预测波动率）
    atr_lower_bound = 0.5
    atr_upper_bound = 2.0
    final_vol = np.clip(
        pred_vol_absolute,
        atr_test_values * atr_lower_bound,
        atr_test_values * atr_upper_bound,
    )
    adaptive_sl_range = stop_loss_r * final_vol
    adaptive_tp_range = take_profit_r * final_vol

    print(
        f"   Fixed SL range - Mean: {np.mean(fixed_sl_range):.2f}, Std: {np.std(fixed_sl_range):.2f}"
    )
    print(
        f"   Adaptive SL range - Mean: {np.mean(adaptive_sl_range):.2f}, Std: {np.std(adaptive_sl_range):.2f}"
    )
    print(
        f"   SL range change - Mean: {np.mean(adaptive_sl_range - fixed_sl_range):.2f}"
    )
    print(
        f"   SL range change % - Mean: {np.mean((adaptive_sl_range - fixed_sl_range) / fixed_sl_range * 100):.2f}%"
    )

    print(
        f"\n   Fixed TP range - Mean: {np.mean(fixed_tp_range):.2f}, Std: {np.std(fixed_tp_range):.2f}"
    )
    print(
        f"   Adaptive TP range - Mean: {np.mean(adaptive_tp_range):.2f}, Std: {np.std(adaptive_tp_range):.2f}"
    )
    print(
        f"   TP range change - Mean: {np.mean(adaptive_tp_range - fixed_tp_range):.2f}"
    )
    print(
        f"   TP range change % - Mean: {np.mean((adaptive_tp_range - fixed_tp_range) / fixed_tp_range * 100):.2f}%"
    )

    # 分析为什么效果差
    print("\n🔍 Root Cause Analysis:")

    # 1. 检查预测波动率是否系统性偏差
    vol_bias = np.mean(pred_vol_relative - actual_vol_relative)
    print(f"   1. Volatility prediction bias: {vol_bias:.6f} ({vol_bias*100:.2f}%)")
    if abs(vol_bias) > 0.001:
        print(
            f"      ⚠️ Systematic bias detected! Predicted vol is {'higher' if vol_bias > 0 else 'lower'} than actual."
        )

    # 2. 检查预测波动率与ATR的匹配
    pred_atr_ratio_mean = np.mean(pred_vol_atr_ratio)
    actual_atr_ratio_mean = np.mean(actual_vol_atr_ratio)
    print(
        f"   2. Predicted Vol/ATR ratio: {pred_atr_ratio_mean:.3f} vs Actual: {actual_atr_ratio_mean:.3f}"
    )
    if abs(pred_atr_ratio_mean - actual_atr_ratio_mean) > 0.2:
        print(f"      ⚠️ Significant mismatch! This may cause incorrect R/R adjustment.")

    # 3. 检查自适应R/R是否导致止损止盈范围过大或过小
    sl_change_pct = np.mean((adaptive_sl_range - fixed_sl_range) / fixed_sl_range * 100)
    tp_change_pct = np.mean((adaptive_tp_range - fixed_tp_range) / fixed_tp_range * 100)
    print(
        f"   3. SL range change: {sl_change_pct:.2f}%, TP range change: {tp_change_pct:.2f}%"
    )
    if abs(sl_change_pct) > 20 or abs(tp_change_pct) > 20:
        print(
            f"      ⚠️ Large R/R adjustment! This may cause too many stop-outs or missed profits."
        )

    # 4. 检查预测波动率的准确性
    vol_mae = np.mean(np.abs(pred_vol_relative - actual_vol_relative))
    vol_rmse = np.sqrt(np.mean((pred_vol_relative - actual_vol_relative) ** 2))
    print(
        f"   4. Volatility prediction accuracy - MAE: {vol_mae:.6f}, RMSE: {vol_rmse:.6f}"
    )
    if vol_mae > 0.002 or vol_rmse > 0.003:
        print(
            f"      ⚠️ Poor prediction accuracy! Model may not be learning volatility well."
        )

    # 5. 检查是否有极端值
    extreme_pred_vol = np.sum((pred_vol_atr_ratio < 0.1) | (pred_vol_atr_ratio > 3.0))
    print(
        f"   5. Extreme predicted vol/ATR ratios (<0.1 or >3.0): {extreme_pred_vol} ({extreme_pred_vol/len(pred_vol_atr_ratio)*100:.2f}%)"
    )
    if extreme_pred_vol > len(pred_vol_atr_ratio) * 0.1:
        print(
            f"      ⚠️ Too many extreme values! May need better clipping or model improvement."
        )

    print("\n💡 Recommendations:")
    print(
        "   1. If prediction bias is large, retrain volatility model with better features"
    )
    print("   2. If Vol/ATR ratio mismatch, adjust atr_lower_bound and atr_upper_bound")
    print("   3. If R/R adjustment is too large, use more conservative bounds")
    print("   4. If prediction accuracy is poor, consider using ATR as fallback")
    print("   5. Consider using ensemble: (predicted_vol + ATR) / 2")


if __name__ == "__main__":
    analyze_ml_volatility_model()
