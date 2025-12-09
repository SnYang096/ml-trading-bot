"""
分析DTW特征和波动率模型
1. 分析哪些DTW特征适合反转策略
2. 检查波动率模型是否使用了GARCH特征
3. 分析波动率预测不准确的原因
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

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
import lightgbm as lgb

warnings.filterwarnings("ignore")


def analyze_dtw_features_for_reversal(
    df_features: pd.DataFrame, labels: pd.Series
) -> pd.DataFrame:
    """
    分析DTW特征与反转标签的相关性
    """
    print("\n" + "=" * 70)
    print("📊 DTW特征分析：适合反转的特征")
    print("=" * 70)

    # 定义反转相关的DTW特征
    reversal_bullish = [
        "dtw_hammer_dist",
        "dtw_head_shoulder_bottom_dist",
        "dtw_double_bottom_dist",
        "dtw_bullish_engulfing_dist",
    ]

    reversal_bearish = [
        "dtw_shooting_star_dist",
        "dtw_head_shoulder_top_dist",
        "dtw_double_top_dist",
        "dtw_bearish_engulfing_dist",
    ]

    continuation = [
        "dtw_bull_flag_dist",
        "dtw_bear_flag_dist",
        "dtw_triangle_dist",
        "dtw_decline_consolidation_dist",
    ]

    # 获取所有DTW特征
    dtw_cols = [col for col in df_features.columns if col.startswith("dtw_")]

    if not dtw_cols:
        print("   ⚠️  No DTW features found!")
        return pd.DataFrame()

    print(f"\n   ✅ Found {len(dtw_cols)} DTW features")

    # 分析每个DTW特征与标签的相关性
    results = []

    # 只分析有标签的样本
    valid_mask = labels.notna()
    df_valid = df_features[valid_mask].copy()
    labels_valid = labels[valid_mask]

    for col in dtw_cols:
        if col not in df_valid.columns:
            continue

        # 确保是数值类型
        dtw_values = pd.to_numeric(df_valid[col], errors="coerce").fillna(
            1e6
        )  # 缺失值用大值填充（距离大）

        # 计算相关性（注意：距离越小，越像该形态）
        # 对于反转形态，我们希望：距离小 → 标签为正（成功反转）
        # 所以应该用负距离或1/距离来算相关性
        dtw_inverse = 1.0 / (dtw_values.values + 1e-6)  # 距离的倒数，越大越好

        correlation = np.corrcoef(dtw_inverse, labels_valid)[0, 1]

        # 分析不同距离区间的胜率
        # 距离小（<0.5）vs 距离大（>1.0）
        small_dist_mask = dtw_values.values < 0.5
        large_dist_mask = dtw_values.values > 1.0

        win_rate_small = (
            labels_valid[small_dist_mask].mean()
            if small_dist_mask.sum() > 0
            else np.nan
        )
        win_rate_large = (
            labels_valid[large_dist_mask].mean()
            if large_dist_mask.sum() > 0
            else np.nan
        )

        # 分类
        category = "Unknown"
        if col in reversal_bullish:
            category = "Bullish Reversal"
        elif col in reversal_bearish:
            category = "Bearish Reversal"
        elif col in continuation:
            category = "Continuation"

        results.append(
            {
                "feature": col,
                "category": category,
                "correlation": correlation,
                "n_samples": len(df_valid),
                "n_small_dist": small_dist_mask.sum(),
                "n_large_dist": large_dist_mask.sum(),
                "win_rate_small_dist": win_rate_small,
                "win_rate_large_dist": win_rate_large,
                "mean_distance": dtw_values.mean(),
                "std_distance": dtw_values.std(),
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("correlation", ascending=False)

    print("\n   📈 DTW特征与反转标签相关性（按相关性排序）：")
    print("   " + "-" * 68)
    print(
        f"   {'特征':<35} {'类别':<20} {'相关性':>8} {'小距离胜率':>12} {'大距离胜率':>12}"
    )
    print("   " + "-" * 68)

    for _, row in results_df.iterrows():
        corr_str = (
            f"{row['correlation']:.4f}" if not np.isnan(row["correlation"]) else "N/A"
        )
        wr_small = (
            f"{row['win_rate_small_dist']:.2%}"
            if not np.isnan(row["win_rate_small_dist"])
            else "N/A"
        )
        wr_large = (
            f"{row['win_rate_large_dist']:.2%}"
            if not np.isnan(row["win_rate_large_dist"])
            else "N/A"
        )
        print(
            f"   {row['feature']:<35} {row['category']:<20} {corr_str:>8} {wr_small:>12} {wr_large:>12}"
        )

    # 推荐适合反转的特征
    print("\n   ✅ 推荐用于反转策略的DTW特征（相关性>0.05或小距离胜率>50%）：")
    recommended = results_df[
        (results_df["correlation"] > 0.05) | (results_df["win_rate_small_dist"] > 0.5)
    ]

    if len(recommended) > 0:
        for _, row in recommended.iterrows():
            print(
                f"      - {row['feature']} ({row['category']}): 相关性={row['correlation']:.4f}, 小距离胜率={row['win_rate_small_dist']:.2%}"
            )
    else:
        print("      ⚠️  没有找到明显适合反转的DTW特征")

    return results_df


def analyze_volatility_model_features(
    df_train: pd.DataFrame,
    y_vol_train: pd.Series,
) -> None:
    """
    分析波动率模型使用的特征
    """
    print("\n" + "=" * 70)
    print("📊 波动率模型特征分析")
    print("=" * 70)

    # 准备特征
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

    # 选择波动率相关特征
    volatility_relevant_features = []

    # GARCH特征
    garch_features = [col for col in numeric_cols if col.startswith("garch_")]
    volatility_relevant_features.extend(garch_features)

    # 注意：EVT特征不用于波动率预测，而是用于风险管理/仓位控制（离场、不加仓）

    # ATR相关特征
    atr_features = [col for col in numeric_cols if "atr" in col.lower()]
    volatility_relevant_features.extend(atr_features)

    # 波动率相关特征
    vol_features = [
        col
        for col in numeric_cols
        if "vol" in col.lower() or "volatility" in col.lower()
    ]
    volatility_relevant_features.extend(vol_features)

    # 其他可能相关的特征
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

    # 去重
    volatility_relevant_features = list(set(volatility_relevant_features))
    available_features = [
        f for f in volatility_relevant_features if f in df_train.columns
    ]

    print(f"\n   ✅ 波动率相关特征统计：")
    print(
        f"      GARCH特征: {len([f for f in available_features if f.startswith('garch_')])}"
    )
    print(
        f"      ATR特征: {len([f for f in available_features if 'atr' in f.lower()])}"
    )
    print(
        f"      其他波动率特征: {len([f for f in available_features if f not in garch_features + atr_features])}"
    )
    print(f"      总计: {len(available_features)}")
    print(f"      注意: EVT特征已排除（用于风险管理，不用于波动率预测）")

    if len(garch_features) > 0:
        print(f"\n   ✅ GARCH特征已加载：")
        for garch_col in garch_features:
            if garch_col in df_train.columns:
                mean_val = df_train[garch_col].mean()
                std_val = df_train[garch_col].std()
                print(f"      - {garch_col}: mean={mean_val:.6f}, std={std_val:.6f}")
    else:
        print(f"\n   ⚠️  未找到GARCH特征！")

    # 分析特征与波动率标签的相关性
    print(f"\n   📈 特征与波动率标签相关性（Top 10）：")
    X_train = df_train[available_features].fillna(0)
    y_vol = y_vol_train.fillna(y_vol_train.median())

    correlations = []
    for col in available_features:
        if col in X_train.columns:
            corr = np.corrcoef(X_train[col].values, y_vol.values)[0, 1]
            if not np.isnan(corr):
                correlations.append((col, abs(corr), corr))

    correlations.sort(key=lambda x: x[1], reverse=True)

    print(f"   {'特征':<35} {'|相关性|':>12} {'相关性':>12}")
    print("   " + "-" * 60)
    for col, abs_corr, corr in correlations[:10]:
        print(f"   {col:<35} {abs_corr:>12.4f} {corr:>12.4f}")

    return available_features


def analyze_volatility_prediction_accuracy(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_vol_train: pd.Series,
    y_vol_test: pd.Series,
    available_features: list,
) -> None:
    """
    分析波动率预测准确性
    """
    print("\n" + "=" * 70)
    print("📊 波动率预测准确性分析")
    print("=" * 70)

    # 准备训练数据
    X_train = df_train[available_features].fillna(0)
    X_test = df_test[available_features].fillna(0)
    y_vol_train_clean = y_vol_train.fillna(y_vol_train.median())
    y_vol_test_clean = y_vol_test.fillna(y_vol_test.median())

    # 训练模型
    print("\n   🔧 训练波动率模型...")
    train_data = lgb.Dataset(X_train.values, label=y_vol_train_clean.values)
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

    # 预测
    pred_vol_train = vol_model.predict(X_train.values)
    pred_vol_test = vol_model.predict(X_test.values)

    # 转换为绝对波动率
    prices_train = df_train["close"].values
    prices_test = df_test["close"].values

    pred_vol_abs_train = pred_vol_train * prices_train
    pred_vol_abs_test = pred_vol_test * prices_test
    actual_vol_abs_train = y_vol_train_clean.values * prices_train
    actual_vol_abs_test = y_vol_test_clean.values * prices_test

    # 获取ATR
    atr_train = (
        df_train["atr"].values
        if "atr" in df_train.columns
        else np.ones(len(df_train)) * 1000
    )
    atr_test = (
        df_test["atr"].values
        if "atr" in df_test.columns
        else np.ones(len(df_test)) * 1000
    )

    print("\n   📊 训练集预测准确性：")
    train_rmse = np.sqrt(np.mean((pred_vol_abs_train - actual_vol_abs_train) ** 2))
    train_mae = np.mean(np.abs(pred_vol_abs_train - actual_vol_abs_train))
    train_corr = np.corrcoef(pred_vol_abs_train, actual_vol_abs_train)[0, 1]

    print(f"      RMSE: {train_rmse:.2f}")
    print(f"      MAE: {train_mae:.2f}")
    print(f"      相关性: {train_corr:.4f}")
    print(
        f"      预测均值: {np.mean(pred_vol_abs_train):.2f}, 实际均值: {np.mean(actual_vol_abs_train):.2f}"
    )
    print(f"      预测/ATR均值: {np.mean(pred_vol_abs_train / (atr_train + 1e-8)):.3f}")
    print(
        f"      实际/ATR均值: {np.mean(actual_vol_abs_train / (atr_train + 1e-8)):.3f}"
    )

    print("\n   📊 测试集预测准确性：")
    test_rmse = np.sqrt(np.mean((pred_vol_abs_test - actual_vol_abs_test) ** 2))
    test_mae = np.mean(np.abs(pred_vol_abs_test - actual_vol_abs_test))
    test_corr = np.corrcoef(pred_vol_abs_test, actual_vol_abs_test)[0, 1]

    print(f"      RMSE: {test_rmse:.2f}")
    print(f"      MAE: {test_mae:.2f}")
    print(f"      相关性: {test_corr:.4f}")
    print(
        f"      预测均值: {np.mean(pred_vol_abs_test):.2f}, 实际均值: {np.mean(actual_vol_abs_test):.2f}"
    )
    print(f"      预测/ATR均值: {np.mean(pred_vol_abs_test / (atr_test + 1e-8)):.3f}")
    print(f"      实际/ATR均值: {np.mean(actual_vol_abs_test / (atr_test + 1e-8)):.3f}")

    # 分析预测偏差
    print("\n   🔍 预测偏差分析：")
    pred_atr_ratio_train = pred_vol_abs_train / (atr_train + 1e-8)
    pred_atr_ratio_test = pred_vol_abs_test / (atr_test + 1e-8)
    actual_atr_ratio_train = actual_vol_abs_train / (atr_train + 1e-8)
    actual_atr_ratio_test = actual_vol_abs_test / (atr_test + 1e-8)

    print(f"      训练集 - 预测/ATR vs 实际/ATR:")
    print(
        f"         预测: {np.mean(pred_atr_ratio_train):.3f} ± {np.std(pred_atr_ratio_train):.3f}"
    )
    print(
        f"         实际: {np.mean(actual_atr_ratio_train):.3f} ± {np.std(actual_atr_ratio_train):.3f}"
    )
    print(
        f"         偏差: {np.mean(pred_atr_ratio_train - actual_atr_ratio_train):.3f}"
    )

    print(f"      测试集 - 预测/ATR vs 实际/ATR:")
    print(
        f"         预测: {np.mean(pred_atr_ratio_test):.3f} ± {np.std(pred_atr_ratio_test):.3f}"
    )
    print(
        f"         实际: {np.mean(actual_atr_ratio_test):.3f} ± {np.std(actual_atr_ratio_test):.3f}"
    )
    print(f"         偏差: {np.mean(pred_atr_ratio_test - actual_atr_ratio_test):.3f}")

    # 特征重要性
    print("\n   📊 特征重要性（Top 10）：")
    feature_importance = vol_model.feature_importance(importance_type="gain")
    feature_names = available_features

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": feature_importance,
        }
    ).sort_values("importance", ascending=False)

    for _, row in importance_df.head(10).iterrows():
        print(f"      {row['feature']:<35} {row['importance']:>12.2f}")


def main():
    """主函数"""
    print("=" * 70)
    print("🔍 DTW特征和波动率模型分析")
    print("=" * 70)

    # 加载数据
    print("\n📊 Loading data...")
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
    test_vol_labels = future_volatility_label(
        df_test["close"],
        horizon=10,
    )

    # 1. 分析DTW特征
    dtw_results = analyze_dtw_features_for_reversal(df_train, train_labels)

    # 2. 分析波动率模型特征
    vol_features = analyze_volatility_model_features(df_train, train_vol_labels)

    # 3. 分析波动率预测准确性
    analyze_volatility_prediction_accuracy(
        df_train,
        df_test,
        train_vol_labels,
        test_vol_labels,
        vol_features,
    )

    print("\n" + "=" * 70)
    print("✅ 分析完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
