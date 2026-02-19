"""
诊断模型预测值集中的问题

检查：
1. 模型预测值的分布
2. 特征重要性
3. 预测值与标签的关系
4. 模型是否真的在学习
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.time_series_model.strategy_config.loader import StrategyConfigLoader
from scripts.train_strategy_pipeline import (
    run_feature_pipeline,
    determine_feature_columns,
    import_callable,
    generate_predictions,
    apply_filters,
    apply_post_label_filters,
)
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.data_tools.data_handler import MarketDataLoader


def diagnose_prediction_concentration(
    strategy_config_path: str = "config/strategies/sr_reversal_long",
    symbol: str = "BTCUSDT",
    data_path: str = "data/parquet_data",
    sample_size: int = 2000,
):
    """诊断预测值集中的问题"""
    print("=" * 80)
    print("诊断模型预测值集中问题")
    print("=" * 80)
    print()

    # 1. 加载配置和数据
    print("1. 加载配置和数据...")
    config_loader = StrategyConfigLoader(strategy_config_path)
    strategy_config = config_loader.load()

    data_loader = MarketDataLoader(data_path=data_path)
    df_raw = data_loader.load_data(symbol=symbol, timeframe="60T")
    print(f"   原始数据: {df_raw.shape}")
    print()

    # 2. 计算特征
    print("2. 计算特征...")
    feature_loader = StrategyFeatureLoader()

    # 配置 tick loader
    try:
        from scripts.train_strategy_pipeline import _ensure_ticks_configured

        start_ts = str(df_raw.index.min())
        end_ts = str(df_raw.index.max())
        requested_features = strategy_config.features.requested_features or []
        _ensure_ticks_configured(
            feature_loader, symbol, data_path, start_ts, end_ts, requested_features
        )
    except Exception as e:
        print(f"   ⚠️  Tick 配置失败: {e}")

    df_features = run_feature_pipeline(
        df_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_config.features,
        fit=True,
    )
    print(f"   特征后数据: {df_features.shape}")
    print()

    # 3. 生成标签
    print("3. 生成标签...")
    label_func = import_callable(
        strategy_config.labels.generator.module,
        strategy_config.labels.generator.function,
    )
    target_col = strategy_config.labels.target_column
    df_features[target_col] = label_func(
        df_features.copy(), **strategy_config.labels.generator.params
    )

    # 应用过滤器
    feature_cols = determine_feature_columns(df_features, strategy_config.features)
    df_filtered = apply_filters(df_features, strategy_config.labels.filters)
    df_filtered = apply_post_label_filters(
        df_filtered, strategy_config.labels.post_label_filters, feature_cols
    )

    # 使用更多样本
    df_sample = df_filtered.head(sample_size).copy()
    if len(df_sample) < 100:
        df_sample = df_filtered.copy()

    print(f"   样本数: {len(df_sample)}")
    print(
        f"   标签分布: 0={(df_sample[target_col] == 0).sum()}, 1={(df_sample[target_col] == 1).sum()}"
    )
    print()

    # 4. 训练模型
    print("4. 训练模型...")
    trainer_func = import_callable(
        strategy_config.model.trainer.module, strategy_config.model.trainer.function
    )
    trainer_params = dict(strategy_config.model.trainer.params)
    target_col_param = trainer_params.pop("target_col", target_col)
    trainer_params["use_gpu"] = False  # 本地环境可能没有 GPU

    # 移除 n_splits（如果存在），因为我们显式传递
    trainer_params_clean = {k: v for k, v in trainer_params.items() if k != "n_splits"}

    available_feature_cols = [col for col in feature_cols if col in df_sample.columns]
    print(f"   使用特征数: {len(available_feature_cols)}")

    models, avg_metric, cv_results, used_features = trainer_func(
        df_sample,
        feature_cols=available_feature_cols,
        target_col=target_col_param,
        n_splits=5,
        **trainer_params_clean,
    )

    print(f"   平均 CV Metric: {avg_metric:.4f}")
    print(f"   使用的特征数: {len(used_features)}")
    print()

    # 5. 生成预测并分析
    print("5. 分析预测值分布...")
    X_test = df_sample[used_features].values
    y_test = df_sample[target_col].values

    preds = generate_predictions(
        models=models,
        model_type=trainer_params.get("model_type", "lightgbm"),
        task_type=trainer_params.get("task_type", "binary"),
        X=X_test,
    )

    print(f"   预测值统计:")
    print(f"     形状: {preds.shape}")
    print(f"     范围: [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"     均值: {preds.mean():.4f}")
    print(f"     中位数: {np.median(preds):.4f}")
    print(f"     标准差: {preds.std():.4f}")
    print(f"     唯一值数量: {len(np.unique(preds))}")
    print()

    # 检查预测值是否过于集中
    if preds.std() < 0.01:
        print("   ⚠️  警告：预测值标准差 < 0.01，模型预测几乎相同！")
        print("      这表明模型没有学到有效的区分信号。")
    elif preds.std() < 0.05:
        print("   ⚠️  警告：预测值标准差 < 0.05，预测值过于集中。")

    # 6. 检查特征重要性
    print("6. 检查特征重要性...")
    try:
        import lightgbm as lgb

        if isinstance(models[0], lgb.Booster):
            feature_importance = models[0].feature_importance(importance_type="gain")
            importance_df = pd.DataFrame(
                {"feature": used_features, "importance": feature_importance}
            ).sort_values("importance", ascending=False)

            print(f"   前 10 个最重要的特征:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"     {row['feature']}: {row['importance']:.2f}")
            print()

            # 检查是否有特征重要性为 0
            zero_importance = (importance_df["importance"] == 0).sum()
            if zero_importance > 0:
                print(f"   ⚠️  警告：{zero_importance} 个特征的重要性为 0（模型未使用）")
    except Exception as e:
        print(f"   ⚠️  无法获取特征重要性: {e}")
    print()

    # 7. 检查预测值与标签的关系
    print("7. 检查预测值与标签的关系...")
    valid_mask = ~np.isnan(y_test)
    if valid_mask.sum() > 0:
        preds_valid = preds[valid_mask]
        y_test_valid = y_test[valid_mask]

        # 按标签分组统计
        preds_label_0 = preds_valid[y_test_valid == 0]
        preds_label_1 = preds_valid[y_test_valid == 1]

        print(f"   Label=0 的预测值:")
        print(f"     均值: {preds_label_0.mean():.4f}")
        print(f"     中位数: {np.median(preds_label_0):.4f}")
        print(f"     标准差: {preds_label_0.std():.4f}")
        print()

        print(f"   Label=1 的预测值:")
        print(f"     均值: {preds_label_1.mean():.4f}")
        print(f"     中位数: {np.median(preds_label_1):.4f}")
        print(f"     标准差: {preds_label_1.std():.4f}")
        print()

        # 检查区分度
        mean_diff = preds_label_1.mean() - preds_label_0.mean()
        print(f"   预测值差异 (Label=1 - Label=0): {mean_diff:.4f}")
        if abs(mean_diff) < 0.05:
            print("   ⚠️  警告：Label=0 和 Label=1 的预测值差异很小！")
            print("      模型无法区分正负样本。")
        elif mean_diff < 0:
            print("   ⚠️  警告：Label=1 的预测值反而更低！")
            print("      模型可能学习反了，或者标签有问题。")
    print()

    # 8. 检查模型是否过拟合或欠拟合
    print("8. 检查模型训练质量...")
    if hasattr(cv_results, "columns") and "metric" in cv_results.columns:
        fold_metrics = cv_results["metric"].values
        print(f"   各 Fold 的 Metric:")
        for i, metric in enumerate(fold_metrics, 1):
            print(f"     Fold {i}: {metric:.4f}")
        print(f"   Metric 标准差: {fold_metrics.std():.4f}")

        if fold_metrics.std() < 0.01:
            print("   ⚠️  警告：各 Fold 的 Metric 几乎相同，可能过拟合。")
        elif fold_metrics.max() - fold_metrics.min() > 0.5:
            print("   ⚠️  警告：各 Fold 的 Metric 差异很大，模型不稳定。")
    print()

    # 9. 检查特征质量
    print("9. 检查特征质量...")
    # 直接使用 numpy 数组，避免 DataFrame 构造问题
    X_test_array = X_test

    # 确保 used_features 和 X_test_array 的列数匹配
    if len(used_features) != X_test_array.shape[1]:
        print(
            f"   ⚠️  警告：特征数量不匹配！used_features={len(used_features)}, X_test_array.shape[1]={X_test_array.shape[1]}"
        )
        # 使用实际的特征数量
        actual_feature_count = min(len(used_features), X_test_array.shape[1])
        used_features = used_features[:actual_feature_count]
        X_test_array = X_test_array[:, :actual_feature_count]

    # 检查特征方差（排除标签列）
    feature_vars = np.var(X_test_array, axis=0)
    # 排除标签列（如果存在）
    feature_names_to_check = [f for f in used_features if f != target_col]
    feature_indices_to_check = [
        i for i, f in enumerate(used_features) if f != target_col
    ]

    zero_var_features = (feature_vars < 1e-8).sum()
    if zero_var_features > 0:
        print(f"   ⚠️  警告：{zero_var_features} 个特征的方差为 0（常数特征）")
        zero_var_feat_indices = np.where(feature_vars < 1e-8)[0]

        # 获取常数特征名称
        zero_var_feat_names = []
        for idx in zero_var_feat_indices:
            if idx < len(used_features):
                feat_name = used_features[idx]
                # 排除标签列
                if feat_name != target_col:
                    zero_var_feat_names.append((feat_name, idx))

        print(f"      常数特征列表（共 {len(zero_var_feat_names)} 个，已排除标签列）:")
        for feat_name, feat_idx in zero_var_feat_names:
            if feat_idx < X_test_array.shape[1]:
                feat_values = X_test_array[:, feat_idx]
                unique_vals = np.unique(feat_values[~np.isnan(feat_values)])
                var_val = np.var(feat_values[~np.isnan(feat_values)])
                if len(unique_vals) <= 5:
                    print(
                        f"        - {feat_name}: 值 = {unique_vals.tolist()}, 方差 = {var_val:.2e}"
                    )
                else:
                    print(
                        f"        - {feat_name}: 值范围 = [{unique_vals[0]:.6f}, {unique_vals[-1]:.6f}], 唯一值数 = {len(unique_vals)}, 方差 = {var_val:.2e}"
                    )
            else:
                print(f"        - {feat_name}: 索引超出范围")
    else:
        print(f"   ✅ 所有特征都有方差（无常数特征）")

    # 检查特征与标签的相关性
    print(f"   特征与标签的相关性（前 10 个）:")
    correlations = []
    for i, feat_name in enumerate(used_features):
        feat_values = X_test_array[:, i]
        # 填充 NaN
        feat_values_clean = np.nan_to_num(feat_values, nan=0.0)
        corr = np.corrcoef(feat_values_clean, y_test)[0, 1]
        if not np.isnan(corr):
            correlations.append((feat_name, abs(corr)))

    correlations.sort(key=lambda x: x[1], reverse=True)
    for feat, corr in correlations[:10]:
        print(f"     {feat}: {corr:.4f}")

    if len(correlations) > 0 and max(c[1] for c in correlations) < 0.1:
        print("   ⚠️  警告：所有特征与标签的相关性都很低（< 0.1）")
        print("      特征可能没有提供有效的预测信号。")
    print()

    # 10. 总结和建议
    print("=" * 80)
    print("诊断总结")
    print("=" * 80)

    issues = []
    if preds.std() < 0.05:
        issues.append("预测值过于集中（标准差 < 0.05）")
    if len(correlations) > 0 and max(c[1] for c in correlations) < 0.1:
        issues.append("特征与标签相关性低（< 0.1）")
    if valid_mask.sum() > 0:
        mean_diff = preds_label_1.mean() - preds_label_0.mean()
        if abs(mean_diff) < 0.05:
            issues.append("模型无法区分正负样本")

    if issues:
        print("发现的问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print()
        print("建议:")
        print("  1. 检查特征工程：特征是否包含有效的预测信号？")
        print("  2. 检查标签质量：标签是否正确生成？")
        print("  3. 检查模型训练：是否有过拟合或欠拟合？")
        print("  4. 考虑调整模型参数：学习率、树深度、正则化等")
        print("  5. 考虑使用更多/更好的特征")
    else:
        print("未发现明显问题。预测值集中可能是正常的，取决于模型和数据的特性。")

    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy-config", default="config/strategies/sr_reversal_long"
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--data-path", default="data/parquet_data")
    parser.add_argument("--sample-size", type=int, default=2000)

    args = parser.parse_args()

    diagnose_prediction_concentration(
        strategy_config_path=args.strategy_config,
        symbol=args.symbol,
        data_path=args.data_path,
        sample_size=args.sample_size,
    )
