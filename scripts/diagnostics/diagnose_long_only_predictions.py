"""
诊断 long_only 策略预测值偏低的问题

检查：
1. 训练数据的标签分布
2. CV 指标是否正常
3. 预测值分布
4. 模型性能
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from src.time_series_model.strategy_config.loader import StrategyConfigLoader
from scripts.train_strategy_pipeline import (
    run_feature_pipeline,
    determine_feature_columns,
    import_callable,
    generate_predictions,
)
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.data_tools.data_handler import MarketDataLoader


def diagnose_long_only_strategy(
    strategy_config_path: str = "config/strategies/sr_reversal_long",
    symbol: str = "BTCUSDT",
    data_path: str = "data/parquet_data",
):
    """诊断 long_only 策略的预测值问题"""
    print("=" * 80)
    print("诊断 long_only 策略预测值偏低问题")
    print("=" * 80)
    print()

    # 1. 加载策略配置
    print("1. 加载策略配置...")
    config_loader = StrategyConfigLoader(strategy_config_path)
    strategy_config = config_loader.load()
    print(f"   策略名称: {strategy_config.name}")
    print(f"   任务类型: {strategy_config.model.trainer.params.get('task_type')}")
    print()

    # 2. 加载数据
    print("2. 加载数据...")
    data_loader = MarketDataLoader(data_path=data_path)
    df_raw = data_loader.load_data(symbol=symbol, timeframe="60T")
    print(f"   数据形状: {df_raw.shape}")
    print(f"   时间范围: {df_raw.index.min()} 到 {df_raw.index.max()}")
    print()

    # 3. 计算特征
    print("3. 计算特征...")
    feature_loader = StrategyFeatureLoader()

    # 配置 tick loader（如果需要）
    try:
        from scripts.train_strategy_pipeline import _ensure_ticks_configured

        start_ts = str(df_raw.index.min())
        end_ts = str(df_raw.index.max())
        requested_features = strategy_config.features.requested_features or []
        _ensure_ticks_configured(
            feature_loader, symbol, data_path, start_ts, end_ts, requested_features
        )
        print("   ✅ Tick loader 配置成功")
    except Exception as e:
        print(f"   ⚠️  Tick 配置失败（可能不需要）: {e}")

    df_features = run_feature_pipeline(
        df_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_config.features,
        fit=True,
    )
    print(f"   特征后数据形状: {df_features.shape}")
    print()

    # 4. 生成标签
    print("4. 生成标签...")
    label_func = import_callable(
        strategy_config.labels.generator.module,
        strategy_config.labels.generator.function,
    )
    target_col = strategy_config.labels.target_column
    df_features[target_col] = label_func(
        df_features.copy(), **strategy_config.labels.generator.params
    )
    print(f"   标签列: {target_col}")
    print()

    # 5. 检查标签分布
    print("5. 检查标签分布...")
    labels = df_features[target_col]
    print(f"   总样本数: {len(labels)}")
    print(f"   有效标签数: {labels.notna().sum()}")
    print(f"   NaN 数量: {labels.isna().sum()}")
    print()

    valid_labels = labels.dropna()
    if len(valid_labels) > 0:
        print(f"   标签值分布:")
        print(
            f"     0 (负样本): {(valid_labels == 0).sum()} ({(valid_labels == 0).mean()*100:.2f}%)"
        )
        print(
            f"     1 (正样本): {(valid_labels == 1).sum()} ({(valid_labels == 1).mean()*100:.2f}%)"
        )
        print(f"     唯一值: {valid_labels.unique()}")
        print()

        if (valid_labels == 1).sum() == 0:
            print("   ⚠️  警告：没有正样本（label=1）！这是问题所在。")
            print("      模型无法学习做多信号，预测值会集中在低概率区间。")
            return
        elif (valid_labels == 1).mean() < 0.1:
            print(f"   ⚠️  警告：正样本比例过低 ({(valid_labels == 1).mean()*100:.2f}%)")
            print("      这会导致模型倾向于预测低概率。")
    else:
        print("   ❌ 错误：没有有效标签！")
        return

    # 6. 应用过滤器
    print("6. 应用过滤器...")
    from scripts.train_strategy_pipeline import apply_filters, apply_post_label_filters

    feature_cols = determine_feature_columns(df_features, strategy_config.features)
    df_filtered = apply_filters(df_features, strategy_config.labels.filters)
    df_filtered = apply_post_label_filters(
        df_filtered, strategy_config.labels.post_label_filters, feature_cols
    )
    print(f"   过滤后数据形状: {df_filtered.shape}")

    filtered_labels = df_filtered[target_col].dropna()
    if len(filtered_labels) > 0:
        print(f"   过滤后标签分布:")
        print(
            f"     0 (负样本): {(filtered_labels == 0).sum()} ({(filtered_labels == 0).mean()*100:.2f}%)"
        )
        print(
            f"     1 (正样本): {(filtered_labels == 1).sum()} ({(filtered_labels == 1).mean()*100:.2f}%)"
        )
    print()

    # 7. 模拟训练（只训练一个 fold 来检查）
    print("7. 检查模型训练...")
    trainer_func = import_callable(
        strategy_config.model.trainer.module, strategy_config.model.trainer.function
    )
    trainer_params = dict(strategy_config.model.trainer.params)
    target_col_param = trainer_params.pop("target_col", target_col)

    # 只使用前 1000 个样本进行快速测试
    df_sample = df_filtered.head(1000).copy()
    if len(df_sample) < 100:
        df_sample = df_filtered.copy()

    # 确保特征列都存在
    available_feature_cols = [col for col in feature_cols if col in df_sample.columns]
    missing_cols = [col for col in feature_cols if col not in df_sample.columns]
    if missing_cols:
        print(f"   ⚠️  缺失特征列: {missing_cols[:10]}... (共 {len(missing_cols)} 个)")
        print(f"   使用可用特征: {len(available_feature_cols)} 个")

    if len(available_feature_cols) == 0:
        print("   ❌ 没有可用的特征列！")
        return

    X_sample = df_sample[available_feature_cols].values
    y_sample = df_sample[target_col].values

    # 检查数据质量
    print(f"   样本数: {len(df_sample)}")
    print(f"   请求特征数: {len(feature_cols)}")
    print(f"   可用特征数: {len(available_feature_cols)}")
    if len(available_feature_cols) > 0:
        # 直接计算 NaN 比例，避免 DataFrame 构造问题
        X_nan_ratio = np.isnan(X_sample).mean()
        print(f"   特征 NaN 比例: {X_nan_ratio:.4f}")
    print(f"   标签 NaN 比例: {pd.Series(y_sample).isna().mean():.4f}")
    print()

    # 8. 训练模型并检查预测
    print("8. 训练模型并检查预测分布...")
    try:
        # 从 trainer_params 中移除 n_splits（如果存在），因为我们显式传递了
        # 同时禁用 GPU（本地环境可能没有 GPU 版本的 LightGBM）
        trainer_params_clean = {
            k: v for k, v in trainer_params.items() if k != "n_splits"
        }
        trainer_params_clean["use_gpu"] = False  # 禁用 GPU 以适配本地环境
        models, avg_metric, cv_results, used_features = trainer_func(
            df_sample,
            feature_cols=available_feature_cols,  # 使用可用的特征列
            target_col=target_col_param,
            n_splits=3,  # 减少 fold 数量以加快测试
            **trainer_params_clean,
        )

        print(f"   平均 CV Metric: {avg_metric:.4f}")
        print(f"   使用的特征数: {len(used_features)}")
        print()

        # 生成预测
        X_test = df_sample[used_features].values
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
        print()

        print(f"   预测值分布:")
        print(f"     < 0.3: {(preds < 0.3).sum()} ({(preds < 0.3).mean()*100:.2f}%)")
        print(
            f"     0.3-0.5: {((preds >= 0.3) & (preds < 0.5)).sum()} ({((preds >= 0.3) & (preds < 0.5)).mean()*100:.2f}%)"
        )
        print(
            f"     0.5-0.6: {((preds >= 0.5) & (preds < 0.6)).sum()} ({((preds >= 0.5) & (preds < 0.6)).mean()*100:.2f}%)"
        )
        print(f"     >= 0.6: {(preds >= 0.6).sum()} ({(preds >= 0.6).mean()*100:.2f}%)")
        print()

        if preds.max() < 0.5:
            print("   ⚠️  警告：所有预测值都低于 0.5！")
            print("      这表明模型没有学到有效的做多信号。")
            print("      可能原因：")
            print("        1. 正样本太少或质量差")
            print("        2. 特征质量不好")
            print("        3. 模型训练有问题")
        elif (preds >= 0.6).sum() == 0:
            print("   ⚠️  警告：没有预测值 >= 0.6（long_entry_threshold）")
            print("      这会导致没有交易信号。")
            print("      建议：")
            print("        1. 降低 long_entry_threshold（如改为 0.5）")
            print("        2. 检查模型训练和特征质量")

    except Exception as e:
        print(f"   ❌ 训练失败: {e}")
        import traceback

        traceback.print_exc()

    print()
    print("=" * 80)
    print("诊断完成")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy-config", default="config/strategies/sr_reversal_long"
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--data-path", default="data/parquet_data")

    args = parser.parse_args()

    diagnose_long_only_strategy(
        strategy_config_path=args.strategy_config,
        symbol=args.symbol,
        data_path=args.data_path,
    )
