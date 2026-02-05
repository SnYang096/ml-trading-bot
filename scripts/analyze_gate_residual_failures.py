#!/usr/bin/env python3
"""
Gate 失败归因分析脚本（改进版 - 方案2）

核心功能：
1. 加载 Gate 模型预测结果（直接读取训练输出的 predictions.parquet）
2. 筛选：success_prob >= threshold 的样本（Gate 通过）
3. 在这些样本上重新计算 failure 标签
4. 对比失败 vs 成功样本的 Evidence 特征分布
5. 输出特征均值差异，判断剩余失败是否来自 Evidence/Execution 层

用法：
    python scripts/analyze_gate_residual_failures.py \
        --model-dir results/train_final_20260205_011545_rr_extreme/bpc \
        --threshold 0.8 \
        --split holdout
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.time_series_model.strategies.labels.failure_first_label import (
    compute_failure_subtypes,
)


# Evidence 特征分组（用于归因分析）
EVIDENCE_FEATURES = {
    "波动regime": [
        "vol_regime_percentile",
        "vol_regime_z_score",
        "vol_expansion_phase",
    ],
    "追末端风险": [
        "terminal_risk_score_f",
        "exhaustion_volume_spike",
        "wick_exhaustion_score",
    ],
    "execution时机": [
        "volume_participation_score_f",
        "liquidity_void_score",
        "noise_ratio_score",
    ],
    "节奏错位": [
        "bpc_pullback_speed_f",
        "bpc_pullback_duration_f",
        "momentum_exhaustion_score",
    ],
    "订单流支持": [
        "cvd_divergence_v2_f",
        "bpc_pullback_delta_absorption_f",
        "vpin_imbalance_score",
    ],
}


def load_predictions(model_dir: Path, split: str = "holdout") -> pd.DataFrame:
    """
    加载模型预测结果

    Args:
        model_dir: 模型目录（如 results/train_final_xxx/bpc）
        split: 数据集划分（train/holdout/all）

    Returns:
        包含预测结果和原始特征的 DataFrame
    """
    import pyarrow.parquet as pq
    import pickle
    import json
    import lightgbm as lgb

    # 方案1: 尝试加载现成的 predictions.parquet
    pred_file = model_dir / "predictions.parquet"
    if pred_file.exists():
        print(f"   ✓ 从 predictions.parquet 加载")
        df = pq.read_table(pred_file).to_pandas()

        # 筛选数据集
        if split != "all":
            if "split" not in df.columns:
                raise ValueError(
                    f"predictions.parquet 中没有 'split' 列，无法筛选 {split}"
                )
            df = df[df["split"] == split].copy()

        print(f"   ✓ 加载 {split} 集: {len(df):,} 样本")
        return df

    # 方案2: 从 results.json 中读取配置，重新加载数据并预测
    print(f"   ⚠️ 未找到 predictions.parquet，尝试重新预测...")

    results_file = model_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"找不到 results.json: {results_file}")

    with open(results_file) as f:
        results = json.load(f)

    # 加载模型
    model_file = model_dir / "model.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"找不到 model.pkl: {model_file}")

    with open(model_file, "rb") as f:
        models = pickle.load(f)

    if isinstance(models, list):
        models = {"models": models}

    # 加载特征列表
    features_file = model_dir / "used_features.json"
    if not features_file.exists():
        raise FileNotFoundError(f"找不到 used_features.json: {features_file}")

    with open(features_file) as f:
        feature_cols = json.load(f)

    print(f"   ✓ 加载模型和特征: {len(feature_cols)} 个特征")

    # 从 results.json 推断数据路径和参数
    # 需要从父目录名推断训练参数
    train_dir_name = model_dir.parent.name  # 如 train_final_20260205_011545_rr_extreme

    # 推断 feature store layer
    # 从模型目录的元数据或使用默认值
    feature_store_layer = "bpc_highcap6_240T_v1"  # 默认值
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
    timeframe = "240T"

    # 尝试从 feature_config.json 获取更多信息
    feature_config_file = model_dir / "feature_config.json"
    if feature_config_file.exists():
        with open(feature_config_file) as f:
            feature_config = json.load(f)
            # 可能包含有用的元数据

    # 加载数据（简化版：直接从 feature store 加载）
    print(f"   正在从 feature store 加载数据...")
    print(f"   ⚠️ 注意：这个脚本需要手动指定数据加载参数")
    print(f"   请使用以下参数重新运行训练并保存 predictions.parquet")
    print(f"   或者使用 analyze_failure_distribution.py 脚本")

    raise NotImplementedError(
        "predictions.parquet 不存在，无法自动重新预测。\n"
        "请确保训练时保存了预测结果，或使用以下命令：\n"
        f"  python scripts/analyze_failure_distribution.py \\\n"
        f"    --model-dir {model_dir} \\\n"
        f"    --symbol BTCUSDT \\\n"
        f"    --timeframe 240T \\\n"
        f"    --entry-threshold 0.8"
    )


def analyze_residual_failures(
    df: pd.DataFrame,
    threshold: float = 0.8,
    direction: str = "long",
    horizon: int = 50,
) -> dict:
    """
    分析 Gate 通过后剩余失败的特征分布

    Args:
        df: 包含预测结果的 DataFrame
        threshold: Gate 阈值
        direction: 交易方向
        horizon: 持仓窗口

    Returns:
        分析结果字典
    """
    # 1. 筛选 Gate 通过的样本
    gate_passed = df[df["pred"] >= threshold].copy()
    print(
        f"\n📊 Gate 通过样本: {len(gate_passed):,} / {len(df):,} ({len(gate_passed)/len(df)*100:.1f}%)"
    )

    # 边界检查：如果没有样本通过，给出诊断提示
    if len(gate_passed) == 0:
        print(f"\n   ⚠️  没有样本通过 Gate (threshold={threshold})")
        print(f"   预测分布:")
        print(f"      min:  {df['pred'].min():.3f}")
        print(f"      max:  {df['pred'].max():.3f}")
        print(f"      mean: {df['pred'].mean():.3f}")
        print(f"      q50:  {df['pred'].quantile(0.50):.3f}")
        print(f"      q75:  {df['pred'].quantile(0.75):.3f}")
        print(f"      q90:  {df['pred'].quantile(0.90):.3f}")
        print(f"\n   💡 建议降低 threshold 或检查模型预测是否正常")
        return {}

    # 2. 计算 failure 标签
    print(f"   计算 failure 子标签...")
    failure_df = compute_failure_subtypes(
        df=gate_passed,
        direction=direction,
        horizon=horizon,
    )

    # 合并
    gate_passed = gate_passed.join(failure_df, how="inner")

    # 过滤有效样本
    valid_mask = gate_passed["failure_any"].notna()
    gate_passed = gate_passed[valid_mask]

    print(f"   有效样本: {len(gate_passed):,}")

    # 3. 分离成功 vs 失败样本
    failures = gate_passed[gate_passed["failure_rr_extreme"] == 1].copy()
    successes = gate_passed[gate_passed["failure_rr_extreme"] == 0].copy()

    failure_rate = len(failures) / len(gate_passed) if len(gate_passed) > 0 else 0

    print(
        f"\n   ✅ 成功样本: {len(successes):,} ({len(successes)/len(gate_passed)*100:.1f}%)"
    )
    print(f"   ❌ 失败样本: {len(failures):,} ({failure_rate*100:.1f}%)")

    if len(failures) == 0:
        print(f"\n   🎉 Gate 完美过滤，没有剩余失败！")
        return {}

    # 4. 特征差异分析
    print(f"\n🔍 分析剩余失败的特征分布...")

    feature_diffs = {}

    for category, features in EVIDENCE_FEATURES.items():
        category_diffs = []

        for feat in features:
            if feat not in gate_passed.columns:
                continue

            fail_mean = failures[feat].mean()
            succ_mean = successes[feat].mean()

            # 跳过全 NaN 的特征
            if pd.isna(fail_mean) or pd.isna(succ_mean):
                continue

            # 计算差异百分比
            if abs(succ_mean) > 1e-6:
                diff_pct = (fail_mean - succ_mean) / abs(succ_mean) * 100
            else:
                diff_pct = 0.0

            # 计算 Cohen's d (效应量)
            fail_std = failures[feat].std()
            succ_std = successes[feat].std()
            pooled_std = np.sqrt((fail_std**2 + succ_std**2) / 2)

            if pooled_std > 1e-6:
                cohens_d = (fail_mean - succ_mean) / pooled_std
            else:
                cohens_d = 0.0

            category_diffs.append(
                {
                    "feature": feat,
                    "fail_mean": fail_mean,
                    "succ_mean": succ_mean,
                    "diff_pct": diff_pct,
                    "cohens_d": cohens_d,
                }
            )

        if category_diffs:
            feature_diffs[category] = sorted(
                category_diffs,
                key=lambda x: abs(x["cohens_d"]),
                reverse=True,
            )

    return {
        "total_samples": len(gate_passed),
        "failure_count": len(failures),
        "success_count": len(successes),
        "failure_rate": failure_rate,
        "feature_diffs": feature_diffs,
        "failures_df": failures,
        "successes_df": successes,
    }


def print_analysis_report(results: dict, threshold: float):
    """打印分析报告"""
    if not results:
        return

    print("\n" + "=" * 70)
    print("📊 Gate 剩余失败归因分析")
    print("=" * 70)

    print(f"\n📈 样本统计:")
    print(f"   Gate threshold: {threshold}")
    print(f"   Gate 通过样本: {results['total_samples']:,}")
    print(
        f"   其中失败: {results['failure_count']:,} ({results['failure_rate']*100:.1f}%)"
    )
    print(
        f"   其中成功: {results['success_count']:,} ({results['success_count']/results['total_samples']*100:.1f}%)"
    )

    print(f"\n🔍 特征差异分析（失败 vs 成功）:")
    print(f"   Cohen's d 解读: |d| < 0.2=小, 0.2-0.8=中, >0.8=大")
    print()

    feature_diffs = results["feature_diffs"]

    # 按类别输出
    for category, diffs in feature_diffs.items():
        if not diffs:
            continue

        print(f"\n   【{category}】")
        for item in diffs[:3]:  # 只显示 top 3
            feat = item["feature"]
            fail_mean = item["fail_mean"]
            succ_mean = item["succ_mean"]
            diff_pct = item["diff_pct"]
            cohens_d = item["cohens_d"]

            # 效应量标记
            if abs(cohens_d) > 0.8:
                marker = "🔴"  # 大效应
            elif abs(cohens_d) > 0.2:
                marker = "🟡"  # 中效应
            else:
                marker = "⚪"  # 小效应

            print(f"      {marker} {feat}")
            print(f"         失败: {fail_mean:.3f}, 成功: {succ_mean:.3f}")
            print(f"         差异: {diff_pct:+.1f}%, Cohen's d: {cohens_d:+.2f}")

    # 诊断建议
    print(f"\n" + "=" * 70)
    print("💡 诊断建议:")
    print("=" * 70)

    # 找出最大效应量的类别
    max_effect_by_category = {}
    for category, diffs in feature_diffs.items():
        if diffs:
            max_d = max(abs(d["cohens_d"]) for d in diffs)
            max_effect_by_category[category] = max_d

    if not max_effect_by_category:
        print("   ⚠️ 没有找到有效的特征差异")
        return

    # 排序
    sorted_categories = sorted(
        max_effect_by_category.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"\n   剩余失败主要来自（按效应量排序）:\n")
    for i, (category, max_d) in enumerate(sorted_categories[:3], 1):
        if max_d > 0.5:
            print(f"   {i}. {category} (效应量: {max_d:.2f})")

    # 判断是否是 Evidence/Execution 层问题
    top_category = sorted_categories[0][0]
    top_effect = sorted_categories[0][1]

    if top_effect > 0.8:
        print(f"\n   ✅ 剩余失败主要来自【{top_category}】")
        print(f"      → 这不是 Gate 的职责范围")
        print(f"      → Gate 已经完成职责，剩余问题属于 Evidence/Execution 层")
    elif top_effect > 0.5:
        print(f"\n   ⚠️ 剩余失败部分来自【{top_category}】")
        print(f"      → Gate 基本完成职责")
        print(f"      → 可以考虑训练 Return Tree 进一步优化")
    else:
        print(f"\n   🚨 特征差异不明显（最大效应量 {top_effect:.2f}）")
        print(f"      → Gate 可能还不够强")
        print(f"      → 或者 Evidence 特征不够丰富")

    print()


def main():
    parser = argparse.ArgumentParser(description="分析 Gate 通过后剩余失败的归因")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="模型目录（如 results/train_final_xxx/bpc）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Gate 阈值（success_prob >= threshold 视为通过）",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="holdout",
        choices=["train", "holdout", "all"],
        help="数据集划分",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="long",
        choices=["long", "short"],
        help="交易方向",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="持仓窗口（bars）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出详细 CSV 路径（可选）",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    print(f"\n🔍 加载模型预测结果...")
    print(f"   模型: {model_dir}")
    print(f"   数据集: {args.split}")

    try:
        df = load_predictions(model_dir, args.split)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        sys.exit(1)

    print(f"\n🎯 分析 Gate 剩余失败...")
    results = analyze_residual_failures(
        df=df,
        threshold=args.threshold,
        direction=args.direction,
        horizon=args.horizon,
    )

    if results:
        print_analysis_report(results, args.threshold)

    # 可选：输出详细 CSV
    if args.output and results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 合并失败和成功样本
        failures = results["failures_df"].copy()
        failures["failure_type"] = "failure"

        successes = results["successes_df"].copy()
        successes["failure_type"] = "success"

        combined = pd.concat([failures, successes], ignore_index=True)
        combined.to_csv(output_path, index=False)

        print(f"📁 详细数据已保存到: {output_path}")


if __name__ == "__main__":
    main()
