#!/usr/bin/env python3
"""
Failure 子标签分布分析脚本

🟥 核心问题：
"当前被模型选中的 trades 里，哪一类 failure 仍然大量存在？"

分析两种 failure 子类型：
1. failure_rr_extreme: 路径极端不利（踩了大坑）
2. failure_no_opportunity: 无机会失败（入场即反向）

用法：
    python scripts/analyze_failure_distribution.py \
        --model-dir models/bpc \
        --data-path data/parquet_data \
        --symbol BTCUSDT \
        --timeframe 240T
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.time_series_model.strategies.labels.failure_first_label import (
    compute_failure_subtypes,
)


def load_model_and_data(
    model_dir: Path,
    data_path: Path,
    symbol: str,
    timeframe: str,
) -> tuple:
    """加载模型和数据"""
    import pickle

    # 加载模型
    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    # 如果是 list，包装成 dict
    if isinstance(model_data, list):
        model_data = {"models": model_data}

    # 加载 used_features
    features_path = model_dir / "used_features.json"
    if features_path.exists():
        import json

        with open(features_path) as f:
            model_data["feature_cols"] = json.load(f)

    # 尝试直接从 parquet 加载
    import pyarrow.parquet as pq

    data_file = Path(data_path) / f"{symbol}_{timeframe}.parquet"
    if data_file.exists():
        df = pq.read_table(data_file).to_pandas()
        return model_data, df

    # 尝试使用 FeatureStore 加载
    try:
        from src.time_series_model.feature_engineering.feature_store import (
            FeatureStore,
        )

        fs = FeatureStore(data_path=str(data_path))
        # 获取全部数据（不指定日期范围）
        df = fs.load_features(
            symbol=symbol,
            timeframe=timeframe,
        )
        if df is not None and len(df) > 0:
            return model_data, df
    except Exception as e:
        print(f"⚠️ FeatureStore 加载失败: {e}")

    raise FileNotFoundError(
        f"数据不存在且无法通过 FeatureStore 加载: {symbol}_{timeframe}"
    )


def get_model_selected_trades(
    model_data: dict,
    df: pd.DataFrame,
    feature_cols: list,
    entry_threshold: float = 0.3,
) -> pd.Series:
    """
    获取模型选中的 trades（success_prob >= threshold）

    Returns:
        pd.Series: 布尔索引，True = 被选中的 trade
    """
    import lightgbm as lgb

    # 获取模型（可能是列表或单个模型）
    models = model_data.get("models", [model_data.get("model")])
    if not models or models[0] is None:
        raise ValueError("无法从 model.pkl 中获取模型")

    # 准备特征，填充缺失列为 NaN
    available_features = [f for f in feature_cols if f in df.columns]
    missing_features = [f for f in feature_cols if f not in df.columns]

    if missing_features:
        print(f"   ⚠️ 缺失特征 ({len(missing_features)}): {missing_features[:5]}...")
        # 添加缺失列为 NaN
        for f in missing_features:
            df[f] = np.nan

    X = df[feature_cols].values

    # 预测（多模型取平均）
    preds_list = []
    for model in models:
        if model is None:
            continue
        if isinstance(model, lgb.Booster):
            pred = model.predict(X)
        else:
            pred = (
                model.predict_proba(X)[:, 1]
                if hasattr(model, "predict_proba")
                else model.predict(X)
            )
        preds_list.append(pred)

    if not preds_list:
        raise ValueError("无有效模型预测")

    preds = np.mean(preds_list, axis=0)

    # 选中的 trades
    selected = preds >= entry_threshold

    return pd.Series(selected, index=df.index, name="model_selected")


def analyze_failure_distribution(
    df: pd.DataFrame,
    selected_mask: pd.Series,
    direction: str = "long",
    horizon: int = 50,
) -> dict:
    """
    分析被选中 trades 中的 failure 分布

    Returns:
        dict: 包含统计信息
    """
    # 计算 failure 子标签
    failure_df = compute_failure_subtypes(
        df=df,
        direction=direction,
        horizon=horizon,
    )

    # 合并
    analysis_df = failure_df.copy()
    analysis_df["selected"] = selected_mask.values

    # 过滤有效样本
    valid_mask = analysis_df["failure_any"].notna()
    analysis_df = analysis_df[valid_mask]

    # 全局统计
    total_samples = len(analysis_df)
    selected_samples = analysis_df["selected"].sum()

    # 全局 failure rate
    global_rr_extreme = (analysis_df["failure_rr_extreme"] == 1).mean()
    global_no_opp = (analysis_df["failure_no_opportunity"] == 1).mean()
    global_any = (analysis_df["failure_any"] == 1).mean()

    # 选中 trades 中的 failure rate
    selected_df = analysis_df[analysis_df["selected"]]
    if len(selected_df) == 0:
        print("⚠️ 没有被模型选中的 trades")
        return {}

    selected_rr_extreme = (selected_df["failure_rr_extreme"] == 1).mean()
    selected_no_opp = (selected_df["failure_no_opportunity"] == 1).mean()
    selected_any = (selected_df["failure_any"] == 1).mean()

    # 未选中 trades 中的 failure rate
    unselected_df = analysis_df[~analysis_df["selected"]]
    unselected_rr_extreme = (unselected_df["failure_rr_extreme"] == 1).mean()
    unselected_no_opp = (unselected_df["failure_no_opportunity"] == 1).mean()
    unselected_any = (unselected_df["failure_any"] == 1).mean()

    # Lift: 选中 vs 全局
    lift_rr_extreme = (
        selected_rr_extreme / global_rr_extreme if global_rr_extreme > 0 else 0
    )
    lift_no_opp = selected_no_opp / global_no_opp if global_no_opp > 0 else 0
    lift_any = selected_any / global_any if global_any > 0 else 0

    results = {
        "total_samples": total_samples,
        "selected_samples": int(selected_samples),
        "selection_rate": selected_samples / total_samples if total_samples > 0 else 0,
        "global": {
            "failure_rr_extreme": global_rr_extreme,
            "failure_no_opportunity": global_no_opp,
            "failure_any": global_any,
        },
        "selected": {
            "failure_rr_extreme": selected_rr_extreme,
            "failure_no_opportunity": selected_no_opp,
            "failure_any": selected_any,
            "forward_rr_mean": selected_df["forward_rr"].mean(),
            "forward_rr_median": selected_df["forward_rr"].median(),
            "mfe_atr_mean": selected_df["mfe_atr"].mean(),
            "mae_atr_mean": selected_df["mae_atr"].mean(),
        },
        "unselected": {
            "failure_rr_extreme": unselected_rr_extreme,
            "failure_no_opportunity": unselected_no_opp,
            "failure_any": unselected_any,
        },
        "lift_vs_global": {
            "failure_rr_extreme": lift_rr_extreme,
            "failure_no_opportunity": lift_no_opp,
            "failure_any": lift_any,
        },
        "reduction_vs_unselected": {
            "failure_rr_extreme": (
                1 - selected_rr_extreme / unselected_rr_extreme
                if unselected_rr_extreme > 0
                else 0
            ),
            "failure_no_opportunity": (
                1 - selected_no_opp / unselected_no_opp if unselected_no_opp > 0 else 0
            ),
            "failure_any": (
                1 - selected_any / unselected_any if unselected_any > 0 else 0
            ),
        },
    }

    return results


def print_analysis_report(results: dict, symbol: str):
    """打印分析报告"""
    if not results:
        return

    print("\n" + "=" * 70)
    print(f"📊 FAILURE 子标签分布分析 - {symbol}")
    print("=" * 70)

    print(f"\n📈 样本统计:")
    print(f"   总样本数: {results['total_samples']:,}")
    print(f"   被选中数: {results['selected_samples']:,}")
    print(f"   选中率: {results['selection_rate']:.1%}")

    print(f"\n🌍 全局 Failure Rate (baseline):")
    g = results["global"]
    print(f"   failure_rr_extreme:     {g['failure_rr_extreme']:.1%}  (踩大坑)")
    print(f"   failure_no_opportunity: {g['failure_no_opportunity']:.1%}  (入场即反)")
    print(f"   failure_any:            {g['failure_any']:.1%}  (任一失败)")

    print(f"\n✅ 被模型选中的 Trades 中 Failure Rate:")
    s = results["selected"]
    print(f"   failure_rr_extreme:     {s['failure_rr_extreme']:.1%}")
    print(f"   failure_no_opportunity: {s['failure_no_opportunity']:.1%}")
    print(f"   failure_any:            {s['failure_any']:.1%}")
    print(f"   ---")
    print(f"   forward_rr 均值: {s['forward_rr_mean']:.2f}R")
    print(f"   forward_rr 中位数: {s['forward_rr_median']:.2f}R")
    print(f"   MFE 均值: {s['mfe_atr_mean']:.2f} ATR")
    print(f"   MAE 均值: {s['mae_atr_mean']:.2f} ATR")

    print(f"\n❌ 未被选中的 Trades 中 Failure Rate:")
    u = results["unselected"]
    print(f"   failure_rr_extreme:     {u['failure_rr_extreme']:.1%}")
    print(f"   failure_no_opportunity: {u['failure_no_opportunity']:.1%}")
    print(f"   failure_any:            {u['failure_any']:.1%}")

    print(f"\n📉 Lift vs 全局 (< 1.0 = 好):")
    l = results["lift_vs_global"]
    print(f"   failure_rr_extreme:     {l['failure_rr_extreme']:.2f}x")
    print(f"   failure_no_opportunity: {l['failure_no_opportunity']:.2f}x")
    print(f"   failure_any:            {l['failure_any']:.2f}x")

    print(f"\n🎯 Reduction vs 未选中 (正数 = 好):")
    r = results["reduction_vs_unselected"]
    print(f"   failure_rr_extreme:     {r['failure_rr_extreme']:+.1%}")
    print(f"   failure_no_opportunity: {r['failure_no_opportunity']:+.1%}")
    print(f"   failure_any:            {r['failure_any']:+.1%}")

    # 诊断建议
    print(f"\n" + "=" * 70)
    print("💡 诊断建议:")
    print("=" * 70)

    if s["failure_rr_extreme"] > 0.15:
        print(f"   ⚠️ failure_rr_extreme 仍有 {s['failure_rr_extreme']:.1%}")
        print(f"      → 模型未能识别'踩大坑'的结构条件")
        print(f"      → 考虑增加 volatility/trend_strength 相关特征")

    if s["failure_no_opportunity"] > 0.20:
        print(f"   ⚠️ failure_no_opportunity 仍有 {s['failure_no_opportunity']:.1%}")
        print(f"      → 大量入场后立刻反向")
        print(f"      → 考虑增加 momentum/timing 相关特征")

    if l["failure_any"] > 0.9:
        print(f"   🚨 模型选中的 trades 与全局 failure rate 几乎无差异!")
        print(f"      → 模型可能没有学到有效的 failure 区分能力")
        print(f"      → 检查特征是否与 failure 真正相关")
    elif l["failure_any"] < 0.7:
        print(f"   ✅ 模型有效降低了 {(1-l['failure_any']):.1%} 的 failure rate")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="分析模型选中 trades 中的 failure 分布"
    )
    parser.add_argument("--model-dir", type=str, default="models/bpc", help="模型目录")
    parser.add_argument(
        "--feature-store-dir",
        type=str,
        default="feature_store",
        help="FeatureStore 目录",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="bpc_highcap6_240T_v1",
        help="FeatureStore layer 名称",
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对")
    parser.add_argument("--timeframe", type=str, default="240T", help="时间周期")
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=0.3,
        help="入场阈值（success_prob >= threshold 视为选中）",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="long",
        choices=["long", "short"],
        help="交易方向",
    )
    parser.add_argument("--horizon", type=int, default=50, help="持仓窗口（bars）")
    parser.add_argument(
        "--output", type=str, default=None, help="输出 CSV 路径（可选）"
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    fs_dir = Path(args.feature_store_dir)

    print(f"\n🔍 加载模型和数据...")
    print(f"   模型: {model_dir}")

    # 加载模型
    import pickle
    import json
    import pyarrow.parquet as pq

    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        print(f"❌ 模型不存在: {model_path}")
        sys.exit(1)

    with open(model_path, "rb") as f:
        models = pickle.load(f)
    if isinstance(models, list):
        models = {"models": models}

    # 加载特征列表
    features_path = model_dir / "used_features.json"
    if features_path.exists():
        with open(features_path) as f:
            feature_cols = json.load(f)
    else:
        print("❌ 无法找到 used_features.json")
        sys.exit(1)

    print(f"   特征数: {len(feature_cols)}")

    # 从 FeatureStore 加载数据
    data_dir = fs_dir / args.layer / args.symbol / args.timeframe
    if not data_dir.exists():
        print(f"❌ FeatureStore 数据不存在: {data_dir}")
        print(f"   请检查 layer 名称是否正确")
        print(f"   可用 layers: {[d.name for d in fs_dir.iterdir() if d.is_dir()]}")
        sys.exit(1)

    # 加载所有月份的 parquet 文件
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"❌ 找不到 parquet 文件: {data_dir}")
        sys.exit(1)

    print(f"   加载 {len(parquet_files)} 个月份的数据...")
    dfs = []
    for pf in parquet_files:
        df_month = pq.read_table(pf).to_pandas()
        dfs.append(df_month)

    df = pd.concat(dfs, ignore_index=False)
    df = df.sort_index()

    print(f"   样本数: {len(df):,}")

    # 获取模型预测
    print(f"\n🎯 获取模型选中的 trades (threshold={args.entry_threshold})...")
    try:
        selected_mask = get_model_selected_trades(
            model_data=models,
            df=df,
            feature_cols=feature_cols,
            entry_threshold=args.entry_threshold,
        )
        print(f"   选中数: {selected_mask.sum():,} / {len(df):,}")
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print(f"\n📊 分析 failure 分布...")
    results = analyze_failure_distribution(
        df=df,
        selected_mask=selected_mask,
        direction=args.direction,
        horizon=args.horizon,
    )

    print_analysis_report(results, args.symbol)

    # 可选：输出到 CSV
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 生成详细 DataFrame
        failure_df = compute_failure_subtypes(
            df=df,
            direction=args.direction,
            horizon=args.horizon,
        )
        failure_df["model_selected"] = selected_mask.values
        failure_df.to_csv(output_path, index=True)
        print(f"📁 详细数据已保存到: {output_path}")


if __name__ == "__main__":
    main()
