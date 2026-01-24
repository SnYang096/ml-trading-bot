#!/usr/bin/env python3
"""
将所有gate规则阈值调到最小并测试

如果优化后仍然没有开仓信号，将所有阈值调到最小（最宽松）进行测试。

使用方法:
    python scripts/test_gate_with_minimal_thresholds.py \
        --logs results/pipeline_with_reflexivity_2024_full/logs_execution.parquet \
        --execution-archetypes config/nnmultihead/execution_archetypes.yaml \
        --output results/gate_minimal_thresholds_test.json \
        --feature-store-layer nnmh_highcap6_240T_2024_with_reflexivity \
        --timeframe 240T
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)
from src.time_series_model.live.tree_gate import apply_gate_rules
from scripts.apply_archetype_gate import _read_feature_store_range


def extract_required_features(
    execution_archetypes_path: str,
) -> list[str]:
    """从execution_archetypes.yaml提取所有gate规则使用的特征"""
    arches = load_execution_archetypes_registry(execution_archetypes_path)
    features = set()

    for arch in arches.values():
        if not arch.gate_rules:
            continue
        rules = arch.gate_rules.get("rules", [])
        for rule in rules:
            feature_key = rule.get("key")
            if feature_key:
                features.add(feature_key)

    return sorted(list(features))


def load_features_from_featurestore(
    logs_df: pd.DataFrame,
    feature_store_root: str,
    feature_store_layer: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """从FeatureStore加载特征并merge到logs DataFrame"""
    symbols = sorted(logs_df["symbol"].astype(str).unique().tolist())

    feats = _read_feature_store_range(
        features_store_root=feature_store_root,
        layer=feature_store_layer,
        symbols=symbols,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
    )

    if feats.empty:
        raise ValueError(
            f"FeatureStore读取失败: layer={feature_store_layer}, "
            f"symbols={symbols}, timeframe={timeframe}"
        )

    # 处理timestamp列
    feats = feats.copy()
    if getattr(feats.index, "name", None) == "timestamp":
        if "timestamp" in feats.columns:
            feats = feats.reset_index(drop=True)
        else:
            feats = feats.reset_index()

    if "timestamp" not in feats.columns:
        if getattr(feats.index, "name", None) == "timestamp":
            feats = feats.reset_index()

    feats["symbol"] = feats["symbol"].astype(str)
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], errors="coerce")
    logs_df = logs_df.copy()
    logs_df["symbol"] = logs_df["symbol"].astype(str)
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"], errors="coerce")

    # Merge特征
    merged = logs_df.merge(
        feats, on=["symbol", "timestamp"], how="left", suffixes=("", "_feat")
    )

    # 处理重复列
    feat_suffix_cols = [c for c in merged.columns if c.endswith("_feat")]
    cols_to_drop = []
    cols_to_rename = {}
    for feat_col in feat_suffix_cols:
        original_col = feat_col[:-5]
        if original_col in merged.columns:
            cols_to_drop.append(feat_col)
        else:
            cols_to_rename[feat_col] = original_col

    if cols_to_drop:
        merged = merged.drop(columns=[c for c in cols_to_drop if c in merged.columns])
    if cols_to_rename:
        merged = merged.rename(
            columns={k: v for k, v in cols_to_rename.items() if k in merged.columns}
        )

    return merged


def set_minimal_thresholds(
    arches: Dict[str, Any],
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    将所有gate规则阈值调到最小（最宽松）

    对于value_*规则：
    - value_lt/value_lte: 设置为特征的最小值（或0.0）
    - value_gt/value_gte: 设置为特征的最大值（或1.0）

    对于quantile_*规则：
    - quantile_lt/quantile_lte: 设置为0.0
    - quantile_gt/quantile_gte: 设置为1.0
    """
    minimal_arches = deepcopy(arches)

    for arch_name, arch in minimal_arches.items():
        if not arch.gate_rules:
            continue

        rules = arch.gate_rules.get("rules", [])
        for rule in rules:
            rule_kind = rule.get("kind", "")
            feature_key = rule.get("key")

            if not feature_key or feature_key not in df.columns:
                continue

            # 获取特征值范围
            feature_values = df[feature_key].dropna()
            if len(feature_values) == 0:
                continue

            min_val = float(feature_values.min())
            max_val = float(feature_values.max())

            # 设置最小阈值
            if rule_kind in ("value_lt", "value_lte"):
                # 对于<规则，设置为最小值（几乎不拒绝）
                rule["threshold"] = min_val - 1.0  # 比最小值还小
            elif rule_kind in ("value_gt", "value_gte"):
                # 对于>规则，设置为最大值（几乎不拒绝）
                rule["threshold"] = max_val + 1.0  # 比最大值还大
            elif rule_kind in ("quantile_lt", "quantile_lte"):
                # 对于quantile <规则，设置为0.0
                rule["quantile"] = 0.0
            elif rule_kind in ("quantile_gt", "quantile_gte"):
                # 对于quantile >规则，设置为1.0
                rule["quantile"] = 1.0

    return minimal_arches


def calculate_kpi_metrics(
    df: pd.DataFrame,
    gate_ok: pd.Series,
    return_col: str = "ret_mean",
) -> Dict[str, float]:
    """计算KPI指标"""
    allowed_df = df[gate_ok].copy()

    if len(allowed_df) == 0:
        return {
            "trade_rate": 0.0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
        }

    trade_rate = float(gate_ok.sum() / len(df))

    if return_col not in allowed_df.columns:
        return {
            "trade_rate": trade_rate,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": len(allowed_df),
        }

    returns = allowed_df[return_col].dropna()

    if len(returns) == 0:
        return {
            "trade_rate": trade_rate,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": len(allowed_df),
        }

    win_rate = float((returns > 0).mean())
    avg_return = float(returns.mean())

    ret_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe_ratio = float(avg_return / ret_std) if ret_std > 0 else 0.0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(abs(drawdown.min())) if len(drawdown) > 0 else 0.0

    return {
        "trade_rate": trade_rate,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "total_trades": len(allowed_df),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将所有gate规则阈值调到最小并测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--logs",
        required=True,
        help="logs文件（parquet）",
    )
    parser.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="execution_archetypes.yaml路径",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出JSON文件",
    )
    parser.add_argument(
        "--feature-store-root",
        default="feature_store",
        help="FeatureStore根目录",
    )
    parser.add_argument(
        "--feature-store-layer",
        default=None,
        help="FeatureStore layer名称",
    )
    parser.add_argument(
        "--timeframe",
        default="240T",
        help="时间框架",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="开始日期",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="结束日期",
    )

    args = parser.parse_args()

    # 读取数据
    print("📊 读取数据...")
    df = pd.read_parquet(args.logs)
    print(f"✅ 读取数据: {len(df)} 行")

    # 加载特征
    if args.feature_store_layer:
        print("📊 从FeatureStore加载特征...")
        required_features = extract_required_features(args.execution_archetypes)
        available_features = set(df.columns)
        missing_features = [f for f in required_features if f not in available_features]

        if missing_features:
            print(f"⚠️  缺少 {len(missing_features)} 个特征，从FeatureStore加载...")
            df = load_features_from_featurestore(
                df,
                args.feature_store_root,
                args.feature_store_layer,
                args.timeframe,
                args.start_date,
                args.end_date,
            )
            print(f"✅ 特征加载完成，DataFrame现在有 {len(df.columns)} 列")

    # 加载archetypes
    print("📊 加载archetypes配置...")
    arches = load_execution_archetypes_registry(args.execution_archetypes)
    arches = {
        k: v for k, v in arches.items() if k != "VolMeanCompressionExpansionReversion"
    }

    # 设置最小阈值
    print("\n🔧 将所有gate规则阈值调到最小...")
    minimal_arches = set_minimal_thresholds(arches, df)

    # 应用最小阈值规则
    print("📊 应用最小阈值规则...")
    gate_ok = pd.Series(False, index=df.index)

    for idx, row in df.iterrows():
        features = row.to_dict()

        for arch_name, arch in minimal_arches.items():
            if not arch.gate_rules:
                gate_ok.loc[idx] = True
                break

            ok, _ = apply_gate_rules(
                gate_rules=arch.gate_rules,
                features=features,
                quantiles=None,
            )

            if ok:
                gate_ok.loc[idx] = True
                break

    # 计算KPI
    print("\n📊 计算KPI指标...")
    kpi = calculate_kpi_metrics(df, gate_ok)

    print(f"\n✅ 最小阈值测试完成:")
    print(f"   交易率: {kpi['trade_rate']:.4f}")
    print(f"   总交易数: {kpi['total_trades']}")
    print(f"   胜率: {kpi['win_rate']:.4f}")
    print(f"   Sharpe: {kpi['sharpe_ratio']:.4f}")

    # 保存结果
    result = {
        "minimal_thresholds": True,
        "kpi": kpi,
        "total_samples": len(df),
        "passed_samples": int(gate_ok.sum()),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 结果已保存: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
