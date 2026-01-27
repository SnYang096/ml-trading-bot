#!/usr/bin/env python3
"""
使用 Optuna 优化 FR (FailureReversionFR) 的 gate 规则参数

优化目标：最大化 Sharpe 比率
优化参数：FR gate 规则的关键阈值
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import optuna
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.apply_archetype_gate import (
    _read_feature_store_range,
    load_evidence_quantiles,
)
from src.time_series_model.live.tree_gate import apply_gate_rules
from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)


def calculate_sharpe(df: pd.DataFrame, return_col: str = "ret_mean") -> float:
    """计算 Sharpe 比率"""
    if return_col not in df.columns or len(df) == 0:
        return -1000.0

    returns = df[return_col].dropna()
    if len(returns) == 0:
        return -1000.0

    ret_mean = returns.mean()
    ret_std = returns.std(ddof=1)

    if ret_std == 0:
        return 0.0

    sharpe = ret_mean / ret_std
    return float(sharpe)


def apply_fr_gate_with_params(
    df: pd.DataFrame,
    arches: Dict[str, Any],
    params: Dict[str, float],
    evidence_quantiles: Optional[Dict[str, Any]] = None,
) -> pd.Series:
    """
    使用给定的参数应用 FR gate 规则

    Args:
        df: 原始数据
        arches: archetype 配置
        params: 优化参数（规则阈值）
        evidence_quantiles: evidence quantiles

    Returns:
        gate_ok Series
    """
    fr_arch = arches.get("FailureReversionFR")
    if not fr_arch:
        return pd.Series(False, index=df.index)

    # ExecutionArchetype 是 dataclass，使用属性访问
    gate_rules = fr_arch.gate_rules if hasattr(fr_arch, "gate_rules") else {}
    rules = gate_rules.get("rules", []) if isinstance(gate_rules, dict) else []

    # 更新规则阈值
    updated_rules = []
    for rule in rules:
        rule_name = rule.get("name", "")
        rule_copy = rule.copy()

        # 如果参数中有这个规则的阈值，使用它
        if rule_name in params:
            if rule_copy.get("kind") in ["quantile_lt", "quantile_gt"]:
                rule_copy["quantile"] = params[rule_name]
            elif rule_copy.get("kind") in [
                "value_lt",
                "value_gt",
                "value_lte",
                "value_gte",
            ]:
                rule_copy["threshold"] = params[rule_name]

        updated_rules.append(rule_copy)

    # 创建临时 gate_rules 配置
    temp_gate_rules = {
        **gate_rules,
        "rules": updated_rules,
    }

    # 应用 gate 规则 - apply_gate_rules 接受 gate_rules dict
    try:
        # 对每一行应用 gate 规则
        gate_ok_list = []
        for idx, row in df.iterrows():
            features = row.to_dict()
            ok, _ = apply_gate_rules(
                gate_rules=temp_gate_rules,
                features=features,
                quantiles=evidence_quantiles,
            )
            gate_ok_list.append(ok)

        return pd.Series(gate_ok_list, index=df.index)
    except Exception as e:
        print(f"    ⚠️ Error applying gate: {e}")
        return pd.Series(False, index=df.index)


def objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    arches: Dict[str, Any],
    evidence_quantiles: Optional[Dict[str, Any]],
    min_trade_rate: float,
    min_trades: int,
) -> float:
    """
    Optuna 目标函数：最大化 Sharpe 比率

    优化参数：
    - deny_if 规则的关键阈值
    - allow_if 规则的阈值（如果需要）
    """
    # 采样 deny_if 规则的关键阈值
    params = {}

    # 关键 deny_if 规则（这些对 Sharpe 影响最大）
    key_deny_rules = {
        "fr_path_efficiency_too_high": (0.0, 0.8),  # quantile_gt
        "fr_price_dir_consistency_too_high": (0.0, 0.8),  # quantile_gt
        "fr_deviation_too_low": (0.0, 0.8),  # quantile_lt
        "fr_not_mean_regime_path_length_too_low": (0.0, 0.8),  # quantile_lt
        "fr_not_mean_regime_atr_percentile_too_low": (0.0, 0.8),  # quantile_lt
        "fr_not_mean_regime_jump_risk_too_high": (0.0, 0.8),  # quantile_gt
        "fr_volume_too_low": (0.0, 0.5),  # quantile_lt
        "fr_bb_width_too_low": (0.0, 0.5),  # quantile_lt
        "fr_mean_adx_too_high": (0.5, 1.0),  # quantile_gt
        "fr_mean_sr_too_far": (0.5, 1.0),  # quantile_gt
        "fr_sqs_too_low": (0.0, 0.5),  # quantile_lt
        "fr_quality_too_low": (0.0, 0.5),  # quantile_lt
        "fr_score_too_low": (0.0, 0.5),  # quantile_lt
    }

    # 采样参数
    for rule_name, (low, high) in key_deny_rules.items():
        params[rule_name] = trial.suggest_float(rule_name, low, high, step=0.02)

    # 可选：优化 allow_if 规则阈值
    key_allow_rules = {
        "fr_divergence": (0.0, 0.8),  # quantile_lt
        "fr_absorption": (0.0, 0.8),  # quantile_gt
        "fr_near_sr": (0.0, 0.8),  # quantile_lt
    }

    for rule_name, (low, high) in key_allow_rules.items():
        params[rule_name] = trial.suggest_float(rule_name, low, high, step=0.02)

    # 应用 gate 规则
    gate_ok = apply_fr_gate_with_params(df, arches, params, evidence_quantiles)

    # 计算指标
    trade_rate = gate_ok.sum() / len(df) if len(df) > 0 else 0.0
    n_trades = gate_ok.sum()

    # 约束检查
    if trade_rate < min_trade_rate:
        trial.set_user_attr("trade_rate", trade_rate)
        trial.set_user_attr("n_trades", int(n_trades))
        return -1000.0  # 惩罚：交易率太低

    if n_trades < min_trades:
        trial.set_user_attr("trade_rate", trade_rate)
        trial.set_user_attr("n_trades", int(n_trades))
        return -1000.0  # 惩罚：交易数太少

    # 计算 Sharpe
    allowed_df = df[gate_ok].copy()
    sharpe = calculate_sharpe(allowed_df, "ret_mean")

    # 记录中间值
    trial.set_user_attr("trade_rate", trade_rate)
    trial.set_user_attr("n_trades", int(n_trades))
    trial.set_user_attr("sharpe", sharpe)

    return sharpe


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 Optuna 优化 FR gate 规则参数")
    parser.add_argument(
        "--raw-logs",
        required=True,
        help="原始 logs 文件（parquet）",
    )
    parser.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="execution_archetypes.yaml 路径",
    )
    parser.add_argument(
        "--feature-store-layer",
        required=True,
        help="FeatureStore layer 名称",
    )
    parser.add_argument(
        "--feature-store-root",
        default="feature_store",
        help="FeatureStore 根目录",
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
    parser.add_argument(
        "--evidence-quantiles",
        default=None,
        help="evidence_quantiles.json 路径（可选）",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Optuna trial 数量",
    )
    parser.add_argument(
        "--min-trade-rate",
        type=float,
        default=0.01,
        help="最小交易率",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=50,
        help="最小交易数",
    )
    parser.add_argument(
        "--output",
        default="results/fr_optuna_optimization.json",
        help="输出 JSON 文件",
    )
    parser.add_argument(
        "--study-name",
        default="fr_gate_optimization",
        help="Optuna study 名称",
    )

    args = parser.parse_args()

    # 读取数据
    print("📊 读取原始数据...")
    df_raw = pd.read_parquet(args.raw_logs)
    print(f"✅ 读取 {len(df_raw)} 行数据")

    # 加载特征
    print("📊 从 FeatureStore 加载特征...")
    symbols = sorted(df_raw["symbol"].astype(str).unique().tolist())
    feats = _read_feature_store_range(
        features_store_root=args.feature_store_root,
        layer=args.feature_store_layer,
        symbols=symbols,
        timeframe=args.timeframe,
        start=args.start_date,
        end=args.end_date,
    )

    # 合并特征 - 使用 apply_archetype_gate 的逻辑
    # 确保 timestamp 在 columns 中
    if df_raw.index.name == "timestamp":
        df_raw = df_raw.reset_index()
    elif "timestamp" not in df_raw.columns and "timestamp" in df_raw.index.names:
        df_raw = df_raw.reset_index()

    if feats.index.name == "timestamp":
        # 如果 timestamp 已经在 columns 中，drop=True 避免重复
        if "timestamp" in feats.columns:
            feats = feats.reset_index(drop=True)
        else:
            feats = feats.reset_index()
    elif "timestamp" not in feats.columns and "timestamp" in feats.index.names:
        feats = feats.reset_index()

    # 确定 merge columns
    merge_cols = ["symbol"]
    if "timestamp" in df_raw.columns and "timestamp" in feats.columns:
        merge_cols.append("timestamp")
    elif "index" in df_raw.columns and "index" in feats.columns:
        merge_cols.append("index")

    df = df_raw.merge(feats, on=merge_cols, how="left", suffixes=("", "_feat"))
    print(f"✅ 合并后 DataFrame 有 {len(df.columns)} 列")

    # 加载 archetype 配置
    print("📊 加载 archetype 配置...")
    arches = load_execution_archetypes_registry(args.execution_archetypes)
    fr_arch = arches.get("FailureReversionFR")
    if not fr_arch:
        print("❌ 未找到 FailureReversionFR archetype")
        return 1

    # 加载 evidence quantiles
    evidence_quantiles = None
    if args.evidence_quantiles and Path(args.evidence_quantiles).exists():
        print("📊 加载 evidence quantiles...")
        evidence_quantiles = load_evidence_quantiles(args.evidence_quantiles)

    # 创建 Optuna study
    print(f"\n🔍 开始 Optuna 优化 ({args.n_trials} trials)...")
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    # 优化
    study.optimize(
        lambda trial: objective(
            trial,
            df,
            arches,
            evidence_quantiles,
            args.min_trade_rate,
            args.min_trades,
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # 保存结果
    print("\n📊 优化完成，保存结果...")
    best_trial = study.best_trial

    result = {
        "best_params": best_trial.params,
        "best_value": best_trial.value,
        "best_trial_number": best_trial.number,
        "best_trial_attrs": best_trial.user_attrs,
        "n_trials": len(study.trials),
        "study_name": args.study_name,
    }

    # 添加所有 trial 的摘要
    trials_summary = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trials_summary.append(
                {
                    "trial_number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "attrs": trial.user_attrs,
                }
            )

    result["trials_summary"] = sorted(
        trials_summary, key=lambda x: x["value"], reverse=True
    )[
        :20
    ]  # 只保存前20个最好的

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ 结果已保存到: {output_path}")
    print(f"\n📈 最佳结果:")
    print(f"   Sharpe: {best_trial.value:.4f}")
    print(f"   Trade Rate: {best_trial.user_attrs.get('trade_rate', 'N/A'):.4f}")
    print(f"   N Trades: {best_trial.user_attrs.get('n_trades', 'N/A')}")
    print(f"\n📋 最佳参数:")
    for key, value in best_trial.params.items():
        print(f"   {key}: {value:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
