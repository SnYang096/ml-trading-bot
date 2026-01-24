#!/usr/bin/env python3
"""
运行Gate优化对比实验

运行以下实验：
1. 基线（当前gate规则）
2. 渐进式优化（三步）
3. 优先级优化（Hard-Gate System）
4. 多目标优化
5. 渐进式 + 优先级 + 多目标

每个实验：
- 运行优化
- 应用优化后的规则
- 计算KPI（Sharpe, Trade Rate, Win Rate等）
- 保存结果

使用方法:
    python scripts/run_gate_optimization_experiments.py \
        --gated-logs results/pipeline_with_reflexivity_2024_full/logs_execution_gated.parquet \
        --raw-logs results/pipeline_with_reflexivity_2024_full/logs_execution.parquet \
        --execution-archetypes config/nnmultihead/execution_archetypes.yaml \
        --output-dir results/gate_optimization_experiments \
        --feature-store-layer nnmh_highcap6_240T_2024_with_reflexivity \
        --timeframe 240T
"""

from __future__ import annotations

import argparse
import json
import subprocess
import yaml
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.optimize_gate_plateau import (
    _apply_single_rule_veto,
    _compute_robustness_score,
    BucketConfig,
    OptimizationConfig,
)
from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)
from src.time_series_model.live.tree_gate import apply_gate_rules
from scripts.apply_archetype_gate import _read_feature_store_range


def calculate_kpi_metrics(
    df: pd.DataFrame,
    gate_ok: pd.Series,
    return_col: str = "ret_mean",
) -> Dict[str, float]:
    """
    计算KPI指标

    Args:
        df: 原始DataFrame
        gate_ok: gate通过/不通过的布尔Series
        return_col: 收益列名

    Returns:
        KPI指标字典
    """
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

    # 计算交易率
    trade_rate = float(gate_ok.sum() / len(df))

    # 计算收益
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

    # 计算胜率
    win_rate = float((returns > 0).mean())

    # 计算平均收益
    avg_return = float(returns.mean())

    # 计算Sharpe比率
    ret_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe_ratio = float(avg_return / ret_std) if ret_std > 0 else 0.0

    # 计算最大回撤
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


def _run_cmd(cmd: list[str]) -> None:
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)


def _apply_threshold_overrides(
    execution_archetypes_path: str,
    optimization_results: Dict[str, Any],
    output_path: Path,
) -> None:
    with open(execution_archetypes_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rule_map: Dict[str, Dict[str, float]] = {}
    for arch_name, arch_results in optimization_results.items():
        if not isinstance(arch_results, list):
            continue
        rule_map[arch_name] = {}
        for rule_result in arch_results:
            rule_name = rule_result.get("rule_name", "")
            threshold = (
                rule_result.get("final_threshold")
                if rule_result.get("final_threshold") is not None
                else rule_result.get("recommended_threshold")
            )
            if rule_name and threshold is not None:
                rule_map[arch_name][rule_name] = float(threshold)

    regimes = data.get("regimes", {})
    updated = 0
    for regime_cfg in regimes.values():
        archetypes = regime_cfg.get("archetypes", {})
        for arch_name, arch_cfg in archetypes.items():
            rules = arch_cfg.get("gate_rules", {}).get("rules", [])
            if arch_name not in rule_map:
                continue
            overrides = rule_map[arch_name]
            for rule in rules:
                name = rule.get("name")
                if name in overrides:
                    if "threshold" in rule:
                        rule["threshold"] = overrides[name]
                    elif "quantile" in rule:
                        rule["quantile"] = overrides[name]
                    updated += 1

    output_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"✅ Updated {updated} rule thresholds -> {output_path}")


def _run_apply_tree_gate(
    *,
    logs_path: str,
    output_path: Path,
    execution_archetypes_path: str,
    feature_store_layer: Optional[str],
    feature_store_root: str,
    timeframe: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "apply_archetype_gate.py"),
        "--logs",
        str(logs_path),
        "--out",
        str(output_path),
        "--features-store-layer",
        str(feature_store_layer),
        "--features-store-root",
        str(feature_store_root),
        "--execution-archetypes",
        str(execution_archetypes_path),
        "--timeframe",
        str(timeframe),
    ]
    if start_date:
        cmd.extend(["--start-date", start_date])
    if end_date:
        cmd.extend(["--end-date", end_date])
    _run_cmd(cmd)


def _run_e2e_kpi(
    *,
    logs_path: Path,
    output_dir: Path,
    label: str,
) -> Dict[str, Any]:
    output_json = output_dir / f"e2e_kpi_{label}.json"
    output_md = output_dir / f"e2e_kpi_{label}.md"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "diagnose_e2e_kpi.py"),
        "--logs",
        str(logs_path),
        "--gate",
        str(logs_path),
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]
    _run_cmd(cmd)
    with open(output_json, "r", encoding="utf-8") as f:
        report = json.load(f)
    overall = report.get("overall", {})
    return {
        "trade_rate": float(overall.get("trade_rate", 0.0) or 0.0),
        "win_rate": float(overall.get("win_rate", 0.0) or 0.0),
        "avg_return": float(overall.get("ret_mean_e2e", 0.0) or 0.0),
        "sharpe_ratio": float(overall.get("sharpe_e2e", 0.0) or 0.0),
        "max_drawdown": 0.0,
        "total_trades": int(overall.get("trade_rows", 0) or 0),
        "e2e_report_json": str(output_json),
        "e2e_report_md": str(output_md),
    }


def apply_optimized_rules(
    df: pd.DataFrame,
    optimization_results: Dict[str, Any],
    arches: Dict[str, Any],
    quantiles: Optional[Dict[str, Any]] = None,
) -> pd.Series:
    """
    应用优化后的规则到DataFrame

    Args:
        df: 原始DataFrame
        optimization_results: 优化结果字典 {archetype: [rule_results]}
        arches: archetypes配置
        quantiles: 分位数字典（可选）

    Returns:
        gate_ok Series
    """
    result_series = pd.Series(False, index=df.index)

    # 构建规则映射：{archetype: {rule_name: threshold}}
    rule_map: Dict[str, Dict[str, float]] = {}

    for arch_name, arch_results in optimization_results.items():
        if not isinstance(arch_results, list):
            continue

        rule_map[arch_name] = {}
        for rule_result in arch_results:
            rule_name = rule_result.get("rule_name", "")
            threshold = rule_result.get("recommended_threshold") or rule_result.get(
                "final_threshold"
            )
            if rule_name and threshold is not None:
                rule_map[arch_name][rule_name] = threshold

    # 对每个样本应用规则
    for idx, row in df.iterrows():
        features = row.to_dict()

        # 尝试每个archetype
        for arch_name, arch in arches.items():
            if not arch.gate_rules:
                result_series.loc[idx] = True
                break

            # 应用优化后的规则
            gate_rules = arch.gate_rules.copy()
            rules = gate_rules.get("rules", [])

            # 更新规则阈值
            if arch_name in rule_map:
                for rule in rules:
                    rule_name = rule.get("name", "")
                    if rule_name in rule_map[arch_name]:
                        threshold = rule_map[arch_name][rule_name]
                        if "threshold" in rule:
                            rule["threshold"] = threshold
                        elif "quantile" in rule:
                            rule["quantile"] = threshold

            # 获取该样本的分位数
            sym_quantiles = None
            if quantiles:
                symbol = str(row.get("symbol", ""))
                if isinstance(quantiles, dict):
                    sym_quantiles = quantiles.get(symbol) or quantiles

            # 应用gate规则
            ok, reasons = apply_gate_rules(
                gate_rules=gate_rules,
                features=features,
                quantiles=sym_quantiles,
            )

            if ok:
                result_series.loc[idx] = True
                break

    return result_series


def _compute_archetype_score(row: pd.Series, arch_name: str) -> float:
    """
    Compute archetype selection score based on mfe, mae, ttm, and archetype-specific semantic score.

    This matches the logic in scripts/apply_archetype_gate.py
    """
    mfe = float(row.get("head_mfe_atr", 0.0) or 0.0)
    mae = float(row.get("head_mae_atr", 0.0) or 0.0)
    ttm = float(row.get("head_t_to_mfe", 0.0) or 0.0)

    eps = 1e-6
    if mae < eps:
        return 0.0

    eff = mfe / (mae + eps)
    time_penalty = 1.0 / (1.0 + ttm / 10.0)
    base_score = eff * mfe * time_penalty

    arch_upper = str(arch_name).upper()
    semantic_col = None
    if "TC" in arch_upper or "TRENDCONTINUATION" in arch_upper:
        semantic_col = "tc_semantic_score"
    elif "TE" in arch_upper or "TRENDEXPANSION" in arch_upper:
        semantic_col = "te_semantic_score"
    elif "FR" in arch_upper or "FAILUREREVERSION" in arch_upper:
        semantic_col = "fr_semantic_score"
    elif "ET" in arch_upper or "EXHAUSTIONTURN" in arch_upper:
        semantic_col = "et_semantic_score"

    semantic_bonus = 1.0
    if semantic_col and semantic_col in row.index:
        semantic_val = float(row.get(semantic_col, 0.0) or 0.0)
        semantic_bonus = 0.5 + semantic_val

    return base_score * semantic_bonus


def load_features_if_needed(
    df: pd.DataFrame,
    feature_store_root: str,
    feature_store_layer: Optional[str],
    timeframe: str,
    start_date: Optional[str],
    end_date: Optional[str],
    execution_archetypes_path: str,
) -> pd.DataFrame:
    """如果需要，从FeatureStore加载特征"""
    if not feature_store_layer:
        return df

    # 提取所需特征
    arches = load_execution_archetypes_registry(execution_archetypes_path)
    required_features = set()
    for arch in arches.values():
        if not arch.gate_rules:
            continue
        rules = arch.gate_rules.get("rules", [])
        for rule in rules:
            feature_key = rule.get("key")
            if feature_key:
                required_features.add(feature_key)

    # 检查缺失特征
    available_features = set(df.columns)
    missing_features = [f for f in required_features if f not in available_features]

    if not missing_features:
        return df

    print(f"⚠️  logs文件缺少 {len(missing_features)} 个特征，从FeatureStore加载...")

    # 从FeatureStore加载
    symbols = sorted(df["symbol"].astype(str).unique().tolist())
    feats = _read_feature_store_range(
        features_store_root=feature_store_root,
        layer=feature_store_layer,
        symbols=symbols,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
    )

    if feats.empty:
        print(f"⚠️  FeatureStore读取失败，使用原始数据")
        return df

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
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Merge特征
    merged = df.merge(
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

    print(f"✅ 特征加载完成，DataFrame现在有 {len(merged.columns)} 列")
    return merged


def run_baseline_experiment(
    df: pd.DataFrame,
    arches: Dict[str, Any],
    logs_path: str,
    output_dir: Path,
    execution_archetypes_path: str,
    feature_store_layer: Optional[str],
    feature_store_root: str,
    timeframe: str,
    start_date: Optional[str],
    end_date: Optional[str],
    quantiles: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """运行基线实验（使用当前gate规则）"""
    print("\n" + "=" * 60)
    print("实验1: 基线（当前gate规则）")
    print("=" * 60)

    gated_path = output_dir / "baseline_gated.parquet"
    _run_apply_tree_gate(
        logs_path=logs_path,
        output_path=gated_path,
        execution_archetypes_path=execution_archetypes_path,
        feature_store_layer=feature_store_layer,
        feature_store_root=feature_store_root,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )
    kpi = _run_e2e_kpi(logs_path=gated_path, output_dir=output_dir, label="baseline")

    print(f"✅ 基线实验完成")
    print(f"   交易率: {kpi['trade_rate']:.4f}")
    print(f"   胜率: {kpi['win_rate']:.4f}")
    print(f"   Sharpe: {kpi['sharpe_ratio']:.4f}")

    return {
        "experiment": "baseline",
        "optimization_results": {},
        "kpi": kpi,
        "gate_logs": str(gated_path),
    }


def run_progressive_experiment(
    df: pd.DataFrame,
    arches: Dict[str, Any],
    logs_path: str,
    output_dir: Path,
    execution_archetypes_path: str,
    feature_store_layer: Optional[str] = None,
    timeframe: str = "240T",
    feature_store_root: str = "feature_store",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """运行渐进式优化实验"""
    print("\n" + "=" * 60)
    print("实验2: 渐进式优化（三步）")
    print("=" * 60)

    # 运行渐进式优化
    output_file = output_dir / "progressive_optimization.json"

    # 注意：optimize_gate_plateau_progressive.py 已经定义了 --execution-archetypes
    # 我们需要确保不重复传递
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "optimize_gate_plateau_progressive.py"),
        "--gated-logs",
        str(output_dir / "temp_gated.parquet"),
        "--raw-logs",
        str(output_dir / "temp_raw.parquet"),
        "--output",
        str(output_file),
        "--target-trades",
        "200",
        "--tighten-step",
        "0.05",
        "--min-trade-rate",
        "0.02",
        "--min-trades-per-bucket",
        "5",
        "--global-trade-budget",
        "0.12",
    ]

    # 添加execution-archetypes（如果脚本需要）
    # 检查脚本是否已经定义了该参数
    if execution_archetypes_path:
        cmd.extend(["--execution-archetypes", execution_archetypes_path])

    if feature_store_layer:
        cmd.extend(
            [
                "--feature-store-root",
                feature_store_root,
                "--feature-store-layer",
                feature_store_layer,
                "--timeframe",
                timeframe,
            ]
        )
        if start_date:
            cmd.extend(["--start-date", start_date])
        if end_date:
            cmd.extend(["--end-date", end_date])

    # 保存临时文件
    df.to_parquet(output_dir / "temp_raw.parquet")
    df.to_parquet(output_dir / "temp_gated.parquet")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ 渐进式优化失败: {result.stderr}")
        return {"experiment": "progressive", "error": result.stderr}

    # 加载优化结果
    if not output_file.exists():
        print(f"❌ 优化结果文件不存在: {output_file}")
        return {"experiment": "progressive", "error": "Result file not found"}

    with open(output_file, "r", encoding="utf-8") as f:
        optimization_results = json.load(f)

    progressive_yaml = output_dir / "execution_archetypes_progressive.yaml"
    _apply_threshold_overrides(
        execution_archetypes_path=execution_archetypes_path,
        optimization_results=optimization_results,
        output_path=progressive_yaml,
    )
    gated_path = output_dir / "progressive_gated.parquet"
    _run_apply_tree_gate(
        logs_path=logs_path,
        output_path=gated_path,
        execution_archetypes_path=str(progressive_yaml),
        feature_store_layer=feature_store_layer,
        feature_store_root=feature_store_root,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )
    kpi = _run_e2e_kpi(logs_path=gated_path, output_dir=output_dir, label="progressive")

    print(f"✅ 渐进式优化实验完成")
    print(f"   交易率: {kpi['trade_rate']:.4f}")
    print(f"   胜率: {kpi['win_rate']:.4f}")
    print(f"   Sharpe: {kpi['sharpe_ratio']:.4f}")

    return {
        "experiment": "progressive",
        "optimization_results": optimization_results,
        "kpi": kpi,
        "gate_logs": str(gated_path),
        "execution_archetypes_override": str(progressive_yaml),
    }


def run_hard_gate_experiment(
    df: pd.DataFrame,
    arches: Dict[str, Any],
    logs_path: str,
    output_dir: Path,
    execution_archetypes_path: str,
    feature_store_layer: Optional[str] = None,
    timeframe: str = "240T",
    feature_store_root: str = "feature_store",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """运行Hard-Gate System优化实验"""
    print("\n" + "=" * 60)
    print("实验3: 优先级优化（Hard-Gate System）")
    print("=" * 60)

    # 运行Hard-Gate优化
    output_file = output_dir / "hard_gate_optimization.json"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "optimize_gate_plateau_hard_gate.py"),
        "--gated-logs",
        str(output_dir / "temp_gated.parquet"),
        "--raw-logs",
        str(output_dir / "temp_raw.parquet"),
        "--execution-archetypes",
        execution_archetypes_path,
        "--output",
        str(output_file),
        "--min-trade-rate",
        "0.001",
        "--min-trades-per-bucket",
        "3",
        "--min-sharpe-threshold",
        "0.05",
        "--threshold-step",
        "0.05",
    ]

    if feature_store_layer:
        cmd.extend(
            [
                "--feature-store-root",
                feature_store_root,
                "--feature-store-layer",
                feature_store_layer,
                "--timeframe",
                timeframe,
            ]
        )
        if start_date:
            cmd.extend(["--start-date", start_date])
        if end_date:
            cmd.extend(["--end-date", end_date])

    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Hard-Gate优化失败: {result.stderr}")
        return {"experiment": "hard_gate", "error": result.stderr}

    # 加载优化结果
    if not output_file.exists():
        print(f"❌ 优化结果文件不存在: {output_file}")
        return {"experiment": "hard_gate", "error": "Result file not found"}

    with open(output_file, "r", encoding="utf-8") as f:
        optimization_results = json.load(f)

    hard_gate_yaml = output_dir / "execution_archetypes_hard_gate.yaml"
    _apply_threshold_overrides(
        execution_archetypes_path=execution_archetypes_path,
        optimization_results=optimization_results,
        output_path=hard_gate_yaml,
    )
    gated_path = output_dir / "hard_gate_gated.parquet"
    _run_apply_tree_gate(
        logs_path=logs_path,
        output_path=gated_path,
        execution_archetypes_path=str(hard_gate_yaml),
        feature_store_layer=feature_store_layer,
        feature_store_root=feature_store_root,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )
    kpi = _run_e2e_kpi(logs_path=gated_path, output_dir=output_dir, label="hard_gate")

    print(f"✅ Hard-Gate优化实验完成")
    print(f"   交易率: {kpi['trade_rate']:.4f}")
    print(f"   胜率: {kpi['win_rate']:.4f}")
    print(f"   Sharpe: {kpi['sharpe_ratio']:.4f}")

    return {
        "experiment": "hard_gate",
        "optimization_results": optimization_results,
        "kpi": kpi,
        "gate_logs": str(gated_path),
        "execution_archetypes_override": str(hard_gate_yaml),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="运行Gate优化对比实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gated-logs",
        required=True,
        help="Gated logs文件（parquet）",
    )
    parser.add_argument(
        "--raw-logs",
        required=True,
        help="原始logs文件（parquet）",
    )
    parser.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="execution_archetypes.yaml路径",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="输出目录",
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
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["baseline", "progressive", "hard_gate", "all"],
        default=["all"],
        help="要运行的实验（默认：all）",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取数据
    print("📊 读取数据...")
    df_raw = pd.read_parquet(args.raw_logs)
    print(f"✅ 读取原始数据: {len(df_raw)} 行")

    # 加载特征（如果需要）
    if args.feature_store_layer:
        df_raw = load_features_if_needed(
            df_raw,
            args.feature_store_root,
            args.feature_store_layer,
            args.timeframe,
            args.start_date,
            args.end_date,
            args.execution_archetypes,
        )

    # 加载archetypes
    arches = load_execution_archetypes_registry(args.execution_archetypes)
    arches = {
        k: v for k, v in arches.items() if k != "VolMeanCompressionExpansionReversion"
    }

    # 确定要运行的实验
    experiments_to_run = []
    if "all" in args.experiments:
        experiments_to_run = ["baseline", "progressive", "hard_gate"]
    else:
        experiments_to_run = args.experiments

    # 运行实验
    all_results = {}

    if "baseline" in experiments_to_run:
        result = run_baseline_experiment(
            df_raw,
            arches,
            args.raw_logs,
            output_dir,
            args.execution_archetypes,
            args.feature_store_layer,
            args.feature_store_root,
            args.timeframe,
            args.start_date,
            args.end_date,
        )
        all_results["baseline"] = result

    if "progressive" in experiments_to_run:
        result = run_progressive_experiment(
            df_raw,
            arches,
            args.raw_logs,
            output_dir,
            args.execution_archetypes,
            args.feature_store_layer,
            args.timeframe,
            args.feature_store_root,
            args.start_date,
            args.end_date,
        )
        all_results["progressive"] = result

    if "hard_gate" in experiments_to_run:
        result = run_hard_gate_experiment(
            df_raw,
            arches,
            args.raw_logs,
            output_dir,
            args.execution_archetypes,
            args.feature_store_layer,
            args.timeframe,
            args.feature_store_root,
            args.start_date,
            args.end_date,
        )
        all_results["hard_gate"] = result

    # 保存所有结果
    results_file = output_dir / "all_experiments_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 所有实验完成，结果保存到: {results_file}")

    # 清理临时文件
    for temp_file in ["temp_gated.parquet", "temp_raw.parquet"]:
        temp_path = output_dir / temp_file
        if temp_path.exists():
            temp_path.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
