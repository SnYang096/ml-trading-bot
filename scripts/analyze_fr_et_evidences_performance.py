#!/usr/bin/env python3
"""
分析FR/ET的evidences单独使用时的表现，以及加上gate后的表现

分析四种情况：
- 情况A: 所有数据，只用FR/ET的evidences（不经过regime和gate）
- 情况B: 所有数据，FR/ET的evidences + gate rules
- 情况C: MEAN_REGIME数据，只用FR/ET的evidences
- 情况D: MEAN_REGIME数据，FR/ET的evidences + gate rules
"""

import pandas as pd
import numpy as np
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_store import FeatureStore, FeatureStoreSpec

from src.time_series_model.core.constitution.execution_evidence import (
    compute_execution_evidence,
)
from src.time_series_model.live.tree_gate import apply_gate_rules


def calculate_sharpe(returns: pd.Series) -> float:
    """计算Sharpe比率（简化版，假设无风险利率为0）"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0


def compute_quantiles_from_data(
    df: pd.DataFrame, symbol_col: str = "symbol"
) -> Dict[str, Any]:
    """从数据中计算quantiles"""
    quantiles = {}
    symbols = df[symbol_col].unique() if symbol_col in df.columns else ["ALL"]

    for symbol in symbols:
        symbol_df = df[df[symbol_col] == symbol] if symbol_col in df.columns else df
        quantiles[symbol] = {}

        # 计算所有数值列的quantiles
        numeric_cols = symbol_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ["ret_mean", "timestamp"]:
                continue
            values = symbol_df[col].dropna()
            if len(values) > 0:
                quantiles[symbol][col] = {}
                for q in [
                    0.1,
                    0.15,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.8,
                    0.9,
                    0.95,
                ]:
                    try:
                        quantiles[symbol][col][f"{q:.2f}"] = float(values.quantile(q))
                    except Exception:
                        pass

    return quantiles


def check_required_evidence(
    evidence_flags: Dict[str, bool],
    required_evidence: List[str],
) -> bool:
    """检查是否满足required_evidence"""
    if not required_evidence:
        return True
    return all(evidence_flags.get(ev, False) for ev in required_evidence)


def apply_evidence_filter(
    df: pd.DataFrame,
    evidence_rules: List[Dict[str, Any]],
    required_evidence: List[str],
    quantiles: Dict[str, Any] | None = None,
    symbol_col: str = "symbol",
    debug: bool = False,
) -> pd.Series:
    """
    应用evidence_rules过滤样本

    注意: 订单流特征一个都不能少。如果缺少vpin等关键特征，应该在调用此函数之前就报错退出。
    """
    mask = pd.Series(False, index=df.index)
    evidence_stats = {ev: 0 for ev in required_evidence}
    total_checked = 0

    for idx, row in df.iterrows():
        # 转换为features字典
        features = row.to_dict()
        symbol = features.get(symbol_col, "ALL")

        # 获取该symbol的quantiles
        symbol_quantiles = quantiles.get(symbol, {}) if quantiles else None

        # 计算evidence flags
        try:
            evidence_flags = compute_execution_evidence(
                features=features,
                rules=evidence_rules,
                quantiles=symbol_quantiles,
            )

            total_checked += 1

            # 统计每个evidence的通过情况
            for ev in required_evidence:
                if evidence_flags.get(ev, False):
                    evidence_stats[ev] += 1

            # 检查required_evidence
            if check_required_evidence(evidence_flags, required_evidence):
                mask.loc[idx] = True
        except Exception as e:
            # 如果计算失败，跳过该样本
            if debug:
                print(f"  Evidence计算失败: {e}")
            continue

    if debug:
        print(f"  检查了 {total_checked} 个样本")
        if total_checked > 0:
            for ev, count in evidence_stats.items():
                print(
                    f"    {ev}: {count}/{total_checked} ({count/total_checked*100:.1f}%)"
                )
        else:
            print(f"  ⚠️  没有样本通过初步检查")

    return mask


def apply_gate_filter(
    df: pd.DataFrame,
    gate_rules: Dict[str, Any],
    quantiles: Dict[str, Any] | None = None,
    symbol_col: str = "symbol",
) -> pd.Series:
    """应用gate_rules过滤样本"""
    mask = pd.Series(False, index=df.index)

    for idx, row in df.iterrows():
        # 转换为features字典
        features = row.to_dict()
        symbol = features.get(symbol_col, "ALL")

        # 获取该symbol的quantiles
        symbol_quantiles = quantiles.get(symbol, {}) if quantiles else None

        # 应用gate rules
        try:
            ok, reasons = apply_gate_rules(
                gate_rules=gate_rules,
                features=features,
                quantiles=symbol_quantiles,
            )
            if ok:
                mask.loc[idx] = True
        except Exception as e:
            # 如果计算失败，跳过该样本
            continue

    return mask


def analyze_scenario(
    df: pd.DataFrame,
    evidence_rules: List[Dict[str, Any]],
    required_evidence: List[str],
    gate_rules: Dict[str, Any] | None = None,
    quantiles: Dict[str, Any] | None = None,
    scenario_name: str = "",
    debug: bool = False,
) -> Dict[str, Any]:
    """分析一个场景"""
    print(f"\n分析场景: {scenario_name}")

    # 应用evidence filter
    evidence_mask = apply_evidence_filter(
        df, evidence_rules, required_evidence, quantiles, debug=debug
    )
    evidence_passed = df[evidence_mask]

    print(f"  通过evidences的样本数: {len(evidence_passed)}/{len(df)}")

    if len(evidence_passed) == 0:
        return {
            "scenario": scenario_name,
            "evidence_count": 0,
            "gate_count": 0,
            "mean_ret": 0.0,
            "win_rate": 0.0,
            "sharpe": 0.0,
        }

    # 如果不需要gate，直接返回evidence结果
    if gate_rules is None:
        if "ret_mean" not in evidence_passed.columns:
            return {
                "scenario": scenario_name,
                "evidence_count": len(evidence_passed),
                "gate_count": len(evidence_passed),
                "mean_ret": 0.0,
                "win_rate": 0.0,
                "sharpe": 0.0,
            }

        ret_mean = evidence_passed["ret_mean"]
        return {
            "scenario": scenario_name,
            "evidence_count": len(evidence_passed),
            "gate_count": len(evidence_passed),
            "mean_ret": float(ret_mean.mean()),
            "win_rate": float((ret_mean > 0).sum() / len(ret_mean)),
            "sharpe": float(calculate_sharpe(ret_mean)),
            "median_ret": float(ret_mean.median()),
            "std_ret": float(ret_mean.std()),
        }

    # 应用gate filter
    gate_mask = apply_gate_filter(evidence_passed, gate_rules, quantiles)
    gate_passed = evidence_passed[gate_mask]

    print(f"  通过gate的样本数: {len(gate_passed)}/{len(evidence_passed)}")

    if len(gate_passed) == 0:
        return {
            "scenario": scenario_name,
            "evidence_count": len(evidence_passed),
            "gate_count": 0,
            "mean_ret": 0.0,
            "win_rate": 0.0,
            "sharpe": 0.0,
        }

    if "ret_mean" not in gate_passed.columns:
        return {
            "scenario": scenario_name,
            "evidence_count": len(evidence_passed),
            "gate_count": len(gate_passed),
            "mean_ret": 0.0,
            "win_rate": 0.0,
            "sharpe": 0.0,
        }

    ret_mean = gate_passed["ret_mean"]
    return {
        "scenario": scenario_name,
        "evidence_count": len(evidence_passed),
        "gate_count": len(gate_passed),
        "mean_ret": float(ret_mean.mean()),
        "win_rate": float((ret_mean > 0).sum() / len(ret_mean)),
        "sharpe": float(calculate_sharpe(ret_mean)),
        "median_ret": float(ret_mean.median()),
        "std_ret": float(ret_mean.std()),
    }


def _read_feature_store_range(
    *,
    features_store_root: str,
    layer: str,
    symbols: List[str],
    timeframe: str,
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    """从FeatureStore读取特征"""
    store = FeatureStore(str(features_store_root))
    parts = []
    for sym in symbols:
        spec = FeatureStoreSpec(
            layer=str(layer), symbol=str(sym), timeframe=str(timeframe)
        )
        start_ts = pd.Timestamp(start) if start else pd.Timestamp("1970-01-01")
        end_ts = pd.Timestamp(end) if end else pd.Timestamp("2100-01-01")
        df_sym = store.read_range(spec, start=start_ts, end=end_ts)
        if df_sym.empty:
            print(f"⚠️  FeatureStore为空: symbol={sym}, layer={layer}")
            continue
        if "symbol" not in df_sym.columns:
            df_sym = df_sym.copy()
            df_sym["symbol"] = sym
        parts.append(df_sym)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, axis=0, ignore_index=False)
    if "timestamp" not in df.columns:
        if getattr(df.index, "name", None) == "timestamp":
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df.index, utc=False, errors="coerce")
    return df


def main():
    parser = argparse.ArgumentParser(description="分析FR/ET Evidences性能")
    parser.add_argument(
        "--logs", default="results/e2e_kpi/logs_3action_regime_optimized.parquet"
    )
    parser.add_argument("--gated", default=None, help="Gated文件路径（如果可用）")
    parser.add_argument("--feature-store-root", default="feature_store")
    parser.add_argument(
        "--feature-store-layer",
        default="nnmh_highcap6_240T_2024_202510_v2",
        help="FeatureStore layer name (default: nnmh_highcap6_240T_2024_202510_v2)",
    )
    parser.add_argument("--timeframe", default="240T")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    print("=" * 80)
    print("FR/ET Evidences性能分析")
    print("=" * 80)

    # 读取基础数据
    if args.gated and Path(args.gated).exists():
        print(f"使用gated文件: {args.gated}")
        df = pd.read_parquet(args.gated)
    elif Path(args.logs).exists():
        print(f"使用logs文件: {args.logs}")
        df = pd.read_parquet(args.logs)
    else:
        print(f"❌ 数据文件不存在: {args.logs}")
        return 1

    print(f"\n总样本数: {len(df)}")
    print(f"数据列数: {len(df.columns)}")

    # 检查缺少的特征
    required_features = ["vpin", "cvd_change_5", "cvd_change_5_normalized"]
    missing_features = [f for f in required_features if f not in df.columns]

    if missing_features:
        print(f"\n⚠️  缺少特征: {missing_features}")
        print("尝试从FeatureStore读取...")

        # 从FeatureStore读取特征
        symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else []
        if symbols:
            feats_df = _read_feature_store_range(
                features_store_root=args.feature_store_root,
                layer=args.feature_store_layer,
                symbols=symbols,
                timeframe=args.timeframe,
                start=args.start_date,
                end=args.end_date,
            )

            if not feats_df.empty:
                # 处理timestamp列
                if getattr(feats_df.index, "name", None) == "timestamp":
                    if "timestamp" not in feats_df.columns:
                        feats_df = feats_df.reset_index()
                    else:
                        # timestamp既是index又是column，删除index
                        feats_df = feats_df.reset_index(drop=True)

                # Merge特征
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                if "timestamp" in feats_df.columns:
                    feats_df["timestamp"] = pd.to_datetime(
                        feats_df["timestamp"], errors="coerce"
                    )

                df = df.merge(
                    feats_df,
                    on=["symbol", "timestamp"],
                    how="left",
                    suffixes=("", "_feat"),
                )
                print(f"✅ 从FeatureStore读取了 {len(feats_df.columns)} 个特征列")
            else:
                print("⚠️  FeatureStore读取失败，将使用现有特征")
        else:
            print("⚠️  无法确定symbols，跳过FeatureStore读取")

    # 检查关键订单流特征（严格要求，一个都不能少）
    required_orderflow_features = {
        "vpin": "VPIN特征（必需）",
        "cvd_change_5": "CVD变化特征（必需）",
        "cvd_change_5_normalized": "CVD归一化特征（必需）",
    }

    missing_features = []
    print(f"\n关键订单流特征检查（严格要求）:")
    for feat, desc in required_orderflow_features.items():
        exists = feat in df.columns
        print(f"  {feat}: {'✅' if exists else '❌'} {desc}")
        if not exists:
            missing_features.append(feat)

    if missing_features:
        print(f"\n❌ 错误: 缺少必需的订单流特征: {missing_features}")
        print(f"订单流特征一个都不能少！缺少这些特征无法进行有效的evidences分析。")
        print(f"\n解决方案:")
        print(f"1. 重新生成FeatureStore，确保包含所有订单流特征")
        print(f"2. 检查FeatureStore配置，确认包含vpin等特征的计算")
        print(f"3. 确保tick数据可用（vpin计算需要tick数据）")
        return 1

    # 检查其他相关特征
    sr_cols = [
        c
        for c in df.columns
        if "sr_" in c.lower() or "sqs" in c.lower() or "poc" in c.lower()
    ]
    print(f"\n其他相关特征:")
    print(f"  sr/sqs/poc相关: {len(sr_cols)} 个 ({sr_cols[:3] if sr_cols else []}...)")
    absorption_cols = [c for c in df.columns if "absorption" in c.lower()]
    print(
        f"  absorption相关: {len(absorption_cols)} 个 ({absorption_cols[:3] if absorption_cols else []}...)"
    )

    # 读取execution_archetypes.yaml
    config_file = Path("config/nnmultihead/execution_archetypes.yaml")
    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return 1

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # 提取FR和ET的配置
    fr_config = None
    et_config = None

    for regime_name, regime_data in config.get("regimes", {}).items():
        for arch_name, arch_data in regime_data.get("archetypes", {}).items():
            if arch_name == "FailureReversionFR":
                fr_config = arch_data
            elif arch_name == "ExhaustionTurnET":
                et_config = arch_data

    if not fr_config or not et_config:
        print("❌ 未找到FR或ET配置")
        return 1

    fr_evidence_rules = fr_config.get("evidence_rules", [])
    fr_required_evidence = fr_config.get("required_evidence", [])
    fr_gate_rules = fr_config.get("gate_rules", {})

    et_evidence_rules = et_config.get("evidence_rules", [])
    et_required_evidence = et_config.get("required_evidence", [])
    et_gate_rules = et_config.get("gate_rules", {})

    print(f"\nFR required_evidence: {fr_required_evidence}")
    print(f"ET required_evidence: {et_required_evidence}")

    # 计算quantiles
    print("\n计算quantiles...")
    quantiles = compute_quantiles_from_data(df)
    print(f"  计算了 {len(quantiles)} 个symbol的quantiles")

    # 分析FR的四种情况
    print("\n" + "=" * 80)
    print("FR (FailureReversion) 分析")
    print("=" * 80)

    fr_results = []

    # 情况A: 所有数据，只用evidences
    result_a = analyze_scenario(
        df,
        fr_evidence_rules,
        fr_required_evidence,
        gate_rules=None,
        quantiles=quantiles,
        scenario_name="A: 所有数据，只用FR evidences",
        debug=True,  # 启用调试
    )
    fr_results.append(result_a)

    # 情况B: 所有数据，evidences + gate
    result_b = analyze_scenario(
        df,
        fr_evidence_rules,
        fr_required_evidence,
        gate_rules=fr_gate_rules,
        quantiles=quantiles,
        scenario_name="B: 所有数据，FR evidences + gate",
    )
    fr_results.append(result_b)

    # 情况C: MEAN_REGIME数据，只用evidences
    mean_df = df[df["regime"] == "MEAN_REGIME"]
    print(f"\nMEAN_REGIME样本数: {len(mean_df)}")

    if len(mean_df) > 0:
        result_c = analyze_scenario(
            mean_df,
            fr_evidence_rules,
            fr_required_evidence,
            gate_rules=None,
            quantiles=quantiles,
            scenario_name="C: MEAN_REGIME数据，只用FR evidences",
        )
        fr_results.append(result_c)

        # 情况D: MEAN_REGIME数据，evidences + gate
        result_d = analyze_scenario(
            mean_df,
            fr_evidence_rules,
            fr_required_evidence,
            gate_rules=fr_gate_rules,
            quantiles=quantiles,
            scenario_name="D: MEAN_REGIME数据，FR evidences + gate",
        )
        fr_results.append(result_d)
    else:
        print("  ⚠️ 没有MEAN_REGIME数据，跳过情况C和D")

    # 分析ET的四种情况
    print("\n" + "=" * 80)
    print("ET (ExhaustionTurn) 分析")
    print("=" * 80)

    et_results = []

    # 情况A: 所有数据，只用evidences
    result_a = analyze_scenario(
        df,
        et_evidence_rules,
        et_required_evidence,
        gate_rules=None,
        quantiles=quantiles,
        scenario_name="A: 所有数据，只用ET evidences",
    )
    et_results.append(result_a)

    # 情况B: 所有数据，evidences + gate
    result_b = analyze_scenario(
        df,
        et_evidence_rules,
        et_required_evidence,
        gate_rules=et_gate_rules,
        quantiles=quantiles,
        scenario_name="B: 所有数据，ET evidences + gate",
    )
    et_results.append(result_b)

    # 情况C: MEAN_REGIME数据，只用evidences
    if len(mean_df) > 0:
        result_c = analyze_scenario(
            mean_df,
            et_evidence_rules,
            et_required_evidence,
            gate_rules=None,
            quantiles=quantiles,
            scenario_name="C: MEAN_REGIME数据，只用ET evidences",
        )
        et_results.append(result_c)

        # 情况D: MEAN_REGIME数据，evidences + gate
        result_d = analyze_scenario(
            mean_df,
            et_evidence_rules,
            et_required_evidence,
            gate_rules=et_gate_rules,
            quantiles=quantiles,
            scenario_name="D: MEAN_REGIME数据，ET evidences + gate",
        )
        et_results.append(result_d)

    # 汇总结果
    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)

    print("\nFR (FailureReversion):")
    print("| 场景 | 通过evidences | 通过gate | 平均ret_mean | 胜率 | Sharpe |")
    print("|------|---------------|----------|--------------|------|--------|")
    for result in fr_results:
        print(
            f"| {result['scenario']} | {result['evidence_count']} | {result['gate_count']} | {result['mean_ret']:.6f} | {result['win_rate']:.1%} | {result['sharpe']:.3f} |"
        )

    print("\nET (ExhaustionTurn):")
    print("| 场景 | 通过evidences | 通过gate | 平均ret_mean | 胜率 | Sharpe |")
    print("|------|---------------|----------|--------------|------|------|")
    for result in et_results:
        print(
            f"| {result['scenario']} | {result['evidence_count']} | {result['gate_count']} | {result['mean_ret']:.6f} | {result['win_rate']:.1%} | {result['sharpe']:.3f} |"
        )

    # 保存结果
    output_file = Path("results/fr_et_evidences_performance.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(
            {
                "fr_results": fr_results,
                "et_results": et_results,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n✅ 结果已保存到: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
