#!/usr/bin/env python3
"""
ET优化分析脚本

分析三个问题：
1. volume_profile特征是否可以计算和恢复
2. ET止损止盈配置是否合理
3. 如何优化ET_REGIME的Sharpe或扩大测试范围
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store import FeatureStore, FeatureStoreSpec


def analyze_volume_profile_features(
    feature_store_root: str,
    layer: str,
    symbols: List[str],
    timeframe: str,
) -> Dict[str, Any]:
    """分析volume_profile特征的可用性"""
    print("=" * 80)
    print("1. Volume Profile特征分析")
    print("=" * 80)

    store = FeatureStore(feature_store_root)
    results = {
        "vpvr_features_found": [],
        "volume_profile_features_found": [],
        "all_vp_features": [],
    }

    for sym in symbols[:3]:  # 测试前3个symbol
        spec = FeatureStoreSpec(layer=layer, symbol=str(sym), timeframe=timeframe)
        try:
            feat_df = store.read_range(
                spec, start=pd.Timestamp("2025-07-01"), end=pd.Timestamp("2025-07-10")
            )
            if not feat_df.empty:
                # 查找vpvr相关特征
                vpvr_cols = [col for col in feat_df.columns if "vpvr" in col.lower()]
                vp_cols = [
                    col
                    for col in feat_df.columns
                    if "vp_" in col.lower() and "vpin" not in col.lower()
                ]
                all_vp = [
                    col
                    for col in feat_df.columns
                    if ("vp" in col.lower() or "volume_profile" in col.lower())
                    and "vpin" not in col.lower()
                ]

                if vpvr_cols:
                    results["vpvr_features_found"].extend(vpvr_cols)
                if vp_cols:
                    results["volume_profile_features_found"].extend(vp_cols)
                if all_vp:
                    results["all_vp_features"].extend(all_vp)
        except Exception as e:
            print(f"  警告: 读取{sym}失败: {e}")

    # 去重
    results["vpvr_features_found"] = sorted(set(results["vpvr_features_found"]))
    results["volume_profile_features_found"] = sorted(
        set(results["volume_profile_features_found"])
    )
    results["all_vp_features"] = sorted(set(results["all_vp_features"]))

    print(f"\n找到的特征:")
    print(f"  vpvr特征: {len(results['vpvr_features_found'])} 个")
    if results["vpvr_features_found"]:
        for col in results["vpvr_features_found"][:10]:
            print(f"    - {col}")

    print(f"  volume_profile特征: {len(results['volume_profile_features_found'])} 个")
    if results["volume_profile_features_found"]:
        for col in results["volume_profile_features_found"][:10]:
            print(f"    - {col}")

    return results


def analyze_stop_loss_take_profit(
    df: pd.DataFrame,
    et_samples: pd.DataFrame,
) -> Dict[str, Any]:
    """分析ET的止损止盈配置"""
    print("\n" + "=" * 80)
    print("2. 止损止盈配置分析")
    print("=" * 80)

    results = {
        "current_config": {
            "stop_loss_r": 1.0,
            "take_profit_r": 2.0,
            "max_holding_bars": 24,
        },
        "mean_mode_config": {
            "stop_loss_r": 3.0,
            "take_profit_r": 5.0,
            "max_holding_bars": 24,
        },
        "ret_mean_stats": {},
        "positive_samples": {},
        "negative_samples": {},
    }

    if "ret_mean" in et_samples.columns:
        ret = et_samples["ret_mean"].dropna()
        results["ret_mean_stats"] = {
            "mean": float(ret.mean()),
            "median": float(ret.median()),
            "std": float(ret.std()),
            "min": float(ret.min()),
            "max": float(ret.max()),
            "positive_count": int((ret > 0).sum()),
            "negative_count": int((ret <= 0).sum()),
        }

        # 分析正负收益样本
        positive = et_samples[et_samples["ret_mean"] > 0]
        negative = et_samples[et_samples["ret_mean"] <= 0]

        if len(positive) > 0:
            results["positive_samples"] = {
                "count": len(positive),
                "avg_ret": float(positive["ret_mean"].mean()),
            }

        if len(negative) > 0:
            results["negative_samples"] = {
                "count": len(negative),
                "avg_ret": float(negative["ret_mean"].mean()),
            }

    print(f"\n当前ET配置:")
    print(f"  stop_loss_r: {results['current_config']['stop_loss_r']}")
    print(f"  take_profit_r: {results['current_config']['take_profit_r']}")
    print(f"  max_holding_bars: {results['current_config']['max_holding_bars']}")

    print(f"\nMEAN模式配置（实际使用）:")
    print(f"  stop_loss_r: {results['mean_mode_config']['stop_loss_r']}")
    print(f"  take_profit_r: {results['mean_mode_config']['take_profit_r']}")
    print(f"  max_holding_bars: {results['mean_mode_config']['max_holding_bars']}")

    print(f"\nret_mean统计:")
    if results["ret_mean_stats"]:
        stats = results["ret_mean_stats"]
        print(f"  平均值: {stats['mean']:.6f}")
        print(
            f"  正收益样本: {stats['positive_count']}/{stats['positive_count'] + stats['negative_count']}"
        )
        print(
            f"  负收益样本: {stats['negative_count']}/{stats['positive_count'] + stats['negative_count']}"
        )

    return results


def analyze_sharpe_optimization(
    df: pd.DataFrame,
    et_samples: pd.DataFrame,
) -> Dict[str, Any]:
    """分析Sharpe优化方案"""
    print("\n" + "=" * 80)
    print("3. Sharpe优化分析")
    print("=" * 80)

    results = {
        "current_sharpe": None,
        "relaxation_tests": [],
        "feature_analysis": {},
    }

    if "ret_mean" in et_samples.columns:
        ret = et_samples["ret_mean"].dropna()
        if len(ret) > 1 and ret.std() > 0:
            sharpe = ret.mean() / ret.std() * np.sqrt(252)
            results["current_sharpe"] = float(sharpe)
            print(f"\n当前Sharpe: {sharpe:.3f}")

    # 分析正负收益样本的特征差异
    positive = et_samples[et_samples["ret_mean"] > 0]
    negative = et_samples[et_samples["ret_mean"] <= 0]

    key_features = [
        "atr_percentile",
        "path_efficiency_pct",
        "jump_risk_pct",
        "path_length_pct",
        "vpin",
    ]

    print(f"\n正负收益样本特征对比:")
    for feat in key_features:
        if feat in et_samples.columns:
            pos_vals = positive[feat].dropna()
            neg_vals = negative[feat].dropna()
            if len(pos_vals) > 0 and len(neg_vals) > 0:
                pos_mean = pos_vals.mean()
                neg_mean = neg_vals.mean()
                diff = pos_mean - neg_mean
                print(f"  {feat}:")
                print(
                    f"    正收益: {pos_mean:.3f}, 负收益: {neg_mean:.3f}, 差异: {diff:.3f}"
                )
                results["feature_analysis"][feat] = {
                    "positive_mean": float(pos_mean),
                    "negative_mean": float(neg_mean),
                    "difference": float(diff),
                }

    # 测试放宽ET_REGIME条件
    print(f"\n测试放宽ET_REGIME条件:")
    mean_regime = df[df["regime"] == "MEAN_REGIME"].copy()
    all_data = df.copy()

    relaxation_tests = [
        {"atr_percentile_min": 0.75, "name": "降低atr_percentile到0.75"},
        {"atr_percentile_min": 0.7, "name": "降低atr_percentile到0.7"},
        {
            "path_efficiency_min": 0.35,
            "path_efficiency_max": 0.65,
            "name": "放宽path_efficiency到0.35-0.65",
        },
        {
            "jump_risk_min": 0.25,
            "jump_risk_max": 0.65,
            "name": "放宽jump_risk到0.25-0.65",
        },
    ]

    for test in relaxation_tests:
        # 构建条件
        conditions = []
        if "atr_percentile_min" in test and "atr_percentile" in all_data.columns:
            conditions.append(all_data["atr_percentile"] >= test["atr_percentile_min"])
        if "path_efficiency_min" in test and "path_efficiency_pct" in all_data.columns:
            conditions.append(
                all_data["path_efficiency_pct"] >= test["path_efficiency_min"]
            )
        if "path_efficiency_max" in test and "path_efficiency_pct" in all_data.columns:
            conditions.append(
                all_data["path_efficiency_pct"] <= test["path_efficiency_max"]
            )
        if "jump_risk_min" in test and "jump_risk_pct" in all_data.columns:
            conditions.append(all_data["jump_risk_pct"] >= test["jump_risk_min"])
        if "jump_risk_max" in test and "jump_risk_pct" in all_data.columns:
            conditions.append(all_data["jump_risk_pct"] < test["jump_risk_max"])
        if "path_length_min" in test and "path_length_pct" in all_data.columns:
            conditions.append(
                all_data["path_length_pct"] >= test.get("path_length_min", 0.5)
            )

        if conditions:
            mask = pd.Series(True, index=all_data.index)
            for cond in conditions:
                mask = mask & cond

            candidates = all_data[mask]
            if len(candidates) > 0 and "ret_mean" in candidates.columns:
                ret = candidates["ret_mean"].dropna()
                if len(ret) > 0:
                    mean_ret = ret.mean()
                    win_rate = (ret > 0).sum() / len(ret) if len(ret) > 0 else 0
                    sharpe = (
                        (ret.mean() / ret.std() * np.sqrt(252))
                        if len(ret) > 1 and ret.std() > 0
                        else 0
                    )

                    test_result = {
                        "name": test["name"],
                        "sample_count": len(candidates),
                        "mean_ret": float(mean_ret),
                        "win_rate": float(win_rate),
                        "sharpe": float(sharpe),
                    }
                    results["relaxation_tests"].append(test_result)

                    print(f"  {test['name']}:")
                    print(f"    样本数: {len(candidates)}")
                    print(f"    平均ret_mean: {mean_ret:.6f}")
                    print(f"    胜率: {win_rate*100:.1f}%")
                    print(f"    Sharpe: {sharpe:.3f}")

    return results


def main() -> int:
    p = argparse.ArgumentParser(description="ET optimization analysis")
    p.add_argument("--logs", required=True, help="Input logs file")
    p.add_argument(
        "--feature-store-root", default="feature_store", help="FeatureStore root"
    )
    p.add_argument(
        "--feature-store-layer",
        default="nnmh_highcap6_240T_2024_202510_v2",
        help="FeatureStore layer",
    )
    p.add_argument("--timeframe", default="240T", help="Timeframe")
    p.add_argument(
        "--output-json",
        default="results/et_optimization_analysis.json",
        help="Output JSON",
    )
    args = p.parse_args()

    # 读取数据
    df = pd.read_parquet(args.logs)
    et_samples = df[
        (df["regime"] == "ET_REGIME")
        & (
            df.get("gate_archetype", pd.Series([""] * len(df))).str.contains(
                "ET", case=False, na=False
            )
        )
    ].copy()

    print(f"总样本数: {len(df)}")
    print(f"ET_REGIME样本数: {len(et_samples)}")

    # 1. Volume Profile特征分析
    symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else []
    vp_results = analyze_volume_profile_features(
        args.feature_store_root,
        args.feature_store_layer,
        symbols,
        args.timeframe,
    )

    # 2. 止损止盈配置分析
    rr_results = analyze_stop_loss_take_profit(df, et_samples)

    # 3. Sharpe优化分析
    sharpe_results = analyze_sharpe_optimization(df, et_samples)

    # 保存结果
    output = {
        "volume_profile_analysis": vp_results,
        "stop_loss_take_profit_analysis": rr_results,
        "sharpe_optimization_analysis": sharpe_results,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ 分析结果已保存到: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
