#!/usr/bin/env python3
"""
诊断 zigzag 特征未被选中的原因

分析：
1. zigzag 特征是否被正确计算
2. zigzag 特征是否被包含在特征列表中
3. zigzag 特征的 IC 值是多少
4. zigzag 特征是否因为归一化问题被排除
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置 PYTHONPATH
import os
os.environ['PYTHONPATH'] = str(project_root) + ':' + os.environ.get('PYTHONPATH', '')

try:
    from src.data_tools.comprehensive_feature_engineering import ComprehensiveFeatureEngineer
    from src.data_tools.data_loader import MarketDataLoader
except ImportError:
    # 如果导入失败，使用简化的检查
    ComprehensiveFeatureEngineer = None
    MarketDataLoader = None


def check_zigzag_in_features(df: pd.DataFrame) -> dict:
    """检查 zigzag 特征是否在数据中"""
    result = {
        "zigzag_exists": "zigzag" in df.columns,
        "zigzag_normalized_exists": "zigzag_normalized" in df.columns,
        "zigzag_stats": None,
        "zigzag_normalized_stats": None,
    }
    
    if result["zigzag_exists"]:
        zigzag = df["zigzag"]
        result["zigzag_stats"] = {
            "mean": float(zigzag.mean()),
            "std": float(zigzag.std()),
            "min": float(zigzag.min()),
            "max": float(zigzag.max()),
            "nan_count": int(zigzag.isna().sum()),
            "zero_count": int((zigzag == 0).sum()),
            "unique_values": int(zigzag.nunique()),
        }
    
    if result["zigzag_normalized_exists"]:
        zigzag_norm = df["zigzag_normalized"]
        result["zigzag_normalized_stats"] = {
            "mean": float(zigzag_norm.mean()),
            "std": float(zigzag_norm.std()),
            "min": float(zigzag_norm.min()),
            "max": float(zigzag_norm.max()),
            "nan_count": int(zigzag_norm.isna().sum()),
        }
    
    return result


def check_zigzag_in_feature_selection(df: pd.DataFrame, feature_cols: list) -> dict:
    """检查 zigzag 是否在特征选择列表中"""
    result = {
        "zigzag_in_features": "zigzag" in feature_cols,
        "zigzag_normalized_in_features": "zigzag_normalized" in feature_cols,
        "total_features": len(feature_cols),
    }
    return result


def calculate_zigzag_ic(df: pd.DataFrame, target_col: str = "binary_signal_1") -> dict:
    """计算 zigzag 特征的 IC 值"""
    result = {}
    
    if target_col not in df.columns:
        return {"error": f"Target column {target_col} not found"}
    
    y = df[target_col].dropna()
    
    for col in ["zigzag", "zigzag_normalized"]:
        if col in df.columns:
            try:
                # 对齐索引
                aligned = pd.DataFrame({
                    "feature": df[col],
                    "target": y
                }).dropna()
                
                if len(aligned) > 10:
                    ic, p_value = spearmanr(
                        aligned["feature"].values,
                        aligned["target"].values,
                        nan_policy="omit"
                    )
                    result[col] = {
                        "ic": float(ic) if not np.isnan(ic) else 0.0,
                        "ic_abs": float(abs(ic)) if not np.isnan(ic) else 0.0,
                        "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
                        "sample_count": len(aligned),
                    }
                else:
                    result[col] = {
                        "error": "Insufficient samples",
                        "sample_count": len(aligned),
                    }
            except Exception as e:
                result[col] = {"error": str(e)}
    
    return result


def check_zigzag_derived_features(df: pd.DataFrame) -> dict:
    """检查是否有基于 zigzag 的衍生特征"""
    zigzag_related = [col for col in df.columns if "zigzag" in col.lower()]
    
    result = {
        "zigzag_related_features": zigzag_related,
        "count": len(zigzag_related),
    }
    
    return result


def main():
    # 读取 top_factors.json
    top_factors_path = Path(
        "results/dim_compare/BTCUSDT-ETHUSDT_comprehensive_20200101_20211231/top_factors.json"
    )
    
    if not top_factors_path.exists():
        print(f"❌ 文件不存在: {top_factors_path}")
        return
    
    with open(top_factors_path, "r") as f:
        top_factors_data = json.load(f)
    
    top_factors = [f["name"] for f in top_factors_data["top_factors"]]
    
    print("=" * 80)
    print("ZigZag 特征诊断报告")
    print("=" * 80)
    
    # 1. 检查 top_factors 中是否有 zigzag
    print("\n1. 检查 top_factors.json 中的 zigzag 特征")
    print("-" * 80)
    zigzag_in_top = [f for f in top_factors if "zigzag" in f.lower()]
    if zigzag_in_top:
        print(f"   ✅ 找到 zigzag 相关特征: {zigzag_in_top}")
    else:
        print(f"   ❌ 未找到 zigzag 相关特征")
        print(f"   📊 总特征数: {len(top_factors)}")
        print(f"   📋 前 20 个特征: {top_factors[:20]}")
    
    # 2. 加载数据并检查 zigzag 特征
    print("\n2. 检查特征工程中的 zigzag")
    print("-" * 80)
    
    # 尝试加载数据（使用较小的数据集进行测试）
    try:
        if MarketDataLoader is None:
            raise ImportError("MarketDataLoader not available")
        
        data_path = "/data/parquet_data"
        loader = MarketDataLoader(data_path)
        
        # 加载 BTCUSDT 数据（使用与 dim_compare 相同的时间范围）
        print("   📂 加载数据...")
        df = loader.load_data(
            symbol="BTCUSDT",
            start_date="2020-01-01",
            end_date="2021-12-31",
        )
        
        if df.empty:
            print("   ⚠️  数据为空，使用模拟数据")
            # 创建模拟数据
            dates = pd.date_range("2020-01-01", periods=1000, freq="5T")
            df = pd.DataFrame({
                "timestamp": dates,
                "open": np.random.randn(1000).cumsum() + 50000,
                "high": np.random.randn(1000).cumsum() + 50500,
                "low": np.random.randn(1000).cumsum() + 49500,
                "close": np.random.randn(1000).cumsum() + 50000,
                "volume": np.random.randint(1000, 10000, 1000),
            })
        
        # 进行特征工程
        print("   🔧 进行特征工程...")
        if ComprehensiveFeatureEngineer is None:
            raise ImportError("ComprehensiveFeatureEngineer not available")
        
        engineer = ComprehensiveFeatureEngineer(feature_types="comprehensive")
        df_features = engineer.engineer_all_features(df, fit=True)
        
        # 检查 zigzag 特征
        zigzag_check = check_zigzag_in_features(df_features)
        print(f"   zigzag 存在: {zigzag_check['zigzag_exists']}")
        print(f"   zigzag_normalized 存在: {zigzag_check['zigzag_normalized_exists']}")
        
        if zigzag_check['zigzag_exists']:
            print(f"   📊 zigzag 统计:")
            stats = zigzag_check['zigzag_stats']
            print(f"      - 均值: {stats['mean']:.4f}")
            print(f"      - 标准差: {stats['std']:.4f}")
            print(f"      - 最小值: {stats['min']:.4f}")
            print(f"      - 最大值: {stats['max']:.4f}")
            print(f"      - NaN 数量: {stats['nan_count']}")
            print(f"      - 零值数量: {stats['zero_count']}")
            print(f"      - 唯一值数量: {stats['unique_values']}")
        
        # 检查特征列
        feature_cols = engineer.get_feature_columns(df_features)
        feature_check = check_zigzag_in_feature_selection(df_features, feature_cols)
        print(f"\n   📋 特征选择检查:")
        print(f"      - zigzag 在特征列表中: {feature_check['zigzag_in_features']}")
        print(f"      - zigzag_normalized 在特征列表中: {feature_check['zigzag_normalized_in_features']}")
        print(f"      - 总特征数: {feature_check['total_features']}")
        
        # 检查 zigzag 相关特征
        zigzag_related = check_zigzag_derived_features(df_features)
        print(f"\n   🔍 zigzag 相关特征:")
        print(f"      - 数量: {zigzag_related['count']}")
        if zigzag_related['zigzag_related_features']:
            print(f"      - 特征列表: {zigzag_related['zigzag_related_features']}")
        
        # 如果有标签，计算 IC
        if "binary_signal_1" in df_features.columns:
            print(f"\n   📈 计算 IC 值...")
            ic_results = calculate_zigzag_ic(df_features, "binary_signal_1")
            for col, result in ic_results.items():
                if "error" not in result:
                    print(f"      - {col}:")
                    print(f"        IC: {result['ic']:.6f}")
                    print(f"        |IC|: {result['ic_abs']:.6f}")
                    print(f"        p-value: {result['p_value']:.6f}")
                    print(f"        样本数: {result['sample_count']}")
                else:
                    print(f"      - {col}: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. 分析原因
    print("\n3. 可能的原因分析")
    print("-" * 80)
    
    reasons = []
    
    if not zigzag_in_top:
        reasons.append("❌ zigzag 特征未被选中")
    
    if zigzag_check.get('zigzag_exists') and not zigzag_check.get('zigzag_normalized_exists'):
        reasons.append("⚠️  zigzag 特征存在但没有归一化版本（zigzag_normalized）")
        reasons.append("   这可能导致 zigzag 因为量纲问题被排除或 IC 值低")
    
    if feature_check.get('zigzag_in_features') == False:
        reasons.append("⚠️  zigzag 特征不在 get_feature_columns() 返回的特征列表中")
        reasons.append("   可能原因：")
        reasons.append("   1. zigzag 是原始价格量纲，类似于 atr，可能被排除")
        reasons.append("   2. 需要创建 zigzag_normalized 特征（类似 atr_normalized）")
    
    if reasons:
        print("   发现的问题:")
        for reason in reasons:
            print(f"   {reason}")
    else:
        print("   ✅ 未发现明显问题")
    
    # 4. 建议
    print("\n4. 建议")
    print("-" * 80)
    print("   1. 创建 zigzag_normalized 特征（类似 atr_normalized）")
    print("      - zigzag_normalized = zigzag / close")
    print("      - 或者使用 ATR 归一化: zigzag_normalized = (zigzag - close) / atr")
    print("   2. 创建基于 zigzag 的衍生特征:")
    print("      - zigzag_distance: 当前价格到最近 zigzag 点的距离")
    print("      - zigzag_turn: zigzag 转折点标记")
    print("      - zigzag_slope: zigzag 段的斜率")
    print("   3. 检查 zigzag 的 IC 值，如果很低，可能需要:")
    print("      - 调整 zigzag 的阈值参数")
    print("      - 使用 zigzag 作为结构确认特征，而不是直接预测特征")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)


if __name__ == "__main__":
    main()

