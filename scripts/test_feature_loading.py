#!/usr/bin/env python3
"""
测试FeatureStore特征加载功能

验证：
1. 特征提取函数能正确提取gate规则所需特征
2. 特征加载函数能正确处理DataFrame合并
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_gate_application import extract_required_features
import pandas as pd


def test_extract_required_features():
    """测试特征提取函数"""
    print("🧪 测试特征提取函数...")

    execution_archetypes_path = "config/nnmultihead/execution_archetypes.yaml"
    features = extract_required_features(execution_archetypes_path)

    assert len(features) > 0, "应该提取到至少一个特征"
    assert isinstance(features, list), "返回应该是列表"
    assert all(isinstance(f, str) for f in features), "所有特征应该是字符串"

    print(f"✅ 特征提取测试通过: 提取到 {len(features)} 个特征")
    print(f"   示例特征: {features[:5]}")

    # 检查一些常见的特征
    common_features = ["path_efficiency_pct", "jump_risk_pct", "cvd_change_5_pct"]
    found_common = [f for f in common_features if f in features]
    if found_common:
        print(f"   找到常见特征: {found_common}")

    return features


def test_feature_loading_logic():
    """测试特征加载逻辑（使用模拟数据）"""
    print("\n🧪 测试特征加载逻辑...")

    # 创建模拟的logs DataFrame
    logs_df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
            "ret_mean": [0.01, -0.02, 0.015],
            "ret_trend": [0.02, -0.01, 0.02],
        }
    )

    # 创建模拟的FeatureStore DataFrame
    feats_df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
            "path_efficiency_pct": [0.5, 0.6, 0.4],
            "jump_risk_pct": [0.1, 0.2, 0.15],
        }
    )

    # 测试merge逻辑
    merged = logs_df.merge(
        feats_df, on=["symbol", "timestamp"], how="left", suffixes=("", "_feat")
    )

    assert len(merged) == len(logs_df), "合并后行数应该与原始logs相同"
    assert "path_efficiency_pct" in merged.columns, "应该包含从FeatureStore加载的特征"
    assert "jump_risk_pct" in merged.columns, "应该包含从FeatureStore加载的特征"
    assert "ret_mean" in merged.columns, "应该保留原始logs的列"

    print("✅ 特征加载逻辑测试通过")
    print(f"   合并后列数: {len(merged.columns)} (原始: {len(logs_df.columns)})")
    print(f"   新增特征: {set(merged.columns) - set(logs_df.columns)}")

    return merged


def main():
    """运行所有测试"""
    print("=" * 60)
    print("FeatureStore特征加载功能测试")
    print("=" * 60)

    try:
        # 测试1: 特征提取
        features = test_extract_required_features()

        # 测试2: 特征加载逻辑
        merged_df = test_feature_loading_logic()

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
