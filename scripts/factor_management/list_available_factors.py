#!/usr/bin/env python3
"""
列出所有可用的因子

功能：
1. 列出指定 feature_type 下的所有可用因子
2. 按类别分组显示
3. 支持搜索和过滤

用法：
    python scripts/factor_management/list_available_factors.py \
        --feature-type comprehensive \
        --output factors_list.txt
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_tools.comprehensive_feature_engineering import ComprehensiveFeatureEngineer
from src.cross_sectional.factor_catalog import categorize_columns


def list_factors(
    feature_type: str = "comprehensive",
    search_pattern: str = None,
    output_file: str = None,
) -> Dict[str, List[str]]:
    """列出所有可用的因子"""
    
    print("=" * 80)
    print("列出可用因子")
    print("=" * 80)
    print(f"特征类型: {feature_type}")
    if search_pattern:
        print(f"搜索模式: {search_pattern}")
    print("=" * 80)
    
    # 创建模拟数据来生成特征
    print("\n1. 生成特征列表...")
    dates = pd.date_range("2024-01-01", periods=100, freq="5T")
    sample_data = pd.DataFrame({
        "timestamp": dates,
        "open": [100.0] * 100,
        "high": [105.0] * 100,
        "low": [95.0] * 100,
        "close": [100.0] * 100,
        "volume": [1000] * 100,
    })
    
    try:
        engineer = ComprehensiveFeatureEngineer(feature_types=feature_type)
        df_features = engineer.engineer_all_features(sample_data, fit=True)
        
        # 获取特征列（排除数据列和标签列）
        feature_cols = engineer.get_feature_columns(df_features)
        
        print(f"   ✅ 找到 {len(feature_cols)} 个因子")
        
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return {}
    
    # 过滤搜索模式
    if search_pattern:
        feature_cols = [
            f for f in feature_cols
            if search_pattern.lower() in f.lower()
        ]
        print(f"   🔍 匹配 '{search_pattern}': {len(feature_cols)} 个因子")
    
    # 按类别分组
    print("\n2. 按类别分组...")
    categories = categorize_columns(feature_cols)
    
    # 打印结果
    print("\n3. 因子列表:")
    print("=" * 80)
    
    total_count = 0
    for category_name, factors in categories.items():
        if factors:
            print(f"\n📁 {category_name} ({len(factors)} 个):")
            for factor in sorted(factors):
                print(f"   - {factor}")
            total_count += len(factors)
    
    print(f"\n总计: {total_count} 个因子")
    
    # 保存到文件
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("可用因子列表\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"特征类型: {feature_type}\n")
            if search_pattern:
                f.write(f"搜索模式: {search_pattern}\n")
            f.write(f"总计: {total_count} 个因子\n\n")
            
            for category_name, factors in categories.items():
                if factors:
                    f.write(f"\n{category_name} ({len(factors)} 个):\n")
                    for factor in sorted(factors):
                        f.write(f"  {factor}\n")
        
        print(f"\n✅ 列表已保存到: {output_path}")
    
    return categories


def main():
    parser = argparse.ArgumentParser(description="列出所有可用的因子")
    parser.add_argument(
        "--feature-type",
        type=str,
        default="comprehensive",
        help="特征类型: baseline, default, enhanced, comprehensive"
    )
    parser.add_argument(
        "--search",
        type=str,
        help="搜索模式（过滤因子名称）"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出文件路径（可选）"
    )
    
    args = parser.parse_args()
    
    categories = list_factors(
        feature_type=args.feature_type,
        search_pattern=args.search,
        output_file=args.output,
    )
    
    if not categories:
        print("\n❌ 未找到因子")
        sys.exit(1)
    
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()

