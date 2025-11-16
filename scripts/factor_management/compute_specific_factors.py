#!/usr/bin/env python3
"""
计算指定因子工具（用于实盘）

功能：
1. 只计算指定的因子，不计算其他因子
2. 输出因子值到文件或数据库
3. 支持实时计算和批量计算

用法：
    # 计算单个因子
    python scripts/factor_management/compute_specific_factors.py \
        --factors rsi_7 \
        --input data/btcusdt_2024.parquet \
        --output factors/rsi_7.csv
    
    # 计算多个因子
    python scripts/factor_management/compute_specific_factors.py \
        --factors rsi_7 zigzag_normalized macd \
        --input data/btcusdt_2024.parquet \
        --output factors/selected_factors.csv
"""

import argparse
import sys
from pathlib import Path
from typing import List, Set
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_tools.comprehensive_feature_engineering import ComprehensiveFeatureEngineer
from src.data_tools.data_loader import MarketDataLoader


def compute_factors(
    factors: List[str],
    input_data: pd.DataFrame,
    feature_type: str = "comprehensive",
    fit: bool = True,
) -> pd.DataFrame:
    """
    计算指定的因子
    
    Args:
        factors: 要计算的因子名称列表
        input_data: 输入数据（OHLCV）
        feature_type: 特征类型
        fit: 是否拟合标准化器
    
    Returns:
        包含指定因子的 DataFrame
    """
    factors_set = set(factors)
    
    # 创建特征工程器
    engineer = ComprehensiveFeatureEngineer(feature_types=feature_type)
    
    # 只计算需要的因子
    df_features = engineer.engineer_all_features(
        input_data,
        fit=fit,
        required_features=factors_set
    )
    
    # 只保留需要的因子列和数据列
    data_cols = {'open', 'high', 'low', 'close', 'volume', 'timestamp', 'datetime'}
    cols_to_keep = [
        col for col in df_features.columns
        if col in factors_set or col in data_cols
    ]
    
    return df_features[cols_to_keep]


def main():
    parser = argparse.ArgumentParser(description="计算指定的因子（用于实盘）")
    parser.add_argument(
        "--factors",
        nargs="+",
        required=True,
        help="要计算的因子名称列表"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="输入文件路径（parquet 或 csv）"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="数据目录路径（如果使用 MarketDataLoader）"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="交易对符号（如果使用 MarketDataLoader）"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="开始日期（如果使用 MarketDataLoader）"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="结束日期（如果使用 MarketDataLoader）"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出文件路径"
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="comprehensive",
        help="特征类型"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet", "json"],
        default="csv",
        help="输出格式"
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"📂 加载数据...")
    if args.input:
        # 从文件加载
        input_path = Path(args.input)
        if input_path.suffix == ".parquet":
            df = pd.read_parquet(input_path)
        elif input_path.suffix == ".csv":
            df = pd.read_csv(input_path, parse_dates=True)
        else:
            print(f"❌ 不支持的文件格式: {input_path.suffix}")
            sys.exit(1)
        print(f"   ✅ 从文件加载: {len(df)} 条数据")
    elif args.data_path and args.symbol:
        # 使用 MarketDataLoader
        loader = MarketDataLoader(args.data_path)
        df = loader.load_data(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        print(f"   ✅ 从数据目录加载: {len(df)} 条数据")
    else:
        print("❌ 必须指定 --input 或 (--data-path + --symbol)")
        sys.exit(1)
    
    # 计算因子
    print(f"\n🔧 计算因子: {', '.join(args.factors)}")
    try:
        df_factors = compute_factors(
            factors=args.factors,
            input_data=df,
            feature_type=args.feature_type,
            fit=True,
        )
        
        # 检查哪些因子被成功计算
        computed_factors = [f for f in args.factors if f in df_factors.columns]
        missing_factors = [f for f in args.factors if f not in df_factors.columns]
        
        print(f"   ✅ 成功计算: {len(computed_factors)} 个因子")
        for factor in computed_factors:
            print(f"      - {factor}")
        
        if missing_factors:
            print(f"   ⚠️  缺失因子: {len(missing_factors)} 个")
            for factor in missing_factors:
                print(f"      - {factor}")
        
    except Exception as e:
        print(f"❌ 计算因子失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 保存结果
    print(f"\n💾 保存结果到: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.format == "csv":
            df_factors.to_csv(output_path, index=False)
        elif args.format == "parquet":
            df_factors.to_parquet(output_path, index=False)
        elif args.format == "json":
            df_factors.to_json(output_path, orient="records", date_format="iso")
        
        print(f"   ✅ 保存成功: {len(df_factors)} 行数据")
        print(f"   📊 列: {list(df_factors.columns)}")
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        sys.exit(1)
    
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()

