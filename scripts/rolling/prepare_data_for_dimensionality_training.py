#!/usr/bin/env python3
"""
为降维训练准备数据
解压zip文件并转换为parquet格式，供滚动训练使用
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import zipfile
import glob


def extract_and_convert_zip_data(data_dir, output_dir):
    """解压zip文件并转换为parquet格式"""
    print(f"📊 Preparing data for dimensionality training...")
    print(f"   Input directory: {data_dir}")
    print(f"   Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # 查找ETH相关的zip文件
    eth_zip_files = glob.glob(os.path.join(data_dir, "ETHUSDT-aggTrades-*.zip"))
    print(f"   Found {len(eth_zip_files)} ETH zip files")

    processed_files = []

    for zip_file in eth_zip_files:
        print(f"   Processing: {os.path.basename(zip_file)}")

        try:
            # 解压zip文件
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                # 获取zip文件中的文件名
                file_list = zip_ref.namelist()
                if not file_list:
                    print(f"     ⚠️  No files in zip: {zip_file}")
                    continue

                # 读取第一个文件（通常是CSV）
                csv_file = file_list[0]
                with zip_ref.open(csv_file) as f:
                    df = pd.read_csv(f)

                print(f"     ✅ Loaded data: {df.shape}")

                # 数据预处理
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                elif "time" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["time"], unit="ms")
                else:
                    print(f"     ⚠️  No timestamp column found")
                    continue

                # 重命名列以匹配标准格式
                column_mapping = {
                    "price": "close",
                    "qty": "volume",
                    "agg_trade_id": "trade_id",
                }
                df = df.rename(columns=column_mapping)

                # 确保必要的列存在
                required_cols = ["timestamp", "close", "volume"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"     ⚠️  Missing columns: {missing_cols}")
                    continue

                # 按时间排序
                df = df.sort_values("timestamp")

                # 重采样到5分钟K线
                df_resampled = (
                    df.set_index("timestamp")
                    .resample("5T")
                    .agg({"close": "last", "volume": "sum", "trade_id": "count"})
                    .dropna()
                )

                # 计算OHLC
                df_ohlc = (
                    df.set_index("timestamp")
                    .resample("5T")
                    .agg(
                        {
                            "close": ["first", "max", "min", "last"],
                            "volume": "sum",
                            "trade_id": "count",
                        }
                    )
                    .dropna()
                )

                # 展平列名
                df_ohlc.columns = [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "trade_count",
                ]

                # 添加其他必要列
                df_ohlc["symbol"] = "ETH-USDT"
                df_ohlc["timestamp"] = df_ohlc.index

                print(f"     ✅ Resampled data: {df_ohlc.shape}")

                # 生成输出文件名
                zip_basename = os.path.basename(zip_file)
                date_part = zip_basename.replace("ETHUSDT-aggTrades-", "").replace(
                    ".zip", ""
                )

                # 确定日期范围
                start_date = df_ohlc.index.min().strftime("%Y-%m-%d")
                end_date = df_ohlc.index.max().strftime("%Y-%m-%d")

                output_file = os.path.join(
                    output_dir, f"ETH-USD_{start_date}_{end_date}.parquet"
                )

                # 保存为parquet
                df_ohlc.to_parquet(output_file)
                print(f"     ✅ Saved to: {output_file}")

                processed_files.append(
                    {
                        "original_file": zip_file,
                        "output_file": output_file,
                        "start_date": start_date,
                        "end_date": end_date,
                        "shape": df_ohlc.shape,
                    }
                )

        except Exception as e:
            print(f"     ❌ Error processing {zip_file}: {e}")
            continue

    print(f"\n✅ Data preparation complete!")
    print(f"   Processed {len(processed_files)} files")

    return processed_files


def create_sample_data_for_testing(
    output_dir, start_date="2024-10-01", end_date="2024-12-31"
):
    """创建样本数据用于测试"""
    print(f"📊 Creating sample data for testing...")

    # 生成日期范围
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # 生成5分钟K线数据
    timestamps = pd.date_range(start=start_dt, end=end_dt, freq="5T")

    # 生成模拟价格数据
    np.random.seed(42)
    n_bars = len(timestamps)

    # 模拟价格走势
    base_price = 3000  # ETH基础价格
    returns = np.random.normal(0, 0.001, n_bars)  # 0.1%的随机波动
    prices = base_price * np.exp(np.cumsum(returns))

    # 生成OHLC数据
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        # 生成OHLC
        volatility = np.random.uniform(0.005, 0.02)  # 0.5%-2%的波动
        high = close * (1 + np.random.uniform(0, volatility))
        low = close * (1 - np.random.uniform(0, volatility))
        open_price = close * (1 + np.random.uniform(-volatility / 2, volatility / 2))

        # 生成成交量
        volume = np.random.uniform(100, 1000)

        data.append(
            {
                "timestamp": timestamp,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "symbol": "ETH-USD",
            }
        )

    df = pd.DataFrame(data)

    # 保存样本数据
    output_file = os.path.join(output_dir, f"ETH-USD_{start_date}_{end_date}.parquet")
    df.to_parquet(output_file)

    print(f"   ✅ Sample data created: {df.shape}")
    print(f"   ✅ Saved to: {output_file}")

    return output_file


def main():
    """主函数"""
    print("🚀 Preparing data for dimensionality training...")

    # 数据目录
    data_dir = "/home/yin/trading/rlbot/data/agg_data"
    output_dir = "/home/yin/trading/rlbot/data/processed_data"

    # 检查输入目录
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return

    # 解压和转换数据
    processed_files = extract_and_convert_zip_data(data_dir, output_dir)

    if not processed_files:
        print("⚠️  No files processed, creating sample data...")
        create_sample_data_for_testing(output_dir)

    print(f"\n🎉 Data preparation complete!")
    print(f"   Output directory: {output_dir}")
    print(f"   Ready for dimensionality training!")


if __name__ == "__main__":
    main()
