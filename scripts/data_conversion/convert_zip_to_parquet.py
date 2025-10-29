#!/usr/bin/env python3
"""
数据转换脚本：将zip格式的交易数据转换为parquet格式
集成到下载功能中，提高后续处理速度
"""

import os
import sys
import pandas as pd
import numpy as np
import zipfile
import glob
from datetime import datetime, timedelta
import shutil
import logging

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataConverter:
    """数据转换器"""

    def __init__(self, input_dir, output_dir, backup_dir=None):
        """
        初始化数据转换器

        Args:
            input_dir: 输入目录（包含zip文件）
            output_dir: 输出目录（存储parquet文件）
            backup_dir: 备份目录（可选，用于备份原始zip文件）
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.backup_dir = backup_dir

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        if self.backup_dir:
            os.makedirs(self.backup_dir, exist_ok=True)

    def convert_zip_to_parquet(self, zip_file, symbol=None):
        """
        将单个zip文件转换为parquet格式

        Args:
            zip_file: zip文件路径
            symbol: 交易对符号（自动从文件名检测）

        Returns:
            dict: 转换结果信息
        """
        try:
            logger.info(f"Converting {os.path.basename(zip_file)}...")

            # 从文件名自动检测交易对
            if symbol is None:
                zip_basename = os.path.basename(zip_file)
                if "ETHUSDT" in zip_basename or "ETH-USDT" in zip_basename:
                    symbol = "ETH-USDT"
                elif "BTCUSDT" in zip_basename or "BTC-USDT" in zip_basename:
                    symbol = "BTC-USDT"
                elif "SOLUSDT" in zip_basename or "SOL-USDT" in zip_basename:
                    symbol = "SOL-USDT"
                else:
                    symbol = "UNKNOWN"
                    logger.warning(
                        f"Could not detect symbol from filename, using {symbol}"
                    )

            # 解压zip文件
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                file_list = zip_ref.namelist()
                if not file_list:
                    logger.warning(f"No files in zip: {zip_file}")
                    return None

                # 读取CSV文件
                csv_file = file_list[0]

                with zip_ref.open(csv_file) as f:
                    # 尝试读取第一行检查是否有header
                    first_line = f.readline().decode("utf-8", errors="ignore")

                read_params = {"low_memory": False}

                # 如果第一行是数字，说明没有header
                if first_line.strip().split(",")[0].replace(".", "").isdigit():
                    logger.info(
                        "No header detected, using default column names")
                    read_params.update({
                        "header":
                        None,
                        "names": [
                            "agg_trade_id",
                            "price",
                            "quantity",
                            "first_trade_id",
                            "last_trade_id",
                            "transact_time",
                            "is_buyer_maker",
                        ],
                    })

                try:
                    with zip_ref.open(csv_file) as csv_handle:
                        df = pd.read_csv(csv_handle, **read_params)
                except Exception as read_error:
                    logger.warning(
                        "Primary CSV load failed for %s, retrying with python engine: %s",
                        zip_basename,
                        read_error,
                    )
                    fallback_params = {**read_params, "engine": "python"}
                    with zip_ref.open(csv_file) as csv_handle:
                        df = pd.read_csv(csv_handle, **fallback_params)

                logger.info(f"Loaded data: {df.shape}")

                # 数据预处理
                df = self._preprocess_data(df)
                if df is None:
                    return None

                # 重采样为5分钟K线
                df_ohlc = self._resample_to_ohlc(df)

                # 设置 symbol
                # 转换符号格式：ETH-USDT -> ETH-USD
                output_symbol = symbol.replace("USDT", "USD")
                df_ohlc["symbol"] = output_symbol

                # 生成输出文件名
                output_file = self._generate_output_filename(
                    zip_file, output_symbol)

                # 保存为parquet
                df_ohlc.to_parquet(output_file, compression="snappy")
                logger.info(f"Saved to: {output_file}")

                # 备份原始文件
                if self.backup_dir:
                    backup_file = os.path.join(self.backup_dir,
                                               os.path.basename(zip_file))
                    shutil.copy2(zip_file, backup_file)
                    logger.info(f"Backed up to: {backup_file}")

                return {
                    "original_file": zip_file,
                    "output_file": output_file,
                    "start_date": df_ohlc.index.min().strftime("%Y-%m-%d"),
                    "end_date": df_ohlc.index.max().strftime("%Y-%m-%d"),
                    "shape": df_ohlc.shape,
                    "symbol": symbol,
                }

        except Exception as e:
            logger.error(f"Error converting {zip_file}: {e}")
            return None

    def _preprocess_data(self, df):
        """数据预处理"""
        try:
            # 检查必要的列
            required_cols = ["transact_time", "price", "quantity"]
            if not all(col in df.columns for col in required_cols):
                logger.warning(
                    f"Missing required columns in {df.columns.tolist()}")
                return None

            # 转换时间戳
            df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")

            # 重命名列
            df = df.rename(columns={"price": "close", "quantity": "volume"})

            # 按时间排序
            df = df.sort_values("timestamp")

            # 去除重复数据
            df = df.drop_duplicates(subset=["timestamp"])

            return df

        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return None

    def _resample_to_ohlc(self, df):
        """重采样为OHLC数据并添加订单流特征"""
        try:
            # 设置时间戳为索引
            df_indexed = df.set_index("timestamp")

            # 重采样为5分钟K线
            df_ohlc = (df_indexed.resample("5min").agg({
                "close": ["first", "max", "min", "last"],
                "volume":
                "sum"
            }).dropna())

            # 展平列名
            df_ohlc.columns = ["open", "high", "low", "close", "volume"]

            # 添加其他必要列 - symbol 从外部传入
            # df_ohlc["symbol"] will be set by convert_zip_to_parquet
            df_ohlc["timestamp"] = df_ohlc.index

            # 计算交易数量
            df_ohlc["trade_count"] = df_indexed.resample("5min").size()

            # 添加订单流特征 (如果有 is_buyer_maker 列)
            if "is_buyer_maker" in df_indexed.columns:
                # 分类买卖方
                df_indexed["taker_buy"] = (
                    ~df_indexed["is_buyer_maker"].astype(bool)).astype(int)
                df_indexed["buy_qty"] = np.where(df_indexed["taker_buy"] == 1,
                                                 df_indexed["volume"], 0.0)
                df_indexed["sell_qty"] = np.where(df_indexed["taker_buy"] == 1,
                                                  0.0, df_indexed["volume"])

                # 重采样订单流
                order_flow = df_indexed.resample("5min").agg({
                    "buy_qty": "sum",
                    "sell_qty": "sum"
                })

                # 计算 taker_buy_ratio
                order_flow["taker_buy_ratio"] = order_flow["buy_qty"] / (
                    order_flow["buy_qty"] + order_flow["sell_qty"]).replace(
                        0, np.nan)
                order_flow["taker_buy_ratio"] = order_flow[
                    "taker_buy_ratio"].fillna(0.5)

                # 计算 CVD 特征
                delta = order_flow["buy_qty"] - order_flow["sell_qty"]
                order_flow["cvd_short"] = delta.rolling(window=20,
                                                        min_periods=1).sum()
                order_flow["cvd_medium"] = delta.rolling(window=60,
                                                         min_periods=1).sum()
                order_flow["cvd_long"] = delta.rolling(window=288,
                                                       min_periods=1).sum()
                order_flow["cvd_change_1"] = delta
                order_flow["cvd_change_5"] = delta.rolling(window=5).sum()
                order_flow["cvd_change_20"] = delta.rolling(window=20).sum()

                # CVD 归一化
                total_volume = order_flow["buy_qty"] + order_flow["sell_qty"]
                order_flow["cvd_normalized"] = delta / total_volume.replace(
                    0, np.nan)
                order_flow["cvd_normalized"] = order_flow[
                    "cvd_normalized"].fillna(0)
                order_flow["cvd"] = delta.cumsum()

                # 合并到 OHLC 数据
                df_ohlc = df_ohlc.join(order_flow[[
                    "buy_qty", "sell_qty", "taker_buy_ratio", "cvd",
                    "cvd_short", "cvd_medium", "cvd_long", "cvd_change_1",
                    "cvd_change_5", "cvd_change_20", "cvd_normalized"
                ]],
                                       how="left").ffill().fillna(0)

            return df_ohlc

        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            return None

    def _generate_output_filename(self, zip_file, symbol):
        """生成输出文件名"""
        zip_basename = os.path.basename(zip_file)

        # 提取日期信息 - 支持多种格式
        if "ETHUSDT-aggTrades-" in zip_basename:
            date_part = zip_basename.replace("ETHUSDT-aggTrades-",
                                             "").replace(".zip", "")
        elif "BTCUSDT-aggTrades-" in zip_basename:
            date_part = zip_basename.replace("BTCUSDT-aggTrades-",
                                             "").replace(".zip", "")
        elif "SOLUSDT-aggTrades-" in zip_basename:
            date_part = zip_basename.replace("SOLUSDT-aggTrades-",
                                             "").replace(".zip", "")
        elif "aggTrades-" in zip_basename:
            # 通用格式
            date_part = zip_basename.split("aggTrades-")[1].replace(".zip", "")
        else:
            # 无法识别，使用当前日期
            date_part = datetime.now().strftime("%Y-%m")
            logger.warning(
                f"Could not extract date from filename, using {date_part}")

        # 生成输出文件名
        output_filename = f"{symbol}_{date_part}.parquet"
        return os.path.join(self.output_dir, output_filename)

    def convert_all_files(self, pattern="*aggTrades-*.zip"):
        """转换所有匹配的zip文件（支持多币种）"""
        logger.info(f"Converting all files matching pattern: {pattern}")

        # 查找所有匹配的zip文件
        zip_files = glob.glob(os.path.join(self.input_dir, pattern))
        logger.info(f"Found {len(zip_files)} files to convert")

        converted_files = []
        failed_files = []

        total_files = len(zip_files)
        if total_files == 0:
            logger.warning("No matching ZIP files found for conversion.")
            return {
                "converted_files": converted_files,
                "failed_files": failed_files,
                "total_files": total_files,
            }

        for index, zip_file in enumerate(zip_files, start=1):
            file_name = os.path.basename(zip_file)
            progress_prefix = f"[{index}/{total_files}]"
            print(f"{progress_prefix} Converting {file_name} ...")

            result = self.convert_zip_to_parquet(zip_file)
            if result:
                converted_files.append(result)
                print(f"{progress_prefix} ✅ Success: {file_name}")
            else:
                failed_files.append(zip_file)
                print(f"{progress_prefix} ❌ Failed: {file_name}")

        logger.info(
            f"Conversion complete: {len(converted_files)} successful, {len(failed_files)} failed"
        )

        return {
            "converted_files": converted_files,
            "failed_files": failed_files,
            "total_files": len(zip_files),
        }

    def cleanup_zip_files(self, converted_files):
        """清理已转换的zip文件"""
        logger.info("Cleaning up converted zip files...")

        cleaned_count = 0
        for file_info in converted_files:
            try:
                original_file = file_info["original_file"]
                if os.path.exists(original_file):
                    os.remove(original_file)
                    cleaned_count += 1
                    logger.info(f"Removed: {original_file}")
            except Exception as e:
                logger.error(f"Error removing {original_file}: {e}")

        logger.info(f"Cleaned up {cleaned_count} zip files")
        return cleaned_count


def main():
    """主函数"""
    print("🚀 Converting ZIP files to Parquet format...")

    # 配置路径（使用绝对路径，指向仓库根目录）
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    input_dir = os.path.join(base_dir, "data", "agg_data")
    output_dir = os.path.join(base_dir, "data", "parquet_data")
    backup_dir = os.path.join(base_dir, "data", "backup_zip")

    print(f"📂 Base directory: {base_dir}")
    print(f"📂 Input directory: {input_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📂 Backup directory: {backup_dir}")
    print()

    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        print(f"💡 Expected data directory structure:")
        print(f"   {base_dir}/data/agg_data/")
        return

    # 创建转换器
    converter = DataConverter(input_dir, output_dir, backup_dir)

    # 转换所有文件
    results = converter.convert_all_files()

    # 打印结果
    print(f"\n📊 Conversion Results:")
    print(f"   Total files: {results['total_files']}")
    print(f"   Converted: {len(results['converted_files'])}")
    print(f"   Failed: {len(results['failed_files'])}")

    if results["converted_files"]:
        print(f"\n✅ Successfully converted files:")
        for file_info in results["converted_files"][:5]:  # 显示前5个
            print(
                f"   {os.path.basename(file_info['original_file'])} -> {os.path.basename(file_info['output_file'])}"
            )

        if len(results["converted_files"]) > 5:
            print(
                f"   ... and {len(results['converted_files']) - 5} more files")

    if results["failed_files"]:
        print(f"\n❌ Failed files:")
        for failed_file in results["failed_files"][:5]:  # 显示前5个
            print(f"   {os.path.basename(failed_file)}")

        if len(results["failed_files"]) > 5:
            print(f"   ... and {len(results['failed_files']) - 5} more files")

    # 询问是否清理zip文件
    if results["converted_files"]:
        response = input(
            f"\n🗑️  Clean up {len(results['converted_files'])} converted zip files? (y/N): "
        )
        if response.lower() == "y":
            cleaned_count = converter.cleanup_zip_files(
                results["converted_files"])
            print(f"✅ Cleaned up {cleaned_count} zip files")

    print(f"\n🎉 Data conversion complete!")
    print(f"   Output directory: {output_dir}")
    print(f"   Backup directory: {backup_dir}")
    print(f"   Ready for fast processing!")


if __name__ == "__main__":
    main()
