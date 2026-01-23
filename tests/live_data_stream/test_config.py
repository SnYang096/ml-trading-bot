"""
测试配置

定义测试数据路径、存储路径、测试symbol和时间范围等配置
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import os


class TestConfig:
    """测试配置类"""

    # 测试数据路径（从项目根目录开始）
    # 使用绝对路径，基于项目根目录
    _PROJECT_ROOT = (
        Path(__file__).resolve().parents[2]
    )  # tests/live_data_stream -> project root
    PARQUET_DATA_1S_DIR = _PROJECT_ROOT / "data" / "parquet_data_1s"

    # 测试存储路径（避免污染生产数据）
    TEST_STORAGE_DIR = _PROJECT_ROOT / "data" / "test_live_storage"

    # 测试symbol（单symbol）
    TEST_SYMBOL = "BTCUSDT"

    # 多symbol测试配置
    TEST_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # 默认测试的symbol列表

    # 每个symbol的最大tick数量（多symbol测试）
    MAX_TICKS_PER_SYMBOL = 5000  # 每个symbol最多处理5000条tick

    # CI/CD测试优化配置
    # 通过环境变量 CI=true 启用CI/CD模式（使用100 ticks/symbol）
    IS_CI = os.getenv("CI", "").lower() in ("true", "1", "yes")
    MAX_TICKS_PER_SYMBOL_CI = 100  # CI/CD模式：每个symbol最多处理100条tick

    # 测试时间范围（默认最近1个月）
    TEST_START_DATE: Optional[datetime] = None
    TEST_END_DATE: Optional[datetime] = None

    # Socket中断时间点（可配置）
    INTERRUPT_AT: Optional[datetime] = None

    # 测试数据量控制（避免测试时间过长）
    MAX_TICKS_PER_TEST = 100000  # 每个测试最多处理10万条tick

    @classmethod
    def get_max_ticks_per_symbol(cls) -> int:
        """
        获取每个symbol的最大tick数量（根据是否在CI/CD环境）

        Returns:
            最大tick数量
        """
        return cls.MAX_TICKS_PER_SYMBOL_CI if cls.IS_CI else cls.MAX_TICKS_PER_SYMBOL

    @classmethod
    def get_test_date_range(cls, days: int = 30) -> tuple[datetime, datetime]:
        """
        获取测试日期范围

        Args:
            days: 天数（默认30天）

        Returns:
            (start_date, end_date)
        """
        if cls.TEST_END_DATE is None:
            end_date = datetime.now()
        else:
            end_date = cls.TEST_END_DATE

        if cls.TEST_START_DATE is None:
            start_date = end_date - timedelta(days=days)
        else:
            start_date = cls.TEST_START_DATE

        return start_date, end_date

    @classmethod
    def get_parquet_file_path(cls, symbol: str, year: int, month: int) -> Path:
        """
        获取parquet文件路径

        Args:
            symbol: 交易对符号
            year: 年份
            month: 月份

        Returns:
            文件路径
        """
        return cls.PARQUET_DATA_1S_DIR / f"{symbol}_{year}-{month:02d}.parquet"

    @classmethod
    def get_test_storage_path(cls) -> Path:
        """获取测试存储路径"""
        return cls.TEST_STORAGE_DIR

    @classmethod
    def get_available_test_symbols(cls) -> list[str]:
        """
        获取可用的测试symbol列表（检查数据文件是否存在）

        Returns:
            可用的symbol列表
        """
        available = []
        for symbol in cls.TEST_SYMBOLS:
            # 检查是否有2024年12月的数据
            file_path = cls.get_parquet_file_path(symbol, 2024, 12)
            if file_path.exists():
                available.append(symbol)
        return (
            available if available else [cls.TEST_SYMBOL]
        )  # 如果没有，至少返回默认symbol
