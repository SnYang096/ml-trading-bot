"""
实时流特征加载集成示例

展示如何在实时流中复用 YAML 特征加载流程，包括：
1. 策略配置加载
2. 特征依赖预编译
3. 历史窗口管理
4. 增量特征计算
"""

from __future__ import annotations

import pandas as pd
from typing import List, Optional, Dict, Any
from pathlib import Path

from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.strategy_config import StrategyConfigLoader


class RealtimeFeatureManager:
    """
    实时流特征管理器

    封装了实时流场景下的特征加载逻辑，包括：
    - 历史窗口维护
    - 特征依赖预编译
    - 增量特征计算
    """

    def __init__(
        self,
        strategy_name: str,
        history_window: int = 1000,
        config_base_path: str = "config/strategies",
    ):
        """
        初始化实时流特征管理器

        Args:
            strategy_name: 策略名称（如 "sr_reversal"）
            history_window: 维护的历史数据窗口大小
            config_base_path: 策略配置基础路径
        """
        self.strategy_name = strategy_name
        self.history_window = history_window

        # 1. 加载策略配置
        config_path = Path(config_base_path) / strategy_name
        config_loader = StrategyConfigLoader(config_path)
        self.strategy_config = config_loader.load()

        # 2. 初始化特征加载器（实时流模式：串行计算，禁用磁盘缓存）
        self.feature_loader = StrategyFeatureLoader(
            max_workers=1,  # 实时流中串行计算，避免并行开销
            use_disk_cache=False,  # 禁用磁盘缓存（实时流中缓存命中率低）
            use_memory_cache=True,  # 保留内存缓存（可能有用）
        )

        # 3. 预编译特征依赖顺序（避免每次调用都解析）
        requested_features = self.strategy_config.features.requested_features
        self.dependency_order = self.feature_loader.resolve_dependencies(
            requested_features
        )
        print(f"✅ 预编译特征依赖顺序: {len(self.dependency_order)} 个特征")

        # 4. 历史数据窗口（用于维护历史数据）
        self.history_df: Optional[pd.DataFrame] = None

        # 5. 特征列名缓存（避免重复计算）
        self.feature_columns: Optional[List[str]] = None

    def append_bar(self, new_bar: pd.DataFrame) -> None:
        """
        追加新的K线数据到历史窗口

        Args:
            new_bar: 新的K线数据（单条或多条）
        """
        if self.history_df is None:
            self.history_df = new_bar.copy()
        else:
            # 合并新数据
            self.history_df = pd.concat([self.history_df, new_bar], ignore_index=True)

        # 维护滑动窗口
        if len(self.history_df) > self.history_window:
            self.history_df = self.history_df.tail(self.history_window).reset_index(
                drop=True
            )

    def compute_features(self, new_bar: pd.DataFrame) -> pd.DataFrame:
        """
        计算新数据的特征

        Args:
            new_bar: 新的K线数据（单条或多条）

        Returns:
            包含特征的 DataFrame（只包含新数据的行）
        """
        # 1. 追加新数据到历史窗口
        self.append_bar(new_bar)

        # 2. 使用历史窗口计算特征（某些特征需要历史数据）
        # 注意：fit=False，实时流中始终使用已训练的参数
        df_with_features = self.feature_loader.load_features_from_requested(
            self.history_df,
            self.strategy_config.features.requested_features,
            fit=False,  # ⚠️ 实时流中强制 False，使用训练时保存的参数
        )

        # 3. 缓存特征列名（首次计算时）
        if self.feature_columns is None:
            base_columns = {
                "timestamp",
                "datetime",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "symbol",
                "_symbol",
            }
            self.feature_columns = [
                col for col in df_with_features.columns if col not in base_columns
            ]
            print(f"✅ 特征列已识别: {len(self.feature_columns)} 个特征")

        # 4. 只返回新数据对应的特征行
        new_bar_count = len(new_bar)
        latest_features = df_with_features.tail(new_bar_count)

        return latest_features

    def get_latest_features(self) -> Optional[pd.DataFrame]:
        """
        获取最新一条数据的特征（用于信号生成）

        Returns:
            最新一条数据的特征 DataFrame，如果没有数据则返回 None
        """
        if self.history_df is None or len(self.history_df) == 0:
            return None

        # 重新计算特征（确保使用最新历史数据）
        df_with_features = self.feature_loader.load_features_from_requested(
            self.history_df,
            self.strategy_config.features.requested_features,
            fit=False,
        )

        # 返回最新一条
        return df_with_features.tail(1)

    def get_feature_columns(self) -> List[str]:
        """获取特征列名列表"""
        if self.feature_columns is None:
            # 如果还没有计算过，先计算一次
            if self.history_df is not None and len(self.history_df) > 0:
                _ = self.get_latest_features()
        return self.feature_columns or []

    def reset_history(self) -> None:
        """清空历史数据窗口（用于重新开始）"""
        self.history_df = None
        self.feature_columns = None
        print("🔄 历史数据窗口已清空")


# ============================================================================
# 使用示例
# ============================================================================


def example_usage():
    """使用示例"""

    # 1. 初始化实时流特征管理器
    feature_manager = RealtimeFeatureManager(
        strategy_name="sr_reversal",
        history_window=1000,  # 维护最近1000条K线
    )

    # 2. 模拟接收实时K线数据
    # 假设从 websocket 接收到新的K线
    new_bar = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp.now()],
            "open": [50000.0],
            "high": [50100.0],
            "low": [49900.0],
            "close": [50050.0],
            "volume": [100.0],
            "symbol": ["BTCUSDT"],
        }
    )

    # 3. 计算特征
    features_df = feature_manager.compute_features(new_bar)

    # 4. 获取最新特征（用于信号生成）
    latest_features = feature_manager.get_latest_features()

    if latest_features is not None:
        print(f"✅ 特征计算完成，共 {len(latest_features.columns)} 列")
        print(f"特征列: {feature_manager.get_feature_columns()[:5]}...")  # 显示前5个

    return feature_manager, latest_features


# ============================================================================
# 与 nautilus_stub.py 集成示例
# ============================================================================


class EnhancedNautilusLiveStub:
    """
    增强版的 Nautilus Live Stub，集成了特征加载
    """

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

        # 初始化特征管理器
        self.feature_manager = RealtimeFeatureManager(
            strategy_name=strategy_name,
            history_window=1000,
        )

    def _convert_payload_to_dataframe(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """
        将 websocket payload 转换为 DataFrame

        根据实际的数据格式调整此函数
        """
        # 示例：假设 payload 包含 K线数据
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp.fromtimestamp(payload.get("t", 0) / 1000)],
                "open": [payload.get("o", 0.0)],
                "high": [payload.get("h", 0.0)],
                "low": [payload.get("l", 0.0)],
                "close": [payload.get("c", 0.0)],
                "volume": [payload.get("v", 0.0)],
                "symbol": [payload.get("s", "UNKNOWN")],
            }
        )

    def _handle_market_data(self, venue: str, payload: Dict[str, Any]) -> None:
        """
        处理实时市场数据，计算特征并生成信号
        """
        try:
            # 1. 转换为 DataFrame
            new_bar = self._convert_payload_to_dataframe(payload)

            # 2. 计算特征
            features_df = self.feature_manager.compute_features(new_bar)

            # 3. 获取最新特征
            latest_features = self.feature_manager.get_latest_features()

            if latest_features is not None:
                # 4. 生成交易信号（这里需要实现实际的信号生成逻辑）
                signal = self._generate_signal(latest_features)

                # 5. 执行交易（如果需要）
                if signal:
                    self._execute_trade(signal, latest_features)

        except Exception as e:
            print(f"❌ 处理市场数据时出错: {e}")
            import traceback

            traceback.print_exc()

    def _generate_signal(self, features_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        根据特征生成交易信号

        这里需要加载训练好的模型，进行预测
        """
        # TODO: 实现实际的信号生成逻辑
        # 1. 加载训练好的模型
        # 2. 使用模型预测
        # 3. 生成交易信号

        return None

    def _execute_trade(self, signal: Dict[str, Any], features_df: pd.DataFrame) -> None:
        """
        执行交易

        这里需要实现实际的交易逻辑
        """
        # TODO: 实现实际的交易执行逻辑
        pass


if __name__ == "__main__":
    # 运行示例
    feature_manager, latest_features = example_usage()

    if latest_features is not None:
        print("\n✅ 实时流特征加载示例运行成功！")
        print(f"特征数量: {len(latest_features.columns)}")
    else:
        print("⚠️ 需要更多历史数据才能计算特征")
