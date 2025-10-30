#!/usr/bin/env python3
"""
综合特征工程模块
整合所有特征工程模块，提供统一的特征工程接口

包含模块：
1. feature_engineering.py - 基础 + TA-Lib 特征工程
2. feature_engineering_enhanced.py - 增强版特征工程（含WPT/订单流等）
3. dl_sequence_features.py - 深度学习序列特征
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings

warnings.filterwarnings("ignore")

# 导入所有特征工程模块
from .feature_engineering import FeatureEngineer
from .feature_engineering_enhanced import EnhancedFeatureEngineer
from .dl_sequence_features import add_dl_sequence_features


class ComprehensiveFeatureEngineer:
    """综合特征工程器 - 整合所有特征工程模块"""

    def __init__(
        self,
        scaler_type: str = "standard",
        wavelet: str = "db4",
        wpt_level: int = 3,
        hurst_window: int = 100,
        dl_backend: str = "auto",
        dl_seq_length: int = 120,
        dl_d_model: int = 64,
        use_fp16: bool = True,
    ):
        """
        初始化综合特征工程器

        Args:
            scaler_type: 标准化类型 ('standard', 'minmax', 'robust')
            wavelet: 小波类型
            wpt_level: 小波包分解层级
            hurst_window: Hurst指数窗口大小
            dl_backend: 深度学习后端 ('mamba', 'flash_attention', 'transformer', 'auto')
            dl_seq_length: 序列长度
            dl_d_model: 模型维度
            use_fp16: 是否使用FP16混合精度
        """
        self.scaler_type = scaler_type
        self.wavelet = wavelet
        self.wpt_level = wpt_level
        self.hurst_window = hurst_window
        self.dl_backend = dl_backend
        self.dl_seq_length = dl_seq_length
        self.dl_d_model = dl_d_model
        self.use_fp16 = use_fp16

        # 初始化特征工程器
        self.basic_engineer = FeatureEngineer()
        self.enhanced_engineer = EnhancedFeatureEngineer(
            scaler_type=scaler_type,
            wavelet=wavelet,
            wpt_level=wpt_level,
            hurst_window=hurst_window,
        )

        # 特征统计
        self.feature_stats = {}
        self.total_features = 0

    def engineer_all_features(self,
                              data: pd.DataFrame,
                              fit: bool = True) -> pd.DataFrame:
        """
        使用三阶段特征工程生成综合特征

        阶段：
        1. 基础 + TA-Lib 指标
        2. 增强版（WPT/Hurst/Hilbert/光谱/订单流）
        3. 深度学习序列特征
        """
        print("🚀 开始综合特征工程...")
        df = data.copy()
        initial_features = len(df.columns)
        prev_count = initial_features

        # 1. 基础特征工程
        print("  📊 1/3 基础特征工程...")
        df = self.basic_engineer.add_technical_indicators(df)
        basic_features = len(df.columns) - prev_count
        prev_count = len(df.columns)
        print(f"     ✅ 基础特征: {basic_features} 个")

        # 2. 增强版特征工程 (WPT + Hurst + Hilbert + 光谱 + 订单流)
        print("  📊 2/3 增强版特征工程...")
        df = self.enhanced_engineer.add_hurst_features(df)
        df = self.enhanced_engineer.add_wavelet_packet_features(df)
        df = self.enhanced_engineer.add_hilbert_features(df)
        df = self.enhanced_engineer.add_spectral_features(df)
        df = self.enhanced_engineer.add_advanced_derived_features(df)
        df = self.enhanced_engineer.add_order_flow_features(df)
        enhanced_features = len(df.columns) - prev_count
        prev_count = len(df.columns)
        print(f"     ✅ 增强版特征: {enhanced_features} 个")

        # 3. 深度学习序列特征
        print("  📊 3/3 深度学习序列特征...")
        try:
            df = add_dl_sequence_features(
                df,
                backend=self.dl_backend,
                seq_length=self.dl_seq_length,
                d_model=self.dl_d_model,
                use_fp16=self.use_fp16,
            )
            dl_features = len(df.columns) - prev_count
            prev_count = len(df.columns)
            print(f"     ✅ 深度学习特征: {dl_features} 个")
        except Exception as e:
            print(f"     ⚠️ 深度学习特征失败: {e}")
            dl_features = 0

        total_new_features = len(df.columns) - initial_features
        self.total_features = total_new_features

        print(f"\n✅ 综合特征工程完成!")
        print(f"  原始特征: {initial_features} 个")
        print(f"  新增特征: {total_new_features} 个")
        print(f"  总特征数: {len(df.columns)} 个")
        print(f"  特征分布:")
        print(f"    - 基础特征: {basic_features} 个")
        print(f"    - 增强版特征: {enhanced_features} 个")
        print(f"    - 深度学习特征: {dl_features} 个")

        self.feature_stats = {
            "basic_features": basic_features,
            "enhanced_features": enhanced_features,
            "dl_features": dl_features,
            "total_new_features": total_new_features,
            "total_features": len(df.columns),
        }

        return df

    def engineer_features(self,
                          data: pd.DataFrame,
                          fit: bool = True) -> pd.DataFrame:
        """
        为单时间框架数据工程特征

        Args:
            data: 输入数据 (OHLCV)
            fit: 是否拟合标准化器

        Returns:
            工程特征后的DataFrame
        """
        return self.engineer_all_features(data, fit=fit)

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """获取特征列名"""
        exclude_columns = [
            "timestamp", "open", "high", "low", "close", "volume"
        ]
        return [col for col in df.columns if col not in exclude_columns]

    def get_feature_stats(self) -> Dict:
        """获取特征统计信息"""
        return self.feature_stats

    def save_scalers(self, path: str):
        """保存所有标准化器"""
        import pickle

        scalers_data = {
            "enhanced_scalers": self.enhanced_engineer.scalers,
            "feature_stats": self.feature_stats,
        }

        with open(path, "wb") as f:
            pickle.dump(scalers_data, f)
        print(f"✅ 标准化器保存到: {path}")

    def load_scalers(self, path: str):
        """加载所有标准化器"""
        import pickle

        with open(path, "rb") as f:
            scalers_data = pickle.load(f)

        self.enhanced_engineer.scalers = scalers_data.get(
            "enhanced_scalers", {})
        self.feature_stats = scalers_data.get("feature_stats", {})

        print(f"✅ 标准化器从 {path} 加载完成")


def create_comprehensive_feature_engineer(
        scaler_type: str = "standard",
        **kwargs) -> ComprehensiveFeatureEngineer:
    """
    创建综合特征工程器的便捷函数

    Args:
        scaler_type: 标准化类型
        **kwargs: 其他参数

    Returns:
        ComprehensiveFeatureEngineer实例
    """
    return ComprehensiveFeatureEngineer(scaler_type=scaler_type, **kwargs)


# 向后兼容的函数
def engineer_features(
    df: pd.DataFrame,
    feature_engineer: Optional[ComprehensiveFeatureEngineer] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, ComprehensiveFeatureEngineer]:
    """
    向后兼容的特征工程函数

    Args:
        df: 输入数据
        feature_engineer: 特征工程器实例
        fit: 是否拟合

    Returns:
        (工程后的DataFrame, 特征工程器)
    """
    if feature_engineer is None:
        feature_engineer = ComprehensiveFeatureEngineer()

    engineered_df = feature_engineer.engineer_features(df, fit=fit)
    return engineered_df, feature_engineer


def add_dl_time_series_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    向后兼容的深度学习序列特征函数

    Args:
        df: 输入数据
        **kwargs: 其他参数

    Returns:
        添加了深度学习特征的DataFrame
    """
    return add_dl_sequence_features(df, **kwargs)


if __name__ == "__main__":
    # 测试综合特征工程
    print("🧪 测试综合特征工程...")

    # 创建测试数据
    dates = pd.date_range("2024-01-01", periods=1000, freq="5T")
    test_data = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.randn(1000).cumsum() + 100,
        "high": np.random.randn(1000).cumsum() + 105,
        "low": np.random.randn(1000).cumsum() + 95,
        "close": np.random.randn(1000).cumsum() + 100,
        "volume": np.random.randint(1000, 10000, 1000),
    })

    # 测试综合特征工程
    engineer = ComprehensiveFeatureEngineer()
    result_df = engineer.engineer_all_features(test_data)

    print(f"\n🎉 测试完成!")
    print(f"  输入特征: {len(test_data.columns)} 个")
    print(f"  输出特征: {len(result_df.columns)} 个")
    print(f"  新增特征: {len(result_df.columns) - len(test_data.columns)} 个")
