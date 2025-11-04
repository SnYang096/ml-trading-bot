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
from .feature_engineering_talib import TalibFeatureEngineer
from .dl_sequence_features import add_dl_sequence_features
from .baseline_feature_engineering import BaselineFeatureEngineer, engineer_baseline_features, get_baseline_feature_columns


class ComprehensiveFeatureEngineer:
    """综合特征工程器 - 整合所有特征工程模块
    
    支持特征切换：
    - baseline: 基线SR+压缩特征
    - default: 默认传统指标（TA-Lib + base_indicators，推荐使用）
    - enhanced: 增强版特征（WPT/Hurst/Hilbert/光谱/订单流）
    - dl_sequence: 深度学习序列特征
    - comprehensive: 所有特征合并
    """

    def __init__(
        self,
        feature_types: str = "comprehensive",
        scaler_type: str = "standard",
        wavelet: str = "db4",
        wpt_level: int = 3,
        hurst_window: int = 100,
        dl_backend: str = "auto",
        dl_seq_length: int = 120,
        dl_d_model: int = 64,
        use_fp16: bool = True,
        baseline_percentile_window: int = 288,
        baseline_compression_threshold_pct: float = 0.2,
    ):
        """
        初始化综合特征工程器

        Args:
            feature_types: 特征类型，支持：
                - 'baseline': 只用基线特征
                - 'default': 默认传统指标（TA-Lib + base_indicators，推荐）
                - 'enhanced': 只用增强版特征
                - 'dl_sequence': 只用深度学习序列特征
                - 'comprehensive': 所有特征合并
                - 逗号分隔的组合: 'baseline,default' 等
            scaler_type: 标准化类型 ('standard', 'minmax', 'robust')
            wavelet: 小波类型
            wpt_level: 小波包分解层级
            hurst_window: Hurst指数窗口大小
            dl_backend: 深度学习后端 ('mamba', 'flash_attention', 'transformer', 'auto')
            dl_seq_length: 序列长度
            dl_d_model: 模型维度
            use_fp16: 是否使用FP16混合精度
            baseline_percentile_window: 基线特征百分位窗口
            baseline_compression_threshold_pct: 基线压缩阈值百分比
        """
        self.feature_types = feature_types
        self.scaler_type = scaler_type
        self.wavelet = wavelet
        self.wpt_level = wpt_level
        self.hurst_window = hurst_window
        self.dl_backend = dl_backend
        self.dl_seq_length = dl_seq_length
        self.dl_d_model = dl_d_model
        self.use_fp16 = use_fp16
        self.baseline_percentile_window = baseline_percentile_window
        self.baseline_compression_threshold_pct = baseline_compression_threshold_pct

        # 解析特征类型
        if feature_types == "comprehensive":
            self.use_baseline = True
            self.use_default = True  # default = talib + base_indicators
            self.use_enhanced = True
            self.use_dl_sequence = True
        else:
            feature_list = [f.strip() for f in feature_types.split(",")]

            self.use_baseline = "baseline" in feature_list
            self.use_default = "default" in feature_list  # default = FeatureEngineer (talib + base_indicators)
            self.use_enhanced = "enhanced" in feature_list
            self.use_dl_sequence = "dl_sequence" in feature_list

        # 初始化特征工程器（按需）
        self.basic_engineer = None  # FeatureEngineer (talib + base_indicators)
        self.enhanced_engineer = None
        self.baseline_engineer = None

        # 默认传统指标：使用 FeatureEngineer (talib + base_indicators)
        if self.use_default or self.use_enhanced or feature_types == "comprehensive":
            self.basic_engineer = FeatureEngineer()

        if self.use_enhanced or feature_types == "comprehensive":
            self.enhanced_engineer = EnhancedFeatureEngineer(
                scaler_type=scaler_type,
                wavelet=wavelet,
                wpt_level=wpt_level,
                hurst_window=hurst_window,
            )

        if self.use_baseline or feature_types == "comprehensive":
            self.baseline_engineer = BaselineFeatureEngineer(
                percentile_window=baseline_percentile_window,
                compression_threshold_pct=baseline_compression_threshold_pct,
            )

        # 特征统计
        self.feature_stats = {}
        self.total_features = 0

    def engineer_all_features(self,
                              data: pd.DataFrame,
                              fit: bool = True) -> pd.DataFrame:
        """
        使用可选的特征工程模块生成特征

        支持的特征类型：
        - baseline: 基线SR+压缩特征
        - default: 默认传统指标（TA-Lib + base_indicators，推荐）
        - enhanced: 增强版特征（WPT/Hurst/Hilbert/光谱/订单流）
        - dl_sequence: 深度学习序列特征
        - comprehensive: 所有特征合并
        """
        print(f"🚀 开始特征工程 (feature_types: {self.feature_types})...")
        df = data.copy()
        initial_features = len(df.columns)
        prev_count = initial_features

        baseline_features = 0
        default_features = 0  # TA-Lib + base_indicators
        enhanced_features = 0
        dl_features = 0

        # 1. 基线特征工程
        if self.use_baseline:
            print("  📊 Baseline特征工程...")
            try:
                df, self.baseline_engineer = engineer_baseline_features(
                    df, self.baseline_engineer, fit=fit)
                baseline_features = len(df.columns) - prev_count
                prev_count = len(df.columns)
                print(f"     ✅ Baseline特征: {baseline_features} 个")
            except Exception as e:
                print(f"     ⚠️  Baseline特征失败: {e}")

        # 2. 默认传统指标特征工程（TA-Lib + base_indicators）
        if self.use_default:
            print("  📊 默认传统指标特征工程（TA-Lib + base_indicators）...")
            try:
                if self.basic_engineer is None:
                    self.basic_engineer = FeatureEngineer()
                df = self.basic_engineer.add_technical_indicators(df)
                default_features = len(df.columns) - prev_count
                prev_count = len(df.columns)
                print(f"     ✅ 默认传统指标特征: {default_features} 个")
            except Exception as e:
                print(f"     ⚠️  默认传统指标特征失败: {e}")

        # 3. 增强版特征工程 (WPT + Hurst + Hilbert + 光谱 + 订单流)
        if self.use_enhanced:
            print("  📊 增强版特征工程...")
            try:
                if self.enhanced_engineer is None:
                    self.enhanced_engineer = EnhancedFeatureEngineer(
                        scaler_type=self.scaler_type,
                        wavelet=self.wavelet,
                        wpt_level=self.wpt_level,
                        hurst_window=self.hurst_window,
                    )
                df = self.enhanced_engineer.add_hurst_features(df)
                df = self.enhanced_engineer.add_wavelet_packet_features(df)
                df = self.enhanced_engineer.add_hilbert_features(df)
                df = self.enhanced_engineer.add_spectral_features(df)
                df = self.enhanced_engineer.add_advanced_derived_features(df)
                df = self.enhanced_engineer.add_order_flow_features(df)
                enhanced_features = len(df.columns) - prev_count
                prev_count = len(df.columns)
                print(f"     ✅ 增强版特征: {enhanced_features} 个")
            except Exception as e:
                print(f"     ⚠️  增强版特征失败: {e}")

        # 4. 深度学习序列特征
        if self.use_dl_sequence:
            print("  📊 深度学习序列特征...")
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
                print(f"     ⚠️  深度学习特征失败: {e}")

        total_new_features = len(df.columns) - initial_features
        self.total_features = total_new_features

        print(f"\n✅ 特征工程完成!")
        print(f"  原始特征: {initial_features} 个")
        print(f"  新增特征: {total_new_features} 个")
        print(f"  总特征数: {len(df.columns)} 个")
        if self.use_baseline or self.use_default or self.use_enhanced or self.use_dl_sequence:
            print(f"  特征分布:")
            if self.use_baseline:
                print(f"    - Baseline特征: {baseline_features} 个")
            if self.use_default:
                print(
                    f"    - 默认传统指标特征（TA-Lib + base_indicators）: {default_features} 个"
                )
            if self.use_enhanced:
                print(f"    - 增强版特征: {enhanced_features} 个")
            if self.use_dl_sequence:
                print(f"    - 深度学习特征: {dl_features} 个")

        self.feature_stats = {
            "baseline_features": baseline_features,
            "default_features": default_features,  # TA-Lib + base_indicators
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
            "timestamp", "open", "high", "low", "close", "volume", "signal",
            "binary_signal", "future_return", "symbol"
        ]
        # 排除多周期标签列
        exclude_patterns = ["signal_", "binary_signal_", "future_return_"]
        feature_cols = []
        for col in df.columns:
            if col not in exclude_columns:
                # 检查是否匹配排除模式
                if not any(
                        col.startswith(pattern)
                        for pattern in exclude_patterns):
                    # 只包含数值类型的列
                    if pd.api.types.is_numeric_dtype(df[col]):
                        feature_cols.append(col)
                    else:
                        # Debug: 打印被排除的非数值列
                        if col not in ["timestamp", "symbol"]:  # 这些已知是非数值的
                            print(
                                f"   ⚠️  Warning: Excluding non-numeric column '{col}' (dtype: {df[col].dtype})"
                            )
        return feature_cols

    def get_feature_stats(self) -> Dict:
        """获取特征统计信息"""
        return self.feature_stats

    def save_scalers(self, path: str):
        """保存所有标准化器"""
        import pickle

        scalers_data = {
            "feature_stats": self.feature_stats,
            "feature_types": self.feature_types,
        }

        # 保存增强版标准化器
        if self.enhanced_engineer is not None:
            scalers_data["enhanced_scalers"] = self.enhanced_engineer.scalers

        # 保存基线标准化器
        if self.baseline_engineer is not None:
            baseline_scalers_data = {
                "fitted_atr_quantiles":
                self.baseline_engineer._fitted_atr_quantiles,
                "fitted_vol_quantiles":
                self.baseline_engineer._fitted_vol_quantiles,
                "percentile_window":
                self.baseline_engineer.percentile_window,
                "compression_threshold_pct":
                self.baseline_engineer.compression_threshold_pct,
            }
            scalers_data["baseline_scalers"] = baseline_scalers_data

        with open(path, "wb") as f:
            pickle.dump(scalers_data, f)
        print(f"✅ 标准化器保存到: {path}")

    def load_scalers(self, path: str):
        """加载所有标准化器"""
        import pickle

        with open(path, "rb") as f:
            scalers_data = pickle.load(f)

        # 加载增强版标准化器
        if self.enhanced_engineer is not None and "enhanced_scalers" in scalers_data:
            self.enhanced_engineer.scalers = scalers_data.get(
                "enhanced_scalers", {})

        # 加载基线标准化器
        if self.baseline_engineer is not None and "baseline_scalers" in scalers_data:
            baseline_scalers = scalers_data.get("baseline_scalers", {})
            self.baseline_engineer._fitted_atr_quantiles = baseline_scalers.get(
                "fitted_atr_quantiles", None)
            self.baseline_engineer._fitted_vol_quantiles = baseline_scalers.get(
                "fitted_vol_quantiles", None)
            self.baseline_engineer.percentile_window = baseline_scalers.get(
                "percentile_window", 288)
            self.baseline_engineer.compression_threshold_pct = baseline_scalers.get(
                "compression_threshold_pct", 0.2)

        self.feature_stats = scalers_data.get("feature_stats", {})

        print(f"✅ 标准化器从 {path} 加载完成")


def create_comprehensive_feature_engineer(
        feature_types: str = "comprehensive",
        scaler_type: str = "standard",
        **kwargs) -> ComprehensiveFeatureEngineer:
    """
    创建综合特征工程器的便捷函数

    Args:
        feature_types: 特征类型 ('baseline', 'talib', 'enhanced', 'dl_sequence', 'comprehensive', 或逗号分隔的组合)
        scaler_type: 标准化类型
        **kwargs: 其他参数

    Returns:
        ComprehensiveFeatureEngineer实例
    """
    return ComprehensiveFeatureEngineer(feature_types=feature_types,
                                        scaler_type=scaler_type,
                                        **kwargs)


def engineer_features_by_type(
    df: pd.DataFrame,
    feature_types: str = "baseline",
    feature_engineer: Optional[ComprehensiveFeatureEngineer] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, ComprehensiveFeatureEngineer]:
    """
    根据特征类型工程特征的便捷函数

    Args:
        df: 输入数据
        feature_types: 特征类型 ('baseline', 'default', 'enhanced', 'dl_sequence', 'comprehensive'，或逗号分隔的组合)
        feature_engineer: 特征工程器实例（如果为None，会创建新的）
        fit: 是否拟合

    Returns:
        (工程后的DataFrame, 特征工程器)
    """
    if feature_engineer is None:
        feature_engineer = ComprehensiveFeatureEngineer(
            feature_types=feature_types)
    elif feature_engineer.feature_types != feature_types:
        # 如果特征类型不匹配，创建新的
        feature_engineer = ComprehensiveFeatureEngineer(
            feature_types=feature_types)

    engineered_df = feature_engineer.engineer_all_features(df, fit=fit)
    return engineered_df, feature_engineer


def get_feature_columns_by_type(df: pd.DataFrame,
                                feature_types: str = "baseline") -> List[str]:
    """
    根据特征类型获取特征列名

    Args:
        df: 数据DataFrame
        feature_types: 特征类型

    Returns:
        特征列名列表
    """
    if feature_types == "baseline":
        return get_baseline_feature_columns(df)
    else:
        # Use ComprehensiveFeatureEngineer's get_feature_columns method
        # Create a temporary engineer to use its method
        temp_engineer = ComprehensiveFeatureEngineer(
            feature_types=feature_types)
        feature_cols = temp_engineer.get_feature_columns(df)

        # Debug: 如果没有特征，打印可用列
        if not feature_cols:
            print(
                f"   ⚠️  Warning in get_feature_columns_by_type: No features found for feature_types='{feature_types}'"
            )
            print(f"   Available columns: {list(df.columns)[:30]}...")
            print(f"   Total columns: {len(df.columns)}")

        return feature_cols


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
