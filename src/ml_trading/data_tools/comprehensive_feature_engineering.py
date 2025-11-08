#!/usr/bin/env python3
"""
综合特征工程模块（无归一化版本）
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
from .baseline_feature_engineering import (
    BaselineFeatureEngineer,
    engineer_baseline_features,
    get_baseline_feature_columns,
)


class ComprehensiveFeatureEngineer:
    """综合特征工程器 - 整合所有特征工程模块（无归一化版本）"""

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
            feature_types: 特征类型字符串，支持逗号组合或 'comprehensive'
            scaler_type: 标准化类型（保留参数以兼容接口，不执行归一化）
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
        self.feature_types = feature_types or "comprehensive"
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

        feature_list = [
            f.strip() for f in self.feature_types.split(",") if f.strip()
        ]
        self._use_comprehensive = (self.feature_types == "comprehensive"
                                   or not feature_list)

        if self._use_comprehensive:
            self.use_baseline = True
            self.use_default = True
            self.use_enhanced = True
            self.use_dl_sequence = True
            self.use_hurst = True
            self.use_wavelet = True
            self.use_hilbert = True
            self.use_spectral = True
            self.use_order_flow = True
        else:
            self.use_baseline = "baseline" in feature_list
            self.use_default = "default" in feature_list
            self.use_enhanced = "enhanced" in feature_list
            self.use_dl_sequence = "dl_sequence" in feature_list

            if self.use_enhanced:
                self.use_hurst = True
                self.use_wavelet = True
                self.use_hilbert = True
                self.use_spectral = True
                self.use_order_flow = True
            else:
                self.use_hurst = "hurst" in feature_list
                self.use_wavelet = "wavelet" in feature_list
                self.use_hilbert = "hilbert" in feature_list
                self.use_spectral = "spectral" in feature_list
                self.use_order_flow = "order_flow" in feature_list

        self.basic_engineer: Optional[FeatureEngineer] = (
            FeatureEngineer() if
            (self.use_default or self.use_enhanced or self.use_hurst
             or self.use_wavelet or self.use_hilbert or self.use_spectral
             or self.use_order_flow) else None)

        use_any_enhanced = (self.use_enhanced or self.use_hurst
                            or self.use_wavelet or self.use_hilbert
                            or self.use_spectral or self.use_order_flow)
        self.enhanced_engineer: Optional[EnhancedFeatureEngineer] = (
            EnhancedFeatureEngineer(
                scaler_type=scaler_type,
                wavelet=wavelet,
                wpt_level=wpt_level,
                hurst_window=hurst_window,
            ) if use_any_enhanced else None)

        self.baseline_engineer: Optional[BaselineFeatureEngineer] = (
            BaselineFeatureEngineer(
                percentile_window=baseline_percentile_window,
                compression_threshold_pct=baseline_compression_threshold_pct,
            ) if self.use_baseline else None)

        # 特征统计
        self.feature_stats: Dict[str, int] = {}
        self.total_features = 0

    def engineer_all_features(self,
                              data: pd.DataFrame,
                              fit: bool = True) -> pd.DataFrame:
        """
        根据配置组合特征工程模块，生成综合特征
        """
        print(f"🚀 开始综合特征工程 (feature_types: {self.feature_types})...")
        df = data.copy()
        initial_features = len(df.columns)
        prev_count = initial_features

        baseline_features = 0
        default_features = 0
        enhanced_features = 0
        dl_features = 0

        if self.use_baseline and self.baseline_engineer is not None:
            print("  📊 Baseline特征工程...")
            try:
                df, self.baseline_engineer = engineer_baseline_features(
                    df,
                    self.baseline_engineer,
                    fit=fit,
                )
                baseline_features = len(df.columns) - prev_count
                prev_count = len(df.columns)
                print(f"     ✅ Baseline特征: {baseline_features} 个")
            except Exception as exc:  # noqa: BLE001
                print(f"     ⚠️  Baseline特征失败: {exc}")

        if self.use_default and self.basic_engineer is not None:
            print("  📊 默认传统指标特征工程（TA-Lib + base_indicators）...")
            try:
                df = self.basic_engineer.add_technical_indicators(df)
                default_features = len(df.columns) - prev_count
                prev_count = len(df.columns)
                print(f"     ✅ 默认特征: {default_features} 个")
            except Exception as exc:  # noqa: BLE001
                print(f"     ⚠️  默认特征失败: {exc}")

        use_any_enhanced = (self.use_enhanced or self.use_hurst
                            or self.use_wavelet or self.use_hilbert
                            or self.use_spectral or self.use_order_flow)
        if use_any_enhanced and self.enhanced_engineer is not None:
            print("  📊 增强版特征工程...")
            try:
                if self.use_hurst or self.use_enhanced:
                    df = self.enhanced_engineer.add_hurst_features(df)
                if self.use_wavelet or self.use_enhanced:
                    df = self.enhanced_engineer.add_wavelet_packet_features(df)
                if self.use_hilbert or self.use_enhanced:
                    df = self.enhanced_engineer.add_hilbert_features(df)
                if self.use_spectral or self.use_enhanced:
                    df = self.enhanced_engineer.add_spectral_features(df)
                if self.use_enhanced or self.use_order_flow:
                    df = self.enhanced_engineer.add_advanced_derived_features(
                        df)
                    df = self.enhanced_engineer.add_order_flow_features(df)

                enhanced_features = len(df.columns) - prev_count
                prev_count = len(df.columns)
                print(f"     ✅ 增强版特征: {enhanced_features} 个")
            except Exception as exc:  # noqa: BLE001
                print(f"     ⚠️  增强版特征失败: {exc}")

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
            except Exception as exc:  # noqa: BLE001
                print(f"     ⚠️ 深度学习特征失败: {exc}")
                dl_features = 0

        total_new_features = len(df.columns) - initial_features
        self.total_features = total_new_features

        print(f"\n✅ 综合特征工程完成!")
        print(f"  原始特征: {initial_features} 个")
        print(f"  新增特征: {total_new_features} 个")
        print(f"  总特征数: {len(df.columns)} 个")
        print(f"  特征分布:")
        print(f"    - Baseline特征: {baseline_features} 个")
        print(f"    - 默认特征: {default_features} 个")
        print(f"    - 增强版特征: {enhanced_features} 个")
        print(f"    - 深度学习特征: {dl_features} 个")

        self.feature_stats = {
            "baseline_features": baseline_features,
            "default_features": default_features,
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
        """保存所有标准化器（无归一化版本仅保留接口）"""
        import pickle

        scalers_data = {
            "enhanced_scalers": getattr(self.enhanced_engineer, "scalers", {}),
            "feature_stats": self.feature_stats,
        }

        with open(path, "wb") as f:
            pickle.dump(scalers_data, f)
        print(f"✅ 标准化器保存到: {path}")

    def load_scalers(self, path: str):
        """加载所有标准化器（无归一化版本仅保留接口）"""
        import pickle

        with open(path, "rb") as f:
            scalers_data = pickle.load(f)

        if self.enhanced_engineer is not None:
            self.enhanced_engineer.scalers = scalers_data.get(
                "enhanced_scalers", {})
        self.feature_stats = scalers_data.get("feature_stats", {})

        print(f"✅ 标准化器从 {path} 加载完成")


def create_comprehensive_feature_engineer(
        scaler_type: str = "standard",
        **kwargs) -> ComprehensiveFeatureEngineer:
    """创建综合特征工程器的便捷函数"""
    return ComprehensiveFeatureEngineer(scaler_type=scaler_type, **kwargs)


# 向后兼容的函数
def engineer_features(
    df: pd.DataFrame,
    feature_engineer: Optional[ComprehensiveFeatureEngineer] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, ComprehensiveFeatureEngineer]:
    """
    向后兼容的特征工程函数
    """
    if feature_engineer is None:
        feature_engineer = ComprehensiveFeatureEngineer()

    engineered_df = feature_engineer.engineer_features(df, fit=fit)
    return engineered_df, feature_engineer


def add_dl_time_series_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    向后兼容的深度学习序列特征函数
    """
    return add_dl_sequence_features(df, **kwargs)


def get_feature_columns_by_type(df: pd.DataFrame,
                                feature_types: str = "baseline") -> List[str]:
    """
    根据特征类型获取特征列名
    """
    if feature_types == "baseline":
        return get_baseline_feature_columns(df)

    temp_engineer = ComprehensiveFeatureEngineer(feature_types=feature_types)
    feature_cols = temp_engineer.get_feature_columns(df)

    if not feature_cols:
        print(
            f"   ⚠️  Warning in get_feature_columns_by_type (no_normal): no features found for feature_types='{feature_types}'"
        )
        print(f"   Available columns: {list(df.columns)[:30]}...")
        print(f"   Total columns: {len(df.columns)}")

    return feature_cols


if __name__ == "__main__":
    # 测试综合特征工程
    print("🧪 测试综合特征工程 (no_normal)...")

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

    engineer = ComprehensiveFeatureEngineer()
    result_df = engineer.engineer_all_features(test_data)

    print(f"\n🎉 测试完成!")
    print(f"  输入特征: {len(test_data.columns)} 个")
    print(f"  输出特征: {len(result_df.columns)} 个")
    print(f"  新增特征: {len(result_df.columns) - len(test_data.columns)} 个")
