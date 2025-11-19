#!/usr/bin/env python3
"""
综合特征工程模块
整合所有特征工程模块，提供统一的特征工程接口

包含模块：
1. feature_engineering.py - 基础 + TA-Lib 特征工程
2. feature_engineering_enhanced.py - 增强版特征工程（含WPT/订单流等）
3. dl_sequence_features.py - 深度学习序列特征
4. alpha_factors/alpha101_feature_engineer.py - WorldQuant Alpha101 因子集
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
from .baseline_features import (
    BaselineFeatureEngineer,
    engineer_baseline_features,
    get_baseline_feature_columns,
)
from .alpha_factors.alpha101_feature_engineer import Alpha101FeatureEngineer


class ComprehensiveFeatureEngineer:
    """综合特征工程器 - 整合所有特征工程模块

    支持特征切换：
    - baseline: 基线SR+压缩特征
    - default: 默认传统指标（TA-Lib + base_indicators，推荐使用）
    - enhanced: 增强版特征（包含所有子模块：Hurst/Wavelet/Hilbert/Spectral/OrderFlow）
    - hurst: Hurst指数特征（趋势持续性分析）
    - wavelet: 小波包变换特征（精细频带分解）
    - hilbert: Hilbert变换特征（瞬时频率/相位分析）
    - spectral: 光谱分析特征（频域特征）
    - order_flow: 订单流特征（CVD/订单流不平衡等）
    - dl_sequence: 深度学习序列特征
    - alpha101: WorldQuant Alpha101 因子集
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
                - 'enhanced': 增强版特征（包含所有子模块：Hurst/Wavelet/Hilbert/Spectral/OrderFlow）
                - 'hurst': 只用Hurst指数特征
                - 'wavelet': 只用小波包变换特征
                - 'hilbert': 只用Hilbert变换特征
                - 'spectral': 只用光谱分析特征
                - 'order_flow': 只用订单流特征
                - 'dl_sequence': 只用深度学习序列特征
                - 'alpha101': 只用 WorldQuant Alpha101 因子集
                - 'comprehensive': 所有特征合并
                - 逗号分隔的组合: 'baseline,alpha101,hurst' 等
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
            self.use_alpha101 = True
            # 如果使用 comprehensive 或 enhanced，默认启用所有子模块
            self.use_hurst = True
            self.use_wavelet = True
            self.use_hilbert = True
            self.use_spectral = True
            self.use_order_flow = True
        else:
            feature_list = [f.strip() for f in feature_types.split(",")]

            # Map 'technical' to 'default' for backward compatibility
            if "technical" in feature_list:
                feature_list.append("default")
                feature_list = list(set(feature_list))  # Remove duplicates

            self.use_baseline = "baseline" in feature_list
            self.use_default = (
                "default" in feature_list
            )  # default = FeatureEngineer (talib + base_indicators)
            self.use_enhanced = "enhanced" in feature_list
            self.use_dl_sequence = "dl_sequence" in feature_list
            self.use_alpha101 = "alpha101" in feature_list

            # 细粒度 enhanced 子模块控制
            # 如果指定了 enhanced，默认启用所有子模块
            if self.use_enhanced:
                self.use_hurst = True
                self.use_wavelet = True
                self.use_hilbert = True
                self.use_spectral = True
                self.use_order_flow = True
            else:
                # 否则根据单独指定的子模块启用
                self.use_hurst = "hurst" in feature_list
                self.use_wavelet = "wavelet" in feature_list
                self.use_hilbert = "hilbert" in feature_list
                self.use_spectral = "spectral" in feature_list
                self.use_order_flow = "order_flow" in feature_list

        # 初始化特征工程器（按需）
        self.basic_engineer = None  # FeatureEngineer (talib + base_indicators)
        self.dl_sequence_extractor = None  # DeepLearningSequenceExtractor (保存状态)
        self.enhanced_engineer = None
        self.baseline_engineer = None
        self.alpha101_engineer: Optional[Alpha101FeatureEngineer] = None

        # 默认传统指标：使用 FeatureEngineer (talib + base_indicators)
        if (
            self.use_default
            or self.use_enhanced
            or self.use_hurst
            or self.use_wavelet
            or self.use_hilbert
            or self.use_spectral
            or self.use_order_flow
            or feature_types == "comprehensive"
        ):
            self.basic_engineer = FeatureEngineer()

        # 如果使用任何 enhanced 子模块，初始化 EnhancedFeatureEngineer
        if (
            self.use_enhanced
            or self.use_hurst
            or self.use_wavelet
            or self.use_hilbert
            or self.use_spectral
            or self.use_order_flow
            or feature_types == "comprehensive"
        ):
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
        if self.use_alpha101:
            self.alpha101_engineer = Alpha101FeatureEngineer()

        # 特征统计
        self.feature_stats = {}
        self.total_features = 0

    def engineer_all_features(
        self,
        data: pd.DataFrame,
        fit: bool = True,
        required_features: Optional[set] = None,
    ) -> pd.DataFrame:
        """
        使用可选的特征工程模块生成特征

        支持的特征类型：
        - baseline: 基线SR+压缩特征
        - default: 默认传统指标（TA-Lib + base_indicators，推荐）
        - enhanced: 增强版特征（WPT/Hurst/Hilbert/光谱/订单流）
        - dl_sequence: 深度学习序列特征
        - comprehensive: 所有特征合并

        Args:
            required_features: 如果指定，只保留这些特征（立即过滤，减少内存占用）
        """
        if required_features:
            print(
                f"🚀 开始特征工程 (feature_types: {self.feature_types}, 仅生成 {len(required_features)} 个指定特征)..."
            )
        else:
            print(f"🚀 开始特征工程 (feature_types: {self.feature_types})...")
        df = data.copy()
        initial_features = len(df.columns)
        prev_count = initial_features

        # 打印初始特征列表（用于调试）
        print(f"   📋 初始特征 ({initial_features} 个): {list(df.columns)}")

        # 数据列（必须保留）
        data_cols = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timestamp",
            "datetime",
            "symbol",
            "_symbol",
            "trade_count",
            "buy_qty",
            "sell_qty",
            "delta",
            "taker_buy_ratio",
            "cvd",
            "cvd_short",
            "cvd_medium",
            "cvd_long",
            "cvd_change_1",
            "cvd_change_5",
            "cvd_change_20",
            "cvd_normalized",
        }

        baseline_features = 0
        default_features = 0  # TA-Lib + base_indicators
        alpha101_features = 0
        enhanced_features = 0
        dl_features = 0

        # 辅助函数：过滤特征
        def _filter_features(df_in: pd.DataFrame, module_name: str) -> pd.DataFrame:
            """在每个模块生成后立即过滤，只保留需要的特征"""
            if required_features is None:
                return df_in
            # 保留数据列和需要的特征列
            cols_before = len(df_in.columns)
            cols_to_keep = [
                c
                for c in df_in.columns
                if c in data_cols
                or c in required_features
                or not pd.api.types.is_numeric_dtype(df_in[c])
            ]
            filtered = df_in[cols_to_keep]
            kept_count = len([c for c in cols_to_keep if c in required_features])
            removed_count = cols_before - len(cols_to_keep)

            # Diagnostic: Check if required features were removed
            required_before = [c for c in df_in.columns if c in required_features]
            required_after = [c for c in filtered.columns if c in required_features]
            if len(required_after) < len(required_before):
                removed_required = set(required_before) - set(required_after)
                print(
                    f"     ⚠️  ERROR: {module_name} removed {len(removed_required)} REQUIRED features!"
                )
                print(
                    f"        Removed required features: {list(removed_required)[:10]}"
                )
                # This should never happen - return original dataframe
                return df_in

            if removed_count > 0:
                print(
                    f"     ✂️ {module_name}: 保留 {kept_count} 个需要的特征，移除 {removed_count} 个不需要的特征"
                )
                print(
                    f"        Before: {cols_before} columns, After: {len(cols_to_keep)} columns"
                )
                # Show some removed features
                removed_cols = set(df_in.columns) - set(cols_to_keep)
                removed_features = [
                    c
                    for c in removed_cols
                    if c not in data_cols
                    and pd.api.types.is_numeric_dtype(
                        df_in.get(c, pd.Series(dtype=float))
                    )
                ]
                if removed_features:
                    print(f"        Example removed features: {removed_features[:10]}")
            return filtered

        # 1. 基线特征工程
        if self.use_baseline:
            print("  📊 Baseline特征工程...")
            try:
                df, self.baseline_engineer = engineer_baseline_features(
                    df,
                    self.baseline_engineer,
                    fit=fit,
                    required_features=required_features,
                )
                baseline_features = len(df.columns) - prev_count
                # 如果已经通过 required_features 过滤，就不需要再次过滤
                if required_features:
                    kept_count = len([c for c in df.columns if c in required_features])
                    print(f"     ✅ Baseline特征: {kept_count} 个需要的特征已计算")
                else:
                    # 在过滤前保存新增的列名（用于统计本次生成的特征）
                    new_columns_before_filter = set(df.columns[prev_count:])
                    df = _filter_features(df, "Baseline")
                    # 计算本次生成的特征中保留了多少（只统计新增的特征列）
                    data_cols = {
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "timestamp",
                        "datetime",
                        "symbol",
                        "_symbol",
                        "trade_count",
                        "buy_qty",
                        "sell_qty",
                        "delta",
                        "taker_buy_ratio",
                        "cvd",
                        "cvd_short",
                        "cvd_medium",
                        "cvd_long",
                        "cvd_change_1",
                        "cvd_change_5",
                        "cvd_change_20",
                        "cvd_normalized",
                    }
                    kept_new_features = len(
                        [
                            c
                            for c in new_columns_before_filter
                            if c in df.columns
                            and c not in data_cols
                            and pd.api.types.is_numeric_dtype(df[c])
                        ]
                    )
                    prev_count = len(df.columns)
                    print(
                        f"     ✅ Baseline特征: {baseline_features} 个生成，{kept_new_features} 个保留"
                    )
            except Exception as e:
                print(f"     ⚠️  Baseline特征失败: {e}")

        # 2. 默认传统指标特征工程（TA-Lib + base_indicators）
        if self.use_default:
            print("  📊 默认传统指标特征工程（TA-Lib + base_indicators）...")
            try:
                if self.basic_engineer is None:
                    self.basic_engineer = FeatureEngineer()
                # 传递 required_features，只计算需要的特征
                df_before = df.copy()
                df = self.basic_engineer.add_technical_indicators(df, required_features)
                default_features = len(df.columns) - prev_count

                # Always apply _filter_features when required_features is specified
                # This ensures we only keep the features we actually need
                if required_features:
                    # Store columns before filter
                    cols_after_add = len(df.columns)
                    # Apply filter to remove unwanted features
                    df = _filter_features(df, "Default")
                    kept_count = len([c for c in df.columns if c in required_features])
                    print(f"     ✅ 默认传统指标特征: {kept_count} 个需要的特征已计算")
                    print(
                        f"     📊 Columns: before={len(df_before.columns)}, after add_technical_indicators={cols_after_add}, after filter={len(df.columns)}"
                    )
                    # Show some example generated features
                    new_cols = set(df.columns) - set(df_before.columns)
                    if new_cols:
                        print(f"     📊 Example new features: {list(new_cols)[:10]}")
                    # Check if features match required_features
                    matched_features = [c for c in new_cols if c in required_features]
                    if matched_features:
                        print(
                            f"     ✅ Matched required features: {len(matched_features)} (examples: {matched_features[:5]})"
                        )
                else:
                    # 在过滤前保存新增的列名（用于统计本次生成的特征）
                    new_columns_before_filter = set(df.columns[prev_count:])
                    df = _filter_features(df, "Default")
                    # 计算本次生成的特征中保留了多少（只统计新增的特征列）
                    data_cols = {
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "timestamp",
                        "datetime",
                        "symbol",
                        "_symbol",
                        "trade_count",
                        "buy_qty",
                        "sell_qty",
                        "delta",
                        "taker_buy_ratio",
                        "cvd",
                        "cvd_short",
                        "cvd_medium",
                        "cvd_long",
                        "cvd_change_1",
                        "cvd_change_5",
                        "cvd_change_20",
                        "cvd_normalized",
                    }
                    kept_new_features = len(
                        [
                            c
                            for c in new_columns_before_filter
                            if c in df.columns
                            and c not in data_cols
                            and pd.api.types.is_numeric_dtype(df[c])
                        ]
                    )
                    prev_count = len(df.columns)
                    print(
                        f"     ✅ 默认传统指标特征: {default_features} 个生成，{kept_new_features} 个保留"
                    )
            except Exception as e:
                print(f"     ⚠️  默认传统指标特征失败: {e}")

        # 3. Alpha101 因子特征（只保留时序版本的关键因子）
        if self.use_alpha101:
            print("  📊 Alpha101 因子特征（时序版本，仅保留关键因子）...")
            try:
                from .alpha_factors.alpha101_timeseries_adapted import (
                    compute_adapted_alpha101_factors,
                )

                # 只计算这4个关键时序因子
                alpha_source = df[["open", "high", "low", "close", "volume"]]
                alpha_df = compute_adapted_alpha101_factors(
                    alpha_source,
                    use_ts_rank=True,
                    alpha001_window=5,  # 波动率过滤器窗口
                    alpha022_corr_window=10,  # 量价相关性窗口
                    alpha022_delta_window=5,  # 相关性变化窗口
                    alpha022_vol_window=20,  # 波动率窗口
                    alpha043_vol_rank_window=20,  # 成交量排名窗口
                    alpha043_mom_rank_window=8,  # 动量排名窗口
                    alpha043_adv_window=20,  # 平均成交量窗口
                    alpha043_mom_period=7,  # 动量周期
                )
                alpha_df = alpha_df.reindex(df.index)
                df = df.join(alpha_df, how="left")
                alpha101_features = len(df.columns) - prev_count
                # 只保留这4个关键因子，不需要额外过滤
                kept_features = [
                    "alpha101_001_ts",  # 波动率过滤器
                    "alpha101_022_ts",  # 量价背离预警
                    "alpha101_043_ts",  # 量价突破信号
                    "alpha101_066_ts",  # K线情绪指标
                ]
                kept_count = len([c for c in kept_features if c in df.columns])
                print(f"     ✅ Alpha101时序因子: {kept_count} 个关键因子已计算")
                print(f"        - alpha101_001_ts: 波动率过滤器 (window=5)")
                print(f"        - alpha101_022_ts: 量价背离预警 (corr_window=10)")
                print(f"        - alpha101_043_ts: 量价突破信号 (mom_period=7)")
                print(f"        - alpha101_066_ts: K线情绪指标")

                # 如果指定了 required_features，只保留需要的特征
                if required_features:
                    # 移除不在 required_features 中的 alpha101 特征
                    alpha101_cols = [c for c in df.columns if c.startswith("alpha101_")]
                    for col in alpha101_cols:
                        if col not in required_features:
                            df = df.drop(columns=[col])
                else:
                    # 移除其他 alpha101 特征（如果有的话）
                    alpha101_cols = [c for c in df.columns if c.startswith("alpha101_")]
                    for col in alpha101_cols:
                        if col not in kept_features:
                            df = df.drop(columns=[col])

                    # 在过滤前保存新增的列名（用于统计本次生成的特征）
                    new_columns_before_filter = set(df.columns[prev_count:])
                    df = _filter_features(df, "Alpha101")
                    # 计算本次生成的特征中保留了多少（只统计新增的特征列）
                    data_cols = {
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "timestamp",
                        "datetime",
                        "symbol",
                        "_symbol",
                        "trade_count",
                        "buy_qty",
                        "sell_qty",
                        "delta",
                        "taker_buy_ratio",
                        "cvd",
                        "cvd_short",
                        "cvd_medium",
                        "cvd_long",
                        "cvd_change_1",
                        "cvd_change_5",
                        "cvd_change_20",
                        "cvd_normalized",
                    }
                    kept_new_features = len(
                        [
                            c
                            for c in new_columns_before_filter
                            if c in df.columns
                            and c not in data_cols
                            and pd.api.types.is_numeric_dtype(df[c])
                        ]
                    )
                    prev_count = len(df.columns)
                    print(
                        f"     ✅ Alpha101特征: {alpha101_features} 个生成，{kept_new_features} 个保留"
                    )
            except Exception as e:
                print(f"     ⚠️  Alpha101特征失败: {e}")

        # 4. 增强版特征工程 (细粒度控制：Hurst, Wavelet, Hilbert, Spectral, Order Flow)
        if required_features:
            essential_base_cols = {
                "cvd",
                "taker_buy_ratio",
                "cvd_short",
                "cvd_medium",
                "cvd_long",
                "cvd_change_1",
                "cvd_change_5",
                "cvd_change_20",
                "cvd_normalized",
                "buy_qty",
                "sell_qty",
                "delta",
            }
            for col in essential_base_cols:
                if col in data.columns and col not in df.columns:
                    df[col] = data[col]

        use_any_enhanced = (
            self.use_enhanced
            or self.use_hurst
            or self.use_wavelet
            or self.use_hilbert
            or self.use_spectral
            or self.use_order_flow
        )
        if use_any_enhanced:
            import os, time

            print("  📊 增强版特征工程...")
            try:
                if self.enhanced_engineer is None:
                    self.enhanced_engineer = EnhancedFeatureEngineer(
                        scaler_type=self.scaler_type,
                        wavelet=self.wavelet,
                        wpt_level=self.wpt_level,
                        hurst_window=self.hurst_window,
                    )

                fast_mode = os.getenv("ENHANCED_FAST", "0").lower() in (
                    "1",
                    "true",
                    "yes",
                )

                def _run(step_name, fn, df_in, module_required_features=None):
                    t0 = time.time()
                    print(f"     ▶️ {step_name} 开始...")
                    # 如果指定了required_features，尝试只计算需要的特征
                    # 注意：enhanced模块的方法可能还不支持required_features参数
                    # 所以先计算所有特征，然后立即过滤
                    try:
                        # 尝试传递required_features（如果方法支持）
                        if module_required_features is not None and hasattr(
                            fn, "__code__"
                        ):
                            # 检查方法是否接受required_features参数
                            import inspect

                            sig = inspect.signature(fn)
                            if "required_features" in sig.parameters:
                                out = fn(
                                    df_in, required_features=module_required_features
                                )
                            else:
                                out = fn(df_in)
                                # 立即过滤
                                if module_required_features:
                                    data_cols = {
                                        "open",
                                        "high",
                                        "low",
                                        "close",
                                        "volume",
                                        "timestamp",
                                        "datetime",
                                        "symbol",
                                        "_symbol",
                                        "trade_count",
                                        "buy_qty",
                                        "sell_qty",
                                        "delta",
                                        "taker_buy_ratio",
                                        "cvd",
                                        "cvd_short",
                                        "cvd_medium",
                                        "cvd_long",
                                        "cvd_change_1",
                                        "cvd_change_5",
                                        "cvd_change_20",
                                        "cvd_normalized",
                                    }
                                    # IMPORTANT: Keep ALL required_features, not just module_required_features
                                    # This preserves features from previous steps
                                    cols_to_keep = [
                                        c
                                        for c in out.columns
                                        if c in data_cols
                                        or c
                                        in required_features  # Keep all required features from all steps
                                        or not pd.api.types.is_numeric_dtype(out[c])
                                    ]
                                    out = out[cols_to_keep]
                        else:
                            out = fn(df_in)
                            # 如果没有传递module_required_features，使用全局required_features过滤
                            # BUT: Only filter NEW features, not existing ones!
                            if required_features:
                                # Only filter if this step generated new features
                                # Keep all existing columns from df_in
                                existing_cols = set(df_in.columns)
                                new_cols = set(out.columns) - existing_cols
                                if new_cols:
                                    # Filter only new columns, keep existing ones
                                    cols_to_keep = list(existing_cols) + [
                                        c
                                        for c in new_cols
                                        if c in required_features
                                        or c
                                        not in required_features  # Keep all new cols for now, filter later
                                    ]
                                    # Actually, apply filter to entire dataframe but preserve existing required features
                                    out = _filter_features(out, step_name)
                    except TypeError:
                        # 如果方法不支持required_features参数，使用默认方式
                        out = fn(df_in)
                        if required_features:
                            # Apply filter but preserve existing required features
                            out = _filter_features(out, step_name)
                    dt = time.time() - t0
                    # 计算实际保留的特征数量
                    if required_features:
                        kept_count = len(
                            [c for c in out.columns if c in required_features]
                        )
                        print(
                            f"     ⏱ {step_name} 完成，用时 {dt:.2f}s，保留 {kept_count} 个需要的特征"
                        )
                    else:
                        # 当 required_features 为 None 时，所有生成的特征都被保留
                        data_cols = {
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "timestamp",
                            "datetime",
                            "symbol",
                            "_symbol",
                            "trade_count",
                            "buy_qty",
                            "sell_qty",
                            "delta",
                            "taker_buy_ratio",
                            "cvd",
                            "cvd_short",
                            "cvd_medium",
                            "cvd_long",
                            "cvd_change_1",
                            "cvd_change_5",
                            "cvd_change_20",
                            "cvd_normalized",
                        }
                        kept_count = len(
                            [
                                c
                                for c in out.columns
                                if c not in data_cols
                                and pd.api.types.is_numeric_dtype(out[c])
                            ]
                        )
                        print(
                            f"     ⏱ {step_name} 完成，用时 {dt:.2f}s，保留 {kept_count} 个特征"
                        )
                    return out

                # 为每个模块提取需要的特征
                hurst_features = None
                wavelet_features = None
                hilbert_features = None
                spectral_features = None
                order_flow_features = None

                if required_features:
                    hurst_features = {
                        f
                        for f in required_features
                        if f.startswith("hurst_") or "hurst" in f.lower()
                    }
                    wavelet_features = {
                        f
                        for f in required_features
                        if f.startswith("wpt_") or "wavelet" in f.lower()
                    }
                    hilbert_features = {
                        f
                        for f in required_features
                        if f.startswith("hilbert_")
                        or "hilbert" in f.lower()
                        or "phase_" in f
                        or "frequency_" in f
                    }
                    spectral_features = {
                        f
                        for f in required_features
                        if f.startswith("spectral_") or "fft_" in f or "psd_" in f
                    }
                    order_flow_features = {
                        f
                        for f in required_features
                        if "cvd" in f.lower()
                        or "ofi" in f.lower()
                        or "order_flow" in f.lower()
                        or "taker_buy" in f.lower()
                    }

                # Hurst（较快）
                if self.use_enhanced or self.use_hurst:
                    if not required_features or hurst_features:
                        df = _run(
                            "Hurst",
                            self.enhanced_engineer.add_hurst_features,
                            df,
                            hurst_features,
                        )

                # Wavelet（较重）
                if self.use_enhanced or self.use_wavelet:
                    if not fast_mode:
                        if not required_features or wavelet_features:
                            df = _run(
                                "WaveletPacket",
                                self.enhanced_engineer.add_wavelet_packet_features,
                                df,
                                wavelet_features,
                            )

                # Spectral（较重）
                if self.use_enhanced or self.use_spectral:
                    if not fast_mode:
                        if not required_features or spectral_features:
                            df = _run(
                                "Spectral",
                                self.enhanced_engineer.add_spectral_features,
                                df,
                                spectral_features,
                            )

                # Hilbert
                if self.use_enhanced or self.use_hilbert:
                    if not required_features or hilbert_features:
                        df = _run(
                            "Hilbert",
                            self.enhanced_engineer.add_hilbert_features,
                            df,
                            hilbert_features,
                        )

                # Order Flow
                if self.use_enhanced or self.use_order_flow:
                    if not required_features or order_flow_features:
                        df = _run(
                            "OrderFlow",
                            self.enhanced_engineer.add_order_flow_features,
                            df,
                            order_flow_features,
                        )

                enhanced_features = len(df.columns) - prev_count
                prev_count = len(df.columns)
                active_modules = []
                if self.use_enhanced or self.use_hurst:
                    active_modules.append("Hurst")
                if self.use_enhanced or self.use_wavelet:
                    active_modules.append("Wavelet")
                if self.use_enhanced or self.use_hilbert:
                    active_modules.append("Hilbert")
                if self.use_enhanced or self.use_spectral:
                    active_modules.append("Spectral")
                if self.use_enhanced or self.use_order_flow:
                    active_modules.append("OrderFlow")
                print(
                    f"     ✅ 增强版特征: {enhanced_features} 个 (模块: {', '.join(active_modules)}, fast_mode={fast_mode})"
                )
            except Exception as e:
                print(f"     ⚠️  增强版特征失败: {e}")

        # 5. 深度学习序列特征
        if self.use_dl_sequence:
            # Check if any dl_seq features are requested
            has_dl_seq_request = False
            if required_features is not None:
                has_dl_seq_request = any(
                    feat.startswith("dl_seq") for feat in required_features
                )
            else:
                # If no required_features specified, generate all (comprehensive mode)
                has_dl_seq_request = True

            if has_dl_seq_request:
                print("  📊 深度学习序列特征...")
                try:
                    # 如果 fit=True，创建新的 extractor；如果 fit=False，使用已保存的 extractor
                    if fit:
                        # Create new extractor and fit
                        from .dl_sequence_features import DeepLearningSequenceExtractor

                        self.dl_sequence_extractor = DeepLearningSequenceExtractor(
                            backend=self.dl_backend,
                            seq_length=self.dl_seq_length,
                            d_model=self.dl_d_model,
                            use_fp16=self.use_fp16,
                            normalization_method="ema",  # 强制使用 EMA（因果安全，已修复泄露）
                        )
                        # Fit: 只初始化模型，不接触数据
                        self.dl_sequence_extractor.fit(df)
                        # Transform: 提取特征（EMA 从头计算，完全因果）
                        df = self.dl_sequence_extractor.add_to_dataframe(df)
                    else:
                        # Use saved extractor (transform only)
                        if self.dl_sequence_extractor is None:
                            raise RuntimeError(
                                "dl_sequence_extractor not fitted. Call engineer_all_features with fit=True first."
                            )
                        # Transform: 提取特征（EMA 从头计算，完全因果）
                        df = self.dl_sequence_extractor.add_to_dataframe(df)

                    # If required_features specified, filter to only requested dl_seq features
                    if required_features is not None:
                        dl_seq_cols = [
                            col for col in df.columns if col.startswith("dl_seq")
                        ]
                        cols_to_remove = [
                            col for col in dl_seq_cols if col not in required_features
                        ]
                        if cols_to_remove:
                            df = df.drop(columns=cols_to_remove)
                            print(
                                f"     ✂️  Filtered to {len(dl_seq_cols) - len(cols_to_remove)} requested dl_seq features"
                            )

                    dl_features = len(df.columns) - prev_count
                    prev_count = len(df.columns)
                    print(f"     ✅ 深度学习特征: {dl_features} 个")
                except Exception as e:
                    print(f"     ⚠️  深度学习特征失败: {e}")
            else:
                print("  📊 深度学习序列特征... (跳过，未在 required_features 中)")

        total_new_features = len(df.columns) - initial_features
        self.total_features = total_new_features

        print(f"\n✅ 特征工程完成!")
        print(f"  原始特征: {initial_features} 个")
        print(f"  新增特征: {total_new_features} 个")
        print(f"  总特征数: {len(df.columns)} 个")

        # Diagnostic: Show feature columns if required_features specified
        if required_features is not None:
            feature_cols_in_df = [c for c in df.columns if c in required_features]
            print(
                f"  📊 Required features found in DataFrame: {len(feature_cols_in_df)}/{len(required_features)}"
            )
            if len(feature_cols_in_df) < len(required_features):
                missing_in_df = required_features - set(feature_cols_in_df)
                print(
                    f"  ⚠️  Missing {len(missing_in_df)} required features in final DataFrame (first 10):"
                )
                for feat in list(missing_in_df)[:10]:
                    print(f"      - {feat}")
            print(
                f"  📊 All columns in final DataFrame ({len(df.columns)}): {list(df.columns)[:40]}"
            )

            # Final check: Apply _filter_features if features are missing
            if len(feature_cols_in_df) == 0 and len(df.columns) > initial_features:
                print(
                    f"  ⚠️  WARNING: No required features found but {len(df.columns) - initial_features} new features were generated!"
                )
                print(
                    f"      This suggests features were generated but then removed. Checking..."
                )
                # Check if _filter_features would help
                filtered_df = _filter_features(df, "FinalCheck")
                filtered_feature_cols = [
                    c for c in filtered_df.columns if c in required_features
                ]
                print(
                    f"      Before final filter: {len(df.columns)} columns, {len(feature_cols_in_df)} required features"
                )
                print(
                    f"      After final filter: {len(filtered_df.columns)} columns, {len(filtered_feature_cols)} required features"
                )
                if len(filtered_feature_cols) > len(feature_cols_in_df):
                    print(
                        f"      ✅ Applying final filter recovered {len(filtered_feature_cols)} required features!"
                    )
                    # Return filtered dataframe
                    df = filtered_df
                else:
                    print(
                        f"      ❌ Final filter did not help. Features may not have been generated correctly."
                    )
                    print(f"      📊 Debug: df.columns = {list(df.columns)[:40]}")
                    print(
                        f"      📊 Debug: required_features (first 20) = {list(required_features)[:20]}"
                    )
                    # Check if there's a mismatch
                    all_generated = set(df.columns) - data_cols
                    required_set = set(required_features)
                    overlap = all_generated & required_set
                    print(
                        f"      📊 Debug: Overlap between generated and required: {len(overlap)} features"
                    )
                    if len(overlap) > 0:
                        print(
                            f"      📊 Debug: Overlapping features: {list(overlap)[:10]}"
                        )
                    else:
                        print(
                            f"      ⚠️  No overlap! This suggests feature names don't match."
                        )
        use_any_enhanced = (
            self.use_enhanced
            or self.use_hurst
            or self.use_wavelet
            or self.use_hilbert
            or self.use_spectral
            or self.use_order_flow
        )
        if (
            self.use_baseline
            or self.use_default
            or self.use_alpha101
            or use_any_enhanced
            or self.use_dl_sequence
        ):
            print(f"  特征分布:")
            if self.use_baseline:
                print(f"    - Baseline特征: {baseline_features} 个")
            if self.use_default:
                print(
                    f"    - 默认传统指标特征（TA-Lib + base_indicators）: {default_features} 个"
                )
            if self.use_alpha101:
                print(f"    - Alpha101因子特征: {alpha101_features} 个")
            if use_any_enhanced:
                print(f"    - 增强版特征: {enhanced_features} 个")
                if self.use_enhanced:
                    print(
                        f"      (包含所有子模块: Hurst, Wavelet, Hilbert, Spectral, OrderFlow)"
                    )
                else:
                    active_modules = []
                    if self.use_hurst:
                        active_modules.append("Hurst")
                    if self.use_wavelet:
                        active_modules.append("Wavelet")
                    if self.use_hilbert:
                        active_modules.append("Hilbert")
                    if self.use_spectral:
                        active_modules.append("Spectral")
                    if self.use_order_flow:
                        active_modules.append("OrderFlow")
                    if active_modules:
                        print(f"      (子模块: {', '.join(active_modules)})")
            if self.use_dl_sequence:
                print(f"    - 深度学习特征: {dl_features} 个")

        self.feature_stats = {
            "baseline_features": baseline_features,
            "default_features": default_features,  # TA-Lib + base_indicators
            "enhanced_features": enhanced_features,
            "alpha101_features": alpha101_features,
            "dl_features": dl_features,
            "total_new_features": total_new_features,
            "total_features": len(df.columns),
            "enhanced_modules": (
                {
                    "hurst": self.use_hurst if use_any_enhanced else False,
                    "wavelet": self.use_wavelet if use_any_enhanced else False,
                    "hilbert": self.use_hilbert if use_any_enhanced else False,
                    "spectral": self.use_spectral if use_any_enhanced else False,
                    "order_flow": self.use_order_flow if use_any_enhanced else False,
                }
                if use_any_enhanced
                else {}
            ),
        }

        return df

    def engineer_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
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
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "signal",
            "binary_signal",
            "future_return",
            "symbol",
        ]
        # 排除多周期标签列
        exclude_patterns = ["signal_", "binary_signal_", "future_return_"]
        # 排除未归一化/原始尺度列（使用对应的归一化替代列）
        exclude_raw = {
            "bb_upper",
            "bb_lower",
            "bb_middle",
            # bb_width 现在已经是归一化的（除以 bb_middle 或 close），所以保留
            "hl",  # 中间变量
            "up_vol",
            "down_vol",  # 中间变量
        }
        # 原始价格量纲的指标前缀（未标准化），统一剔除
        # 注意：只排除原始值（如 sma_5），保留归一化版本（如 sma_5_pct_close）
        raw_prefixes = (
            "sma_",
            "ema_",
            "wma_",
            "tema_",
            "kama_",  # 均线族（价格量纲）
            "volume_sma_",  # 量均线（未标准化）
            "atr_",  # 原始ATR（未归一化），保留natr和atr_normalized
        )
        # 归一化版本的后缀（这些应该保留）
        normalized_suffixes = (
            "_pct_close",
            "_ratio",
            "_normalized",
            "_percentile",
        )
        # MACD 原始量纲（价格差异），为稳妥剔除原始MACD系
        raw_exact = {
            "macd",
            "macd_signal",
            "macd_hist",
            "macd_ext_hist",
            "macd_fix_hist",
            "atr",  # 原始ATR（未归一化），保留natr和atr_normalized
        }
        # 排除未归一化的小波特征（保留归一化的小波特征）
        # 未归一化：wpt_*_energy, wpt_*_mean, wpt_*_std
        # 已归一化：wpt_*_energy_ratio, wpt_shannon_entropy, wpt_energy_concentration, wpt_high_low_ratio, wpt_dominant_band
        wpt_raw_patterns = ("_energy", "_mean", "_std")
        feature_cols = []
        for col in df.columns:
            if (
                col not in exclude_columns
                and col not in exclude_raw
                and col not in raw_exact
            ):
                # 检查是否匹配排除模式
                if not any(col.startswith(pattern) for pattern in exclude_patterns):
                    # 过滤原始未归一化前缀（但保留归一化版本）
                    if any(col.startswith(p) for p in raw_prefixes):
                        # 检查是否是归一化版本（有归一化后缀）
                        if not any(
                            col.endswith(suffix) for suffix in normalized_suffixes
                        ):
                            continue
                    # 排除未归一化的小波特征（wpt_*_energy, wpt_*_mean, wpt_*_std）
                    # 但保留归一化的小波特征（wpt_*_energy_ratio, wpt_shannon_entropy 等）
                    if "wpt_" in col and any(col.endswith(p) for p in wpt_raw_patterns):
                        # 检查是否是 energy_ratio（已归一化）
                        if not col.endswith("_energy_ratio"):
                            continue
                    # 排除 channel 的原始价格量纲特征（保留归一化的距离特征）
                    if col in ["channel_mid", "channel_upper", "channel_lower"]:
                        continue
                    # 排除 Hilbert 的原始幅度和频率（保留归一化的相位）
                    if col.endswith("_hilbert_amplitude") or col.endswith(
                        "_hilbert_frequency"
                    ):
                        continue
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
                "percentile_window": self.baseline_engineer.percentile_window,
                "compression_threshold_pct": self.baseline_engineer.compression_threshold_pct,
                "vwap_window": self.baseline_engineer.vwap_window,
                "feature_clip_bound": self.baseline_engineer.feature_clip_bound,
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
            self.enhanced_engineer.scalers = scalers_data.get("enhanced_scalers", {})

        # 加载基线标准化器
        if self.baseline_engineer is not None and "baseline_scalers" in scalers_data:
            baseline_scalers = scalers_data.get("baseline_scalers", {})
            # 向后兼容：如果存在旧的 quantiles 字段，忽略它们
            self.baseline_engineer.percentile_window = baseline_scalers.get(
                "percentile_window", 288
            )
            self.baseline_engineer.compression_threshold_pct = baseline_scalers.get(
                "compression_threshold_pct", 0.2
            )
            self.baseline_engineer.vwap_window = baseline_scalers.get(
                "vwap_window", 160
            )
            self.baseline_engineer.feature_clip_bound = baseline_scalers.get(
                "feature_clip_bound", 10.0
            )

        self.feature_stats = scalers_data.get("feature_stats", {})

        print(f"✅ 标准化器从 {path} 加载完成")


def create_comprehensive_feature_engineer(
    feature_types: str = "comprehensive", scaler_type: str = "standard", **kwargs
) -> ComprehensiveFeatureEngineer:
    """
    创建综合特征工程器的便捷函数

    Args:
        feature_types: 特征类型 ('baseline', 'talib', 'alpha101', 'enhanced', 'dl_sequence', 'comprehensive', 或逗号分隔的组合)
        scaler_type: 标准化类型
        **kwargs: 其他参数

    Returns:
        ComprehensiveFeatureEngineer实例
    """
    return ComprehensiveFeatureEngineer(
        feature_types=feature_types, scaler_type=scaler_type, **kwargs
    )


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
        feature_types: 特征类型 ('baseline', 'default', 'alpha101', 'enhanced', 'dl_sequence', 'comprehensive'，或逗号分隔的组合)
        feature_engineer: 特征工程器实例（如果为None，会创建新的）
        fit: 是否拟合

    Returns:
        (工程后的DataFrame, 特征工程器)
    """
    if feature_engineer is None:
        feature_engineer = ComprehensiveFeatureEngineer(feature_types=feature_types)
    elif feature_engineer.feature_types != feature_types:
        # 如果特征类型不匹配，创建新的
        feature_engineer = ComprehensiveFeatureEngineer(feature_types=feature_types)

    engineered_df = feature_engineer.engineer_all_features(df, fit=fit)
    return engineered_df, feature_engineer


def get_feature_columns_by_type(
    df: pd.DataFrame, feature_types: str = "baseline"
) -> List[str]:
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
        temp_engineer = ComprehensiveFeatureEngineer(feature_types=feature_types)
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
    test_data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": np.random.randn(1000).cumsum() + 100,
            "high": np.random.randn(1000).cumsum() + 105,
            "low": np.random.randn(1000).cumsum() + 95,
            "close": np.random.randn(1000).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 1000),
        }
    )

    # 测试综合特征工程
    engineer = ComprehensiveFeatureEngineer()
    result_df = engineer.engineer_all_features(test_data)

    print(f"\n🎉 测试完成!")
    print(f"  输入特征: {len(test_data.columns)} 个")
    print(f"  输出特征: {len(result_df.columns)} 个")
    print(f"  新增特征: {len(result_df.columns) - len(test_data.columns)} 个")
