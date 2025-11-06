"""
安全的双标的预处理流程

每个标的完全独立处理：
1. 独立 resample
2. 独立特征工程
3. 独立计算标签
4. 独立截断（删除最后 fb 行）
5. 最后合并

这样可以完全避免跨标的的数据泄露。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from ml_trading.data_tools.baseline_feature_engineering import (
    engineer_baseline_features, )
from ml_trading.data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer, )


def safe_multi_asset_preprocessing(
    files: List[str],
    feature_type: str,
    timeframe: str,
    forward_bars: int,
    feature_engineer: Optional[ComprehensiveFeatureEngineer] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    安全的双标的预处理流程
    
    Args:
        files: 数据文件列表
        feature_type: 特征类型
        timeframe: 时间框架（如 '240T', '5T'）
        forward_bars: 前向K线数
        feature_engineer: 特征工程器（已废弃，每个标的独立创建，保留以兼容）
    
    Returns:
        (processed_df, metadata): 处理后的数据框和元数据
    
    Note:
        - 每个标的独立创建和拟合特征工程器，避免状态污染
        - future_volatility 使用 abs(future_return)，避免未来信息泄露
        - 使用 log return + winsorize 避免极端值
    """
    # Import helper functions
    from ml_trading.pipeline.training.train import _resample_single_asset

    # Helper function to load parquet files
    def load_parquet_file(file_path: str) -> pd.DataFrame:
        """Load a single parquet file."""
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            print(f"   ⚠️  警告: 无法加载文件 {file_path}: {e}")
            return pd.DataFrame()

    def _load_many(file_list: List[str]) -> pd.DataFrame:
        """Load and merge multiple parquet files."""
        frames = []
        for f in file_list:
            df = load_parquet_file(f) if f.endswith(".parquet") else None
            if df is not None and len(df) > 0:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, axis=0).sort_index()
        return merged

    # 按标的分组文件
    symbol_files: Dict[str, List[str]] = {}
    for file_path in files:
        # 从文件名推断标的
        filename = Path(file_path).stem.upper()
        symbol = None
        if "BTC" in filename:
            symbol = "BTCUSDT"
        elif "ETH" in filename:
            symbol = "ETHUSDT"
        elif "SOL" in filename:
            symbol = "SOLUSDT"
        elif "BNB" in filename:
            symbol = "BNBUSDT"
        elif "ADA" in filename:
            symbol = "ADAUSDT"
        else:
            # 尝试从数据中读取
            try:
                df_test = pd.read_parquet(file_path)
                if "symbol" in df_test.columns:
                    symbol = df_test["symbol"].iloc[0]
            except Exception:
                pass

        if symbol is None:
            # 🔒 CRITICAL: 如果没有 symbol 信息，无法安全合并多标的
            # 必须能够识别每个样本属于哪个标的，否则会导致：
            # 1. 标签混淆（同一时间戳多个资产的标签混在一起）
            # 2. 评估失真（无法按标的分组评估）
            # 3. 模型学偏（模型不知道样本属于哪个资产）
            print(f"   ⚠️  警告: 无法从文件 {file_path} 中提取 symbol 信息")
            print(f"      尝试从文件名推断...")
            # 最后尝试：从文件路径中提取
            file_path_upper = str(file_path).upper()
            if "BTC" in file_path_upper:
                symbol = "BTCUSDT"
            elif "ETH" in file_path_upper:
                symbol = "ETHUSDT"
            elif "SOL" in file_path_upper:
                symbol = "SOLUSDT"
            elif "BNB" in file_path_upper:
                symbol = "BNBUSDT"
            elif "ADA" in file_path_upper:
                symbol = "ADAUSDT"
            else:
                symbol = "UNKNOWN"
                print(f"      ❌ 无法确定 symbol，将使用 'UNKNOWN'")
                print(f"      ⚠️  严重警告：没有 symbol 信息的多标的合并是不安全的！")
                print(f"         可能导致：")
                print(f"         1. 标签混淆（同一时间戳多个资产的标签混在一起）")
                print(f"         2. 评估失真（无法按标的分组评估）")
                print(f"         3. 模型学偏（模型不知道样本属于哪个资产）")
                print(f"         4. 推理时无法确定预测适用于哪个标的")
                print(f"      💡 建议：")
                print(f"         - 确保数据文件包含 symbol 列")
                print(f"         - 或确保文件名包含标的标识（如 BTC、ETH、SOL）")
                print(f"         - 或使用单标的训练模式")

        if symbol not in symbol_files:
            symbol_files[symbol] = []
        symbol_files[symbol].append(file_path)

    print(f"\n{'='*70}")
    print(f"🔒 安全的多标的预处理（完全隔离）")
    print(f"{'='*70}")
    print(f"   标的数量: {len(symbol_files)}")
    print(f"   时间框架: {timeframe}")
    print(f"   前向K线: {forward_bars}")
    print(f"{'='*70}\n")

    all_processed_dfs = []
    metadata = {
        "symbol_stats": {},
        "total_samples_before": 0,
        "total_samples_after": 0,
    }

    # 每个标的独立处理
    for symbol, symbol_file_list in symbol_files.items():
        print(f"📊 处理标的: {symbol} ({len(symbol_file_list)} 个文件)")

        # 1. 加载该标的的所有数据
        symbol_raw = _load_many(symbol_file_list)
        if len(symbol_raw) == 0:
            print(f"   ⚠️  警告: {symbol} 没有数据，跳过")
            continue

        # 确保时间索引
        if not isinstance(symbol_raw.index, pd.DatetimeIndex):
            if "timestamp" in symbol_raw.columns:
                symbol_raw.set_index("timestamp", inplace=True)
            else:
                print(f"   ❌ 错误: {symbol} 无法确定时间索引，跳过")
                continue

        symbol_raw = symbol_raw.sort_index()
        metadata["total_samples_before"] += len(symbol_raw)

        # 2. 独立 resample（仅用本标的的数据）
        print(f"   🔄 Resampling {symbol}...")
        try:
            # 移除 symbol 列（如果存在）以便 resample
            symbol_data_for_resample = symbol_raw.copy()
            if "symbol" in symbol_data_for_resample.columns:
                symbol_data_for_resample = symbol_data_for_resample.drop(
                    columns=["symbol"])

            symbol_resampled = _resample_single_asset(symbol_data_for_resample,
                                                      timeframe)
            # 🔒 CRITICAL: 必须确保每个样本都有 symbol 信息
            # 这是多标的合并安全性的基础
            symbol_resampled["symbol"] = symbol

            # 验证：确保 symbol 列已正确添加
            if "symbol" not in symbol_resampled.columns:
                raise ValueError(f"❌ 严重错误：{symbol} 的 resampled 数据缺少 symbol 列！"
                                 "这会导致多标的合并不安全。请检查 _resample_single_asset 函数。")

            # 验证：确保所有行的 symbol 值都正确
            if not (symbol_resampled["symbol"] == symbol).all():
                raise ValueError(
                    f"❌ 严重错误：{symbol} 的 resampled 数据中 symbol 列值不一致！"
                    "这会导致多标的合并不安全。")

            print(
                f"      Resampled: {len(symbol_raw):,} → {len(symbol_resampled):,} 条"
            )
        except Exception as e:
            print(f"   ❌ 错误: {symbol} resample 失败: {e}")
            continue

        # 验证：统计 volume=0 的数量（这是正常的市场状态，不是错误）
        # ✅ 根据最佳实践：无交易窗口应该保留（fillna(0)），而不是删除（dropna）
        # 这样可以避免样本选择偏差，让模型学习真实的市场状态（包括静默时段）
        volume_zero = (symbol_resampled["volume"] == 0).sum()
        if volume_zero > 0:
            zero_volume_ratio = volume_zero / len(symbol_resampled) * 100
            print(
                f"   📊 统计: {symbol} 有 {volume_zero} 个无交易窗口（{zero_volume_ratio:.1f}%），"
                f"已使用 fillna(0) 保留（符合最佳实践，避免样本选择偏差）")

        # 🔧 CRITICAL FIX: 显式分离特征与标签的时间边界
        # 特征工程只使用到时间 t 的数据（截断最后 fb 行）
        # 标签计算需要到时间 t+fb 的数据（完整数据）
        # 这样可以物理隔离特征计算和标签生成，杜绝任何滚动窗口越界可能

        # 3a. 准备特征工程数据（只用到 t，不包含最后 fb 行）
        if len(symbol_resampled) > forward_bars:
            symbol_for_features = symbol_resampled.iloc[:-forward_bars].copy()
        else:
            print(
                f"   ⚠️  警告: {symbol} 数据量不足（{len(symbol_resampled)} <= {forward_bars}），跳过"
            )
            continue

        # 3b. 独立特征工程（仅用本标的的数据，且已截断）
        # 🔒 CRITICAL FIX: 每个标的必须拥有独立的、仅基于自身数据拟合的特征工程器
        # 不能复用 feature_engineer，否则会导致状态污染（第二个标的的 fit 会覆盖第一个标的的统计量）
        print(f"   🔧 特征工程 {symbol}（使用 {len(symbol_for_features):,} 条数据）...")

        # 🔒 CRITICAL: 保存 symbol 列，确保特征工程后不会丢失
        symbol_col_backup = symbol_for_features["symbol"].copy(
        ) if "symbol" in symbol_for_features.columns else None

        try:
            if feature_type == "baseline":
                # 每个标的独立创建和拟合特征工程器
                symbol_feat_df, _ = engineer_baseline_features(
                    symbol_for_features, None, fit=True)
            else:
                # 每个标的独立创建和拟合特征工程器
                local_feature_engineer = ComprehensiveFeatureEngineer(
                    feature_types=feature_type)
                symbol_feat_df = local_feature_engineer.engineer_all_features(
                    symbol_for_features, fit=True)

            # 🔒 CRITICAL: 恢复 symbol 列（特征工程可能会删除它）
            if "symbol" not in symbol_feat_df.columns and symbol_col_backup is not None:
                # 确保索引对齐
                symbol_feat_df["symbol"] = symbol_col_backup.reindex(
                    symbol_feat_df.index, fill_value=symbol)
            elif "symbol" not in symbol_feat_df.columns:
                # 如果备份也不存在，直接添加
                symbol_feat_df["symbol"] = symbol
            else:
                # 验证 symbol 列的值是否正确
                if not (symbol_feat_df["symbol"] == symbol).all():
                    symbol_feat_df["symbol"] = symbol

            print(
                f"      特征数: {len([c for c in symbol_feat_df.columns if c not in ['symbol', 'future_return', 'future_volatility']])}"
            )

            # 🔧 FIX: 排除原始的未标准化特征，只保留标准化后的版本
            # 1. 排除原始订单流特征（如原始 cvd）
            # 2. 排除原始价格量纲特征（如 roll_high_*, roll_low_*）
            raw_cols_to_exclude = {
                "cvd",  # 原始 CVD（未标准化），使用 cumulative_ofi, ofi_* 等标准化版本
                "roll_high_s",  # 原始滚动高点（使用原始 high），使用 sr_dist_high_s（标准化）
                "roll_low_s",  # 原始滚动低点（使用原始 low），使用 sr_dist_low_s（标准化）
                "roll_high_l",  # 原始滚动高点（使用原始 high），使用 sr_dist_high_l（标准化）
                "roll_low_l",  # 原始滚动低点（使用原始 low），使用 sr_dist_low_l（标准化）
            }

            # 检查并排除原始未标准化特征
            cols_to_drop = []
            for col in symbol_feat_df.columns:
                if col in raw_cols_to_exclude:
                    # 对于 cvd，检查是否有标准化版本
                    if col == "cvd":
                        has_normalized_version = any(
                            norm_col in symbol_feat_df.columns
                            for norm_col in [
                                "cumulative_ofi", "ofi_short", "ofi_medium",
                                "ofi_long"
                            ])
                        if has_normalized_version:
                            cols_to_drop.append(col)
                            print(f"      ✅ 排除原始未标准化特征: {col}（已使用标准化版本）")
                        else:
                            # 如果没有标准化版本，检查值范围并标准化
                            col_values = symbol_feat_df[col].dropna()
                            if len(col_values) > 0:
                                col_abs_max = col_values.abs().max()
                                if col_abs_max > 1000:
                                    print(
                                        f"      🔧 标准化原始特征: {col} (max_abs={col_abs_max:.2f})"
                                    )
                                    from sklearn.preprocessing import RobustScaler
                                    scaler = RobustScaler()
                                    col_values_scaled = scaler.fit_transform(
                                        col_values.values.reshape(
                                            -1, 1)).flatten()
                                    symbol_feat_df.loc[col_values.index,
                                                       col] = col_values_scaled
                                    print(
                                        f"         标准化后: max_abs={np.abs(col_values_scaled).max():.2f}"
                                    )
                    # 对于 roll_high_* 和 roll_low_*，直接排除（已有标准化版本 sr_dist_*）
                    elif col.startswith("roll_"):
                        # 检查是否有对应的标准化版本
                        if col == "roll_high_s" and "sr_dist_high_s" in symbol_feat_df.columns:
                            cols_to_drop.append(col)
                            print(
                                f"      ✅ 排除原始未标准化特征: {col}（已使用标准化版本 sr_dist_high_s）"
                            )
                        elif col == "roll_low_s" and "sr_dist_low_s" in symbol_feat_df.columns:
                            cols_to_drop.append(col)
                            print(
                                f"      ✅ 排除原始未标准化特征: {col}（已使用标准化版本 sr_dist_low_s）"
                            )
                        elif col == "roll_high_l" and "sr_dist_high_l" in symbol_feat_df.columns:
                            cols_to_drop.append(col)
                            print(
                                f"      ✅ 排除原始未标准化特征: {col}（已使用标准化版本 sr_dist_high_l）"
                            )
                        elif col == "roll_low_l" and "sr_dist_low_l" in symbol_feat_df.columns:
                            cols_to_drop.append(col)
                            print(
                                f"      ✅ 排除原始未标准化特征: {col}（已使用标准化版本 sr_dist_low_l）"
                            )
                        else:
                            # 如果没有标准化版本，直接删除（这些特征不应该被使用）
                            cols_to_drop.append(col)
                            print(f"      ✅ 排除原始未标准化特征: {col}（无标准化版本，不应使用）")

            # 删除原始未标准化特征
            if cols_to_drop:
                symbol_feat_df = symbol_feat_df.drop(columns=cols_to_drop)
        except Exception as e:
            print(f"   ❌ 错误: {symbol} 特征工程失败: {e}")
            import traceback
            traceback.print_exc()
            continue

        # 4. 独立计算标签（使用完整数据，需要 t+fb）
        print(f"   📈 计算标签 {symbol}（使用完整数据 {len(symbol_resampled):,} 条）...")

        # 🔍 DIAGNOSTIC: 打印原始价格数据统计（用于排查数据问题）
        print(f"      🔍 原始价格数据诊断:")
        close_prices = symbol_resampled["close"].dropna()
        if len(close_prices) > 0:
            print(
                f"         close 价格范围: [{close_prices.min():.2f}, {close_prices.max():.2f}]"
            )
            print(
                f"         close 价格均值: {close_prices.mean():.2f}, 中位数: {close_prices.median():.2f}"
            )
            print(f"         close 价格标准差: {close_prices.std():.2f}")
            # 检查是否有异常价格跳空
            price_changes = close_prices.pct_change().dropna()
            extreme_jumps = (price_changes.abs() > 0.5).sum()  # >50% 跳空
            if extreme_jumps > 0:
                print(f"         ⚠️  警告: 检测到 {extreme_jumps} 个极端价格跳空（>50%）")
                extreme_jump_values = price_changes[price_changes.abs() > 0.5]
                print(
                    f"         极端跳空范围: [{extreme_jump_values.min():.4f}, {extreme_jump_values.max():.4f}]"
                )

        # 🔧 OPTIMIZATION: 使用 log return + winsorize 避免极端值
        # Log return 更稳定且对称，winsorize 防止极端值主导 loss
        try:
            from scipy.stats.mstats import winsorize

            # 计算 simple return（用于诊断）
            simple_returns_raw = (
                symbol_resampled["close"].shift(-forward_bars) /
                symbol_resampled["close"] - 1)

            # 🔍 DIAGNOSTIC: 打印 simple return 统计（处理前）
            simple_returns_valid = simple_returns_raw.dropna()
            if len(simple_returns_valid) > 0:
                print(f"      🔍 Simple Return 诊断（处理前，fb={forward_bars}）:")
                print(
                    f"         范围: [{simple_returns_valid.min():.6f}, {simple_returns_valid.max():.6f}]"
                )
                print(
                    f"         均值: {simple_returns_valid.mean():.6f}, 中位数: {simple_returns_valid.median():.6f}"
                )
                print(f"         标准差: {simple_returns_valid.std():.6f}")
                print(
                    f"         绝对值最大值: {simple_returns_valid.abs().max():.6f} ({simple_returns_valid.abs().max()*100:.2f}%)"
                )
                # 检查极端值
                extreme_simple = (simple_returns_valid.abs()
                                  > 1.0).sum()  # >100%
                if extreme_simple > 0:
                    extreme_ratio = extreme_simple / len(
                        simple_returns_valid) * 100
                    print(
                        f"         ⚠️  警告: {extreme_simple} 个极端值（>100%），占比 {extreme_ratio:.2f}%"
                    )
                    extreme_values = simple_returns_valid[
                        simple_returns_valid.abs() > 1.0]
                    print(
                        f"         极端值范围: [{extreme_values.min():.6f}, {extreme_values.max():.6f}]"
                    )
                    print(
                        f"         极端值示例（前5个）: {extreme_values.head().tolist()}"
                    )

            # 计算 log return（使用完整数据）
            log_returns = np.log(
                symbol_resampled["close"].shift(-forward_bars) /
                symbol_resampled["close"])

            # 只取有效部分（去掉最后 fb 行，与 symbol_for_features 对齐）
            # 确保索引与 symbol_feat_df 一致
            valid_log_returns = log_returns.iloc[:-forward_bars].copy()

            # 🔍 DIAGNOSTIC: 打印 log return 统计（处理前）
            valid_log_returns_clean_diag = valid_log_returns.dropna()
            if len(valid_log_returns_clean_diag) > 0:
                print(f"      🔍 Log Return 诊断（处理前，fb={forward_bars}）:")
                print(
                    f"         范围: [{valid_log_returns_clean_diag.min():.6f}, {valid_log_returns_clean_diag.max():.6f}]"
                )
                print(
                    f"         均值: {valid_log_returns_clean_diag.mean():.6f}, 中位数: {valid_log_returns_clean_diag.median():.6f}"
                )
                print(
                    f"         标准差: {valid_log_returns_clean_diag.std():.6f}")
                # 检查是否有 Inf 或极端值
                inf_count = np.isinf(valid_log_returns_clean_diag).sum()
                if inf_count > 0:
                    print(
                        f"         ⚠️  警告: 检测到 {inf_count} 个 Inf 值（可能是价格=0导致）")
                extreme_log = (valid_log_returns_clean_diag.abs()
                               > 2.0).sum()  # log(3) ≈ 1.1, log(10) ≈ 2.3
                if extreme_log > 0:
                    extreme_ratio = extreme_log / len(
                        valid_log_returns_clean_diag) * 100
                    print(
                        f"         ⚠️  警告: {extreme_log} 个极端 log return（|log|>2.0），占比 {extreme_ratio:.2f}%"
                    )

            # 确保索引与 symbol_feat_df 对齐（特征工程可能改变了索引）
            # 如果特征工程删除了某些行，我们需要对齐
            if not valid_log_returns.index.equals(symbol_feat_df.index):
                # 使用 symbol_feat_df 的索引，从 valid_log_returns 中提取对应的值
                valid_log_returns = valid_log_returns.reindex(
                    symbol_feat_df.index)

            # Winsorize at fixed percentiles
            # 🔧 OPTIMIZATION: 根据 forward_bars 动态调整边界
            # 对于短期预测（fb=1），使用 ±50% 收益率
            # 对于中期预测（fb=5-15），使用 ±100% 收益率
            # 对于长期预测（fb=45），使用 ±200% 收益率
            valid_log_returns_clean = valid_log_returns.dropna()
            if len(valid_log_returns_clean) > 0:
                # 根据 forward_bars 动态调整 log return 边界
                if forward_bars <= 1:
                    max_log_return = np.log(1.5)  # ±50% 收益率
                elif forward_bars <= 15:
                    max_log_return = np.log(2.0)  # ±100% 收益率
                else:
                    max_log_return = np.log(3.0)  # ±200% 收益率

                # Winsorize（限制在 1% 和 99% 分位数）
                try:
                    winsorized_values = winsorize(
                        valid_log_returns_clean.values,
                        limits=(0.01, 0.01)  # 限制在 1% 和 99% 分位数
                    )
                    # 进一步限制在合理范围内（根据 forward_bars 动态调整）
                    winsorized_values = np.clip(winsorized_values,
                                                -max_log_return,
                                                max_log_return)

                    # 转换回 simple return（用于兼容现有代码）
                    future_return_values = np.exp(winsorized_values) - 1

                    # 对齐索引（确保 future_return 与 symbol_feat_df 的索引一致）
                    # valid_log_returns_clean 的索引应该已经是 symbol_feat_df.index 的子集
                    future_return_series = pd.Series(
                        future_return_values,
                        index=valid_log_returns_clean.index)

                    # 确保索引对齐（使用 symbol_feat_df 的索引）
                    symbol_feat_df["future_return"] = (
                        future_return_series.reindex(symbol_feat_df.index,
                                                     fill_value=np.nan))

                    # 🔧 OPTIMIZATION: 验证 future_return 的合理性
                    future_return_stats = symbol_feat_df[
                        "future_return"].dropna()
                    if len(future_return_stats) > 0:
                        fr_max = future_return_stats.max()
                        fr_min = future_return_stats.min()
                        fr_abs_max = future_return_stats.abs().max()
                        # 检查是否有极端值（>500% 收益率）
                        extreme_count = (future_return_stats.abs() > 5.0).sum()
                        if extreme_count > 0:
                            extreme_ratio = (extreme_count /
                                             len(future_return_stats) * 100)
                            print(
                                f"      ⚠️  警告: {symbol} 仍有 {extreme_count} 个极端值（>500%），"
                                f"占比 {extreme_ratio:.2f}%")
                            print(
                                f"         future_return 范围: [{fr_min:.4f}, {fr_max:.4f}], "
                                f"max_abs={fr_abs_max:.4f}")
                            # 进一步限制极端值
                            symbol_feat_df.loc[
                                symbol_feat_df["future_return"].abs() > 5.0,
                                "future_return"] = np.nan

                    max_simple_return = (np.exp(max_log_return) - 1) * 100
                    print(
                        f"      ✅ 使用 log return + winsorize（限制在 ±{max_simple_return:.0f}% 收益率，fb={forward_bars}）"
                    )

                    # 🔍 DIAGNOSTIC: 打印处理后的 future_return 统计
                    future_return_final = symbol_feat_df[
                        "future_return"].dropna()
                    if len(future_return_final) > 0:
                        print(f"      🔍 Future Return 诊断（处理后）:")
                        print(
                            f"         范围: [{future_return_final.min():.6f}, {future_return_final.max():.6f}]"
                        )
                        print(
                            f"         均值: {future_return_final.mean():.6f}, 中位数: {future_return_final.median():.6f}"
                        )
                        print(f"         标准差: {future_return_final.std():.6f}")
                        print(
                            f"         绝对值最大值: {future_return_final.abs().max():.6f} ({future_return_final.abs().max()*100:.2f}%)"
                        )
                        # 检查是否还有极端值
                        extreme_final = (future_return_final.abs()
                                         > 1.0).sum()  # >100%
                        if extreme_final > 0:
                            extreme_ratio = extreme_final / len(
                                future_return_final) * 100
                            print(
                                f"         ⚠️  警告: 仍有 {extreme_final} 个极端值（>100%），占比 {extreme_ratio:.2f}%"
                            )
                except Exception as e:
                    print(f"      ⚠️  Winsorize 失败，使用原始 log return: {e}")
                    # 回退到 log return（不 winsorize）
                    future_return_values = np.exp(
                        valid_log_returns_clean.values) - 1
                    future_return_series = pd.Series(
                        future_return_values,
                        index=valid_log_returns_clean.index)
                    symbol_feat_df[
                        "future_return"] = future_return_series.reindex(
                            symbol_feat_df.index, fill_value=np.nan)
            else:
                # 如果没有有效数据，使用 simple return 作为回退
                simple_returns = (
                    symbol_resampled["close"].shift(-forward_bars) /
                    symbol_resampled["close"] - 1).iloc[:-forward_bars]
                symbol_feat_df["future_return"] = simple_returns.reindex(
                    symbol_feat_df.index, fill_value=np.nan)
                print(f"      ⚠️  使用 simple return（log return 计算失败）")
        except ImportError:
            # 如果 scipy 不可用，使用 simple return
            print(f"      ⚠️  scipy 不可用，使用 simple return")
            simple_returns = (symbol_resampled["close"].shift(-forward_bars) /
                              symbol_resampled["close"] -
                              1).iloc[:-forward_bars]
            symbol_feat_df["future_return"] = simple_returns.reindex(
                symbol_feat_df.index, fill_value=np.nan)

        # future_volatility（基于 future_return）
        # 🔒 CRITICAL FIX: 不能使用 rolling std，因为会引入未来信息
        # future_volatility[t] = std(future_return[t:t+window]) 需要 future_return[t+1], ..., future_return[t+window-1]
        # 但这些值对应的是未来的收益（如 future_return[t+1] 需要 close[t+1+fb]），引入了未来信息
        # ✅ 正确做法：使用 abs(future_return) 或 future_return^2 作为波动代理
        symbol_feat_df["future_volatility"] = symbol_feat_df[
            "future_return"].abs()

        # 🔒 CRITICAL: 确保 symbol 列存在且值正确
        # 特征工程后 symbol 列应该已经被恢复（在上面的代码中），这里只是双重检查
        if "symbol" not in symbol_feat_df.columns:
            symbol_feat_df["symbol"] = symbol
        elif not (symbol_feat_df["symbol"] == symbol).all():
            symbol_feat_df["symbol"] = symbol

        # 5. 删除 NaN 值（标签相关的）
        # 🔒 CRITICAL: 确保在 dropna 时保留 symbol 列
        before_dropna = len(symbol_feat_df)
        # 保存 symbol 列（如果存在）
        symbol_col = symbol_feat_df["symbol"].copy(
        ) if "symbol" in symbol_feat_df.columns else None
        symbol_feat_df = symbol_feat_df.dropna(
            subset=["future_return", "future_volatility"]).copy()
        # 重新添加 symbol 列（如果被 dropna 删除了）
        if symbol_col is not None and "symbol" not in symbol_feat_df.columns:
            symbol_feat_df["symbol"] = symbol_col.reindex(symbol_feat_df.index,
                                                          fill_value=symbol)
        elif symbol_col is not None:
            # 确保 symbol 列的值正确（dropna 可能改变了索引）
            symbol_feat_df["symbol"] = symbol
        else:
            # 如果 symbol 列不存在，添加它
            symbol_feat_df["symbol"] = symbol
        after_dropna = len(symbol_feat_df)

        after_truncate = len(symbol_feat_df)

        # 统计信息
        metadata["symbol_stats"][symbol] = {
            "raw_samples": len(symbol_raw),
            "resampled_samples": len(symbol_resampled),
            "after_dropna": after_dropna,
            "after_truncate": after_truncate,
            "final_samples": len(symbol_feat_df),
        }
        metadata["total_samples_after"] += len(symbol_feat_df)

        # 🔒 CRITICAL: 最终验证 symbol 列存在且正确
        if "symbol" not in symbol_feat_df.columns:
            raise ValueError(f"❌ 严重错误：{symbol} 的最终数据缺少 symbol 列！"
                             "这会导致多标的合并不安全。")
        if not (symbol_feat_df["symbol"] == symbol).all():
            print(f"      ⚠️  警告: {symbol} 的最终数据中 symbol 列值不一致，已修正")
            symbol_feat_df["symbol"] = symbol

        print(f"      ✅ {symbol} 处理完成: {len(symbol_feat_df):,} 条最终数据")
        print(
            f"         原始: {len(symbol_raw):,} → Resample: {len(symbol_resampled):,} → "
            f"Dropna: {after_dropna:,} → 截断: {after_truncate:,}")
        # 验证 symbol 列
        if "symbol" in symbol_feat_df.columns:
            print(
                f"         ✅ Symbol 列验证通过: {symbol_feat_df['symbol'].unique()}"
            )

        all_processed_dfs.append(symbol_feat_df)
        print()

    # 7. 最后合并所有标的
    if not all_processed_dfs:
        raise ValueError("没有成功处理任何标的的数据")

    print(f"{'='*70}")
    print(f"🔗 合并所有标的的数据...")
    final_df = pd.concat(all_processed_dfs, axis=0,
                         ignore_index=False).sort_index()
    print(f"   最终数据量: {len(final_df):,} 条")

    # 🔒 CRITICAL: 验证合并后的数据必须包含 symbol 列
    if "symbol" not in final_df.columns:
        raise ValueError("❌ 严重错误：合并后的数据缺少 symbol 列！"
                         "没有 symbol 信息的多标的合并是不安全的，会导致："
                         "1. 标签混淆（同一时间戳多个资产的标签混在一起）"
                         "2. 评估失真（无法按标的分组评估）"
                         "3. 模型学偏（模型不知道样本属于哪个资产）"
                         "4. 推理时无法确定预测适用于哪个标的")

    # 🔒 CRITICAL: 检查是否有 UNKNOWN symbol（可能表示 symbol 信息缺失）
    if "UNKNOWN" in final_df["symbol"].unique():
        unknown_count = (final_df["symbol"] == "UNKNOWN").sum()
        unknown_ratio = unknown_count / len(final_df) * 100
        print(
            f"\n   ⚠️  警告: 检测到 {unknown_count} 个样本的 symbol 为 'UNKNOWN'（占比 {unknown_ratio:.2f}%）"
        )
        print(f"      这可能表示某些数据文件缺少 symbol 信息")
        print(f"      💡 建议：检查数据文件，确保所有文件都包含 symbol 列或文件名包含标的标识")
        if unknown_ratio > 10:
            print(f"      🚨 严重警告：UNKNOWN 样本占比过高（>10%），建议停止训练并修复数据")
            raise ValueError(f"UNKNOWN symbol 样本占比过高（{unknown_ratio:.2f}%），"
                             "这会导致多标的合并不安全。请检查数据文件并确保所有文件都包含 symbol 信息。")

    # 🔒 CRITICAL: 检查是否有重复的时间戳（同一时间多个标的的数据）
    duplicate_timestamps = final_df.index.duplicated(keep=False)
    if duplicate_timestamps.any():
        duplicate_count = duplicate_timestamps.sum()
        duplicate_ratio = duplicate_count / len(final_df) * 100
        print(
            f"\n   ✅ 检测到 {duplicate_count} 个重复时间戳（占比 {duplicate_ratio:.2f}%）")
        print(f"      这是正常的（多标的训练中，同一时间可能有多个标的的数据）")
        print(f"      但必须通过 symbol 列区分它们")

        # 验证：重复时间戳的样本是否都有不同的 symbol
        duplicate_mask = final_df.index.duplicated(keep=False)
        if duplicate_mask.any():
            duplicate_df = final_df[duplicate_mask]
            # 检查是否有相同时间戳但相同 symbol 的情况（这不应该发生）
            duplicate_same_symbol = duplicate_df.groupby(
                [duplicate_df.index, "symbol"]).size()
            if (duplicate_same_symbol > 1).any():
                print(f"      🚨 严重警告：检测到相同时间戳且相同 symbol 的重复样本！")
                print(f"         这可能是数据问题，需要检查")
                problem_pairs = duplicate_same_symbol[duplicate_same_symbol >
                                                      1]
                print(f"         问题样本数: {len(problem_pairs)}")
            else:
                print(f"      ✅ 验证通过：所有重复时间戳的样本都有不同的 symbol")

    print(f"   标的分布:")
    if "symbol" in final_df.columns:
        for symbol, count in final_df["symbol"].value_counts().items():
            pct = count / len(final_df) * 100
            print(f"      {symbol}: {count:,} ({pct:.1f}%)")

    # 🔍 DIAGNOSTIC: 打印合并后的 future_return 统计（按标的）
    print(f"\n   🔍 合并后 Future Return 诊断（按标的）:")
    if "symbol" in final_df.columns and "future_return" in final_df.columns:
        for symbol in final_df["symbol"].unique():
            symbol_mask = final_df["symbol"] == symbol
            symbol_fr = final_df.loc[symbol_mask, "future_return"].dropna()
            if len(symbol_fr) > 0:
                print(f"      {symbol}:")
                print(f"         样本数: {len(symbol_fr):,}")
                print(
                    f"         范围: [{symbol_fr.min():.6f}, {symbol_fr.max():.6f}]"
                )
                print(
                    f"         均值: {symbol_fr.mean():.6f}, 中位数: {symbol_fr.median():.6f}"
                )
                print(f"         标准差: {symbol_fr.std():.6f}")
                print(
                    f"         绝对值最大值: {symbol_fr.abs().max():.6f} ({symbol_fr.abs().max()*100:.2f}%)"
                )
                # 检查极端值
                extreme_count = (symbol_fr.abs() > 1.0).sum()  # >100%
                if extreme_count > 0:
                    extreme_ratio = extreme_count / len(symbol_fr) * 100
                    print(
                        f"         ⚠️  警告: {extreme_count} 个极端值（>100%），占比 {extreme_ratio:.2f}%"
                    )

        # 🔍 DIAGNOSTIC: 检查不同标的的 future_return 分布是否一致
        print(f"\n   🔍 Future Return 分布一致性检查:")
        fr_stats_by_symbol = {}
        for symbol in final_df["symbol"].unique():
            symbol_mask = final_df["symbol"] == symbol
            symbol_fr = final_df.loc[symbol_mask, "future_return"].dropna()
            if len(symbol_fr) > 0:
                fr_stats_by_symbol[symbol] = {
                    "mean": symbol_fr.mean(),
                    "std": symbol_fr.std(),
                    "median": symbol_fr.median(),
                    "abs_max": symbol_fr.abs().max(),
                }

        if len(fr_stats_by_symbol) > 1:
            # 检查均值和标准差是否差异过大
            means = [stats["mean"] for stats in fr_stats_by_symbol.values()]
            stds = [stats["std"] for stats in fr_stats_by_symbol.values()]
            abs_maxs = [
                stats["abs_max"] for stats in fr_stats_by_symbol.values()
            ]

            mean_range = max(means) - min(means)
            std_range = max(stds) - min(stds)
            abs_max_range = max(abs_maxs) - min(abs_maxs)

            print(
                f"      均值范围: [{min(means):.6f}, {max(means):.6f}], 差异: {mean_range:.6f}"
            )
            print(
                f"      标准差范围: [{min(stds):.6f}, {max(stds):.6f}], 差异: {std_range:.6f}"
            )
            print(
                f"      绝对值最大值范围: [{min(abs_maxs):.6f}, {max(abs_maxs):.6f}], 差异: {abs_max_range:.6f}"
            )

            # 如果差异过大，可能是归一化问题
            if abs_max_range > 10.0:  # 差异 > 1000%
                print(
                    f"      🚨 严重警告: 不同标的的 future_return 绝对值最大值差异过大（{abs_max_range:.2f}）"
                )
                print(f"         这可能表明：")
                print(f"         1. 不同标的的 future_return 计算方式不一致")
                print(f"         2. 某些标的的数据有问题（价格跳空、数据错误）")
                print(f"         3. 需要按标的分别归一化 future_return")
            elif mean_range > 1.0 or std_range > 1.0:
                print(f"      ⚠️  警告: 不同标的的 future_return 分布差异较大")
                print(
                    f"         建议：考虑按标的分别归一化 future_return（Z-score 或 MinMax）")

    print(f"{'='*70}\n")

    return final_df, metadata
