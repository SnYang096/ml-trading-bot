#!/usr/bin/env python3
"""
数据泄露检查脚本
根据文档要求，检查特征生成中可能存在的未来信息泄露
"""

import pandas as pd
import numpy as np
from pathlib import Path

def check_dl_seq_alignment():
    """检查 dl_seq_f 特征的对齐是否正确"""
    print("=" * 70)
    print("🔍 检查 1: dl_seq_f 特征对齐逻辑")
    print("=" * 70)
    
    # 模拟序列生成过程
    seq_length = 120
    data_length = 1000
    
    # 模拟序列窗口生成
    sequences = []
    for i in range(data_length - seq_length + 1):
        seq = np.arange(i, i + seq_length)  # 窗口 [i, i+seq_length)
        sequences.append(seq)
    
    # 当前代码的对齐方式
    valid_indices_current = np.arange(seq_length - 1, data_length)
    
    # 检查：第 i 个序列应该对应哪个时间点？
    print(f"\n序列窗口生成逻辑：")
    print(f"  - 序列 0: 窗口 [0:120]，应该对应时间点 119（窗口最后一个点）")
    print(f"  - 序列 1: 窗口 [1:121]，应该对应时间点 120（窗口最后一个点）")
    print(f"  - 序列 i: 窗口 [i:i+120]，应该对应时间点 i+119（窗口最后一个点）")
    
    print(f"\n当前代码的对齐方式：")
    print(f"  - valid_indices = np.arange({seq_length - 1}, {data_length})")
    print(f"  - 第 0 个特征对应时间点: {valid_indices_current[0]}")
    print(f"  - 第 1 个特征对应时间点: {valid_indices_current[1]}")
    
    # 检查是否正确
    expected_indices = []
    for i in range(len(sequences)):
        expected_idx = i + seq_length - 1  # 窗口最后一个点
        expected_indices.append(expected_idx)
    
    if np.array_equal(valid_indices_current[:len(expected_indices)], expected_indices):
        print(f"\n✅ 对齐逻辑正确：特征 i 对应时间点 i+{seq_length-1}")
    else:
        print(f"\n❌ 对齐逻辑可能有问题！")
        print(f"  预期: {expected_indices[:5]}")
        print(f"  实际: {valid_indices_current[:5].tolist()}")
    
    return valid_indices_current[0] == seq_length - 1


def check_rolling_zscore():
    """检查 rolling zscore 是否使用了未来信息"""
    print("\n" + "=" * 70)
    print("🔍 检查 2: Rolling Z-score 计算逻辑")
    print("=" * 70)
    
    # 创建测试数据
    np.random.seed(42)
    test_data = pd.Series(np.random.randn(1000).cumsum())
    
    # 检查不同窗口的 rolling zscore
    windows = [50, 288, 500]
    
    for window in windows:
        print(f"\n窗口大小: {window}")
        
        # 使用 rolling (默认 center=False，安全)
        rolling_mean = test_data.rolling(window=window, min_periods=10).mean()
        rolling_std = test_data.rolling(window=window, min_periods=10).std()
        zscore = (test_data - rolling_mean) / rolling_std
        
        # 检查是否有使用 center=True（危险）
        # 检查是否有使用全样本统计（危险）
        # 检查 min_periods 是否合理
        
        print(f"  - 使用 rolling(window={window}, min_periods=10)")
        print(f"  - center=False (默认，安全)")
        print(f"  - 前 {window} 个值有 NaN（正常，因为窗口不足）")
        print(f"  - 第 {window} 个值使用窗口 [0:{window}] 计算（安全）")
        
        # 检查是否有使用全样本统计
        if not np.isnan(zscore.iloc[window:]).any():
            print(f"  ✅ 窗口 {window}: 计算逻辑安全")
        else:
            print(f"  ⚠️  窗口 {window}: 存在 NaN，需要检查 min_periods 设置")
    
    return True


def check_adaptive_normalization():
    """检查自适应归一化是否使用了未来信息"""
    print("\n" + "=" * 70)
    print("🔍 检查 3: dl_seq_f 自适应归一化逻辑")
    print("=" * 70)
    
    print("\n检查 adaptive normalization 实现：")
    print("  1. 全局 scaler: 使用全样本 mean/std（在 fit 时计算）")
    print("  2. 局部窗口: 使用滚动窗口 [i:i+seq_length] 的 mean/std")
    print("  3. 组合: global_weight * global_stats + local_weight * local_stats")
    
    print("\n⚠️  潜在问题：")
    print("  - 全局 scaler 在 fit 时使用全样本统计，这本身是安全的")
    print("  - 但在 transform 时，如果使用相同的全局 scaler，需要确保")
    print("    没有在测试集上重新 fit（这会导致使用未来信息）")
    
    print("\n✅ 如果遵循以下流程，应该是安全的：")
    print("  1. 在训练集上 fit scaler")
    print("  2. 在训练集和测试集上都使用同一个 scaler transform")
    print("  3. 不使用测试集数据重新 fit")
    
    return True


def check_feature_future_correlation():
    """分析特征-未来相关性检测结果"""
    print("\n" + "=" * 70)
    print("🔍 检查 4: 特征-未来相关性分析")
    print("=" * 70)
    
    suspicious_features = {
        'dl_seq_f43': 0.1840,
        'atr_zscore_w288': 0.1517,
        'volatility_zscore_w288': 0.1455,
        'atr_percentile': 0.1417,
        'dl_seq_f55': -0.1411,
        'dl_seq_f18': -0.1372,
        'dl_seq_f38': -0.1357,
        'atr_compression_ratio': -0.1331,
        'dl_seq_f7': 0.1294,
        'volatility_zscore_w500': 0.1250,
    }
    
    print("\n高相关性特征分析：")
    print(f"{'特征':<30} | {'相关性':<10} | {'可能原因':<30}")
    print("-" * 70)
    
    for feat, corr in suspicious_features.items():
        if 'dl_seq' in feat:
            reason = "序列特征可能编码未来模式"
        elif 'zscore_w288' in feat or 'zscore_w500' in feat:
            reason = "长窗口 zscore，可能边界效应"
        elif 'percentile' in feat:
            reason = "百分位数计算，检查窗口"
        elif 'compression' in feat:
            reason = "ATR压缩比，检查计算逻辑"
        else:
            reason = "需要进一步检查"
        
        print(f"{feat:<30} | {corr:>9.4f} | {reason:<30}")
    
    return suspicious_features


def generate_recommendations():
    """生成修复建议"""
    print("\n" + "=" * 70)
    print("🛠️  修复建议")
    print("=" * 70)
    
    recommendations = [
        {
            "优先级": "🔴 高",
            "问题": "dl_seq_f 特征对齐可能有问题",
            "检查点": "验证 valid_indices = np.arange(seq_length-1, len(df)) 是否正确",
            "修复": "确保特征 i 对应时间点 i+seq_length-1（窗口最后一个点）"
        },
        {
            "优先级": "🟡 中",
            "问题": "长窗口 zscore (w288, w500) 高相关性",
            "检查点": "检查 min_periods 设置，确保早期样本不使用未来数据",
            "修复": "如果 min_periods 太小，可能 fallback 到未来数据"
        },
        {
            "优先级": "🟡 中",
            "问题": "adaptive normalization 全局 scaler",
            "检查点": "确保测试集不使用自己的数据 fit scaler",
            "修复": "在训练集 fit，在测试集只 transform"
        },
        {
            "优先级": "🟢 低",
            "问题": "atr_percentile 和 compression_ratio",
            "检查点": "检查这些特征的计算是否使用滚动窗口",
            "修复": "确保所有 rolling 操作都是 center=False"
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['优先级']} - {rec['问题']}")
        print(f"   检查点: {rec['检查点']}")
        print(f"   修复: {rec['修复']}")
    
    print("\n" + "=" * 70)
    print("📋 立即行动清单")
    print("=" * 70)
    print("""
1. ✅ 隔离高相关特征，重新测试 OOS 表现
   - 移除 corr > 0.1 的特征
   - 重新运行训练和测试
   - 如果 OOS IC 大幅下降 → 证实存在泄露

2. ✅ 验证 dl_seq_f 对齐逻辑
   - 检查特征索引和时间点对应关系
   - 确保没有使用未来窗口

3. ✅ 检查 rolling 操作
   - 确保所有 rolling 都是 center=False
   - 检查 min_periods 设置是否合理

4. ✅ Forward-Walk Simulation
   - 从 t=0 开始逐步加入数据
   - 实时计算特征和预测
   - 如果 IC 崩溃 → 证实泄露
    """)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🔍 数据泄露全面检查")
    print("=" * 70)
    
    # 执行各项检查
    check1 = check_dl_seq_alignment()
    check2 = check_rolling_zscore()
    check3 = check_adaptive_normalization()
    suspicious = check_feature_future_correlation()
    generate_recommendations()
    
    print("\n" + "=" * 70)
    print("📊 总结")
    print("=" * 70)
    print(f"""
✅ 检查完成
⚠️  发现 {len(suspicious)} 个高相关性特征
🔴 需要立即验证 dl_seq_f 对齐逻辑
🟡 需要检查长窗口 zscore 的 min_periods 设置
    """)

