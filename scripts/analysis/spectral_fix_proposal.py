"""
修复后的Spectral特征实现
使用滚动窗口计算spectral特征，生成时间序列
"""

import numpy as np
import pandas as pd
from scipy.signal import periodogram


def add_spectral_features_fixed(data: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """
    修复版：使用滚动窗口计算spectral特征

    Args:
        data: DataFrame with OHLCV and other columns
        window: 滚动窗口大小（默认100根K线）

    Returns:
        DataFrame with spectral features added
    """
    df = data.copy()

    # 定义需要计算光谱的信号源
    signal_sources = {
        "close": df["close"].values,
        "open": df["open"].values,
        "volume": df["volume"].values,
    }

    if "cvd" in df.columns:
        signal_sources["cvd"] = df["cvd"].values
    if "taker_buy_ratio" in df.columns:
        signal_sources["taker_buy_ratio"] = df["taker_buy_ratio"].values

    print(f"      光谱分析信号源: {list(signal_sources.keys())} (滚动窗口={window})")

    # 对每个信号源计算滚动spectral特征
    for source_name, source_data in signal_sources.items():
        print(f"        计算 {source_name} spectral features...")

        # 初始化特征数组
        n_samples = len(source_data)
        spectral_centroid = np.zeros(n_samples)
        spectral_bandwidth = np.zeros(n_samples)
        spectral_rolloff = np.zeros(n_samples)
        spectral_entropy = np.zeros(n_samples)
        dominant_freq = np.zeros(n_samples)

        # 滚动窗口计算
        for i in range(n_samples):
            if i < window:
                # 窗口不足，填充0或使用可用数据
                continue

            # 获取窗口数据
            window_data = source_data[i - window : i]

            # 移除NaN
            valid_data = window_data[np.isfinite(window_data)]

            if len(valid_data) < 10:
                continue

            try:
                # 方法1：使用periodogram计算功率谱密度（推荐）
                # 对收益率序列计算更有意义
                if source_name in ["close", "open"]:
                    # 对价格信号，先转换为收益率
                    returns = np.diff(valid_data) / (valid_data[:-1] + 1e-10)
                    freqs, psd = periodogram(returns, fs=1.0)
                else:
                    # 对volume、cvd等，直接使用
                    freqs, psd = periodogram(valid_data, fs=1.0)

                # 归一化
                psd_sum = np.sum(psd)
                if psd_sum == 0:
                    continue

                psd_norm = psd / psd_sum

                # 1. Spectral Centroid（谱质心）
                spectral_centroid[i] = np.sum(freqs * psd_norm)

                # 2. Spectral Bandwidth（谱带宽）
                spectral_bandwidth[i] = np.sqrt(
                    np.sum(((freqs - spectral_centroid[i]) ** 2) * psd_norm)
                )

                # 3. Spectral Rolloff（谱滚降，95%能量点）
                cumsum_psd = np.cumsum(psd_norm)
                rolloff_idx = np.where(cumsum_psd >= 0.95)[0]
                if len(rolloff_idx) > 0:
                    spectral_rolloff[i] = freqs[rolloff_idx[0]]

                # 4. Spectral Entropy（谱熵）
                spectral_entropy[i] = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

                # 5. Dominant Frequency（主导频率）
                dominant_freq[i] = freqs[np.argmax(psd)]

            except Exception as e:
                # 如果计算失败，保持为0
                continue

        # 添加到DataFrame
        df[f"{source_name}_spectral_centroid"] = spectral_centroid
        df[f"{source_name}_spectral_bandwidth"] = spectral_bandwidth
        df[f"{source_name}_spectral_rolloff"] = spectral_rolloff
        df[f"{source_name}_spectral_entropy"] = spectral_entropy
        df[f"{source_name}_spectral_dominant_freq"] = dominant_freq

    return df


# 对比：错误的实现 vs 正确的实现
def compare_implementations():
    """
    对比错误和正确的实现
    """
    print("=" * 80)
    print("🔍 Spectral特征实现对比")
    print("=" * 80)

    print("\n❌ 错误的实现（当前）：")
    print("```python")
    print("# 对整个序列计算一次")
    print("fft = np.fft.fft(valid_data)")
    print("spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)")
    print("")
    print("# 把标量赋值给整个列")
    print("df['close_spectral_centroid'] = spectral_centroid  # 所有行相同！")
    print("```")

    print("\n结果：")
    print("  Row 0: spectral_centroid = 0.123")
    print("  Row 1: spectral_centroid = 0.123  (相同)")
    print("  Row 2: spectral_centroid = 0.123  (相同)")
    print("  ...")
    print("  → LightGBM重要性 = 0（无信息量）")

    print("\n" + "=" * 80)
    print("✅ 正确的实现（修复后）：")
    print("```python")
    print("# 滚动窗口计算")
    print("for i in range(n_samples):")
    print("    window_data = source_data[i-window:i]")
    print("    freqs, psd = periodogram(window_data)")
    print("    spectral_centroid[i] = np.sum(freqs * psd) / np.sum(psd)")
    print("")
    print("# 时间序列赋值")
    print("df['close_spectral_centroid'] = spectral_centroid  # 每行不同！")
    print("```")

    print("\n结果：")
    print("  Row 0: spectral_centroid = 0.000  (窗口不足)")
    print("  Row 100: spectral_centroid = 0.123")
    print("  Row 101: spectral_centroid = 0.125  (不同)")
    print("  Row 102: spectral_centroid = 0.118  (不同)")
    print("  ...")
    print("  → LightGBM重要性 > 0（有信息量）")

    print("\n" + "=" * 80)
    print("💡 关键区别：")
    print("=" * 80)
    print("❌ 错误：对整个序列计算一次 → 标量 → 所有行相同")
    print("✅ 正确：滚动窗口计算 → 时间序列 → 每行不同")
    print("")
    print("📊 这就是为什么：")
    print("  - Hilbert有效：对每个时间点计算瞬时频率")
    print("  - Spectral无效：只计算一次，没有时间变化")


if __name__ == "__main__":
    # 示例：生成测试数据
    np.random.seed(42)
    n = 500
    test_data = pd.DataFrame(
        {
            "close": 100 + np.cumsum(np.random.randn(n) * 0.5),
            "open": 100 + np.cumsum(np.random.randn(n) * 0.5),
            "volume": np.abs(np.random.randn(n) * 1000 + 5000),
        }
    )

    print("测试修复后的实现...")
    result = add_spectral_features_fixed(test_data, window=100)

    print("\n✅ 修复后的特征（前10行）：")
    spectral_cols = [c for c in result.columns if "spectral" in c]
    print(result[spectral_cols].head(110).tail(10))

    print("\n验证时间序列性：")
    for col in spectral_cols[:3]:
        unique_vals = result[col].nunique()
        print(f"  {col}: {unique_vals} 个唯一值 (总{len(result)}行)")

    print("\n" + "=" * 80)
    compare_implementations()
