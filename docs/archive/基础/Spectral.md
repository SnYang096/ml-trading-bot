import numpy as np
from scipy.signal import periodogram, welch
import pandas as pd

def compute_spectral_features(returns, fs=1.0):
    """
    计算谱特征（输入：收益率序列，fs=采样频率，如 60 表示 1分钟数据）
    """
    # 方法1：周期图（Periodogram）
    freqs, psd = periodogram(returns, fs)
    
    # 主导频率
    dominant_freq = freqs[np.argmax(psd)]
    
    # 谱熵（衡量频率分布的混乱度）
    psd_norm = psd / np.sum(psd)
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-8))
    
    # 低频能量（如 < 0.01 Hz，对应周期 > 100分钟）
    low_band = psd[(freqs >= 0.001) & (freqs < 0.01)].sum()
    
    # 高频能量（> 0.05 Hz，对应周期 < 20分钟）
    high_band = psd[freqs >= 0.05].sum()
    
    # 谱质心
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
    
    return {
        'dominant_freq': dominant_freq,
        'spectral_entropy': spectral_entropy,
        'low_freq_power': low_band,
        'high_freq_power': high_band,
        'spectral_centroid': spectral_centroid
    }

# 使用示例
# returns = df['close'].pct_change().dropna()
# features = compute_spectral_features(returns, fs=60)  # 1分钟数据