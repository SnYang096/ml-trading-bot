from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pywt


def freedman_diaconis_bins(data: np.ndarray, min_bins: int = 10, max_bins: int = 100) -> int:
    """
    使用 Freedman–Diaconis rule 自动计算直方图 bin 数量。
    
    该方法特别适合处理非正态分布或含异常值的数据，比传统的 Sturges 公式更鲁棒。
    
    公式：
        bin_width = 2 * IQR(x) / n^(1/3)
        bins = ceil((max(x) - min(x)) / bin_width)
    
    Args:
        data: 数据序列
        min_bins: 最小 bin 数量（默认 10）
        max_bins: 最大 bin 数量（默认 100）
    
    Returns:
        计算得到的 bin 数量（限制在 [min_bins, max_bins] 范围内）
    
    Examples:
        >>> x = np.random.normal(100, 5, 1000)
        >>> bins = freedman_diaconis_bins(x)
        >>> assert 20 <= bins <= 40
    """
    data = np.asarray(data)
    n = len(data)
    
    if n < 2:
        return min_bins
    
    # 计算 IQR（四分位距）
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    
    # 如果 IQR = 0（所有值相同或几乎相同），使用 fallback
    if iqr <= 0:
        # fallback: 使用数据范围，假设需要 10 个 bins
        data_range = np.ptp(data)  # peak-to-peak (max - min)
        if data_range <= 1e-10:
            # 所有值完全相同，使用默认值
            return min_bins
        # 使用固定比例：假设每个 bin 占数据范围的 1/10
        bin_width = data_range / 10.0
    else:
        # 标准 FD rule：bin_width = 2 * IQR / n^(1/3)
        bin_width = 2 * iqr / (n ** (1/3))
    
    if bin_width <= 0:
        return min_bins
    
    # 计算 bins 数量：ceil((max - min) / bin_width)
    data_range = np.max(data) - np.min(data)
    bins = int(np.ceil(data_range / bin_width))
    
    # 限制在合理范围内
    return int(np.clip(bins, min_bins, max_bins))


@dataclass
class VolumeProfileResult:
    hist: np.ndarray
    edges: np.ndarray
    centers: np.ndarray
    price_min: float
    price_max: float
    price_denoised: Optional[np.ndarray] = None


def compute_wpt_volume_profile(
    price_window: np.ndarray,
    volume_window: np.ndarray,
    bins: int | str = "auto",
    wavelet: str = "db4",
    level: int = 4,
    drop_high_freq: bool = True,
) -> Optional[VolumeProfileResult]:
    """
    使用 WPT 降噪后的价格序列构建 volume profile 直方图。

    将该函数独立出来，供 VPVR 与 baseline POC/HAL 共享，避免重复计算。

    Args:
        price_window: 价格序列窗口
        volume_window: 成交量序列窗口（与 price_window 等长）
        bins: 直方图 bins 数量。如果为 "auto"（默认），则使用 Freedman-Diaconis rule 自动计算
        wavelet: 小波函数名称（默认 "db4"）
        level: WPT 分解层数（默认 4）
        drop_high_freq: 是否剔除高频子带进行降噪（默认 True）
            - 如果为 True，将按频率排序剔除最高频的 25% 子带

    Returns:
        VolumeProfileResult 对象，包含：
        - hist: 成交量直方图
        - edges: bins 边界
        - centers: bins 中心价格
        - price_min: 价格最小值
        - price_max: 价格最大值
        - price_denoised: 降噪后的价格序列（可选，用于上层特征扩展）
    """

    if price_window is None or volume_window is None:
        return None

    if len(price_window) != len(volume_window) or len(price_window) < 10:
        return None

    price_window = np.asarray(price_window, dtype=float)
    volume_window = np.asarray(volume_window, dtype=float)

    # Step 1: WPT 降噪（改进版：按频率排序剔除高频子带）
    # 修复1: 动态限制 level，防止超过最大允许分解层数
    # 先验证小波函数是否有效（避免在 dwt_max_level 阶段抛出异常）
    try:
        # 验证小波函数是否有效
        _ = pywt.Wavelet(wavelet)
        max_level = pywt.dwt_max_level(len(price_window), wavelet)
        actual_level = min(level, max_level) if max_level > 0 else 1
    except (ValueError, RuntimeError, TypeError):
        # 无效小波函数，直接使用原始价格
        price_denoised = price_window
        actual_level = 0
    
    if actual_level < 1:
        # 无法分解，直接使用原始价格
        price_denoised = price_window
    else:
        try:
            wp = pywt.WaveletPacket(
                data=price_window, wavelet=wavelet, mode="symmetric", maxlevel=actual_level
            )
            if drop_high_freq:
                # 使用 "freq" 排序获取按频率升序排列的子带
                freq_nodes = wp.get_level(actual_level, "freq")
                if len(freq_nodes) > 0:
                    # 剔除最高频的 25% 子带（至少剔除1个）
                    n_drop = max(1, len(freq_nodes) // 4)
                    for node in freq_nodes[-n_drop:]:
                        wp[node.path].data = np.zeros_like(node.data)
            price_denoised = wp.reconstruct(update=True)
            
            # 修复2: 强制对齐长度（WPT 重建后长度可能不一致）
            if len(price_denoised) != len(price_window):
                # 截断到原始长度（最安全的方法）
                if len(price_denoised) > len(price_window):
                    price_denoised = price_denoised[:len(price_window)]
                else:
                    # 如果重建后长度更短，使用线性插值扩展（较少见）
                    # 或者直接使用原始价格（更安全）
                    price_denoised = price_window
        except (ValueError, RuntimeError, TypeError) as e:
            # 修复4: 只捕获预期的小波相关异常，不捕获 MemoryError、KeyboardInterrupt 等
            price_denoised = price_window

    # Step 2: 过滤无效值
    valid_mask = (
        np.isfinite(price_denoised)
        & np.isfinite(volume_window)
        & (volume_window > 0)
    )

    if not np.any(valid_mask):
        return None

    price_valid = price_denoised[valid_mask]
    volume_valid = volume_window[valid_mask]

    if len(price_valid) < 10:
        return None

    price_min = price_valid.min()
    price_max = price_valid.max()

    # 检查价格范围（考虑浮点数精度问题）
    price_range = price_max - price_min
    if not np.isfinite(price_min) or not np.isfinite(price_max) or price_range <= 1e-10:
        return None

    # Step 3: 动态 bins 计算（如果 bins="auto"）
    if isinstance(bins, str) and bins == "auto":
        bins = freedman_diaconis_bins(price_valid, min_bins=10, max_bins=100)
    
    # 修复3: 防止 bins > 数据点数（避免过度分箱）
    bins = min(bins, len(price_valid))
    if bins < 1:
        bins = 1

    # Step 4: 构建 volume profile
    hist, edges = np.histogram(
        price_valid,
        bins=bins,
        range=(price_min, price_max),
        weights=volume_valid,
    )

    if np.sum(hist) <= 0:
        return None

    centers = (edges[:-1] + edges[1:]) / 2

    return VolumeProfileResult(
        hist=hist,
        edges=edges,
        centers=centers,
        price_min=float(price_min),
        price_max=float(price_max),
        price_denoised=price_denoised,
    )

