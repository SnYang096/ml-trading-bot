from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pywt


@dataclass
class VolumeProfileResult:
    hist: np.ndarray
    edges: np.ndarray
    centers: np.ndarray
    price_min: float
    price_max: float


def compute_wpt_volume_profile(
    price_window: np.ndarray,
    volume_window: np.ndarray,
    bins: int = 50,
    wavelet: str = "db4",
    level: int = 4,
    drop_high_freq: bool = True,
) -> Optional[VolumeProfileResult]:
    """
    使用 WPT 降噪后的价格序列构建 volume profile 直方图。

    将该函数独立出来，供 VPVR 与 baseline POC/HAL 共享，避免重复计算。
    """

    if price_window is None or volume_window is None:
        return None

    if len(price_window) != len(volume_window) or len(price_window) < 10:
        return None

    price_window = np.asarray(price_window, dtype=float)
    volume_window = np.asarray(volume_window, dtype=float)

    # Step 1: WPT 降噪
    try:
        wp = pywt.WaveletPacket(
            data=price_window, wavelet=wavelet, mode="symmetric", maxlevel=level
        )
        if drop_high_freq:
            for node in wp.get_level(level, "freq"):
                if all(c == "d" for c in node.path):
                    wp[node.path].data = np.zeros_like(wp[node.path].data)
        price_denoised = wp.reconstruct()
    except Exception:
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

    if not np.isfinite(price_min) or not np.isfinite(price_max) or price_max <= price_min:
        return None

    # Step 3: 构建 volume profile
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
    )

