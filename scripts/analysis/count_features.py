"""统计项目中所有特征数量的脚本.

扫描所有特征工程模块，统计每个模块生成的特征数量，并生成详细报告。
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
import re

# Add src to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))


def count_basic_features() -> Dict[str, List[str]]:
    """统计 feature_engineering.py 中的基础特征."""
    features = {
        "RSI": ["rsi"],
        "MACD": ["macd", "macd_signal", "macd_histogram"],
        "Bollinger Bands": ["bb_upper", "bb_middle", "bb_lower"],
        "ATR": ["atr"],
        "ZigZag": ["zigzag"],
        "Price Features": ["price_change", "volatility"],
        "Volume Features": ["volume_sma", "volume_ratio"],
    }
    return features


def count_improved_features() -> Dict[str, List[str]]:
    """统计 feature_engineering_improved.py 中的改进特征."""
    features = count_basic_features()

    # Add improved features
    features["Normalized Features"] = [
        "bb_position",
        "rsi_normalized",
        "macd_normalized",
        "atr_normalized",
    ]

    features["Momentum Features"] = ["momentum_5", "momentum_10", "momentum_20"]

    features["Moving Averages"] = [
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_ratio_5_20",
        "sma_ratio_10_20",
    ]

    return features


def count_enhanced_features() -> Dict[str, List[str]]:
    """统计 feature_engineering_enhanced.py 中的增强特征."""
    features = {}

    # Basic features (from add_basic_features method)
    features["Price-based Features"] = [
        "returns",
        "log_returns",
        "price_change",
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_50",
        "ema_5",
        "ema_10",
        "ema_20",
        "ema_50",
    ]

    features["Volatility Features"] = ["volatility", "atr", "atr_normalized"]

    features["Momentum Features"] = [
        "momentum_5",
        "momentum_10",
        "momentum_20",
        "roc_5",
        "roc_10",
        "roc_20",
    ]

    features["RSI & Bollinger"] = ["rsi_14", "bb_upper", "bb_lower", "bb_position"]

    features["MACD"] = ["macd", "macd_signal", "macd_histogram"]

    features["Volume Features"] = ["volume_sma_20", "volume_ratio"]

    # Hurst features (per signal source)
    signal_sources = ["close", "open", "volume", "cvd", "taker_buy_ratio"]
    hurst_features_per_source = [
        "_hurst",
        "_hurst_deviation",
        "_hurst_trend_signal",
        "_hurst_mean_revert_signal",
        "_hurst_change",
        "_hurst_acceleration",
    ]
    features["Hurst Features"] = [
        f"{source}{feat}"
        for source in signal_sources
        for feat in hurst_features_per_source
    ]

    # WPT features (complex, estimate per source)
    # Each source gets: energy, mean, std per node + entropy + concentration etc
    # For level 3, we have 2^3 = 8 nodes
    # Per node: energy, mean, std, energy_ratio = ~4 features
    # Global: shannon_entropy, energy_concentration, high_low_ratio, dominant_band = 4 features
    # Total per source: 8*4 + 4 = 36 features
    wpt_features = []
    for source in signal_sources:
        for i in range(8):  # 8 nodes at level 3
            wpt_features.extend(
                [
                    f"{source}_wpt_{i}_energy",
                    f"{source}_wpt_{i}_mean",
                    f"{source}_wpt_{i}_std",
                    f"{source}_wpt_{i}_energy_ratio",
                ]
            )
        wpt_features.extend(
            [
                f"{source}_wpt_shannon_entropy",
                f"{source}_wpt_energy_concentration",
                f"{source}_wpt_high_low_ratio",
                f"{source}_wpt_dominant_band",
            ]
        )
    features["Wavelet Packet Transform"] = wpt_features

    # Hilbert features (per signal source)
    hilbert_features_per_source = [
        "_hilbert_amplitude",
        "_hilbert_phase",
        "_hilbert_frequency",
    ]
    features["Hilbert Transform"] = [
        f"{source}{feat}"
        for source in signal_sources
        for feat in hilbert_features_per_source
    ]

    # Spectral features (per signal source)
    spectral_features_per_source = [
        "_spectral_centroid",
        "_spectral_bandwidth",
        "_spectral_rolloff",
    ]
    features["Spectral Analysis"] = [
        f"{source}{feat}"
        for source in signal_sources
        for feat in spectral_features_per_source
    ]

    # Advanced derived features
    features["Advanced Derived"] = [
        "bb_width",
        "bb_width_normalized",
        "hl",
        "range_ratio_5bar",
        "compression_duration",
        "compression_energy",
        "atr_percentile",
        "volatility_reversal_score",
        "volatility_squeeze_flag",
        "price_range_symmetry",
        "volume_anomaly",
        "up_vol",
        "down_vol",
        "upvol_downvol_ratio",
        "roc_5",
        "acceleration_3",
        "price_vs_ema_distance",
        "momentum_persistence",
        "slope_consistency_score",
        "hour_of_day_sin",
        "hour_of_day_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "structure_tension",
        "trend_volatility_alignment",
        "compression_to_breakout_prob",
    ]

    # Order flow features
    features["Order Flow"] = [
        "order_flow_imbalance",
        "ofi_short",
        "ofi_medium",
        "ofi_long",
        "cumulative_ofi",
        "ofi_momentum_5",
        "ofi_momentum_20",
        "ofi_volatility",
        "delta_divergence_5",
        "delta_divergence_20",
        "divergence_strength",
        "cvd_short_trend",
        "cvd_short_momentum",
        "cvd_medium_trend",
        "cvd_medium_momentum",
        "cvd_long_trend",
        "cvd_short_medium_ratio",
        "cvd_medium_long_ratio",
        "cvd_trend_alignment",
        "cvd_norm_momentum",
        "cvd_norm_extreme",
        "tbr_momentum_5",
        "tbr_momentum_20",
        "tbr_extreme_buy",
        "tbr_extreme_sell",
        "tbr_neutral",
        "cvd_slope_3",
        "cvd_slope_10",
        "cvd_slope_30",
        "cvd_acceleration",
        "liquidity_drain",
        "liquidity_ratio",
        "buy_sell_pressure_ratio",
        "pressure_diff",
        "pressure_diff_norm",
        "volume_price_divergence",
    ]

    return features


def count_wavelet_features() -> Dict[str, List[str]]:
    """统计 feature_engineering_wavelet.py 中的小波特征."""
    features = count_improved_features()

    # Wavelet features for close price
    wavelet_base = [
        "wavelet_energy",
        "wavelet_entropy",
        "wavelet_std",
        "wavelet_skewness",
        "wavelet_kurtosis",
        "wavelet_approx_energy",
        "wavelet_approx_std",
        "wavelet_approx_mean",
    ]

    # Detail coefficients (4 levels)
    for i in range(1, 5):
        wavelet_base.extend(
            [
                f"wavelet_detail_{i}_energy",
                f"wavelet_detail_{i}_std",
                f"wavelet_detail_{i}_mean",
            ]
        )

    features["Wavelet Features (Close)"] = wavelet_base

    # Volume wavelet features
    features["Wavelet Features (Volume)"] = [f"volume_{feat}" for feat in wavelet_base]

    # Hilbert features
    features["Hilbert Transform"] = [
        "hilbert_amplitude",
        "hilbert_phase",
        "hilbert_frequency",
    ]

    # Spectral features
    features["Spectral Analysis"] = [
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
    ]

    return features


def count_dl_features(d_model: int = 64) -> Dict[str, List[str]]:
    """统计 dl_sequence_features.py 中的深度学习序列特征."""
    features = {
        "Deep Learning Sequence Features": [f"dl_seq_f{i}" for i in range(d_model)]
    }
    return features


def generate_report() -> str:
    """生成完整的特征统计报告."""
    report = []
    report.append("=" * 80)
    report.append("特征统计报告 (Feature Count Report)")
    report.append("=" * 80)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    all_features = {}
    total_count = 0

    # 1. Basic Features
    report.append("1. 基础特征工程 (feature_engineering.py)")
    report.append("-" * 80)
    basic_features = count_basic_features()
    module_total = 0
    for category, features in basic_features.items():
        count = len(features)
        module_total += count
        report.append(f"  {category}: {count} 个特征")
        report.append(
            f"    {', '.join(features[:5])}" + ("..." if len(features) > 5 else "")
        )
    report.append(f"  小计: {module_total} 个特征")
    report.append("")
    all_features["basic"] = basic_features
    total_count += module_total

    # 2. Improved Features
    report.append("2. 改进特征工程 (feature_engineering_improved.py)")
    report.append("-" * 80)
    improved_features = count_improved_features()
    module_total = 0
    for category, features in improved_features.items():
        count = len(features)
        module_total += count
        report.append(f"  {category}: {count} 个特征")
        report.append(
            f"    {', '.join(features[:5])}" + ("..." if len(features) > 5 else "")
        )
    report.append(f"  小计: {module_total} 个特征")
    report.append("")
    all_features["improved"] = improved_features
    total_count += module_total

    # 3. Enhanced Features (最复杂)
    report.append("3. 增强特征工程 (feature_engineering_enhanced.py)")
    report.append("-" * 80)
    enhanced_features = count_enhanced_features()
    module_total = 0
    for category, features in enhanced_features.items():
        count = len(features)
        module_total += count
        report.append(f"  {category}: {count} 个特征")
        if count <= 10:
            report.append(f"    {', '.join(features)}")
        else:
            report.append(f"    {', '.join(features[:5])}...")
    report.append(f"  小计: {module_total} 个特征")
    report.append("")
    all_features["enhanced"] = enhanced_features
    total_count += module_total

    # 4. Wavelet Features
    report.append("4. 小波特征工程 (feature_engineering_wavelet.py)")
    report.append("-" * 80)
    wavelet_features = count_wavelet_features()
    module_total = 0
    for category, features in wavelet_features.items():
        count = len(features)
        module_total += count
        report.append(f"  {category}: {count} 个特征")
        if count <= 10:
            report.append(f"    {', '.join(features)}")
        else:
            report.append(f"    {', '.join(features[:5])}...")
    report.append(f"  小计: {module_total} 个特征")
    report.append("")
    all_features["wavelet"] = wavelet_features
    total_count += module_total

    # 5. Deep Learning Features
    report.append("5. 深度学习序列特征 (dl_sequence_features.py)")
    report.append("-" * 80)
    dl_features = count_dl_features()
    module_total = 0
    for category, features in dl_features.items():
        count = len(features)
        module_total += count
        report.append(f"  {category}: {count} 个特征 (默认 d_model=64)")
        report.append(f"    可配置维度: 32, 64, 128, 256")
    report.append(f"  小计: {module_total} 个特征")
    report.append("")
    all_features["deep_learning"] = dl_features
    total_count += module_total

    # Summary
    report.append("=" * 80)
    report.append("总结 (Summary)")
    report.append("=" * 80)
    report.append(
        f"1. 基础特征工程: {sum(len(f) for f in count_basic_features().values())} 个"
    )
    report.append(
        f"2. 改进特征工程: {sum(len(f) for f in count_improved_features().values())} 个"
    )
    report.append(
        f"3. 增强特征工程: {sum(len(f) for f in count_enhanced_features().values())} 个"
    )
    report.append(
        f"4. 小波特征工程: {sum(len(f) for f in count_wavelet_features().values())} 个"
    )
    report.append(
        f"5. 深度学习特征: {sum(len(f) for f in count_dl_features().values())} 个"
    )
    report.append("")
    report.append(f"**总计: {total_count} 个特征**")
    report.append("")

    # Recommendations
    report.append("=" * 80)
    report.append("建议 (Recommendations)")
    report.append("=" * 80)
    report.append("1. 特征选择:")
    report.append("   - 使用特征重要性分析选择 top 100-200 个特征")
    report.append("   - 避免特征过多导致过拟合")
    report.append("")
    report.append("2. 特征工程模块选择:")
    report.append("   - 快速原型: 使用 feature_engineering.py (基础版)")
    report.append("   - 标准训练: 使用 feature_engineering_improved.py (改进版)")
    report.append("   - 高级研究: 使用 feature_engineering_enhanced.py (增强版)")
    report.append("   - 深度学习: 添加 dl_sequence_features.py")
    report.append("")
    report.append("3. 计算成本:")
    report.append("   - 基础版: 最快")
    report.append("   - 改进版: 快")
    report.append("   - 增强版: 慢 (WPT + Hurst 计算密集)")
    report.append("   - 深度学习: 中等 (需要 GPU 加速)")
    report.append("")

    report.append("=" * 80)

    return "\n".join(report)


def save_report(report: str, output_dir: Path):
    """保存报告到文件."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save text report
    txt_path = output_dir / "feature_count_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✓ 报告已保存: {txt_path}")

    # Save JSON data
    json_path = output_dir / "feature_count_data.json"
    data = {
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "basic": {
                "file": "feature_engineering.py",
                "features": count_basic_features(),
                "total": sum(len(f) for f in count_basic_features().values()),
            },
            "improved": {
                "file": "feature_engineering_improved.py",
                "features": count_improved_features(),
                "total": sum(len(f) for f in count_improved_features().values()),
            },
            "enhanced": {
                "file": "feature_engineering_enhanced.py",
                "features": count_enhanced_features(),
                "total": sum(len(f) for f in count_enhanced_features().values()),
            },
            "wavelet": {
                "file": "feature_engineering_wavelet.py",
                "features": count_wavelet_features(),
                "total": sum(len(f) for f in count_wavelet_features().values()),
            },
            "deep_learning": {
                "file": "dl_sequence_features.py",
                "features": count_dl_features(),
                "total": sum(len(f) for f in count_dl_features().values()),
            },
        },
        "grand_total": (
            sum(len(f) for f in count_basic_features().values())
            + sum(len(f) for f in count_improved_features().values())
            + sum(len(f) for f in count_enhanced_features().values())
            + sum(len(f) for f in count_wavelet_features().values())
            + sum(len(f) for f in count_dl_features().values())
        ),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON 数据已保存: {json_path}")


def main():
    """主函数."""
    print("\n🔍 统计项目特征数量...\n")

    # Generate report
    report = generate_report()

    # Print to console
    print(report)

    # Save to files
    output_dir = project_root / "reports"
    save_report(report, output_dir)

    print("\n✅ 特征统计完成!")


if __name__ == "__main__":
    main()
