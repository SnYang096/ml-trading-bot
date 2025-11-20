#!/usr/bin/env python3
"""
分析可疑特征：打乱后相关性仍 > 50% 的特征

这些特征可能是：
1. 虚假相关（spurious correlation）
2. 数据泄漏（data leakage）
3. 时间模式（temporal patterns）但不够稳定
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict


def analyze_suspicious_features(results_file: str = None):
    """
    分析可疑特征的模式和可能原因
    """
    # 从终端输出中提取的可疑特征（衰减比 > 50%）
    suspicious_features = {
        # 时间特征
        "hour_sin": 438.07,
        "hour_cos": 875.00,
        "Hour_of_Day": 2247.90,
        "minutes_since_reset": 2247.90,
        "Is_Weekend": 96.64,
        # 成交量相关
        "taker_buy_ratio": 59.99,
        "cvd_long": 328.25,
        "cvd_normalized": 59.99,
        "trade_count": 205.19,
        "volume_ratio": 272.41,
        "order_flow_imbalance": 59.99,
        "ofi_long": 438.80,
        # WPT mean 特征（未归一化）
        "close_wpt_aad_mean": 16035.89,
        "close_wpt_ada_mean": 148.08,
        "close_wpt_add_mean": 142.84,
        "close_wpt_daa_mean": 70.18,
        "close_wpt_dad_mean": 451.42,
        "close_wpt_dda_mean": 81.21,
        "close_wpt_ddd_mean": 143.63,
        "open_wpt_aad_mean": 94.74,
        "open_wpt_ada_mean": 104.93,
        "open_wpt_add_mean": 95.55,
        "open_wpt_dad_mean": 84.26,
        "open_wpt_dda_mean": 4290.05,
        "open_wpt_ddd_mean": 57.39,
        "volume_wpt_aaa_mean": 84.43,
        "volume_wpt_aad_mean": 60.91,
        "volume_wpt_ada_mean": 174.68,
        "volume_wpt_daa_mean": 102.99,
        "volume_wpt_dad_mean": 200.82,
        "volume_wpt_dda_mean": 255.91,
        "volume_wpt_ddd_mean": 181.48,
        "cvd_wpt_add_mean": 100.96,
        "cvd_wpt_daa_mean": 90.38,
        "cvd_wpt_dad_mean": 183.07,
        "cvd_wpt_dda_mean": 1363.50,
        "cvd_wpt_ddd_mean": 181.48,
        "taker_buy_ratio_wpt_aad_mean": 102.50,
        "taker_buy_ratio_wpt_ada_mean": 94.86,
        "taker_buy_ratio_wpt_add_mean": 1554.32,
        "taker_buy_ratio_wpt_daa_mean": 56.81,
        "taker_buy_ratio_wpt_dad_mean": 110.01,
        "taker_buy_ratio_wpt_ddd_mean": 618.69,
        # Hurst change/acceleration 特征
        "open_hurst_change": 140.85,
        "open_hurst_acceleration": 757.80,
        "volume_hurst_acceleration": 239.74,
        "cvd_hurst_acceleration": 152.38,
        "taker_buy_ratio_hurst_change": 181.64,
        "taker_buy_ratio_hurst_acceleration": 85.77,
        # 其他技术指标
        "price_to_vwap_pct": 55.39,
        "price_to_vwap_atr": 54.46,
        "volume_percentile": 55.86,
        "trange": 79.24,
        "stddev": 366.67,
        "var": 366.67,
        "cdl_hammer": 669.93,
        "cdl_hanging_man": 83.67,
        "cdl_3outside": 51.12,
        "cdl_closingmarubozu": 125.76,
        "cdl_eveningdojistar": 66.99,
        "cdl_eveningstar": 2799.56,
        "cdl_gapsidesidewhite": 68.53,
        "cdl_gravestonedoji": 223.22,
        "cdl_identical3crows": 857.61,
        "cdl_marubozu": 64.97,
        "cdl_matchinglow": 60.20,
        "cdl_takuri": 66.03,
        "tbr_extreme_buy": 355.65,
        "tbr_extreme_sell": 4165.71,
        "cvd_short_momentum": 64.01,
        "cvd_medium_momentum": 109.37,
        "cvd_medium_long_ratio": 116.78,
        "cvd_trend_alignment": 85.81,
        "tbr_momentum_5": 78.27,
        "cvd_acceleration": 37.24,
        "volume_wpt_aaa_energy": 161.96,
        "volume_wpt_aad_energy": 110.94,
        "volume_wpt_ada_energy": 68.17,
        "volume_wpt_add_energy": 28.87,
        "cvd_wpt_aaa_std": 327.24,
        "cvd_wpt_add_energy": 53.73,
        "cvd_wpt_dda_energy": 74.60,
        "cvd_wpt_ddd_energy": 53.89,
        "taker_buy_ratio_wpt_aaa_energy": 29.25,
        "taker_buy_ratio_spectral_bandwidth": 87.20,
        "open_hilbert_phase_acceleration": 89.16,
        "volume_hilbert_phase_acceleration": 81.42,
        "taker_buy_ratio_hilbert_phase_acceleration": 387.71,
    }

    # 按类别分组
    categories = defaultdict(list)

    for feat, ratio in suspicious_features.items():
        if (
            "hour" in feat.lower()
            or "Hour" in feat
            or "minutes" in feat
            or "Weekend" in feat
        ):
            categories["时间特征"].append((feat, ratio))
        elif "wpt" in feat and "_mean" in feat:
            categories["WPT_mean特征（未归一化）"].append((feat, ratio))
        elif "hurst" in feat and ("change" in feat or "acceleration" in feat):
            categories["Hurst_change/acceleration特征"].append((feat, ratio))
        elif "wpt" in feat and ("_energy" in feat or "_std" in feat):
            categories["WPT_energy/std特征（未归一化）"].append((feat, ratio))
        elif "cdl_" in feat:
            categories["K线形态特征"].append((feat, ratio))
        elif "momentum" in feat or "acceleration" in feat or "change" in feat:
            categories["动量/变化特征"].append((feat, ratio))
        elif "extreme" in feat or "ratio" in feat:
            categories["极值/比率特征"].append((feat, ratio))
        elif "percentile" in feat or "pct" in feat:
            categories["百分位特征"].append((feat, ratio))
        elif "spectral" in feat or "hilbert" in feat:
            categories["频域特征"].append((feat, ratio))
        else:
            categories["其他"].append((feat, ratio))

    # 打印分析结果
    print("=" * 80)
    print("🔍 可疑特征分析报告")
    print("=" * 80)
    print(f"\n总计发现 {len(suspicious_features)} 个可疑特征（衰减比 > 50%）\n")

    for category, features in sorted(
        categories.items(), key=lambda x: len(x[1]), reverse=True
    ):
        print(f"\n{'='*80}")
        print(f"📊 {category} ({len(features)} 个)")
        print(f"{'='*80}")

        # 按衰减比排序
        features_sorted = sorted(features, key=lambda x: x[1], reverse=True)

        print(f"\n{'特征名':<50} {'衰减比':>15}")
        print("-" * 80)
        for feat, ratio in features_sorted:
            print(f"{feat:<50} {ratio:>15.2f}%")

    # 分析可能原因
    print("\n" + "=" * 80)
    print("🔬 可能原因分析")
    print("=" * 80)

    print(
        "\n1. 时间特征（hour_sin, hour_cos, Hour_of_Day, minutes_since_reset, Is_Weekend）"
    )
    print("   问题：这些特征与未来收益的虚假相关")
    print("   原因：")
    print("   - 时间特征本身不包含价格信息，不应该有预测能力")
    print("   - 如果存在相关性，可能是数据中的时间模式（如某些时段更容易上涨）")
    print("   - 但这种模式可能不稳定，导致打乱后相关性仍然存在（虚假相关）")
    print("   建议：")
    print("   - 这些特征可能不适合用于预测，建议移除或谨慎使用")
    print("   - 如果确实存在时间模式，应该通过更稳健的方法验证")

    print("\n2. WPT mean 特征（未归一化）")
    print("   问题：小波包变换的均值特征未归一化，可能包含量纲信息")
    print("   原因：")
    print("   - wpt_*_mean 特征直接使用原始小波系数的均值")
    print("   - 这些值可能包含价格/成交量的绝对量纲信息")
    print("   - 如果价格趋势向上，mean 值会持续增大，形成虚假相关")
    print("   建议：")
    print("   - 代码中已经尝试排除这些特征，但可能还有遗漏")
    print("   - 应该只使用归一化的 WPT 特征（如 wpt_*_energy_ratio）")
    print("   - 检查 comprehensive_feature_engineering.py 中的过滤逻辑")

    print("\n3. Hurst change/acceleration 特征")
    print("   问题：使用 .diff() 计算变化率，可能引入未来信息")
    print("   原因：")
    print("   - hurst_change = hurst.diff() 计算相邻时间点的差值")
    print("   - 如果 Hurst 指数计算使用了未来窗口，diff() 会放大泄漏")
    print("   - acceleration = change.diff() 是二阶差分，更容易累积误差")
    print("   建议：")
    print("   - 检查 Hurst 指数的计算是否使用了未来数据")
    print("   - 如果 Hurst 计算是安全的，diff() 应该是安全的（但需要 shift(1)）")
    print("   - 建议使用 shift(1) 确保时间对齐")

    print("\n4. K线形态特征（cdl_*）")
    print("   问题：某些 K 线形态特征存在虚假相关")
    print("   原因：")
    print("   - K 线形态特征通常是二值特征（0/1）")
    print("   - 如果某些形态出现频率低，可能产生虚假相关")
    print("   - 打乱后相关性仍然存在，说明不是真实的预测信号")
    print("   建议：")
    print("   - 检查这些形态特征的计算逻辑")
    print("   - 如果形态出现频率过低，考虑移除或合并")

    print("\n5. 极值/比率特征")
    print("   问题：某些极值和比率特征存在虚假相关")
    print("   原因：")
    print("   - 极值特征可能对异常值敏感")
    print("   - 比率特征如果分母接近0，可能产生异常值")
    print("   - 这些异常值可能导致虚假相关")
    print("   建议：")
    print("   - 检查极值特征的计算逻辑，确保处理了异常值")
    print("   - 检查比率特征的分母是否为0的情况")

    print("\n" + "=" * 80)
    print("🛠️  修复建议")
    print("=" * 80)

    print("\n1. 立即移除的特征：")
    print(
        "   - 所有时间特征（hour_sin, hour_cos, Hour_of_Day, minutes_since_reset, Is_Weekend）"
    )
    print("   - 所有 WPT mean 特征（wpt_*_mean）")
    print("   - 所有 WPT energy/std 特征（如果未归一化）")

    print("\n2. 需要检查并修复的特征：")
    print("   - Hurst change/acceleration 特征：确保 Hurst 计算安全，并添加 shift(1)")
    print("   - K线形态特征：检查计算逻辑，移除出现频率过低的形态")
    print("   - 极值/比率特征：添加异常值处理")

    print("\n3. 代码修改建议：")
    print("   - 在 comprehensive_feature_engineering.py 中加强 WPT 特征过滤")
    print("   - 在 baseline_features.py 中移除或标记时间特征")
    print("   - 在 feature_engineering_enhanced.py 中修复 Hurst 衍生特征")

    # 生成特征列表文件
    features_to_remove = {
        "high_priority": [
            # 时间特征
            "hour_sin",
            "hour_cos",
            "Hour_of_Day",
            "minutes_since_reset",
            "Is_Weekend",
            # WPT mean 特征
            *[f for f in suspicious_features.keys() if "wpt" in f and "_mean" in f],
        ],
        "medium_priority": [
            # Hurst change/acceleration
            *[
                f
                for f in suspicious_features.keys()
                if "hurst" in f and ("change" in f or "acceleration" in f)
            ],
            # 极值特征
            "tbr_extreme_buy",
            "tbr_extreme_sell",
        ],
        "low_priority": [
            # K线形态（出现频率低的）
            "cdl_eveningstar",
            "cdl_identical3crows",
            "cdl_gravestonedoji",
            "cdl_hammer",
            "cdl_closingmarubozu",
        ],
    }

    # 尝试写入文件
    try:
        output_file = (
            Path(__file__).parent.parent
            / "results"
            / "suspicious_features_to_remove.json"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(features_to_remove, f, indent=2)
        print(f"\n✅ 已生成特征移除列表: {output_file}")
    except PermissionError:
        # 如果无法写入 results 目录，输出到脚本目录
        output_file = Path(__file__).parent / "suspicious_features_to_remove.json"
        with open(output_file, "w") as f:
            json.dump(features_to_remove, f, indent=2)
        print(f"\n✅ 已生成特征移除列表: {output_file}")

    print(f"   - 高优先级: {len(features_to_remove['high_priority'])} 个特征")
    print(f"   - 中优先级: {len(features_to_remove['medium_priority'])} 个特征")
    print(f"   - 低优先级: {len(features_to_remove['low_priority'])} 个特征")


if __name__ == "__main__":
    analyze_suspicious_features()
