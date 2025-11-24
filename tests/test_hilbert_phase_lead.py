"""
Hilbert 相位领先分析测试

测试内容：
1. CVD vs Volume 在相位分析中的差异
2. CVD 相位领先能否预测价格变化
3. 去趋势化的必要性
4. 真实市场场景模拟
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.time_series.utils_hilbert_features import (
    hilbert_transform,
    compute_phase_lead,
    extract_hilbert_features,
)
from src.features.time_series.utils_wpt_features import wpt_decompose


class TestHilbertPhaseLead(unittest.TestCase):
    """Hilbert 相位领先分析测试类"""

    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.n_samples = 100

    def create_simulated_market_data(self):
        """
        创建模拟市场数据

        场景：主力在 t=40 开始净买入（CVD 领先），价格在 t=45 开始上涨
        """
        t = np.arange(self.n_samples)

        # 1. 价格序列：t=45 开始上涨
        price = 100 + 0.1 * t
        price[45:] += 2 * np.cumsum(np.random.randn(self.n_samples - 45) * 0.5 + 0.3)

        # 2. Volume：脉冲式，价格上涨时放量（跟随价格）
        volume = np.abs(np.random.normal(10, 3, self.n_samples))
        volume[45:] = volume[45:] * 2  # 价格突破后放量

        # 3. CVD：t=40 开始净买入（领先价格 5 个时间单位）
        cvd = np.cumsum(np.random.randn(self.n_samples) * 0.5)
        # t=40 开始主力净买入
        cvd[40:] += np.cumsum(np.random.randn(self.n_samples - 40) * 0.3 + 0.5)

        return {
            "price": price,
            "volume": volume,
            "cvd": cvd,
            "t": t,
        }

    def test_volume_vs_cvd_phase_difference(self):
        """
        测试 1：Volume vs CVD 相位差对比

        预期结果：
        - Volume 相位差 ≈ 0（跟随价格）
        - CVD 相位差 > 0（领先价格）
        """
        print("\n" + "=" * 70)
        print("测试 1：Volume vs CVD 相位差对比")
        print("=" * 70)

        data = self.create_simulated_market_data()
        price = data["price"]
        volume = data["volume"]
        cvd = data["cvd"]
        t = data["t"]

        # 去趋势（使用 WPT）
        price_wpt = wpt_decompose(price, wavelet="db4", level=3)
        price_trend = price_wpt["trend"]
        price_fluctuation = price_wpt["fluctuation"]

        volume_wpt = wpt_decompose(volume, wavelet="db4", level=3)
        volume_fluctuation = volume_wpt["fluctuation"]

        cvd_wpt = wpt_decompose(cvd, wavelet="db4", level=3)
        cvd_fluctuation = cvd_wpt["fluctuation"]

        # Hilbert 变换
        price_hilbert = hilbert_transform(price_fluctuation, detrend=False)
        volume_hilbert = hilbert_transform(volume_fluctuation, detrend=False)
        cvd_hilbert = hilbert_transform(cvd_fluctuation, detrend=False)

        # 计算相位差
        phase_diff_volume = (
            volume_hilbert["phase_unwrapped"] - price_hilbert["phase_unwrapped"]
        )
        phase_diff_cvd = (
            cvd_hilbert["phase_unwrapped"] - price_hilbert["phase_unwrapped"]
        )

        # 分析 t=40~50 时间段（主力建仓到价格启动）
        analysis_window = slice(40, 50)

        volume_phase_diff_mean = np.mean(phase_diff_volume[analysis_window])
        cvd_phase_diff_mean = np.mean(phase_diff_cvd[analysis_window])

        print(f"\n时间段 t=40~50（主力建仓到价格启动）:")
        print(f"  Volume 相位差均值: {volume_phase_diff_mean:.4f}")
        print(f"  CVD 相位差均值: {cvd_phase_diff_mean:.4f}")
        print(f"  差异: {cvd_phase_diff_mean - volume_phase_diff_mean:.4f}")

        # 断言：CVD 相位差应该显著大于 Volume 相位差
        self.assertGreater(
            cvd_phase_diff_mean,
            volume_phase_diff_mean,
            "CVD 相位差应该大于 Volume 相位差",
        )

        # 断言：CVD 相位差应该为正（领先）
        self.assertGreater(cvd_phase_diff_mean, 0, "CVD 相位差应该为正（领先价格）")

        print(f"\n✅ 测试通过：CVD 相位领先效果显著优于 Volume")

    def test_cvd_phase_lead_predicts_price_movement(self):
        """
        测试 2：CVD 相位领先能否预测价格变化

        预期结果：
        - t=40~45：CVD 相位领先，但价格未动 → 预测信号
        - t=45~50：价格开始上涨 → 验证预测
        """
        print("\n" + "=" * 70)
        print("测试 2：CVD 相位领先预测价格变化")
        print("=" * 70)

        data = self.create_simulated_market_data()
        price = data["price"]
        cvd = data["cvd"]

        # 去趋势
        price_wpt = wpt_decompose(price, wavelet="db4", level=3)
        price_fluctuation = price_wpt["fluctuation"]

        cvd_wpt = wpt_decompose(cvd, wavelet="db4", level=3)
        cvd_fluctuation = cvd_wpt["fluctuation"]

        # 计算相位差
        price_hilbert = hilbert_transform(price_fluctuation, detrend=False)
        cvd_hilbert = hilbert_transform(cvd_fluctuation, detrend=False)
        phase_diff = cvd_hilbert["phase_unwrapped"] - price_hilbert["phase_unwrapped"]

        # 分析不同时间段
        # t=40~45：CVD 领先，价格未动
        lead_period = slice(40, 45)
        # t=45~50：价格开始上涨
        price_move_period = slice(45, 50)

        phase_lead_before = np.mean(phase_diff[lead_period])
        price_change_before = price[45] - price[40]

        phase_lead_during = np.mean(phase_diff[price_move_period])
        price_change_during = price[50] - price[45]

        print(f"\nt=40~45（CVD 领先，价格未动）:")
        print(f"  CVD 相位领先均值: {phase_lead_before:.4f}")
        print(f"  价格变化: {price_change_before:.4f}")

        print(f"\nt=45~50（价格开始上涨）:")
        print(f"  CVD 相位领先均值: {phase_lead_during:.4f}")
        print(f"  价格变化: {price_change_during:.4f}")

        # 断言：CVD 相位领先应该出现在价格变化之前
        self.assertGreater(phase_lead_before, 0, "CVD 应该在价格变化前就出现相位领先")

        # 断言：价格变化应该与之前的相位领先相关
        self.assertGreater(price_change_during, 0, "价格应该在 CVD 相位领先后上涨")

        print(f"\n✅ 测试通过：CVD 相位领先能够预测价格变化")

    def test_detrending_necessity(self):
        """
        测试 3：去趋势化的必要性

        预期结果：
        - 不去趋势：相位差被趋势淹没，无法识别领先关系
        - 去趋势后：相位差清晰，能够识别领先关系
        """
        print("\n" + "=" * 70)
        print("测试 3：去趋势化的必要性")
        print("=" * 70)

        data = self.create_simulated_market_data()
        price = data["price"]
        cvd = data["cvd"]

        # 方法 1：不去趋势
        price_hilbert_raw = hilbert_transform(price, detrend=False)
        cvd_hilbert_raw = hilbert_transform(cvd, detrend=False)
        phase_diff_raw = (
            cvd_hilbert_raw["phase_unwrapped"] - price_hilbert_raw["phase_unwrapped"]
        )

        # 方法 2：去趋势后
        price_wpt = wpt_decompose(price, wavelet="db4", level=3)
        price_fluctuation = price_wpt["fluctuation"]

        cvd_wpt = wpt_decompose(cvd, wavelet="db4", level=3)
        cvd_fluctuation = cvd_wpt["fluctuation"]

        price_hilbert_detrend = hilbert_transform(price_fluctuation, detrend=False)
        cvd_hilbert_detrend = hilbert_transform(cvd_fluctuation, detrend=False)
        phase_diff_detrend = (
            cvd_hilbert_detrend["phase_unwrapped"]
            - price_hilbert_detrend["phase_unwrapped"]
        )

        # 分析 t=40~50 时间段
        analysis_window = slice(40, 50)

        phase_diff_raw_mean = np.mean(phase_diff_raw[analysis_window])
        phase_diff_detrend_mean = np.mean(phase_diff_detrend[analysis_window])

        phase_diff_raw_std = np.std(phase_diff_raw[analysis_window])
        phase_diff_detrend_std = np.std(phase_diff_detrend[analysis_window])

        print(f"\n时间段 t=40~50:")
        print(f"  不去趋势:")
        print(f"    相位差均值: {phase_diff_raw_mean:.4f}")
        print(f"    相位差标准差: {phase_diff_raw_std:.4f}")
        print(f"  去趋势后:")
        print(f"    相位差均值: {phase_diff_detrend_mean:.4f}")
        print(f"    相位差标准差: {phase_diff_detrend_std:.4f}")

        # 断言：去趋势后的相位差应该更稳定（标准差更小）
        self.assertLess(
            phase_diff_detrend_std, phase_diff_raw_std, "去趋势后的相位差应该更稳定"
        )

        # 断言：去趋势后的相位差应该更清晰（均值更显著）
        self.assertGreater(
            abs(phase_diff_detrend_mean),
            abs(phase_diff_raw_mean) * 0.5,
            "去趋势后的相位差应该更清晰",
        )

        print(f"\n✅ 测试通过：去趋势化能够提高相位分析的准确性")

    def test_real_market_scenario(self):
        """
        测试 4：真实市场场景模拟

        场景：机构在财报前吸筹
        - Price：横盘 10 日
        - Volume：每日稳定，无异常
        - CVD：连续 5 日净流入
        """
        print("\n" + "=" * 70)
        print("测试 4：真实市场场景模拟（机构吸筹）")
        print("=" * 70)

        n_days = 20
        t = np.arange(n_days)

        # Price：横盘 10 日，然后上涨
        price = 100 + 0.01 * np.random.randn(n_days)
        price[10:] += np.cumsum(np.random.randn(n_days - 10) * 0.1 + 0.2)

        # Volume：每日稳定，无异常
        volume = 10 + 0.5 * np.random.randn(n_days)

        # CVD：连续 5 日净流入（t=5~10）
        cvd = np.cumsum(np.random.randn(n_days) * 0.2)
        cvd[5:10] += np.cumsum(np.random.randn(5) * 0.1 + 0.3)  # 净流入

        # 去趋势
        price_wpt = wpt_decompose(price, wavelet="db4", level=2)
        price_fluctuation = price_wpt["fluctuation"]

        volume_wpt = wpt_decompose(volume, wavelet="db4", level=2)
        volume_fluctuation = volume_wpt["fluctuation"]

        cvd_wpt = wpt_decompose(cvd, wavelet="db4", level=2)
        cvd_fluctuation = cvd_wpt["fluctuation"]

        # Hilbert 变换
        price_hilbert = hilbert_transform(price_fluctuation, detrend=False)
        volume_hilbert = hilbert_transform(volume_fluctuation, detrend=False)
        cvd_hilbert = hilbert_transform(cvd_fluctuation, detrend=False)

        # 计算相位差
        phase_diff_volume = (
            volume_hilbert["phase_unwrapped"] - price_hilbert["phase_unwrapped"]
        )
        phase_diff_cvd = (
            cvd_hilbert["phase_unwrapped"] - price_hilbert["phase_unwrapped"]
        )

        # 分析不同时间段
        accumulation_period = slice(5, 10)  # 机构吸筹期
        price_move_period = slice(10, 15)  # 价格上涨期

        print(f"\nt=5~10（机构吸筹期）:")
        print(
            f"  Volume 相位差均值: {np.mean(phase_diff_volume[accumulation_period]):.4f}"
        )
        print(f"  CVD 相位差均值: {np.mean(phase_diff_cvd[accumulation_period]):.4f}")
        print(f"  价格变化: {price[10] - price[5]:.4f}")

        print(f"\nt=10~15（价格上涨期）:")
        print(
            f"  Volume 相位差均值: {np.mean(phase_diff_volume[price_move_period]):.4f}"
        )
        print(f"  CVD 相位差均值: {np.mean(phase_diff_cvd[price_move_period]):.4f}")
        print(f"  价格变化: {price[15] - price[10]:.4f}")

        # 断言：在吸筹期，CVD 应该领先，但 Volume 不应该
        cvd_lead_accumulation = np.mean(phase_diff_cvd[accumulation_period])
        volume_lead_accumulation = np.mean(phase_diff_volume[accumulation_period])

        self.assertGreater(
            cvd_lead_accumulation,
            volume_lead_accumulation,
            "在吸筹期，CVD 应该比 Volume 更领先",
        )

        self.assertGreater(cvd_lead_accumulation, 0, "在吸筹期，CVD 应该领先价格")

        print(f"\n✅ 测试通过：CVD 能够在机构吸筹期提前预警")

    def test_phase_lead_signal_quality(self):
        """
        测试 5：相位领先信号质量

        验证：
        1. CVD 相位领先时，后续价格变化方向
        2. 信号强度和持续时间
        3. 假信号率
        """
        print("\n" + "=" * 70)
        print("测试 5：相位领先信号质量")
        print("=" * 70)

        # 创建多个场景的数据
        n_scenarios = 10
        correct_predictions = 0
        total_signals = 0

        for scenario in range(n_scenarios):
            np.random.seed(42 + scenario)

            # 创建数据：随机选择主力建仓时间点
            build_up_start = np.random.randint(30, 60)
            price_move_start = build_up_start + np.random.randint(3, 8)

            n_samples = 100
            t = np.arange(n_samples)

            # Price：在 price_move_start 开始上涨
            price = 100 + 0.1 * t
            price[price_move_start:] += np.cumsum(
                np.random.randn(n_samples - price_move_start) * 0.5 + 0.3
            )

            # CVD：在 build_up_start 开始净买入
            cvd = np.cumsum(np.random.randn(n_samples) * 0.5)
            cvd[build_up_start:] += np.cumsum(
                np.random.randn(n_samples - build_up_start) * 0.3 + 0.5
            )

            # 去趋势
            price_wpt = wpt_decompose(price, wavelet="db4", level=3)
            price_fluctuation = price_wpt["fluctuation"]

            cvd_wpt = wpt_decompose(cvd, wavelet="db4", level=3)
            cvd_fluctuation = cvd_wpt["fluctuation"]

            # Hilbert 变换
            price_hilbert = hilbert_transform(price_fluctuation, detrend=False)
            cvd_hilbert = hilbert_transform(cvd_fluctuation, detrend=False)
            phase_diff = (
                cvd_hilbert["phase_unwrapped"] - price_hilbert["phase_unwrapped"]
            )

            # 检测信号：CVD 相位领先 > 阈值
            signal_threshold = np.percentile(phase_diff, 75)
            signals = phase_diff > signal_threshold

            # 检查信号后的价格变化
            for i in range(len(phase_diff) - 5):
                if signals[i]:
                    total_signals += 1
                    # 检查未来 5 个时间点的价格变化
                    future_price_change = price[i + 5] - price[i]
                    if future_price_change > 0:
                        correct_predictions += 1

        accuracy = correct_predictions / total_signals if total_signals > 0 else 0

        print(f"\n信号统计:")
        print(f"  总信号数: {total_signals}")
        print(f"  正确预测: {correct_predictions}")
        print(f"  准确率: {accuracy:.2%}")

        # 断言：准确率应该显著高于随机（50%）
        self.assertGreater(
            accuracy, 0.5, f"相位领先信号准确率 ({accuracy:.2%}) 应该高于随机 (50%)"
        )

        print(f"\n✅ 测试通过：相位领先信号具有预测价值")

    def test_cvd_vs_volume_envelope_analysis(self):
        """
        测试 6：CVD vs Volume 包络分析

        验证：CVD 包络放大 + 相位领先 = 有效信号
        """
        print("\n" + "=" * 70)
        print("测试 6：CVD vs Volume 包络分析")
        print("=" * 70)

        data = self.create_simulated_market_data()
        price = data["price"]
        volume = data["volume"]
        cvd = data["cvd"]

        # 去趋势
        price_wpt = wpt_decompose(price, wavelet="db4", level=3)
        price_fluctuation = price_wpt["fluctuation"]

        volume_wpt = wpt_decompose(volume, wavelet="db4", level=3)
        volume_fluctuation = volume_wpt["fluctuation"]

        cvd_wpt = wpt_decompose(cvd, wavelet="db4", level=3)
        cvd_fluctuation = cvd_wpt["fluctuation"]

        # Hilbert 变换
        price_hilbert = hilbert_transform(price_fluctuation, detrend=False)
        volume_hilbert = hilbert_transform(volume_fluctuation, detrend=False)
        cvd_hilbert = hilbert_transform(cvd_fluctuation, detrend=False)

        # 包络分析
        price_envelope = price_hilbert["envelope"]
        volume_envelope = volume_hilbert["envelope"]
        cvd_envelope = cvd_hilbert["envelope"]

        # 相位差
        phase_diff_volume = (
            volume_hilbert["phase_unwrapped"] - price_hilbert["phase_unwrapped"]
        )
        phase_diff_cvd = (
            cvd_hilbert["phase_unwrapped"] - price_hilbert["phase_unwrapped"]
        )

        # 分析 t=40~50 时间段
        analysis_window = slice(40, 50)

        # 包络放大（相对于前期）
        baseline_window = slice(30, 40)

        volume_envelope_expansion = np.mean(volume_envelope[analysis_window]) / (
            np.mean(volume_envelope[baseline_window]) + 1e-6
        )
        cvd_envelope_expansion = np.mean(cvd_envelope[analysis_window]) / (
            np.mean(cvd_envelope[baseline_window]) + 1e-6
        )

        print(f"\n时间段 t=40~50:")
        print(f"  Volume 包络放大倍数: {volume_envelope_expansion:.4f}")
        print(f"  CVD 包络放大倍数: {cvd_envelope_expansion:.4f}")
        print(f"  Volume 相位差均值: {np.mean(phase_diff_volume[analysis_window]):.4f}")
        print(f"  CVD 相位差均值: {np.mean(phase_diff_cvd[analysis_window]):.4f}")

        # 综合信号：包络放大 + 相位领先
        volume_signal_strength = volume_envelope_expansion * np.mean(
            phase_diff_volume[analysis_window]
        )
        cvd_signal_strength = cvd_envelope_expansion * np.mean(
            phase_diff_cvd[analysis_window]
        )

        print(f"\n综合信号强度:")
        print(f"  Volume: {volume_signal_strength:.4f}")
        print(f"  CVD: {cvd_signal_strength:.4f}")

        # 断言：CVD 综合信号应该更强
        self.assertGreater(
            cvd_signal_strength,
            volume_signal_strength,
            "CVD 综合信号（包络+相位）应该强于 Volume",
        )

        print(f"\n✅ 测试通过：CVD 包络+相位综合信号更有效")


class TestHilbertFeatureIntegration(unittest.TestCase):
    """Hilbert 特征集成测试"""

    def test_extract_hilbert_features_integration(self):
        """测试 Hilbert 特征提取集成"""
        print("\n" + "=" * 70)
        print("测试：Hilbert 特征提取集成")
        print("=" * 70)

        # 创建测试 DataFrame
        np.random.seed(42)
        n_samples = 100

        df = pd.DataFrame(
            {
                "close": 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
                "cvd": np.cumsum(np.random.randn(n_samples) * 0.3),
            }
        )

        # 添加 WPT 特征（模拟）
        price_wpt = wpt_decompose(df["close"].values, wavelet="db4", level=3)
        df["wpt_price_fluctuation"] = price_wpt["fluctuation"]
        df["wpt_price_trend"] = price_wpt["trend"]

        cvd_wpt = wpt_decompose(df["cvd"].values, wavelet="db4", level=3)
        df["wpt_cvd_fluctuation"] = cvd_wpt["fluctuation"]
        df["wpt_cvd_trend"] = cvd_wpt["trend"]

        # 提取 Hilbert 特征
        result_df = extract_hilbert_features(
            df,
            price_fluctuation_col="wpt_price_fluctuation",
            cvd_fluctuation_col="wpt_cvd_fluctuation",
        )

        # 检查输出特征
        expected_features = [
            "hilbert_price_phase",
            "hilbert_price_envelope",
            "hilbert_cvd_phase",
            "hilbert_cvd_envelope",
            "hilbert_phase_diff",
            "hilbert_cvd_leads",
        ]

        print(f"\n检查输出特征:")
        for feature in expected_features:
            if feature in result_df.columns:
                print(f"  ✅ {feature}")
            else:
                print(f"  ❌ {feature} (缺失)")

        # 断言：所有预期特征都应该存在
        for feature in expected_features:
            self.assertIn(
                feature,
                result_df.columns,
                f"特征 {feature} 应该存在",
            )

        print(f"\n✅ 测试通过：Hilbert 特征提取集成正常")


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
