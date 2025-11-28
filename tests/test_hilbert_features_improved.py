"""
Hilbert 变换特征工程（改进版）测试

测试内容：
1. 基础包络特征有效性（波动强度检测）
2. 背离信号检测（价格新高但CVD未新高）
3. 成交量融合（识别假突破）
4. 分位数标准化（跨品种可比）
5. 自适应窗口（动态周期匹配）
6. 无未来信息验证（因果性测试）
7. 边界情况处理（NaN、异常值）
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.time_series.utils_hilbert_features import (
    extract_hilbert_features,
    compute_hilbert_envelope,
    estimate_local_period,
    rolling_quantile_normalize,
)


class TestHilbertFeaturesImproved(unittest.TestCase):
    """Hilbert 特征（改进版）测试类"""

    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.n_samples = 500

    def create_base_data(self, n_samples=None):
        """创建基础测试数据（包含WPT波动分量）"""
        if n_samples is None:
            n_samples = self.n_samples

        # 创建价格和CVD的波动分量（模拟WPT分解后的结果）
        t = np.arange(n_samples)

        # 价格波动：包含趋势和震荡
        price_fluc = np.sin(2 * np.pi * t / 50) + 0.5 * np.random.randn(n_samples)

        # CVD波动：与价格相关但可能有领先/滞后
        cvd_fluc = 0.8 * price_fluc + 0.2 * np.random.randn(n_samples)

        # 成交量（用于成交量融合测试）
        volume = 1000 + 200 * np.abs(np.random.randn(n_samples))

        df = pd.DataFrame(
            {
                "wpt_price_fluctuation": price_fluc,
                "wpt_cvd_fluctuation": cvd_fluc,
                "volume": volume,
                "close": 100 + np.cumsum(price_fluc * 0.1),  # 价格序列
            }
        )

        return df

    def test_basic_envelope_features(self):
        """
        测试 1：基础包络特征有效性

        验证：
        - hilbert_price_env 能捕捉波动强度
        - hilbert_cvd_env 能捕捉资金流波动
        - hilbert_cvd_price_env_ratio 能反映背离
        """
        print("\n" + "=" * 70)
        print("测试 1：基础包络特征有效性")
        print("=" * 70)

        df = self.create_base_data()

        # 在 t=200 处制造一个价格波动突增
        df.loc[200:250, "wpt_price_fluctuation"] *= 3

        result = extract_hilbert_features(
            df,
            price_fluctuation_col="wpt_price_fluctuation",
            cvd_fluctuation_col="wpt_cvd_fluctuation",
            window=64,
            ema_span=10,
        )

        # 检查特征是否存在
        self.assertIn("hilbert_price_env", result.columns)
        self.assertIn("hilbert_cvd_env", result.columns)
        self.assertIn("hilbert_cvd_price_env_ratio", result.columns)
        self.assertIn("hilbert_price_env_slope", result.columns)

        # 验证波动突增被捕捉（t=200附近包络应该增大）
        price_env = result["hilbert_price_env"].dropna()
        if len(price_env) > 200:
            # 检查突增后的包络是否大于突增前
            before_avg = price_env.iloc[150:200].mean()
            after_avg = price_env.iloc[200:250].mean()

            print(f"  突增前平均包络: {before_avg:.4f}")
            print(f"  突增后平均包络: {after_avg:.4f}")
            print(f"  包络增长: {(after_avg / before_avg - 1) * 100:.2f}%")

            # 突增后包络应该明显增大
            self.assertGreater(
                after_avg, before_avg * 1.2, "价格波动突增应该导致包络增大"
            )

        print("  ✅ 基础包络特征能有效捕捉波动强度变化")

    def test_divergence_detection(self):
        """
        测试 2：背离信号检测

        场景：价格包络创新高，但CVD包络未创新高 → 背离信号
        """
        print("\n" + "=" * 70)
        print("测试 2：背离信号检测")
        print("=" * 70)

        df = self.create_base_data()

        # 制造背离场景：t=200后价格波动增大，但CVD波动不变
        df.loc[200:300, "wpt_price_fluctuation"] *= 2.5
        # CVD波动保持原样（不增大）

        result = extract_hilbert_features(
            df,
            price_fluctuation_col="wpt_price_fluctuation",
            cvd_fluctuation_col="wpt_cvd_fluctuation",
            window=64,
            ema_span=10,
        )

        price_env = result["hilbert_price_env"].dropna()
        cvd_env = result["hilbert_cvd_env"].dropna()
        ratio = result["hilbert_cvd_price_env_ratio"].dropna()

        if len(price_env) > 250 and len(cvd_env) > 250:
            # 计算背离程度
            price_before = price_env.iloc[150:200].mean()
            price_after = price_env.iloc[250:300].mean()

            cvd_before = cvd_env.iloc[150:200].mean()
            cvd_after = cvd_env.iloc[250:300].mean()

            price_change = (price_after / price_before - 1) * 100
            cvd_change = (cvd_after / cvd_before - 1) * 100

            print(f"  价格包络变化: {price_change:.2f}%")
            print(f"  CVD包络变化: {cvd_change:.2f}%")
            print(f"  背离程度: {price_change - cvd_change:.2f}%")

            # 验证背离被捕捉（价格增长明显大于CVD）
            self.assertGreater(price_change, cvd_change + 50, "背离场景应该被正确识别")

            # 验证比值下降（CVD/Price 比值应该下降）
            ratio_before = ratio.iloc[150:200].mean()
            ratio_after = ratio.iloc[250:300].mean()
            self.assertLess(
                ratio_after, ratio_before * 0.8, "背离时CVD/Price比值应该下降"
            )

        print("  ✅ 背离信号能被有效检测")

    def test_volume_fusion_fake_breakout(self):
        """
        测试 3：成交量融合 - 识别假突破

        场景：价格波动增大，但成交量波动下降 → 假突破信号
        """
        print("\n" + "=" * 70)
        print("测试 3：成交量融合 - 识别假突破")
        print("=" * 70)

        df = self.create_base_data()

        # 制造假突破：t=200后价格波动增大，但成交量下降
        df.loc[200:300, "wpt_price_fluctuation"] *= 2.0
        df.loc[200:300, "volume"] *= 0.5  # 成交量下降

        result = extract_hilbert_features(
            df,
            price_fluctuation_col="wpt_price_fluctuation",
            cvd_fluctuation_col="wpt_cvd_fluctuation",
            volume_col="volume",
            window=64,
            ema_span=10,
            use_volume_fusion=True,
        )

        # 检查成交量融合特征
        self.assertIn("hilbert_volume_env", result.columns)
        self.assertIn("hilbert_env_price_vol_ratio", result.columns)

        price_env = result["hilbert_price_env"].dropna()
        vol_env = result["hilbert_volume_env"].dropna()
        ratio = result["hilbert_env_price_vol_ratio"].dropna()

        if len(price_env) > 250 and len(vol_env) > 250:
            # 验证假突破被识别
            price_before = price_env.iloc[150:200].mean()
            price_after = price_env.iloc[250:300].mean()

            vol_before = vol_env.iloc[150:200].mean()
            vol_after = vol_env.iloc[250:300].mean()

            # 价格/成交量比值应该大幅上升（价格涨但成交量跌）
            ratio_before = ratio.iloc[150:200].mean()
            ratio_after = ratio.iloc[250:300].mean()

            print(f"  价格包络变化: {(price_after/price_before - 1)*100:.2f}%")
            print(f"  成交量包络变化: {(vol_after/vol_before - 1)*100:.2f}%")
            print(f"  价格/成交量比值变化: {(ratio_after/ratio_before - 1)*100:.2f}%")

            # 假突破时，价格/成交量比值应该大幅上升
            self.assertGreater(
                ratio_after, ratio_before * 1.5, "假突破时价格/成交量比值应该大幅上升"
            )

        print("  ✅ 成交量融合能有效识别假突破")

    def test_triple_divergence_signal(self):
        """
        测试 4：三元背离信号

        场景：价格包络新高 + CVD包络未新高 + 成交量包络下降 → 强烈背离
        """
        print("\n" + "=" * 70)
        print("测试 4：三元背离信号")
        print("=" * 70)

        df = self.create_base_data()

        # 制造三元背离：t=200后
        # 1. 价格波动增大（新高）
        df.loc[200:300, "wpt_price_fluctuation"] *= 2.5
        # 2. CVD波动不变（未新高）
        # 3. 成交量下降
        df.loc[200:300, "volume"] *= 0.6

        result = extract_hilbert_features(
            df,
            price_fluctuation_col="wpt_price_fluctuation",
            cvd_fluctuation_col="wpt_cvd_fluctuation",
            volume_col="volume",
            window=64,
            ema_span=10,
            use_volume_fusion=True,
        )

        self.assertIn("hilbert_triple_divergence", result.columns)

        divergence = result["hilbert_triple_divergence"].dropna()

        if len(divergence) > 250:
            # 检查背离信号是否在正确位置触发
            divergence_after = divergence.iloc[250:300]
            divergence_rate = divergence_after.mean()

            print(f"  背离信号触发率: {divergence_rate:.2%}")

            # 在背离场景中，应该有相当比例的背离信号
            self.assertGreater(divergence_rate, 0.3, "三元背离场景应该触发背离信号")

        print("  ✅ 三元背离信号能有效识别强烈背离")

    def test_quantile_normalization_cross_asset(self):
        """
        测试 5：分位数标准化 - 跨品种可比

        验证：不同价格水平的资产，标准化后分布相似
        """
        print("\n" + "=" * 70)
        print("测试 5：分位数标准化 - 跨品种可比")
        print("=" * 70)

        # 创建三个不同价格水平的资产
        assets = {}

        # 高价格资产（如BTC）
        df_btc = self.create_base_data(300)
        df_btc["wpt_price_fluctuation"] *= 100  # 放大波动幅度
        assets["BTC"] = df_btc

        # 中价格资产（如ETH）
        df_eth = self.create_base_data(300)
        df_eth["wpt_price_fluctuation"] *= 10
        assets["ETH"] = df_eth

        # 低价格资产（如SOL）
        df_sol = self.create_base_data(300)
        assets["SOL"] = df_sol

        results = {}
        for symbol, df in assets.items():
            result = extract_hilbert_features(
                df,
                price_fluctuation_col="wpt_price_fluctuation",
                cvd_fluctuation_col="wpt_cvd_fluctuation",
                window=64,
                ema_span=10,
                use_quantile_normalize=True,
                quantile_window=100,
            )
            results[symbol] = result["hilbert_price_env_qnorm"].dropna()

        # 验证标准化后的分布相似
        for symbol, qnorm in results.items():
            print(
                f"  {symbol} - 均值: {qnorm.mean():.4f}, 标准差: {qnorm.std():.4f}, "
                f"范围: [{qnorm.min():.4f}, {qnorm.max():.4f}]"
            )

            # 验证值域在[0, 1]附近
            self.assertGreaterEqual(
                qnorm.min(),
                -0.1,  # 允许轻微负值（边界情况）
                f"{symbol} 标准化值应该 >= 0",
            )
            self.assertLessEqual(
                qnorm.max(), 1.1, f"{symbol} 标准化值应该 <= 1"  # 允许轻微超过1
            )

        # 验证不同资产的分布相似（均值接近0.5，标准差接近0.3）
        means = [qnorm.mean() for qnorm in results.values()]
        stds = [qnorm.std() for qnorm in results.values()]

        mean_std = np.std(means)
        std_std = np.std(stds)

        print(f"  均值标准差: {mean_std:.4f} (越小越好)")
        print(f"  标准差的标准差: {std_std:.4f} (越小越好)")

        # 不同资产的标准化分布应该相似
        self.assertLess(mean_std, 0.2, "不同资产的标准化均值应该相似")
        self.assertLess(std_std, 0.2, "不同资产的标准化标准差应该相似")

        print("  ✅ 分位数标准化使不同价格水平的资产具有可比性")

    def test_adaptive_window(self):
        """
        测试 6：自适应窗口

        验证：自适应窗口能根据局部周期调整窗口大小
        """
        print("\n" + "=" * 70)
        print("测试 6：自适应窗口")
        print("=" * 70)

        df = self.create_base_data()

        # 制造周期变化：前半段周期短，后半段周期长
        t = np.arange(len(df))
        df.loc[:250, "wpt_price_fluctuation"] = np.sin(
            2 * np.pi * t[:250] / 30
        )  # 周期30
        df.loc[250:, "wpt_price_fluctuation"] = np.sin(
            2 * np.pi * t[250:] / 60
        )  # 周期60

        result = extract_hilbert_features(
            df,
            price_fluctuation_col="wpt_price_fluctuation",
            cvd_fluctuation_col="wpt_cvd_fluctuation",
            window=64,
            ema_span=10,
            use_adaptive_window=True,
            base_window_min=32,
            base_window_max=128,
            period_lookback=64,
        )

        self.assertIn("hilbert_adaptive_window", result.columns)

        adaptive_windows = result["hilbert_adaptive_window"].dropna()

        if len(adaptive_windows) > 100:
            # 前半段窗口应该较小（短周期）
            window_first_half = adaptive_windows.iloc[100:200].mean()
            # 后半段窗口应该较大（长周期）
            window_second_half = adaptive_windows.iloc[300:400].mean()

            print(f"  前半段平均窗口: {window_first_half:.2f}")
            print(f"  后半段平均窗口: {window_second_half:.2f}")

            # 后半段窗口应该大于前半段
            self.assertGreater(
                window_second_half,
                window_first_half * 1.1,
                "长周期应该导致更大的自适应窗口",
            )

        print("  ✅ 自适应窗口能根据局部周期动态调整")

    def test_no_future_information(self):
        """
        测试 7：无未来信息验证（因果性测试）

        验证：特征不会使用未来信息
        """
        print("\n" + "=" * 70)
        print("测试 7：无未来信息验证（因果性测试）")
        print("=" * 70)

        df = self.create_base_data()

        # 在 t=200 处制造一个突变
        df.loc[200:, "wpt_price_fluctuation"] *= 3

        result = extract_hilbert_features(
            df,
            price_fluctuation_col="wpt_price_fluctuation",
            cvd_fluctuation_col="wpt_cvd_fluctuation",
            window=64,
            ema_span=10,
        )

        price_env = result["hilbert_price_env"].dropna()

        # 检查突变点之前的值是否稳定（不应该提前变化）
        if len(price_env) > 200:
            # 突变前（t=150-190）的值应该相对稳定
            before_mutation = price_env.iloc[150:190]
            before_std = before_mutation.std()

            # 突变后（t=210-250）的值应该明显不同
            after_mutation = price_env.iloc[210:250]
            after_mean = after_mutation.mean()
            before_mean = before_mutation.mean()

            print(f"  突变前均值: {before_mean:.4f}, 标准差: {before_std:.4f}")
            print(f"  突变后均值: {after_mean:.4f}")
            print(f"  变化幅度: {(after_mean/before_mean - 1)*100:.2f}%")

            # 突变前应该稳定（标准差小）
            self.assertLess(before_std, before_mean * 0.3, "突变前的特征值应该相对稳定")

            # 突变后应该明显不同
            self.assertGreater(
                abs(after_mean - before_mean),
                before_std * 2,
                "突变后的特征值应该明显变化",
            )

        print("  ✅ 特征不会使用未来信息（因果性验证通过）")

    def test_edge_cases(self):
        """
        测试 8：边界情况处理

        验证：NaN、异常值、数据不足等情况能正确处理
        """
        print("\n" + "=" * 70)
        print("测试 8：边界情况处理")
        print("=" * 70)

        # 测试1：前N行全NaN
        df = self.create_base_data()
        df.loc[:100, "wpt_price_fluctuation"] = np.nan

        result = extract_hilbert_features(
            df,
            price_fluctuation_col="wpt_price_fluctuation",
            cvd_fluctuation_col="wpt_cvd_fluctuation",
            window=64,
            ema_span=10,
        )

        # 前100行应该全是NaN（数据不足）
        price_env = result["hilbert_price_env"]
        self.assertTrue(price_env.iloc[:100].isna().all(), "数据不足时应该输出NaN")

        print("  ✅ NaN处理正确")

        # 测试2：数据长度小于窗口
        df_short = self.create_base_data(50)
        result_short = extract_hilbert_features(
            df_short,
            price_fluctuation_col="wpt_price_fluctuation",
            cvd_fluctuation_col="wpt_cvd_fluctuation",
            window=64,
            ema_span=10,
        )

        # 应该能正常处理（输出NaN）
        self.assertIn("hilbert_price_env", result_short.columns)
        print("  ✅ 短数据序列处理正确")

        # 测试3：全NaN输入
        df_nan = pd.DataFrame(
            {
                "wpt_price_fluctuation": np.nan * np.ones(100),
                "wpt_cvd_fluctuation": np.nan * np.ones(100),
            }
        )

        result_nan = extract_hilbert_features(
            df_nan,
            price_fluctuation_col="wpt_price_fluctuation",
            cvd_fluctuation_col="wpt_cvd_fluctuation",
            window=64,
            ema_span=10,
        )

        # 应该全部是NaN
        self.assertTrue(
            result_nan["hilbert_price_env"].isna().all(), "全NaN输入应该输出全NaN"
        )
        print("  ✅ 全NaN输入处理正确")

        print("  ✅ 所有边界情况处理正确")

    def test_performance(self):
        """
        测试 9：性能测试

        验证：计算速度在可接受范围内
        """
        print("\n" + "=" * 70)
        print("测试 9：性能测试")
        print("=" * 70)

        import time

        # 创建较大的数据集
        df = self.create_base_data(2000)

        start_time = time.time()
        result = extract_hilbert_features(
            df,
            price_fluctuation_col="wpt_price_fluctuation",
            cvd_fluctuation_col="wpt_cvd_fluctuation",
            volume_col="volume",
            window=64,
            ema_span=10,
            use_adaptive_window=True,
            use_quantile_normalize=True,
            use_volume_fusion=True,
        )
        elapsed_time = time.time() - start_time

        print(f"  数据量: {len(df)} 行")
        print(f"  计算时间: {elapsed_time:.2f} 秒")
        print(f"  速度: {len(df)/elapsed_time:.0f} 行/秒")

        # 2000行数据应该在10秒内完成
        self.assertLess(
            elapsed_time,
            10,
            f"2000行数据应该在10秒内完成，实际耗时 {elapsed_time:.2f}秒",
        )

        print("  ✅ 性能测试通过")


def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("Hilbert 特征（改进版）全面测试")
    print("=" * 70)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestHilbertFeaturesImproved)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ 所有测试通过！Hilbert特征工程实现正确且有效。")
    else:
        print("\n❌ 部分测试失败，请检查实现。")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
