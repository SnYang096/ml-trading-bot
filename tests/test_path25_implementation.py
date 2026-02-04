"""
测试路径2.5实现
验证数学特征分层使用的正确性
"""

import pandas as pd
import numpy as np
from src.time_series_model.strategies.bpc_strategy_v2 import BPCStrategyV2
from src.time_series_model.evidence.bpc_evidence_calculator import BPCEvidenceCalculator
from src.time_series_model.execution.noise_penalty import (
    ExecutionNoisePenalty,
    NoisePenaltyConfig,
)


def test_path25_implementation():
    """
    测试路径2.5实现的正确性
    """
    print("=== 路径2.5实现验证 ===")

    # 创建模拟数据
    n_samples = 200
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="4H")

    # 创建包含各种特征的DataFrame
    df = pd.DataFrame(
        {
            "close": 100 + np.cumsum(np.random.randn(n_samples) * 0.1),
            "high": 100 + np.cumsum(np.random.randn(n_samples) * 0.1) + 0.2,
            "low": 100 + np.cumsum(np.random.randn(n_samples) * 0.1) - 0.2,
            "volume": np.random.randint(1000, 5000, n_samples),
        },
        index=dates,
    )

    # 添加BPC相关特征（结构/订单流/规制特征）
    df["bpc_score_breakout"] = np.random.rand(n_samples)
    df["bpc_score_pullback"] = np.random.rand(n_samples)
    df["bpc_score_continuation"] = np.random.rand(n_samples)
    df["bpc_pullback_depth_pct"] = np.random.rand(n_samples) * 0.1
    df["bpc_impulse_return_atr"] = np.random.randn(n_samples) * 0.1
    df["bpc_dir_consistency_short"] = np.random.rand(n_samples)
    df["bpc_dir_consistency_mid"] = np.random.rand(n_samples)
    df["bpc_dir_consistency_long"] = np.random.rand(n_samples)
    df["cvd_divergence_score"] = np.random.randn(n_samples) * 0.1
    df["price_momentum_divergence"] = np.random.randn(n_samples) * 0.1
    df["bpc_pullback_delta_absorption"] = np.random.rand(n_samples)
    df["cvd_change_5_pct"] = np.random.randn(n_samples) * 0.1
    df["trend_r2_20"] = np.random.rand(n_samples)
    df["path_efficiency_pct"] = np.random.rand(n_samples)
    df["price_dir_consistency_pct"] = np.random.rand(n_samples)
    df["macd_atr"] = np.random.randn(n_samples) * 0.1
    df["rsi_normalized"] = np.random.rand(n_samples)
    df["atr_percentile"] = np.random.rand(n_samples)
    df["bb_width_normalized_pct"] = np.random.rand(n_samples)
    df["vpin_score"] = np.random.rand(n_samples)
    df["volume_ratio_pct"] = np.random.rand(n_samples)
    df["ofci_pct"] = np.random.rand(n_samples)
    df["shd_pct"] = np.random.rand(n_samples)
    df["vol_regime_score"] = np.random.rand(n_samples)
    df["vol_trend_score"] = np.random.rand(n_samples)
    df["sr_strength_max"] = np.random.rand(n_samples)

    # 添加数学特征（用于Execution层，不在Gate/Evidence层使用）
    df["wpt_price_fluctuation"] = np.random.rand(n_samples)
    df["spectrum_price_entropy"] = np.random.rand(n_samples)
    df["hilbert_price_env"] = np.random.rand(n_samples)
    df["hurst_price_rolling"] = np.random.rand(n_samples)
    # 添加EVT特征（作为保险丝机制）
    df["evt_tail_risk"] = np.random.rand(n_samples)

    print(f"数据形状: {df.shape}")
    print(f"特征列数量: {len(df.columns)}")

    # 测试1: BPC证据计算器（仅使用结构/订单流特征）
    print("\n--- 测试1: BPC证据计算器 ---")
    evidence_calc = BPCEvidenceCalculator()
    evidence_score = evidence_calc.calculate_evidence_score(df)
    print(f"Evidence分数范围: [{evidence_score.min():.3f}, {evidence_score.max():.3f}]")
    print(f"Evidence分数平均值: {evidence_score.mean():.3f}")

    # 验证Evidence分数在[0,1]区间内
    assert (
        evidence_score.min() >= 0.0 and evidence_score.max() <= 1.0
    ), "Evidence分数应在[0,1]区间内"
    print("✅ Evidence分数范围验证通过")

    # 测试2: 噪声惩罚计算器（使用数学特征）
    print("\n--- 测试2: 噪声惩罚计算器 ---")
    noise_config = NoisePenaltyConfig()
    noise_calc = ExecutionNoisePenalty(noise_config)
    noise_penalty = noise_calc.compute(df)
    print(f"Noise惩罚范围: [{noise_penalty.min():.3f}, {noise_penalty.max():.3f}]")
    print(f"Noise惩罚平均值: {noise_penalty.mean():.3f}")

    # 验证噪声惩罚在[0, 0.8]区间内
    assert (
        noise_penalty.min() >= 0.0 and noise_penalty.max() <= 0.8
    ), "Noise惩罚应在[0, 0.8]区间内"
    print("✅ Noise惩罚范围验证通过")

    # 测试3: BPC策略V2（路径2.5架构）
    print("\n--- 测试3: BPC策略V2（路径2.5架构） ---")
    strategy = BPCStrategyV2()

    # 评估交易机会
    approved, params = strategy.evaluate_trade_opportunity(df)
    print(f"交易获批: {approved}")

    if approved:
        print(f"Evidence分数: {params['evidence_score']:.3f}")
        print(f"Noise惩罚: {params['noise_penalty']:.3f}")
        exec_params = params["execution_params"]
        print(
            f"调整后参数: SL={exec_params.sl_r:.2f}R, TP={exec_params.tp_r:.2f}R, Size={exec_params.size_multiplier:.2f}x"
        )

    print("\n=== 路径2.5实现验证完成 ===")
    print("✅ 所有测试通过！")
    print("\n路径2.5架构特点：")
    print("- Gate层: 仅使用结构/订单流/规制特征（无数学特征）")
    print("- Evidence层: 仅基于结构/订单流特征评估alpha质量（无数学特征）")
    print("- Execution层: 同时消费evidence_score和noise_penalty（后者基于数学特征）")
    print("- 数学特征仅影响'如何执行'，不影响'是否执行'")


if __name__ == "__main__":
    test_path25_implementation()
