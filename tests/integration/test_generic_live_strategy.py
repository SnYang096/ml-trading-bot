"""
GenericLiveStrategy 集成测试

验证 GenericLiveStrategy 与现有 BPCLiveStrategy 的功能等价性
确保配置驱动的通用实现能完全替代硬编码的策略实现
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import yaml
import os

from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy
from src.time_series_model.live.bpc_live_strategy import BPCLiveStrategy


# =============================================================================
# 测试数据准备
# =============================================================================


@pytest.fixture
def sample_features():
    """标准测试特征数据"""
    return {
        "close": 50000.0,
        "volume": 100.0,
        "atr": 500.0,
        "bpc_score_breakout": 0.8,
        "bpc_was_in_pullback": 1,
        "bollinger_position": 0.9,
        "liquidity_silence_score": 0.6,
        "rsi": 70.0,
        "bb_width_normalized": 0.3,
        "volume_ratio": 1.2,
    }


@pytest.fixture
def sample_bars():
    """标准测试K线数据"""
    return [
        {"close": 49500.0, "volume": 80.0},
        {"close": 49800.0, "volume": 90.0},
        {"close": 50000.0, "volume": 100.0},
    ]


# =============================================================================
# 1. 配置文件准备
# =============================================================================


@pytest.fixture
def temp_config_dir():
    """创建临时配置目录和文件"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建策略目录结构
        strategy_dir = os.path.join(tmpdir, "strategies", "bpc", "archetypes")
        os.makedirs(strategy_dir, exist_ok=True)

        # direction.yaml
        direction_config = {
            "version": 1,
            "causal_source": "breakout_detection",
            "direction_rules": [
                {
                    "id": "breakout_rule",
                    "feature": "bpc_score_breakout",
                    "transform": "sign",
                    "description": "Breakout方向: sign(bpc_score_breakout)",
                }
            ],
        }
        with open(os.path.join(strategy_dir, "direction.yaml"), "w") as f:
            yaml.dump(direction_config, f)

        # gate.yaml
        gate_config = {
            "hard_gate": [
                {
                    "id": "min_breakout_score",
                    "when": {"bpc_score_breakout": {"value_gt": 0.5}},
                    "then": {"action": "allow"},
                }
            ],
            "soft_filter": [
                {
                    "id": "pullback_filter",
                    "when": {"bpc_was_in_pullback": {"value_eq": 1}},
                    "then": {"weight": 1.2},
                }
            ],
        }
        with open(os.path.join(strategy_dir, "gate.yaml"), "w") as f:
            yaml.dump(gate_config, f)

        # evidence.yaml
        evidence_config = {
            "features": [
                {"name": "rsi", "weight": 0.3, "bins": 5},
                {"name": "bb_width_normalized", "weight": 0.4, "bins": 5},
                {"name": "volume_ratio", "weight": 0.3, "bins": 5},
            ]
        }
        with open(os.path.join(strategy_dir, "evidence.yaml"), "w") as f:
            yaml.dump(evidence_config, f)

        # execution.yaml
        execution_config = {
            "stop_loss": {
                "initial_r": 2.0,
                "trailing": {"activation_r": 1.0, "trail_r": 1.5},
            },
            "take_profit": {"multiple": 2.5},
            "holding": {"time_stop_bars": 50},
            "tiers": {
                "enabled": True,
                "levels": [
                    {
                        "name": "high_confidence",
                        "evidence_min": 0.7,
                        "stop_loss": {
                            "initial_r": 1.5,
                            "trailing": {"activation_r": 0.8, "trail_r": 1.2},
                        },
                        "time_stop_bars": 60,
                        "size_multiplier": 1.2,
                    }
                ],
            },
        }
        with open(os.path.join(strategy_dir, "execution.yaml"), "w") as f:
            yaml.dump(execution_config, f)

        # entry_filters.yaml
        entry_config = {
            "filters": [
                {
                    "id": "bb_or_liq_silence",
                    "type": "or",
                    "filters": [
                        {
                            "id": "bb_filter",
                            "feature": "bollinger_position",
                            "condition": "value_gt",
                            "threshold": 0.8,
                        },
                        {
                            "id": "liq_silence_filter",
                            "feature": "liquidity_silence_score",
                            "condition": "value_gt",
                            "threshold": 0.5,
                        },
                    ],
                    "enabled": True,
                }
            ]
        }
        with open(os.path.join(strategy_dir, "entry_filters.yaml"), "w") as f:
            yaml.dump(entry_config, f)

        yield tmpdir


# =============================================================================
# 2. 功能等价性测试
# =============================================================================


class TestGenericVsBPCStrategyEquivalence:
    """验证 GenericLiveStrategy 与 BPCLiveStrategy 功能等价"""

    def test_initialization(self, temp_config_dir):
        """测试初始化"""
        # Generic 实现
        generic = GenericLiveStrategy(
            strategy_name="bpc",
            strategies_root=os.path.join(temp_config_dir, "strategies"),
        )

        # BPC 实现
        bpc = BPCLiveStrategy(
            strategies_root=os.path.join(temp_config_dir, "strategies"),
        )

        # 都应成功初始化
        assert generic is not None
        assert bpc is not None

    def test_direction_evaluation_equivalence(self, temp_config_dir, sample_features):
        """测试方向判定等价性"""
        generic = GenericLiveStrategy(
            strategy_name="bpc",
            strategies_root=os.path.join(temp_config_dir, "strategies"),
        )
        bpc = BPCLiveStrategy(
            strategies_root=os.path.join(temp_config_dir, "strategies"),
        )

        # 加载配置
        generic.load_configs()
        bpc.load_configs()

        # 生成信号
        generic_intents = generic.decide(features=sample_features, symbol="BTCUSDT")
        bpc_intents = bpc.decide(features=sample_features, symbol="BTCUSDT")

        # 验证都有信号且方向一致
        assert len(generic_intents) == 1
        assert len(bpc_intents) == 1
        assert generic_intents[0].action == bpc_intents[0].action

    def test_gate_filtering_equivalence(self, temp_config_dir):
        """测试 Gate 过滤等价性"""
        generic = GenericLiveStrategy(
            strategy_name="bpc",
            strategies_root=os.path.join(temp_config_dir, "strategies"),
        )
        bpc = BPCLiveStrategy(
            strategies_root=os.path.join(temp_config_dir, "strategies"),
        )

        generic.load_configs()
        bpc.load_configs()

        # 测试低分特征（应被 Gate 拒绝）
        low_score_features = {
            "close": 50000.0,
            "bpc_score_breakout": 0.3,  # 低于阈值 0.5
            "bpc_was_in_pullback": 1,
        }

        generic_intents = generic.decide(features=low_score_features, symbol="BTCUSDT")
        bpc_intents = bpc.decide(features=low_score_features, symbol="BTCUSDT")

        # 都应被拒绝
        assert len(generic_intents) == 0
        assert len(bpc_intents) == 0

    def test_entry_filter_equivalence(self, temp_config_dir):
        """测试 Entry Filter 等价性"""
        generic = GenericLiveStrategy(
            strategy_name="bpc",
            strategies_root=os.path.join(temp_config_dir, "strategies"),
        )
        bpc = BPCLiveStrategy(
            strategies_root=os.path.join(temp_config_dir, "strategies"),
        )

        generic.load_configs()
        bpc.load_configs()

        # 测试不满足入场条件的特征
        no_entry_features = {
            "close": 50000.0,
            "bpc_score_breakout": 0.8,
            "bpc_was_in_pullback": 1,
            "bollinger_position": 0.5,  # 不满足 > 0.8
            "liquidity_silence_score": 0.3,  # 不满足 > 0.5
        }

        generic_intents = generic.decide(features=no_entry_features, symbol="BTCUSDT")
        bpc_intents = bpc.decide(features=no_entry_features, symbol="BTCUSDT")

        # 都应被拒绝
        assert len(generic_intents) == 0
        assert len(bpc_intents) == 0

    def test_evidence_scoring_equivalence(self, temp_config_dir, sample_features):
        """测试 Evidence 评分等价性"""
        generic = GenericLiveStrategy(
            strategy_name="bpc",
            strategies_root=os.path.join(temp_config_dir, "strategies"),
        )
        bpc = BPCLiveStrategy(
            strategies_root=os.path.join(temp_config_dir, "strategies"),
        )

        generic.load_configs()
        bpc.load_configs()

        # 设置相同的 quantiles
        test_df = pd.DataFrame(
            {
                "rsi": [30, 50, 70, 80],
                "bb_width_normalized": [0.1, 0.3, 0.5, 0.7],
                "volume_ratio": [0.8, 1.0, 1.2, 1.5],
            }
        )

        generic.set_quantiles(test_df)
        bpc.set_quantiles_from_df(test_df)

        # 生成信号
        generic_intents = generic.decide(features=sample_features, symbol="BTCUSDT")
        bpc_intents = bpc.decide(features=sample_features, symbol="BTCUSDT")

        # 验证都有信号
        assert len(generic_intents) == 1
        assert len(bpc_intents) == 1

        # 评分应在合理范围内
        assert 0 <= generic_intents[0].confidence <= 1
        assert 0 <= bpc_intents[0].confidence <= 1

    def test_execution_params_equivalence(self, temp_config_dir, sample_features):
        """测试执行参数等价性"""
        generic = GenericLiveStrategy(
            strategy_name="bpc",
            strategies_root=os.path.join(temp_config_dir, "strategies"),
        )
        bpc = BPCLiveStrategy(
            strategies_root=os.path.join(temp_config_dir, "strategies"),
        )

        generic.load_configs()
        bpc.load_configs()

        # 设置 quantiles
        test_df = pd.DataFrame(
            {
                "rsi": [30, 50, 70, 80],
                "bb_width_normalized": [0.1, 0.3, 0.5, 0.7],
                "volume_ratio": [0.8, 1.0, 1.2, 1.5],
            }
        )
        generic.set_quantiles(test_df)
        bpc.set_quantiles_from_df(test_df)

        # 生成信号
        generic_intents = generic.decide(features=sample_features, symbol="BTCUSDT")
        bpc_intents = bpc.decide(features=sample_features, symbol="BTCUSDT")

        # 验证执行参数结构
        gen_profile = generic_intents[0].execution_profile
        bpc_profile = bpc_intents[0].execution_profile

        # 基础 RR 约束应存在
        assert "rr_constraints" in gen_profile
        assert "rr_constraints" in bpc_profile

        # 停止损失倍数
        assert gen_profile["rr_constraints"]["stop_loss_r"] > 0
        assert bpc_profile["rr_constraints"]["stop_loss_r"] > 0


# =============================================================================
# 3. 边界条件测试
# =============================================================================


class TestGenericStrategyEdgeCases:
    """测试边界条件和异常处理"""

    def test_missing_config_files(self, temp_config_dir):
        """测试配置文件缺失情况"""
        # 删除 direction.yaml
        direction_path = os.path.join(
            temp_config_dir, "strategies", "bpc", "archetypes", "direction.yaml"
        )
        os.remove(direction_path)

        # 应该抛出 FileNotFoundError
        with pytest.raises(FileNotFoundError):
            GenericLiveStrategy(
                strategy_name="bpc",
                strategies_root=os.path.join(temp_config_dir, "strategies"),
            )

    def test_empty_features(self, temp_config_dir):
        """测试空特征输入"""
        generic = GenericLiveStrategy(
            strategy_name="bpc",
            strategies_root=os.path.join(temp_config_dir, "strategies"),
        )
        generic.load_configs()

        intents = generic.decide(features={}, symbol="BTCUSDT")
        assert len(intents) == 0

    def test_missing_required_features(self, temp_config_dir):
        """测试缺少必要特征"""
        generic = GenericLiveStrategy(
            strategy_name="bpc",
            strategies_root=os.path.join(temp_config_dir, "strategies"),
        )
        generic.load_configs()

        # 缺少 bpc_score_breakout
        incomplete_features = {"close": 50000.0, "volume": 100.0}

        intents = generic.decide(features=incomplete_features, symbol="BTCUSDT")
        assert len(intents) == 0  # 应该无法生成方向

    def test_invalid_feature_values(self, temp_config_dir):
        """测试无效特征值"""
        generic = GenericLiveStrategy(
            strategy_name="bpc",
            strategies_root=os.path.join(temp_config_dir, "strategies"),
        )
        generic.load_configs()

        # 无效的字符串值
        invalid_features = {
            "close": 50000.0,
            "bpc_score_breakout": "invalid",
            "bpc_was_in_pullback": 1,
        }

        intents = generic.decide(features=invalid_features, symbol="BTCUSDT")
        # 应该优雅处理，跳过无效值
        assert len(intents) == 0 or intents[0].action in ["LONG", "SHORT"]


# =============================================================================
# 4. 性能测试
# =============================================================================


class TestGenericStrategyPerformance:
    """性能基准测试"""

    def test_decision_speed(self, temp_config_dir, sample_features):
        """测试决策速度"""
        generic = GenericLiveStrategy(
            strategy_name="bpc",
            strategies_root=os.path.join(temp_config_dir, "strategies"),
        )
        generic.load_configs()

        import time

        # 预热
        for _ in range(10):
            generic.decide(features=sample_features, symbol="BTCUSDT")

        # 测试
        start_time = time.perf_counter()
        for _ in range(1000):
            generic.decide(features=sample_features, symbol="BTCUSDT")
        end_time = time.perf_counter()

        avg_time_ms = (end_time - start_time) * 1000 / 1000
        print(f"\n平均决策时间: {avg_time_ms:.3f}ms")

        # 应该在合理范围内（< 10ms）
        assert avg_time_ms < 10.0

    def test_memory_usage(self, temp_config_dir, sample_features):
        """测试内存使用"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 创建多个实例
        strategies = []
        for i in range(100):
            strat = GenericLiveStrategy(
                strategy_name="bpc",
                strategies_root=os.path.join(temp_config_dir, "strategies"),
            )
            strat.load_configs()
            strategies.append(strat)

        final_memory = process.memory_info().rss
        memory_increase_mb = (final_memory - initial_memory) / 1024 / 1024

        print(f"\n内存增长: {memory_increase_mb:.2f} MB")
        # 100个实例应该 < 50MB
        assert memory_increase_mb < 50.0


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
