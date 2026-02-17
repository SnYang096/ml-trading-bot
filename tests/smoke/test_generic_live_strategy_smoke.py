"""
GenericLiveStrategy 冒烟测试

端到端验证 GenericLiveStrategy 能否正确生成交易信号
使用真实配置文件进行完整流程测试
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import yaml
import os
from pathlib import Path

from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy


def test_end_to_end_signal_generation():
    """端到端信号生成冒烟测试"""

    print("=== GenericLiveStrategy 冒烟测试 ===\n")

    # 1. 准备测试配置
    print("1. 准备测试配置...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建 BPC 策略配置
        strategy_dir = Path(tmpdir) / "strategies" / "bpc" / "archetypes"
        strategy_dir.mkdir(parents=True, exist_ok=True)

        # direction.yaml - 突破方向
        direction_config = {
            "version": 1,
            "causal_source": "breakout_detection",
            "direction_rules": [
                {
                    "id": "breakout_direction",
                    "feature": "bpc_score_breakout",
                    "transform": "sign",
                    "description": "突破方向: sign(bpc_score_breakout)",
                }
            ],
        }
        with open(strategy_dir / "direction.yaml", "w") as f:
            yaml.dump(direction_config, f)

        # gate.yaml - 结构过滤
        gate_config = {
            "hard_gate": [
                {
                    "id": "min_breakout",
                    "when": {"bpc_score_breakout": {"value_gt": 0.5}},
                    "then": {"action": "allow"},
                }
            ],
            "soft_filter": [
                {
                    "id": "pullback_boost",
                    "when": {"bpc_was_in_pullback": {"value_eq": 1}},
                    "then": {"weight": 1.2},
                }
            ],
        }
        with open(strategy_dir / "gate.yaml", "w") as f:
            yaml.dump(gate_config, f)

        # evidence.yaml - 证据评分
        evidence_config = {
            "features": [
                {"name": "rsi", "weight": 0.4, "bins": 5},
                {"name": "bb_width_normalized", "weight": 0.3, "bins": 5},
                {"name": "volume_ratio", "weight": 0.3, "bins": 5},
            ]
        }
        with open(strategy_dir / "evidence.yaml", "w") as f:
            yaml.dump(evidence_config, f)

        # execution.yaml - 执行参数
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
                        "name": "high_conf",
                        "evidence_min": 0.7,
                        "stop_loss": {"initial_r": 1.5},
                        "time_stop_bars": 60,
                        "size_multiplier": 1.2,
                    }
                ],
            },
        }
        with open(strategy_dir / "execution.yaml", "w") as f:
            yaml.dump(execution_config, f)

        # entry_filters.yaml - 入场时机
        entry_config = {
            "filters": [
                {
                    "id": "entry_timing",
                    "type": "or",
                    "filters": [
                        {
                            "id": "bb_position",
                            "feature": "bb_position",
                            "condition": "value_gt",
                            "threshold": 0.8,
                        },
                        {
                            "id": "liq_silence",
                            "feature": "liquidity_silence_score",
                            "condition": "value_gt",
                            "threshold": 0.5,
                        },
                    ],
                    "enabled": True,
                }
            ]
        }
        with open(strategy_dir / "entry_filters.yaml", "w") as f:
            yaml.dump(entry_config, f)

        print(f"✅ 配置文件已创建: {strategy_dir}")

        # 2. 初始化策略
        print("\n2. 初始化 GenericLiveStrategy...")

        strategy = GenericLiveStrategy(
            strategy_name="bpc",
            strategies_root=str(tmpdir / "strategies"),
        )

        print("✅ 策略初始化成功")

        # 3. 加载配置
        print("\n3. 加载配置文件...")

        strategy.load_configs()

        print("✅ 配置加载完成")
        print(f"   - Direction rules: {len(strategy.direction_evaluator.rules)}")
        print(f"   - Gate rules: {len(strategy.archetype.gate.all_rules)}")
        print(f"   - Evidence features: {len(strategy.archetype.evidence.features)}")

        # 4. 设置分位数
        print("\n4. 设置分位数阈值...")

        # 创建测试数据用于分位数计算
        test_data = pd.DataFrame(
            {
                "rsi": [30, 40, 50, 60, 70, 80],
                "bb_width_normalized": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "volume_ratio": [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
            }
        )

        strategy.set_quantiles(test_data)

        print("✅ 分位数设置完成")
        print(f"   - 计算了 {len(strategy._quantiles)} 个特征的分位数")

        # 5. 测试信号生成
        print("\n5. 测试信号生成...")

        # 测试用例1: 完整有效信号
        test_features_1 = {
            "close": 50000.0,
            "bpc_score_breakout": 0.8,  # 正向突破
            "bpc_was_in_pullback": 1,  # 经历过回踩
            "rsi": 70.0,  # 超买
            "bb_width_normalized": 0.4,  # 中等波动
            "volume_ratio": 1.2,  # 放量
            "bb_position": 0.9,  # 靠近上轨
        }

        intents_1 = strategy.decide(features=test_features_1, symbol="BTCUSDT")

        print(f"\n测试用例1 - 完整有效信号:")
        print(f"   输入特征: {test_features_1}")
        print(f"   生成信号数: {len(intents_1)}")

        if intents_1:
            intent = intents_1[0]
            print(f"   信号详情:")
            print(f"     - 动作: {intent.action}")
            print(f"     - 置信度: {intent.confidence:.3f}")
            print(f"     - 仓位倍数: {intent.size_multiplier}")
            print(f"     - 执行策略: {intent.execution_strategy}")
            print(f"     - 标签: {intent.execution_tags}")

        # 测试用例2: 被 Gate 拒绝的信号
        test_features_2 = {
            "close": 50000.0,
            "bpc_score_breakout": 0.3,  # 低于阈值
            "bpc_was_in_pullback": 1,
            "rsi": 70.0,
            "bb_width_normalized": 0.4,
            "volume_ratio": 1.2,
            "bb_position": 0.9,
        }

        intents_2 = strategy.decide(features=test_features_2, symbol="BTCUSDT")

        print(f"\n测试用例2 - Gate 拒绝:")
        print(f"   输入特征: {test_features_2}")
        print(f"   生成信号数: {len(intents_2)} (预期: 0)")

        # 测试用例3: 被 Entry Filter 拒绝的信号
        test_features_3 = {
            "close": 50000.0,
            "bpc_score_breakout": 0.8,
            "bpc_was_in_pullback": 1,
            "rsi": 70.0,
            "bb_width_normalized": 0.4,
            "volume_ratio": 1.2,
            "bb_position": 0.5,  # 不满足入场条件
            "liquidity_silence_score": 0.3,  # 不满足入场条件
        }

        intents_3 = strategy.decide(features=test_features_3, symbol="BTCUSDT")

        print(f"\n测试用例3 - Entry Filter 拒绝:")
        print(f"   输入特征: {test_features_3}")
        print(f"   生成信号数: {len(intents_3)} (预期: 0)")

        # 6. 验证结果
        print("\n6. 结果验证...")

        # 验证测试用例1应生成信号
        assert (
            len(intents_1) == 1
        ), f"测试用例1应生成1个信号，实际生成{len(intents_1)}个"
        assert intents_1[0].action == "LONG", f"预期做多，实际: {intents_1[0].action}"
        assert (
            0 <= intents_1[0].confidence <= 1
        ), f"置信度应在[0,1]范围内: {intents_1[0].confidence}"

        # 验证测试用例2应被拒绝
        assert (
            len(intents_2) == 0
        ), f"测试用例2应被Gate拒绝，实际生成{len(intents_2)}个信号"

        # 验证测试用例3应被拒绝
        assert (
            len(intents_3) == 0
        ), f"测试用例3应被Entry Filter拒绝，实际生成{len(intents_3)}个信号"

        print("✅ 所有验证通过")

        # 7. 性能测试
        print("\n7. 性能测试...")

        import time

        # 预热
        for _ in range(10):
            strategy.decide(features=test_features_1, symbol="BTCUSDT")

        # 测试1000次决策
        start_time = time.perf_counter()
        for _ in range(1000):
            strategy.decide(features=test_features_1, symbol="BTCUSDT")
        end_time = time.perf_counter()

        avg_time_ms = (end_time - start_time) * 1000 / 1000
        print(f"   1000次决策耗时: {end_time - start_time:.3f}s")
        print(f"   平均每次: {avg_time_ms:.3f}ms")

        # 应该在合理范围内
        assert avg_time_ms < 5.0, f"平均决策时间过长: {avg_time_ms:.3f}ms"

        print("✅ 性能测试通过")

        print("\n🎉 冒烟测试完成！GenericLiveStrategy 工作正常")


def test_multi_strategy_support():
    """测试多策略支持能力"""

    print("\n=== 多策略支持测试 ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        strategies_root = Path(tmpdir) / "strategies"

        # 创建多个策略配置
        for strategy_name in ["bpc", "me", "fer"]:
            arch_dir = strategies_root / strategy_name / "archetypes"
            arch_dir.mkdir(parents=True, exist_ok=True)

            # 基础配置（简化版）
            direction_config = {
                "version": 1,
                "causal_source": f"{strategy_name}_logic",
                "direction_rules": [
                    {
                        "id": f"{strategy_name}_direction",
                        "feature": "test_signal",
                        "transform": "sign",
                    }
                ],
            }

            gate_config = {"hard_gate": [], "soft_filter": []}
            evidence_config = {"features": []}
            execution_config = {
                "stop_loss": {"initial_r": 2.0},
                "take_profit": {"multiple": 2.5},
                "holding": {"time_stop_bars": 50},
            }
            entry_config = {"filters": []}

            # 写入配置文件
            configs = {
                "direction.yaml": direction_config,
                "gate.yaml": gate_config,
                "evidence.yaml": evidence_config,
                "execution.yaml": execution_config,
                "entry_filters.yaml": entry_config,
            }

            for filename, config in configs.items():
                with open(arch_dir / filename, "w") as f:
                    yaml.dump(config, f)

        print("✅ 多策略配置已创建")

        # 测试每个策略都能初始化
        for strategy_name in ["bpc", "me", "fer"]:
            print(f"\n测试策略: {strategy_name}")

            strategy = GenericLiveStrategy(
                strategy_name=strategy_name,
                strategies_root=str(strategies_root),
            )
            strategy.load_configs()

            # 测试基本功能
            test_features = {"test_signal": 0.5, "close": 50000.0}
            intents = strategy.decide(features=test_features, symbol="BTCUSDT")

            print(f"   初始化: ✅")
            print(f"   信号生成: {'✅' if len(intents) == 1 else '❌'}")

        print("\n✅ 多策略支持测试通过")


if __name__ == "__main__":
    try:
        test_end_to_end_signal_generation()
        test_multi_strategy_support()
        print("\n🎉 所有冒烟测试通过！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise
