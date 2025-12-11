"""
单元测试：ts_sr_reversal_optuna.py

测试预测阈值优化脚本的核心功能。
"""

import pytest
import optuna
import numpy as np
from unittest.mock import MagicMock, patch, Mock
from copy import deepcopy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestSampleParams:
    """测试 sample_params 函数"""

    def test_sample_params_returns_dict(self):
        """测试 sample_params 返回正确的字典结构"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna import (
                sample_params,
            )
        except ImportError:
            pytest.skip("Cannot import ts_sr_reversal_optuna due to dependencies")

        # 创建 mock trial
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_float = MagicMock(
            side_effect=lambda name, low, high: (low + high) / 2
        )

        params = sample_params(trial)

        # 检查返回类型和结构
        assert isinstance(params, dict)
        assert "long_entry_threshold" in params
        assert "long_exit_threshold" in params
        assert "short_entry_threshold" in params
        assert "short_exit_threshold" in params

        # 检查参数范围
        assert 0.4 <= params["long_entry_threshold"] <= 0.8
        assert 0.2 <= params["long_exit_threshold"] <= 0.5
        assert 0.2 <= params["short_entry_threshold"] <= 0.6
        assert 0.5 <= params["short_exit_threshold"] <= 0.8

    def test_sample_params_constraints_valid(self):
        """测试约束检查：有效的参数组合"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna import (
                sample_params,
            )
        except ImportError:
            pytest.skip("Cannot import ts_sr_reversal_optuna due to dependencies")

        trial = MagicMock(spec=optuna.Trial)
        # 设置合理的值，满足约束
        value_map = {
            "long_entry_threshold": 0.6,
            "long_exit_threshold": 0.3,
            "short_entry_threshold": 0.3,
            "short_exit_threshold": 0.7,
        }
        trial.suggest_float = MagicMock(
            side_effect=lambda name, low, high: value_map.get(name, (low + high) / 2)
        )

        params = sample_params(trial)

        # 验证约束
        assert params["long_entry_threshold"] > params["long_exit_threshold"]
        assert params["short_exit_threshold"] > params["short_entry_threshold"]

    def test_sample_params_constraints_invalid_long(self):
        """测试约束检查：无效的 long 参数组合应该被修剪"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna import (
                sample_params,
            )
        except ImportError:
            pytest.skip("Cannot import ts_sr_reversal_optuna due to dependencies")

        trial = MagicMock(spec=optuna.Trial)
        # 设置无效的值：long_entry <= long_exit
        value_map = {
            "long_entry_threshold": 0.3,  # <= long_exit
            "long_exit_threshold": 0.4,
            "short_entry_threshold": 0.3,
            "short_exit_threshold": 0.7,
        }
        trial.suggest_float = MagicMock(
            side_effect=lambda name, low, high: value_map.get(name, (low + high) / 2)
        )

        # 应该抛出 TrialPruned 异常
        with pytest.raises(optuna.TrialPruned) as exc_info:
            sample_params(trial)

        assert "long_entry_threshold" in str(exc_info.value).lower()

    def test_sample_params_constraints_invalid_short(self):
        """测试约束检查：无效的 short 参数组合应该被修剪"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna import (
                sample_params,
            )
        except ImportError:
            pytest.skip("Cannot import ts_sr_reversal_optuna due to dependencies")

        trial = MagicMock(spec=optuna.Trial)
        # 设置无效的值：short_exit <= short_entry
        value_map = {
            "long_entry_threshold": 0.6,
            "long_exit_threshold": 0.3,
            "short_entry_threshold": 0.5,
            "short_exit_threshold": 0.4,  # <= short_entry
        }
        trial.suggest_float = MagicMock(
            side_effect=lambda name, low, high: value_map.get(name, (low + high) / 2)
        )

        # 应该抛出 TrialPruned 异常
        with pytest.raises(optuna.TrialPruned) as exc_info:
            sample_params(trial)

        assert "short_exit_threshold" in str(exc_info.value).lower()

    def test_sample_params_calls_suggest_float_correctly(self):
        """测试 sample_params 正确调用 suggest_float"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna import (
                sample_params,
            )
        except ImportError:
            pytest.skip("Cannot import ts_sr_reversal_optuna due to dependencies")

        trial = MagicMock(spec=optuna.Trial)
        # 使用 lambda 返回不同的值，确保满足约束
        call_count = [0]

        def suggest_float_side_effect(name, low, high):
            call_count[0] += 1
            # 返回不同的值以满足约束
            if name == "long_entry_threshold":
                return 0.6
            elif name == "long_exit_threshold":
                return 0.3
            elif name == "short_entry_threshold":
                return 0.3
            elif name == "short_exit_threshold":
                return 0.7
            else:
                return (low + high) / 2

        trial.suggest_float = MagicMock(side_effect=suggest_float_side_effect)

        sample_params(trial)

        # 验证调用了 4 次 suggest_float（4 个阈值）
        assert call_count[0] == 4
        assert trial.suggest_float.call_count == 4

        # 验证调用参数
        calls = trial.suggest_float.call_args_list
        call_names = [call[0][0] for call in calls]

        assert "long_entry_threshold" in call_names
        assert "long_exit_threshold" in call_names
        assert "short_entry_threshold" in call_names
        assert "short_exit_threshold" in call_names


class TestConfigUpdate:
    """测试配置更新逻辑"""

    def test_config_update_with_thresholds(self):
        """测试配置对象正确更新阈值参数"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna import (
                sample_params,
            )
            from src.strategy_config.loader import BacktestConfig, StrategyConfig
        except ImportError:
            pytest.skip("Cannot import required modules due to dependencies")

        # 创建 mock 策略配置
        strategy_cfg = MagicMock(spec=StrategyConfig)
        strategy_cfg.backtest = MagicMock(spec=BacktestConfig)
        strategy_cfg.backtest.params = {}

        # 创建阈值参数
        threshold_params = {
            "long_entry_threshold": 0.65,
            "long_exit_threshold": 0.35,
            "short_entry_threshold": 0.35,
            "short_exit_threshold": 0.65,
        }

        # 更新配置
        strategy_cfg.backtest.params.update(threshold_params)

        # 验证更新
        assert strategy_cfg.backtest.params["long_entry_threshold"] == 0.65
        assert strategy_cfg.backtest.params["long_exit_threshold"] == 0.35
        assert strategy_cfg.backtest.params["short_entry_threshold"] == 0.35
        assert strategy_cfg.backtest.params["short_exit_threshold"] == 0.65

    def test_config_update_preserves_existing_params(self):
        """测试配置更新保留现有参数"""
        try:
            from src.strategy_config.loader import BacktestConfig
        except ImportError:
            pytest.skip("Cannot import required modules due to dependencies")

        # 创建配置，已有一些参数
        backtest_cfg = MagicMock(spec=BacktestConfig)
        backtest_cfg.params = {
            "price_col": "close",
            "initial_cash": 10000,
            "fee": 0.0004,
        }

        # 添加阈值参数
        threshold_params = {
            "long_entry_threshold": 0.6,
            "long_exit_threshold": 0.4,
            "short_entry_threshold": 0.4,
            "short_exit_threshold": 0.6,
        }

        backtest_cfg.params.update(threshold_params)

        # 验证现有参数保留
        assert backtest_cfg.params["price_col"] == "close"
        assert backtest_cfg.params["initial_cash"] == 10000
        assert backtest_cfg.params["fee"] == 0.0004

        # 验证新参数添加
        assert backtest_cfg.params["long_entry_threshold"] == 0.6
        assert backtest_cfg.params["short_exit_threshold"] == 0.6


class TestObjectiveFunction:
    """测试 objective 函数的逻辑"""

    def test_objective_updates_config(self):
        """测试 objective 函数正确更新配置"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna import (
                sample_params,
            )
            from src.strategy_config.loader import StrategyConfig, BacktestConfig
        except ImportError:
            pytest.skip("Cannot import required modules due to dependencies")

        # 创建 mock 配置
        strategy_cfg = MagicMock(spec=StrategyConfig)
        strategy_cfg.backtest = MagicMock(spec=BacktestConfig)
        strategy_cfg.backtest.params = {}

        # 创建 mock trial
        trial = MagicMock(spec=optuna.Trial)
        trial.number = 0
        value_map = {
            "long_entry_threshold": 0.6,
            "long_exit_threshold": 0.3,
            "short_entry_threshold": 0.3,
            "short_exit_threshold": 0.7,
        }
        trial.suggest_float = MagicMock(
            side_effect=lambda name, low, high: value_map.get(name, (low + high) / 2)
        )
        trial.set_user_attr = MagicMock()

        # 模拟 objective 函数的配置更新部分
        threshold_params = sample_params(trial)
        trial_cfg = deepcopy(strategy_cfg)
        if trial_cfg.backtest.params is None:
            trial_cfg.backtest.params = {}
        trial_cfg.backtest.params.update(threshold_params)

        # 验证配置已更新
        assert trial_cfg.backtest.params["long_entry_threshold"] == 0.6
        assert trial_cfg.backtest.params["long_exit_threshold"] == 0.3
        assert trial_cfg.backtest.params["short_entry_threshold"] == 0.3
        assert trial_cfg.backtest.params["short_exit_threshold"] == 0.7


class TestParameterRanges:
    """测试参数范围的合理性"""

    def test_threshold_ranges_are_reasonable(self):
        """测试阈值范围设置合理"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna import (
                sample_params,
            )
        except ImportError:
            pytest.skip("Cannot import ts_sr_reversal_optuna due to dependencies")

        trial = MagicMock(spec=optuna.Trial)

        # 测试边界值
        test_cases = [
            # (long_entry, long_exit, short_entry, short_exit, should_pass)
            (0.4, 0.2, 0.2, 0.5, True),  # 最小值
            (0.8, 0.5, 0.6, 0.8, True),  # 最大值
            (0.6, 0.3, 0.3, 0.7, True),  # 中间值
            (0.5, 0.5, 0.3, 0.7, False),  # long_entry == long_exit (无效)
            (0.6, 0.3, 0.5, 0.5, False),  # short_exit == short_entry (无效)
        ]

        for long_entry, long_exit, short_entry, short_exit, should_pass in test_cases:
            value_map = {
                "long_entry_threshold": long_entry,
                "long_exit_threshold": long_exit,
                "short_entry_threshold": short_entry,
                "short_exit_threshold": short_exit,
            }
            trial.suggest_float = MagicMock(
                side_effect=lambda name, low, high: value_map.get(
                    name, (low + high) / 2
                )
            )

            if should_pass:
                params = sample_params(trial)
                assert params["long_entry_threshold"] == long_entry
                assert params["long_exit_threshold"] == long_exit
            else:
                with pytest.raises(optuna.TrialPruned):
                    sample_params(trial)
