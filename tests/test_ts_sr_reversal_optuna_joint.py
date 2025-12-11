"""
单元测试：ts_sr_reversal_optuna_joint.py

测试联合优化脚本的核心功能。
"""

import pytest
import optuna
import numpy as np
from unittest.mock import MagicMock
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestSampleModelParams:
    """测试 sample_model_params 函数"""

    def test_sample_xgboost_params(self):
        """测试 XGBoost 参数采样"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna_joint import (
                sample_model_params,
            )
        except ImportError:
            pytest.skip("Cannot import ts_sr_reversal_optuna_joint due to dependencies")

        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_int = MagicMock(return_value=6)
        # suggest_float 可能被调用时带有 log=True 参数
        trial.suggest_float = MagicMock(
            side_effect=lambda name, low, high, **kwargs: 0.05
        )

        params = sample_model_params(trial, model_type="xgboost")

        assert isinstance(params, dict)
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "n_estimators" in params
        assert "subsample" in params
        assert "colsample_bytree" in params

    def test_sample_lightgbm_params(self):
        """测试 LightGBM 参数采样"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna_joint import (
                sample_model_params,
            )
        except ImportError:
            pytest.skip("Cannot import ts_sr_reversal_optuna_joint due to dependencies")

        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_int = MagicMock(return_value=31)
        # suggest_float 可能被调用时带有 log=True 参数
        trial.suggest_float = MagicMock(
            side_effect=lambda name, low, high, **kwargs: 0.05
        )

        params = sample_model_params(trial, model_type="lightgbm")

        assert isinstance(params, dict)
        assert "num_leaves" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "feature_fraction" in params
        assert "bagging_fraction" in params

    def test_unsupported_model_type(self):
        """测试不支持的模型类型"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna_joint import (
                sample_model_params,
            )
        except ImportError:
            pytest.skip("Cannot import ts_sr_reversal_optuna_joint due to dependencies")

        trial = MagicMock(spec=optuna.Trial)

        with pytest.raises(ValueError, match="Unsupported model_type"):
            sample_model_params(trial, model_type="unsupported")


class TestSampleThresholdParams:
    """测试 sample_threshold_params 函数"""

    def test_sample_threshold_params_returns_dict(self):
        """测试 sample_threshold_params 返回正确的字典结构"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna_joint import (
                sample_threshold_params,
            )
        except ImportError:
            pytest.skip("Cannot import ts_sr_reversal_optuna_joint due to dependencies")

        trial = MagicMock(spec=optuna.Trial)
        value_map = {
            "long_entry_threshold": 0.6,
            "long_exit_threshold": 0.3,
            "short_entry_threshold": 0.3,
            "short_exit_threshold": 0.7,
        }
        trial.suggest_float = MagicMock(
            side_effect=lambda name, low, high, **kwargs: value_map.get(
                name, (low + high) / 2
            )
        )

        params = sample_threshold_params(trial)

        assert isinstance(params, dict)
        assert "long_entry_threshold" in params
        assert "long_exit_threshold" in params
        assert "short_entry_threshold" in params
        assert "short_exit_threshold" in params

        # 验证约束
        assert params["long_entry_threshold"] > params["long_exit_threshold"]
        assert params["short_exit_threshold"] > params["short_entry_threshold"]

    def test_threshold_constraints_invalid(self):
        """测试阈值约束检查"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna_joint import (
                sample_threshold_params,
            )
        except ImportError:
            pytest.skip("Cannot import ts_sr_reversal_optuna_joint due to dependencies")

        trial = MagicMock(spec=optuna.Trial)
        # 设置无效的值：long_entry <= long_exit
        value_map = {
            "long_entry_threshold": 0.3,
            "long_exit_threshold": 0.4,  # > long_entry (无效)
            "short_entry_threshold": 0.3,
            "short_exit_threshold": 0.7,
        }
        trial.suggest_float = MagicMock(
            side_effect=lambda name, low, high, **kwargs: value_map.get(
                name, (low + high) / 2
            )
        )

        with pytest.raises(optuna.TrialPruned):
            sample_threshold_params(trial)


class TestJointOptimization:
    """测试联合优化逻辑"""

    def test_model_and_threshold_params_both_sampled(self):
        """测试同时采样模型参数和阈值参数"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna_joint import (
                sample_model_params,
                sample_threshold_params,
            )
        except ImportError:
            pytest.skip("Cannot import ts_sr_reversal_optuna_joint due to dependencies")

        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_int = MagicMock(return_value=6)
        value_map = {
            "learning_rate": 0.05,
            "subsample": 0.8,
            "long_entry_threshold": 0.6,
            "long_exit_threshold": 0.3,
            "short_entry_threshold": 0.3,
            "short_exit_threshold": 0.7,
        }
        trial.suggest_float = MagicMock(
            side_effect=lambda name, low, high, **kwargs: value_map.get(
                name, (low + high) / 2
            )
        )

        model_params = sample_model_params(trial, model_type="xgboost")
        threshold_params = sample_threshold_params(trial)

        assert len(model_params) > 0
        assert len(threshold_params) > 0
        assert "max_depth" in model_params
        assert "long_entry_threshold" in threshold_params
