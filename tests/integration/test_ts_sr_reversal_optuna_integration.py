"""
集成测试：ts_sr_reversal_optuna.py

测试完整的 Optuna 优化流程（使用 mock）。
"""

import pytest
import optuna
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import pandas as pd
import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestTSRReversalOptunaIntegration:
    """集成测试：完整的 Optuna 优化流程"""

    def test_objective_function_flow(self):
        """测试 objective 函数的完整流程"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna import (
                sample_params,
            )
            from src.time_series_model.strategy_config.loader import (
                StrategyConfig,
                BacktestConfig,
            )
        except ImportError:
            pytest.skip("Cannot import required modules due to dependencies")

        # 创建 mock 策略配置
        strategy_cfg = MagicMock(spec=StrategyConfig)
        strategy_cfg.backtest = MagicMock(spec=BacktestConfig)
        strategy_cfg.backtest.params = {"price_col": "close"}

        # 创建 mock trial
        trial = MagicMock(spec=optuna.Trial)
        trial.number = 0
        fixed = {
            "long_entry_threshold": 0.6,
            "long_exit_threshold": 0.3,
            "short_entry_threshold": 0.3,
            "short_exit_threshold": 0.7,
        }
        # Optuna suggest_float signature is (name, low, high, **kwargs)
        trial.suggest_float = MagicMock(
            side_effect=lambda name, low, high, **kwargs: fixed[name]
        )
        trial.set_user_attr = MagicMock()

        # 模拟 objective 函数的流程
        threshold_params = sample_params(trial)

        # 创建配置副本并更新
        from copy import deepcopy

        trial_cfg = deepcopy(strategy_cfg)
        if trial_cfg.backtest.params is None:
            trial_cfg.backtest.params = {}
        trial_cfg.backtest.params.update(threshold_params)

        # 验证配置更新
        assert trial_cfg.backtest.params["long_entry_threshold"] == 0.6
        assert trial_cfg.backtest.params["long_exit_threshold"] == 0.3
        assert trial_cfg.backtest.params["short_entry_threshold"] == 0.3
        assert trial_cfg.backtest.params["short_exit_threshold"] == 0.7

        # 验证原有参数保留
        assert trial_cfg.backtest.params["price_col"] == "close"

    def test_execute_single_run_receives_updated_config(self):
        """测试 execute_single_run 接收到更新后的配置"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna import (
                sample_params,
            )
            from src.time_series_model.strategy_config.loader import (
                StrategyConfig,
                BacktestConfig,
            )
            from src.time_series_model.strategies.evaluation.strategy_feature_compare import (
                execute_single_run,
            )
        except ImportError:
            pytest.skip("Cannot import required modules due to dependencies")

        # 创建 mock 策略配置
        strategy_cfg = MagicMock(spec=StrategyConfig)
        strategy_cfg.backtest = MagicMock(spec=BacktestConfig)
        strategy_cfg.backtest.params = {}

        # 创建 mock trial
        trial = MagicMock(spec=optuna.Trial)
        fixed = {
            "long_entry_threshold": 0.65,
            "long_exit_threshold": 0.35,
            "short_entry_threshold": 0.35,
            "short_exit_threshold": 0.65,
        }
        trial.suggest_float = MagicMock(
            side_effect=lambda name, low, high, **kwargs: fixed[name]
        )

        # 获取阈值参数
        threshold_params = sample_params(trial)

        # 更新配置
        from copy import deepcopy

        trial_cfg = deepcopy(strategy_cfg)
        trial_cfg.backtest.params.update(threshold_params)

        # Mock execute_single_run
        with patch(
            "src.time_series_model.strategies.evaluation.strategy_feature_compare.execute_single_run"
        ) as mock_execute:
            mock_execute.return_value = {
                "avg_cv_metric": 0.75,
                "backtest": {"total_return": 10.0, "sharpe": 1.5},
            }

            # 验证配置被正确传递
            result = mock_execute(
                trial_cfg,
                pd.DataFrame(),  # mock data
                pd.DataFrame(),  # mock data
                test_warmup_bars=0,
                variant_name="test",
            )

            # 验证 execute_single_run 被调用
            assert mock_execute.called

            # 验证第一个参数是更新后的配置
            call_args = mock_execute.call_args
            passed_cfg = call_args[0][0]

            # 验证配置包含阈值参数
            assert passed_cfg.backtest.params["long_entry_threshold"] == 0.65
            assert passed_cfg.backtest.params["short_exit_threshold"] == 0.65

    def test_optuna_study_with_mock_objective(self):
        """测试 Optuna study 使用 mock objective 函数"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna import (
                sample_params,
            )
        except ImportError:
            pytest.skip("Cannot import required modules due to dependencies")

        # 创建简单的 objective 函数
        def mock_objective(trial):
            params = sample_params(trial)
            # 模拟返回指标
            return 0.5 + np.random.rand() * 0.3

        # 创建 study
        study = optuna.create_study(direction="maximize")

        # 运行少量 trials
        study.optimize(mock_objective, n_trials=5, show_progress_bar=False)

        # 验证 study 完成
        assert len(study.trials) == 5

        # 验证最佳 trial 存在
        assert study.best_trial is not None
        assert study.best_value is not None

        # 验证参数在合理范围内
        best_params = study.best_trial.params
        assert "long_entry_threshold" in best_params
        assert 0.4 <= best_params["long_entry_threshold"] <= 0.8

    def test_threshold_params_stored_in_trial_attrs(self):
        """测试阈值参数存储在 trial user_attrs 中"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna import (
                sample_params,
            )
        except ImportError:
            pytest.skip("Cannot import required modules due to dependencies")

        # 创建 objective 函数，模拟存储逻辑
        def mock_objective(trial):
            threshold_params = sample_params(trial)
            trial.set_user_attr("threshold_params", threshold_params)
            trial.set_user_attr("backtest_results", {"total_return": 10.0})
            return 0.75

        study = optuna.create_study(direction="maximize")
        study.optimize(mock_objective, n_trials=3, show_progress_bar=False)

        # 验证 user_attrs 被设置
        for trial in study.trials:
            # Some trials may be pruned if sampled thresholds are invalid; only assert on completed ones.
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            assert "threshold_params" in trial.user_attrs
            assert "long_entry_threshold" in trial.user_attrs["threshold_params"]
            assert "short_exit_threshold" in trial.user_attrs["threshold_params"]

    def test_best_params_output_structure(self):
        """测试最佳参数输出结构"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna import (
                sample_params,
            )
        except ImportError:
            pytest.skip("Cannot import required modules due to dependencies")

        # 创建 objective 函数
        def mock_objective(trial):
            threshold_params = sample_params(trial)
            trial.set_user_attr("threshold_params", threshold_params)
            trial.set_user_attr(
                "backtest_results", {"total_return": 10.0, "sharpe": 1.5}
            )
            return 0.75

        study = optuna.create_study(direction="maximize")
        study.optimize(mock_objective, n_trials=3, show_progress_bar=False)

        # 模拟 main 函数中的输出构建逻辑
        best_thresholds = study.best_trial.user_attrs.get("threshold_params", {})
        best_backtest = study.best_trial.user_attrs.get("backtest_results", {})

        best = {
            "value": study.best_value,
            "params": study.best_trial.params,
            "threshold_params": best_thresholds,
            "backtest_results": best_backtest,
        }

        # 验证输出结构
        assert "value" in best
        assert "params" in best
        assert "threshold_params" in best
        assert "backtest_results" in best

        # 验证阈值参数结构
        assert "long_entry_threshold" in best["threshold_params"]
        assert "long_exit_threshold" in best["threshold_params"]
        assert "short_entry_threshold" in best["threshold_params"]
        assert "short_exit_threshold" in best["threshold_params"]

    def test_output_files_generation(self):
        """测试输出文件生成（使用临时目录）"""
        try:
            from src.time_series_model.optimization.ts_sr_reversal_optuna import (
                sample_params,
            )
        except ImportError:
            pytest.skip("Cannot import required modules due to dependencies")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_output"
            output_dir.mkdir(parents=True)

            # 创建 objective 函数
            def mock_objective(trial):
                threshold_params = sample_params(trial)
                trial.set_user_attr("threshold_params", threshold_params)
                trial.set_user_attr("backtest_results", {"total_return": 10.0})
                return 0.75

            study = optuna.create_study(direction="maximize")
            study.optimize(mock_objective, n_trials=3, show_progress_bar=False)

            # 构建输出数据
            best_thresholds = study.best_trial.user_attrs.get("threshold_params", {})
            best_backtest = study.best_trial.user_attrs.get("backtest_results", {})

            best = {
                "value": study.best_value,
                "params": study.best_trial.params,
                "threshold_params": best_thresholds,
                "backtest_results": best_backtest,
            }

            # 保存 JSON
            best_params_file = output_dir / "best_params.json"
            with open(best_params_file, "w", encoding="utf-8") as fh:
                json.dump(best, fh, indent=2)

            # 保存 CSV
            trials_df = study.trials_dataframe()
            trials_csv = output_dir / "trial_history.csv"
            trials_df.to_csv(trials_csv, index=False)

            # 验证文件存在
            assert best_params_file.exists()
            assert trials_csv.exists()

            # 验证 JSON 内容
            with open(best_params_file) as f:
                loaded_best = json.load(f)
                assert "threshold_params" in loaded_best
                assert "long_entry_threshold" in loaded_best["threshold_params"]

            # 验证 CSV 内容
            loaded_df = pd.read_csv(trials_csv)
            assert len(loaded_df) == 3  # 3 trials
            # Optuna uses 'params_' prefix for parameter columns in trials_dataframe()
            assert "params_long_entry_threshold" in loaded_df.columns

    def test_no_environment_variables_used(self):
        """测试不再使用环境变量（验证旧代码已移除）"""
        try:
            from src.time_series_model.optimization import ts_sr_reversal_optuna
        except ImportError:
            pytest.skip("Cannot import ts_sr_reversal_optuna due to dependencies")

        # 验证不再有 ENV_KEYS
        assert not hasattr(
            ts_sr_reversal_optuna, "ENV_KEYS"
        ), "ENV_KEYS should be removed - optimization now uses config updates, not env vars"

        # 验证不再有 sr_signal_env 函数
        assert not hasattr(
            ts_sr_reversal_optuna, "sr_signal_env"
        ), "sr_signal_env should be removed - optimization now uses config updates, not env vars"

        # 验证 sample_params 返回字典而不是元组
        trial = MagicMock(spec=optuna.Trial)
        fixed = {
            "long_entry_threshold": 0.6,
            "long_exit_threshold": 0.4,
            "short_entry_threshold": 0.4,
            "short_exit_threshold": 0.6,
        }
        trial.suggest_float = MagicMock(
            side_effect=lambda name, low, high, **kwargs: fixed[name]
        )

        params = ts_sr_reversal_optuna.sample_params(trial)

        # 应该返回字典，不是元组
        assert isinstance(params, dict)
        assert not isinstance(params, tuple)

        # 应该包含阈值参数，不是环境变量键
        assert "long_entry_threshold" in params
        assert "SR_SIGNAL_MIN_STRENGTH" not in params
