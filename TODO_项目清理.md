# 项目清理 TODO

> 生成时间: 2026-02-18
> 当前活跃策略: BPC / ME / FER / LV (archetype 架构)
> 已废弃策略: sr_reversal / sr_breakout / compression_breakout / trend_following (tree 模型策略)

---

## 一、可安全删除的测试文件

### 1.1 空壳 / 依赖已删除
- [ ] `tests/test_pipeline.py` — 全部代码已注释（MultiTimeframePipeline 已删）
- [ ] `tests/sample_module.py` — fake_label/fake_trainer helper，无引用
- [ ] `tests/unit/test_task_spec_evidence_quantiles_default.py` — 依赖已删除的 `config/tasks/`

### 1.2 旧策略专属测试（sr_reversal / sr_breakout）
- [ ] `tests/test_sr_reversal_label_full_scan.py`
- [ ] `tests/test_sr_reversal_rr_consistency.py`
- [ ] `tests/test_sr_reversal_rule_optimization.py`
- [ ] `tests/test_ts_sr_reversal_optuna.py`
- [ ] `tests/test_ts_sr_reversal_optuna_joint.py`
- [ ] `tests/integration/test_sr_reversal_backtest_autosignal.py`
- [ ] `tests/integration/test_sr_reversal_backtest_sr_fuse.py`
- [ ] `tests/integration/test_ts_sr_reversal_optuna_integration.py`
- [ ] `tests/strategies/test_sr_breakout_label_autosignal.py`
- [ ] `tests/strategies/test_sr_breakout_label_multisymbol.py`
- [ ] `tests/strategies/test_sr_breakout_multisymbol_baseline.py`

### 1.3 Docker 验证脚本
- [ ] `tests/test_alphalens.py` — Docker 环境 alphalens 安装验证

### 1.4 空目录
- [ ] `tests/backtesting/` — 完全空目录

---

## 二、src/ 死代码

### 2.1 rule_based_strategies/ 整个目录
- [ ] `src/rule_based_strategies/sr_reversal_rule_strategy.py`
- [ ] `src/rule_based_strategies/bt_sr_fake_breakout/` 子目录
- [ ] `src/rule_based_strategies/intraday_sniper/` 子目录
> 注: cli/main.py L8203 有一处 import 引用，删除时需同步清理

### 2.2 diagnostics/ sr_reversal 相关 (5个)
- [ ] `src/time_series_model/diagnostics/sr_reversal_ml_parameter_sweep.py`
- [ ] `src/time_series_model/diagnostics/sr_reversal_model_comparison.py`
- [ ] `src/time_series_model/diagnostics/sr_reversal_model_diagnosis.py`
- [ ] `src/time_series_model/diagnostics/sr_reversal_rule_baseline.py`
- [ ] `src/time_series_model/diagnostics/sr_reversal_rule_optimization.py`

### 2.3 旧策略回测 (4个)
- [ ] `src/time_series_model/backtesting/sr_reversal_backtest.py`
- [ ] `src/time_series_model/backtesting/sr_breakout_backtest.py`
- [ ] `src/time_series_model/backtesting/compression_breakout_backtest.py`
- [ ] `src/time_series_model/backtesting/trend_following_backtest.py`

### 2.4 旧策略优化 (2个)
- [ ] `src/time_series_model/optimization/ts_sr_reversal_optuna.py`
- [ ] `src/time_series_model/optimization/ts_sr_reversal_optuna_joint.py`

### 2.5 旧策略 backtesting 框架 (2个)
- [ ] `src/time_series_model/strategies/backtesting/base_backtest.py`
- [ ] `src/time_series_model/strategies/backtesting/vectorbt_backtest.py`

---

## 三、tree_strategies 整体清理

### 3.1 config/strategies/tree_strategies/ (162 个文件, 852K)
- [ ] 整个 `config/strategies/tree_strategies/` 目录
> 包含 sr_reversal_rr_reg_long / sr_breakout / compression_breakout / trend_following
> 及 strategies_exported/tree_best/ 下的导出快照
> 现状: src/ 零引用, live/ 零引用, train_all 零引用, 仅 CLI default 参数还在引用

### 3.2 旧策略语义配置 (4个 yaml)
- [ ] `config/feature_groups_sr_reversal_semantic.yaml`
- [ ] `config/feature_groups_sr_breakout_semantic.yaml`
- [ ] `config/feature_groups_compression_breakout_semantic.yaml`
- [ ] `config/feature_groups_trend_following_semantic.yaml`

### 3.3 旧策略标签文件（src/ 中仍存在，但仅被旧策略使用）
> 这些需谨慎: 虽然 4 archetype 不用，但 label 函数是通用基础设施，部分仍可能被引用
- `src/time_series_model/strategies/labels/sr_reversal_label.py` — 仅被旧诊断+测试引用
- `src/time_series_model/strategies/labels/sr_breakout_label.py` — 仅被旧诊断+测试引用
- `src/time_series_model/strategies/labels/compression_breakout_label.py` — 仅被旧测试引用
- `src/time_series_model/strategies/labels/trend_following_label.py` — 仅被旧测试引用

---

## 四、mlbot CLI 命令清理

### 4.1 旧策略专属 CLI 命令 (可删除)
- [ ] `mlbot train sr-reversal-long` (L6277) — 专用于旧 sr_reversal 训练
- [ ] `mlbot train sr-reversal-short` (L6326) — 专用于旧 sr_reversal 训练
- [ ] `mlbot diagnose sr-reversal-model-comparison` (L8366) — 旧策略诊断
- [ ] `mlbot diagnose rule-baseline` (L8135) — 调用 sr_reversal_rule_baseline
- [ ] `mlbot optimize ml-param-sweep` (L9519) — 调用 sr_reversal_ml_parameter_sweep
- [ ] `mlbot optimize rule-plateau-charts` (L9489) — 调用 sr_reversal_rule_optimization
- [ ] `mlbot diagnose test-vpin-thresholds` (L8210) — 调用 diagnostics/test_vpin_thresholds.py (旧)

### 4.2 通用命令中的旧策略 default 值 (需修改)
- [ ] L1556: `help="Strategy name"` default `sr_reversal_rr_reg_long` → 改为 `bpc`
- [ ] L1743: default `sr_reversal_rr_reg_long,sr_breakout,...` → 改为 `bpc,me,fer,lv`
- [ ] L6298/6371/6432/6802/6947/8139/8214/8285/8370/9078/9417/9523: 多处 default 为 `config/strategies/sr_reversal_long` → 改为 `config/strategies/bpc`
- [ ] L8773: default `sr_reversal_rr_reg_long,...` → `bpc,me,fer,lv`

> cli/main.py 共 11274 行, 其中 33 处引用旧策略名

---

## 五、tests/ 目录重组

### 5.1 非测试脚本移出 tests/
- [ ] `tests/check_feature_quality.py` → `scripts/diagnostics/`
- [ ] `tests/check_feature_test_coverage.py` → `scripts/diagnostics/`
- [ ] `tests/validate_test_structure.py` → `scripts/diagnostics/`
- [ ] `tests/integration/check_all_feature_dependencies.py` → `scripts/diagnostics/`
- [ ] `tests/integration/check_constant_feature_dependencies.py` → `scripts/diagnostics/`
- [ ] `tests/integration/check_dtw_nan_reason.py` → `scripts/diagnostics/`
- [ ] `tests/integration/check_feature_dependencies.py` → `scripts/diagnostics/`
- [ ] `tests/integration/debug_ticks_config.py` → `scripts/diagnostics/`
- [ ] `tests/integration/diagnose_long_only_predictions.py` → `scripts/diagnostics/`
- [ ] `tests/integration/diagnose_prediction_concentration.py` → `scripts/diagnostics/`

### 5.2 .md 文档移出 tests/
共 15 个 .md 文件分布在 tests/ 各子目录:
- tests/README_SYS_PATH.md, README_PYTEST_INI.md
- tests/features/INF_NAN_ROOT_CAUSE_SUMMARY.md, standard.md
- tests/live_data_stream/README.md, CI_CD_CONFIG.md, OPTIMIZE_TEST_SPEED.md
- tests/event_driven/README.md
- tests/integration/ 下 7 个诊断报告 .md
> 建议: 移到 docs/tests/ 或删除

### 5.3 根目录散落 → 迁移到子目录
| 文件 | 建议归属 |
|---|---|
| test_archetype_e2e.py, test_archetype_cli_integration.py | → integration/ |
| test_live_startup_integration.py, test_live_vs_batch_features.py | → integration/ 或 live_data_stream/ |
| test_binance_testnet_connection.py | → smoke/ |
| test_binance_tick_aggregator.py, test_tick_storage.py | → features/ 或 integration/ |
| test_bpc_breakout_direction.py, test_dtw_individual.py, test_evt_safety_fuse.py | → features/ |
| test_zip_to_parquet.py, test_zip_to_parquet_agg.py | → unit/ |
| test_monthly_cache.py | → unit/ |
| test_feature_config_loading.py, test_feature_loader.py, test_validate_feature_config.py | → unit/ |
| test_strategy_config_loader.py, test_volatility_model_config.py | → unit/ |
| test_system_mode.py | → live_data_stream/ |
| test_compute_lift_for_threshold.py, test_drop_inf_rows.py | → unit/ |
| test_rolling_train.py, test_optuna_imbalanced_data.py | → integration/ |
| test_label_generators.py, test_rr_exit_consistency.py | → unit/ |
| test_project_structure.py | → smoke/ |

### 5.4 碎片化合并建议
| 主题 | 当前文件数 | 建议 |
|---|---|---|
| VPIN | 12 | 合并为 2-3 个 |
| WPT | 7 | 合并为 1-2 个 |
| normalization | 18 | 按阶段合并 |
| trade_clustering | 5 | 合并为 1 个 |

---

## 六、工程债务

### 6.1 sys.path hack
- 94 个文件包含 `sys.path.insert` hack
- 项目已有 `setup.py`，安装为 editable (`pip install -e .`) 后可全部去除

### 6.2 __pycache__ 清理
- tests/ 下有 10 个 __pycache__ 目录

---

## 优先级建议

| 阶段 | 内容 | 风险 |
|---|---|---|
| P0 | 删除空壳 (test_pipeline/sample_module/test_task_spec) | 零 |
| P1 | 删除 tree_strategies 全套 (config + src + tests + CLI) | 低 (确认无引用) |
| P2 | 迁移根目录散落测试到子目录 | 低 |
| P3 | 移出非测试脚本和 .md 文件 | 低 |
| P4 | CLI default 参数更新 | 中 (需测试) |
| P5 | 碎片化合并 + sys.path hack | 中 (工作量大) |
