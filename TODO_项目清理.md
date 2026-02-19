# 项目清理 TODO

> 生成时间: 2026-02-18
> **完成时间: 2026-02-17**
> 当前活跃策略: BPC / ME / FER / LV (archetype 架构)
> 已废弃策略: sr_reversal / sr_breakout / compression_breakout / trend_following (tree 模型策略)

---

## 一、可安全删除的测试文件 ✅ 全部完成

### 1.1 空壳 / 依赖已删除
- [x] `tests/test_pipeline.py` — 已删除
- [x] `tests/sample_module.py` — 已删除
- [x] `tests/unit/test_task_spec_evidence_quantiles_default.py` — 已删除

### 1.2 旧策略专属测试（sr_reversal / sr_breakout）
- [x] `tests/test_sr_reversal_label_full_scan.py` — 已删除
- [x] `tests/test_sr_reversal_rr_consistency.py` — 已删除
- [x] `tests/test_sr_reversal_rule_optimization.py` — 已删除
- [x] `tests/test_ts_sr_reversal_optuna.py` — 已删除
- [x] `tests/test_ts_sr_reversal_optuna_joint.py` — 已删除
- [x] `tests/integration/test_sr_reversal_backtest_autosignal.py` — 已删除
- [x] `tests/integration/test_sr_reversal_backtest_sr_fuse.py` — 已删除
- [x] `tests/integration/test_ts_sr_reversal_optuna_integration.py` — 已删除
- [x] `tests/strategies/test_sr_breakout_label_autosignal.py` — 已删除
- [x] `tests/strategies/test_sr_breakout_label_multisymbol.py` — 已删除
- [x] `tests/strategies/test_sr_breakout_multisymbol_baseline.py` — 已删除

### 1.3 Docker 验证脚本
- [x] `tests/test_alphalens.py` — 已删除

### 1.4 空目录
- [x] `tests/backtesting/` — 已删除

---

## 二、src/ 死代码 ✅ 全部完成

### 2.1 rule_based_strategies/ 整个目录
- [x] `src/rule_based_strategies/` — 整目录已删除 (含 CLI import 清理)

### 2.2 diagnostics/ sr_reversal 相关 (5个)
- [x] 全部已删除 (含 analyze_dtw_and_volatility, analyze_ml_volatility_model 等诊断文件)

### 2.3 旧策略回测 (4个)
- [x] 全部已删除 (sr_reversal/sr_breakout/compression_breakout/trend_following backtest)

### 2.4 旧策略优化 (2个)
- [x] 全部已删除 (ts_sr_reversal_optuna / joint)

### 2.5 旧策略 backtesting 框架 (2个)
- [x] 全部已删除 (base_backtest / vectorbt_backtest)

---

## 三、tree_strategies 整体清理 ✅ 全部完成

### 3.1 config/strategies/tree_strategies/
- [x] 整个 `config/strategies/tree_strategies/` 目录 — 已删除 (162 文件)

### 3.2 旧策略语义配置 (4个 yaml)
- [x] 全部已删除

### 3.3 旧策略标签文件
- [x] 4 个旧标签文件已删除
- [x] `_ensure_atr` 函数迁移到 `label_utils.py`，6 个活跃标签 import 已更新
- [x] 关联诊断文件 + 测试文件 + CLI 命令同步清理

---

## 四、mlbot CLI 命令清理 ✅ 全部完成

### 4.1 旧策略专属 CLI 命令
- [x] 全部已删除 (sr-reversal-long/short, diagnose, optimize 等 7 个命令)
- [x] cross_section 命令组也已删除 (~1188 行)

### 4.2 通用命令中的旧策略 default 值
- [x] 所有 default 已从 `sr_reversal_rr_reg_long` 更新为 `bpc`
- [x] 多策略 default 已更新为 `bpc,me,fer,lv`

---

## 五、tests/ 目录重组 ✅ 全部完成

### 5.1 非测试脚本移出 tests/
- [x] 10 个非测试脚本已移到 `scripts/diagnostics/`

### 5.2 .md 文档移出 tests/
- [x] 24 个 .md 文件已移到 `docs/tests/`

### 5.3 根目录散落 → 迁移到子目录
- [x] 26 个根目录测试文件已迁移到 integration/smoke/features/unit/live_data_stream 子目录
- [x] tests/ 根目录零散落测试文件

### 5.4 碎片化合并建议
> 暂不执行 — 工作量大、风险中等，留待后续优化

---

## 六、工程债务 ✅ 全部完成

### 6.1 sys.path hack
- [x] 275 个文件的 `sys.path.insert` hack 已清理
- [x] `setup.py` 已修复 (`package_dir` 映射)
- [x] `src/__init__.py` 已创建
- [x] conftest.py 负责 PROJECT_ROOT 注入，测试无需 sys.path hack

### 6.2 __pycache__ 清理
- [x] 53 个 __pycache__ 目录已清理（整个项目范围）

---

## 优先级建议

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | 删除空壳 (test_pipeline/sample_module/test_task_spec) | ✅ 完成 |
| P1 | 删除 tree_strategies 全套 (config + src + tests + CLI) | ✅ 完成 |
| P2 | 迁移根目录散落测试到子目录 | ✅ 完成 |
| P3 | 移出非测试脚本和 .md 文件 | ✅ 完成 |
| P4 | CLI default 参数更新 | ✅ 完成 |
| P5 | sys.path hack 清理 + __pycache__ | ✅ 完成 |
| P5b | 碎片化合并 (VPIN/WPT/normalization) | ⏳ 暂缓 |
