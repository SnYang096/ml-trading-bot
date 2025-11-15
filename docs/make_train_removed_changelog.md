# make train 移除变更日志

## 📋 变更概述

**`make train` 已被移除**，统一使用 `make rolling` 进行所有训练。

**变更日期**: 2025-01-15

**原因**:
- 滚动训练提供更好的评估（通过扩展窗口训练和多个模型检查点）
- 可以观察模型在不同时间段的性能变化
- 更接近真实交易场景
- 减少代码重复和维护成本

---

## 🔄 变更内容

### 1. Makefile

**删除**:
- `train` 目标（第 365-400 行）
- `train-quantile` 相关说明

**更新**:
- Help 信息中移除了 `make train` 相关说明
- 添加注释说明使用 `make rolling` 替代

### 2. train.py

**标记为 deprecated**:
- `main()` 函数添加 deprecation warning
- 保留工具函数（`_compute_direction_threshold`, `_resample_ohlcv` 等）供其他模块使用

### 3. 文档更新

**更新的文件**:
- `README.md` - 更新训练流程说明
- `scripts/training/README.md` - 添加重要变更说明

---

## 📝 迁移指南

### 原来的 make train

```bash
# 旧方式（已移除）
make train SYMBOLS=BTCUSDT \
  START_DATE=2024-11-01 END_DATE=2024-12-31 \
  TRAIN_FEATURE_TYPE=comprehensive
```

### 新的 make rolling

```bash
# 新方式：滚动训练（推荐）
make rolling SYMBOLS=BTCUSDT \
  ROLLING_START=2024-11 ROLLING_END=2024-12 \
  INITIAL_TRAIN_MONTHS=6 \
  ROLLING_FEATURE_TYPE=comprehensive

# 单个月训练（相当于原来的 make train）
make rolling SYMBOLS=BTCUSDT \
  ROLLING_START=2024-11 ROLLING_END=2024-11 \
  INITIAL_TRAIN_MONTHS=1
```

---

## ⚠️ 向后兼容性

### 保留的功能

1. **train.py 工具函数**:
   - `_compute_direction_threshold()` - 仍被 `rolling.py` 使用
   - `_resample_ohlcv()` - 仍被其他模块使用
   - `_collect_files()` - 仍被其他模块使用

2. **vectorbot.py Legacy 格式支持**:
   - 仍然支持从 `.pkl` 文件加载旧的 `MLTradingStrategy` 对象
   - 用于向后兼容旧的模型文件

### 不兼容的变更

1. **Makefile**:
   - `make train` 命令不再可用
   - 必须使用 `make rolling` 替代

2. **输出格式**:
   - 旧格式：`models/trained_model_*.pkl`（包含整个 MLTradingStrategy）
   - 新格式：`results/rolling_*/latest/*.pkl`（QuantTradingModel 管道）

---

## 🔍 依赖关系检查

### 仍使用 train.py 的模块

1. **rolling.py**:
   - 导入 `_compute_direction_threshold` ✅ 保留

2. **auto_workflow.py**:
   - 导入 `train_module` ⚠️ 需要检查是否仍在使用

3. **其他脚本**:
   - `tune_q50_params.py` - 使用工具函数 ✅ 保留
   - `safe_multi_asset_preprocessing.py` - 使用工具函数 ✅ 保留

### 建议

- 如果 `auto_workflow.py` 仍在使用 `train_module`，需要迁移到 `rolling_module`
- 其他工具函数可以保留，因为它们被多个模块使用

---

## 📊 影响评估

### 低风险

- ✅ 工具函数保留，不影响其他模块
- ✅ `vectorbot.py` 仍支持 legacy 格式加载
- ✅ 文档已更新

### 需要注意

- ⚠️ 如果有脚本直接调用 `train.py main()`，需要迁移到 `rolling.py`
- ⚠️ 如果有 CI/CD 流程使用 `make train`，需要更新为 `make rolling`

---

## ✅ 验证清单

- [x] Makefile 中 `train` 目标已删除
- [x] `train.py` 标记为 deprecated
- [x] 文档已更新
- [x] 工具函数保留
- [x] `vectorbot.py` 仍支持 legacy 格式
- [ ] 检查 `auto_workflow.py` 是否需要更新
- [ ] 检查是否有 CI/CD 流程需要更新

---

## 🎯 下一步

1. **测试**:
   - 验证 `make rolling` 在单个月训练时工作正常
   - 验证 `vectorbot.py` 仍能加载旧的 `.pkl` 文件

2. **清理**（可选）:
   - 如果确认不再需要，可以考虑将 `train.py` 中的工具函数迁移到独立的工具模块
   - 但这不是必须的，保留它们也不会有问题

3. **监控**:
   - 观察是否有用户或脚本仍在使用 `make train`
   - 如果有，提供迁移指导

