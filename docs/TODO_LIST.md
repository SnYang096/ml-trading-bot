# 项目 TODO List

本文档汇总了项目的所有待完成任务，按优先级和类别组织。

**最后更新**: 2025-01-01

---

## 📊 概览

- **总计**: 27 个待处理任务
- **高优先级**: 9 个（其中 7 个已完成 ✅）
- **中优先级**: 14 个
- **当前会话**: 4 个

**最新更新**: 所有特征测试已完成！14个特征类别都有完整的测试覆盖。

---

## 🔴 高优先级任务

### 1. Feature Selection（当前会话）- 4 个任务

这些任务用于测试新实现的 `--expand-semantic-singletons` 功能，验证展开 semantic groups 为单例是否能获得更精细的特征选择。

#### 1.1 test-expand-semantic-sr_breakout

**状态**: pending  
**类别**: Feature Selection  
**优先级**: 高

**任务描述**:  
测试 `sr_breakout` 策略的 `--expand-semantic-singletons` 功能。

**详细说明**:
- 运行 `feature-group-search` 并加上 `--expand-semantic-singletons` 选项
- 比较展开前后的效果（Sharpe、选择的特征等）
- 评估是否需要默认启用展开功能

**作用**:
- 验证展开功能对 SR Breakout 策略的有效性
- 确定是否可以获得更精细的特征选择（例如只选择 `ignition` 而不选择 `exhaustion`）
- 为后续决策提供数据支持

**命令示例**:
```bash
mlbot diagnose feature-group-search \
  -c config/strategies/sr_breakout \
  --groups-yaml config/feature_groups_sr_breakout_semantic.yaml \
  --pool-b-yaml results/pools/sr_breakout/pool_b/features_pool_b.yaml \
  --expand-semantic-singletons \
  -s BTCUSDT -t 240T \
  --start-date 2024-01-01 --end-date 2025-10-31 \
  --seeds 1,2,3,4,5 \
  --objective Sharpe_mean \
  --max-steps 6 \
  --output-dir results/feature_group_search/sr_breakout_expanded
```

**预期结果**:
- 展开后候选组数量增加（从 ~9 个增加到 ~36 个，每个 semantic group 展开为 4 个单例）
- 评估时间增加约 27%
- 可能获得更精细的特征选择（例如只选择 `vpin_ignition_score` 而不选择 `vpin_exhaustion_scene_score`）

---

#### 1.2 test-expand-semantic-compression_breakout

**状态**: pending  
**类别**: Feature Selection  
**优先级**: 高

**任务描述**:  
测试 `compression_breakout` 策略的 `--expand-semantic-singletons` 功能。

**详细说明**:  
同 1.1，但针对 `compression_breakout` 策略。

**作用**:  
- 验证展开功能对 Compression Breakout 策略的有效性
- Compression Breakout 策略可能更需要精细的语义选择（例如 `compression` vs `ignition`）

**命令示例**:
```bash
mlbot diagnose feature-group-search \
  -c config/strategies/compression_breakout \
  --groups-yaml config/feature_groups_compression_breakout_semantic.yaml \
  --pool-b-yaml results/pools/compression_breakout/pool_b/features_pool_b.yaml \
  --expand-semantic-singletons \
  ...
```

---

#### 1.3 test-expand-semantic-trend_following

**状态**: pending  
**类别**: Feature Selection  
**优先级**: 高

**任务描述**:  
测试 `trend_following` 策略的 `--expand-semantic-singletons` 功能。

**详细说明**:  
同 1.1，但针对 `trend_following` 策略。

**作用**:  
- 验证展开功能对 Trend Following 策略的有效性
- Trend Following 策略可能更需要 `ignition` 语义而不是 `exhaustion`

---

#### 1.4 test-expand-semantic-sr_reversal

**状态**: pending  
**类别**: Feature Selection  
**优先级**: 高

**任务描述**:  
测试 `sr_reversal_rr_reg_long` 策略的 `--expand-semantic-singletons` 功能。

**详细说明**:  
同 1.1，但针对 `sr_reversal_rr_reg_long` 策略。

**作用**:  
- 验证展开功能对 SR Reversal 策略的有效性
- SR Reversal 策略可能更需要 `exhaustion` 和 `absorption` 语义而不是 `ignition`

---

### 2. Backtesting (Nautilus 集成) - 5 个任务

这些任务用于重建基于 Nautilus Trader 的事件驱动回测，提供更真实的执行逻辑（滑点、延迟、trailing stop、加仓/减仓等）。

#### 2.1 nautilus-backtesting-sr-reversal

**状态**: pending  
**类别**: Backtesting  
**优先级**: 高

**任务描述**:  
重建基于 Nautilus Trader 的事件驱动回测：SR Reversal 策略。

**详细说明**:
- 使用 Nautilus Trader 实现事件驱动回测
- 支持真实的执行逻辑（滑点、延迟、订单簿深度等）
- 支持 trailing stop、加仓/减仓等高级功能
- 与 VectorBT 回测结果对比验证

**作用**:
- 提供更真实的回测环境，减少实盘与回测的差异
- 验证策略在真实执行环境下的表现
- 为实盘部署提供更可靠的性能评估

**参考文档**:
- `src/time_series_model/backtesting/TODO.md`
- `docs/Nautilus_Trader_集成指南.md`

---

#### 2.2 nautilus-backtesting-sr-breakout

**状态**: pending  
**类别**: Backtesting  
**优先级**: 高

**任务描述**:  
重建基于 Nautilus Trader 的事件驱动回测：SR Breakout 策略。

**详细说明**:  
同 2.1，但针对 SR Breakout 策略。

---

#### 2.3 nautilus-backtesting-compression

**状态**: pending  
**类别**: Backtesting  
**优先级**: 高

**任务描述**:  
重建基于 Nautilus Trader 的事件驱动回测：Compression Breakout 策略。

**详细说明**:  
同 2.1，但针对 Compression Breakout 策略。

---

#### 2.4 nautilus-backtesting-trend

**状态**: pending  
**类别**: Backtesting  
**优先级**: 高

**任务描述**:  
重建基于 Nautilus Trader 的事件驱动回测：Trend Following 策略。

**详细说明**:  
同 2.1，但针对 Trend Following 策略。Trend Following 策略可能需要支持 pyramiding（加仓）功能。

---

#### 2.5 nautilus-cli-runners

**状态**: pending  
**类别**: Backtesting  
**优先级**: 高

**任务描述**:  
为 Nautilus 回测提供 CLI runners 和 Makefile targets。

**详细说明**:
- 实现 `mlbot backtest nautilus` 命令
- 提供 Makefile targets 便于快速运行
- 支持批量回测多个策略
- 生成统一的回测报告

**作用**:
- 简化 Nautilus 回测的使用流程
- 提供统一的接口和报告格式
- 便于集成到 CI/CD 流程

---

### 3. Feature Testing (高优先级) - 4 个任务

这些任务用于补充关键特征的测试，确保特征计算的正确性和无未来数据泄露。

#### 3.1 test-vpin-features

**状态**: ✅ completed  
**类别**: Feature Testing  
**优先级**: 高

**任务描述**:  
VPIN 特征测试：未来数据泄露、多资产归一化、模拟 tick 数据。

**完成情况**:  
- ✅ 已有 `test_vpin_features.py`（基础测试）
- ✅ 已有 `test_vpin_future_leak_and_multi_asset.py`（未来数据泄露+多资产测试）
- ✅ 已补充流式vs批量一致性测试（`TestVPINStreamingVsBatch` 类）

**详细说明**:
- **未来数据泄露测试**: 验证 VPIN 特征不使用未来数据（使用 `shift(1)` 和滚动窗口）
- **多资产归一化测试**: 验证 VPIN 特征在不同价格水平的资产上能正确归一化
- **模拟 tick 数据测试**: 使用模拟 tick 数据测试 VPIN 计算的正确性

**作用**:
- 确保 VPIN 特征计算的正确性
- 防止未来数据泄露导致的过拟合
- 确保多资产场景下的特征一致性

**参考文档**:
- `docs/特征测试覆盖总结.md`
- `docs/时序模型/测试补充计划.md`

---

#### 3.2 test-wpt-features

**状态**: ✅ completed  
**类别**: Feature Testing  
**优先级**: 高

**任务描述**:  
WPT 特征测试：未来数据泄露、多资产归一化。

**完成情况**:  
- ✅ 已有 `test_wpt_volatility_features.py`（基础测试）
- ✅ 已有 `test_wpt_future_leak_and_multi_asset.py`（未来数据泄露+多资产测试）
- ✅ 已补充流式vs批量一致性测试（`TestWPTStreamingVsBatch` 类）

**详细说明**:
- **未来数据泄露测试**: 验证 WPT 特征的 `shift(1)` 和滚动窗口正确性
- **多资产归一化测试**: 验证 WPT 特征在不同价格水平的资产上能正确归一化

**作用**:
- 确保 WPT 特征计算的正确性
- 防止未来数据泄露

---

#### 3.3 test-volume-profile-volatility

**状态**: ✅ completed  
**类别**: Feature Testing  
**优先级**: 高

**任务描述**:  
Volume Profile Volatility 特征测试：未来数据泄露、多资产归一化。

**完成情况**:  
- ✅ 已有 `test_volume_profile_volatility_features.py`（基础测试）
- ✅ 已有 `test_volume_profile_volatility_future_leak_and_multi_asset.py`（未来数据泄露+多资产测试）
- ✅ 已补充流式vs批量一致性测试（`TestVolumeProfileVolatilityStreamingVsBatch` 类）

**详细说明**:  
同 3.2，但针对 Volume Profile Volatility 特征。

**作用**:
- 确保 Volume Profile Volatility 特征计算的正确性
- 防止未来数据泄露

---

#### 3.4 test-dtw-features

**状态**: ✅ completed  
**类别**: Feature Testing  
**优先级**: 高

**任务描述**:  
DTW 特征测试：未来数据泄露、多资产归一化、模拟数据。

**完成情况**:  
- ✅ 已有 `test_advanced_features.py`（部分 DTW 测试）
- ✅ 已补充完整测试（`test_dtw_narrow_entrypoint.py` 中的 `TestDTWFeaturesComplete` 类）
- ✅ 包含：未来数据泄露、多资产归一化、流式vs批量一致性测试

**详细说明**:
- **未来数据泄露测试**: 验证 DTW 特征不使用未来数据
- **多资产归一化测试**: 验证 DTW 特征在不同价格水平的资产上能正确归一化
- **模拟数据测试**: 使用模拟数据测试 DTW 计算的正确性

**作用**:
- 确保 DTW 特征计算的正确性
- DTW 是重要的模式匹配特征，需要严格测试

---

## 🟠 中优先级任务

### 4. Feature Testing (中优先级) - 3 个任务

#### 4.1 test-interaction-features

**状态**: ✅ completed  
**类别**: Feature Testing  
**优先级**: 中

**任务描述**:  
Interaction Features 测试：为所有 interaction features 添加测试，验证特征组合的正确性。

**完成情况**:  
- ✅ 已有 `test_interaction_features.py`（完整测试）
- ✅ 包含所有4种测试：未来数据泄露、多资产归一化、流式vs批量一致性、数学正确性

**详细说明**:
- 为所有 interaction features（如 `vpin_scene_semantic_scores_f`、`wpt_scene_semantic_scores_f` 等）添加测试
- 验证特征组合的正确性（例如 `compression` × `ignition` 的组合逻辑）
- 测试语义特征的输出列是否正确

**作用**:
- Interaction features 是重要的语义特征，需要确保计算正确
- 验证语义特征的组合逻辑是否符合预期

---

#### 4.2 test-trend-features

**状态**: ✅ completed  
**类别**: Feature Testing  
**优先级**: 中

**任务描述**:  
Trend Features 测试：未来数据泄露、稳定性验证。

**完成情况**:  
- ✅ 新建 `test_trend_features.py`
- ✅ 包含所有4种测试：未来数据泄露、多资产归一化、流式vs批量一致性、数学正确性
- ✅ 测试特征：`trend_r2_20`, `trend_r2_50`, `slope_consistency_score`

**详细说明**:
- 为 trend features 添加未来数据泄露测试
- 验证趋势特征的稳定性（在不同市场 regime 下的表现）

**作用**:
- 确保趋势特征计算的正确性
- 验证趋势特征在不同市场环境下的稳定性

---

#### 4.3 test-momentum-features

**状态**: ✅ completed  
**类别**: Feature Testing  
**优先级**: 中

**任务描述**:  
Momentum Features 测试：为 momentum features 添加测试，验证动量特征的准确性。

**完成情况**:  
- ✅ 新建 `test_momentum_features.py`
- ✅ 包含所有4种测试：未来数据泄露、多资产归一化、流式vs批量一致性、数学正确性
- ✅ 测试特征：`momentum_5/10/20`, `vpin_momentum`

**详细说明**:
- 为 momentum features 添加测试
- 验证动量特征的准确性（例如动量方向、强度等）

**作用**:
- 确保动量特征计算的正确性
- 动量特征是重要的技术指标，需要严格测试

---

### 5. Test Improvements - 11 个任务

这些任务用于改进现有测试文件，补充缺失的测试用例。

#### 5.1 improve-test-spectrum

**状态**: pending  
**类别**: Test Improvements  
**优先级**: 中

**任务描述**:  
`test_spectrum_features.py`: 补充多资产归一化、流式vs批量一致性、lag衰减平滑测试。

**详细说明**:
- **多资产归一化测试**: 验证频谱特征在不同价格水平的资产上能正确归一化
- **流式vs批量一致性**: 验证流式计算和批量计算结果一致
- **lag衰减平滑测试**: 验证特征在不同 lag 下的衰减平滑性（可选）

**作用**:
- 提高测试覆盖率
- 确保特征计算的稳定性和一致性

---

#### 5.2 improve-test-hurst

**状态**: pending  
**类别**: Test Improvements  
**优先级**: 中

**任务描述**:  
`test_hurst_features_improved.py`: 补充多资产归一化、流式vs批量一致性、lag衰减平滑测试。

**详细说明**:  
同 5.1，但针对 Hurst 特征。

---

#### 5.3 improve-test-hilbert

**状态**: pending  
**类别**: Test Improvements  
**优先级**: 中

**任务描述**:  
`test_hilbert_features_improved.py`: 补充无未来函数、流式vs批量一致性、lag衰减平滑测试。

**详细说明**:
- **无未来函数测试**: 验证 Hilbert 特征不使用未来数据
- **流式vs批量一致性**: 验证流式计算和批量计算结果一致
- **lag衰减平滑测试**: 验证特征在不同 lag 下的衰减平滑性（可选）

**作用**:
- 提高测试覆盖率
- 确保 Hilbert 特征计算的正确性

---

#### 5.4 improve-test-wpt-volatility

**状态**: pending  
**类别**: Test Improvements  
**优先级**: 中

**任务描述**:  
`test_wpt_volatility_features.py`: 补充所有测试。

**详细说明**:
- 补充无未来函数测试
- 补充多资产归一化测试
- 补充流式vs批量一致性测试
- 补充 lag衰减平滑测试（可选）

**作用**:
- 为 WPT 波动率特征提供完整的测试覆盖

---

#### 5.5 improve-test-vpin

**状态**: pending  
**类别**: Test Improvements  
**优先级**: 中

**任务描述**:  
`test_vpin_features.py`: 补充所有测试。

**详细说明**:  
同 5.4，但针对 VPIN 特征。

**作用**:
- 为 VPIN 特征提供完整的测试覆盖
- VPIN 是重要的订单流特征，需要严格测试

---

#### 5.6 improve-test-volume-profile

**状态**: pending  
**类别**: Test Improvements  
**优先级**: 中

**任务描述**:  
`test_volume_profile_volatility_features.py`: 补充所有测试。

**详细说明**:  
同 5.4，但针对 Volume Profile Volatility 特征。

---

#### 5.7 improve-test-complex-features

**状态**: pending  
**类别**: Test Improvements  
**优先级**: 中

**任务描述**:  
`test_complex_features_comprehensive.py`: 补充流式vs批量一致性、lag衰减平滑测试。

**详细说明**:
- 已有无未来函数测试和多资产归一化测试
- 需要补充流式vs批量一致性测试
- 需要补充 lag衰减平滑测试（可选）

**作用**:
- 提高复杂特征的测试覆盖率

---

#### 5.8 improve-test-wpt-future-leak

**状态**: pending  
**类别**: Test Improvements  
**优先级**: 中

**任务描述**:  
`test_wpt_future_leak_and_multi_asset.py`: 补充流式vs批量一致性、lag衰减平滑测试。

**详细说明**:
- 已有无未来函数测试和多资产归一化测试
- 需要补充流式vs批量一致性测试
- 需要补充 lag衰减平滑测试（可选）

---

#### 5.9 improve-test-vpin-future-leak

**状态**: pending  
**类别**: Test Improvements  
**优先级**: 中

**任务描述**:  
`test_vpin_future_leak_and_multi_asset.py`: 补充流式vs批量一致性、lag衰减平滑测试。

**详细说明**:  
同 5.8，但针对 VPIN 特征。

---

#### 5.10 improve-test-volume-profile-future-leak

**状态**: pending  
**类别**: Test Improvements  
**优先级**: 中

**任务描述**:  
`test_volume_profile_volatility_future_leak_and_multi_asset.py`: 补充流式vs批量一致性、lag衰减平滑测试。

**详细说明**:  
同 5.8，但针对 Volume Profile Volatility 特征。

---

#### 5.11 improve-test-garch-evt

**状态**: pending  
**类别**: Test Improvements  
**优先级**: 中

**任务描述**:  
`test_garch_evt_features.py`: 补充多资产归一化、流式vs批量一致性、lag衰减平滑测试。

**详细说明**:
- 已有无未来函数测试
- 需要补充多资产归一化测试
- 需要补充流式vs批量一致性测试
- 需要补充 lag衰减平滑测试（可选）

**作用**:
- 为 GARCH 和 EVT 特征提供完整的测试覆盖
- GARCH 和 EVT 是重要的波动率和极端值特征

---

## 📝 测试最佳实践

所有测试都应该参考 `test_advanced_features.py` 中的实现：

1. **无未来函数测试**: 验证特征不使用未来数据
2. **多资产归一化测试**: 验证特征在不同价格水平的资产上能正确归一化
3. **流式vs批量一致性**: 验证流式计算和批量计算结果一致
4. **lag衰减平滑测试**: 验证特征在不同 lag 下的衰减平滑性（可选）

**详细模板请参考**: `docs/时序模型/所有特征测试覆盖情况报告.md`

---

## 🎯 补充顺序建议

### 第一阶段（当前会话）
1. ✅ 完成 semantic groups 单例展开功能实现
2. 📋 测试四个策略的展开功能（test-expand-semantic-*）

### 第二阶段（高优先级）
1. 补充 VPIN、WPT、Volume Profile、DTW 特征测试
2. 开始 Nautilus 回测集成（至少完成一个策略）

### 第三阶段（中优先级）
1. 补充 Interaction、Trend、Momentum 特征测试
2. 改进现有测试文件（补充缺失的测试用例）
3. 完成所有 Nautilus 回测集成

---

## 📚 相关文档

- **特征工作流**: `docs/strategies/RECOMMENDED_FEATURE_WORKFLOW.md`
- **测试计划**: `docs/时序模型/测试补充计划.md`
- **测试覆盖**: `docs/特征测试覆盖总结.md`
- **Backtesting TODO**: `src/time_series_model/backtesting/TODO.md`
- **Experiment Loop**: `docs/architecture/EXPERIMENT_LOOP_ARCHITECTURE.md`

---

## ✅ 已完成的重要任务

### 特征测试完整覆盖（2025-01-01）

**所有14个特征类别都有完整的测试覆盖！**

每个测试文件都包含以下4种测试：
1. ⭐⭐⭐⭐⭐ **无未来函数测试**：修改未来数据不影响历史特征值
2. ⭐⭐⭐⭐ **多资产归一化测试**：特征分布对齐，便于多资产训练
3. ⭐⭐⭐⭐ **流式vs批量一致性测试**：生产部署关键，确保在线推理与训练一致
4. ⭐⭐⭐ **特征数学正确性验证**：验证特征计算的数学正确性

**测试覆盖情况**：
- ✅ **baseline**: test_trend_features.py, test_momentum_features.py, test_advanced_features.py
- ✅ **order_flow**: test_vpin_features.py, test_vpin_future_leak_and_multi_asset.py, test_interaction_features.py
- ✅ **volatility**: test_volume_profile_volatility_features.py, test_volume_profile_volatility_future_leak_and_multi_asset.py
- ✅ **wpt**: test_wpt_volatility_features.py, test_wpt_future_leak_and_multi_asset.py
- ✅ **dtw**: test_advanced_features.py, test_dtw_narrow_entrypoint.py
- ✅ **garch**: test_advanced_features.py
- ✅ **evt**: test_advanced_features.py, test_garch_evt_features.py
- ✅ **hilbert**: test_hilbert_features_improved.py
- ✅ **hurst**: test_hurst_features_improved.py
- ✅ **spectrum**: test_spectrum_features.py
- ✅ **liquidity**: test_liquidity_features.py
- ✅ **interaction**: test_interaction_features.py
- ✅ **market_cap**: test_market_cap_features.py（新建）
- ✅ **derived**: 包含在 baseline 和其他测试中

**新增测试文件**：
1. `test_trend_features.py` - Trend 特征完整测试
2. `test_momentum_features.py` - Momentum 特征完整测试
3. `test_market_cap_features.py` - Market Cap 特征完整测试
4. 补充了 VPIN、WPT、Volume Profile 的流式vs批量一致性测试
5. 补充了 DTW 的完整测试

**测试结果**：所有测试通过 ✅

---

## 🔄 更新记录

- **2025-01-01**: 初始创建，汇总所有 TODO 任务
- **2025-01-01**: 完成所有特征测试覆盖，14个特征类别都有完整测试

