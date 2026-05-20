# ME 训练任务进度

## ✅ 准备阶段

| 任务 | 状态 | 说明 |
|------|------|------|
| 创建配置目录 | ✅ 已完成 | config/strategies/me/archetypes/ |
| 创建 labels_rr_extreme.yaml | ✅ 已完成 | Gate 训练标签 |
| 创建 labels_return_tree.yaml | ✅ 已完成 | Evidence 训练标签 |
| 创建 features_gate.yaml | ✅ 已完成 | Gate 训练特征 |
| 创建 features_evidence.yaml | ✅ 已完成 | Evidence 训练特征 |
| 实现 ME 标签函数 | ✅ 已完成 | compute_me_failure_rr_extreme_label, compute_me_return_tree_label |
| 创建 archetype 模板 | ✅ 已完成 | gate/evidence/entry_filters/execution/holding.yaml |

## 🔄 研究阶段（待执行）

| 任务 | 状态 | 命令 |
|------|------|------|
| Step 1: Feature Store | ⏳ 待执行 | mlbot feature-store build |
| Step 2: Gate 训练 | ⏳ 待执行 | mlbot train final --labels labels_rr_extreme |
| Step 3: Gate 优化 | ⏳ 待执行 | optimize_gate_unified.py |
| Step 4: Evidence 训练 | ⏳ 待执行 | mlbot train final --labels labels_return_tree |
| Step 5: Evidence 优化 | ⏳ 待执行 | optimize_evidence_plateau.py |
| Step 6: Entry Filter 优化 | ⏳ 待执行 | optimize_entry_filter_plateau.py |
| Step 7: Execution 优化 | ⏳ 待执行 | optimize_execution_grid.py |
| Step 8: PCM 联合回测 | ⏳ 待执行 | backtest_execution_layer.py --pcm |
| Step 9: 输出训练报告 | ⏳ 待执行 | 汇总 KPI 到 实验报告.md |

## 🚀 实盘阶段（待创建）

| 任务 | 状态 | 说明 |
|------|------|------|
| 实现 MELiveStrategy | ⏳ 待执行 | src/time_series_model/live/me_live_strategy.py |
| LivePCM 注册 ME | ⏳ 待执行 | run_live.py 集成 |
| 特征一致性验证 | ⏳ 待执行 | compare_same_data.py |
| PCM 仲裁测试 | ⏳ 待执行 | ME+BPC 同时触发验证 |
| E2E 冒烟测试 | ⏳ 待执行 | tick → 特征 → 信号 → 开仓 |

---

## 关键 KPI 目标

| 阶段 | KPI | 目标 |
|------|-----|------|
| Gate | Lift | > 1.0（降低失败率） |
| Evidence | bad_suppression | > 0.05 |
| Entry Filter | snotio | 提升 mean(R-multiples) |
| Execution | Sharpe | 最大化 |
| PCM 联合 | Sharpe 提升 | ME + BPC > max(ME, BPC) |

---

## 注意事项

1. **训练前检查**：确保 Feature Store 已构建完成
2. **训练后审核**：每个优化脚本的输出需人工审核后再更新配置
3. **配置同步**：研究阶段完成后，需将 archetypes 配置同步到 live/highcap/
