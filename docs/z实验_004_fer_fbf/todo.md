# FER 训练任务进度

## ✅ 准备阶段

| 任务 | 状态 | 说明 |
|------|------|------|
| 创建配置目录 | ✅ 已完成 | config/strategies/fer/archetypes/ |
| 创建 labels_rr_extreme.yaml | ✅ 已完成 | Gate 训练标签 |
| 创建 labels_return_tree.yaml | ✅ 已完成 | Evidence 训练标签 |
| 创建 features_gate.yaml | ✅ 已完成 | Gate 训练特征（5类因果结构） |
| 创建 features_evidence.yaml | ✅ 已完成 | Evidence 训练特征 |
| 创建 archetype 模板 | ✅ 已完成 | gate/evidence/entry_filters/execution/holding.yaml |
| 实现 FER 标签函数 | ⏳ 待执行 | 复用 BPC 标签函数 |
| 实现 FER 特征函数 | ⏳ 待执行 | 5类因果结构特征 |

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
| 实现 FERLiveStrategy | ⏳ 待执行 | src/time_series_model/live/fer_live_strategy.py |
| LivePCM 注册 FER | ⏳ 待执行 | run_live.py 集成 |
| 特征一致性验证 | ⏳ 待执行 | compare_same_data.py |
| PCM 仲裁测试 | ⏳ 待执行 | FER+ME+BPC 三策略仲裁 |
| E2E 冒烟测试 | ⏳ 待执行 | tick → 特征 → 信号 → 开仓 |

---

## 关键 KPI 目标

| 阶段 | KPI | 目标 |
|------|-----|------|
| Gate | Lift | > 1.0（降低失败率） |
| Evidence | bad_suppression | > 0.05 |
| Entry Filter | snotio | 提升 mean(R-multiples) |
| Execution | Sharpe | 最大化 |
| PCM 联合 | Sharpe 提升 | FER + ME + BPC > max(单策略) |

---

## FER 特定注意事项

### 1. 特征实现优先级（按因果结构）

**必须实现**：
- ① 推进效率下降：price_delta_efficiency_f, momentum_efficiency_decay_f
- ② 吸收特征：aggressor_absorption_ratio_f, trapped_flow_indicator_f
- ⑤ 能量衰减：impulse_failure_score_f

**可选实现**：
- ③ Trapped Cluster：trapped_longs_ratio_f
- ④ 流动性错配：sweep_failure_rate_f

**可复用**：
- 辅助特征：vpin, rsi, macd, atr（已有实现）

### 2. 语义验证重点

- **区分疲劳 vs 失败**：
  - 疲劳：只是慢，但方向仍在
  - 失败：推进已死，吸收明显
  
- **Impulse 必须存在**：
  - 必须先有单边impulse
  - 检查 time_since_impulse 特征

### 3. PCM 仲裁验证

- FER 优先级 = 0（最高）
- 测试场景：FER + ME 同时触发 → FER 优先
- 测试场景：FER + BPC 同时触发 → FER 优先

---

## 配置同步检查

研究阶段完成后：

```bash
# 同步到实盘目录
cp -r config/strategies/fer/archetypes live/highcap/config/strategies/fer/
```

验证清单：
- [ ] live/highcap/config/strategies/fer/archetypes/ 存在
- [ ] 5 个 archetype 文件都已同步
- [ ] PCM 优先级配置正确（fer=0）
