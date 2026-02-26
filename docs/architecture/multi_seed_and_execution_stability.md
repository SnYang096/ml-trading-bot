# 多 Seed 搜索 + Execution 基线稳定性

## 1. 问题背景

### 1.1 非确定性根因

LightGBM `deterministic=True` 不保证跨运行一致性，因为 OpenMP 多线程浮点归约顺序不固定：
- 8 线程并行求 `loss = Σ(residual²)`，完成顺序每次不同
- `(a+b)+c ≠ a+(b+c)` 在浮点精度下成立
- 微小差异导致 early stopping 在不同 round 触发 → 完全不同的模型

**当前策略**：不强制单线程，而是用多 Seed 搜索将非确定性转化为"免费的额外探索"。
- 默认 `num_threads=-1`（多线程，快 ~3x）
- `--deterministic` CLI flag 保留作为调试选项（强制 `num_threads=1`）
- 代价：管线不可逐字复现，但产出的 archetypes 配置是确定性的

### 1.2 模型对 Seed 敏感度差异

| 策略 | 多线程时代跨 run 稳定性 | 原因 |
|------|------------------------|------|
| BPC  | ✅ 4次运行完全一致       | Gate 特征信号强（evt_var_99, macd_signal_atr），任何扰动都选中相同特征 |
| FER  | ⚠️ 2次变化              | 中等信号，特征竞争存在 |
| ME   | ❌ 每次都不同            | Gate 特征信号弱，微小扰动就换特征 |

**结论**：跨 seed 不稳定 = 模型信号弱的诊断信号。

### 1.3 Execution 小样本过拟合

当 gate 筛选后交易数过少时，execution grid search 在极小样本上 "优化" 出极端参数：

| Gate allows | Execution 结果 | 问题 |
|-------------|---------------|------|
| 136 trades  | initial_r=2.0, act_r=3.0 | 合理 |
| 27 trades   | initial_r=1.0, act_r=1.0 | 过拟合（1R 止损几乎没有呼吸空间） |

---

## 2. 方案设计

### 2.1 多 Seed 搜索

**目标**：将不可控的多线程随机搜索 → 可控的系统性 seed 搜索。

**配置**（research_pipeline.yaml）：
```yaml
training:
  seeds: [42, 1, 2, 3, 4]    # 5 个 seed
  seed_selection: best_sharpe  # 选 Sharpe 最高的 seed
```

**流程**：
```
For each strategy:
  For each seed in seeds:
    1. 训练模型 (seed=N, deterministic)
    2. Gate optimize → Entry filter → Execution → Backtest
    3. 记录: {seed, sharpe, trades, gate_rules, ...}
  
  Select best seed:
    - 筛选: trades >= min_trades
    - 排序: sharpe_per_trade 降序
    - 选中: 排名第一的 seed
    - 记录: selected_seed 写入 report.json
```

**优化（预留）**：Step 0-3（download, feature store, prepare, prefilter）与 seed 无关，只需跑一次。当前先简单实现全流程多次运行，后续优化为共享前置步骤。

**诊断输出**：
```
📊 ME Seed 搜索结果:
  seed=42: Sharpe=0.26  trades=27   gate=[sma_200_position, wpt_vper_mid]
  seed=1:  Sharpe=0.58  trades=136  gate=[shd_pct, evt_scale]
  seed=2:  Sharpe=0.41  trades=89   gate=[shd_pct, wpt_vper_mid]
  seed=3:  Sharpe=0.52  trades=110  gate=[evt_scale, sma_200_position]
  seed=4:  Sharpe=0.47  trades=95   gate=[shd_pct, evt_scale]
  
  🏆 Best: seed=1 (Sharpe=0.58, trades=136)
  📈 Stability: 3/5 seeds 选中 shd_pct → 该特征是稳定信号
  ⚠️ seed=42 表现最差 → 单 seed 不可靠
```

### 2.2 Execution 基线保护

**规则**：当 gate 筛选后交易数 < `min_trades_for_execution_opt` 时，跳过 execution grid search，保留现有 execution.yaml 不动。

**配置**（research_pipeline.yaml 的 kpi_gates）：
```yaml
kpi_gates:
  execution:
    min_trades: 50  # gate 后 < 50 trades 则不优化 execution
```

**原理**：
- Execution grid search 需要足够样本量来估计 Sharpe 的统计意义
- < 50 trades 时，grid search 找到的 "最优" 参数是噪声
- 保留已有参数（上次大样本时优化的）更安全

**实现**：在 auto_research_pipeline.py Step 8 前检查 logs_gated 的 allow 数量。

---

## 3. 实现清单

| # | 文件 | 改动 |
|---|------|------|
| 1 | `config/research_pipeline.yaml` | 添加 `training.seeds` + `kpi_gates.execution.min_trades` |
| 2 | `scripts/auto_research_pipeline.py` | 多 seed 循环 + seed 选优 + execution 交易数检查 |
| 3 | 无新文件 | — |

---

## 4. 不采纳的方案

### 4.1 换 sklearn DecisionTree 替代 LightGBM

**优点**：单棵树，天然确定性，无 early stopping 级联。
**不采纳原因**：
- 改动面太大（训练流程、gate 优化器、规则提取全要重写）
- 表达力下降（depth=5 仅 32 叶，复杂交互无法捕捉）
- 多 seed 搜索已解决稳定性问题，不需要降级模型
- 可作为后续优化方向（如果多 seed 后 ME 仍不稳定）
