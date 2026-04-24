# FER 策略训练与实盘

## 实验目的

为 FER (Failure Exhaustion Reversal) 策略完成从研究到实盘的完整流程：

1. **研究阶段**：找到适合 FER 的 Gate/Evidence/Entry 规则
2. **实盘阶段**：FER + BPC + ME 通过 LivePCM 仲裁发信号开仓

---

## FER 策略语义

**核心逻辑**：单边博弈已经失败 → 反向清算

### 三个核心条件

1. **之前存在单边 impulse** (有趋势)
2. **impulse 出现结构性失败** (推进效率下降)
3. **反向资金开始接管** (trapped 清算)

### 与其他策略的区别

| 策略 | 核心语义 | 触发条件 |
|------|----------|----------|
| **BPC** | 布林带压缩 → 回踩 → 延续 | 压缩 + 回踩 + 延续 |
| **ME** | 压缩 → 波动扩张 → 突破 | ATR 扩张 + 放量突破 |
| **FER** | 单边失败 → 反向清算 | 推进效率下降 + trapped + 结构打穿 |

**关键判断**：
> **资金强度没有下降，但价格推进已经死亡。**

---

## 训练流水线

```
Feature Store → Gate 训练 → Evidence 训练 → Entry Filter → Execution → 输出报告
```

### Step 1: 数据准备

```bash
mlbot feature-store build --no-docker \
  --config config/strategies/fer \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --start-date 2023-01-01 --end-date 2026-01-01 \
  --warmup-months 6
```

### Step 2: Gate 训练 (KPI: Lift)

**目标**：识别"踩大坑"的结构条件

```bash
# 训练
mlbot train final --no-docker \
  --config config/strategies/fer \
  --features config/strategies/fer/features_gate.yaml \
  --labels config/strategies/fer/labels_rr_extreme.yaml \
  --symbol BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T --data-path data/parquet_data \
  --start-date 2023-01-01 --end-date 2026-01-01 \
  --holdout-start-date 2024-05-01 --holdout-end-date 2026-01-01 --seed 42

# 优化
mlbot gate apply-archetype \
  --logs results/train_final_<ts>_rr_extreme/fer/predictions.parquet --strategy fer

python scripts/optimize_gate_unified.py --strategy fer \
  --logs results/train_final_<ts>_rr_extreme/fer/logs_gated.parquet \
  --output results/train_final_<ts>_rr_extreme/fer/gate_optimization.json
```

### Step 3: Evidence 训练 (KPI: bad_suppression)

**目标**：在 GOOD 样本上学习如何放大 RR

```bash
# 训练
mlbot train final --no-docker \
  --config config/strategies/fer \
  --features config/strategies/fer/features_evidence.yaml \
  --labels config/strategies/fer/labels_return_tree.yaml \
  --symbol BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T --data-path data/parquet_data \
  --start-date 2023-01-01 --end-date 2026-01-01 \
  --holdout-start-date 2024-05-01 --holdout-end-date 2026-01-01 --seed 42

# 优化 (基于 RR 分层)
python scripts/optimize_evidence_plateau.py --strategy fer \
  --logs results/train_final_<ts>_return_tree/fer/logs_gated.parquet \
  --output results/train_final_<ts>_return_tree/fer/evidence_optimization.json
```

### Step 4: Entry Filter (KPI: snotio)

```bash
python scripts/optimize_entry_filter_plateau.py \
  --logs results/train_final_<ts>_return_tree/fer/predictions.parquet --strategy fer
```

### Step 5: Execution (KPI: Sharpe)

```bash
python scripts/optimize_execution_grid.py \
  --logs results/train_final_<ts>_return_tree/fer/logs_gated.parquet --strategy fer \
  --output results/train_final_<ts>_return_tree/fer/execution_grid.json
```

### Step 6: PCM 联合回测

```bash
# 单独回测 FER
python scripts/backtest_execution_layer.py \
  --logs results/train_final_<ts>_return_tree/fer/predictions.parquet \
  --strategy fer

# FER + BPC + ME PCM 联合回测
python scripts/backtest_execution_layer.py \
  --pcm bpc:results/bpc/predictions.parquet \
       me:results/me/predictions.parquet \
       fer:results/fer/predictions.parquet \
  --quantile-train-start 2025-02-01 \
  --quantile-train-end 2025-08-01
```

---

## 研究阶段产出

| 产出 | 文件 | KPI |
|------|------|-----|
| Gate 规则 | archetypes/gate.yaml | Lift > 1.0 |
| Evidence 规则 | archetypes/evidence.yaml | bad_suppression > 0.05 |
| Entry Filter | archetypes/entry_filters.yaml | snotio 提升 |
| Execution 参数 | archetypes/execution.yaml | Sharpe 最大化 |
| 训练报告 | z实验_004_fer/实验报告.md | 汇总 KPI |

---

## FER 特征体系（按因果结构）

### ① 推进效率下降（第一性原理）

**核心**: 单位"钱"换来的"位移"变小

- `delta_efficiency` = ΔPrice / ΔDelta
- `volume_efficiency` = ΔPrice / Volume  
- `cluster_efficiency` = bar_range / cluster_size
- `wpt_efficiency` = 实际推进 / WPT 能量

### ② 吸收 (Absorption)

**核心**: 大量 aggressor 但价格不动

- `absorption_score` = high_delta_bar AND close < open AND delta_efficiency < threshold
- `wick_scene.long_wick` + 高成交
- `fp_scene.aggressive_but_stuck`

### ③ Trapped Cluster（被套证据）

**核心**: 单边冲高后被困

- cluster 集中在极值区
- 价格迅速回到 value area
- VPIN spike

### ④ 流动性错配

**核心**: Sweep 后立即回到高流动区

- Sweep 后未延续
- liquidity void 未被填补
- delta spike 后无 follow-through

### ⑤ 能量衰减

**核心**: 动能还在，但推进已死

- WPT peak 后连续下降
- energy divergence
- bar range 收窄但 cluster 变大

---

## PCM 仲裁优先级

| Archetype | Priority | 说明 |
|-----------|----------|------|
| **Reversal (FER)** | **0** | **条件最严格，单边失败清算** |
| ME | 1 | 需确认扩张 |
| BPC | 2 | 最宽松 |

**决策依据**：按条件严格性排序 — FER 的触发条件最严格（需单边失败 + 结构打穿 + 反向接管），优先级最高。

---

## 实盘阶段

见 `z实验_005_fer实盘/` 目录（待创建）

---

## 关键 KPI 目标

| 阶段 | KPI | 目标 |
|------|-----|------|
| Gate | Lift | > 1.0（降低失败率） |
| Evidence | bad_suppression | > 0.05 |
| Entry Filter | snotio | 提升 mean(R-multiples) |
| Execution | Sharpe | 最大化 |
| PCM 联合 | Sharpe 提升 | FER + ME + BPC > max(单独) |

---

## 注意事项

1. **FER 最大风险**：误判"疲劳"为"失败"
   - 疲劳：资金还在 + 推进慢 → 继续趋势
   - 失败：资金还在 + 推进死 → 反转 ✅

2. **区分方法**：
   | 状态 | 订单流 | 价格推进 | FER 判断 |
   |------|--------|----------|----------|
   | 健康趋势 | 强 | 正常 | ❌ |
   | 疲劳趋势 | 强 | 慢 | ❌ |
   | **失败趋势** | **强** | **死/反** | **✅** |

3. **特征选择原则**：
   - ✅ 推进效率、吸收、trapped、sweep、衰竭
   - ❌ 单纯 RSI 背离、ATR 收缩、MA 交叉
