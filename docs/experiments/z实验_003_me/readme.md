# 实验目的

为 ME (MomentumExpansion) 策略完成从研究到实盘的完整流程：

1. **研究阶段**：找到适合 ME 的 Gate/Evidence/Entry 规则
2. **实盘阶段**：ME + BPC 共同启动，通过 LivePCM 仲裁发信号开仓

---

## ME 策略语义

**核心逻辑**：压缩后波动/区间扩张，放量突破

- **触发条件**：ATR 扩张 + 放量突破
- **方向判断**：breakout_sign（突破方向）
- **与 BPC 的区别**：
  - BPC：布林带压缩 → 回踩 → 延续
  - ME：压缩 → 波动扩张 → 突破

---

## 训练流水线

```
Feature Store → Gate 训练 → Evidence 训练 → Entry Filter → Execution → 输出报告
```

### Step 1: 数据准备

```bash
mlbot feature-store build --no-docker \
  --config config/strategies/me \
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
  --config config/strategies/me \
  --features config/strategies/me/features_gate.yaml \
  --labels config/strategies/me/labels_rr_extreme.yaml \
  --symbol BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T --data-path data/parquet_data \
  --start-date 2023-01-01 --end-date 2026-01-01 \
  --holdout-start-date 2024-05-01 --holdout-end-date 2026-01-01 --seed 42

# 优化
mlbot gate apply-archetype \
  --logs results/train_final_<ts>_rr_extreme/me/predictions.parquet --strategy me

python scripts/optimize_gate_unified.py --strategy me \
  --logs results/train_final_<ts>_rr_extreme/me/logs_gated.parquet \
  --output results/train_final_<ts>_rr_extreme/me/gate_optimization.json

# 手动审核更新 config/strategies/me/archetypes/gate.yaml
```

### Step 3: Evidence 训练 (KPI: bad_suppression)

**目标**：在 GOOD 样本上学习如何放大 RR

```bash
# 训练
mlbot train final --no-docker \
  --config config/strategies/me \
  --features config/strategies/me/features_evidence.yaml \
  --labels config/strategies/me/labels_return_tree.yaml \
  --symbol BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T --data-path data/parquet_data \
  --start-date 2023-01-01 --end-date 2026-01-01 \
  --holdout-start-date 2024-05-01 --holdout-end-date 2026-01-01 --seed 42

# 优化
python scripts/optimize_evidence_plateau.py --strategy me \
  --logs results/train_final_<ts>_return_tree/me/logs_gated.parquet \
  --output results/train_final_<ts>_return_tree/me/evidence_optimization.json

# 手动审核更新 config/strategies/me/archetypes/evidence.yaml
```

### Step 4: Entry Filter (KPI: snotio)

```bash
python scripts/optimize_entry_filter_snotio.py --all \
  --logs results/train_final_<ts>_return_tree/me/predictions.parquet --strategy me

python scripts/optimize_entry_filter_plateau.py \
  --logs results/train_final_<ts>_return_tree/me/predictions.parquet --strategy me

# 手动审核更新 config/strategies/me/archetypes/entry_filters.yaml
```

### Step 5: Execution (KPI: Sharpe)

```bash
python scripts/optimize_execution_grid.py \
  --logs results/train_final_<ts>_return_tree/me/logs_gated.parquet --strategy me \
  --output results/train_final_<ts>_return_tree/me/execution_grid.json

# 手动审核更新 config/strategies/me/archetypes/execution.yaml
```

### Step 6: PCM 联合回测

```bash
# 单独回测 ME
python scripts/backtest_execution_layer.py \
  --logs results/train_final_<ts>_return_tree/me/predictions.parquet \
  --strategy me

# ME + BPC PCM 联合回测
python scripts/backtest_execution_layer.py \
  --pcm bpc:results/bpc/predictions.parquet \
       me:results/me/predictions.parquet \
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
| 训练报告 | docs/z实验_003_me/实验报告.md | 汇总 KPI |

---

## 实盘阶段

见 `z实验_004_me实盘/` 目录（待创建）
