# 本地研究 Pipeline 命令速查

> 更新日期：2026-02-22
> 配置文件：`config/research_pipeline.yaml`
> 脚本入口：`scripts/auto_research_pipeline.py`

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    auto_research_pipeline.py                     │
│                                                                  │
│  config/strategies/fer/   ──复制──▶  实验工作区/strategies/fer/  │
│  (生产 config, 不被覆盖)              (所有 --promote 写这里)    │
│                                                                  │
│  ADOPT 时:  实验/archetypes/ ──复制──▶ 生产/archetypes/          │
└─────────────────────────────────────────────────────────────────┘
```

### 实验目录结构

```
results/research_history/{strategy}/{YYYYMMDD_HHMMSS}/
  ├── strategies/{strategy}/          # 隔离的 config 副本
  │   ├── archetypes/                 # 所有 --promote 写到这里
  │   │   ├── gate.yaml
  │   │   ├── evidence.yaml
  │   │   ├── entry_filters.yaml
  │   │   ├── execution.yaml
  │   │   ├── prefilter.yaml
  │   │   └── direction.yaml
  │   ├── features_gate.yaml
  │   ├── prefilter.yaml
  │   └── ...
  ├── archetypes/                     # archetypes 快照副本 (方便查看)
  ├── report.json                     # 结构化指标 + 阈值 + 对比决策
  ├── comparison.json                 # 与上次实验的对比详情
  ├── pipeline.log                    # 完整运行日志
  └── training_baseline.json          # 训练基线 (供监控使用)
```

---

## 一、一键自动化 (推荐)

### 单策略全流程

```bash
cd /home/yin/trading/ml_trading_bot

# FER 策略一键研究 (自动检测最新数据日期)
python scripts/auto_research_pipeline.py --strategy fer

# BPC 策略
python scripts/auto_research_pipeline.py --strategy bpc

# ME 策略
python scripts/auto_research_pipeline.py --strategy me
```

### 全部策略

```bash
python scripts/auto_research_pipeline.py --all
```

### 指定时间范围

```bash
# 指定 end-date
python scripts/auto_research_pipeline.py --strategy fer --end-date 2026-03-01
```

### Dry-run (只打印命令不执行)

```bash
python scripts/auto_research_pipeline.py --strategy fer --dry-run
```

### 只保存实验不自动采纳

```bash
# --no-adopt: ADOPT 决策时不自动覆盖生产 config, 需手动 --adopt
python scripts/auto_research_pipeline.py --strategy fer --no-adopt
```

---

## 二、实验管理

### 列出历史实验

```bash
# 列出某策略的所有实验
python scripts/auto_research_pipeline.py --strategy fer --list

# 列出所有策略的实验
python scripts/auto_research_pipeline.py --all --list
```

输出示例:
```
📋 FER 历史实验 (3 次):
────────────────────────────────────────────────────────────────────────────
  时间戳                  Sharpe    Trades   决策    备注
────────────────────────────────────────────────────────────────────────────
  20260220_100000          0.1234       45  ✅ ADOPT  2023-01-01~2026-01-01
  20260221_140000          0.0987       38  ⚠️ ALERT  2023-01-01~2026-02-01
  20260222_120000          0.1456       52  ✅ ADOPT  2023-01-01~2026-03-01
```

### 手动采纳指定实验

```bash
# 采纳某次实验的 config → 覆盖生产 config/strategies/fer/archetypes/
python scripts/auto_research_pipeline.py --strategy fer --adopt 20260222_120000
```

### 对比两次实验

```bash
# YAML 级别 diff: 逐 key 对比 archetypes 差异
python scripts/auto_research_pipeline.py --strategy fer --diff 20260220_100000 20260222_120000
```

---

## 三、自动流水线 11 步详解

```
Step 0:  Data Download + Convert (增量, 已有月份跳过, 失败不中断)
Step 1:  Feature Store (增量, 已有月份自动跳过)
Step 2:  Prepare Only (--prepare-only, 导出 features_labeled.parquet)
Step 3:  Prefilter --promote (分析环境条件 → 实验/archetypes/prefilter.yaml)
Step 4:  Direction --promote (验证方向质量 → 实验/archetypes/direction.yaml)
Step 5:  Gate (train + apply + optimize --promote → 实验/archetypes/gate.yaml)
Step 6:  Evidence (train + apply + optimize --promote → 实验/archetypes/evidence.yaml)
Step 7:  Entry Filter --promote (plateau 验证 → 实验/archetypes/entry_filters.yaml)
Step 8:  Execution --promote (grid search → 实验/archetypes/execution.yaml)
Step 9:  Backtest (单策略回测, 提取 Sharpe/Trades/WinRate)
Step 10: Export Training Baseline (training_baseline.json, 容错)
```

### 决策规则 (全确定性, 无人工)

| 条件 | 决策 |
|------|------|
| trades < 10 | ERROR |
| sharpe <= 0 | ALERT |
| new/prev >= 0.7 | ADOPT → 自动覆盖生产 config |
| new/prev < 0.7 | ALERT (显著衰减) |
| 首次运行 | ADOPT |

---

## 四、手动分步执行

适用于调试或需要在中间步骤手动干预的场景。

### 通用参数

```bash
STRATEGY="fer"              # bpc / fer / me
TIMEFRAME="240T"            # fer/bpc=240T, me=60T
SYMBOLS="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT"
START="2023-01-01"
END="2026-03-01"            # 最新数据月+1
HOLDOUT="2025-01-01"        # = END - 14个月
```

### Step 1: Feature Store

```bash
mlbot feature-store build --no-docker \
  --config config/strategies/${STRATEGY} \
  --symbols ${SYMBOLS} --timeframe ${TIMEFRAME} \
  --start-date ${START} --end-date ${END} --warmup-months 6
```

### Step 2: Prepare Only

```bash
mlbot train final --no-docker --prepare-only \
  --config config/strategies/${STRATEGY} \
  --features config/strategies/${STRATEGY}/features_gate.yaml \
  --labels config/strategies/${STRATEGY}/labels_rr_extreme.yaml \
  --symbol ${SYMBOLS} --timeframe ${TIMEFRAME} --data-path data/parquet_data \
  --start-date ${START} --end-date ${END} \
  --holdout-start-date ${HOLDOUT} --holdout-end-date ${END} --seed 42

PREPARE_DIR="results/train_final_<timestamp>_rr_extreme/${STRATEGY}"
```

### Step 3: Prefilter

```bash
python scripts/analyze_archetype_feature_stratification.py \
  --logs ${PREPARE_DIR}/features_labeled.parquet \
  --strategy ${STRATEGY} --select-recent 6 --promote
```

### Step 4: Direction

```bash
python z实验_005_统一研究/direction_strict_validation.py \
  --logs ${PREPARE_DIR}/features_labeled.parquet \
  --strategy ${STRATEGY} --compare-features --temporal --promote
```

### Step 5: Gate 训练 + 优化

```bash
# 训练 (带 prefilter)
mlbot train final --no-docker \
  --config config/strategies/${STRATEGY} \
  --features config/strategies/${STRATEGY}/features_gate.yaml \
  --labels config/strategies/${STRATEGY}/labels_rr_extreme.yaml \
  --archetype-prefilter config/strategies/${STRATEGY}/archetypes/prefilter.yaml \
  --symbol ${SYMBOLS} --timeframe ${TIMEFRAME} --data-path data/parquet_data \
  --start-date ${START} --end-date ${END} \
  --holdout-start-date ${HOLDOUT} --holdout-end-date ${END} --seed 42

GATE_DIR="results/train_final_<timestamp>_rr_extreme/${STRATEGY}"

# Apply draft gate
mlbot gate apply-archetype \
  --logs ${GATE_DIR}/predictions.parquet --strategy ${STRATEGY} \
  --gate-path config/strategies/${STRATEGY}/gate_draft.yaml

# Optimize + promote
python scripts/optimize_gate_unified.py --strategy ${STRATEGY} \
  --logs ${GATE_DIR}/logs_gated.parquet \
  --output ${GATE_DIR}/gate_optimization.json \
  --gate-path config/strategies/${STRATEGY}/gate_draft.yaml --promote

# Re-apply with optimized gate
mlbot gate apply-archetype \
  --logs ${GATE_DIR}/predictions.parquet --strategy ${STRATEGY} \
  --gate-path config/strategies/${STRATEGY}/archetypes/gate.yaml
```

### Step 6: Evidence 训练 + 优化

```bash
mlbot train final --no-docker \
  --config config/strategies/${STRATEGY} \
  --features config/strategies/${STRATEGY}/features_evidence.yaml \
  --labels config/strategies/${STRATEGY}/labels_return_tree.yaml \
  --archetype-prefilter config/strategies/${STRATEGY}/archetypes/prefilter.yaml \
  --symbol ${SYMBOLS} --timeframe ${TIMEFRAME} --data-path data/parquet_data \
  --start-date ${START} --end-date ${END} \
  --holdout-start-date ${HOLDOUT} --holdout-end-date ${END} --seed 42

EV_DIR="results/train_final_<timestamp>_return_tree/${STRATEGY}"

# Gate apply
mlbot gate apply-archetype \
  --logs ${EV_DIR}/predictions.parquet --out ${EV_DIR}/logs_gated.parquet \
  --gate-path config/strategies/${STRATEGY}/archetypes/gate.yaml --strategy ${STRATEGY}

# Evidence optimize + promote
python scripts/optimize_evidence_plateau.py --strategy ${STRATEGY} \
  --candidates ${EV_DIR}/evidence_candidates.yaml \
  --predictions ${EV_DIR}/predictions.parquet \
  --logs ${EV_DIR}/logs_gated.parquet \
  --output ${EV_DIR}/evidence_optimization.json --promote
```

### Step 7: Entry Filter

```bash
python scripts/optimize_entry_filter_plateau.py \
  --logs ${EV_DIR}/predictions.parquet --strategy ${STRATEGY} \
  --research --promote
```

### Step 8: Execution

```bash
python scripts/optimize_execution_grid.py \
  --logs ${EV_DIR}/logs_gated.parquet --strategy ${STRATEGY} \
  --output ${EV_DIR}/execution_grid.json --promote
```

### Step 9: Backtest

```bash
# 单策略回测
python scripts/backtest_execution_layer.py \
  --logs ${EV_DIR}/predictions.parquet --strategy ${STRATEGY}

# PCM 联合回测
python scripts/backtest_execution_layer.py \
  --pcm fer:${FER_DIR}/predictions.parquet \
       bpc:${BPC_DIR}/predictions.parquet \
       me:${ME_DIR}/predictions.parquet
```

### Step 10: Baseline 导出

```bash
python scripts/export_training_baseline.py \
  --strategy ${STRATEGY} --result-dir ${EV_DIR} \
  --gate-dir ${GATE_DIR} --evidence-dir ${EV_DIR}
```

---

## 五、监控命令

### 特征漂移检测

```bash
python scripts/local_monitor_feature_drift.py \
  --baseline ${EV_DIR}/training_baseline.json \
  --new-data <新数据.parquet>
# Exit: 0=stable, 1=drift, 2=severe
```

### 周频快速检查

```bash
python scripts/local_monitor_weekly.py \
  --baseline ${EV_DIR}/training_baseline.json \
  --new-data <新数据.parquet> --strategy ${STRATEGY}
# Exit: 0=healthy, 1=attention, 2=retrain
```

### 月频全层报告

```bash
python scripts/local_monitor_monthly.py \
  --baseline ${EV_DIR}/training_baseline.json \
  --new-data <新数据.parquet> --strategy ${STRATEGY}
# Exit: 0=healthy, 1=attention, 2=retrain
```

---

## 六、配置文件说明

### config/research_pipeline.yaml

```yaml
dates:
  start_date: "2023-01-01"     # 最早可用训练数据
  holdout_months: 14           # Holdout 固定窗口

symbols: "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT"
data_path: "data/parquet_data"

download:
  enabled: true                # false 跳过下载
  data_dir: "data/agg_data"
  parquet_dir: "data/parquet_data"

strategies:
  bpc:
    config: config/strategies/bpc
    timeframe: "240T"
    has_prefilter: true
    has_direction: true
  fer:
    config: config/strategies/fer
    timeframe: "240T"
    has_prefilter: true
    has_direction: true
  me:
    config: config/strategies/me
    timeframe: "60T"
    has_prefilter: true
    has_direction: true

comparison:
  min_trades: 10
  sharpe_adopt_ratio: 0.7
  sharpe_reject_floor: 0.0
```

---

## 七、改进方向

- [ ] 训练产出 (predictions.parquet 等) 也移入实验目录 (需修改 mlbot train 输出路径)
- [ ] 线上/线下特征对比 (`scripts/compare_live_vs_batch_features.py`)
- [ ] PCM 联合回测自动化 (需多策略结果目录)
- [ ] Web UI 可视化实验对比
