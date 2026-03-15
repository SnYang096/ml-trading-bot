# 快速启动命令 — 研究 Pipeline 工作流

> 更新时间: 2026-03-10  
> 策略: bpc-long, bpc-short, fer-long, fer-short, me-long, me-short, lv（共 7 策略）  
> 信号栈: Prefilter → Gate → Execution（Evidence + Entry Filter 已移除）  
> CLI 入口: `mlbot pipeline <subcommand>`

---

## 一、前置条件

```bash
# 确认数据目录有数据
ls data/parquet_data/ | head -5

# 确认策略配置目录
ls config/strategies/
```

---

## 二、数据下载

> 每次研究前更新数据，`--start-year/month` 指定数据拉取起点

```bash
# 1. OHLCV K 线数据（4H / 1min）
mlbot data pipeline-universe --no-docker \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-groups highcap \
  --start-year 2023 --start-month 1

# 2. 市值数据（7 天内已更新则跳过）
mlbot data update-market-cap \
  --config config/market_cap/market_cap.yaml \
  --max-age-days 7 \
  --no-docker

# 3. 资金费率
mlbot data download-funding-rate \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-groups highcap \
  --start-year 2023 --start-month 1 \
  --no-docker

# 4. 未平仓合约
mlbot data download-open-interest \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-groups highcap \
  --start-year 2023 --start-month 1 --progress-every 1
```

---

## 三、Feature Store 构建

> Feature Store 是模型训练的输入，**只需在特征代码或数据更新时重建**  
> `--timeframe` 必须与策略匹配：BPC/FER=240T，ME=60T，LV=15T  
> `--warmup-months 6` 为特征计算提供预热期，不计入训练

```bash
# BPC / FER 策略（共用 240T 特征）
mlbot feature-store build --no-docker \
  --config config/strategies/bpc-long \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-groups highcap \
  --timeframe 240T \
  --start-date 2023-01-01 --end-date 2026-03-01 \
  --warmup-months 6

# ME 策略（60T）
mlbot feature-store build --no-docker \
  --config config/strategies/me-long \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-groups highcap \
  --timeframe 60T \
  --start-date 2023-01-01 --end-date 2026-03-01 \
  --warmup-months 6

# LV 策略（15T）
mlbot feature-store build --no-docker \
  --config config/strategies/lv \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-groups highcap \
  --timeframe 15T \
  --start-date 2023-01-01 --end-date 2026-03-01 \
  --warmup-months 6
```

> BPC/FER 共享同一个 feature store，`config/strategies/bpc-long` 和 `config/strategies/fer-long` 指向相同特征节点，构建一次即可。

---

## 四、Research Pipeline

> 核心命令：`mlbot pipeline run`  
> 自动完成：Prepare → SHAP → Prefilter → Gate → 向量回测 → PCM 联合回测 → ADOPT 决策

### 4.0 数据划分

1. 应该划分训练集合（1年+），验证集合（调整阈值 3个月），holdout集合（3个月）
2. pipeline跨regime稳定
   1. （策略+参数不一定能跨regime，但我们要求管线重新训练后，能稳定）
   2. 实盘会和regime shift 探测结合重新训练
3. 最好能设计滚动，因为我们有202301~202602的数据
   1. 虽然我们导出规则，拟合情况少，如果能设计良好的滚动验证，也是不错的


### 4.1 全策略研究（正式）

```bash
# 全策略串行训练 + PCM 联合回测 + 自动 ADOPT
mlbot pipeline run --all
```

参数说明：
- `--all`：依次跑所有 7 个策略，最后做 PCM 联合仲裁
- `--end-date 2026-03-01`：手动指定数据截止日期（默认自动检测最新数据）
- `--no-adopt`：只保存实验结果，不自动写回 config（需手动 `pipeline adopt`）
- `--dry-run`：打印所有命令但不执行（检查参数用）
- `--skip-shap`：跳过 SHAP 特征筛选（快速迭代调试用，正式研究不建议）
- `--event-backtest`：训练完后自动跑事件回测 execution 优化（sym-r grid search）
- `--event-sym-r 1.0:0.5:4.0`：execution 优化 sym-r 搜索范围（配合 `--event-backtest`）

### 4.2 单策略研究（调试）

```bash
# 只跑一个策略（无 PCM，仅看信号质量和交易地图）
mlbot pipeline run --strategy me-long

# 快速验证（跳过 SHAP）
mlbot pipeline run --strategy fer-long --skip-shap
# --skip-shap 的作用：跳过 Walk-Forward SHAP 特征筛选步骤，直接复用上次缓存的 features_gate_shap.yaml，省约 5~10 分钟。有没有必要——取决于这次跑的目的：
# 情况	用不用 --skip-shap
# 只改了 prefilter（当前情况）	✅ 可以跳过，特征集没变
# 改了 features_gate_*.yaml（候选特征变了）	❌ 必须重跑 SHAP
# 改了训练时间窗口 / 数据范围	❌ 重跑，SHAP 结果会不同
# 快速验证 prefilter 多算法效果	✅ 跳过，节省时间


mlbot pipeline run --strategy fer-short 

# 精细模式（1min bar 执行层）
mlbot pipeline run --strategy bpc-long --use-1min
```

> 单策略模式下不运行 PCM 联合回测（无仲裁意义），ADOPT 决策仅基于单策略指标。

### 4.3 输出产物

每次实验结果保存在 `results/research_history/{strategy}/{YYYYMMDD_HHMMSS}/`：

```
results/research_history/me-long/20260313_234448/
├── report.json                          # 实验报告 + ADOPT 决策
├── strategies/                          # 训练后的 gate/prefilter 配置快照
└── results/
    ├── logs_gated.parquet               # 经 gate 过滤的信号（核心输出）
    ├── trading_map_me-long.html         # 交易地图（simple-execution，信号质量）
    ├── trading_map_me-long_exec.html    # 交易地图（真实 trailing stop 行为）
    └── shap/                            # SHAP 特征分析
```

---

## 五、实验管理

### 5.1 列出历史实验

```bash
# 列出单策略所有历史实验（含 Sharpe / WinRate / ADOPT 决策）
mlbot pipeline list --strategy me-long

# 列出全部策略
mlbot pipeline list --all
```

### 5.2 手动采纳实验

```bash
# 采纳指定时间戳的实验（将该实验的 config 写回 config/strategies/）
mlbot pipeline adopt 20260313_234448 --strategy me-long
```

### 5.3 对比两次实验

```bash
# 对比两次实验的 archetype 配置差异
mlbot pipeline diff 20260310_120000 20260313_234448 --strategy me-long
```

### 5.4 删除历史实验

```bash
# 预览（--dry-run 先看会删哪些）
mlbot pipeline delete --strategy me-long --status error --dry-run

# 按状态批量删除
mlbot pipeline delete --strategy me-long --status error

# 删除指定时间戳
mlbot pipeline delete --strategy me-long --timestamp 20260310_120000

# 删除全部历史实验（谨慎）
mlbot pipeline delete --strategy me-long --all
```

---

## 六、事件回测（Event Backtest）

> 用真实 1min bar 逐笔触发信号，完全模拟实盘时序，验证 execution 参数效果  
> 输出：交易地图 HTML + 交易明细 CSV

### 6.1 对实验运行事件回测

```bash
# 对最新实验运行事件回测（无 execution 优化）
mlbot pipeline event-backtest --strategy me-long

# 对指定实验运行（--hash 指定时间戳）
mlbot pipeline event-backtest --strategy me-long --hash 20260313_234448

# 同时做 execution 参数 grid search 优化（推荐 ME 策略）

mlbot pipeline event-backtest \
  --strategy me-short \
  --hash 20260313_234144 \
  --sym-r 1.0:0.5:4.0 \
  --promote  

# 先导出向量回测的交易明细
python scripts/backtest_execution_layer.py \
  --logs results/research_history/me-short/20260313_234144/results/logs_gated.parquet \
  --strategy me-short --strategies-root config/strategies \
  --test-start 2025-09-01 --test-end 2026-03-01 \
  --simple-execution --export /tmp/vector_trades_me-short.csv

# 在事件地图上叠加蓝圈对比向量回测
python scripts/event_backtest.py \
  --strategy me-short \
  --start-date 2025-09-01 --end-date 2026-03-01 \
  --strategies-root results/research_history/me-short/20260313_234144/strategies \
  --trading-map /tmp/event_map_compare.html \
  --compare-trades /tmp/vector_trades_me-short.csv  
```

参数说明：
- `--hash`：实验时间戳，不填则自动使用最新实验
- `--sym-r start:step:end`：开启对称 execution 优化，`initial_r=activation_r=trail_r` 联动搜索
- `--promote`：将最优参数写回实验目录的 `execution.yaml`
- `--fast` / `--no-fast`：快速模式（60T bar，默认开启）/ 精细模式（1min bar）

输出文件（保存到实验的 `results/` 目录）：
- `trading_map_{strategy}_event.html`：事件回测交易地图
- `event_trades_{strategy}.csv`：交易明细
- `event_exec_opt.json`：execution 优化结果（有 `--sym-r` 时生成）

### 6.2 手动运行事件回测脚本

```bash
# 基础用法（默认用研究数据 data/parquet_data，最近 180 天）
python scripts/event_backtest.py \
  --strategy me-long \
  --days 180

# 指定日期范围 + 导出交易明细
python scripts/event_backtest.py \
  --strategy me-long,bpc-long \
  --start-date 2025-06-01 --end-date 2026-03-01 \
  --trading-map results/trading_map_me_bpc.html \
  --export results/event_trades.csv
```

### 6.3 Execution 参数 Grid Search（手动）

```bash
# ME 策略对称优化（initial_r=activation_r=trail_r 联动）
python scripts/optimize_event_execution.py \
  --strategy me-long \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT \
  --start-date 2025-06-01 --end-date 2026-03-01 \
  --sym-r 1.0:0.5:4.0 \
  --promote    # 自动写回 config/strategies/me-long/archetypes/execution.yaml
```

---

## 七、DEPLOY（config → live）

> 将研究确认的 `config/strategies/` 部署到 `live/highcap/config/strategies/`

### 7.1 查看差异（不部署）

```bash
# 查看所有策略 config/ vs live/ 差异
python scripts/deploy_config_to_live.py --diff

# 只看某个策略
python scripts/deploy_config_to_live.py --diff --strategy me-long
```

### 7.2 执行部署

```bash
# 部署指定策略（交互确认）
python scripts/deploy_config_to_live.py --deploy --strategy me-long

# 部署所有策略
python scripts/deploy_config_to_live.py --deploy

# 部署 + 自动 git commit
python scripts/deploy_config_to_live.py --deploy --strategy me-long --git-commit
```

### 7.3 完整研究→部署流程

```bash
# Step 1: 运行研究
mlbot pipeline run --all

# Step 2: 查看决策（应显示 ADOPT）
mlbot pipeline list --all

# Step 3: 查看配置差异
python scripts/deploy_config_to_live.py --diff

# Step 4: 部署
python scripts/deploy_config_to_live.py --deploy --git-commit
```

---

## 八、手动分步执行（调试用）

> `mlbot pipeline run` 已自动完成以下所有步骤，仅在需要单独调试时手动执行

```bash
# 变量设置
STRATEGY="me-long"
CONFIG="config/strategies/me-long"
TIMEFRAME="60T"
SYMBOLS="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT"
START="2024-01-01"
END="2026-03-01"
HOLDOUT="2025-09-01"
GATE_DIR="results/research_history/me-long/<timestamp>/results"
```

**Step 2: Prepare（生成 features_labeled.parquet）**
```bash
mlbot train final --no-docker --prepare-only \
  --config ${CONFIG} \
  --features ${CONFIG}/features.yaml \
  --labels ${CONFIG}/labels_rr_extreme.yaml \
  --symbol ${SYMBOLS} --timeframe ${TIMEFRAME} \
  --data-path data/parquet_data \
  --start-date ${START} --end-date ${END} \
  --holdout-start-date ${HOLDOUT} --holdout-end-date ${END} \
  --seed 42 --non-deterministic
```

**Step 2.5: SHAP 特征筛选**
```bash
python scripts/shap_feature_selection.py \
  --logs ${GATE_DIR}/features_labeled.parquet \
  --strategy ${STRATEGY} \
  --strategies-root config/strategies \
  --pipeline-config config/research_pipeline.yaml \
  --output ${GATE_DIR}/shap \
  --promote
```

**Step 3: Prefilter**
```bash
python scripts/analyze_archetype_feature_stratification.py \
  --logs ${GATE_DIR}/features_labeled.parquet \
  --strategy ${STRATEGY} \
  --meta-algorithm \
  --features-prefilter ${CONFIG}/features_prefilter.yaml \
  --config ${CONFIG} \
  --promote
```

**Step 5: Gate Train**
```bash
mlbot train final --no-docker \
  --config ${CONFIG} \
  --features ${CONFIG}/features_gate_shap.yaml \
  --labels ${CONFIG}/labels_rr_extreme.yaml \
  --archetype-prefilter ${CONFIG}/archetypes/prefilter.yaml \
  --symbol ${SYMBOLS} --timeframe ${TIMEFRAME} \
  --data-path data/parquet_data \
  --start-date ${START} --end-date ${END} \
  --holdout-start-date ${HOLDOUT} --holdout-end-date ${END} \
  --seed 42 --non-deterministic
```

**Step 5b: Gate Optimize**
```bash
python scripts/optimize_gate_unified.py \
  --strategy ${STRATEGY} \
  --strategies-root config/strategies \
  --logs ${GATE_DIR}/logs_gated.parquet \
  --output ${GATE_DIR}/gate_optimization.json \
  --gate-path ${CONFIG}/gate_draft.yaml \
  --promote
```

**Step 9: 向量回测（信号质量评估）**
```bash
python scripts/backtest_execution_layer.py \
  --logs ${GATE_DIR}/logs_gated.parquet \
  --strategy ${STRATEGY} \
  --strategies-root config/strategies \
  --test-start ${HOLDOUT} --test-end ${END} \
  --simple-execution
  # --simple-execution：固定 SL=1.5R / TP=3R / 50bar，中性评估信号质量
  # 去掉 --simple-execution：使用 execution.yaml 真实 trailing stop 配置
```

---

## 九、配置文件说明

### 时间窗口（config/research_pipeline.yaml）

```yaml
dates:
  start_date: "2024-01-01"   # 训练起点（覆盖 2024 牛市 + 2025 调整）
  holdout_months: 6          # Holdout 窗口 = end_date 往前 6 个月
```

### 策略 Timeframe

| 策略                 | Timeframe | Feature Store |
| -------------------- | --------- | ------------- |
| bpc-long / bpc-short | 240T (4H) | 与 fer 共享   |
| fer-long / fer-short | 240T (4H) | 与 bpc 共享   |
| me-long / me-short   | 60T (1H)  | 独立          |
| lv                   | 15T       | 独立          |

### 实验目录结构

```
results/research_history/
└── {strategy}/
    └── {YYYYMMDD_HHMMSS}/
        ├── report.json          # 实验报告
        ├── strategies/          # config 快照
        └── results/
            ├── logs_gated.parquet
            ├── trading_map_{strategy}.html
            └── trading_map_{strategy}_exec.html
```
