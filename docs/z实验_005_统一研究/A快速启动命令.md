# 快速启动命令 — 研究 Pipeline 工作流

> 更新时间: 2026-04-10  
> 策略配置目录: `config/strategies/{bpc,fer,me}`（**短名**；管线里 `strategies.*` 的键与目录名一致，不再使用 `bpc-long`、`fer-short-120T` 等旧目录名）  
> **周期 `timeframe`**: 一律以各策略 `meta.yaml` 里的 `strategy.timeframe` 为准；管线 YAML 中 **`strategies.<name>.timeframe` 可不写**（`auto_research_pipeline` 以 meta 解析为准；若写了且与 meta 不一致会触发 ConfigCheck 告警）。  
> 典型信号栈（以各策略 `archetypes/*.yaml` 为准）: Prefilter → Gate → Entry Filter（若有）→ Direction → Evidence / Execution / 事件回测  
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

> Feature Store 是模型训练的输入，**在特征代码、依赖或 `feature_dependencies.yaml` 变更后需重建或让管线触发增量构建**。  
> `--config` 指向策略根目录；`--timeframe` **须与对应策略 `meta.yaml` 的 `strategy.timeframe` 一致**。  
> 当前仓库约定：**BPC / FER / ME 默认均为 `120T`**（若你改过 `meta.yaml`，以文件为准）。  
> `--warmup-months 6` 为特征预热窗，不计入训练统计。

```bash
# BPC
mlbot feature-store build --no-docker \
  --config config/strategies/bpc \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-groups highcap \
  --timeframe 120T \
  --start-date 2023-01-01 --end-date 2026-03-01 \
  --warmup-months 6

# FER
mlbot feature-store build --no-docker \
  --config config/strategies/fer \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-groups highcap \
  --timeframe 120T \
  --start-date 2023-01-01 --end-date 2026-03-01 \
  --warmup-months 6

# ME
mlbot feature-store build --no-docker \
  --config config/strategies/me \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-groups highcap \
  --timeframe 120T \
  --start-date 2023-01-01 --end-date 2026-03-01 \
  --warmup-months 6
```

> 不同策略若命中同一 feature layer（hash 相同），构建会复用/跳过已完成月份；**新增 FER 锚点列等改动后**，应对相关策略重跑 build 或走完整管线以刷新 layer。

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
# 全策略串行（以所选管线 YAML 的 strategies.* 为准）+ PCM 联合回测 + 自动 ADOPT
mlbot pipeline run --all --config config/prod_train_pipeline_2h.yaml
```

参数说明：
- `--all`：依次跑当前配置里声明的每个 `strategies.<name>`，多策略时含 PCM 联合仲裁（单策略 YAML 会关闭 `pcm_eval` 等）
- `--end-date 2026-03-01`：手动指定数据截止日期（默认自动检测最新数据）
- `--no-adopt`：只保存实验结果，不自动写回 config（需手动 `pipeline adopt`）
- `--dry-run`：打印所有命令但不执行（检查参数用）
- `--skip-shap`：跳过 SHAP 特征筛选（快速迭代调试用，正式研究不建议）
- `--event-backtest`：训练完后自动跑事件回测 execution 优化（sym-r grid search）
- `--event-sym-r 1.0:0.5:4.0`：execution 优化 sym-r 搜索范围（配合 `--event-backtest`）
- `--stage`：分层运行（见下方 4.2.1）

### 4.2 单策略研究（调试）

```bash
# 只跑一个策略（与主 2H 管线一致）
mlbot pipeline run --strategy me --config config/prod_train_pipeline_2h.yaml
mlbot pipeline run --strategy fer --config config/prod_train_pipeline_2h.yaml
mlbot pipeline run --strategy bpc --config config/prod_train_pipeline_2h.yaml

# 快速验证（跳过 SHAP）
mlbot pipeline run --strategy fer --config config/prod_train_pipeline_2h.yaml --skip-shap
# --skip-shap 的作用：跳过 Walk-Forward SHAP 特征筛选步骤，直接复用上次缓存的 features_gate_shap.yaml，省约 5~10 分钟。有没有必要——取决于这次跑的目的：
# 情况	用不用 --skip-shap
# 只改了 prefilter（当前情况）	✅ 可以跳过，特征集没变
# 改了 features_gate_*.yaml（候选特征变了）	❌ 必须重跑 SHAP
# 改了训练时间窗口 / 数据范围	❌ 重跑，SHAP 结果会不同
# 快速验证 prefilter 多算法效果	✅ 跳过，节省时间

# 精细模式（1min bar 执行层）
mlbot pipeline run --strategy bpc --config config/prod_train_pipeline_2h.yaml --use-1min
```

> 单策略且配置里仅一条 `strategies.*` 时，通常不跑 PCM 联合；ADOPT 仅基于该策略指标。专用 **fer/bpc/me-only** 管线见下表。

### 4.2.1 Stage 分层运行（快速迭代）

```bash
# 默认：完整管线
mlbot pipeline run --all --config config/prod_train_pipeline_2h.yaml --stage full

# 只跑单策略分层（会按层前置依赖执行并在该层停止）
mlbot pipeline run --strategy fer --config config/prod_train_pipeline_2h.yaml --stage prefilter
mlbot pipeline run --strategy fer --config config/prod_train_pipeline_2h.yaml --stage gate
mlbot pipeline run --strategy fer --config config/prod_train_pipeline_2h.yaml --stage entry_filter

# 只跑 execution 参数网格（不跑事件回测）
mlbot pipeline run --all --config config/prod_train_pipeline_2h.yaml --stage execution_opt

# 只跑事件回测（含 execution 优化 + 地图）
mlbot pipeline run --all --config config/prod_train_pipeline_2h.yaml --stage event_backtest

# 只跑 PCM 联合回测
mlbot pipeline run --all --config config/prod_train_pipeline_2h.yaml --stage pcm_joint

# 只跑 PCM slot 网格
mlbot pipeline run --all --config config/prod_train_pipeline_2h.yaml --stage pcm_slot_grid

# 仅跑慢变量快照（训练到 entry_filter 停止，并生成 slow snapshot manifest）
mlbot pipeline run --all --config config/prod_train_pipeline_2h.yaml --stage slow_snapshot

# 仅复盘某一个月快变量（默认: 前3个月调阈值, 回测当月）
mlbot pipeline run --all --config config/prod_train_pipeline_2h.yaml --stage fast_month --month 2025-07

# 按月滚动模拟（从 holdout_start 到 end_date 自动逐月）
# - slow_realistic: 每季度更新慢变量结构（过去12个月），每月前3个月调阈值，再测当月
# - turbo_fixed_features: 固定特征，不做特征搜索；每月前3个月调阈值，再测当月
mlbot pipeline run --all --config config/prod_train_pipeline_2h.yaml --stage rolling_sim

# ── 常用专用管线（output.history_dir 互不覆盖）──
# FER-only / turbo / 阈值链 + 事件回测 R 网格（历史目录 results/fer/turbo-rolling-sim）
mlbot pipeline run --all --config config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_fer_only.yaml --stage rolling_sim

# 同上：只复盘 2024-07～09（快变量月，适合调 prefilter/entry/gate）
mlbot pipeline run --all --config config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_fer_only.yaml --stage fast_month --month 2024-07,2024-08,2024-09 --skip-shap 2>&1 | tee log.fer.txt

# BPC-only / turbo（results/bpc/turbo-rolling-sim）
mlbot pipeline run --all --config config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_bpc_only.yaml --stage rolling_sim

# ME-only / turbo（results/me/turbo-rolling-sim）
mlbot pipeline run --all --config config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_me_only.yaml --stage rolling_sim

# BPC-only / 慢模式 / 近似季度结构 + 月度快变量（results/bpc/slow-rolling-sim）
mlbot pipeline run --all --config config/prod_train_pipeline_2h_slow_bpc_only.yaml --stage rolling_sim

# ME-only / 慢模式（results/me/slow-rolling-sim）
mlbot pipeline run --all --config config/prod_train_pipeline_2h_slow_me_only.yaml --stage rolling_sim

# rolling_sim 现支持跨月仓位续跑：
# - 中间月份：月末保留未平仓并写出 end_state.json
# - 下个月：自动加载上月 end_state.json
# - 最后一个月：自动强平收口，便于汇总总收益
```

`rolling` 模式配置示例（`config/prod_train_pipeline_2h.yaml`）：

```yaml
rolling:
  mode: slow_realistic  # slow_realistic | turbo_fixed_features | legacy
  windows:
    calibration_months: 3
    structure_lookback_months: 12
  slow_realistic:
    cadence_months: 3
    triggered_retrain_enabled: true
  turbo_fixed_features:
    fixed_strategies_root: config/strategies
    disable_feature_search: true
```

可用 stage：
- `full`
- `prefilter`
- `gate`
- `entry_filter`
- `slow_snapshot`
- `execution_opt`
- `event_backtest`
- `fast_month`
- `rolling_sim`
- `pcm_joint`
- `pcm_slot_grid`

### 4.2.2 当前仓库中的 2H 管线 YAML（速查）

| 文件                                                                         | 用途                                             | `rolling.mode`         | `output.history_dir`（滚动/快月输出根） |
| ---------------------------------------------------------------------------- | ------------------------------------------------ | ---------------------- | --------------------------------------- |
| `config/prod_train_pipeline_2h.yaml`                                         | 主生产向：多策略 + PCM                           | `slow_realistic`       | `results/120T/prod_train_history`       |
| `config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_fer_only.yaml` | FER-only，阈值链 + `execution_opt`，不做特征搜索 | `turbo_fixed_features` | `results/fer/turbo-rolling-sim`         |
| `config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_bpc_only.yaml` | BPC-only turbo                                   | `turbo_fixed_features` | `results/bpc/turbo-rolling-sim`         |
| `config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_me_only.yaml`  | ME-only turbo                                    | `turbo_fixed_features` | `results/me/turbo-rolling-sim`          |
| `config/prod_train_pipeline_2h_slow_bpc_only.yaml`                           | BPC-only 慢模式（约季度结构 + 月度快变量）       | `slow_realistic`       | `results/bpc/slow-rolling-sim`          |
| `config/prod_train_pipeline_2h_slow_me_only.yaml`                            | ME-only 慢模式                                   | `slow_realistic`       | `results/me/slow-rolling-sim`           |

> `fast_month` / `rolling_sim` 的逐月目录在 `history_dir` 下的 `_rolling_sim/<run_ts>/fast_month_YYYY-MM/...`。事件回测产物多为各月 `.../<strategy>/event_backtest_*.json` 与 `event_trades_*.csv`。

新增辅助命令：

```bash
# 查看某次 rolling_sim 的 side 状态摘要
mlbot pipeline report-side-state --run-id 20260326_120001 --config config/prod_train_pipeline_2h.yaml

# 查看某次 rolling_sim 在指定月份的 PCM 候选池明细
mlbot pipeline debug-pcm-candidates --run-id 20260326_120001 --month 2025-07 --config config/prod_train_pipeline_2h.yaml
```

### 4.3 输出产物

**主 2H 管线**（`prod_train_pipeline_2h.yaml`）默认写入：

`results/120T/prod_train_history/<strategy>/<YYYYMMDD_HHMMSS>/`

```
results/120T/prod_train_history/me/20260313_234448/
├── report.json                          # 实验报告 + ADOPT 决策
├── strategies/                          # 训练后的 gate/prefilter 等快照
└── results/
    ├── logs_gated.parquet               # 经 gate 过滤的信号（核心输出）
    ├── trading_map_*.html               # 交易地图
    └── shap/                            # SHAP 特征分析（未 --skip-shap 时）
```

**专用管线**（FER/BPC/ME-only）以各自 YAML 的 `output.history_dir` 为准，例如 FER turbo：`results/fer/turbo-rolling-sim/`。若仍存在旧路径 `results/research_history/...`，为历史实验遗留，新跑以当前 `history_dir` 为准。

---

## 五、实验管理

### 5.1 列出历史实验

```bash
# 列出单策略所有历史实验（含 Sharpe / WinRate / ADOPT 决策）
mlbot pipeline list --strategy me

# 列出全部策略（与 CLI 实现一致；若需按目录过滤可结合 results/*/）
mlbot pipeline list --all
```

### 5.2 手动采纳实验

```bash
# 采纳指定时间戳的实验（将该实验的 config 写回 config/strategies/<name>/）
mlbot pipeline adopt 20260313_234448 --strategy me
```

### 5.3 对比两次实验

```bash
# 对比两次实验的 archetype 配置差异
mlbot pipeline diff 20260310_120000 20260313_234448 --strategy me
```

### 5.4 删除历史实验

```bash
# 预览（--dry-run 先看会删哪些）
mlbot pipeline delete --strategy me --status error --dry-run

# 按状态批量删除
mlbot pipeline delete --strategy me --status error

# 预览：批量删除各策略 ERROR 实验（短名 bpc / fer / me）
for s in bpc fer me; do
  mlbot pipeline delete --strategy "$s" --status error --dry-run
done

# 执行：批量删除
for s in bpc fer me; do
  mlbot pipeline delete --strategy "$s" --status error
done

# 删除指定时间戳
mlbot pipeline delete --strategy me --timestamp 20260310_120000

# 删除全部历史实验（谨慎）
mlbot pipeline delete --strategy me --all
```

---

## 六、事件回测（Event Backtest）

> 用真实 1min bar 逐笔触发信号，完全模拟实盘时序，验证 execution 参数效果  
> 输出：交易地图 HTML + 交易明细 CSV

### 6.1 对实验运行事件回测

```bash
# 对最新实验运行事件回测（无 execution 优化）
mlbot pipeline event-backtest --strategy me bpc --start-date 2024-01-01 --end-date 2026-03-01

python scripts/event_backtest.py \
  --strategy me,bpc,fer \
  --start-date 2024-01-01 \
  --end-date 2026-03-01 \
  --strategies-root config/strategies \
  --data-path data/parquet_data \
  --fast
# 对指定实验运行（--hash 指定时间戳）
mlbot pipeline event-backtest --strategy me --hash 20260313_234448

# 同时做 execution 参数 grid search 优化（推荐 ME 策略）

mlbot pipeline event-backtest \
  --strategy me \
  --hash 20260313_234144 \
  --sym-r 1.0:0.5:4.0 \
  --promote  

# 先导出向量回测的交易明细
python scripts/backtest_execution_layer.py \
  --logs results/120T/prod_train_history/me/20260313_234144/results/logs_gated.parquet \
  --strategy me --strategies-root config/strategies \
  --test-start 2025-09-01 --test-end 2026-03-01 \
  --simple-execution --export /tmp/vector_trades_me-short.csv

# 在事件地图上叠加蓝圈对比向量回测
python scripts/event_backtest.py \
  --strategy bpc,fer,me \
  --start-date 2025-09-01 --end-date 2026-03-01 \
  --strategies-root config/strategies \
  --trading-map /tmp/event_map_compare.html \
  --compare-trades /tmp/vector_trades_me-short.csv  

python scripts/event_backtest.py \
  --strategy bpc,fer,me \
  --start-date 2025-12-01 \
  --end-date 2026-03-01 \
  --data-path data/parquet_data \
  --strategies-root config/strategies \
  --output results/prod_train_history/pcm_event_120T_latest.json \
  --export results/prod_train_history/pcm_event_120T_latest_trades.csv \
  --trading-map results/prod_train_history/event_map_compare.html
```

### 6.1.1 Slot 网格回测（推荐，管线内置）

> 不再手动改 `constitution.yaml`。  
> 直接在 `config/prod_train_pipeline_2h.yaml` 配置 `pcm_slot_grid`（cases + 评分惩罚 + plateau），
> `mlbot pipeline run --all` 会在 Step 9.5 后自动执行 Step 9.6 的 slot 网格 PCM 对比。

```bash
# 运行全策略（会自动执行 PCM Slot Grid）
mlbot pipeline run --all --config config/prod_train_pipeline_2h.yaml

# 仅运行 PCM Slot Grid（不重跑训练/筛选，最快）
mlbot pipeline run --all --config config/prod_train_pipeline_2h.yaml --stage pcm_slot_grid
```

输出文件（在首个策略本次实验目录）：
- `pcm_slot_grid_report.json`：各 case 指标 + 综合 score + 推荐 case
- `pcm_slot_grid_report.md`：可读表格版（便于复盘）
- `pcm_slot_grid_<case>.json`：每个 case 的 PCM 事件回测明细

建议重点对比：
- `sharpe_daily`
- `total_r`
- `max_drawdown_r`
- `slot_full_rate`（=`reject_pcm_slot_full / total_signals_checked`）
- `per_archetype.*.total_r`

参数说明：
- `--hash`：实验时间戳，不填则自动使用最新实验
- `--sym-r start:step:end`：开启对称 execution 优化，`initial_r=activation_r=trail_r` 联动搜索
- `--promote`：将最优参数写回实验目录的 `execution.yaml`
- `--fast` / `--no-fast`：快速模式（按策略主周期 K 线，如 120T；默认开启）/ 精细模式（1min bar）

输出文件（保存到实验的 `results/` 目录）：
- `trading_map_{strategy}_event.html`：事件回测交易地图
- `event_trades_{strategy}.csv`：交易明细
- `event_exec_opt.json`：execution 优化结果（有 `--sym-r` 时生成）

### 6.2 手动运行事件回测脚本

```bash
# 基础用法（默认用研究数据 data/parquet_data，最近 180 天）
python scripts/event_backtest.py \
  --strategy me \
  --strategies-root config/strategies \
  --days 180

# 指定日期范围 + 导出交易明细
python scripts/event_backtest.py \
  --strategy me,bpc \
  --strategies-root config/strategies \
  --data-path data/parquet_data \
  --start-date 2025-06-01 --end-date 2026-03-01 \
  --trading-map results/trading_map_me_bpc.html \
  --export results/event_trades.csv

# 跨月续跑（示例：2025-07 结束状态 -> 2025-08 恢复）
python scripts/event_backtest.py \
  --strategy me \
  --strategies-root config/strategies \
  --data-path data/parquet_data \
  --start-date 2025-07-01 --end-date 2025-07-31 \
  --output /tmp/event_2025_07.json \
  --dump-end-state /tmp/end_state_2025_07.json \
  --keep-open-positions

python scripts/event_backtest.py \
  --strategy me \
  --strategies-root config/strategies \
  --data-path data/parquet_data \
  --start-date 2025-08-01 --end-date 2025-08-31 \
  --resume-state /tmp/end_state_2025_07.json \
  --output /tmp/event_2025_08.json \
  --dump-end-state /tmp/end_state_2025_08.json \
  --keep-open-positions
```

### 6.3 Execution 参数 Grid Search（手动）

```bash
# ME 策略对称优化（initial_r=activation_r=trail_r 联动）
python scripts/optimize_event_execution.py \
  --strategy me \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT \
  --start-date 2025-06-01 --end-date 2026-03-01 \
  --sym-r 1.0:0.5:4.0 \
  --promote    # 自动写回 config/strategies/me/archetypes/execution.yaml
```

---

## 七、DEPLOY（config → live）

> 将研究确认的 `config/strategies/` 部署到 `live/highcap/config/strategies/`

### 7.1 查看差异（不部署）

```bash
# 查看所有策略 config/ vs live/ 差异
python scripts/deploy_config_to_live.py --diff

# 只看某个策略
python scripts/deploy_config_to_live.py --diff --strategy me
```

### 7.2 执行部署

```bash
# 部署指定策略（交互确认）
python scripts/deploy_config_to_live.py --deploy --strategy me

# 部署所有策略
python scripts/deploy_config_to_live.py --deploy

# 部署 + 自动 git commit
python scripts/deploy_config_to_live.py --deploy --strategy me --git-commit
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
STRATEGY="me"
CONFIG="config/strategies/me"
TIMEFRAME="120T"
SYMBOLS="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT"
START="2024-01-01"
END="2026-03-01"
HOLDOUT="2025-09-01"
GATE_DIR="results/120T/prod_train_history/me/<timestamp>/results"
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

### 策略 Timeframe（当前仓库）

| 策略目录 `config/strategies/*` | 默认 `meta.yaml` | 说明                            |
| ------------------------------ | ---------------- | ------------------------------- |
| `bpc`                          | `120T`           | Breakout–Pullback–Continuation  |
| `fer`                          | `120T`           | Failure / Exhaustion / Reversal |
| `me`                           | `120T`           | Momentum Expansion              |

改周期：只改对应策略的 `meta.yaml` 中 `strategy.timeframe`，并重建 Feature Store / 重跑管线。

### 实验目录结构（当前）

主配置 `prod_train_pipeline_2h.yaml`：

```
results/120T/prod_train_history/
└── {bpc|fer|me}/
    └── {YYYYMMDD_HHMMSS}/
        ├── report.json
        ├── strategies/
        └── results/
            ├── logs_gated.parquet
            ├── trading_map_*.html
            └── shap/   # 未 skip 时
```

专用管线见 **§4.2.2** 的 `output.history_dir`（如 `results/fer/turbo-rolling-sim`）。

---

> **存档说明**：下文「# 滚动测试…」起至文档末尾，多为 **历史多周期（60T/120T/240T）与旧策略键**（如 `fer-short-120T`）的实验笔记，**与当前短名 `bpc` / `fer` / `me` 及默认 120T 不一定一致**。复现请改用 §一～§四的命令，并以 `meta.yaml` 为准。

# 滚动测试 先用 6,3 做滚动多窗口验证 要怎么做

可以按下面这个最小流程做，不改代码、只用现有命令。

1) 固定配置不动
先确认 config/research_pipeline.yaml 里就是：

holdout_months: 6
validation_months: 3
并且暂时不要再改其他策略参数（保证实验可比）。

2) 选一组滚动 end-date
建议先跑 6 个窗口（每月一个）：

2025-10-01
2025-11-01
2025-12-01
2026-01-01
2026-02-01
2026-03-01
3) 批量运行（关键：加 --no-adopt）
这样每次只产实验结果，不覆盖生产配置。
```bash

for d in 2025-10-01 2025-11-01 2025-12-01 2026-01-01 2026-02-01 2026-03-01; do
  mlbot pipeline run --strategy fer-short-60T --end-date "$d" --no-adopt
done

for d in 2025-10-01 2025-11-01 2025-12-01 2026-01-01 2026-02-01 2026-03-01; do
  mlbot pipeline run --strategy fer-short-120T --end-date "$d" --no-adopt
done

for d in 2025-10-01 2025-11-01 2025-12-01 2026-01-01 2026-02-01 2026-03-01; do
  mlbot pipeline run --strategy fer-short-240T --end-date "$d" --no-adopt
done

# -------------------------------------------
for d in 2025-10-01 2025-11-01 2025-12-01 2026-01-01 2026-02-01 2026-03-01; do
  mlbot pipeline run --strategy bpc-short-60T --end-date "$d" --no-adopt
done

for d in 2025-10-01 2025-11-01 2025-12-01 2026-01-01 2026-02-01 2026-03-01; do
  mlbot pipeline run --strategy bpc-short-120T --end-date "$d" --no-adopt
done

for d in 2025-10-01 2025-11-01 2025-12-01 2026-01-01 2026-02-01 2026-03-01; do
  mlbot pipeline run --strategy bpc-short-240T --end-date "$d" --no-adopt
done
# -------------------------------------------

for d in 2025-10-01 2025-11-01 2025-12-01 2026-01-01 2026-02-01 2026-03-01; do
  mlbot pipeline run --strategy me-short-60T --end-date "$d" --no-adopt
done

for d in 2026-01-01 2026-02-01 2026-03-01; do
  mlbot pipeline run --strategy me-short-120T --end-date "$d" --no-adopt
done

for d in 2026-01-01 2026-02-01 2026-03-01; do
  mlbot pipeline run --strategy me-short-240T --end-date "$d" --no-adopt
done


mlbot pipeline run --strategy bpc-short-60T
mlbot pipeline run --strategy bpc-short-120T
mlbot pipeline run --strategy bpc-short-240T

mlbot pipeline run --strategy me-short-60T
mlbot pipeline run --strategy me-short-120T
mlbot pipeline run --strategy me-short-240T

mlbot pipeline run --strategy fer-short-60T
mlbot pipeline run --strategy fer-short-120T
mlbot pipeline run --strategy fer-short-240T
```
4) 看每个窗口结果
先快速看列表：

python scripts/auto_research_pipeline.py --strategy bpc --list
再做一次聚合（看中位数/稳定性）：

```bash
python - <<'PY'
import json, pathlib, statistics
root = pathlib.Path("results/research_history/me-short-60T")
rows = []
for d in sorted([p for p in root.iterdir() if p.is_dir()])[-20:]:
    rp = d / "report.json"
    if not rp.exists(): 
        continue
    r = json.loads(rp.read_text())
    bt = r.get("backtest_metrics", {})
    rows.append((d.name, bt.get("sharpe_per_trade"), bt.get("total_trades")))
rows = [x for x in rows if x[1] is not None and x[2] is not None]
print("\n".join(f"{t}  sharpe={s:.4f}  trades={n}" for t,s,n in rows))
if rows:
    ss = [x[1] for x in rows]
    nn = [x[2] for x in rows]
    print(f"\nmedian_sharpe={statistics.median(ss):.4f}")
    print(f"positive_ratio={sum(s>0 for s in ss)/len(ss):.1%}")
    print(f"median_trades={statistics.median(nn):.1f}")
PY
```
5) 决策标准（建议）
median_sharpe > 0
positive_ratio >= 70%
median_trades >= 80（你可按 FER 调成 60/100）
如果这三条达标，再考虑进入 deploy；不达标再讨论 1H 分支实验。

6) 稳定性目标（推荐，避免“Sharpe 提升但交易数失控”）
在 `config/research_pipeline.yaml` 的每个策略里，给 `kpi_gates.prefilter` 和
`kpi_gates.entry_filter` 配置软目标区间：

```yaml
target_trades_min: 35
target_trades_max: 220
trade_penalty_low: 0.002
trade_penalty_high: 0.001
stability_penalty: 0.0
```

说明：
- 不是硬拦截；仍然自动化搜索，但会优先选择“Sharpe + 合理交易密度”的方案
- `trade_penalty_low` 惩罚交易过少，`trade_penalty_high` 惩罚交易过多
- 日志里会显示 `Score`（`Sharpe - trade penalties`）并按 `Score` 选优

## 稳定性实验
配置
dates:
  start_date: "2024-01-01" # Regime窗口起点: 覆盖2024牛市+震荡+2025熊市转换, 贴近当前市场
  # start_date: "2023-01-01"   # 全量历史模式 (备用)
  holdout_months: 6 # OOS 总窗口 (Val + Test)
  validation_months: 3 # OOS 前 N 个月用于 Gate 调阈值; 剩余 = 纯 Test (0=不分离)

20260315_161028  sharpe=0.0000  trades=0
20260315_162206  sharpe=0.0000  trades=0
20260315_162746  sharpe=0.9421  trades=164
20260315_165506  sharpe=0.1246  trades=954
20260315_165921  sharpe=0.0000  trades=0
20260315_170111  sharpe=0.0486  trades=559
20260315_171545  sharpe=0.0000  trades=0
20260315_171829  sharpe=0.3623  trades=88
20260315_191412  sharpe=0.3224  trades=21
20260315_192607  sharpe=0.0000  trades=0
20260315_192721  sharpe=0.3224  trades=21
20260315_203309  sharpe=0.3224  trades=21
20260315_204634  sharpe=0.4062  trades=62
20260315_210402  sharpe=0.0486  trades=559
20260315_211633  sharpe=0.6021  trades=27
20260315_211913  sharpe=-0.0323  trades=456
20260315_212211  sharpe=0.3410  trades=744
20260315_212512  sharpe=0.3801  trades=1431
20260315_212957  sharpe=0.3370  trades=182
20260315_213324  sharpe=0.4062  trades=62

median_sharpe=0.3224
positive_ratio=70.0%
median_trades=62.0

就你贴出来这组结果看，6,3 + 4H 是“可用但不够稳”的状态。

按你文档里的判定线：

median_sharpe=0.3224 ✅（>0）
positive_ratio=70.0% ✅（刚好踩线）
median_trades=62.0
按通用门槛 >=80 ❌
按你给 FER 放宽门槛 >=60 ✅
所以结论分两层：

研究阶段：合理，可以继续作为主配置。
直接 deploy 决策：偏勉强，建议先再加一层稳定性约束（比如“0 trades 窗口占比”）。
你这批里 0 trades 的窗口不少（我数到约 5/20），说明仍有“有时能跑得很好、有时直接塌掉”的不稳定性。
建议你把上线门槛改成 FER 专用版：

median_sharpe > 0
positive_ratio >= 70%
median_trades >= 60
nonzero_trade_ratio >= 80%（新增）
这样 6,3 + 4H 就更客观了。

## 我后面做12,6的滚动测试，还是加上sr约束做，还是切到1h做？什么顺序才能把头绪理清楚

推荐顺序（先做什么）
先在 4H + 6,3 上加 SR 约束（你现在主线）
再做 4H + 12,6（同一套 SR 约束）
最后才开 1H 分支实验（只做 FER，不要三策略一起改）
为什么这个顺序最清晰
你现在最大问题是“语义锚不稳”（有时追波动爆发），不是先天 timeframe 不够。
先加 SR 约束能验证：不换频率，仅靠语义锚能不能稳住。
如果 4H 两种窗口都稳了，就没必要急着 1H。
如果 4H 仍不稳，再上 1H，才能明确是“频率问题”不是“语义问题”。
具体实验矩阵（建议）
A 组：4H + 6,3 + SR约束（滚动）
B 组：4H + 12,6 + SR约束（滚动）
C 组：1H + 6,3 + 同样SR约束（小规模滚动）
每组只改一个变量，不要混着改。

你该先做的下一步
先跑 A 组。
A 组达标后再跑 B；只有 A/B 都不理想才上 C。

一句话：先“锁语义”，再“拉窗口”，最后“改频率”。

## Locked 阈值调优工具（新）

目标：保持 locked 语义特征不变，只调阈值；每个窗口独立跑 `Val/Test`。

1) 单窗口快速扫描（先小网格）：

```bash
python scripts/tune_locked_prefilter_thresholds.py \
  --strategy fer-short \
  --end-dates 2026-03-01 \
  --fer-lower-values 0.0,0.05 \
  --fer-upper-values 0.30,0.35,0.40 \
  --sr-min-values 0.45,0.55,0.65 \
  --dist-max-values 0.8,1.2,1.6 \
  --max-cases 12 \
  --skip-shap
```

2) 多窗口滚动评分（推荐）：

```bash
python scripts/tune_locked_prefilter_thresholds.py \
  --strategy fer-short \
  --end-dates 2025-10-01,2025-11-01,2025-12-01,2026-01-01,2026-02-01,2026-03-01 \
  --fer-lower-values 0.0,0.05 \
  --fer-upper-values 0.25,0.35,0.45 \
  --sr-min-values 0.45,0.55,0.65 \
  --dist-max-values 0.8,1.2,1.6 \
  --min-trades-target 60 \
  --trade-penalty 0.002
```

输出目录：`results/locked_tuning/fer-short/<timestamp>/`
- `summary.csv`: 每组参数的聚合分数
- `summary.json`: 每个窗口明细（含 run_id/report 路径）

3) 主管线内置自动调优（已启用，且带缓存）：

- 当 `prefilter.yaml` 存在 `locked: true` 规则时，`mlbot pipeline run` 会自动触发阈值调优。
- 同一窗口已调优过则命中缓存并跳过调优（cache：`results/locked_tuning/cache/`）。
- 如需临时禁用自动调优：在命令里加 `--disable-auto-locked-tuning`。
