# ML Trading Bot（中文）

**English**: [README.md](README.md)  
**文档索引**: [docs/README.md](docs/README.md)

## 框架目的

核心地位的是启发式规则（domain heuristics）：
比如“订单流真空不要做突破”、“BPC pullback 太浅是假形态”，这些本质上是你的交易哲学
框架里所有的树模型、lift 曲线、Failure Analysis，都是在帮你验证这些启发式在历史数据上的有效性，而不是要推翻它们

大概率启发式规则本身（比如订单流/结构/VP 场景）是对的，框架的职责是在历史上反证“在哪些子场景成功率明显掉下去”，然后把这些场景排除掉或降权。

## 底层哲学
[系统哲学基础](docs/archive/architecture/系统哲学基础.md)
[顶级量化团队.md](docs/archive/architecture/顶级量化团队.md)
[一道一宿速正道](docs/archive/architecture/一道一宿速正道.md)

**Alpha不是收集的，是雕刻的。**

本仓库包含因子研究、模型训练、回测与数据管道的生产就绪组件。**本 README 保持尽量短**：只提供"命令 + 推荐流程 + 入口文档链接"。研究解释型内容已迁移到独立文档。

---

## 快速开始

1) 创建虚拟环境（conda、venv 等）并激活  
2) 以可编辑模式安装：

```bash
pip install -e .[dev]
```

3) （可选但推荐）安装 Git pre-commit 钩子：

```bash
make install-hooks
```

4) 查看命令：

```bash
mlbot --help
mlbot analyze --help
mlbot train --help
mlbot diagnose --help
mlbot optimize --help
mlbot data --help
```

---

## 数据管道（下载 → 转换 → 训练）

```bash
mlbot data download \
  --symbols BTCUSDT,ETHUSDT \
  --start-year 2021 \
  --start-month 1

mlbot data convert

# 或一次性跑完
mlbot data pipeline \
  --symbols BTCUSDT,ETHUSDT
```

### Universe 驱动（多币种批量：下载 + 转 parquet）

`pipeline-universe` 会按 universe 配置解析 symbol 列表，并 **下载后立刻转成 Parquet**：

```bash
mlbot data pipeline-universe --no-docker \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-groups highcap \
  --start-year 2023 \
  --start-month 1
```

### 扩展数据：市值（Market Cap）与资金费率（Funding Rate）

#### 1) 市值快照（CoinGecko）

```bash
export COINGECKO_API_KEY='...'

mlbot data update-market-cap \
  --config config/market_cap/market_cap.yaml \
  --max-age-days 7 \
  --no-docker
```

结果默认落盘：
- `data/market_cap/<SYMBOL>.parquet`
- `data/market_cap/market_cap_manifest.json`

#### 2) Binance 资金费率（按月 ZIP → Parquet）

```bash
# 方式 A：Universe 驱动（推荐）
mlbot data download-funding-rate --no-docker \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-groups highcap \
  --start-year 2023 --start-month 1

# 方式 B：手工传 symbol 列表
mlbot data download-funding-rate --no-docker \
  --symbols BTCUSDT,ETHUSDT \
  --start-year 2023 --start-month 1 \
  --progress-every 10
```

结果默认落盘：
- ZIP：`data/funding_rate/zip/`
- Parquet：`data/funding_rate/parquet/`

#### 3) Binance 未平仓合约（OI）

```bash
mlbot data download-open-interest \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-groups highcap \
  --start-year 2023 --start-month 1 --progress-every 1
```

---

## 推荐工作流（当前主线：`mlbot pipeline run`）

> **当前主线不再以 `mlbot train final` 为入口**。研究/回测/阈值调优/Event Backtest 全部走统一管线 `mlbot pipeline run`（底层会调训练/评估的各子步）。
>
> **最全命令手册**（含所有 stage / flag / 跨月续跑 / 稳定性实验）：
> - `docs/z实验_005_统一研究/A快速启动命令.md`（最新，长文档）
> - `docs/z实验_005_统一研究/实施文档_01_2024牛市_5x趋势骑乘.md`（5x 趋势骑乘实验 + turbo 调阈值工作流）

### 活跃策略 & 对应管线 YAML

**当前 `config/strategies/` 顶层活跃目录**（不含 `bad-candidates/`）：

- **经典单腿链**（prefilter / gate / direction、`TradeIntent` 路径）：`bpc`、`tpc`、`me`。
- **多腿独立策略**（自有 inventory；研究 adopt / deploy / 实盘见下文 **§6.1**）：`chop_grid`、`dual_add_trend`。

**`bad-candidates/`**：历史实验与废弃候选（含 `fbf`、`fer`、`msr`、`lv`、`lottery`、以及 **`srb` / `crf` 等当前主树副本**）；日常主线以顶层五目录为准。

| 策略目录          | 语义                                                         | 默认 `meta.yaml` timeframe |
| ----------------- | ------------------------------------------------------------ | -------------------------- |
| `bpc`             | Breakout → Pullback → Continuation（趋势延续）               | `120T`                     |
| `tpc`             | Trend → Pullback → Continuation（趋势回踩）                  | `120T`                     |
| `me`              | Momentum Expansion（动量扩张）                               | `120T`                     |
| `chop_grid`       | 语义 chop + 盒过滤下的小网格段（**多腿**；根配置 `grid.yaml`） | `120T`                     |
| `dual_add_trend`  | 趋势置信 + chop/盒过滤下双腿加仓（**多腿**；根配置 `dual_add.yaml`） | `120T`                     |

**常用 pipeline YAML**（按 **`config/strategies/<策略>/`** 归类；**`mlbot pipeline run --config …`** 仍使用 **`config/` 根目录下** 的 `prod_train_pipeline_*.yaml` 路径）。

> **与 `live/highcap/config/strategies/` 的分工**：实盘镜像里是同名的 **`meta.yaml` / `features.yaml` / `archetypes/` / 根引擎 yaml**（由 `scripts/deploy_config_to_live.py` 同步）；**不会**把 `prod_train_pipeline_*.yaml` 拷到 `live/…/strategies/bpc/`——那些文件是**研究编排入口**，留在 `config/` 根目录便于脚本与 CI 引用。若将来要把某策略的 pipeline 物理挪到 `config/strategies/<slug>/pipelines/`，需同步改所有 `--config` 引用。

#### 全局 / 多策略

| YAML | 用途 | `rolling.mode` |
| ---- | ---- | ---------------- |
| `config/prod_train_pipeline_2h.yaml` | 主生产向：多策略 + PCM 联合回测 | `slow_realistic` |

#### `bpc` — `config/strategies/bpc/`（镜像：`live/highcap/config/strategies/bpc/`）

| YAML | 用途 | `rolling.mode` |
| ---- | ---- | ---------------- |
| `config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_bpc_only.yaml` | turbo：阈值链 + execution 优化，不做特征搜索 | `turbo_fixed_features` |
| `config/prod_train_pipeline_2h_slow_bpc_only.yaml` | 慢模式：季度结构 + 月度快变量 | `slow_realistic` |

#### `tpc` — `config/strategies/tpc/`（镜像：`live/highcap/config/strategies/tpc/`）

| YAML | 用途 | `rolling.mode` |
| ---- | ---- | ---------------- |
| `config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_tpc_only.yaml` | turbo | `turbo_fixed_features` |
| `config/prod_train_pipeline_2h_slow_tpc_only.yaml` | 慢模式 | `slow_realistic` |

#### `me` — `config/strategies/me/`（镜像：`live/highcap/config/strategies/me/`）

| YAML | 用途 | `rolling.mode` |
| ---- | ---- | ---------------- |
| `config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_me_only.yaml` | turbo | `turbo_fixed_features` |
| `config/prod_train_pipeline_2h_slow_me_only.yaml` | 慢模式 | `slow_realistic` |

#### `chop_grid` — `config/strategies/chop_grid/`（镜像：`live/highcap/config/strategies/chop_grid/`）

| YAML | 用途 | `rolling.mode` |
| ---- | ---- | ---------------- |
| `config/prod_train_pipeline_2h_turbo_chop_grid_only.yaml` | 多腿网格 rolling（turbo） | `turbo_fixed_features` |
| `config/prod_train_pipeline_2h_slow_chop_grid_only.yaml` | 多腿慢模式 | `slow_realistic` |

#### `dual_add_trend` — `config/strategies/dual_add_trend/`（镜像：`live/highcap/config/strategies/dual_add_trend/`）

| YAML | 用途 | `rolling.mode` |
| ---- | ---- | ---------------- |
| `config/prod_train_pipeline_2h_turbo_dual_add_trend_only.yaml` | 多腿双腿策略 rolling（turbo） | `turbo_fixed_features` |
| `config/prod_train_pipeline_2h_slow_dual_add_trend_only.yaml` | 多腿慢模式 | `slow_realistic` |

#### `bad-candidates/`（策略树 + 专用管线；非顶层主线）

| YAML | 用途 | `rolling.mode` |
| ---- | ---- | ---------------- |
| `config/prod_train_pipeline_2h_turbo_crf_only.yaml` | CRF / turbo（`bad-candidates/crf`，`box_structure_f`） | `turbo_fixed_features` |
| `config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_srb_only.yaml` | SRB / turbo（`bad-candidates/srb`） | `turbo_fixed_features` |
| `config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_srb_quickstrike_only.yaml` | SRB quickstrike / turbo | `turbo_fixed_features` |
| `config/prod_train_pipeline_2h_slow_srb_only.yaml` | SRB 慢模式 | `slow_realistic` |
| `config/strategies/bad-candidates/pipelines/*.yaml` | 历史 FBF / FER / MSR 等实验编排 | （见各文件） |

> `turbo_fixed_features`：特征集固定，只做阈值链 / execution 优化 / 月度滚动 → **快**。  
> `slow_realistic`：每季度重做结构快照（prefilter/gate 元算法），月度 fast_loop 调阈值 → **稳**。

### 0) 质量闸门（推荐）

```bash
make test-key-features-all
mlbot diagnose feature-contract --no-docker
```

### 1) 数据下载 + Feature Store 构建

```bash
# 一次性拉齐 highcap universe 所有 symbol 的 2H OHLCV（含 1min 原始）
mlbot data pipeline-universe --no-docker \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-groups highcap \
  --start-year 2023 --start-month 1

# 资金费率 + OI（可选但推荐）
mlbot data download-funding-rate --no-docker \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-groups highcap \
  --start-year 2023 --start-month 1

mlbot data download-open-interest \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-groups highcap \
  --start-year 2023 --start-month 1

# Feature Store（按策略构建；新增/修改特征后必须重跑对应策略）
mlbot feature-store build --no-docker \
  --config config/strategies/bpc \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-groups highcap \
  --timeframe 120T \
  --start-date 2023-01-01 --end-date 2026-03-01 \
  --warmup-months 6
```

> 同一 feature layer（hash 相同）会跨策略复用；新增/改特征代码或 `config/feature_dependencies.yaml` 后需要对相关策略重跑 `feature-store build`。

### 2) 研究管线（turbo 快模式：只调阈值，不搜特征）

```bash
# BPC turbo（推荐先跑单月验证，再开全 rolling）
mlbot pipeline run --all \
  --config config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_bpc_only.yaml \
  --stage fast_month --month 2024-09 --skip-shap 2>&1 | tee log.bpc.txt

# 全量 rolling_sim（从 holdout_start 到 end_date 自动逐月）
mlbot pipeline run --all \
  --config config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_bpc_only.yaml \
  --stage rolling_sim --skip-shap 2>&1 | tee log.bpc.txt

# ME / TPC / SRB / CRF 同理，只换 --config
mlbot pipeline run --all \
  --config config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_me_only.yaml \
  --stage rolling_sim --skip-shap

mlbot pipeline run --all \
  --config config/prod_train_pipeline_2h_turbo_crf_only.yaml \
  --stage rolling_sim --skip-shap
```

**常用 stage**（分层调试）：

```
prefilter → gate → entry_filter → slow_snapshot → execution_opt → event_backtest
fast_month   # 仅复盘某个月（默认：前 3 个月调阈值，回测当月）
rolling_sim  # 按月滚动：结构快照 / 阈值调优 / 当月回测 / 月间仓位续跑
pcm_joint    # PCM 联合仲裁
pcm_slot_grid  # Slot 网格（替代手动改 constitution.yaml）
```

### 3) 研究管线（slow 慢模式：季度结构 + 月度阈值 — 上线前完整验证）

```bash
mlbot pipeline run --all \
  --config config/prod_train_pipeline_2h_slow_bpc_only.yaml \
  --stage rolling_sim 2>&1 | tee log.bpc.slow.txt

# 单月复盘（调试用）
mlbot pipeline run --all \
  --config config/prod_train_pipeline_2h_slow_bpc_only.yaml \
  --stage fast_month --month 2024-09
```

### 4) 事件回测（Event Backtest）

> 用真实 1min bar 逐笔触发信号，与实盘时序严格对齐；支持 execution 参数 grid search。

```bash
# A. 走 pipeline（最省事，会自动跑 execution 优化并写回 archetypes/execution.yaml）
mlbot pipeline run --all \
  --config config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_bpc_only.yaml \
  --stage event_backtest --skip-shap

# B. 直接跑单次 event_backtest（最快）
python scripts/event_backtest.py \
  --strategy bpc,me,tpc,srb,crf \
  --start-date 2024-01-01 --end-date 2026-03-01 \
  --strategies-root config/strategies \
  --data-path data/parquet_data \
  --fast

# C. 对已有实验重放（不重训，带交易地图）
mlbot pipeline event-backtest \
  --strategy bpc --hash 20260313_234448 \
  --sym-r 1.0:0.5:4.0 --promote    # 同步网格优化 execution，并写回实验目录
```

### 5) 慢管线产物对比（`slow_candidate_report.py`）

> 当 slow pipeline 跑完后，用这组命令把「每月 Prefilter/Gate/EF 选了什么 / 和 turbo 基线的差异 / 月度 R delta」一张表产出来，**无需重跑**。

```bash
RUN=results/bpc/slow-rolling-sim/_rolling_sim/20260423_223716
BASE=results/bpc/turbo-rolling-sim/_rolling_sim/<turbo_ts>
OUT=results/bpc/slow_candidate_reports/${RUN##*/}
mkdir -p "$OUT"

PYTHONPATH=. python3 scripts/slow_candidate_report.py review \
  --slow-run-dir "$RUN" --baseline-run-dir "$BASE" \
  --strategy bpc --output "$OUT/review.md"

# 其他子命令：manifest / drift / digest / consensus
```

### 6) Adopt & Deploy（研究 → 实盘）

```bash
# 列出历史实验 + 决策（ADOPT/REJECT）
mlbot pipeline list --all
mlbot pipeline list --strategy bpc

# 采纳指定实验（把该实验的 config 写回 config/strategies/<name>/）
mlbot pipeline adopt 20260313_234448 --strategy bpc

# 研究仓 → 实盘 highcap
python scripts/deploy_config_to_live.py --diff --strategy bpc
python scripts/deploy_config_to_live.py --deploy --strategy bpc --git-commit
```

### 6.1) 多腿策略（`chop_grid` / `dual_add_trend`）：配置、研究 adopt、同步实盘、多腿进程

与 BPC 同一套心智：**研究配置**在 `config/strategies/<策略名>/`，根目录 **`grid.yaml`**（chop_grid）或 **`dual_add.yaml`**（dual_add_trend）为**主配置**；**`archetypes/*.yaml`** 为可推广 / 可 adopt 的薄层；**实盘镜像**在 `live/highcap/config/strategies/<策略名>/`（用下方 deploy 同步）。

**1）离线诊断（不跑 pipeline）**

```bash
# chop_grid：语义 chop + 盒过滤 + 网格段回测（读 grid.yaml）
python scripts/diagnose_chop_grid.py \
  --start 2024-01-01 --end 2024-12-31 \
  --symbols BTCUSDT,ETHUSDT --timeframe 2h

# dual_add_trend：趋势段 + 双腿加仓仿真（读 dual_add.yaml；trend 列来自注册特征同一公式）
python scripts/diagnose_dual_add_trend.py \
  --start 2024-01-01 --end 2024-12-31 \
  --symbols BTCUSDT,ETHUSDT --timeframe 2h
# 可选：覆盖 trend 用的多周期回看（默认与 feature_dependencies 中 trend_confidence_f 的 horizons 一致时可不写）
# python scripts/diagnose_dual_add_trend.py ... --trend-return-horizons 3,5,10
```

**2）研究管线 + rolling 产物（便于 adopt）**

多腿 turbo 示例配置：`config/prod_train_pipeline_2h_turbo_chop_grid_only.yaml`（其中 `output.history_dir` 决定结果根目录）。跑完 **`rolling_sim`** 后，流水线会把**最后一月**的 `strategies_calibrated/<策略>/` 拷到：

`{history_dir}/<策略名>/<本次 run 时间戳>/strategies/<策略名>/`

这样 **`mlbot pipeline adopt`** 能按与 BPC 相同的目录约定找到 `strategies/<策略>/archetypes`（多腿 adopt **不**走 BPC 的 locked prefilter/gate 校验，以复制为主）。

**重要：`list` / `adopt` / `diff` 的实验根目录 = 当前 `--config` 里的 `output.history_dir`。**  
若 rolling 用的是 `prod_train_pipeline_2h_turbo_chop_grid_only.yaml`（`history_dir` 在 `results/chop_grid/...`），则 **`mlbot pipeline adopt` 必须带同一 `--config`**；否则 CLI 默认读 `config/research_pipeline.yaml`，会去 `results/research_history/...`，会报「实验不存在」。

```bash
CHOP_CFG=config/prod_train_pipeline_2h_turbo_chop_grid_only.yaml
DUAL_CFG=config/prod_train_pipeline_2h_turbo_dual_add_trend_only.yaml

mlbot pipeline run --strategy chop_grid --config "$CHOP_CFG" \
  --stage rolling_sim --skip-shap

mlbot pipeline run --strategy dual_add_trend --config "$DUAL_CFG" \
  --stage rolling_sim --skip-shap

# 列出 / 采纳须与上面同一 --config
mlbot pipeline list --strategy chop_grid --config "$CHOP_CFG"
mlbot pipeline adopt <时间戳> --strategy chop_grid --config "$CHOP_CFG"

mlbot pipeline list --strategy dual_add_trend --config "$DUAL_CFG"
mlbot pipeline adopt <时间戳> --strategy dual_add_trend --config "$DUAL_CFG"
```

**3）研究仓 → 实盘 highcap（多腿会 diff/deploy 全部 archetypes + 根 engine yaml）**

```bash
python scripts/deploy_config_to_live.py --diff --strategy chop_grid
python scripts/deploy_config_to_live.py --deploy --strategy chop_grid --git-commit

python scripts/deploy_config_to_live.py --diff --strategy dual_add_trend
python scripts/deploy_config_to_live.py --deploy --strategy dual_add_trend --git-commit
```

**4）多腿实盘 / 影子进程（与 `scripts/run_live.py` 分离）**

```bash
# 默认 shadow + parquet 回放；bar 源还可选 websocket / feature-store
python scripts/run_multi_leg_live.py --mode shadow --bar-source parquet --once

# 指定策略与策略 yaml（默认已指向 config/strategies/...）
python scripts/run_multi_leg_live.py \
  --strategies chop_grid \
  --chop-grid-config config/strategies/chop_grid/grid.yaml \
  --once
```

密钥与账户隔离见脚本首屏 docstring（推荐 `MULTI_LEG_BINANCE_FUTURES_*`）；更完整的 live 事件流说明见 `docs/architecture/live_stream/README.md`。

### 7) （可选）最终验收 & 最终训练模型

> 这两条命令对应 **nnmultihead / 传统 tree 最终模型的单次一致性验收**，**不是日常 rolling_sim 的替代**；日常研究用 §2-§5 的 rolling 工作流即可。

```bash
# 6 个月 Holdout 验收（只验收，不调参）
mlbot diagnose holdout-eval \
  --config config/strategies/bpc \
  --symbol BTCUSDT --timeframe 120T \
  --train-start-date 2024-01-01 \
  --holdout-start-date 2025-10-01 \
  --holdout-end-date 2026-03-31 \
  --output-root results/holdout_eval --deterministic --no-docker

# 全窗训练最终模型
mlbot train final \
  --config config/strategies/bpc \
  --symbol BTCUSDT --timeframe 120T \
  --start-date 2024-01-01 --end-date 2026-03-31 \
  --output-root models --deterministic --no-docker
```

### 8) 近期实验 & 分支路径（可选；不含 Nautilus — 已弃用）

#### 8.1 盒子特征（`box_structure_f`）+ CRF 立项

> 基础设施：`src/features/time_series/box_structure_features.py`（因果、滚动，26 列 `box_*`）  
> 诊断脚本（oracle vs causal 对照）：`scripts/diag_consolidation_structure.py --mode {oracle,causal}`  
> 方法与结论文档：
> - `docs/z实验_005_统一研究/box_features_causal_vs_oracle_20260424.md`
> - `docs/z实验_005_统一研究/CRF_CBC_structure_diagnosis_20260424.md`

CRF 策略跑法（box-based 双向均值回归）：

```bash
mlbot feature-store build --no-docker \
  --config config/strategies/crf \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-groups highcap \
  --timeframe 120T \
  --start-date 2023-01-01 --end-date 2026-03-01 --warmup-months 6

mlbot pipeline run --all \
  --config config/prod_train_pipeline_2h_turbo_crf_only.yaml \
  --stage rolling_sim --skip-shap 2>&1 | tee log.crf.txt
```

SRB / BPC / ME 已把 `box_*` 作为 prefilter 草稿（`locked: false`），跑现有 turbo 管线即可生效。

#### 8.2 Pool-B + 语义组特征搜索（分支，非主线）

```bash
mlbot diagnose poolb-semantic-search \
  --strategies bpc \
  --symbol BTCUSDT --timeframe 120T \
  --start-date 2024-01-01 --end-date 2025-04-30 \
  --search-algo pipeline --expand-semantic-singletons \
  --regen-poolb --rerun-search
```

> 详细 A/B/C 预算预设见 `docs/archive/guides/tree/FEATURE_GROUP_SEARCH_PRESETS_CN.md`。

#### 8.3 Locked 阈值调参（分支工具）

> 保持 locked 语义特征不变，只扫阈值；多窗口滚动评分。

```bash
python scripts/tune_locked_prefilter_thresholds.py \
  --strategy bpc \
  --end-dates 2025-10-01,2025-11-01,2025-12-01,2026-01-01,2026-02-01,2026-03-01 \
  --min-trades-target 60 --trade-penalty 0.002
```

> 当 `prefilter.yaml` 有 `locked: true` 规则时，`mlbot pipeline run` 会自动触发；缓存在 `results/locked_tuning/cache/`，加 `--disable-auto-locked-tuning` 可关闭。

#### 8.4 TaskSpec 驱动的 Tier0/Tier1 对比（nnmultihead，分支）

你问的“Tier0/Tier1 会如何影响训练？是不是跑两次看报告？”——**是的**，但需要做到两点才能可复盘：  
1) 每个 Tier 生成一个**具体可执行的 config 目录**（不直接靠“标签”）  
2) 用各自 config 训练出 model，再用统一流程评估（A-layer + system/e2e）  

##### 1) 先从 TaskSpec 生成派生 config（让 tiers 变成真实 features.yaml）

```bash
mlbot nnmultihead materialize-config-from-task-spec --no-docker \
  --task-spec config/tasks/task_spec.yaml \
  --base-config config/nnmultihead/path_primitives_4h_80h_min \
  --out-config results/derived_cfg/tier01
```

> `task_spec.yaml` 里通过 `feature_plan.tiers_enabled` + `tier_feature_files` 显式定义 Tier0/Tier1 的 feature nodes 列表。  

##### 2) 训练（TaskSpec-only：命令会自动 materialize 派生 config）

```bash
mlbot nnmultihead train --no-docker \
  --task-spec config/tasks/task_spec.yaml \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT
```

##### 3) 跑主链路评估（predict → router → build-logs → e2e）（TaskSpec-only）

```bash
mlbot nnmultihead pipeline-3action-e2e --no-docker \
  --task-spec config/tasks/task_spec.yaml \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --start-date 2025-05-01 --end-date 2025-12-31 \
  --model <PATH_TO_MODEL_PT_FROM_TRAIN> \
  --returns-source rr_execution \
  --out results/nnmh_e2e/tier01
```

> **详细工作流文档**: 完整的命令序列、Gate 过滤说明、ET/FR 交易缺失原因分析等，见 [`docs/workflow/PIPELINE_WORKFLOW.md`](docs/workflow/PIPELINE_WORKFLOW.md)

##### 完整 NN Pipeline 工作流（固定流程）

**推荐使用一键脚本**（见 `scripts/run_full_pipeline.py`）：

```bash
python scripts/run_full_pipeline.py \
  --task-spec config/tasks/task_spec_highcap6_2024_202510.yaml \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --model results/nnmultihead/.../model.pt \
  --feature-store-layer nnmh_highcap6_240T_2024_with_reflexivity \
  --run-id pipeline_2024_reflexivity_validation
```

**手动执行步骤**（每一步都有日志输出，支持断点续传）：

1. **FeatureStore构建**（如果需要新特征）
2. **模型预测** → `preds/`
3. **Regime分类** → `physics_regime.parquet`
4. **构建Execution日志** → `logs_execution.parquet`
5. **应用Gate过滤** → `logs_execution_gated.parquet`
6. **添加反身性特征**（可选）→ `exec_logs/features/`
7. **构建Stage Logs** → `exec_logs/{preds,router,gate,execution,returns,features}/`
8. **聚合Canonical Log** → `execution_log.jsonl`
9. **生成E2E KPI报告** → `e2e_kpi_report.md`

详细命令和说明见 [`docs/workflow/PIPELINE_WORKFLOW.md`](docs/workflow/PIPELINE_WORKFLOW.md)

##### 3.1) Router 阈值：用"平坦高原"协议做稳健调参

> 目的：避免“找尖峰”导致的炼丹，优先选多窗口/bootstrapped 都稳的阈值组合。  
> 详细解释见：`docs/architecture/guides/THRESHOLD_PLATEAU_TUNING_PROTOCOL_CN.md`
>
> 说明：`mlbot nnmultihead pipeline-3action-e2e` 会在输出目录下自动写出
> `router_thresholds_baseline.json`（使用你传入的阈值覆盖 + 未传入则用 Router 默认值），
> 供 plateau 命令直接复用。
>
> 默认 tuned-threshold 流程已包含：**heuristic bounds**（防离谱阈值）与 **trend_rate 约束**（防 TREND 趋零）。

```bash
mlbot diagnose threshold-plateau --no-docker \
  --preds results/nnmh_e2e/tier01/preds \
  --logs  results/nnmh_e2e/tier01/logs_3action.parquet \
  --model <PATH_TO_MODEL_PT_FROM_TRAIN> \
  --baseline-json results/nnmh_e2e/tier01/router_thresholds_baseline.json \
  --out results/plateau/router3action_tier01_oos_v1 \
  --trend-rate-min 0.005 --trend-rate-penalty 2.0 \
  --heuristic-bounds --heuristic-qmin 0.05 --heuristic-qmax 0.95
```

**用法（推荐两步法）**

- **Step A：先跑一次 pipeline 产出 `preds/` + `logs_3action.parquet` + baseline thresholds**

```bash
mlbot nnmultihead pipeline-3action-e2e --no-docker \
  --task-spec config/tasks/task_spec.yaml \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 --end-date 2024-06-30 \
  --feature-store-root feature_store \
  --feature-store-layer nnmh_tree_union_all_240T_v2 \
  --model <PATH_TO_MODEL_PT_FROM_TRAIN> \
  --returns-source rr_execution \
  --out results/nnmh_e2e/tier01
```

- **Step B：做 plateau tuning，得到 `router_thresholds_best.json`，再把它喂回 pipeline 重跑**
  - 输出位置：`results/plateau/router3action_tier01_oos_v1/router_thresholds_best.json`
  - 重要：**阈值调参只能用 train/oos（可调参）窗口**；`holdout`（只验收）不要用来调参。

```bash
# 1) tune -> best thresholds
mlbot diagnose threshold-plateau --no-docker \
  --preds results/nnmh_e2e/tier01/preds \
  --logs  results/nnmh_e2e/tier01/logs_3action.parquet \
  --model <PATH_TO_MODEL_PT_FROM_TRAIN> \
  --baseline-json results/nnmh_e2e/tier01/router_thresholds_baseline.json \
  --out results/plateau/router3action_tier01_oos_v1 \
  --trend-rate-min 0.005 --trend-rate-penalty 2.0 \
  --heuristic-bounds --heuristic-qmin 0.05 --heuristic-qmax 0.95

# 2) rerun pipeline with tuned thresholds (explicitly applied)
mlbot nnmultihead pipeline-3action-e2e --no-docker \
  --task-spec config/tasks/task_spec.yaml \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 --end-date 2024-06-30 \
  --feature-store-root feature_store \
  --feature-store-layer nnmh_tree_union_all_240T_v2 \
  --model <PATH_TO_MODEL_PT_FROM_TRAIN> \
  --router-thresholds-json results/plateau/router3action_tier01_oos_v1/router_thresholds_best.json \
  --returns-source rr_execution \
  --out results/nnmh_e2e/tier01_tuned
```

##### 3.2) 灭绝回放（Extinction Replay）：产出 survival labels

> 目的：把 “在极端路径里会不会死” 变成可回放、可产物、可训练的标签（`labels.parquet`）。
> 对应长文：`docs/archive/architecture/archetype灭绝级回测.md`、`docs/archive/leagcy/ood头的训练.md`

```bash
mlbot diagnose extinction-replay-3action --no-docker \
  --logs results/nnmh_e2e/tier01/logs_3action.parquet \
  --out  results/extinction_replay/tier01_v1
```

##### 3.3) 训练 Survival Head（MLP）：产出 survival_prob

> 输入：`logs_3action.parquet` + 上一步产出的 `labels.parquet`  
> 输出：`model.pt` + `survival_preds.parquet` + `report.html`（含 ROC/PR/Calibration 曲线）

```bash
mlbot diagnose survival-head-train --no-docker \
  --logs   results/nnmh_e2e/tier01/logs_3action.parquet \
  --labels results/extinction_replay/tier01_v1/labels.parquet \
  --out    results/survival_head/tier01_v1
```

##### 3.4) Conditional Survival Table：学习 OOD → Archetype 生存权重

> 目的：先用最稳的“表格基线”学习 `survival_rate(archetype | ood_bin)`，并导出可部署的 `weights.yaml`。  
> 备注：需要 `logs_3action.parquet` 中存在 `ood_score` 与 `active_archetype` 列（通常来自 LiveDashboard/Router 产物合并）。

```bash
mlbot diagnose ood-to-archetype-weights --no-docker \
  --logs   results/nnmh_e2e/tier01/logs_3action.parquet \
  --labels results/extinction_replay/tier01_v1/labels.parquet \
  --out    results/ood_to_archetype/tier01_v1
```

---

## Research Notes（近期结论）

- **Tier2 / Orderflow-only 对多头模型无增强**（HighCap6 / 2024H1）
  - Orderflow baseline：`results/runs/tier02_highcap6_2024H1_orderflow_20260115_041919/`
  - Orderflow tuned（plateau 后）：`results/runs/tier02_highcap6_2024H1_orderflow_tuned_20260115_044953/`
  - 对比报告：`results/compare/nnmh_runs/20260115_045141/report.md`
  - 结论：A-layer 提升不明显，系统层 Sharpe/收益无改善，trade_rate 下降；可暂时放弃该方向。

- **Tier2 / Spectrum+Math 对多头模型无增强**（HighCap6 / 2024H1）
  - Spectrum+Math baseline：`results/runs/tier02_highcap6_2024H1_spectrum_math_20260115_042103/`
  - Spectrum+Math warm3+extfill：`results/runs/tier02_highcap6_2024H1_spectrum_math_warm3_extfill_20260115_050924/`
  - Spectrum+Math nocache：`results/runs/tier02_highcap6_2024H1_spectrum_math_nocache_20260115_075757/`
  - 对比报告（同组内部）：`results/compare/nnmh_runs/20260115_095830/report.md`
  - 对比报告（与 Tier01 baseline）：`results/compare/nnmh_runs/20260115_100550/report.md`
  - 结论：系统层 Sharpe/收益无明显改善且 trade_rate 更低；无 plateau 报告产出，优先级降低。


对比方式：
- 跑两份 TaskSpec（Tier0-only vs Tier0+Tier1），生成两份 derived config / model / 报告
- 对比：A-layer（head eval）+ system（e2e counterfactual + KPI gate + snapshot）

---

## 文档入口（建议先读）

> **📚 统一文档索引**: [docs/README.md](docs/README.md) - 推荐从这里开始浏览所有文档

### ⭐ 日常研究必备（最新）

- **`docs/z实验_005_统一研究/A快速启动命令.md`** — 最完整的命令手册：数据下载 / Feature Store / pipeline run 所有 stage / 事件回测 / 滚动多窗口 / locked 阈值调优工具
- **`docs/z实验_005_统一研究/实施文档_01_2024牛市_5x趋势骑乘.md`** — 2024 牛市 5x 趋势骑乘实验：turbo 快模式（只调阈值）/ slow 模式 / event_backtest / FS 问题修复 / adopt-deploy 流程
- **`docs/z实验_005_统一研究/box_features_causal_vs_oracle_20260424.md`** — 盒子结构特征（`box_*`）的因果 vs oracle 对照
- **`docs/z实验_005_统一研究/CRF_CBC_structure_diagnosis_20260424.md`** — CRF/CBC 策略结构诊断

### 核心工作流文档

- **上线 MVP 闭环（树模型，已归档）**：[docs/archive/guides/tree/DEPLOYMENT_MVP_WORKFLOW_CN.md](docs/archive/guides/tree/DEPLOYMENT_MVP_WORKFLOW_CN.md)（当前主线见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)）
  - Pool‑B + 语义组搜索 → 6 个月 holdout 验收 → 训练最终上线模型
  - rolling 与事件回测的职责边界（OOS vs 实盘一致性）

- **多资产合约实盘落地路线图（从 1w→10w 的可执行路线，低维护）**：`docs/archive/guides/LIVE_TRADING_ROADMAP_MULTI_ASSET_CN.md`
- **ETH 拖累处理与 Universe 演进**：仓库内暂无独立文档（若需可从 Git 历史检索 `ETH_DRAG_AND_UNIVERSE_EVOLUTION_CN`）
- **特征搜索 Playbook（详细算法/命令/概念）**：`docs/architecture/strategies/FEATURE_SEARCH_PLAYBOOK_CN.md`
- **语义特征单列展开说明**：`docs/architecture/strategies/SEMANTIC_GROUPS_SINGLETON_EXPANSION.md`
- **归一化契约与检查**：`docs/architecture/NORMALIZATION_CONTRACT_AND_CHECKS.md`
- **“保留但不喂给模型”的列排除机制（exclude_columns）**：`docs/architecture/guides/FEATURE_PIPELINE_EXCLUDE_COLUMNS_CN.md`
- **Feature-group-search / pipeline 调参指南**：`docs/archive/guides/tree/FEATURE_GROUP_SEARCH_TUNING_GUIDE_CN.md`
  - nnmultihead 推荐顺序：search → train(primitives) → OOS predict → build-logs → Router 阈值调参 → BC/RL
- **特征测试设计与覆盖（4类测试 + 覆盖快照保存）**：`docs/tests/FEATURE_TEST_DESIGN_AND_COVERAGE_CN.md`
- **实盘特征契约与证据字段（缺失策略/has_orderflow/has_sr_quality）**：`docs/archive/LIVE_FEATURE_CONTRACT_AND_EVIDENCE_CN.md`
- **Archetype 上线前 Checklist（v0）**：`docs/architecture/ARCHETYPE_PRELIVE_CHECKLIST_CN.md`

- **项目 TODO / Roadmap**：`docs/architecture/ARCH_UPGRADE_TASKSPEC_CONSTITUTION_V1_CN.md`
  - TODO 已内聚到架构升级文档中（按 P0/P1/P2 分层）

### 架构文档

- **文档索引**：[docs/README.md](docs/README.md) - 统一文档导航入口
- **系统架构图（已更新到当前 CLI/工作流）**：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

- **特征列表与归一化状态**：`docs/architecture/FEATURE_CATALOG.md`
  - 全部 208 个特征节点的归一化状态
  - 归一化方法说明（ATR 归一化、百分比归一化、相似度转换等）
  - 按类别分组的特征列表
  - 树模型 vs NN 模型的使用建议

- **特征归一化策略**：`docs/architecture/FEATURE_NORMALIZATION_POLICY.md`
  - Phase 1/2/3 归一化实现进度
  - 跨资产可比性验证
  - 因果性归一化方法（避免未来泄露）

- **工业化 Experiment Loop（Layer A/B/C、TaskSpec、Filter→Wrapper、稳定性口径）**：`docs/architecture/EXPERIMENT_LOOP_ARCHITECTURE.md`
- **NN 多头 Path Primitives + Router→Execution（NO/MEAN/TREND）**：`docs/时序模型/架构：NN多头路径原语（Path Primitives）+Router解耦升级.md`
- **研究 Playbook（标签/执行一致性、timeframe、仓位管理）**：`docs/archive/guides/tree/RESEARCH_PLAYBOOK_CN.md`
- **Policy 执行假设（intrabar vs close）**：`docs/architecture/strategies/POLICY_EXECUTION_ASSUMPTIONS_CN.md`
- **NN 多头 → 3-action → RL/BC e2e（长文档）**：`docs/archive/NNMULTIHEAD_3ACTION_E2E_CN.md`
- [多头NN和订单流的使用分类和评估](/workspaces/ml_trading_bot/docs/architecture/多头NN和订单流.md)
- [训练落地文档](docs/architecture/guides/FEATURE_COMPLEXITY_LAYERS_CN.md)
- [谁对sharp负责](docs/archive/architecture/谁对sharp负责.md)
- [删除的策略(该不做什么)](docs/architecture/删除的策略该不做什么.md)
- [alpha可以更多吗](docs/archive/architecture/alpha可以更多吗.md)
- [VolMean难在哪里](docs/architecture/VolMean难在哪里.md)
- [时间框架高级甜点区](docs/archive/architecture/时间框架高级甜点区.md)
- [职责坍缩](docs/archive/architecture/职责坍缩.md)
- [GATE_FEATURE_MAPPING_VS_TREE_PHASE1](docs/architecture/GATE_FEATURE_MAPPING_VS_TREE_PHASE1.md)
- [PLATEAU_OPTIMIZATION_METHODOLOGY](docs/architecture/guides/PLATEAU_OPTIMIZATION_METHODOLOGY.md)
- [NNMULTIHEAD_COMMANDS_CN](docs/archive/NNMULTIHEAD_COMMANDS_CN.md)
- [meta_router_core_pipeline](docs/architecture/meta_router_core_pipeline.md)
- [数学特征如何使用](docs/architecture/数学特征如何使用.md)
---

## 获取帮助

```bash
mlbot --help
```

---

## 实盘（Live：WebSocket + MetaRouterCore）

> 实盘入口与事件流/回放/对账等细节：见 `docs/architecture/live_stream/README.md`。

启动实盘交易系统（WebSocket → OrderFlowListener → MetaRouterCore → OrderManager）：

```bash
MLBOT_LIVE_SYMBOLS=BTCUSDT \
MLBOT_LIVE_USE_FUTURES=true \
MLBOT_ORDER_MANAGEMENT_DB_PATH=data/order_management.db \
python scripts/run_live.py
```
