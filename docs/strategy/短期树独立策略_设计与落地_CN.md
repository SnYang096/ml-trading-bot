# 短期树独立策略：设计与落地（fast_scalp / short_term_swing）

> 配套：[`树模型方法论演进与短期树重建指南_CN.md`](树模型方法论演进与短期树重建指南_CN.md)（六阶段史 + §4.2.5 阶段 5）、[`WORKFLOW_整体架构与管线改进计划_CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md)（ABC × 层职责）、[`方法论_R_and_D流程_CN.md`](方法论_R_and_D流程_CN.md) §3.4（树通道命令速查）。
>
> 本文档回答两件事：
>
> 1. `config/strategies/tree_strategies/` 是什么，对应 ABC 哪个系统？
> 2. 「单棵树直接给开仓方向，按 IC 衰减分 fast/swing 两档」是否合理？怎么落地？

---

## 0. 摘要

- **`tree_strategies/` 现状 = 考古资产**（阶段 2 Pool-B + FGS）。它**不是** B 系统，也不是 C 系统；它是一条**独立的 ML 通道**，在阶段 4 让位给 archetype 规则主线（[`树模型方法论演进*.md`](树模型方法论演进与短期树重建指南_CN.md) §4.2.2、§4.4.1）。
- **用户提案的合理性**：✅ 合理且与「阶段 5 短期树重建」完全一致：**IC 衰减期 → H → 浅树 → 单 τ plateau**（[`树模型方法论演进*.md`](树模型方法论演进与短期树重建指南_CN.md) §4.2.5、§4.2.2 阶段 5）。  
  唯一要纠正的口径：**不是「B/C + 树」**，而是「与 B/C 并行的独立 slug」。
- **落地推荐**：建两个 slug
  - `config/strategies/fast_scalp/`（H ≈ 3 bar，对标 C 的 trend_scalp 节奏，独立单腿执行）
  - `config/strategies/short_term_swing/`（H ≈ 20 bar，对标 B 的趋势 swing 集合，独立单腿执行）
- **明确不做**（用户已说明，与 README §9.2、`B 系统运维心智梳理.md` 一致）：把树 score 接进 B 的 evidence 做仓位放大缩小。

---

## 1. `tree_strategies/` 现状与 ABC 对应

### 1.1 实际目录

```
config/strategies/tree_strategies/
├── trend_following/           # 阶段 2: 50 bar rank label
├── sr_reversal_rr_reg_long/   # 50 bar R/R 回归（在 strategies_exported/tree_best/ 下）
├── sr_breakout/
├── compression_breakout/
└── strategies_exported/tree_best/<策略>_<日期>/...
```

特点（[`树模型方法论演进*.md`](树模型方法论演进与短期树重建指南_CN.md) §4.3.3、§4.4.1）：

- **标签都是 50 bar 量级**（rank / R-R 回归 / barrier）
- **特征是 Pool-B + FGS 选出的「Sharpe_mean 最优组合」**
- **训练后由 `train_strategy_pipeline.py` 直接出 `predict_proba` → `long_entry_threshold = 0.6` 入场**
- 现已 `_EXCLUDED_STRATEGY_SUBDIRS = {"bad-candidates", "tree_strategies"}`（`scripts/rolling_dashboard/pipeline_jobs.py`），**不进默认 rolling**。

### 1.2 它属于 B 还是 C？

**都不是**。它对应 [`树模型方法论演进*.md`](树模型方法论演进与短期树重建指南_CN.md) §4.2.3 的「并行线 C：机器学习形态」，是一条 **从 B/C 之外** 起步的 ML 通道。它当时**试图**用四棵树模拟 SR/突破/趋势剧本（替代 B 的 archetype 叙事），但因「样本窄 + y 与特征时间尺度错位 + FGS Sharpe_mean 优化」三个原因被阶段 4 弃用（§4.4.1）。

### 1.3 与 B/C 的边界（写进 doctrine）

| | B (BPC/TPC/ME/SRB) | C (chop_grid/trend_scalp) | Tree 通道（新短期树） |
|---|---|---|---|
| **决策载体** | yaml 规则（archetypes/*.yaml）| yaml 规则（多腿引擎）| **浅树 score + 单 τ plateau** |
| **典型 horizon** | 趋势 swing（数日）| 段内（数小时～1 天）| 与 IC 衰减对齐：3 bar / 20 bar |
| **特征工厂** | locked 规则 + Q 级 quick_layer_scan | 季度语义代理 R&D（§2.2.1）| **IC@H → best_lag 冻结 30–50 列** |
| **PCM 槽位** | 4 个槽 | 2 个槽 | **独立槽（新增）** |
| **能合并仓位吗** | — | — | ❌ 不并入 B 的 evidence（用户决定） |

---

## 2. 用户提案是否合理？

> 提案原文：
> - 一个单独的树直接给出开仓方向
> - 训练对齐 IC 衰减期：3 bar 内有效 → 类似 C trend_scalp；20 bar 衰减 → 类似 B trend swing 集合
> - **不**把 tree 做 evidence 合并到 B 做仓位放大缩小

### 2.1 ✅ 合理之处

1. **IC 衰减对齐 horizon** 正是阶段 2 失败、阶段 5 重建的核心修复（[`树模型方法论演进*.md`](树模型方法论演进与短期树重建指南_CN.md) §4.2.5、§4.4.1 第 2 条）：旧做法把 3 bar IC 特征训 50 bar rank，信号被抹平。
2. **单棵树直接给方向** 与 §1.2.5 "新流程（决策树 + 单高原）" 一致：树 score → 单维 τ plateau → 写 `backtest.yaml` 的 `entry_threshold`，比逐特征 plateau 简单。
3. **fast / swing 两档分流** 直接对应 §4.2.5 表："`fast_scalp` / `short_term_swing` 分两档 H"。
4. **不做 evidence 合并** 与 `README_CN.md` §9.2、`B 系统运维心智梳理.md` 一致：B 入场已定稿，树不持续优化它的仓位。

### 2.2 ⚠ 仍要注意的边界

| 风险 | 建议 |
|---|---|
| 单棵树纯方向、无尾部 veto，在 crypto fat-tail 段会顶穿 maxDD | 树 score 之上叠 **一条** 软 veto：`evt_var_99 < x` 或 `bull_share` 极端时 size×0.5；不要叠 B 那一套 gate 全套 |
| 不带 regime，可能在 bear 抢多 | **regime 与 B 共享**：复用 `archetypes/regime.yaml` 的 `|ema_1200_position|>0.10`（或自带一条更紧的 chop ≤ 0.4），但**不复用 B 的 prefilter/entry/gate** |
| 「方向」如果输出 LONG/SHORT 二分类，遇到弱信号会强行选边 | 三态：score < τ_short → SHORT；τ_short ≤ score ≤ τ_long → 不开仓；score > τ_long → LONG（与现 `tree_strategies/*/backtest.yaml` 的 `long/short_entry/exit_threshold` 一致） |
| 与 B 的 TPC/BPC 撞同一时刻的同方向多腿 → PCM 槽位被吃光 | PCM 仲裁层加新 slug 的预算配比；不抢 B 的现有 4 个槽 |
| FGS Sharpe_mean 选特征的过拟合 | 不用 FGS；用 `mlbot analyze factor-eval --ic-decay-lags` 出 IC + best_lag，人工冻结 30–50 列，与 §4.2.5 阶段 5 一致 |

### 2.3 为什么**不**走 `regime → prefilter → gate → tree`

- prefilter 是 **archetype 叙事**（BPC 突破后回踩 / TPC 趋势内深回调），它**就是规则故事**；让树跟在 prefilter 后面，等于把树绑死在某一种叙事里 → 又回到阶段 2 "样本窄、叶不稳" 的老坑。
- gate 是 B 的尾部 veto，它需要 `pnl_r` 实测验证（[`docs/decisions/tpc_gate_vol_ABH_experiment_20260526.md`](../decisions/tpc_gate_vol_ABH_experiment_20260526.md)）；树本身就是分数排序，自己的 τ plateau 就解决了同一类问题。
- 这两层叠上去后，树训练样本可能从 100k+ bar 降到 < 5k，模型不稳。

**结论**：`regime（最薄一层）→ 单棵树 → 单 τ plateau → execution`。不要再加 prefilter / gate。

---

## 3. 推荐架构

```text
                                    共享 B 的 regime（仅 |ema_1200_position|>0.10 或自定 chop≤0.4）
                                                │
config/strategies/fast_scalp/                  ▼              config/strategies/short_term_swing/
  features.yaml  (best_lag ∈ [1,3,5])  ─┐                ┌─  features.yaml  (best_lag ∈ [10,20])
  labels.yaml    forward_rr @ H=3      ─┤  独立单棵树    ├─  labels.yaml    forward_rr @ H=20
  model.yaml     浅 LightGBM            ─┤  predict_proba ├─  model.yaml     浅 LightGBM
  backtest.yaml  long/short τ plateau   ─┘                └─  backtest.yaml  long/short τ plateau
                                                │
                                                ▼
                                  独立单腿 execution（SL / TP / time stop）
                                                │
                                                ▼
                                  PCM 仲裁（新槽，不抢 B/C 的 6 个）
```

| 层 | 处理 |
|---|---|
| Regime | **共享**：复用 B 的 `|ema_1200_position|≥0.10` 与（可选）`chop≤0.4`；不重新发现 |
| Prefilter | **没有** |
| Direction | **由树 score 决定**：score > τ_long → LONG，< τ_short → SHORT，中间不开 |
| Gate | **没有 B 那套**；可选「软 veto」最多 1 条（fat-tail 极端） |
| Entry | **没有** OR rules；树就是择时 |
| Execution | 单腿；time stop 与 H 对齐（H=3 → ≤ 6 bar time stop；H=20 → ≤ 40 bar）|

---

## 4. 两个候选 slug 草图

### 4.1 `fast_scalp`（H = 3 bar，对标 C 的 trend_scalp 节奏）

```yaml
# config/strategies/fast_scalp/labels.yaml
target_column: label
label_generator:
  module: src.time_series_model.strategies.labels.forward_rr
  function: compute_forward_rr_label
  params:
    price_col: close
    horizon: 3
    rr_floor: 0.30        # |forward_rr| 低于此值视为 0（与 H 的方差匹配）
    use_log_return: true
```

```yaml
# config/strategies/fast_scalp/features.yaml
# 仅用 IC 在 1–5 bar 仍显著的列（由 mlbot analyze factor-eval 出表后冻结）
features:
  - macd_atr
  - bb_width_normalized_pct
  - tpc_semantic_chop
  - vp_absorption_score
  - cvd_short
  - vpin_short
  - hurst_short
  - atr_percentile
  # ≤ 30 列；详细列由 IC 阶段定
```

```yaml
# config/strategies/fast_scalp/model.yaml
type: lightgbm
params:
  objective: regression
  num_leaves: 15           # 浅
  max_depth: 4
  learning_rate: 0.03
  n_estimators: 400
  min_data_in_leaf: 200
  feature_fraction: 0.7
  bagging_fraction: 0.7
  bagging_freq: 5
```

```yaml
# config/strategies/fast_scalp/backtest.yaml
backtest:
  enabled: true
  params:
    price_col: close
    freq: "120T"
    use_signal_direction: true
    long_entry_threshold: 0.55      # 由 holdout τ plateau 定
    long_exit_threshold: 0.50
    short_entry_threshold: 0.45
    short_exit_threshold: 0.50
    initial_cash: 10000
    fee: 0.0004
    slippage: 0.0001
    max_holding_bars: 6             # 与 H=3 对齐
    use_trailing_stop: true
    trailing_atr_mult: 1.5
    position_sizing:
      type: atr_risk
      risk_pct: 0.005               # 比 B 紧
      atr_col: atr
      atr_window: 14
```

### 4.2 `short_term_swing`（H = 20 bar，对标 B 趋势 swing 集合）

差异（其余结构同上）：

```yaml
# labels.yaml
params:
  horizon: 20
  rr_floor: 0.80
```

```yaml
# features.yaml
# IC 在 10–20 bar 仍显著的列（趋势 / 慢均线 / WPT scene / ema_1200_*）
features:
  - ema_1200_position
  - ema_1200_slope_10
  - trend_confidence
  - hurst_long
  - wpt_scene_*
  - bb_width_normalized_pct
  - macd_atr
  # ≤ 50 列
```

```yaml
# backtest.yaml — params 差异
max_holding_bars: 40
trailing_atr_mult: 2.0
position_sizing.risk_pct: 0.01
```

---

## 5. 训练 / 验证 / promote 命令

> 全部不动 live；产物在 `results/<slug>/...`。

### 5.1 第一步：IC 对齐 horizon

```bash
PYTHONPATH=src:scripts python -m mlbot analyze factor-eval \
  --strategy fast_scalp \
  --features-yaml config/strategies/fast_scalp/features.yaml \
  --ic-decay-lags 1,3,5,10,20,50 \
  --out results/fast_scalp/factor_eval/<日期>.md

# 等价的轻量入口（无须 features.yaml，已有 parquet 即可）
PYTHONPATH=src:scripts python scripts/quick_layer_scan.py ic-decay \
  --features-parquet results/.../features_labeled.parquet \
  --features macd_atr,bb_width_normalized_pct,tpc_semantic_chop,... \
  --horizons 1,3,5,10,20 \
  --baseline-json config/monitoring/factor_ic_baseline_tpc_20260526.json \
  --out results/fast_scalp/ic_decay_<日期>.md
```

判读：

- **fast_scalp** 保留 best_lag ∈ {1,3,5} 且 \|IC\|>0.02 的列。
- **short_term_swing** 保留 best_lag ∈ {10,20} 且 \|IC\|>0.015 的列。
- 任一列在 H=50 仍最强 → 不该进这两个 slug（属于阶段 2 旧 trend_following 范畴，不再训）。

### 5.2 第二步：训练 + 出 score

```bash
PYTHONPATH=src:scripts python -m mlbot train final \
  -c config/strategies/fast_scalp \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --output-dir results/train_final/fast_scalp/$(date +%Y%m%d_%H%M%S)
# 产物：predictions.parquet（含 score 列）+ features_labeled.parquet
```

### 5.3 第三步：τ plateau 标定（holdout 上的 score 单维扫描）

```bash
# 沿用已有的单维 plateau 工具（语义同 scan_chop_plateau / _identify_plateau）
PYTHONPATH=src:scripts python scripts/regime_threshold_calibrate.py \
  --features-parquet results/train_final/fast_scalp/<ts>/predictions.parquet \
  --feature score --operator ">=" \
  --grid 0.45,0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.65 \
  --label forward_rr \
  --out results/fast_scalp/tau_plateau_<日期>.md
```

把 plateau 中心写回 `backtest.yaml` 的 `long/short_entry_threshold`，**人审 + commit**，不自动 promote。

### 5.4 第四步：event_backtest 双段验证

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest --variant-grid \
  config/experiments/fast_scalp_dual_period.yaml
# 产物：results/fast_scalp/experiments/{H_recent,H_bull_2024}/ + EXPERIMENT_INDEX.json
```

判读规则**与 B 完全一致**（[`方法论_R_and_D流程_CN.md`](方法论_R_and_D流程_CN.md) §2.4 Pareto rule）：两段都不劣才 promote；只一段好就做 regime-conditional 或 drop。

### 5.5 第五步：决策文档 + promote

```bash
PYTHONPATH=src:scripts python scripts/_new_decision_doc.py \
  --experiment-index results/fast_scalp/experiments/EXPERIMENT_INDEX.json \
  --topic fast_scalp_<H>_initial \
  --out docs/decisions/fast_scalp_<H>_<日期>.md
```

人审后才动 `live/highcap/config/strategies/fast_scalp/`。

---

## 6. 与 B / C 并行运行（PCM 仲裁）

> 加 slug 不抢现有槽位；落 live 前确认下列三件。

1. **`config/constitution/constitution.yaml`**：把 `fast_scalp` / `short_term_swing` 加入 `strategies` 与 PCM 槽位分配（建议各 1 槽起，与 B/C 总和不超过现宪法上限）。
2. **同方向冲突**：同一 symbol 同一 bar 出现 `fast_scalp.LONG + TPC.LONG` 时，PCM 默认按 `evidence` 排序；树通道 evidence 用 `score`，与 B 的 `pnl_r_pred` 不同尺度——**必须** normalize（min-max 到 [0,1]）。
3. **Kill switch**：树通道 maxDD 超出 PCM 上限时单独 kill，不影响 B/C。

---

## 7. 实施步骤（不动 live）

| 步骤 | 命令 / 工作 | 产物 | 改 live？ |
|---|---|---|---|
| 1. 建 slug 目录 | `mkdir -p config/strategies/{fast_scalp,short_term_swing}` + 写 5 个 yaml（§4）| 5 个 yaml | 否 |
| 2. IC 对齐 | `mlbot analyze factor-eval` + `quick_layer_scan ic-decay` | 因子表 + 冻结 features.yaml | 否 |
| 3. 训练 | `mlbot train final` | predictions.parquet | 否 |
| 4. τ plateau | `regime_threshold_calibrate.py` 单维 | plateau 报告，写 backtest.yaml | 否 |
| 5. 双段回测 | `event_backtest --variant-grid` | EXPERIMENT_INDEX | 否 |
| 6. 决策文档 | `_new_decision_doc.py` | `docs/decisions/...` 骨架 | 否 |
| 7. PCM 配比 | 修 `config/constitution/constitution.yaml`（仅 config，不动 live） | constitution diff | 否 |
| 8. shadow 跑 ≥ 1 季度 | 与 B/C 并行 | watchdog + drift 报告 | 否 |
| 9. 人审 → live | `deploy_config_to_live.py` | live yaml | ✅ |

---

## 8. FAQ

**Q1**：训出来的树要不要导出可读规则，写回 `gate.yaml`？

A：**不要**。`tree_strategies` 时代试过 `export_tree_rules_imodels*.py`（[`树模型方法论演进*.md`](树模型方法论演进与短期树重建指南_CN.md) §4.3.2），上线后阈值漂移多、维护难。这里 promote 的是 **τ plateau 中心 + 树模型权重**，不是 if/else 规则。

**Q2**：能不能复用 `tree_strategies/trend_following/` 的现有 yaml？

A：**结构可以参考、内容必须重写**。`trend_following/labels.yaml` 是 `horizon=50` + rank label，与新 slug 的 `forward_rr @ H=3/20` 完全不同；其 features 是 FGS Sharpe_mean 选出来的，过拟合风险高（[`树模型方法论演进*.md`](树模型方法论演进与短期树重建指南_CN.md) §4.4.1）。

**Q3**：为什么不直接在 B 的 TPC 上加一个 score 列做仓位缩放？

A：用户已明确不做（理由："用处太小"）。文档侧的额外理由：B 的 evidence 缩放与 PCM 槽位绑定，新增一个 score 通道会让宪法 `evidence_position_scale` 的语义不再单调；这正是阶段 3 `OUTCOME_BASED_TREE_LABELING` 之后被弃的方向（[`树模型方法论演进*.md`](树模型方法论演进与短期树重建指南_CN.md) §4.2.2 阶段 3）。

**Q4**：fast_scalp 与现有 `trend_scalp` 不会冲突吗？

A：现有 `trend_scalp` 是 C 系统的**规则多腿**（语义 chop → trend transition）；`fast_scalp` 是树通道**单腿**。两者 PCM 槽位独立。命名上：如果觉得歧义，可把新 slug 命名为 `tree_fast`、`tree_swing`。

**Q5**：要不要训一棵共享树、按 horizon 切两个头？

A：起步先**两棵独立**，原因：训练样本可以各自挑 IC-aligned 特征池，互不污染；管线（features.yaml / labels.yaml / model.yaml / backtest.yaml）都按 slug 拆，PCM 看到的也是独立预算。等 fast 与 swing 都跑稳后，可再考虑「共享 backbone + 双头」作为 v2。

---

## 9. 不做（明确决定）

- ❌ 把树 score 接进 B 的 `evidence_position_scale`（用户决定）。
- ❌ 树跟在 B 的 prefilter/gate 之后做尾部 / 加仓判断（样本窄 + 与规则叙事冲突）。
- ❌ 用 FGS Sharpe_mean 选特征（已是阶段 5 明确弃用，见 [`树模型方法论演进*.md`](树模型方法论演进与短期树重建指南_CN.md) §4.4.1）。
- ❌ 重启 `tree_strategies/sr_reversal_*` 等四策略树（考古资产保留为参考，不再训）。

---

## 10. 与现有文档的关系

| 文档 | 关系 |
|---|---|
| [`树模型方法论演进与短期树重建指南_CN.md`](树模型方法论演进与短期树重建指南_CN.md) | 本文是其 §4.2.5「阶段 5 短期树重建」与 §1.2 决策层 ML 的**落地手册** |
| [`WORKFLOW_整体架构与管线改进计划_CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md) | ABC 系统职责矩阵，本文把「树通道」与 A/B/C 并列入 §3 算法分工 |
| [`方法论_R_and_D流程_CN.md`](方法论_R_and_D流程_CN.md) §3.4 | 命令速查表（与本文 §5 对齐） |
| [`研究工具重构计划_CN.md`](研究工具重构计划_CN.md) §14 | Phase 9+ 新命令族默认路径 + tree rd_loop 示例 |

**树通道 R&D playbook（rd_loop）**：
- `config/experiments/fast_scalp/rd_loop_fast_scalp_ic_plateau.yaml` — IC decay → snotio plateau → variant-grid
- `config/experiments/short_term_swing/rd_loop_short_term_swing_ic_plateau.yaml` — 同上
| [`B 系统运维心智梳理.md`](B系统运维心智梳理.md) / `README_CN.md` §9.2 | 树**不**做 B 的持续优化、不接 evidence 的口径 |
