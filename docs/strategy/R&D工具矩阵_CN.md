# R&D 工具矩阵（B 系统主线）

> **替代口径**：日常 R&D 与运维以本文为准，不再把 `research_roll` / `validate_static.*` 当作发现特征或调阈值的主入口。  
> **迁移计划**：[`配置与监控_manifest迁移计划_CN.md`](配置与监控_manifest迁移计划_CN.md)（`pre_deploy` / `calibrate_roll` → experiments + `config/monitoring`）。  
> **顶层框架（ABC × 规则/树 统一视图）**：[`ABC统一研究框架_CN.md`](ABC统一研究框架_CN.md) —— 本文是它的"工具单"。  
> 配套执行手册：[`方法论_R_and_D流程_CN.md`](方法论_R_and_D流程_CN.md) · 架构背景：[`WORKFLOW_整体架构与管线改进计划_CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md)

---

## 0. 核心原则（30 秒）

| 代号 | 名称 | 能自动化？ | 改生产 yaml？ |
|------|------|-----------|---------------|
| **①** | 假设 / 离线筛查 | 工具固定，**假设由人定** | 否 |
| **②** | 因果验证 / 候选标定 | 实验步骤固定，**变体由人定** | 人审后才改 |
| **③** | 稳定性 / drift / 上线 contract | **可 cron** | 否（deploy 除外） |

**禁止**：在同一趟 `mlbot pipeline run` 里同时做 ①+②+③ 并 `--adopt` —— 无法归因。

**Phase 1 勿重复造轮子**：box × depth × lookback 等 B/C 规则扫描一律走 **`mlbot research scan`** + [`config/experiments/` 下 `rd_loop_*.yaml`](../../config/experiments/README.md)；binding 窗宽用 **N 次 `prepare-only`** 各产 parquet。仅 TPC `scan_tpc_pullback_lookback.py` 为 OHLC 重算 binding 的**例外**（见 [`LAYER_PROMOTION_CRITERIA.md`](../../config/experiments/LAYER_PROMOTION_CRITERIA.md)）。

**管线（pipeline yaml）的历史角色**：把 prefilter / gate / entry 各层脚本 **串成一条自动化 R&D 链**。  
新 doctrine：**各层脚本仍有用，但应分层、分阶段显式调用**；bundle yaml 仅保留监控与 contract。

---

## 1. 三阶段工具矩阵

### 阶段 A — 假设（①，不动 yaml）

| 层 | 首选工具 | 输入 | 产出 | 验收 |
|----|----------|------|------|------|
| **Regime** | `mlbot research scan condition-set` | `features_labeled.parquet` | scan md / json | `\|z\|>2` 且 meaningful Δpp |
| **Prefilter** | `mlbot research scan feature-plateau` / `mlbot research plateau` | 同上 | plateau md + `plateau.json` | plateau 存在 / 方向与锁定值一致 |
| **Gate** | `mlbot research plateau --kpi lift` / `rd_loop gate-plateau` | 同上 | lift plateau json + gate_draft + **`.skips.json`** | parity vs legacy optional |
| **Entry** | `mlbot research plateau`（加 regime+prefilter filter） | 同上 | entry 候选 + calibrate draft | 同上 |
| **跨层 IC** | `mlbot research ic` / `mlbot analyze factor-eval` | 同上 | IC@H json | 符号与层叙事不矛盾 |
| **分层** | `mlbot research segment` | 同上 | stratify json | 子群 lift 方向一致 |
| **编排** | `scripts/rd_loop.py` | `config/experiments/rd_loop_*.yaml` | `results/rd_loop/<topic>/` | 各 scan md 齐全；可选 **`monitor_bundle` draft** |
| **监控 baseline** | `mlbot research promote-baseline` | `monitor_bundle/bundle.json` | git monitoring JSON + PSI ref | Phase 5；见 [`研发与监控打通_CN.md`](研发与监控打通_CN.md) |
| **实验脚手架** | `mlbot research init <topic>` | `config/experiments/_template/` | `config/experiments/<topic>/` | rd_loop phase1 + promote_baseline.yaml |
| **人审写回** | `mlbot research calibrate` → `mlbot research promote` | plateau / batch json | draft yaml + skip manifest | promote 保留 `locked`；必 `--yes` 或 `--dry-run` |

**`mlbot research` 子命令**（统一入口，替代直接调 `quick_layer_scan` / 单层 optimize 做 ①）：

```bash
mlbot research scan condition-set --strategy tpc --layer prefilter --parquet ... --condition '...'
mlbot research scan feature-plateau --strategy tpc --feature pulse_z --operator '<=' --grid '0,1,0.1' ...
mlbot research ic --strategy tpc --features pulse_z --horizons 1,3,6 ...
mlbot research scan pair-scan --strategy tpc --pair-a 'vol_persistence:>:0.003,0.01' --pair-b 'vpin:<=:0.5,0.7' ...
mlbot research plateau --kpi lift --strategy tpc --layer gate --feature tpc_semantic_chop --operator '>' --grid '0.2,0.3,0.4,0.5' ...
mlbot research plateau --kpi snotio --snotio-mode entry_rr --strategy srb --subject 'feature:vol_persistence' ...  # full RR sim (needs OHLC+atr logs_gated)
mlbot research plateau --subject 'model.score:results/research/fit/tpc/prefilter/<run_id>' --grid '0.3,0.4,0.5' ...
mlbot research segment --strategy tpc --feature pulse_z --bins 5 ...
mlbot research fit --strategy tpc --layer prefilter ...        # 树 audit（gain + SHAP）
mlbot research calibrate --from-plateau results/.../gate_plateau_batch.json \
  --output results/.../gate_draft.yaml --strategy tpc
mlbot research promote --from results/.../gate_draft.yaml \
  --to config/strategies/tpc/archetypes/gate.yaml --layer gate --dry-run --yes
mlbot research compare results/a/plateau.json results/b/plateau.json
mlbot research robustness --kernel temporal|gate --feature ... --threshold ...
```

> 旧入口（`quick_layer_scan`、`optimize_*_plateau`）仍可用但会打印 **DEPRECATED**；新假设筛查一律走 `mlbot research *`。

#### calibrate skip 清单（gate batch）

`mlbot research calibrate` 在 **gate batch**（`kpi: lift` + `rules: {...}`）模式下，对无法安全写回的规则 **保持生产 yaml 不变**，并输出 skip 清单：

| 产出 | 路径 | 内容 |
|------|------|------|
| draft yaml | `--output` | 可写回规则的 threshold 更新；头部注释 `# calibrate skips (N): ...` |
| skip manifest | `<output>.skips.json`（默认）或 `--skips-output` | `{ "skip_count", "skips": [{ "rule_id", "reason", "detail" }] }` |

常见 `reason`：

| reason | 含义 |
|--------|------|
| `optimizer_skipped_locked` / `optimizer_frozen` | batch 阶段已跳过 locked/frozen 规则 |
| `optimizer_status_not_applicable` | 优化未产出可写回状态（如 `no_valid_threshold`） |
| `unsafe_any_of` | `when: any_of` 不能单点改写（会 OR→AND） |
| `unsafe_band_no_interval` | 双边界 band 规则缺 `threshold_interval`，单点会丢上/下界 |
| `missing_optimization_fields` / `missing_recommended_threshold` | batch json 字段不全 |

**人审时**：先看 `.skips.json` 和 `gate_plateau_summary.md`，再决定是否手工改 band / any_of 规则或重跑 batch（加 `write_back_intervals: true`）。

#### promote（显式写生产，无 auto-promote）

```bash
# 预览 diff，不写盘（推荐第一步）
mlbot research promote --from results/.../gate_draft.yaml \
  --to config/strategies/tpc/archetypes/gate.yaml --layer gate --dry-run --yes

# 确认后写盘（自动 timestamp backup + locked merge）
mlbot research promote --from results/.../gate_draft.yaml \
  --to config/strategies/tpc/archetypes/gate.yaml --layer gate --yes
```

- `locked` / `frozen` / `promote_never_disable` 规则 **始终保留生产侧**。
- 无 `--yes` 且非 `--dry-run` 时 **拒绝执行**（exit 2）。

**数据准备**（按需，非每次）：

```bash
# 真实数据（推荐）：特征 + 标签，不训模型
RUN_ID=train_final_$(date +%Y%m%d_%H%M%S)
mlbot train final --no-docker --prepare-only \
  -c config/strategies/tpc \
  -t 120T \
  --symbol BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --start-date 2024-01-01 --end-date 2026-04-01 \
  --output-root results/train_final/tpc/${RUN_ID}
# → results/train_final/tpc/${RUN_ID}/tpc/features_labeled.parquet

# 取最近一次产物（省略 RUN_ID 时）
PARQ=$(ls -t results/train_final/tpc/train_final_*/tpc/features_labeled.parquet 2>/dev/null | head -1)

# smoke 链路学习（合成 parquet，数字无实盘意义）：
PYTHONPATH=src:scripts python scripts/_validation_smoke_assets.py
# → results/validation_smoke/tpc/features_labeled.parquet
```

> 其它策略把 `-c` / `-t` / `--output-root` 换成对应目录即可（如 `bpc`、`-t 240T`）。`--prepare-only` 只跑特征管线 + 打标签，**跳过** LightGBM 训练。

#### 三通道端到端示例（TPC · 树 · chop grid）

> 本文是 **工具单**；三条通道 **数据形态与写回对象不同**，不要混用同一套「只扫 features_labeled label」流程。

| | **TPC（B 规则层）** | **树（fast_scalp / short_term_swing）** | **chop grid（C 段级 KPI）** |
|--|---------------------|------------------------------------------|----------------------------|
| 数据 | `features_labeled.parquet` | 同上 | `grid_segments.csv` + features → `seg_labeled.parquet` |
| ① 主工具 | `scan` / `plateau --kpi lift` / `rd_loop gate-plateau` | `factor-eval` / `research ic` + `fit` + `plateau --kpi snotio` | `event_backtest` → `_build_grid_segment_labels` → `scan condition-set` on seg KPI |
| ② 验证 | `event_backtest --variant-grid`（recent + bull） | 同上 | `chop_grid_semantic_proxy_grid.yaml` |
| 写回 yaml | `archetypes/gate.yaml` 等 | 树 `features` / `model` / `backtest` | `chop_grid` prefilter / grid config |
| 示例 yaml | [`rd_loop_tpc_gate_plateau.yaml`](../../config/experiments/tpc/rd_loop_tpc_gate_plateau.yaml) | [`rd_loop_fast_scalp_ic_plateau.yaml`](../../config/experiments/fast_scalp/rd_loop_fast_scalp_ic_plateau.yaml) | [`chop_grid_semantic_proxy_grid.yaml`](../../config/experiments/chop_grid/chop_grid_semantic_proxy_grid.yaml) |

**TPC（B）— 条件筛查 → gate lift → calibrate → variant-grid → promote**

```bash
# 数据：smoke 或 train_final（二选一）
# PYTHONPATH=src:scripts python scripts/_validation_smoke_assets.py
# PARQ=results/validation_smoke/tpc/features_labeled.parquet
RUN_ID=train_final_$(date +%Y%m%d_%H%M%S)
mlbot train final --no-docker --prepare-only \
  -c config/strategies/tpc -t 120T \
  --symbol BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --start-date 2024-01-01 --end-date 2026-04-01 \
  --output-root results/train_final/tpc/${RUN_ID}
PARQ=results/train_final/tpc/${RUN_ID}/tpc/features_labeled.parquet

# ①A 条件假设（不动 yaml）
mlbot research scan condition-set --strategy tpc --layer gate \
  --parquet "$PARQ" --label success_no_rr_extreme \
  --filter 'tpc_semantic_chop<=0.4' \
  --condition 'H: abs(ema_1200_position)>0.10'

# ①B Gate lift 单特征 / batch（rd_loop 编排见 rd_loop_tpc_gate_plateau.yaml）
mlbot research plateau --kpi lift --strategy tpc --layer gate \
  --parquet "$PARQ" --feature tpc_semantic_chop --operator gt \
  --grid '0.2,0.3,0.4,0.5'

PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/tpc/rd_loop_tpc_gate_plateau.yaml

mlbot research calibrate \
  --from-plateau results/rd_loop/tpc_gate_plateau/quick_scan/gate_plateau/gate_plateau_batch.json \
  --output results/rd_loop/tpc_gate_plateau/gate_draft.yaml --strategy tpc

# ② 因果验证（必做）
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/tpc/tpc_variant_grid_smoke.yaml

# 人审后 promote（TPC 生产 gate 多为 locked；batch 默认 skip_locked）
mlbot research promote --from results/rd_loop/tpc_gate_plateau/gate_draft.yaml \
  --to config/strategies/tpc/archetypes/gate.yaml --layer gate --dry-run --yes
```

**树通道（fast_scalp / short_term_swing）— IC → fit → τ plateau → variant-grid**

> 树通道 **不走** B 的 prefilter/gate/entry 分层 yaml；① 以 IC + 树 audit + entry τ 为主。详见 [`短期树独立策略_设计与落地_CN.md`](短期树独立策略_设计与落地_CN.md)。

```bash
PARQ=results/train_final/fast_scalp/<run_id>/fast_scalp/features_labeled.parquet

# ①A 因子 IC（树通道首选之一）
mlbot analyze factor-eval --strategy fast_scalp \
  --features-yaml config/strategies/fast_scalp/features.yaml \
  --parquet "$PARQ" --ic-decay-lags 1,3,5,10,20

mlbot research ic --strategy fast_scalp --parquet "$PARQ" \
  --features pulse_z,macd_atr,bb_width_normalized_pct \
  --horizons 1,3,5,10,20 --target forward_rr

# ①B 树 audit + entry τ（proxy 或 entry_rr）
mlbot research fit --strategy fast_scalp --layer prefilter --parquet "$PARQ"

mlbot research plateau --kpi snotio --snotio-mode proxy \
  --strategy fast_scalp --layer entry --parquet "$PARQ" \
  --feature pulse_z --operator '<=' --grid '0,1,0.1'

PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/fast_scalp/rd_loop_fast_scalp_ic_plateau.yaml

# ② variant-grid（short_term_swing 换对应 grid yaml）
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/fast_scalp/fast_scalp_direction_grid.yaml
```

**chop grid（C）— grid 回测 → segment KPI 桥 → 条件扫描 → 变体 grid**

> chop grid 的 ① **不是** 直接在 `features_labeled` 上扫 `success_no_rr_extreme`；核心是 **段内 KPI**（`seg_total_r_over_dd` 等）。

```bash
# ② 先跑变体（engine=chop_grid）
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/chop_grid/chop_grid_semantic_proxy_grid.yaml
# → results/chop_grid/experiments/<variant>/grid_segments.csv

# ① segment label 桥
PYTHONPATH=src:scripts python scripts/_build_grid_segment_labels.py \
  --segments results/chop_grid/experiments/baseline_recent/grid_segments.csv \
  --features-parquet results/validation_smoke/chop_grid/features_labeled.parquet \
  --out results/chop_grid/experiments/baseline_recent/seg_labeled.parquet

# ① 段 KPI 条件扫描
mlbot research scan condition-set --strategy chop_grid \
  --parquet results/chop_grid/experiments/baseline_recent/seg_labeled.parquet \
  --label seg_total_r_over_dd \
  --condition 'high_chop: bpc_semantic_chop>=0.50' \
  --condition 'low_chop: bpc_semantic_chop<0.50'

# 决策留痕（C 专用 template）
PYTHONPATH=src:scripts python scripts/_new_decision_doc.py \
  --experiment-index results/chop_grid/experiments/EXPERIMENT_INDEX.json \
  --topic chop_grid_proxy_sweep --topic-template c_semantic_proxy \
  --out docs/decisions/chop_grid_proxy_20260526.md
```

操作记录与 smoke 判读：[`ABC验证操作记录_20260526_CN.md`](ABC验证操作记录_20260526_CN.md)。

---

### 阶段 B — 验因果（②，只动 config_experiments）

| 任务 | 工具 | 产出 | 验收 |
|------|------|------|------|
| 改 1–2 条规则 / 阈值 | `cp -r config/strategies → config_experiments/<variant>_strategies` | variant 策略树 | diff 仅 1–2 个 yaml |
| 双段 R-multiple | `event_backtest --variant-grid` | `EXPERIMENT_INDEX.json` + trades | **recent + bull** 两段 Pareto |
| 决策留痕 | `_new_decision_doc.py` | `docs/decisions/*.md` | 变体表 + 双段结果 + by-side |
| 写生产 | `mlbot research promote`（或人工 `cp`） | `config/strategies/*/archetypes/*.yaml` | 与 decision doc 一致；locked 保留 |

**树通道（fast_scalp / short_term_swing）**：① 用 `factor-eval` / ic-decay 定 H → `mlbot train final` → τ plateau；② 仍走 variant-grid 双段。

---

### 阶段 C — 监控与上线（③，固定流程）

> **权威命令**：[`漂移监控_mlbot_monitor_CN.md`](漂移监控_mlbot_monitor_CN.md)（`mlbot monitor`）。

| 频率 | 工具 | 产出 | 验收 |
|------|------|------|------|
| **周** | `mlbot monitor weekly`（或 `watchdog` + `drift`） | `results/monitoring/weekly_watchdog/<ts>/` | exit 0；IC/PSI / plateau 无 ALERT |
| **月** | `calibrate_roll.default` `--stage rolling_sim` **或** 固定 config 的 `event_backtest` | `stitched_summary` / ledger | sharpe、trades 趋势正常 |
| **上线前** | `pre_deploy_replay.yaml` **或** `pre_deploy_contract_checks.py` | `contract_checks.json` | 无 BLOCKED |
| **上线** | `deploy_config_to_live.py` | live 镜像 | 人 confirm |

---

## 2. Pipeline yaml 状态（ROUTINE_R&D_DEPRECATED）

| YAML | 日常 R&D | 仍保留用途 |
|------|----------|------------|
| `calibrate_roll.default.yaml` | 否（③ 月监控） | ✅ 固定 yaml 多月 replay + ledger |
| `calibrate_roll.no_prefilter_threshold_search.yaml` | 否 | ✅ 阈值手工锁死时的 turbo 变体 |
| `pre_deploy_replay.yaml` | 否 | ✅ 上线前 frozen replay + contract |
| `research_roll.features_on.yaml` | **ROUTINE_R&D_DEPRECATED** | 遗留：季度 bundle 只读体检（不 adopt optimize 产出） |
| `validate_static.full_study.yaml` | **ROUTINE_R&D_DEPRECATED** | 遗留：可选 formal `deploy_gate` 整段评分 |
| `validate_static.constrained.yaml` | **ROUTINE_R&D_DEPRECATED** | 遗留：可被 variant-grid + 单层脚本替代 |

> 标记 **ROUTINE_R&D_DEPRECATED** 的 yaml **不删除**（历史 run 路径、adopt 脚本仍引用），但 **不应作为新实验入口**。

---

## 3. 端到端流程（BPC 示例）

```
[设计] 经验 → features.yaml + archetypes（语义锚 locked）
[数据] train final --prepare-only → features_labeled.parquet

[①] rd_loop / mlbot research scan|ic|plateau
     例：breakout_strength label 反向 → 假设「去掉该锚」

[②] config_experiments/bpc_no_breakout_strategies/
     event_backtest --variant-grid（recent + bull）

[人审] cp prefilter.yaml + docs/decisions/bpc_*.md

[③ 周] regime_watchdog / regime_drift_monitor
[③ 月] calibrate_roll rolling_sim（或月 cron event_backtest）
[③ 上线] pre_deploy_replay → deploy

不跑：research_roll / full_study / constrained（除非要 legacy deploy_gate JSON）
```

---

## 4. 各层 pipeline 脚本 vs rd_loop — 深度对照

> **结论先说**：`rd_loop` today 只是 **① 的薄编排**（调 `mlbot research` + 可选 variant-grid + decision doc）。  
> **各层 optimize 脚本比 rd_loop 复杂得多**，价值在 **② 的单层精细标定**；不应删掉，也不应无实验设计地塞进 bundle pipeline。

### 4.1 能力对比表

| 层 | Pipeline 内常用脚本 | 做什么 | rd_loop / quick_layer_scan 覆盖？ | 何时单独用 pipeline 脚本 |
|----|---------------------|--------|-----------------------------------|-------------------------|
| **Prefilter** | `analyze_archetype_feature_stratification.py` | meta 多法（KS / mean_effect / …）在 predictions 上搜**候选特征** + `--promote` | ❌ 不覆盖（① 只做 label plateau，不训树不 promote） | 探索 **新 prefilter 候选列**（需人审，禁止 auto-promote 进生产） |
| | `locked_prefilter_parquet_tune.py` | locked 规则在 parquet 上坐标 plateau | ⚠️ 部分（`feature-plateau` 更简单） | 已定 locked 规则，要在 holdout 上出 **数值写回提案** |
| | `tune_locked_prefilter_thresholds.py` | 多 case / 多窗 grid + summary.json | ❌ | 批量 locked 阈值网格（constrained yaml 曾包这一层） |
| **Gate** | `optimize_gate_unified.py` | lift + plateau + robustness；区间 deny | ⚠️ `condition-set` / `pair-scan` 只做 label 效应 | gate **数值区间**已定结构，要在 logs 上精调 τ / deny band。**已支持 `features_labeled.parquet` 输入**（路线 B，无需 pipeline run） |
| **Entry** | `optimize_entry_filter_plateau.py` ⚠️ DEPRECATED | 按 filter 扫 snotio plateau | ⚠️ 推荐 `mlbot research plateau --kpi snotio [--snotio-mode entry_rr]` | entry OR 规则 **逐条** 扫阈值；`entry_rr` 需 OHLC+atr+方向 |
| **Direction** | `direction_strict_validation.py` | 方向公式 + 可选 compare-features | ❌ | 动 direction.yaml 前 |
| **跨层** | `posthoc_layer_effectiveness.py` | 各层 rule pass/fail vs success 的 effect/z | ⚠️ 类似 condition-set，但更贴 yaml 规则 | pre_deploy **strict-locked-features**；复盘已锁定规则 |
| **执行** | `event_backtest` / pipeline_events | 1m 重放、R-multiple | ✅ variant-grid 直接用 | ② 验因果 **必用** |
| **SHAP** | pipeline + `multileg_feature_selection` | 特征重要性 → features 候选 | ❌ rd_loop 无 SHAP | **仅 audit**（`audit_only: true`）；禁止 auto-promote |

### 4.1bis 输入文件 / 是否需要先跑 pipeline

> 直接判断"我现在能不能跑"，决定要不要先 `mlbot train final`。

| 脚本 | 输入文件 | 需要先跑 pipeline？ | 备注 |
|------|---------|---------------------|------|
| `quick_layer_scan.py` | `features_labeled.parquet` | **不需要**（`--prepare-only` 即可） | 1-2 分钟出 markdown |
| `rd_loop.py` | 同上 | **不需要** | 内部调 scan + 可选 variant-grid |
| `optimize_gate_unified.py` | `features_labeled.parquet` ✅ 或 `predictions.parquet` | **不需要**（路线 B） | `forward_rr` 在；自动生成 `is_good` |
| `optimize_entry_filter_plateau.py` | `features_labeled.parquet` ✅ 或 `predictions.parquet` | **不需要**（路线 B） | 缺 `gate_decision` 时打 INFO 并继续 |
| `locked_prefilter_parquet_tune.py` | `features_labeled.parquet` | **不需要** | 直接在 parquet 上坐标 plateau |
| `tune_locked_prefilter_thresholds.py` | `features_labeled.parquet` | **不需要** | 多 case 多窗 grid |
| `analyze_archetype_feature_stratification.py` | `predictions.parquet`（内含 `_train_lgb`） | **需要**（含模型训练）；除非加 `--no-model` flag（未实现） | meta 多法搜候选特征 |
| `posthoc_layer_effectiveness.py` | `predictions.parquet`（含 gate/entry decision 列） | **需要** | 复盘已锁规则在生产 funnel 中的 effect |
| `direction_strict_validation.py` | `predictions.parquet`（含 direction 决策） | **需要** | direction.yaml 变更前的检验 |
| `factor-eval`（mlbot analyze） | `features_labeled.parquet` | **不需要** | IC 衰减 / best_lag |
| `event_backtest --variant-grid` | `strategies_root` + 日期窗 + 数据 | **不需要 train pipeline**（直接 1m 重放） | ② 唯一可信因果验证 |
| `regime_watchdog.py` | `features.parquet`（recent window） + baseline | **不需要** | 周度 cron |

**口径**：
- ✅ **新支持** `features_labeled.parquet` = 路线 B 已落地，详见 [`ABC统一研究框架_CN.md`](ABC统一研究框架_CN.md) §5。
- "**需要**" = 该脚本逻辑里有模型 score / decision 列依赖，没跑完 pipeline 就无法运行。
- 凡输入是 `features_labeled.parquet` 的，工作流：`mlbot train final --prepare-only -c ... → 直接调脚本`；不需要等模型训练。

---

### 4.2 为什么 rd_loop「看起来更简单」

| rd_loop 设计目标 | 老 pipeline bundle 设计目标 |
|------------------|----------------------------|
| 2 分钟排除错误假设 | 2 小时「自动找最优」 |
| 单特征 / 单条件可归因 | 多层串联 over-fit 同一段噪声 |
| 输出 markdown 证据 | 输出 draft yaml 诱惑直接 adopt |

**rd_loop 简单是刻意的** —— 它不应复制 `analyze_archetype_*` + `optimize_gate_*` 的全部分支。

### 4.3 推荐分工（当前仓库可落地）

```
假设层（①）     → rd_loop / mlbot research scan|ic|plateau  （快、全层扫一遍）
                → posthoc_layer_effectiveness （贴 yaml 规则核对）

单层精标（②b）  → 在 variant-grid 证明「方向对」之后，按需调用：
                  · gate: optimize_gate_unified.py
                  · entry: optimize_entry_filter_plateau.py
                  · locked prefilter 数值: locked_prefilter_parquet_tune.py
                → 产出 proposal yaml，人审后再 cp 生产

因果层（②a）    → event_backtest --variant-grid（必做）

监控（③）       → watchdog + calibrate_roll / 月 event_backtest + pre_deploy contract
```

### 4.4 rd_loop 增强（部分已落地）

| 增强 | 状态 | 说明 |
|------|------|------|
| `rd_loop` mode: `entry-plateau` | ✅ | auto-loop `entry_filters.yaml` → `entry_plateau_scan` + `logs_gated.parquet` |
| `rd_loop` step: `gate-plateau` | ✅ | batch/single lift；内核 `src/research/stat_kernels/gate_optimize.py` |
| `rd_loop` step: `locked-prefilter-tune` | ✅ | locked 数值提案 | `locked_prefilter_parquet_tune.py` |
| 仍 **不** 纳入 | — | 全自动 promote / 多层串联 optimize | pipeline bundle |

`entry-plateau` 示例：[`config/experiments/srb/rd_loop_srb_entry_plateau.yaml`](../../config/experiments/srb/rd_loop_srb_entry_plateau.yaml)  
`gate-plateau` 示例：[`config/experiments/tpc/rd_loop_tpc_gate_plateau.yaml`](../../config/experiments/tpc/rd_loop_tpc_gate_plateau.yaml)

---

## 5. calibrate_roll vs event_backtest vs pre_deploy

| | calibrate_roll（turbo，optimize 全关） | 裸 event_backtest | pre_deploy_replay |
|--|----------------------------------------|-------------------|-------------------|
| 改生产 yaml | 否 | 否 | 否 |
| 多月 + 仓位续跑 | 有 | 需自行拼 | 有（rolling_sim） |
| monthly ledger | 有 | 无 | 有 |
| locked_features contract | 无 | 无 | **有（BLOCKED）** |
| regime.yaml 检查 | 无 | 无 | 有 |
| plateau_stability | 无 | 无 | yaml 有；**代码暂 deferred** |

月 drift：**固定 config 的 replay 趋势**，不是调参。若 calibrate_roll 过重，可改为 cron 跑固定窗 `event_backtest`。

---

## 6. 反模式

| 反模式 | 应用 |
|--------|------|
| 跑 `research_roll` 并 `--adopt` | ❌ |
| 用 `full_study` **发现**特征（在 rd_loop 之前） | ❌ |
| label scan 通过就改 yaml（跳过 variant-grid） | ❌ |
| 把 `optimize_gate_unified` 包进月度 cron | ❌ |
| 删掉 layer 脚本「因为 rd_loop 有了」 | ❌ — 它们是 **②b 精标** 武器 |

---

## 7. 命令速查

```bash
# ① 假设（BPC 通用）
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/bpc/rd_loop_bpc.yaml

# ① TPC gate-plateau → calibrate（含 skip 清单）
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/tpc/rd_loop_tpc_gate_plateau.yaml
mlbot research calibrate \
  --from-plateau results/rd_loop/tpc_gate_plateau/quick_scan/gate_plateau/gate_plateau_batch.json \
  --output results/rd_loop/tpc_gate_plateau/gate_draft.yaml --strategy tpc
# → gate_draft.yaml + gate_draft.yaml.skips.json（若有 skip）

# ① 树通道（fast_scalp）
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/fast_scalp/rd_loop_fast_scalp_ic_plateau.yaml
mlbot analyze factor-eval --strategy fast_scalp \
  --features-yaml config/strategies/fast_scalp/features.yaml \
  --parquet results/train_final/fast_scalp/.../features_labeled.parquet \
  --ic-decay-lags 1,3,5,10,20

# ① chop grid（段 KPI 桥 + 扫描）
PYTHONPATH=src:scripts python scripts/_build_grid_segment_labels.py \
  --segments results/chop_grid/experiments/<variant>/grid_segments.csv \
  --features-parquet results/validation_smoke/chop_grid/features_labeled.parquet \
  --out results/chop_grid/experiments/<variant>/seg_labeled.parquet
mlbot research scan condition-set --strategy chop_grid \
  --parquet results/chop_grid/experiments/<variant>/seg_labeled.parquet \
  --label seg_total_r_over_dd --condition 'high_chop: bpc_semantic_chop>=0.50'

# 人审写回（preview → 确认）
mlbot research promote --from results/.../gate_draft.yaml \
  --to config/strategies/tpc/archetypes/gate.yaml --layer gate --dry-run --yes
mlbot research promote --from results/.../gate_draft.yaml \
  --to config/strategies/tpc/archetypes/gate.yaml --layer gate --yes

# ② 验因果
PYTHONPATH=src:scripts python -m scripts.event_backtest --variant-grid \
  config/experiments/<your_grid>.yaml

# ③ 周监控（见 漂移监控_mlbot_monitor_CN.md）
export WATCHDOG_PARQUET=results/<recent>/features_labeled.parquet
mlbot monitor weekly
# 或：mlbot monitor drift --window-parquet "$WATCHDOG_PARQUET" --emit-rd-loop-suggestions

# ③ 月监控
mlbot pipeline run --all \
  --config config/strategies/bpc/research/calibrate_roll.default.yaml \
  --stage rolling_sim --skip-shap

# ③ 上线 contract
mlbot pipeline run --all \
  --config config/strategies/bpc/research/pre_deploy_replay.yaml \
  --stage rolling_sim --skip-shap
```

---

## 8. 与 README / 老文档的关系

### 8.1 `docs/strategy` 文档分工（避免命令口径分裂）

| 文档 | 写什么 | 命令写在哪 |
|------|--------|------------|
| **本文** | ①②③ 工具矩阵、pipeline 弃用、`rd_loop` vs 各层 optimize | **§1 子命令全文** + §7 速查 |
| [`方法论_R_and_D流程_CN.md`](方法论_R_and_D流程_CN.md) | 五步流程、双段 Pareto、promote / watchdog cron | 引用本文 §1；§2 为可 copy 示例 |
| [`ABC统一研究框架_CN.md`](ABC统一研究框架_CN.md) | ABC、数据三件套、Phase 0–4 | §3.2 与本文对齐 |
| [`WORKFLOW_整体架构与管线改进计划_CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md) | 架构、层宪法、里程碑 | 历史「待建 quick_layer_scan」段落仅作背景，**执行以本文为准** |
| [`ABC新流程验证checklist_CN.md`](ABC新流程验证checklist_CN.md) | 验收 checklist | 已改为 `mlbot research`；遗留 `quick_layer_scan` 仅对拍 |
| [`label_scan_vs_IC_说明_CN.md`](label_scan_vs_IC_说明_CN.md) | Δpp / IC 判读语义 | 统计含义不变；CLI 名见本文 §1 |

**统一规则**：新假设筛查 = `mlbot research *` 或 `rd_loop`；② = `event_backtest --variant-grid`；③ = watchdog / calibrate_roll / pre_deploy。  
`scripts/quick_layer_scan.py` 仅遗留 / 单测对拍，不在新文档示例里当主入口。

### 8.2 其它引用

- **README §2–§3**：已改为指向本文；老「calibrate_roll 调阈值 / research_roll 季度结构」描述作废。
- **WORKFLOW §5 里程碑 M1/M4**：turbo 纯验证、SHAP audit-only 与本文一致。
- **层脚本源码**：仍挂在 `scripts/auto_research_pipeline.py` 的 `run_strategy_pipeline` 路径下；拆出来单独调用 = 正确用法。
