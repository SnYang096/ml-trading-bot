# R&D 工具矩阵（B 系统主线）

> **替代口径**：日常 R&D 与运维以本文为准，不再把 `research_roll` / `validate_static.*` 当作发现特征或调阈值的主入口。  
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

**管线（pipeline yaml）的历史角色**：把 prefilter / gate / entry 各层脚本 **串成一条自动化 R&D 链**。  
新 doctrine：**各层脚本仍有用，但应分层、分阶段显式调用**；bundle yaml 仅保留监控与 contract。

---

## 1. 三阶段工具矩阵

### 阶段 A — 假设（①，不动 yaml）

| 层 | 首选工具 | 输入 | 产出 | 验收 |
|----|----------|------|------|------|
| **Regime** | `mlbot research scan condition-set` | `features_labeled.parquet` | scan md / json | `\|z\|>2` 且 meaningful Δpp |
| **Prefilter** | `mlbot research scan feature-plateau` / `mlbot research plateau` | 同上 | plateau md + `plateau.json` | plateau 存在 / 方向与锁定值一致 |
| **Gate** | `mlbot research scan condition-set` / `pair-scan` / `mlbot research ic` | 同上 | deny 区效应 / IC decay | deny 区 label / IC 与叙事一致 |
| **Entry** | `mlbot research plateau`（加 regime+prefilter filter） | 同上 | entry 候选 + calibrate draft | 同上 |
| **跨层 IC** | `mlbot research ic` | 同上 | IC@H json | 符号与层叙事不矛盾 |
| **分层** | `mlbot research segment` | 同上 | stratify json | 子群 lift 方向一致 |
| **编排** | `scripts/rd_loop.py`（内部仍可调 `quick_layer_scan`） | `config/experiments/rd_loop_*.yaml` | `results/rd_loop/<topic>/` | 各 scan md 齐全 |

**`mlbot research` 子命令**（统一入口，替代直接调 `quick_layer_scan` / 单层 optimize 做 ①）：

```bash
mlbot research scan condition-set --strategy tpc --layer prefilter --parquet ... --condition '...'
mlbot research scan feature-plateau --strategy tpc --feature pulse_z --operator '<=' --grid '0,1,0.1' ...
mlbot research ic --strategy tpc --features pulse_z --horizons 1,3,6 ...
mlbot research scan pair-scan --strategy tpc --pair-a 'vol_persistence:>:0.003,0.01' --pair-b 'vpin:<=:0.5,0.7' ...
mlbot research plateau --kpi snotio --snotio-mode proxy --feature tpc_pullback_depth ...   # mean(forward_rr)
mlbot research plateau --kpi snotio --snotio-mode entry_rr --strategy srb --subject 'feature:vol_persistence' ...  # full RR sim (needs OHLC+atr logs_gated)
mlbot research plateau --subject 'model.score:results/research/fit/tpc/prefilter/<run_id>' --grid '0.3,0.4,0.5' ...
mlbot research segment --strategy tpc --feature pulse_z --bins 5 ...
mlbot research fit --strategy tpc --layer prefilter ...        # 树 audit（gain + SHAP）
mlbot research calibrate --plateau-json results/.../plateau.json ...
mlbot research compare results/a/plateau.json results/b/plateau.json
mlbot research robustness --kernel temporal|gate --feature ... --threshold ...
```

> 旧入口（`quick_layer_scan`、`optimize_*_plateau`）仍可用但会打印 **DEPRECATED**；新假设筛查一律走 `mlbot research *`。

**数据准备**（按需，非每次）：

```bash
mlbot train final --no-docker --prepare-only \
  -c config/strategies/bpc \
  --output-dir results/train_final/bpc/<run_id>
# → features_labeled.parquet
```

---

### 阶段 B — 验因果（②，只动 config_experiments）

| 任务 | 工具 | 产出 | 验收 |
|------|------|------|------|
| 改 1–2 条规则 / 阈值 | `cp -r config/strategies → config_experiments/<variant>_strategies` | variant 策略树 | diff 仅 1–2 个 yaml |
| 双段 R-multiple | `event_backtest --variant-grid` | `EXPERIMENT_INDEX.json` + trades | **recent + bull** 两段 Pareto |
| 决策留痕 | `_new_decision_doc.py` | `docs/decisions/*.md` | 变体表 + 双段结果 + by-side |
| 写生产 | 人工 `cp` | `config/strategies/*/archetypes/*.yaml` | 与 decision doc 一致 |

**树通道（fast_scalp / short_term_swing）**：① 用 `factor-eval` / ic-decay 定 H → `mlbot train final` → τ plateau；② 仍走 variant-grid 双段。

---

### 阶段 C — 监控与上线（③，固定流程）

| 频率 | 工具 | 产出 | 验收 |
|------|------|------|------|
| **周** | `regime_watchdog.py` | `report.json` | exit 0；IC/PSI 无 ALERT |
| **周** | `regime_drift_monitor.py` | drift report | plateau 未漂出 |
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

[①] rd_loop / quick_layer_scan
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

> **结论先说**：`rd_loop`  today 只是 **① 的薄编排**（调 `quick_layer_scan` + 可选 variant-grid + decision doc）。  
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
假设层（①）     → rd_loop / quick_layer_scan  （快、全层扫一遍）
                → posthoc_layer_effectiveness （贴 yaml 规则核对）

单层精标（②b）  → 在 variant-grid 证明「方向对」之后，按需调用：
                  · gate: optimize_gate_unified.py
                  · entry: optimize_entry_filter_plateau.py
                  · locked prefilter 数值: locked_prefilter_parquet_tune.py
                → 产出 proposal yaml，人审后再 cp 生产

因果层（②a）    → event_backtest --variant-grid（必做）

监控（③）       → watchdog + calibrate_roll / 月 event_backtest + pre_deploy contract
```

### 4.4 后续可增强 rd_loop 的方向（未实现）

| 增强 | 作用 | 对应现有脚本 |
|------|------|--------------|
| `rd_loop` step: `gate-plateau` | ①→② 桥接：scan 显著后调 gate optimizer 出 proposal | `optimize_gate_unified.py` |
| `rd_loop` step: `entry-plateau` | 同上 entry | `optimize_entry_filter_plateau.py` |
| `rd_loop` step: `locked-prefilter-tune` | locked 数值提案 | `locked_prefilter_parquet_tune.py` |
| 仍 **不** 纳入 | 全自动 promote / 多层串联 optimize | pipeline bundle |

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
# ① 假设
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/rd_loop_bpc.yaml

# ② 验因果
PYTHONPATH=src:scripts python -m scripts.event_backtest --variant-grid \
  config/experiments/<your_grid>.yaml

# ③ 周监控
PYTHONPATH=src:scripts python scripts/regime_watchdog.py \
  --strategies bpc,tpc,me,srb \
  --window-parquet results/<recent>/features_labeled.parquet \
  --baseline-json config/monitoring/regime_watchdog_baseline.json

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

- **README §2–§3**：已改为指向本文；老「calibrate_roll 调阈值 / research_roll 季度结构」描述作废。
- **WORKFLOW §5 里程碑 M1/M4**：turbo 纯验证、SHAP audit-only 与本文一致。
- **层脚本源码**：仍挂在 `scripts/auto_research_pipeline.py` 的 `run_strategy_pipeline` 路径下；拆出来单独调用 = 正确用法。
