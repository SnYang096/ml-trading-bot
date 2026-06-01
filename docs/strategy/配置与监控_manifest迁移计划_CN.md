# 配置与监控 Manifest 迁移计划

> **状态**：计划稿（2026-06-02）  
> **目的**：把 R&D / 监控 / 上线从 `config/strategies/*/research/` 的 pipeline 模板，迁到与 `config/experiments` 同构的 **YAML manifest + CLI**，`config/strategies` 只保留生产契约。  
> **关联**：[`config/experiments/README.md`](../../config/experiments/README.md) · [`漂移监控_mlbot_monitor_CN.md`](漂移监控_mlbot_monitor_CN.md) · [`R&D工具矩阵_CN.md`](R&D工具矩阵_CN.md) §2 · [`LAYER_PROMOTION_CRITERIA.md`](../../config/experiments/LAYER_PROMOTION_CRITERIA.md)

---

## 1. 架构共识（已确认）

### 1.1 目录分工（目标态）

| 路径 | 角色 | 放什么 | 不放什么 |
|------|------|--------|----------|
| **`config/experiments/<YYYYMMDD>_<topic>/`** | R&D 实验卡片 | `rd_loop_*.yaml`、`*_grid.yaml`、`DECISION.md`、可选 **pre_deploy / holdout** grid | 生产 `archetypes` 终态（promote 后才进 strategies） |
| **`config_experiments/<topic>_strategies/`** | 变体策略树 | A/B 快照（variant-grid 的 `strategies_root`） | 日常生产配置 |
| **`config/strategies/<slug>/`** | **生产契约** | `archetypes/*.yaml`、`features*.yaml`、`meta.yaml` | `research_roll`、静态研究模板、月/周 job |
| **`config/monitoring/`**（新建） | 运维 / 漂移 job | 周 watchdog+drift、月固定 config replay、contract 门禁 manifest | 假设扫描、变体发现 |
| **`config/market_segment.yaml`** | 日历分段 | bear / bull / recent 等日期（回测 + monitor 窗对齐） | — |
| **`live/highcap/config/strategies/`** | 实盘镜像 | `deploy_config_to_live.py` 从 strategies 拷贝 | 实验草稿 |

### 1.2 三阶段与工具（不变）

| 阶段 | 做什么 | 目标入口 | 改生产 yaml？ |
|------|--------|----------|---------------|
| **① 假设** | 特征 / plateau / IC | `mlbot research` / `rd_loop`（`config/experiments`） | 否 |
| **② 验因果** | 分段 Pareto、双段/多段 R | `event_backtest --variant-grid` + `segment_matrix` + `market_segment.yaml` | 否（人审 promote 才改） |
| **③ 监控 / 门禁** | drift、固定 config replay、contract | **`mlbot monitor`** + monitoring manifest（计划） | 否 |

**原则**：管线只做验证，不做研究；禁止同一趟 bundle 里 ①+②+③ 并 `--adopt`。

### 1.3 废弃口径（已生效，勿新开入口）

| 遗留物 | 状态 | 替代 |
|--------|------|------|
| `research_roll.features_on.yaml` | `ROUTINE_R&D_DEPRECATED` | `config/experiments` + `event_backtest --variant-grid` |
| `validate_static.*.yaml` | 同上 | 同上；上线门禁迁 manifest（见 §3.2） |
| `auto_research_pipeline` 作 **发现/调参/adopt** | 不推新实验 | 仍是 `mlbot pipeline run` 的**执行引擎**（迁移期） |
| `auto_research_pipeline --compare-only` 当 Drift-Only | **未实现** | `mlbot monitor`（见 [`漂移监控_mlbot_monitor_CN.md`](漂移监控_mlbot_monitor_CN.md)） |

**结论**：本地/远程 **不必再跑 research_roll** 做 promote 决策；多周期因果用 **segment_matrix** 即可。长期「整段体检」仅为可选遗产路径。

### 1.4 Live 与实验 snapshot

1. **研究期**：`config_experiments/<topic>_strategies/` 供 variant-grid 对照。  
2. **人审 promote**：`mlbot research promote` → **`config/strategies/<slug>/archetypes/`**（canonical 生产）。  
3. **实盘**：`deploy_config_to_live.py` → **`live/highcap/config/strategies/`**。

### 1.5 远程优先 · CMS · Telegram（2026-06 共识）

| 原则 | 说明 |
|------|------|
| **主路径在远程** | systemd cron 自动跑 `mlbot monitor`；本地仅人审后复核 |
| **数据** | 远程 **bus → export-window**（不重算特征）；长窗/IC 才 archive batch 或 prepare（§7.0–7.4） |
| **告警** | ALERT → **Telegram** + **CMS 卡片**（非仅写磁盘） |
| **处置** | 人看 CMS/TG → **本地** `rd_loop` / variant-grid → promote；远程不 auto-fix |

---

## 2. 与现有实现的对照

| 能力 | 今天 | 目标 |
|------|------|------|
| 假设 + 分段因果 | ✅ `config/experiments` + variant-grid | 保持，作为唯一 R&D 入口 |
| 周 drift | ⚠️ `mlbot monitor` 已落地，**默认窗错误（C1）** | manifest + 近端窗 + P0.5 修复 |
| 月 replay 趋势 | ⚠️ `config/strategies/*/research/calibrate_roll.default.yaml` + pipeline | manifest + 固定 config replay（`event_backtest` 或薄 rolling） |
| 上线 pre_deploy | ⚠️ `strategies/*/research/pre_deploy_replay.yaml`（`extends` legacy） | `config/experiments/.../pre_deploy_*.yaml` 或 `config/monitoring/pre_deploy_*.yaml` |
| deploy 强制 contract | ❌ WORKFLOW 附录 A #7 | `mlbot monitor contract` + deploy hard-check |
| Grafana / RD 控制台 | 设计稿 | 消费 `results/monitoring/<job_id>/`（P3） |

---

## 3. 迁移项（按优先级）

### P0 — Monitor manifest（本地 + 远程同命令）

**动机**：与 `rd_loop` / `variant-grid` 同构 — YAML 描述 job，CLI 只执行。与 **`mlbot research` 命令族对称**的 **`mlbot monitor` 统一 verb** 见 [`漂移监控_mlbot_monitor_CN.md`](漂移监控_mlbot_monitor_CN.md) §5（`export-window` / `distribution` / `plateau` / … / `run`）；B/C/树 差异在 manifest `steps`，不是三套脚本。

**交付**：

1. 新建 `config/monitoring/README.md`（job 字段说明、env、`WATCHDOG_PARQUET`）。  
2. 示例 manifest，例如：  
   - `config/monitoring/weekly_rule_stack.yaml` — watchdog（7d bus）+ drift（6m archive-batch）+ heartbeat  
   - （计划）`monthly_fixed_replay_*.yaml`、`quarterly_regime_review.yaml` — 见 §漂移监控 §4.5  
   - （可选）`config/monitoring/monthly_replay_bpc.yaml` — 固定 `strategies_root` + segment / 日期窗  
3. `mlbot monitor run --config <path>` — 解析 manifest，顺序调用现有脚本。  
4. 更新 [`漂移监控_mlbot_monitor_CN.md`](漂移监控_mlbot_monitor_CN.md) §3、`etc/systemd` 注释指向 manifest。

**Manifest 示意**（实现时可微调字段名）：

```yaml
monitor_id: weekly_rule_stack
output_dir: results/monitoring/weekly_rule_stack/{run_ts}
# 周跑双窗（§漂移监控 §4.5 C6）：gate 短窗 + regime 长窗
windows:
  short:
    source: feature_bus_export
    lookback_days: 7
    timeframe: 120T
    parquet: results/monitoring/window/{run_ts}/features_current_7d.parquet
  long:
    source: archive_batch
    segment: recent_6m_oos   # config/market_segment.yaml
    parquet: results/monitoring/window/{run_ts}/features_current_6m.parquet
strategies: [bpc, tpc, me, srb]
steps:
  - export-window: { window: short }
  - archive-batch: { window: long }
  - watchdog: { window: short, baseline: config/monitoring/regime_watchdog_baseline.json }
  - drift: { window: long, emit_rd_loop_suggestions: true }
# schedule 仅文档/systemd 引用，CLI 不调度
```

**P0 约束**：

- manifest **必须**声明 `window.source`（`feature_bus_export` | `archive_batch` | `prepare_labeled` | `parquet_path`）。**生产 cron 默认 `feature_bus_export`**（§7.0）。
- **禁止**未设 `window` 时 fallback 到 `results/train_final/**/features_labeled.parquet`（当前 [`run_weekly.sh`](../../scripts/monitoring/run_weekly.sh) 行为视为待移除）。
- reference baseline（PSI/IC）可与 current 窗不同：reference=标定全历史，current=近端窗。

**P0 必做**：`export-window`（bus 拼窗）。**P2 可选**：按 segment 的 `archive_batch` / `prepare_labeled`（见 T1）。

---

### P0.5 — 监控正确性修复（C1–C4，优先于 manifest 美化）

**动机**：审查发现默认周跑几乎永不 ALERT（见 [`漂移监控_mlbot_monitor_CN.md`](漂移监控_mlbot_monitor_CN.md) §1.1）。manifest 落地前应先修「能告警」。

**交付**：

| 项 | 对应缺口 | 动作 |
|----|----------|------|
| 近端窗口 | C1 | 远程：**bus export-window** → current parquet；长窗/IC 可选 batch/prepare；`run_weekly.sh` 缺窗 **exit 3** |
| exit code | C2 | `regime_drift_monitor.py` ALERT 改为 **1** |
| 多策略 baseline | C3 | bpc / me / srb 的 watchdog + IC baseline |
| 告警通道 | C4 | **Telegram**（monitor ALERT）+ **CMS 漂移卡片**（读 report.json）；systemd `OnFailure` 可选 |
| CMS 展示 | C4 延伸 | mlbot_console `/rd` 或 monitor 专页：ALERT 列表 + 链本地验证命令 |

**验收**：用**近端窗** parquet 跑 `mlbot monitor weekly`，故意选与 baseline 不同分布的窗，至少一项 PSI / bull_share / plateau 可触发 ALERT；远程 cron 非零 exit 有通知。

---

### P1 — Pre-deploy 迁入 experiments / monitoring manifest

**动机**：`pre_deploy_replay.yaml` 是**门禁 job**，不是策略定义；且 `extends: validate_static.constrained` 绑死废弃链。

**交付**：

1. 定义 manifest schema（与 variant-grid 共用 `market_segment_path`、`strategies_root`）：  
   - `role: pre_deploy`  
   - `contract_checks`（locked_features、plateau_stability）  
   - `threshold_calibration.*.optimize: false`、`shap: false`  
   - `event_backtest` 长窗或指定 `segment`  
2. 在**已通过 promote 的实验目录**增加 `pre_deploy_confirm.yaml` 示例（如 TPC gate canonical 实验旁）。  
3. 实现路径二选一（实现时定）：  
   - **A**：`event_backtest` + `scripts/pre_deploy_contract_checks.py` 组合 CLI；  
   - **B**：薄 wrapper `mlbot monitor pre-deploy --config ...` 不调全量 pipeline。  
4. `config/strategies/*/research/pre_deploy_replay.yaml` 头注释标 **LEGACY**，指向新 manifest。

**验收**：新 promote 流程不再要求 `mlbot pipeline run -c .../pre_deploy_replay.yaml`。

---

### P2 — Calibrate_roll / 月 replay 迁出 strategies/research

**动机**：月任务 = **固定 config 验证趋势**，不是调参。

**交付**：

1. `config/monitoring/monthly_replay_<slug>.yaml` — 引用 `config/strategies` 生产 root、`optimize: false`、日期/rolling 段。  
2. 优先用 **固定窗 `event_backtest`** 或现有 `rolling_sim` 薄封装，避免全 bundle `auto_research_pipeline`。  
3. 废弃或仅只读保留 `config/strategies/*/research/calibrate_roll.default.yaml`。  
4. `scripts/monitoring/run_monthly.sh` 改为 `mlbot monitor run --config ...`（或等价）。

---

### P3 — 文档与交叉引用清理

**交付**：

1. [`config/experiments/README.md`](../../config/experiments/README.md) — 增加「pre_deploy / 监控 job 不放 strategies/research」。  
2. [`ABC统一研究框架_CN.md`](ABC统一研究框架_CN.md) / [`R&D工具矩阵_CN.md`](R&D工具矩阵_CN.md) — README 图 4 的 L2「research_roll 体检」改为「drift alert → experiments R&D」；注明 research_roll **可选/考古**。  
3. [`WORKFLOW_整体架构与管线改进计划_CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md) §4.5–4.6 — 上线门禁指向 manifest。  
4. 归档 [`研究模式与上线流程.md`](../experiments/z实验_005_统一研究/archive/研究模式与上线流程.md) — 已标 Drift-Only 过时，链到本计划 §1.3。

---

### P4 — 远期（非阻塞）

| 项 | 说明 |
|----|------|
| `mlbot monitor export-window` + 可选 `archive_batch` | bus 默认；长窗/IC 再 batch 或 prepare |
| `live/manifest.yaml` | pin experiment_id + commit |
| `deploy_config_to_live` hard-check | 7 天内 pre_deploy contract 无 BLOCKED |
| RD 控制台 / Grafana `quant_rd.json` | 索引 `results/monitoring/<job_id>/` |
| 实时 Regime 分类 / OOD | 不在 monitor v1（见 NP 问题归档） |

---

## 4. 目标工作流（端到端）

```
[本地 R&D]
  config/experiments/<topic>/
    rd_loop_*.yaml          → ① 假设
    *_grid.yaml             → ② 分段因果（segment_matrix + market_segment.yaml）
    DECISION.md             → 人审
    pre_deploy_confirm.yaml → ②b 长期稳健 / contract（P1 后）

  config_experiments/<topic>_strategies/  → 变体树

  mlbot research promote → config/strategies/<slug>/archetypes/

[远程 / 本地 运维]
  config/monitoring/weekly_rule_stack.yaml  → mlbot monitor run --config
  （月）config/monitoring/monthly_replay_*.yaml

  deploy_config_to_live.py → live/highcap/config/strategies/
```

---

## 5. 反模式（迁移期间禁止）

| 反模式 | 应该 |
|--------|------|
| 在 `config/strategies/*/research/` 新增研究模板 | 新 job → `config/experiments` 或 `config/monitoring` |
| 用 `research_roll` / `full_study` 做新 promote 决策 | variant-grid + LAYER_PROMOTION_CRITERIA |
| 用 pipeline 月跑 optimize | 月 manifest 仅 replay / alert |
| 监控改 yaml | ALERT → 人审 → 新 experiment → promote |
| Live 直接指向 `config_experiments` 未 promote 树 | promote → strategies → deploy |

---

## 6. 实施检查表（开工时用）

- [ ] P0：`config/monitoring/README.md` + 示例 yaml + `mlbot monitor run --config`（含 `window.source`）
- [ ] P0.5：监控正确性修复（T1–T4，见 §6.1）
- [ ] P0：更新 `漂移监控_mlbot_monitor_CN.md`、`run_weekly.sh` / systemd 文档
- [ ] P1：pre_deploy manifest schema + 一个策略 smoke（如 tpc）
- [ ] P1：legacy `pre_deploy_replay.yaml` 注释与迁移说明
- [ ] P2：monthly replay manifest；`run_monthly.sh` 切换
- [ ] P3：ABC / 工具矩阵 / WORKFLOW / experiments README 交叉引用
- [ ] P4：deploy hard-check、segment prepare（按需）

---

## 6.1 监控缺口 TODO（审查 2026-06-02）

与 [`漂移监控_mlbot_monitor_CN.md`](漂移监控_mlbot_monitor_CN.md) §1.1 一一对应；**T1–T2 为阻断项**。

| ID | 优先级 | 任务 | 对应缺口 | 状态 |
|----|--------|------|----------|------|
| **T1** | **高** | 远程 **`export-window`（7d）+ `archive-batch`（6m）**；周 manifest 分窗；移除 train_final fallback | C1,C6 | 待做 |
| **T10** | 中 | 周/月/季/年 manifest 族（§漂移监控 §4.5）；月任务改为 fixed replay | C6 | 待做 |
| **T11** | 中 | Promote **监控 bundle** checklist（§10.6）写入方法论；baseline 去绝对路径；deploy 可选校验 | §10.5 | 待做 |
| **T2** | **高** | 统一 exit code（drift 2→1） | C2 | **已修**（2026-06-02 P0.5） |
| **T12** | 中 | 与 **T8** 合并：`stat_kernels/drift.py` + 重构 watchdog/drift（**C7**） | C7 | 待做（里程碑 2） |
| **T3** | 中 | bpc / me / srb baseline | C3 | 待做 |
| **T4** | 中 | **Telegram** monitor ALERT + **CMS** 漂移卡片（远程主展示） | C4 | 待做 |
| **T5** | 低/远期 | realized-R vs expected-R（execution 层「该下线」证据） | C5 | 待做 |
| **T6** | 校验 | regime plateau 与 gate 规则一致性 | — | 例行 |
| **T7** | 中 | 远程 cron 日更 export-window+monitor（可选，默认周更） | — | 待做 |
| **T8** | 中 | 统一 monitor verb 族 + `stat_kernels/drift.py` 内核（§漂移监控 §5，含 **C7/T12**） | C7 | 待做（里程碑 2） |
| **T9** | 中 | manifest `window.source` 多分支 + `export-window`；§7.4 验收：远程 bus 导出、本地 labeled 复核 | §7.4 | 待做 |

**远程数据拼窗（T1）**：

- Archive 日级 bars + publisher 长窗 compute（~150d，运维口径 ~6 月 warmup）+ bus 最新 features
- 产出 `results/monitoring/window/<ts>/features_labeled.parquet` 供 `WATCHDOG_PARQUET`
- **本地**仅在 CMS/TG 告警后复跑验证

---

## 7. 变更记录

| 日期 | 说明 |
|------|------|
| 2026-06-02 | 初稿：汇总 experiments-centric 架构、废弃 research_roll、monitor/pre_deploy/calibrate_roll manifest 迁移优先级 |
| 2026-06-02 | Regime shift 审查：P0 `window.source` 约束、P0.5 正确性修复、§6.1 TODO T1–T6 |
| 2026-06-02 | 远程优先 + archive/warmup/bus 数据分层 + CMS/TG 目标流 + T7 日更 cron |
| 2026-06-02 | §漂移监控 §4 算法原理 + §5 统一 monitor 命令族（ABC+树+C）；T8 |
| 2026-06-02 | §7.4 本地 parquet / 远程分工；T9 |
| 2026-06-02 | §7.0 远程默认 bus export，非每周 prepare（IC/长窗例外） |
| 2026-06-02 | §4.5 节奏×层×算法；周 manifest 双窗；T10 |
| 2026-06-02 | §10 平台基线 / 本地→远程交付物；T11 promote bundle |
| 2026-06-02 | 代码审查 C7、§漂移监控 §1.2；T2 已修；T12↔T8；P0.5 周跑去 fallback + DRIFT_PARQUET |
