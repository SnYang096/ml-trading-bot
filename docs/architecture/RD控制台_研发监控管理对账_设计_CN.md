# RD 控制台设计：研发 / 监控 / 管理 / 对账 一体化

> **定位**：替代已落伍的 `scripts/rolling_dashboard/`（绑定旧 `auto_research_pipeline` 管线），围绕新 ABC 研发流程（`mlbot research` / `rd_loop` → `event_backtest --variant-grid` → `calibrate`/`promote` → `watchdog`/`drift`/`pre_deploy`）重建一个**研发控制平面 (RD Control Plane)**，并与已有的 `mlbot_console`（实盘 CMS/对账）与 `deploy/monitoring`（Grafana 栈）拼成「研发 + 监控 + 管理 + 对账」四件套。
>
> **状态**：设计稿（v0，待评审）。本文只做设计，不动代码。
>
> **关联**：
> - 流程口径：[`../strategy/方法论_R_and_D流程_CN.md`](../strategy/方法论_R_and_D流程_CN.md) · [`../strategy/R&D工具矩阵_CN.md`](../strategy/R&D工具矩阵_CN.md)
> - 遗留清理：[`../strategy/遗留研究命令清理计划_CN.md`](../strategy/遗留研究命令清理计划_CN.md)
> - 命令清单：[`../完整命令速查表.md`](../完整命令速查表.md)

---

## 1. 问题陈述

### 1.1 老 dashboard 为什么落伍

`mlbot rolling-dashboard`（`scripts/rolling_dashboard/`）是围绕**旧自动化管线**建的：

| 老 dashboard 的隐含假设 | 与新流程的冲突 |
|------------------------|----------------|
| 只有 `rolling`（`_rolling_sim`）与 `flat`（History）两种 run | 新流程产物是 `rd_loop/` + `experiments/` + 决策文档，不是 rolling_sim |
| 主指标来自 `stitched_summary.json` | 新流程主指标来自 `EXPERIMENT_INDEX.json` + `capital_report.json`（双段 R/DD） |
| 唯一 job 启动器 = `auto_research_pipeline.py` | 新流程入口是 `rd_loop.py` / `event_backtest --variant-grid` / `mlbot research` |
| 升级路径 = `mlbot pipeline adopt` | 新流程是 `mlbot research calibrate` → `promote`（locked merge + 人审） |
| 无实验谱系 | 假设→证据→变体→决策→上线→监控 这条 lineage 完全不可见 |

结果：**新流程跑出来的东西，控制台一个都看不到**——rd_loop 状态、plateau/skip 清单、变体双段对比、决策文档、watchdog/drift/contract 告警，全部散落在 `results/**` 和 `docs/decisions/**` 的 JSON/MD 里，靠人肉 `ls` 和读文件。

### 1.2 但基础设施并不空白

仓库里已经有**三块可复用**的资产，不该推倒重来：

| 资产 | 路径 | 职责 | 在四件套里的角色 |
|------|------|------|------------------|
| `rolling_dashboard` | `scripts/rolling_dashboard/` | 旧管线 run 卡片 + job 启动 + 目录浏览 | **部分退役**，保留静态文件服务与 job 框架 |
| `mlbot_console` | `src/mlbot_console/`（FastAPI） | 实盘 CMS：Trade Map / orders / account / regime / spot ledger / **对账** | **管理 + 对账**（已就绪，扩展即可） |
| `deploy/monitoring` | Prometheus + Grafana + Loki + Promtail + sqlite-web + Telegram 告警 | 实盘指标 / 日志 / 告警 | **监控**（已就绪，加 R&D 指标即可） |

**真正缺的只有一块：研发控制平面**——把新流程产物索引化、谱系化、可编排。

---

## 2. 目标与非目标

### 2.1 目标

1. **研发 (R&D)**：一个「实验注册表 + 看板」，索引 `rd_loop` / `EXPERIMENT_INDEX` / plateau / calibrate / 决策文档，展示 **假设 → 证据 → 双段验证 → 决策 → 上线** 的谱系；可从 UI 启动 `rd_loop` / `variant-grid` job。
2. **监控 (Monitoring)**：把 `regime_watchdog` / `regime_drift_monitor` / `pre_deploy_contract_checks` 的 JSON 变成 **Prometheus 指标 + Grafana 面板 + Telegram 告警**，与实盘指标并排。
3. **管理 (Management/CMS)**：策略各层状态（locked/active/disabled）、config/yaml 版本与 diff、**promote 审计日志**（谁、何时、改了哪条规则、skip 了什么）、上线门禁（contract 状态）。
4. **对账 (Reconciliation)**：复用 `mlbot_console` 已有的账户/订单/现货账本对账，**与研发侧打通**（live 表现回流到对应实验/决策）。

### 2.2 非目标

- 不做自动 promote（doctrine：人审才能抓 cross-regime bug）。控制台只**呈现证据 + 一键生成 draft + 显式确认**。
- 不替换 Grafana 自己造时序图（时序留给 Grafana/Prometheus）。
- 不做实时交易控制（kill switch 等仍在宪法 / 实盘侧）。
- v0 不引入重前端框架；沿用 `mlbot_console` 的 FastAPI + 原生 JS 风格。

---

## 3. 对标 qlib 的研发管线管理

> qlib 的研发管理是「**实验追踪 (experiment tracking)**」范式，我们要学它的**谱系与可复现**，但保留我们更强的**实盘对账 + 双段因果验证**。

| 维度 | qlib | 本仓库现状 | 设计取向 |
|------|------|------------|----------|
| 工作流编排 | `qrun workflow_config.yaml`（声明式） | `rd_loop` hypothesis yaml（已声明式，接近 qrun） | ✅ 保留 rd_loop，做成可 UI 触发 |
| 实验追踪 | `qlib.workflow.R` + MLflow（Experiment/Recorder，参数/指标/artifact 自动记录） | **无**；产物散落 results/ + docs/decisions | ❌ 缺口 → 建**实验注册表**（轻量 SQLite 索引，非 MLflow 重栈） |
| 记录器 | `SignalRecord` / `PortAnaRecord`（signal、回测、风险分析） | `EXPERIMENT_INDEX.json` + `capital_report.json`（双段 R/DD/Sharpe） | ✅ 已有等价物，缺统一索引 |
| 报告 | `qlib.contrib.report`（Plotly 离线报告） | `trading_map_*.html` + decision doc + Grafana | ✅ 更丰富，缺聚合入口 |
| 模型/因子库 | Model Zoo / Alpha360 | archetypes + features_*.yaml + 树通道 | ✅ 有，缺「版本 × 状态」视图 |
| 自动 R&D | RD-Agent（假设→代码→回测自动 loop） | **刻意不做全自动**（doctrine：人审 promote） | ⚠️ 只借「假设可追溯」，不借「自动改 yaml」 |
| 实盘 / 对账 | **无**（qlib 偏研究） | `mlbot_console` 对账 + Grafana 实盘指标 | ✅ 我们更强，要把它和研发谱系打通 |

**结论**：qlib 强在「实验追踪 + 可复现谱系」，弱在「实盘闭环」。我们正好相反。设计核心 = **补一个轻量实验注册表（学 MLflow Recorder 的谱系，但不引入 MLflow），把已有的实盘对账 + Grafana 接到谱系末端**。

---

## 4. 目标架构：四平面 + 一注册表

```text
                         ┌───────────────────────────────────────────────┐
                         │            RD Control Plane (新建)              │
                         │   FastAPI app: src/mlbot_console (扩展) 下新增   │
                         │   /rd/* 路由 + 静态页；复用 auth/config         │
                         └───────────────────┬───────────────────────────┘
                                             │ 读
                ┌────────────────────────────┼────────────────────────────┐
                ▼                            ▼                            ▼
   ① 研发 (R&D)                  ② 监控 (Monitoring)            ③ 管理 (Management/CMS)
   实验注册表 + 看板             Grafana + Prometheus + Loki     config/promote/contract
   - 假设/证据/变体/决策谱系     - R&D 指标 exporter (新建)       - 各层 locked/active 状态
   - rd_loop / EXPERIMENT_INDEX  - watchdog/drift/contract → TS   - promote 审计 + diff/skip
   - plateau / calibrate / skip  - 实盘指标(已有)                 - 上线门禁(contract gate)
                │                            │                            │
                └────────────┬───────────────┴────────────┬───────────────┘
                             ▼                            ▼
                  ┌─────────────────────┐      ④ 对账 (Reconciliation)
                  │  rd_registry.sqlite │      mlbot_console (已有)
                  │  (实验注册表索引)   │      - account / orders / spot ledger
                  │  由 indexers 周期构建│      - live PnL ↔ 实验/决策 回流
                  └──────────┬──────────┘
                             │ 扫描
   ┌─────────────────────────┼──────────────────────────────────────────────┐
   │  results/rd_loop/**/rd_loop_state.json                                   │
   │  results/**/experiments/EXPERIMENT_INDEX.json + <variant>/capital_report.json │
   │  results/**/quick_scan/**/{plateau.json, gate_plateau_batch.json, *.skips.json} │
   │  results/regime_watchdog/**/report.json                                  │
   │  results/regime_drift_monitor/**/drift_report.json                       │
   │  <run>/contract_checks.json                                              │
   │  docs/decisions/*.md  (YAML front-matter)                                │
   │  config/strategies/**/archetypes/*.yaml  (+ .bak.<ts> promote 备份)      │
   └──────────────────────────────────────────────────────────────────────────┘
```

**一句话**：新建一个**只读索引层 (`rd_registry.sqlite`) + 薄 API/UI**，挂在已有的 `mlbot_console` 进程里；监控复用 Grafana；对账复用 mlbot_console；老 rolling_dashboard 的 job 框架抽出来复用。

---

## 5. 核心新件：实验注册表 (Experiment Registry)

### 5.1 为什么要索引层而不是直接读文件

- `results/**` 被 gitignore，文件散、量大、schema 各异；UI 每次现扫太慢（老 dashboard 已用 15s 缓存绕）。
- 需要**跨产物 join**（一个决策文档对应哪几个变体？哪个实验导致了这次 promote？promote 后 watchdog baseline 是否更新？）。
- 学 qlib MLflow Recorder：用一个**轻量 SQLite**（对齐已有 `results/.pipeline_run_dashboard.sqlite` 的做法）做索引，**事实仍在文件里**，DB 只存指针 + 摘要 + 谱系边。

### 5.2 数据模型（SQLite 表）

```sql
-- 实验：一个 rd_loop topic 或一次 variant-grid 的逻辑单元
CREATE TABLE experiment (
  id            TEXT PRIMARY KEY,         -- topic 或 experiment_id
  strategy      TEXT,                     -- tpc / me / fast_scalp / chop_grid ...
  system        TEXT,                     -- A / B / C / tree
  layer         TEXT,                     -- regime/prefilter/gate/entry/direction/execution
  kind          TEXT,                     -- rd_loop | variant_grid | scan | monitor
  status        TEXT,                     -- running | completed | failed | abandoned
  created_at    TEXT,
  updated_at    TEXT,
  output_dir    TEXT,                     -- results/rd_loop/<topic> 等
  hypothesis    TEXT                      -- 一句话假设（来自 yaml/decision front-matter）
);

-- 运行/步骤：rd_loop step、variant run、scan、monitor run
CREATE TABLE run (
  id            TEXT PRIMARY KEY,
  experiment_id TEXT REFERENCES experiment(id),
  step          TEXT,                     -- research_scan|variant_grid|decision_doc|<variant>
  period        TEXT,                     -- recent | bull | <date-range>
  exit_code     INTEGER,
  metrics_json  TEXT,                     -- {total_r,max_dd,sharpe,trades,lift,plateau...}
  artifact_path TEXT,                     -- capital_report.json / plateau.json ...
  finished_at   TEXT
);

-- 证据：scan/plateau/ic/skip 摘要（用于看板「假设是否成立」）
CREATE TABLE evidence (
  id            TEXT PRIMARY KEY,
  experiment_id TEXT REFERENCES experiment(id),
  evid_type     TEXT,                     -- condition_set|feature_plateau|lift|ic|skip
  feature       TEXT,
  summary_json  TEXT,                     -- {z, dpp, lift, plateau_start/end, skip_reason...}
  source_path   TEXT
);

-- 决策：docs/decisions/*.md front-matter
CREATE TABLE decision (
  id            TEXT PRIMARY KEY,         -- decision doc slug
  experiment_id TEXT,
  strategy      TEXT,
  verdict       TEXT,                     -- promote | reject | park | needs-more
  decided_by    TEXT,
  decided_at    TEXT,
  doc_path      TEXT
);

-- promote 审计：mlbot research promote 的每次写回
CREATE TABLE promotion (
  id            TEXT PRIMARY KEY,
  decision_id   TEXT REFERENCES decision(id),
  strategy      TEXT,
  layer         TEXT,
  target_yaml   TEXT,                     -- config/strategies/<s>/archetypes/gate.yaml
  backup_path   TEXT,                     -- .bak.<ts>
  skips_json    TEXT,                     -- calibrate skip 清单
  diff_summary  TEXT,
  promoted_at   TEXT
);

-- 监控告警：watchdog/drift/contract 的时间序列汇总（细粒度走 Prometheus）
CREATE TABLE monitor_event (
  id            TEXT PRIMARY KEY,
  source        TEXT,                     -- watchdog|drift|contract
  strategy      TEXT,
  status        TEXT,                     -- OK|ALERT|BLOCKED
  detail_json   TEXT,
  report_path   TEXT,
  ts            TEXT
);
```

### 5.3 谱系 (lineage)

把上面表连成一条链，这是控制台最大的价值：

```text
hypothesis (experiment.hypothesis)
   └─► evidence (scan/plateau/lift/ic)      ← ① 假设是否成立
         └─► run[period=recent], run[period=bull]   ← ② 双段因果
               └─► decision (verdict)        ← 人审
                     └─► promotion (diff/skip/backup)  ← 写回生产
                           └─► monitor_event (watchdog/drift/contract)  ← ③ 上线后
                                 └─► live reconciliation (mlbot_console)  ← 真金白银回流
```

UI 用一条横向 timeline 呈现；任意节点可下钻到原始文件（plateau.json / capital_report.html / decision.md / gate.yaml diff）。

### 5.4 indexers（构建注册表）

新增 `scripts/rd_registry/` 包，一组幂等扫描器（可 cron / 可 UI 手动刷新 / 可 watch 文件变更）：

| indexer | 扫描源 | 写表 |
|---------|--------|------|
| `index_rd_loop.py` | `results/rd_loop/**/rd_loop_state.json` + quick_scan | experiment, run, evidence |
| `index_experiments.py` | `results/**/experiments/EXPERIMENT_INDEX.json` + `<variant>/capital_report.json` | experiment, run |
| `index_decisions.py` | `docs/decisions/*.md` front-matter | decision |
| `index_promotions.py` | `*.skips.json` + `*.bak.<ts>` + git log of archetypes | promotion |
| `index_monitor.py` | `regime_watchdog/**`, `regime_drift_monitor/**`, `contract_checks.json` | monitor_event |

> 复用老 dashboard 的 `scan.py` / `response_cache.py` 思路（rglob + TTL 缓存），只是换 schema。

---

## 6. 决策文档 front-matter 约定（让人审产物可索引）

当前 `docs/decisions/*.md` 是纯 markdown，机器无法 join。建议 `_new_decision_doc.py` 在头部生成 **YAML front-matter**（不破坏可读性，且 doctrine「人审」不变）：

```markdown
---
id: tpc_gate_vol_ABH_20260527
experiment_id: tpc_gate_plateau
strategy: tpc
system: B
layer: gate
verdict: park            # promote | reject | park | needs-more
decided_by: yin
decided_at: 2026-05-27
evidence:
  - results/rd_loop/tpc_gate_plateau/quick_scan/gate_plateau/gate_plateau_batch.json
variants:
  - experiment_index: results/tpc/experiments/EXPERIMENT_INDEX.json
    promoted_variant: H
periods: [recent, bull]
promote:
  target: config/strategies/tpc/archetypes/gate.yaml
  applied: false
---

# TPC Gate vol ABH 实验
（正文照旧：变体表 + 双段结果 + 决策理由 + 复现命令 + 已知坑）
```

`index_decisions.py` 读 front-matter → `decision` 表 → 谱系即可贯通。**这是整套设计里唯一需要轻改产物格式的地方**，收益最大。

---

## 7. 监控平面：把 R&D JSON 接进 Grafana

不重复造时序图，新增一个 **exporter** 把研发/监控 JSON 翻译成 Prometheus 指标（或 Loki 日志）：

```text
scripts/rd_registry/exporter.py
  读 monitor_event 表 (或直接读 report.json)
  → 暴露 /metrics (prometheus_client) 或写 textfile collector
```

建议指标（labels: strategy, layer, feature, period）：

| 指标 | 来源 | 用途 |
|------|------|------|
| `rd_watchdog_alert{strategy}` | regime_watchdog report.json | IC/PSI 告警 → Grafana + Telegram |
| `rd_drift_status{strategy,feature}` | regime_drift_monitor | plateau 漂出 0/1 |
| `rd_contract_status{strategy,check}` | contract_checks.json | BLOCKED → 阻断上线面板红 |
| `rd_experiment_total{status}` | registry | 在跑/完成/失败实验数 |
| `rd_decision_pending` | registry | 待人审决策数 |
| `rd_promote_lag_days{strategy,layer}` | registry | 距上次 promote 天数（防过拟合月月动） |

Grafana 侧加一个 `quant_rd.json` 看板（与已有 `quant_system.json` / `quant_strategy_map.json` 并列），复用现成 Telegram contact-point。**实盘指标已在 Grafana**，研发指标补进去即「监控」闭环。

---

## 8. 管理平面 (CMS)：config × 状态 × promote 审计

挂在 `mlbot_console` 下新增 `/rd/manage`：

1. **层状态矩阵**：读 archetypes/*.yaml + features_*.yaml，渲染 ABC × 层 的 `locked / active / disabled / draft` 状态（对应 [`方法论_R_and_D流程_CN.md`](../strategy/方法论_R_and_D流程_CN.md) §3 的矩阵）。
2. **config 版本/diff**：archetypes 在 git 里，直接 `git log -p` 该 yaml；promote 备份 `.bak.<ts>` 做并排 diff。
3. **promote 审计**：`promotion` 表渲染「谁/何时/改哪条规则/skip 了什么/对应哪个决策」。从 UI 触发 `mlbot research promote --dry-run` 出 diff，人确认后才 `--yes`（沿用 localhost-only POST 安全模型）。
4. **上线门禁**：`contract_checks.json` 状态卡；BLOCKED 时禁用 deploy 按钮。

---

## 9. 对账平面：复用 mlbot_console，并回流到谱系

`mlbot_console` 已有 account / orders / spot ledger / 多腿对账。新增的打通点：

- live 成交（order_management.db）按 strategy + 时间窗聚合 → 关联到最近一次该层 `promotion` → 在实验谱系末端显示「上线后真实 R / PnL vs 决策文档里的回测预期」。
- 形成闭环：**回测预期 R ↔ 实盘实际 R 的偏差**，作为下一轮 R&D 的触发器（写回 `monitor_event`，drift 同级别）。

---

## 10. 路由与组件落点

统一进 `src/mlbot_console`（已有 FastAPI + auth + 静态页），避免再起一个进程：

| 路由 | 平面 | 数据源 |
|------|------|--------|
| `/rd` | 研发首页（看板：在跑实验 / 待审决策 / 告警） | registry |
| `/rd/experiments`、`/rd/experiment/{id}` | 实验列表 + 谱系下钻 | registry + 原始文件 |
| `/rd/evidence/{id}` | plateau/scan/skip 详情 | quick_scan/*.json |
| `/rd/decisions` | 决策列表（verdict 过滤） | decision 表 |
| `/rd/manage` | 层状态 + config diff + promote 审计 | yaml + git + promotion 表 |
| `/rd/run`（POST，localhost） | 启动 rd_loop / variant-grid job | 复用 `pipeline_jobs.py` 框架 |
| `/api/rd/*.json` | 上述数据的 JSON API | registry |
| `/metrics` | Prometheus exporter | registry/report.json |

job 启动框架：把老 `scripts/rolling_dashboard/pipeline_jobs.py`（SQLite job 表 + 日志 tail + 进度）**抽成通用 runner**，命令从 `auto_research_pipeline.py` 换成 `rd_loop.py` / `python -m scripts.event_backtest --variant-grid` / `mlbot research *`。

---

## 11. 分阶段路线图

| Phase | 内容 | 产出 | 依赖 |
|-------|------|------|------|
| **P0** | 决策文档 front-matter 约定 + `_new_decision_doc.py` 生成 | 可索引决策 | 无（先行，收益大） |
| **P1** | `rd_registry` 包 + indexers + `rd_registry.sqlite` | CLI: `mlbot rd index` 出注册表 | P0 |
| **P2** | `mlbot_console` 加 `/rd` 研发看板（只读：实验/证据/谱系/决策） | 研发平面 MVP | P1 |
| **P3** | exporter + Grafana `quant_rd.json` 看板 + Telegram | 监控平面（R&D 告警进 Grafana） | P1 |
| **P4** | `/rd/manage`：层状态 + config diff + promote 审计 + dry-run promote | 管理平面 | P1 |
| **P5** | job runner 复用：UI 触发 rd_loop / variant-grid | 编排平面 | P2 |
| **P6** | 对账回流：live R ↔ 回测 R 偏差 → monitor_event | 对账闭环 | P2 + mlbot_console |
| **P7** | 老 `rolling_dashboard` 退役（仅留静态文件服务 / 迁移 job 框架） | 单一控制台 | P5 |

> 建议先做 **P0 + P1 + P2**：最小可用 = 「决策可索引 + 注册表 + 只读研发看板」，立刻解决「新流程产物看不见」的核心痛点；监控/管理/对账可增量叠加。

---

## 12. 从 rolling_dashboard 迁移

| 老能力 | 去向 |
|--------|------|
| rolling_sim 卡片 | 退役（新流程无 rolling_sim）；历史 run 走 `/browse` 静态 |
| pipeline job 启动 | 抽 `pipeline_jobs.py` 为通用 runner，命令换新流程 |
| adopt 按钮 | 替换为 `/rd/manage` 的 `research promote` dry-run → 确认 |
| 目录浏览 `/browse` | 保留（仍有用） |
| 静态文件服务（trading_map_*.html 等） | 保留（registry 下钻会用到） |
| `.pipeline_run_dashboard.sqlite` | 与 `rd_registry.sqlite` 合并或并存 |

---

## 13. 开放问题（评审待定）

1. **进程拓扑**：研发控制台挂进 `mlbot_console`（同进程，简单）还是独立服务（研究机 vs 实盘机隔离）？实盘机通常不跑研究，建议研发控制台可单独部署、对账平面只读连 live DB。
2. **registry 刷新策略**：cron（简单）vs 文件 watch（实时）vs UI 手动？建议 cron(5min) + 手动刷新按钮。
3. **front-matter 回填**：历史 `docs/decisions/*.md` 要不要补 front-matter？建议只对新文档强制，老文档按需。
4. **多机 results**：研究在多台机器跑，registry 要不要支持远程/汇总？v0 先单机，路径用相对 + 机器标签预留。
5. **权限**：promote / job 触发要不要比 localhost-only 更强的鉴权（复用 `mlbot_console` BasicAuth）？

---

## 14. 一页纸总结

- **不重写**：监控用 Grafana、对账用 mlbot_console、job 框架用 rolling_dashboard 抽出来的 runner。
- **新建一块**：轻量**实验注册表**（SQLite 索引 + indexers）+ `mlbot_console` 下的 `/rd` 看板。
- **学 qlib**：实验谱系可追溯、可复现（但不引入 MLflow，也不做自动 promote）。
- **杠杆点**：决策文档加 YAML front-matter（P0），整条「假设→证据→双段→决策→promote→监控→对账」谱系就能贯通。
- **先做 P0–P2**，最小可用即解决「新流程产物看不见」。
