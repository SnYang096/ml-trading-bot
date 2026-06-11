# 漂移监控：B / C 系统统一设计

> **状态**：设计稿（2026-06-11）  
> **读者**：实盘运维、策略 promote、CMS 使用方  
> **上位文档**：[漂移监控_mlbot_monitor_CN.md](漂移监控_mlbot_monitor_CN.md)（命令与远程分工）· [配置与监控_manifest迁移计划_CN.md](配置与监控_manifest迁移计划_CN.md)（manifest 迁移）  
> **代码锚点**：`config/monitoring/*.yaml` · `src/monitoring/` · `scripts/regime_*.py` · `scripts/multileg_monitor.py` · CMS `/monitoring`

---

## 1. 要解决的问题

| 现状 | 目标 |
|------|------|
| 日/周 cron 只跑 **B 四策略**（bpc/tpc/me/srb） | **B + C** 均有明确 cadence 与 CMS 卡片 |
| TPC regime 已换 **labeled `allowed_regimes`**（ADX+EMA），监控仍部分按旧 plateau/硬编码 EMA | **读 regime.yaml 本身**，schema 变则检测逻辑跟着变 |
| C（chop_grid/trend_scalp）只有 **`multileg monitor` 月报**，未接 feature bus 链 | C **Regime/Prefilter** 走 bus verb；**执行层** 保留 multileg 月报 |
| CMS 只见 `_factor_health`，看不出 regime 语义 | 卡片 + 表展示 **PSI / regime mix / multileg KPI** 分项 |

**原则（与迁移计划一致）**：一套 `mlbot monitor schedule`，**B/C 差异写在 manifest `steps`**，不维护第二套哲学。

---

## 2. 系统总览

```mermaid
flowchart TB
  subgraph prod [生产配置 — git + live 镜像]
    BS["config/strategies<br/>bpc/tpc/me/srb"]
    CS["config/strategies<br/>chop_grid/trend_scalp"]
    BL["config/monitoring<br/>baseline + schedules"]
    LIVE["live/highcap/config/strategies"]
  end

  subgraph data [远程数据 — VPS]
    BUS["shared_feature_bus<br/>features/120T/*.parquet"]
    RES["results/monitoring/<br/>index.json + reports"]
    ROLL["results/*/rolling_sim<br/>monthly_ledger"]
  end

  subgraph cron [systemd timers]
    D["daily_health"]
    Wb["weekly_b_rule_stack"]
    Wc["weekly_c_regime"]
    Mb["monthly_b_drift"]
    Mc["monthly_c_multileg"]
  end

  subgraph verbs [mlbot monitor steps]
    EW[export-window]
    WD[watchdog]
    DR[drift]
    ML[multileg-kpi]
  end

  subgraph cms [CMS mlbot_console]
    API["/api/monitoring/dashboard"]
    UI["漂移监控页<br/>日→周→月→季→年"]
  end

  BS --> LIVE
  CS --> LIVE
  BL --> cron
  BUS --> EW
  cron --> verbs
  LIVE --> WD
  LIVE --> DR
  verbs --> RES
  ROLL --> ML
  RES --> API --> UI
```

**主路径**：远程 timer → manifest → bus（+ rolling 产物）→ report → `index.json` → CMS。  
**处置路径**：人看 CMS/TG → 本地 `rd_loop` / variant-grid → promote → 更新 baseline → git push → 远程 pull。

---

## 3. B 系与 C 系：同一模型、不同 manifest

```mermaid
flowchart LR
  subgraph layers [策略层 — ABC 共用]
    R[Regime 慢变量]
    P[Prefilter 结构]
    G[Gate / Entry — 主要 B]
    E[Execution 执行]
  end

  subgraph B [B 系 bpc/tpc/me/srb]
    B1[周: PSI + regime mix]
    B2[周: plateau / labeled drift]
    B3[日: bus 心跳 + 缺勤]
  end

  subgraph C [C 系 chop_grid/trend_scalp]
    C1[周: entry_feature 分布<br/>extensions.multileg]
    C2[周: prefilter 特征 PSI]
    C3[月: multileg KPI 环比]
  end

  R --> B1
  R --> B2
  R --> C1
  P --> C2
  G --> B1
  E --> C3
```

| 维度 | B 系 | C 系 |
|------|------|------|
| **策略 slug** | `bpc`, `tpc`, `me`, `srb` | `chop_grid`, `trend_scalp` |
| **regime.yaml 形态** | 旧：`rules` + plateaus；新 TPC：`allowed_regimes{rules}` | `extensions.multileg`（`entry_feature`, `entry_min`, …） |
| **特征来源** | highcap feature bus（与 trend 同源） | 同 bus（需 publisher 输出 `bpc_semantic_chop` / `trend_confidence`） |
| **周监控** | watchdog(near 7d) + drift(deep bus) | watchdog-C（entry 通过率 + PSI）+ drift-C（prefilter 列，可选） |
| **月监控** | `monthly_drift`（30d plateau） | **`multileg monitor`**（rolling 月 KPI） |
| **执行层** | 未来 `ledger` / realized-R（T5） | **已有** multileg 月报 |

---

## 4. Regime 检测：三种 schema 统一适配

监控不再假设「全是 `last_calibration.plateaus`」。`src/monitoring/regime_health.py` 按 **regime.yaml 实际形态** 分支（已实现 / 计划）：

```mermaid
flowchart TD
  START[读取 strategies/.../regime.yaml]
  P{有 last_calibration.plateaus?}
  L{allowed_regimes 为 dict<br/>且含 per-label rules?}
  M{有 extensions.multileg?}

  START --> P
  P -->|是| PLAT[plateau P50 vs 区间<br/>regime_drift_monitor legacy]
  P -->|否| L
  L -->|是| LAB[classify 每 bar → bull/bear/neutral<br/>比 regime_shares vs baseline]
  L -->|否| M
  M -->|是| CEXT[entry_feature 分布<br/>pass_rate vs entry_min baseline]
  M -->|否| SKIP[NO_PLATEAUS / 跳过并 CMS 标注]

  PLAT --> OUT[写入 drift_report + monitor_event]
  LAB --> OUT
  CEXT --> OUT
  SKIP --> OUT
```

| Schema | 示例策略 | 检测什么 | baseline 写哪 |
|--------|----------|----------|---------------|
| **plateau** | bpc（旧） | 特征 P50 是否在 plateau 带内 | `regime.yaml` `last_calibration.plateaus` |
| **labeled** | tpc（E22 ADX+EMA） | 分类后 bull/bear/neutral **占比** | `regime_watchdog_baseline.json` → `regime_shares`，或 `last_calibration.regime_shares` |
| **multileg** | chop_grid, trend_scalp | `entry_feature >= entry_min` **通过率**；`exit_below` 滞后不在此检（引擎职责） | `last_calibration.multileg_baseline` 或 monitoring JSON |

**regime 改了能否检测到？**

- **B labeled（TPC）**：改 `allowed_regimes` 规则 → 下周 classify 结果变 → **regime_shares 漂移**（需 baseline）。✅ 已接线（`9eb4a88e`）。
- **B plateau（bpc/me/srb）**：改 plateau 或 rules → plateau drift / PSI。✅ 原有逻辑。
- **C multileg**：改 `entry_min` / `entry_feature` → **pass_rate 变**；🔲 本文 Phase 2 实现 `evaluate_multileg_regime_health`。

---

## 5. Cadence × Manifest 矩阵（目标态）

卡片顺序（CMS）：**日 → 周 → 月 → 季 → 年**。

```mermaid
gantt
  title 监控节奏（示意）
  dateFormat YYYY-MM-DD
  section 日
  daily_health B心跳+缺勤     :a1, 2026-06-01, 1d
  section 周
  weekly_b B watchdog+drift  :a2, 2026-06-07, 7d
  weekly_c C regime+prefilter:a3, 2026-06-07, 7d
  section 月
  monthly_b 30d drift        :a4, 2026-07-01, 30d
  monthly_c multileg KPI       :a5, 2026-07-01, 30d
  section 季/年
  quarterly/yearly 归档复核   :a6, 2026-10-01, 90d
```

| Cadence | manifest（计划路径） | strategies | 窗 | steps |
|---------|---------------------|------------|-----|-------|
| **daily** | `daily_health.yaml`（已有） | bpc,tpc,me,srb | near 1d | export-window → watchdog |
| **weekly** | `weekly_rule_stack.yaml`（已有） | bpc,tpc,me,srb | near 7d + deep 0d | export×2 → watchdog → drift |
| **weekly_c** | `weekly_c_regime.yaml`（**新增**） | chop_grid,trend_scalp | near 7d | export-window → watchdog-c → drift-c（可选） |
| **monthly** | `monthly_drift.yaml`（已有） | bpc,tpc,me,srb | 30d | export → drift |
| **monthly_c** | `monthly_multileg_c.yaml`（**新增**） | chop_grid,trend_scalp | rolling 月 | multileg-kpi |
| **quarterly/yearly** | 已有 | B 全系 | 长窗 | watchdog + drift 归档 |

**staleness_hours**（`schedules.yaml`）：weekly_c / monthly_c 与对应 B cadence 同级上限（周 192h、月 840h）。

---

## 6. 单次周更数据流（B + C 并列）

```mermaid
sequenceDiagram
  participant T as systemd timer
  participant S as monitor_scheduler
  participant E as export-window
  participant B as feature_bus
  participant W as watchdog
  participant D as drift
  participant I as index.json
  participant C as CMS

  T->>S: schedule --cadence weekly
  S->>E: near lookback_days=7
  E->>B: 读 120T parquet
  E-->>S: features_current_7d.parquet
  S->>W: strategies=bpc,tpc,me,srb
  Note over W: PSI vs Tier-0 ref<br/>regime_shares vs baseline
  W-->>S: report.json
  S->>E: deep lookback_days=0
  E-->>S: features_current_deep.parquet
  S->>D: drift per strategy regime.yaml
  D-->>S: drift_report.json
  S->>I: index_monitor_run
  C->>I: GET /api/monitoring/dashboard

  Note over T,C: weekly_c 另跑一条 manifest<br/>同 bus，不同 strategies + watchdog-c
```

---

## 7. C 系月更：multileg 执行层

```mermaid
flowchart LR
  RS[rolling_sim 历史 run]
  ML[monthly_ledger.jsonl]
  MM[multileg_monitor.py]
  REP[results/multileg_monitor/]
  IDX[index.json cadence=monthly_c]

  RS --> ML --> MM --> REP --> IDX
```

| 信号 | 含义 |
|------|------|
| `trend_regime_shift` | trend 腿 flip 频率月环比 |
| `chop_regime_shift` | chop entry 语义漂移 |
| `trade_shift` / `forced_shift` | 成交量、forced 率 |
| `threshold_shift` | 总 R 低于地板 |

与 B 的 **特征 parquet 链正交**：CMS **同页**展示，cadence 分卡（`月更·C`）。

---

## 8. Baseline 与 promote 后维护

```mermaid
flowchart LR
  T0[Tier-0 / promote 标定窗]
  W1[写 regime.yaml<br/>last_calibration.*]
  W2[写 config/monitoring<br/>regime_watchdog_baseline.json]
  W3[写 PSI ref parquet]
  GIT[git push]
  VPS[远程 pull + live deploy]

  T0 --> W1
  T0 --> W2
  T0 --> W3
  W1 --> GIT
  W2 --> GIT
  W3 --> GIT
  GIT --> VPS
```

**B 系 minimum（按策略）**

```yaml
# regime_watchdog_baseline.json 片段
"tpc": {
  "regime_shares": { "bull": 0.05, "bear": 0.45, "neutral": 0.50 },
  "source": "results/monitoring/tier0/..."
}
```

**C 系 minimum（计划）**

```yaml
# regime.yaml 或 monitoring JSON
last_calibration:
  multileg_baseline:
    chop_grid:
      entry_pass_rate: 0.38    # P(entry_feature >= entry_min)
      median_entry_feature: 0.55
    trend_scalp:
      entry_pass_rate: 0.22
```

Promote **E22 TPC ADX regime** 后：在标定窗上跑 `mlbot monitor catalog` 选 parquet → 算 `regime_shares` → 写入 baseline（**不要**沿用仅 `bull_share` 的旧 Tier-0）。

---

## 9. CMS 展示模型

| 卡片字段 | B 周更 | C 周更 | C 月更 |
|----------|--------|--------|--------|
| watchdog | PSI + regime_shares ALERT | entry_pass_rate ALERT | — |
| drift | plateau / labeled OK | prefilter PSI（可选） | multileg KPI WATCH |
| 详情表 | `因子健康 (PSI/IC)` | `chop_grid` / `trend_scalp` | `trend_regime_shift` 等 |

索引：`results/monitoring/index.json` + `rd_registry.sqlite` `monitor_event`（含 `detail_json`）。

---

## 10. 实施阶段

| 阶段 | 内容 | 状态 |
|------|------|------|
| **P0** | bus export + daily/weekly B + CMS 卡片 + Telegram | ✅ 已上线 |
| **P0.5** | labeled regime_shares（TPC ADX） | ✅ `regime_health.py` |
| **P1 B** | 补全 bpc/me/srb `regime_shares` baseline；live 同步 regime.yaml | 🔲 运维 |
| **P1 C manifest** | 新增 `weekly_c_regime.yaml`、`monthly_multileg_c.yaml`、`schedules.yaml` 条目 | ✅ |
| **P2 C verb** | `watchdog-c`：`extensions.multileg` pass_rate；`multileg-kpi` step 纳入 scheduler | ✅ |
| **P2 统一** | `strategies_source: constitution` + `strategy_support.yaml`（B 仅 `tpc` drift-ready） | ✅ |
| **P2 部署** | VPS timer 启用 `weekly_c` / `monthly_c`；补 C `multileg_baseline` | 🔲 运维 |
| **P3** | PSI 列从 gate/regime 自动推导；ledger realized-R（B 执行层） | 远期 |

---

## 11. 计划新增 manifest 示意

### `config/monitoring/weekly_c_regime.yaml`

```yaml
monitor_id: weekly_c_regime
output_dir: results/monitoring/weekly_c_regime/{run_ts}
strategies: [chop_grid, trend_scalp]
strategies_root: live/highcap/config/strategies   # 远程：与实盘一致

windows:
  near:
    source: feature_bus_export
    lookback_days: 7
    timeframe: 120T
    parquet: results/monitoring/window/{run_ts}/features_c_7d.parquet

watchdog_defaults:
  regime_share_tol: 0.10

steps:
  - export-window: { window: near }
  - watchdog-c:      # Phase 2：multileg entry pass_rate + PSI on entry_feature
      window: near
  - drift-c:         # 可选：prefilter 列 plateau
      window: near
      layer: prefilter
```

### `config/monitoring/monthly_multileg_c.yaml`

```yaml
monitor_id: monthly_multileg_c
output_dir: results/monitoring/monthly_multileg_c/{run_ts}
strategies: [chop_grid, trend_scalp]

steps:
  - multileg-kpi:
      rolling_root: results/trend_scalp/validate_static.full_study/_rolling_sim
      strategies: [chop_grid, trend_scalp]
```

### `schedules.yaml` 增补

```yaml
  weekly_c:
    manifest: config/monitoring/weekly_c_regime.yaml
    description: C regime/prefilter on 7d bus
  monthly_c:
    manifest: config/monitoring/monthly_multileg_c.yaml
    description: C execution layer multileg month-over-month
```

---

## 12. 与现有文档分工

| 文档 | 职责 |
|------|------|
| [漂移监控_mlbot_monitor_CN.md](漂移监控_mlbot_monitor_CN.md) | CLI、远程 cron、数据分层 §7、缺口表 |
| [配置与监控_manifest迁移计划_CN.md](配置与监控_manifest迁移计划_CN.md) | manifest 迁移 P0–P3、废弃 research_roll |
| **本文** | **B+C 统一拓扑、regime schema 适配、cadence 矩阵、CMS 模型、实施路线图** |
| [config/monitoring/README.md](../../config/monitoring/README.md) | 操作手册：catalog、Tier-0、systemd |

---

## 13. 验收清单（B+C 全接线后）

- [ ] `schedules.yaml` 含 daily / weekly / weekly_c / monthly / monthly_c / quarterly / yearly
- [ ] VPS timer 全 enable；`index.json` 无 MISSED（除故意停用的 cadence）
- [ ] TPC 改 `allowed_regimes` 后，周更 drift/watchdog 显示 **regime_shares** 而非仅 EMA 硬阈值
- [ ] chop_grid 改 `entry_min` 后，周更 C 卡 **entry_pass_rate** 可 ALERT（需 baseline）
- [ ] 月更 C 卡显示 multileg **WATCH/RETUNE** 与详情表
- [ ] live `regime.yaml` 与 git promote 同步；baseline 与标定窗同源

---

*文档版本：2026-06-11 · 对应代码：`regime_health` labeled 分支已落地；C verb 为 Phase 2。*
