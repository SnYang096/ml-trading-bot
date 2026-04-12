# 实盘与滚动窗归因工作流

本文描述：**当线上或滚动评估窗口相对基线变差时**，如何用仓库里**现成**的诊断入口定位问题层，并落到可执行的下一步（改 Gate、改 PCM、补数据、重训等）。

> **范围**：以 **Parquet 形态的执行 / 回测 logs** 为主（与 `mlbot gate apply-archetype`、`mlbot diagnose e2e-kpi`、管线产物列名一致）。不等价于「交易所账户级对账」，后者需另接资金曲线与成交回报。

---

## 1. 数据契约（两份 Parquet 要对齐）

综合脚本 `scripts/diagnose_production_attribution.py` 会：

- 用 `ret_mean` / `ret_trend` 与 `gate_archetype`（或 `archetype`）合成 **archetype 对齐收益**；
- **仅统计 `gate_ok == True` 的行**（若无该列则 gated 子集为空，指标会失真）。

建议在进入归因前确认（或先跑 gate）：

| 列名（常见） | 用途 |
|--------------|------|
| `gate_ok` | 是否通过 Gate；无则须先补或勿用 gated 统计 |
| `ret_mean`, `ret_trend` | 与 archetype 类型选择收益口径 |
| `gate_archetype` / `archetype` | TC/TE vs FR/ET 收益列切换 |
| `timestamp` / 索引时间 | 切窗、对齐 |
| `slot_id` / `position_id`（可选） | PCM 诊断里 slot 统计 |

**基线（baseline）** 与 **当前（production）** 应是**同一 schema**、可比时间粒度（例如同为 4H bar 或同为逐笔聚合后的日志行）。

---

## 2. 触发与阈值（显式配置）

`diagnose_production_attribution.py` 计算的 **`trade_count_drop = (prod_trades - base_trades) / base_trades`**，相对缩量时为**负数**。告警条件为 **`trade_count_drop <= 阈值`**，因此「相对基线少 20% 就告警」应设 **`"trade_count_drop": -0.2`**。

> **注意**：`mlbot diagnose production-attribution` 在 `src/cli/main.py` 里自带的 `--alert-thresholds` 默认 JSON 字符串含 `"trade_count_drop": 0.2`（正数），与脚本内 `get(..., -0.2)` 的语义**易混淆**。**生产使用请在命令行显式传入 JSON**（见下节示例）。

其它键（与脚本一致）：

- `consecutive_losses`：连续亏损条数阈值（在 gated 收益序列上数负收益「连续段」的最大长度）。
- `sharpe_drop`：生产 Sharpe − 基线 Sharpe，**小于等于**该值告警（常为负数，如 `-0.5`）。

---

## 3. 推荐路径（先总览，再分层）

### 3.1 一键退化检测（已有）

```bash
mlbot diagnose production-attribution \
  --production-logs results/live_or_rollout/logs.parquet \
  --baseline-logs results/baseline_smoke_test/logs_baseline.parquet \
  --output-dir results/diagnostics/production_attribution \
  --alert-thresholds '{"consecutive_losses":5,"sharpe_drop":-0.5,"trade_count_drop":-0.2}'
```

或直调脚本：

```bash
python scripts/diagnose_production_attribution.py \
  --production-logs results/live_or_rollout/logs.parquet \
  --baseline-logs results/baseline_smoke_test/logs_baseline.parquet \
  --output-dir results/diagnostics/production_attribution \
  --alert-thresholds '{"consecutive_losses":5,"sharpe_drop":-0.5,"trade_count_drop":-0.2}'
```

**产出**：`degradation_report.json`、`degradation_report.md`。若检测到退化，脚本以 **exit code 1** 退出（便于 CI / cron）。

### 3.2 分层深挖（按现象选工具）

下列均为仓库内**存在**的入口；参数以 `mlbot diagnose <子命令> --help` 或 `python scripts/<name>.py --help` 为准。

| 关注点 | 建议入口 | 说明 |
|--------|----------|------|
| **整体 KPI / 漏斗** | `mlbot diagnose e2e-kpi --logs ...` | 对应 `scripts/diagnose_e2e_kpi.py` |
| **PCM / slot 占用** | `mlbot diagnose pcm-performance --logs ... [--baseline ...] --output ...` | `scripts/diagnose_pcm_performance.py`；slot 列可选 |
| **Archetype 条数与 gate 决策** | `mlbot diagnose archetype-trade-counts --mode ... --out ...` | 需 `mode_3action` 类 parquet |
| **Gate 应用差异** | `scripts/diagnose_gate_application.py`、`diagnose_gate_diff.py`、`diagnose_gate_filtering.py` | 与基线对比规则效果 |
| **执行层 plateau / 约束** | `mlbot diagnose execution-gate-plateau`、`execution-constraints-plateau` | 对应 `scripts/diagnose_execution_gate_plateau.py` 等 |
| **基线退化** | `scripts/diagnose_baseline_performance_drop.py` | 与历史基线对比 |
| **预测集中度（NN 多头侧）** | `scripts/diagnostics/diagnose_prediction_concentration.py` 等 | 需 preds 路径 |

**与 BPC / 事件链路对齐**：若问题出在「实盘 vs 事件回测」一致性与持仓路径，见 `scripts/event_backtest.py` 与 `docs/architecture/event_drive_backtest/` 下说明。

---

## 4. `mlbot diagnose outcome-attribution` 状态

`src/cli/main.py` 注册了 **`mlbot diagnose outcome-attribution`**，但调用的 **`scripts/diagnose_outcome_attribution.py` 当前不在仓库**。在该脚本恢复或 CLI 改指其它实现之前，**不要使用该子命令**。

可改用 **`mlbot diagnose e2e-kpi`** 或自研对 `preds` / `logs` 的 PnL 归因表作为替代。

---

## 5. PCM 与 Execution 的边界（概念）

- **Execution**：单笔 archetype 的 RR、止损止盈、时间止损等（策略 `execution.yaml` 与共享执行逻辑）。
- **PCM**：多策略 / 多 archetype 的 **slot、优先级、宪法层预算**（见 `config/constitution/`、`LivePCM` 相关文档）。

**`max_slots` 等数值以当前 `constitution.yaml` 与 PCM 配置为准**，本文不硬编码为 2。

---

## 6. 运维建议

1. **固定基线版本**：每次改 Gate / 改模型前导出一份 `logs_*.parquet` 作为 baseline，文件名带日期与 Git SHA。  
2. **先确认 `gate_ok`**：再跑 `production-attribution` 与 PCM 诊断。  
3. **告警 JSON 写进版本库或 Run 元数据**，避免依赖 CLI 默认值。  
4. **告警后按上表只做一两层深挖**，避免同时改 Gate + PCM + 模型导致不可归因。  
5. **滚动月**：可与 [BASELINE_TESTING_WORKFLOW.md](./BASELINE_TESTING_WORKFLOW.md)、管线 `event_backtest` 产物一并存档。

---

## 7. 相关文档

- [BASELINE_TESTING_WORKFLOW.md](./BASELINE_TESTING_WORKFLOW.md) — 基线 logs 与 KPI  
- [PLATEAU_OPTIMIZATION_METHODOLOGY.md](./PLATEAU_OPTIMIZATION_METHODOLOGY.md) — Gate 阈值与自由度  
- [GATE_WHEN_THEN_EXECUTION_ORDER.md](./GATE_WHEN_THEN_EXECUTION_ORDER.md) — Gate 求值两条路径（`tree_gate` vs `loader.apply_gate`）  
- 事件回测与实盘对齐：`docs/architecture/event_drive_backtest/事件驱动架构说明.md`
