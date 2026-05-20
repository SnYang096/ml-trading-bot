# Rolling Sim 与时间相关配置说明

**状态**: 与当前 `scripts/auto_research_pipeline.py` / `scripts/pipeline/config.py` 行为对齐  
**最后更新**: 2026-04

本文说明生产流水线 YAML 里各类「日期 / 月份 / 窗口」配置的含义与层次，避免与「事件回测」「结构快照」等概念混淆。

**相关代码**：

- 滚动月份枚举：`scripts/auto_research_pipeline.py` 中 `_iter_month_tokens` → `scripts/pipeline/config.py` 的 `iter_month_tokens`
- 每月标定/测试窗：`scripts/auto_research_pipeline.py` 中 `_calib_and_test_windows`
- 结构快照（仅 slow）：`_run_slow_structure_snapshot_for_month`
- 配置归一化与校验：`scripts/pipeline/config.py` 的 `load_pipeline_config`

---

## 1. 全局时间轴：`dates.*`（整次实验的 Train / Holdout）

| 配置 | 含义 |
|------|------|
| `dates.start_date` / `dates.end_date` | 数据与训练的**总日历范围**（`end_date` 可被 CLI `--end-date` 或自动探测最新数据日期覆盖）。 |
| `dates.holdout_months` | 从 `end_date` **往前**数若干个月，得到 **holdout 起点**；该起点之后直到 `end_date` 为样本外（OOS）区间。流水线启动时会打印 `Train: … ~ holdout_start`。 |
| `dates.validation_months` | 当 `0 < validation_months < holdout_months` 时，将 holdout 再切为 **Val**（前段，常用于 Gate 等调参）与 **Test**（后段，纯 OOS）。 |

**与「每月滚动」的关系**：`rolling_sim` 枚举的月份来自 **holdout 起点所在月 → `end_date` 所在月**，一般**不是**从 `start_date` 开始按月滚动。

---

## 2. 滚动仿真走哪些月：`rolling_sim` 的「事件月历」

在 `args.stage == "rolling_sim"` 路径下，月份列表为：

```text
month_tokens = _iter_month_tokens(_display_holdout_start, _display_end)
```

即：从 **holdout 首日的日历月** 起，到 **`end_date` 所在月** 止，**每个自然月**一个 `YYYY-MM`。  
因此这里的「周期」本质是：**在 holdout 内逐月重放**。

---

## 3. 每个月内部两扇窗：`rolling.windows.calibration_months`

对某一个滚动月 `M`，`_calib_and_test_windows` 定义：

| 名称 | 区间（直觉） |
|------|----------------|
| **test** | **当月 M** 的 `[月初, 月末]`：本月产出结果、事件回测（若开启）、写入当月 ledger。 |
| **calib** | **标定窗**：从 **M 月月初往前数 `calibration_months` 个月** 起，到 **M 月第一天之前** 止。快变量（阈值、prefilter、方向网格等）主要在该段拟合/选参，再在当月 test 上评估。 |

例如 `calibration_months: 6` → 每个滚动月为 **6 个月标定 + 1 个月测试**。

**注意**：这与 `dates.holdout_months` **不是同一维度**：后者是**整段数据**的 OOS 划分；前者是**每个滚动月内部的局部窗**。

---

## 4. 慢结构快照：`structure_lookback_months` 与 `slow_realistic.*`

| 配置 | 含义 |
|------|------|
| `rolling.windows.structure_lookback_months` | 仅在 **`rolling.mode: slow_realistic`** 且按 cadence 跑 **slow snapshot** 时：从「目标月上月末」往回，用**多长历史**跑结构流水线（特征搜索等到 `entry_filter`），生成冻结的 `strategies` 快照目录。 |
| `rolling.slow_realistic.cadence_months` | 隔多少个月做一次上述结构快照（与 `calibration_months` 无关）。 |

当 **`rolling.mode: turbo_fixed_features`** 时：`rolling_sim` **不调用** slow snapshot，`structure_lookback_months` / `cadence_months` **基本不参与**当月逻辑；配置可视为契约占位或便于日后切回 `slow_realistic`。

顶层 YAML 中的 `slow_loop.*` 为运营/文档契约；`rolling_sim` 在 `slow_realistic` 下实际读取的是 `rolling.slow_realistic.*`（详见 `load_pipeline_config` 中的说明与告警）。

---

## 5. 事件回测 / 交易地图：`event_backtest.*`

| 配置 | 含义 |
|------|------|
| `event_backtest.enabled` | 是否在**每个滚动月**的 fast month 流程中调用 `scripts/event_backtest.py`（以及是否写出 `event_trades_*.csv`、当月 HTML 地图等）。 |
| `event_backtest.map_extra_months` | **仅**在运行事件回测并生成交易地图时：从 `data_path` **多向前加载**若干个月的 **1m 数据**，用于 **VWAP 等指标热身**；**图表时间轴仍以当月 test 窗为主**，不把整段向前扩展的历史全部画在主轴上（见 `scripts/event_backtest.py` 中 `generate_trading_map_html` 的说明）。 |

**补充（2H VWAP 满窗）**：地图默认在 2H K 线上用 **`map_vwap_window_bars`（默认 1200）** 做滚动典型价 VWAP，约需 **~100 天**的 2H 历史才能在显示窗**左缘**也吃满窗口；日历 **3 个月**往往略短，**建议 `map_extra_months` ≥ 4**（例如 `prod_train_pipeline_2h_turbo_2024bull_thresholds_only_bpc_only.yaml` 使用 **4**）。

这与 **标定窗长度**、**holdout**、**结构快照**相互独立。

---

## 6. 方向调优节奏：`rolling_calibration.direction_tuning.cadence_months`

若配置该项：表示 **每 N 个滚动月** 才执行一次方向相关步骤（与 `rolling.windows.calibration_months` **勿混用**：后者是「标定窗长度（月）」，不是「隔几月跑一次方向」）。

---

## 7. `rolling_calibration.step_months`

`load_pipeline_config` 会校验 `rolling_calibration.step_months > 0` 并写入归一化后的 `rolling_calibration`。  
当前 **`rolling_sim` 主循环**使用 `iter_month_tokens(holdout_start, end)` **逐自然月**推进，**未**按 `step_months` 做「隔 N 月一步」的稀疏枚举；若将来要实现「两月一步」等节奏，需在 `auto_research_pipeline` 侧显式接入该字段。

---

## 8. 心智图（仅 `rolling_sim`）

```text
dates: 整段 [start_date ─────────────── end_date]
              └─ train ─┘ └──── holdout ──────────┘
                              └─ (可选) validation / test 子切分 ┘

rolling_sim 遍历：holdout_start ～ end_date 的每个自然月 M

对每个 M：
  calib = [M 月初 - calibration_months,  M 月初 - 1 天]
  test  = [M 月初, M 月末]
  （若 event_backtest.enabled）事件回测主窗 ≈ test；
      map_extra_months 仅影响多加载的 1m 数据量（用于指标计算）

slow_realistic 另分支：按 cadence 用 structure_lookback_months 做结构快照（turbo 模式无此分支）
```

---

## 9. 与 `prod_train_pipeline_*` 示例的对应关系

- **`dates` + `rolling.windows.calibration_months`**：决定「滚哪些月」以及「每月标定/测试怎么切」。
- **`turbo_fixed_features`**：固定策略根目录，不走 slow 结构快照。
- **`event_backtest`**：控制是否跑事件链路与地图向前多取的 1m 月数。

更完整的命令与工作流步骤见 [docs/workflow/PIPELINE_WORKFLOW.md](../docs/workflow/PIPELINE_WORKFLOW.md)。
