# Lottery100 管线集成（当前口径）

## 当前结论

- **主评估链路**：`auto_research_pipeline.py` 识别 `strategy_family: lottery100` 后，走  
  **Feature Store → Event Execution Optimize → event_backtest（含 trading map）**。
- **与其它策略一致点**：仍使用 `config/strategies/<strategy>/archetypes/*.yaml`、实验目录隔离、`--adopt` 合并回生产。
- **B+ 定位**：`run_lottery_research_bundle.py` 保留为离线容量研究辅助，不是当前主决策链路。

---

## 对齐关系（按用户心智）

| 维度 | BPC/ME 等常规策略 | Lottery100 当前 |
|------|-------------------|-----------------|
| 策略目录 | `config/strategies/<name>` | `config/strategies/bad-candidates/lottery100` |
| 运行入口 | `auto_research_pipeline.py` | 同一入口（family 分支） |
| 实验目录 | `results/research_history/...` | 相同 |
| 回测主指标 | backtest + event_backtest | event_backtest（主） |
| 地图产物 | `trading_map_*_event.html` | 相同（event 产出） |
| 采纳机制 | `--adopt` | 相同 |

---

## KPI 映射（保持字段结构一致）

- `kpi_gates.prefilter`：宏观体制覆盖率（当前更多用于诊断，不做 NN prefilter 过滤）。
- `kpi_gates.gate`：特征门或结构门阈值（沿用字段名，便于统一认知）。
- `kpi_gates.backtest`：事件回测主门禁（`min_trades`、`min_win_rate`、`max_drawdown_approx`）。
- `kpi_gates.deploy`：上线前最小样本约束（与其它策略同名同语义）。

---

## 输出位置

- 单次 run：`results/research_history/lottery100/<timestamp>/results/lottery100_event/`
  - `event_backtest_lottery100.json`
  - `event_trades_lottery100.csv`
  - `trading_map_lottery100_event.html`
- 离线研究（可选）：`results/lottery100_bundle/`（B+ 与容量统计）。
