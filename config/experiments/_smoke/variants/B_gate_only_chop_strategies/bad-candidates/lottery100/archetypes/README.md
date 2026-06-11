# Lottery100 `archetypes/` — 与 ME/BPC **同目录契约**

本目录与 **`config/strategies/*/archetypes`** 对齐，供：

- **`GenericLiveStrategy`** / **`event_backtest.py`** 加载 `direction` / `prefilter` / `gate` / `entry_filters` / `execution`
- **`auto_research_pipeline.py`** 在 **`strategy_family: lottery100`** 下跑 event_backtest 并把事件指标写入 **`execution.yaml`** 的 `last_evaluation`（与其它策略 promote 写入 archetypes **同一 adopt 路径**）

## 与其它策略的对应关系

| 文件 | 语义（Lottery100） |
|------|---------------------|
| **prefilter.yaml** | Regime / 宏观可交易体制占位；主线体制仍见 `../leverage_capacity_v4.yaml` |
| **gate.yaml** | 容量安全 + 末端追涨 / funding（FS 特征名） |
| **direction.yaml** | **fixed_direction: long**（必选，否则 GenericLiveStrategy 拒绝下单） |
| **entry_filters.yaml** | 特征备忘门（默认空）；详细备忘见 `../gate_draft.yaml` |
| **evidence.yaml** | 可选软评分（默认空） |
| **execution.yaml** | H=120、费用、名义杠杆、`cap_leverage_to_lmax` 契约（防「纸面全仓爆仓」语义） |

## KPI

全局阈值见 **`config/kpi_gates/lottery100.yaml`**；prod 管线里 **`strategies.lottery100.kpi_gates`** 与之对齐，可自行只改各层 KPI 数值。
