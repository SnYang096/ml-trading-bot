# TaskSpecs（仅用于 nn/RL 侧的“任务定义工件”）

本目录用于把 **Task（任务）** 固化成可落盘、可审计、可复现的 YAML 工件。

重要约束：

- 这些 TaskSpec **不改动**树模型的策略优先流程（`config/strategies/**` + `train_strategy_pipeline.py` 仍然独立运行）。
- TaskSpec 的目的，是让 nnmultihead / rl 这条线在产物（`meta.json/metrics.json/report.html`）里携带一个稳定的 `task_id`，从而：
  - 不同训练方法（Rule/BC/Offline RL/Tree policy）可在同一评估口径下横向对比
  - 后续 recorder/报告索引更简单（按 `task_id` 聚合）

字段约定（最小集）：

- `task_id`：稳定 ID（建议只包含字母/数字/下划线）
- `family`：任务族，例如 `primitives` / `router_3action`
- `data`：timeframe、bar_hours 等“数据口径”
- `labels`：horizon、entry_offset 等“任务定义”
- `evaluation`：rolling IC/ICIR、conditional slices（near_sr/trend_high/compression_high）
- `feature_contract`：引用 nnmultihead config 下的 `feature_contract.yaml`（可选）


