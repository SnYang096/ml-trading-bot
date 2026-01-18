## 统一执行日志（分阶段 + 可聚合 Canonical）

目标：让 **Nautilus 事件流** 与 **研发 pipeline** 复用同一日志形态，同时保持分阶段产物，便于排查与聚合对齐。

### 规范文件
- `config/nnmultihead/execution_log_schema.yaml`

### 分阶段日志（默认）
- 文件按 **stage** 拆分：`features` / `preds` / `router` / `gate` / `evidence` / `execution` / `returns` / `observability`
- **聚合键**：`decision_id = strategy_name + symbol + decision_ts_ns`
- **月度滚动**：每月一个 jsonl 文件（便于审计与回放）
- **gate 子原因**：`gate.reasons` 拆分为 `contract / evidence / heuristics / execution_rules` 等类别，便于定位拒绝来源

### Canonical 日志（可选聚合）
聚合后包含：
- `features` / `preds` / `router` / `gate` / `evidence` / `execution` / `returns` / `observability`
用于对比 live 与 pipeline 的 end-to-end 行为。

### 研发 pipeline 对齐方式
pipeline 现有产物：
- `preds_*.parquet`
- `mode_3action.parquet`
- `logs_3action.parquet`

可以通过 `scripts/build_execution_log_stages.py` 生成分阶段日志，再用聚合脚本生成 canonical：
- `features` / `evidence` / `gate` 在 pipeline 中可能为空
- `execution.intent` 可由 `router.mode != NO_TRADE` 推导
- `execution.submit_order` 保持 `false/null`（pipeline 是离线评估）
- `returns` 来自 `logs_3action.parquet`

### Nautilus 事件流对齐方式
在 `MetaRouterStrategy`（或其它 live 策略）中，每次 timer 触发：
1) 计算 `features`  
2) optional 预测 → `preds`  
3) router → `router`  
4) gate/evidence/heuristics → `gate` / `evidence`  
5) execution → `execution`

并以 JSONL 追加写入分阶段日志（推荐路径：`results/live_logs/<stage>/YYYY-MM.jsonl`）。

### 聚合脚本
- `scripts/aggregate_execution_log_stages.py --stage-dir results/live_logs --out execution_log.jsonl`
