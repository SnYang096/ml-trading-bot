# 统一研发与实盘模拟管线设计

## 1. 背景与目标

当前 `pipeline run` 更偏“单次窗口最优”，对长期实盘行为模拟不足。  
本方案目标是建立一个统一框架：

- 慢变量（结构）低频更新：特征集合、prefilter/gate/entry_filter 结构、策略方向启停基线
- 快变量（参数）高频更新：月度阈值、execution、slot 分配
- 输出可拼接的长期月度 OOS 结果，用于上线前评估与回归

## 2. 术语与边界

- **慢变量**：变化慢、影响结构稳定性，默认每 3 个月或触发式更新
- **快变量**：变化快、影响当月执行表现，默认每月更新
- **carry_forward**：当月未满足 adopt 但可继承上期参数继续运行
- **quality ranking**：在同侧机会拥挤时，对 `(symbol, side)` 进行优先级排序

## 3. 配置契约

统一在 `config/prod_train_pipeline_2h.yaml` / `config/prod_train_pipeline_4h.yaml` 定义：

- `pipeline_mode`: `classic | rolling_live_sim`
- `slow_loop`: cadence、freeze 输出
- `fast_loop`: step、阈值校准、execution、pcm 开关
- `symbol_policy`: carry-forward 与 hard-fail 规则
- `direction_stack`: `ema200 | vwap_long_anchor | ensemble`（V1 先以 EMA200 为主）
- `slot_allocation`: quality ranking 参数
- `stitching`: 拼接输出指标与地图导出

## 4. 命令与阶段

### 4.1 现有阶段（兼容保留）

- `full`
- `prefilter`
- `gate`
- `entry_filter`
- `execution_opt`
- `event_backtest`
- `pcm_joint`
- `pcm_slot_grid`

### 4.2 新增阶段（本方案）

- `slow_snapshot`：仅慢变量快照（到 entry_filter）
- `fast_month --month YYYY-MM`：单月快变量复盘
- `rolling_sim`：按 holdout 月份逐月执行 fast loop 并拼接

### 4.3 新增辅助命令

- `mlbot pipeline report-side-state --run-id <run_id>`
- `mlbot pipeline debug-quality --run-id <run_id> --month YYYY-MM`

## 5. 产物契约

滚动模拟根目录：`results/.../_rolling_sim/<run_id>/`

- `monthly_ledger.jsonl`：月度摘要流水
- `stitched_summary.json`：拼接汇总指标
- `trading_map_stitched.html`：月度交易地图索引
- `fast_month_<YYYY-MM>/`
  - `fast_month_summary.json`
  - `quality_ranking_<YYYY-MM>.json`
  - `symbol_side_state.json`

## 6. 质量分（V1）

### 6.1 评分

V1 用轻量可解释分数：

- `Qv1 = 0.55 * history_edge + 0.45 * now_strength`
- 当前实现默认使用事件回测指标近似 `history_edge + 风险惩罚`
- 预留 `cvd_accel_aligned / price_efficiency_aligned` 作为 `now_strength` 的增强输入

### 6.2 排序与并列规则（Tie-break）

主排序按 `Qv1` 降序；并列时按：

1. `near_stop_rate` 低优先
2. `max_drawdown_r` 低优先
3. `n_trades` 高优先
4. `strategy` 字典序（保证复现）

## 7. Symbol Side 状态机（V1）

每个策略/方向状态：

- `active`
- `carry_forward`
- `disabled`

更新规则（简化）：

- 若当月 `sharpe_r > enable_threshold` 且 `n_trades >= min_symbol_trades_soft` -> `active`
- 否则若为 long 且上月为 `active/carry_forward` 且未触发 hard-fail -> `carry_forward`
- 否则 `disabled`

## 8. 与实盘主循环对齐

`run_live.py` 对齐建议：

- 热更新：阈值、execution、side state
- 需重启：特征集合与 constitution 硬约束
- retrain 检查仍由现有周期任务触发，rolling 结果作为决策依据

## 9. 验证计划

1. **兼容性**：`classic` 流程输出不变
2. **功能性**：新阶段产物齐全、CLI 可调用
3. **一致性**：`monthly_ledger` 与 `stitched_summary` 聚合一致
4. **可复现**：固定 seed/date 结果稳定
5. **策略对比**：`global_only` vs `hybrid_carry_forward`，`quality_ranked` vs baseline

## 10. 风险与后续

- V1 风险：小样本导致质量分波动
  - 缓解：最小交易数门槛、hard-fail、可复现排序
- V1.1 方向：
  - 引入 `vwap_long_anchor` 方向模式实现
  - 引入更细粒度 symbol 级 CVD/效率特征与权重学习
