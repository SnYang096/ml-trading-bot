# 多腿联合 Timeline 回测 vs 实盘对齐结论

**日期：** 2026-06-16  
**配置：** `chop_prod` + `trend_prod` + `live/highcap/config/constitution/constitution.yaml`  
**引擎：** `scripts/backtest_multileg_timeline.py`（Path B：bar-level pending order book）  
**Sizing：** equity 10,000，`segment_dd_target: 0.072`，compound sizing 默认开启

---

## 1. 当前联合回测结果（recent_6m_oos）

唯一在 Path B 修复 + ledger 清理后完整跑通并验证的 canonical 段：

| 指标 | 数值 |
|------|------|
| 区间 | 2025-12-01 → 2026-05-31 |
| 起始 / 结束权益 | 10,000 → **20,483** |
| 收益率 | **+104.83%**（6 个月） |
| 峰值权益 | 21,150 |
| 最大回撤（从峰值） | **-11.55%** |
| 成交 / 被拒 | 8073 ok / **2384 rej** |
| 是否 halt | **否** |
| 累计手续费 | **9,789 USDT**（mock 逐笔真实扣费） |
| `realized_pnl` | **+10,483**（= equity 变化，与 wallet 一致） |

四段 canonical grid 在修复后需重跑落盘，见 `variant_grid.yaml` → `results/multileg_joint/sizing_072_20260613_timeline/`。

---

## 2. 和实盘一样吗？

**结论：代码路径高度对齐，执行环境不等价；回测数字不能当实盘预期。**

### 2.1 已对齐（与 `run_multi_leg_live.py` 共用）

```text
ChopGridLiveEngine / DualAddTrendLiveEngine
  → MultiLegLiveOrchestrator
  → MultiLegPortfolioRiskGovernor
  → MultiLegExecutionAdapter
  → MockBinanceAPI（替代 BinanceAPI）
```

- chop 使用 `bar_simulation=False`；成交经 adapter + pending order book；inventory 由 `on_execution_results → on_execution_report` 更新
- 同一账户 **symbol owner** 互斥（chop ↔ trend 同 bar 换手）
- **compound sizing** + constitution 风控（gross/net、daily loss、max DD）
- **1min 执行 + 2h 信号**；parity 测试 timeline vs daemon 在同 bar handoff 上一致

### 2.2 仍与实盘有差距

| 维度 | 回测 | 实盘 |
|------|------|------|
| 成交来源 | 1m bar high/low 触价即成交 | User Data Stream + 交易所真实撮合 |
| 对账 | `reconcile=False` | 每 60s / 有 action 时对账 |
| 滑点 / 排队 | 无排队，LIMIT 按挂单价全成 | 可能部分成交、拒单、late fill |
| 保证金 | mock 无 margin gate | 交易所拒单 / 强平风险 |
| Funding | 无 | 有 |
| 精度 / 步进 | 简化 | 交易所 lot/tick 规则 |
| 特征来源 | parquet 离线预计算 | Feature Bus 磁盘流 |
| 生产成熟度 | 可批量回放 | testnet/shadow 硬化中，非 mainnet 生产就绪 |

Mock 撮合：LIMIT 在 bar 触价时按挂单价**全量成交**（`mock_binance_api.match_pending_orders`），无 order book 深度与排队竞争。

### 2.3 修复前后对比（为何旧数字不可信）

| 阶段 | recent_6m 典型表现 | 原因 |
|------|-------------------|------|
| 修复前（DECISION §5 旧 timeline） | +1657%（5 万本金）、MaxDD -3% | limit 不成交、ledger 假账、governor 未真实生效 |
| Path B 初版（side 大小写 + chop inventory 未桥接） | 开局数小时 halt -20% | 零成交 / 裸仓 / 假 PnL |
| **当前（a7561287 + ledger 方案 A）** | **+104.8% / MaxDD -11.55%** | 撮合、inventory、wallet 口径自洽 |

旧 `ledger_realized_pnl = -101k` 与 equity +104% 冲突 — 旁路 ledger 已删除；`total_fees_usdt` 改由 mock 逐笔累计（约 9789 USDT，旧 ledger 报告的 3470 同样不可信）。

### 2.4 与实盘数字对照

仓库内**无**同配置、同区间的 live PnL 对照表。`DECISION.md` 中 trend-only recent_6m 约 **+14.4%**（regime_only）；当前联合 **+104.8%** 含 chop grid + compound + 共享账户复利，不可与单策略或实盘直接混比。

---

## 3. 关键修复清单（2026-06-16）

| Bug | 修复 |
|-----|------|
| `match_pending_orders` side 大小写 | `str(order["side"]).upper()` |
| chop `bar_simulation=False` 不建 inventory | filled → `on_execution_report` 桥接 |
| `daily_pnl` 恒为 0 | `_day_start_equity` 日初快照 |
| 旁路 ledger 与 wallet 严重偏离 | 删除 ledger 记账；equity/fee 以 mock wallet 为准 |
| `sync_engine_realized_bridge` 误改 ledger | 仅 `bar_simulation=True` 时更新 wallet（Path B chop 不走此路径） |

单测：`test_mock_binance_pending_match_case.py`、`test_chop_execution_bridge.py`、`test_multileg_timeline_account.py`；parity：`test_multileg_timeline_daemon_parity.py`。

---

## 4. 如何使用这些结果

**可以：**

- chop + trend 联合逻辑、风控、sizing 的**相对排序**（四段对比、参数矩阵）
- 与修复前 garbage 回测区分，避免 +8000% / +1657% 类假象

**不可以：**

- 将 +104.8% 当作 testnet/mainnet 未来 6 个月预期
- 用已删除的 `ledger_realized_pnl` 做归因

**建议：**

1. 重跑四段 canonical grid 落盘（本实验 `variant_grid.yaml`）
2. testnet shadow 同配置跑 2–4 周，对比成交数、拒单率、持仓漂移
3. 实盘 gap 收敛后再更新 `LAYER_PROMOTION_CRITERIA` 三条杠

---

## 5. 重跑命令

```bash
python -m scripts.event_backtest \
  --variant-grid config/experiments/20260613_multileg_sizing_validate/variant_grid.yaml
```

产物：`results/multileg_joint/sizing_072_20260613_timeline/timeline/{segment}/summary.json` + `joint/summary.json`。

---

## 6. 四段 canonical grid 结果（2026-06-16 重跑）

**命令：** `python -m scripts.event_backtest --variant-grid config/experiments/20260613_multileg_sizing_validate/variant_grid.yaml`  
**落盘：** `results/multileg_joint/sizing_072_20260613_timeline/joint/summary.json`

| 段 | 区间 | 收益 | MaxDD | 成交/拒 | Halt | 备注 |
|----|------|------|-------|---------|------|------|
| bear_2022 | 2022-01 → 2023-11 | **+128.3%** | -81.9% | 9459 / 17699 | ✅ 2022-06-14 | halt 后仍继续跑（仅 block 新开仓） |
| bull_2023_2024 | 2023-06 → 2025-01 | **+56.9%** | -16.4% | 20542 / 12384 | ❌ | 全程未触发 kill switch |
| recent_range_to_bear | 2025-01 → 2026-05 | ~~**-144.0%**~~ → **+4.87%** | ~~-131.1%~~ → -26.6% | 14093 / 13643 | ✅ 2025-02-03 | 修复 A+B 后 equity_end **+10,487**（见 §8）；旧穿仓数已废弃 |
| recent_6m_oos | 2025-12 → 2026-05 | **+104.8%** | -11.6% | 8073 / 2384 | ❌ | 与 sanity 一致 |

**解读：**

- `recent_6m_oos` / `bull_2023_2024` 表现稳健，适合作为当前 regime 参考。
- `bear_2022` halt 后权益仍为正（+128%），因 backtest **不在 halt 时 break**——pending 成交与降风险 action 仍继续；MaxDD -82% 反映 halt 前峰值到谷值的极端波动。
- `recent_range_to_bear` 负权益是 **mock 缺少 margin gate** 的已知局限：governor 拒单 16976 次仍无法阻止账户穿仓至负值；实盘会被交易所拒单/强平截断，此段数字**不可直接外推**。
- 四段均为独立 10k 本金、compound sizing；段间不滚存权益。

---

## 7. `recent_range_to_bear` 穿仓根因分析（2026-06-16）

**现象：** 2025-02-03 触发 `dd>20%` halt 后，权益从 10,959 继续恶化至 **-4,396**（2025-07-18 平仓完毕），段末 `equity_end < 0`。

### 7.1 时间线

| 时点 | 权益 | Wallet | 未实现 | 持仓 gross | 说明 |
|------|------|--------|--------|------------|------|
| 峰值 2025-02-02 | **14,134** | 14,014 | ≈0 | 12.6k | compound sizing 放大后峰值 |
| **Halt 2025-02-03 04:00** | **10,959** | 13,939 | **-2,979** | **83.8k** | DD kill switch 触发；5 个 mock 持仓、102 个 TP/SL 挂单 |
| 同日 2025-02-03 20:00 | 1,981 | 11,813 | -9,832 | 71.4k | halt 后 **12h 内** 权益再跌 ~82%（纯 mark-to-market） |
| 2025-05-22 | **< 0** | 仍 >0 | 大负 | 有仓 | 权益首次穿零（未实现主导） |
| 2025-07-18 | **-4,396** | **-4,396** | 0 | 0 | 全部平仓完毕，wallet 定格 |

Halt 后成交统计：**0** 笔 entry fill、**8** 笔 reduce fill（instrumented run）；16976 次 governor 拒单发生在 halt **之前**的前 5 周。

### 7.2 根因链（按重要性）

**① Halt 后无法主动平仓（主因）**

```540:541:scripts/backtest_multileg_timeline.py
            if account.halted:
                continue
```

`halted=True` 时 **整段跳过 orchestrator**，包括 `market_exit` / `cancel_protection`。注释写「halt 后继续跑 pending 成交与降风险 action」，但 **market_exit 根本进不了 orchestrator**。

后果：2025-02-03 留下的 5 个持仓（gross **83.8k**，约为 10k 本金的 **8.4 倍**）无法被 engine 的 regime/risk exit 平掉，只能靠 mock 里已挂的 TP/SL 触价平仓；价格不对路时持仓长期「烂」在账上，1m mark 持续计未实现亏损。

**② Mock 无 margin gate / 允许负 wallet（穿仓放大器）**

- `_apply_open` 不检查可用保证金，wallet 可随 realized 亏损降至 **负数**
- 实盘：交易所拒单或强平，账户不可能到 -4,396 USDT
- `on_bar_close` 的 `equity<=0` halt **仅在 `not self.halted` 时检查**——已被 DD halt 后，即使权益穿零也不会二次熔断或强平

**③ Halt 前 gross Exposure 膨胀（触发条件）**

- Compound sizing 把 peak equity 滚到 14.1k → governor 上限同步放大（`max_gross ≈ 2.7 × equity ≈ 37.8k`）
- 2025-02-03 02:00–04:00 间，5 个 symbol 的 limit 单在 1m 撮合中集中成交，mock gross 飙至 **83.8k**（超 portfolio cap ~2.2×）
- Governor 只在 **place 时刻** 做投影检查；已挂 pending limit 在后续 1m bar **不再过 governor**，多 level 同 bar Fill 可突破 cap
- Halt 时 wallet 仍 +13.9k（realized 盈利），但 **-3k 未实现** 叠加峰值回撤 → 触发 20% DD

**④ TP/SL 挂单过多且 halt 后大量失效（102 → 45 orphan）**

- Halt 时 102 个 `reduce_only` 保护单；仅少数触价成交
- 2025-07-18 平仓后仍剩 **45** 个 pending reduce（orphan protection），对账/清理缺失

### 7.3 与实盘差异

| 机制 | 回测实际行为 | 实盘预期 |
|------|-------------|----------|
| DD halt 后 | 跳过 orchestrator，持仓可无限期 mark 亏损 | Safety gate / 人工 flatten |
| 保证金 | 无，wallet 可到 -4396 | 拒单 / 强平 |
| Gross cap | place 时检查，pending fill 可超限 | 交易所 + 账户级 gate |
| Kill switch 后 | 回测继续 16 个月到段末 | 应停止开新仓并清仓 |

### 7.4 修复建议（优先级）

1. **Halt 后仍执行降风险 action**：`halted` 时只 skip `place`/`place_protection`，**保留** `market_exit` / `cancel` / `cancel_protection` 进 orchestrator；或 halt 瞬间 mock 市价全平
2. **Mock margin gate**：`_apply_open` 在 `wallet < required_margin` 时 reject；`wallet_usdt` floor 0
3. **Pending fill 后再验 cap**：1m 撮合后若 gross 超限，自动 cancel 最远 entry pending
4. **Halt 后 break 或冻结 mark**：halt 时 snapshot 权益，后续仅处理 reduce fill，不再计未实现恶化（可选，偏保守）

**结论：** `-144%` 不是策略 alpha 信号，而是 **「halt 不平仓 + mock 无保证金 + pending fill 突破 gross cap」** 叠加的仿真 artifact；该段 **不可用于 promotion**，需先修仿真/account safety 再重跑。

---

## 8. 处置与修复落地（2026-06-16）

### 8.1 已修复

| 代号 | 范围 | 文件 | 改动 |
|------|------|------|------|
| **A** | 回测 | `scripts/backtest_multileg_timeline.py` | 删除 `if account.halted: continue`；halt 时只挡 `place`，放行 `market_exit`/`cancel`/`cancel_protection`/`place_protection`（reduce-only）进 orchestrator，对齐 live kill switch（`_RISK_INCREASING_ACTIONS`） |
| **B** | 回测/mock | `src/order_management/mock_binance_api.py` | `_apply_open` 加 margin gate：`wallet + 未实现 <= 0` 拒绝开仓并返回 `status=rejected/insufficient_margin`；`match_pending_orders` 的 entry fill 同样被拒（丢单不成交）；reduce-only 不受影响 |
| **C** | **实盘** | `src/time_series_model/live/dual_add_trend_live_engine.py` | `_target_exits`（内部 TP）/ `_exit_all`（regime exit）在发 `market_exit` 的同一 bar，对该腿 `protection_order_ids` 发 `cancel_protection`，关掉「reduce-only 保护单存活到下次 60s reconcile」的 orphan 窗口（避免误平同向新仓） |

单测：`tests/unit/test_dual_add_trend_live_engine.py`（`test_exit_all_cancels_position_protection_orders` / `test_target_exit_cancels_position_protection_orders`）、`tests/unit/test_mock_binance_pending_match_case.py::TestMarginGate`（4 例）。

A 的有效性由 `recent_range_to_bear` 重跑验证（halt 后持仓被及时降险平掉，`equity_end` 不再穿零）：

| 指标 | 修复前 | 修复后（A+B） |
|------|--------|---------------|
| `equity_end` | **-4,396** | **+10,487** |
| `return_pct` | **-144.0%** | **+4.87%** |
| `max_drawdown_pct` | -131.1% | -26.6% |
| `trades_ok` | 1,867 | 14,093 |
| halt | 2025-02-03 04:00 | 2025-02-03 04:00（同点） |

同一 DD halt 触发点，但 halt 后 `market_exit`/`cancel` 进 orchestrator → 持仓被及时平掉，权益稳定在小幅正收益，而非 16 个月 mark 亏损至 -4,396。该段仍**不建议用于 promotion**（compound sizing 在峰值放大 + DD halt 的极端段），但已不再是穿仓 artifact。

### 8.2 记录不改（设计性差异 / 超范围）

| 代号 | 项 | 结论 |
|------|----|------|
| **P1** | 回测无 reconcile（protection 永不重检、orphan 不清理） | **暂不在回测启用全量 reconcile**：开销大且改变回测性质（用于策略相对排序）。C 已覆盖「exit 时主动 cancel」这一主路径；chop 侧已有 reconcile 期 orphan/stale 剪枝。实盘 reconcile 每 60s 正常运行。 |
| **P2** | 回测 halt 模型与 live `MultiLegKillSwitchTracker` 不一致（无持久化/cooldown/周月亏损/自动恢复） | **有意分离**：回测单次连续运行不需要状态持久化与 cooldown 恢复；live 已有完整 tracker。A 修复后回测 halt 行为（block 新开仓、放行降险）已与 live 语义对齐，足够用于段内评估。 |
| **P3** | mock 不模拟交易所拒单（min-notional / 限频 / 价距） | **部分覆盖**：B 已补上保证金维度的拒单（最易造成穿仓的一类）。min-notional/限频等属过度仿真，对策略相对排序影响有限，暂不补。 |

### 8.3 顺带 review（已提交代码 / 未提交文档）

- **`_sync_phantom_positions`（orchestrator Plan A，已提交）**：复核通过——class 内定义正确、`__init__` 已初始化 `_inventory_synced`/`_phantom_confirm`，调用的 `storage.close_positions_by_leg_ids(strategy, symbol, leg_ids, reason, run_id)` 签名一致；仅在首次对账后、连续 N cycle 确认 `(sym,side)` 交易所为空时才清理，防瞬时空快照。
- **测试隔离 flake（预存，与本次改动无关）**：`pytest-randomly` 随机序下，组合跑 `tests/order_management/test_multileg_timeline_daemon_parity.py` 偶发 `AttributeError: ... has no attribute '_sync_phantom_positions'`；固定序（`-p no:randomly`）106 例全过。疑似某测试改动了 `MultiLegLiveOrchestrator` 类对象，属独立的 test-isolation 问题，待单独排查。
- **`docs/.../20260616_phantom_positions_postmortem_CN.md`（未提交）**：Bug #8 幻影仓位为独立的 live 议题（Plan A 已落地代码，长期需 Plan C user-stream/protection 对账），不在本次回测对齐范围。
