# TPC trend_pool_guard 并发 symbol 上限验证（2026-06-12）

## 问题

生产 `live/highcap` 当前为 **1→2**（`max_unprotected_symbols=1`，`max_symbols_after_unlock=2`）：

- 空仓时只能先开 **1 个 symbol** 的裸仓；
- 必须有一笔 **breakeven 锁仓** 后才能开第 2 个 symbol；
- 总 symbol 数硬顶 **2**。

担心：等第一笔 BE 时，后续信号可能已变差（错过最佳入场窗口）。

## 假设（你的 3,3 配对）

| 阶段 | 期望行为 |
|------|----------|
| 初始 | 可同时开 **3** 个 symbol 裸仓 |
| 每 1 笔 BE | 多开 **1** 个 symbol 槽 |
| 3 笔初始全 BE | 最多 **6** 个 symbol |

## 代码实际语义（`live_pcm._trend_pool_guard_reject_reason`）

现有参数 **不是**「3 + count(BE)」的动态公式，而是两个独立硬顶：

1. **`max_unprotected_symbols`**：同时 **未 protected**（未 breakeven 锁仓）的 distinct symbol 数上限。
2. **`max_symbols_after_unlock`**：同时 **持仓** 的 distinct trend symbol **总数** 上限（名称易误解，实为全局 symbol cap）。

近似映射：

| 实验 ID | max_unprot | max_total | 行为摘要 |
|---------|------------|-----------|----------|
| **G0_prod_1_2** | 1 | 2 | 生产现状 |
| **G1_be1_3** | 1 | 3 | 先 1 裸仓，BE 后可扩到 3 symbol |
| **G2_be3_3** | 3 | 3 | 初始最多 3 裸仓，总 cap 3（不能 BE 后再扩） |
| **G3_be3_6** | 3 | 6 | **最接近你的 3→6 模型**：初始 3 裸仓；每 BE 降 unprot 计数可开新 symbol；总 cap 6 |
| **G4_guard_off** | — | — | 关闭 guard（对照：slot_count=10 为实际上限） |

**G3 与「+1/BE」的差异**：只要 `unprotected < 3` 且 `total < 6` 即可开新 symbol，不要求「恰好 BE 一笔才 +1」；但 BE 会降低 unprotected 计数，效果上接近阶梯解锁。

TPC prod execution：`breakeven.trigger_r=6.0 ATR`（约 1.5R 锁仓），与 live 一致。

## Phase 3 跑法

```bash
# 0) 刷新 constitution 变体（改 base 后重跑）
PYTHONPATH=src:scripts python scripts/research/prepare_tpc_pool_guard_constitutions.py

# 1) 全窗 canonical（2022→2026-04，6 symbols）
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260612_tpc_trend_pool_guard_sweep/tpc_pool_guard_grid.yaml \
  --quiet-signal-logs

# 2) 快速 smoke（单 variant）
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --strategy tpc \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT \
  --start-date 2024-01-01 --end-date 2025-01-01 \
  --constitution-yaml config/experiments/20260612_tpc_trend_pool_guard_sweep/constitution/G3_be3_6.yaml \
  --output-dir results/tpc/experiments/pool_guard_20260612/smoke/G3_be3_6 \
  --quiet-signal-logs
```

## 读结果

每个 variant 的 `event_backtest.json` / `capital_report.json`：

- **total_r** / **max_drawdown_r**（主 KPI）
- funnel：`reject_pcm_trend_pool_unprotected_cap`、`reject_pcm_trend_pool_post_unlock_cap`、`reject_pcm_trend_pool_corr`

排序：先 **maxDD 升序**，同 DD 再 **total_r 降序**（见 `docs/experiments/z实验_trend_slot_guard/trend_slot_guard_validation_matrix.md`）。

## 决策

见 [`DECISION.md`](DECISION.md)（跑完 grid 后填写）。

| 字段 | 值 |
|------|-----|
| 策略 | tpc（prod tree `config/strategies`） |
| Grid | [`tpc_pool_guard_grid.yaml`](tpc_pool_guard_grid.yaml) |
| Constitution base | [`constitution/base_tpc_prod.yaml`](constitution/base_tpc_prod.yaml)
