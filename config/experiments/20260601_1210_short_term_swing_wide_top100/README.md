# short_term_swing wide-pool top-100 re-test on recent_6m_oos (20260601)

| 字段 | 值 |
|------|-----|
| 目录 | `20260601_1210_short_term_swing_wide_top100/` |
| 日期 | 2026-06-01 |
| 策略 | `short_term_swing` (tree) |
| 前置 | `20260601` wide prepare + IC prune (`ic_prune_wide_20260601`, 252 pass columns from ~882-col pool) |
| 决策 | (待填，跑完后写 DECISION.md) |

## 目的与假设

**问题**：之前 5 月底 fast_scalp / short_term_swing 在 curated 小池子上看到 alt 子集或特定 τ 下有正 holdout RR；但 20260601 wide 全特征池子（~880+ 列）IC 挑 top-20 后，6 币 pooled holdout τ 全面负（最佳 q=0.05 Sharpe -1.16）。

**本次实验**：
- 复用同一次 wide IC 产生的 ranked 列表（252 pass → top-100，而非之前的 top-20）。
- **关键改进**：使用 `config/market_segment.yaml` 新增的 `recent_6m_oos`（2025-10-01 → 2026-03-31）作为主要 OOS 窗口。
  - 之前 wide IC holdout 用到 2026-04-01，略超当前数据上限（~2026-03-30）；本次用干净的最近 6 个月真实 OOS。
  - 与 TPC gate 三个 canonical 段的“当前 regime”评估口径对齐（便于未来 gate + tree 联合决策）。
- 目标：观察放宽到 100 列（仍来自同一个 IC 排序）是否能在**当前 regime 的 focused 6m OOS** 上产生可交易的 τ plateau（6 币 pooled 或至少 alt 子集）。

**假设**：
- H1：top-100 在 recent_6m_oos 上 τ RR 优于之前的 top-20（更多特征 → 树有更多信号可组合）。
- H2：即使整体 Pearson 仍弱/负，极端 quantile（q=0.05~0.10）下 6 币或 alt 子集能出现正 Sharpe（类似 5 月 fast_scalp alts 的条件 promote 现象）。
- H3：若仍全面负 → 进一步确认“大池子 IC 自动选材”对 short_term_swing 当前不 work，需要更强的 curation / regime-conditional / 多段稳定性过滤。

## 规范说明（本次严格遵守）

- 实验目录命名：`YYYYMMDD_HHMM_<简短主题>`（推荐格式，见 `config/experiments/20260601_1130_tpc_gate_final_lock/README.md`）。
- 策略配置变更放在独立 slug `config/strategies/tree_strategies/short_term_swing_wide_top100_test/`（不污染 canonical `short_term_swing`）。
- 市场阶段使用 `config/market_segment.yaml` 的 `recent_6m_oos`（已于本次实验同期添加，便于 variant-grid / 未来 event_backtest 复用）。
- 产物路径、rd_loop yaml、README/DECISION 全部在 `config/experiments/` 下 traceable。

## 物料

- `rd_loop_short_term_swing_wide_top100.yaml` — 编排（复用 wide prepare parquet，train scoped to top-100，tau on recent_6m_oos）。
- 策略配置：`config/strategies/tree_strategies/short_term_swing_wide_top100_test/`（features.yaml 已写入 top-100 + atr_f；ic_screen.yaml 等继承自 canonical）。
- 前置 wide IC 排名：`results/rd_loop/short_term_swing_ic_plateau/ic_prune_wide_20260601/ic_prune_holdout.json`（top-100 即该列表前 100）。

## 如何运行

```bash
# 一条命令驱动（推荐）
PYTHONPATH=src:scripts:. python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260601_1210_short_term_swing_wide_top100/rd_loop_short_term_swing_wide_top100.yaml
```

分步（如果想精确控制）：
1. Train（会自动用该 slug 的 features.yaml 做列 scoping）：
   ```bash
   PYTHONPATH=src:scripts:. python scripts/rd_loop.py \
     --hypothesis-yaml ... --only-steps train
   ```
2. τ scan（关键：用 recent 6m 日期窗）：
   ```bash
   ... --only-steps tau-scan
   ```

## 预期产物

- `results/rd_loop/short_term_swing_wide_top100_recent6m/`（或 rd_loop yaml 里指定的 output_dir）
  - train artifact（short_term_swing_wide_top100_test）
  - `holdout_rr_recent6m/` 或类似（用 2025-10-01 → 2026-03-31 的 τ 网格 + vectorbt RR）
- 更新后的 `DECISION.md`（跑完后填写 Go/No-Go + 对比 top-20、对比 5 月结果、是否值得继续 100~200 列或改用 tree_core 子池等）。

## 后续清理 / 决策输入

跑完后：
1. 汇总 recent_6m_oos 下 6 币 pooled + alt-only / majors-only 子集的 Sharpe / Return / trade count。
2. 与之前 top-20 wide + 5 月 curated 结果做 Pareto 对比。
3. 决定：
   - 是否 promote 任何 variant（带约束 paper/shadow）？
   - 是否需要进一步实验（例如只在 tree_core_120T ~95 nodes 上 wide、或加 regime-conditional IC、或换 forward_rr 做 IC target）？
4. 同步 `config/strategies/tree_strategies/short_term_swing/ic_screen.yaml` 或 features（如果有可复用的洞见）。
5. 如涉及 event_backtest grid，参考 TPC gate 做法建 variant + segment_matrix（可用新 `recent_6m_oos`）。

## 数据与窗口对齐说明

- 当前可用数据上限 ≈ 2026-03-30（120T parquet）。
- `recent_6m_oos` 设为 2025-10-01 → 2026-03-31（安全、无越界）。
- 之前 wide IC/holdout 用到 2026-04-01 仅多出极少量 bar；本次用严格 6m 更干净，也便于未来与 TPC gate 的“当前 regime”决策口径统一。

---

**状态**：目录 + 规范结构已建立（2026-06-01）。等待用户确认后可立即启动 rd_loop 跑数 + 监控。
