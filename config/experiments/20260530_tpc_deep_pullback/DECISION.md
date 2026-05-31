# TPC「深回撤 + 吸收」假设与实验记录（2026）

**实验目录：** `config/experiments/20260530_tpc_deep_pullback/`  
**rd_loop：** `rd_loop_tpc_deep_pullback.yaml` → `results/rd_loop/tpc_deep_pullback/`  
**因果验证：** `tpc_deep_pullback_ablation_grid.yaml` → `results/tpc/experiments/deep_pullback_ablation/`

## 核心假设

| 编号 | 假设 | 离线怎么证 | 回测怎么证 |
|------|------|------------|------------|
| H1 | 牛市深回踩（`tpc_pullback_depth` 高）比浅回踩/延续更有性价比 | bull plateau：depth ≥ τ → good-rate / forward_rr ↑ | E1/E4 入场 depth 中位 ≥ 0.30 |
| H2 | 高 `path_efficiency` = 延续区，应留给 BPC，TPC 应 deny | path_efficiency ≤ τ_pe lift | E3/E4 入场 PE 中位 ≤ 0.55 |
| H3 | 订单流吸收 + 回踩缩量确认后再入场 | deep_absorb condition-set > prod_entry | funnel 少「冲动段」入场 |
| H4 | 深回踩可用更紧止损（不必死守 4 ATR） | — | E5 vs E4：bull CAGR/DD 不恶化 |

## 与 BPC 划界

- **BPC：** `gate_pullback_on_breakout=True`，突破→回踩→延续。
- **TPC：** `gate_pullback_on_breakout=False`；当前 entry 过松（cvd≥0.629, recovery≥0.039），约 76% 入场 depth≈0 → 语义偏 BPC 延续。
- **direction：** 已 promote（EMA1200 align）；6 币 promote 验证 structural ≈0%，本实验 focus **入场语义**。

## Phase 0 — 数据

```bash
RUN_ID=train_final_tpc_deep_$(date +%Y%m%d_%H%M%S)
mlbot train final --no-docker --prepare-only \
  -c config/strategies/tpc -t 120T \
  --symbol BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,SOLUSDT,XRPUSDT \
  --start-date 2023-01-01 --end-date 2026-04-01 \
  --output-root results/train_final/tpc/${RUN_ID}
```

- **① 扫描 / bull：** 2023-01-01 → 2025-01-01，`ema_1200_position>=0.10`
- **② holdout recent：** 2025-04-01 → 2026-04-01

## Phase 1 — 两轮 rd_loop（①）

1. **Pass 1** `rd_loop_tpc_deep_pullback.yaml`：depth / PE / cvd / vol 分量 plateau + prod 基线 + IC（**不在此步写死 deep_absorb 组合**）。
2. **Pass 2** `rd_loop_tpc_deep_pullback_entry_compare.yaml`：把 plateau 选定的 τ 写入 `deep_absorb` 行，再与 `prod_entry` 做 condition-set。

`rd_loop` **不会**自动把 plateau 结果写回 condition；Pass 2 阈值必须来自 Pass 1 产出，禁止与 plateau 无关的手拍。

**label / KPI 口径（按层）：**

| 层 | 特征 | KPI |
|----|------|-----|
| prefilter | `tpc_pullback_depth` | `success_no_rr_extreme`（少踩坑） |
| gate 语义 | `path_efficiency_pct` | `success_no_rr_extreme` |
| entry 候选 | `tpc_cvd_absorption` / `tpc_vol_pullback_confirm` | 各跑 **`snotio-plateau` + `entry_rr`** 与 **`success_no_rr_extreme`** 双 KPI 交叉核对 |
| 符号一致性 | 全部 | `ic-decay` + `forward_rr` |

Entry 阈值 promote 前必须看 snotio 或 E4 回测的 Mean R，不能只凭 success 率。

## 离线通过标准（①）

- [ ] 存在 τ_deep ∈ [0.30, 0.50]，bull 段 \|z\|>2 且 good-rate Δpp > 0
- [ ] path_efficiency ≤ τ_pe（初值 0.55）优于当前入场分布
- [ ] deep_absorb good-rate > prod_entry，触发率 > ~30% 原 entry
- [ ] IC 符号与假设一致

## Phase 2 — 变体树（ablation）

| ID | 改动 | 树路径 |
|----|------|--------|
| E0 | prod（direction align + 旧 entry） | `config/strategies` |
| E1 | prefilter depth 双边带 | `config_experiments/tpc_deep_prefilter_strategies` |
| E2 | entry 吸收 + 反延续 | `config_experiments/tpc_deep_entry_strategies` |
| E3 | gate 挡延续 | `config_experiments/tpc_anti_cont_gate_strategies` |
| E4 | E1+E2+E3 | `config_experiments/tpc_deep_pullback_full_strategies` |
| E5 | E4 + initial_r:2, min_stop_pct:0.015 | `config_experiments/tpc_deep_tight_stop_strategies` |

## Phase 3 — Promote 前检查（②）

| 项 | 要求 |
|----|------|
| 语义 | E4 median depth ≥ 0.30，median path_eff ≤ 0.55 |
| 收益 | bull：E4 Mean R / PF ≥ E0 或 CAGR 不劣；trades ≥ ~50% E0 |
| 划界 | 相对 E0 信号变少、entry_filter 拒单增多 |
| 风控 | E5 maxDD 不劣于 E4 |
| recent | 人审，无自动 promote |

## Promote（2026-05-31）

- **上线**：`config/strategies/tpc` + `live/highcap` — prod prefilter + **E2_or entry**（vol≥0.45 OR delta≥0.15）；gate 无 PE deny。
- **不上线**：E1 depth 下界、E3/E4 PE gate + 深回撤 full stack（smoke 均劣于 E2_or）。
- **语法统一**：`docs/strategy/layer_condition_schema_unify_plan.md`（prefilter/regime 待复用 gate `when`/`all_of`）。

## 结论

- **E2_or** 为当前最优；E1/E3 两两组合 + E4 全栈均不 promote。
- 下一 R&D：Phase B prefilter `when` 实现 → regime-conditional depth；6 币 / recent 复验。
