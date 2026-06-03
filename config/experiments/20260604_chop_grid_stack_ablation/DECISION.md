# chop_grid stack ablation — DECISION

**Status:** E0/E2/E3/E5 已完成（recent_6m_oos · 5 symbols · timeline return_pct）  
**Run date:** 2026-06-04  
**Artifacts:** `results/chop_grid/experiments/stack_ablation_20260604/`

---

## Executive summary

| 层 | 结论 | 动作 |
|---|---|---|
| **prefilter box_pos 0.40–0.60** | **强必要**；去掉后 −14.9% vs +1.16% | **保留 locked** |
| **stable_box block** | 在 prod box_pos 下 **与 block_off 完全相同**（55 trades，+1.16%） | live 语义保留；研究 YAML 可简化讨论 |
| **dense 3L @ 2bps maker** | +4.29% vs baseline +2.75%（同成本口径） | **开实验分支**；四段 validate 后再 promote |

---

## E0 Baseline

| variant | return_pct_timeline | n_trades | n_segments | maxDD | daily_sharpe |
|---------|---------------------|----------|------------|-------|--------------|
| prod_research | **+1.164%** | 55 | 93 | −0.00147 | 3.44 |

路径：`E0_baseline/segment_summary.csv`

---

## E3 prefilter box_pos（核心）

| variant | box_pos | return_pct_timeline | n_trades | n_segments | maxDD | vs prod |
|---------|---------|---------------------|----------|------------|-------|---------|
| **pos_prod** | 0.40–0.60 | **+1.164%** | 55 | 93 | −0.00147 | — |
| pos_wide | 0.35–0.65 | +0.352% | 149 | 168 | −0.00548 | **−0.81 pp** |
| pos_off | 0–1（无） | **−14.916%** | 659 | 228 | −0.155 | **−16.08 pp** |

**Decision:** ☑ **keep prod 0.40–0.60** · ☐ widen · ☐ remove prefilter

**解读：**

- `prefilter.yaml` **不是过度设计** —— 它是当前 stack 里 **ROI 最高的一层**。
- 去掉 box_pos → segment 数 93→228、成交 55→659，但 **forced exit 主导**，在 20bps 成本下巨亏。
- 放宽到 0.35–0.65 仍明显差于 prod（+0.35%），说明 **tight 中间带** 优于宽带。

路径：`E3_box_pos/*/segment_summary.csv`

---

## E2 stable_box block

| variant | return_pct_timeline | n_trades | n_segments | maxDD | vs block_on |
|---------|---------------------|----------|------------|-------|-------------|
| block_on (prod) | **+1.164%** | 55 | 93 | −0.00147 | — |
| block_off | **+1.164%** | 55 | 98 | −0.00147 | **0 pp** |

**Decision:** ☑ **keep block_on for live parity** · ☐ drop for research simplification

**解读：**

- 在 **prod box_pos prefilter 已启用** 时，stable_box block **不改变任何成交**（同一 55 笔）。
- block_off 仅多识别 5 个 segment（98 vs 93），无增量 trade —— 与 20260603 box_prefilter threshold sweep「在 tight box_pos 下无效」一致。
- **不是重复过滤的独立增量**，但 live 引擎仍依赖该语义；保留 YAML，不必再调 threshold。

路径：`E2_stable_box/*/segment_summary.csv`

---

## E5 Grid 密度 @ live_binance_perp

成本假设：maker **2 bps**；regime 强平 taker **5 bps** + slippage。  
Stack：prod regime + box_pos 0.40–0.60 + block_stable_box。

| variant | fee | return_pct_timeline | n_trades | tp_rate | maxDD |
|---------|-----|---------------------|----------|---------|-------|
| baseline 2L / 1.1% | 2 bps | **+2.745%** | 55 | 56.4% | −0.0008 |
| **dense 3L / 0.33%** | 2 bps | **+4.289%** | 650 | 89.8% | −0.0019 |
| baseline 2L / 1.1% | 20 bps | +1.971% | 55 | 56.4% | −0.0009 |
| dense 3L / 0.33% | 20 bps | −3.115% | 650 | 89.8% | −0.0311 |

**Decision:** ☑ **promote dense 3L to experiment branch** · ☐ defer · ☐ keep 2L only

**解读：**

- 在 **实盘成本口径** 下 dense 3L **+1.54 pp** vs baseline（+56% relative）。
- 在 **20bps 研究成本** 下 dense **仍然亏损** —— 解释为何 `min_pct=1.1%` 被 promote（覆盖保守成本）。
- **下一步 E7：** 四段 market_segment validate + 更新 `grid_backtest.costs` 文档化 live 2bps 假设。

---

## E7 — 四段 market_segment validate（done 2026-06-04）

**Artifacts:** `results/chop_grid/experiments/stack_ablation_20260604/E7_four_segment/`  
**Stack 固定：** prod regime + box_pos 0.40–0.60 + block_stable_box

### Timeline return_pct  by segment

| segment | baseline_prod (20bps) | baseline_live (2L@2bps) | **dense_3l_live (3L@2bps)** | dense − baseline_live |
|---------|----------------------:|------------------------:|----------------------------:|----------------------:|
| bear_2022 | +2.15% | +3.55% | **+5.98%** | **+2.43 pp** |
| bull_2023_2024 | +2.91% | +7.23% | **+9.99%** | **+2.76 pp** |
| recent_range_to_bear | +2.40% | +5.14% | **+5.23%** | +0.09 pp |
| recent_6m_oos | +1.16% | +2.30% | **+3.09%** | **+0.79 pp** |

**四段全正；dense 在 live 成本口径下每一段都优于 baseline 2L。**

### 风险（max_drawdown_portfolio）

| segment | baseline_live | dense_3l_live |
|---------|--------------:|--------------:|
| bear_2022 | −0.13% | −0.66% |
| bull_2023_2024 | −0.21% | −0.81% |
| recent_range | −0.11% | −0.65% |
| recent_6m_oos | −0.10% | −0.23% |

dense 收益更高，但 **路径 maxDD 约为 baseline 的 3–4×**（仍 <1% 量级）。按 [`LAYER_PROMOTION_CRITERIA.md`](../LAYER_PROMOTION_CRITERIA.md) 需用户接受 **R↑ vs DD↑** 权衡。

### E7 Decision

- ☑ **dense 3L 进入实验分支**（[`dense_3l_execution.yaml`](dense_3l_execution.yaml)）— 四段 R 全正且均 beat baseline@2bps
- ☐ **暂不替换 prod `execution.yaml`** — maxDD 恶化；建议 live paper / 小资金验证 maker 成交率后再 promote
- ☑ **research 成本文档化**：保守 20bps 与 live 2bps 分开报告（见 [`E7_four_segment.md`](E7_four_segment.md)）

### 复现

```bash
BASE=results/chop_grid/experiments/stack_ablation_20260604/E7_four_segment
CFG=config/strategies/chop_grid/research/calibrate_roll.default.yaml
SYM=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT
COMMON=(--config "$CFG" --symbols "$SYM" --timeframe 2h --execution-timeframe 1min --no-maps)

python scripts/experiment_chop_grid_market_segment.py --out-root "$BASE/dense_3l_live" -- \
  "${COMMON[@]}" --max-levels 3 --grid-atr-mult 0.01 --grid-pct 0.0033 \
  --maker-fee-bps 2 --taker-fee-bps 5 --forced-exit-slippage-bps 5
```

---

## E8 — 真实账户口径 + sizing（2026-06-04）

> 动机：`pnl_per_capital` 把每个 symbol 的资金按 `2×levels` 等分、pooled 再 ÷`n_symbols`，
> 隐含「每 symbol 满仓 + 5 symbol 平摊」。实盘是 **单账户 + 固定 `unit_notional`/格**，
> 时间轴上不可能所有 symbol/层同时开仓。用 [`scripts/sim_chop_grid_account.py`](../../../scripts/sim_chop_grid_account.py)
> 重放 entry/exit 事件（单 10k 账户，固定名义/格）确认真实收益口径。

### 真实账户结果（equity=10k，固定 unit_notional/格）

| 段 | unit=400 ret% | maxDD% | peak gross% | peak/symbol% |
|----|---------------|--------|-------------|--------------|
| recent_6m_oos | 3.71 | -0.27 | 52 | 12 |
| bear_2022 | 7.17 | -0.79 | 36 | 12 |
| bull_2023_2024 | 11.98 | -0.98 | 32 | 12 |
| recent_range_to_bear | 6.27 | -0.78 | 52 | 12 |

- **峰值不是满仓**：单 symbol 峰值同时仅 **3 层**（peak/symbol 12% = 3×400），portfolio 峰值 ~13 legs（52%）。
  证实「时间轴上不会所有 symbol 同时开仓」——但峰值仍需 portfolio gross cap ≥52%。
- **prod 风控修正**：`max_gross_notional_pct` 0.20→**0.60**（覆盖单段峰值 52%，否则实盘 fill 被拒，
  实盘收益远低于回测）；`max_symbol_gross_notional_pct` 0.15 已够（peak/sym 12%）。
- ⚠️ `ALL combined 104%` 是 4 段不同年份 trades 拼接的假象（时间戳重叠），真实单段峰值 ≤52%。

### 1% maxDD sizing

| 锚定窗口 | unit_notional | ret% | peak gross% |
|----------|---------------|------|-------------|
| OOS（当前 chop regime） | ~1465 | ~13.6 | ~190 |
| bull_2023_2024（最差段，all-weather） | ~410 | ~12.3 | ~33 |

- **prod `unit_notional=400` ≈ all-weather 1% DD 档**（最差 bull 段 0.98% DD）。
- **aggressive 档** = regime-aware unit≈1465 → 见 [`aggressive_sizing.yaml`](aggressive_sizing.yaml)
  （需 gross cap 2.0 / per-sym 0.50，max_gross_leverage 3.0 兜底；仅 chop regime 确认期使用）。

### 复现

```bash
python scripts/sim_chop_grid_account.py \
  --root results/chop_grid/experiments/stack_ablation_20260604/E7_four_segment/dense_3l_live \
  --equity 10000 --dd-target 1.0
```

---

## 待跑

| ID | 主题 | 状态 |
|----|------|------|
| E1 | Regime chop 0.50/0.32 vs 0.52/0.33 | pending（边际，低优先） |
| E4 | box_prefilter 阈值 | skip |
| E6 | Replenish | 见 replenish_ablation |
| E7 | 四段 joint | **done** |

---

## Promote checklist

- [x] E3 → 保留 `prefilter.yaml` box_pos 0.40–0.60
- [x] E2 → 保留 live stable_box block
- [x] E7 → dense 3L 四段 validate @ 2bps
- [x] E8 → 真实账户口径确认 + prod gross cap 0.20→0.60 修正 + aggressive_sizing.yaml
- [x] dense 3L → 已写入 `live/highcap` archetypes（execution/regime/prefilter）+ unit_notional 400
- [ ] live paper / 小资金验证 maker 成交率 + 真实 gross 峰值 vs cap
- [ ] 更新 `grid_backtest.costs` 文档：conservative vs live_binance_perp
