# chop_grid 入场栈 — 分层 ablation 计划

**日期：** 2026-06-04  
**动机：** 当前 stack 为「chop 高 + box_pos 中间带 + stable box 禁入 + spacing/min_pct」，怀疑部分层重复或过度设计；dense 3L 在 ~2bps maker 下可能显著优于 baseline，需与层 ablation 分开验证。

**Canonical 窗口：** `recent_6m_oos`（2025-10-01 .. 2026-03-31）为主；promote 前补跑 `config/market_segment.yaml` 四段（见 [`LAYER_PROMOTION_CRITERIA.md`](../LAYER_PROMOTION_CRITERIA.md)）。

**成本口径（两套，分开报告）：**

| 口径 | maker/taker | 用途 |
|------|-------------|------|
| `research_conservative` | 20 bps（当前 `grid_backtest.costs`） | 与历史 OOS 文档可比 |
| `live_binance_perp` | maker 2 bps；regime 强平 taker 2 bps + slippage 5 bps | dense grid / 实盘决策 |

---

## 当前 stack（prod research archetype）

```
signal 2h  →  exec 1min (live-aligned)
│
├─ regime (extensions.multileg)
│    entry_feature: bpc_semantic_chop
│    entry_min / exit_below  →  hysteresis 开/关 grid
│    exclude_box_prefilter: false  →  stable box bar 上 block 新开仓
│    box_prefilter: {stability, width, touches}  →  定义「stable box」
│
├─ prefilter (archetypes/prefilter.yaml)
│    box_pos_60 ∈ [0.40, 0.60]  →  价格在 box 垂直位置中间带
│
└─ execution
     spacing atr_mult=1.18, min_pct=0.011, max_levels_per_side=2
```

### 两层「box」不要混

| 概念 | 配置位置 | 语义 |
|------|----------|------|
| **stable box** | `regime.extensions.multileg.box_prefilter` + `exclude_box_prefilter=false` | box **结构**够稳（stability/width/touches）→ **禁止**开新 grid |
| **box_pos 中间带** | `prefilter.rules` | 价格在 box **几何位置** 40–60%（不贴顶/底）→ **允许**开仓 |

二者正交，但 **20260603 box_prefilter sweep** 在 tight box_pos 下 threshold 几乎无增量（~0.06 pp）—— stable box **block** 是否仍有独立价值，需 **E2** 单独 ablate。

### 已有证据（非正式 promote）

| 层 | 来源 | 结论（待 ablation 确认） |
|----|------|--------------------------|
| box_pos 0.40–0.60 | `oos_phase2_20260603` | 与 spacing 同为 **主杠杆**；无 prefilter 未系统 ablate |
| regime 0.52/0.33 | 同上 | 相对 0.50/0.32 **边际** |
| stable_box block | `box_prefilter_sweep` + live 对齐 | threshold 微调无效；**on vs off** 未单独 ablate |
| dense 3L | `levels_density_20260603` | 20bps **亏**；2bps **+4.5% vs +2.9%** baseline |

---

## 实验顺序（一次只动一层）

原则：固定 prod 其余层，单变量 ablation；通过 [`LAYER_PROMOTION_CRITERIA.md`](../LAYER_PROMOTION_CRITERIA.md) 三条杠再 promote。

| ID | 主题 | 假设 | 变体 | 脚本 / 命令 |
|----|------|------|------|-------------|
| **E0** | Baseline 锁定 | 复现 tuned prod stack | `prod_research` | `experiment_chop_grid_market_segment.py` + prod yaml |
| **E1** | Regime chop | hysteresis 必要 | `chop_loose` 0.50/0.32 · `chop_tight` 0.55/0.35 · `chop_off`（仅 exit） | `sweep_chop_oos_layers.py` regime 段 或 backtest CLI |
| **E2** | stable_box block | live 语义必要；与 box_pos 是否重复 | `block_on`（prod）· `block_off` | `--block-stable-box` / `--no-block-stable-box` |
| **E3** | **prefilter box_pos** | 中间带避免贴边开仓，提升 R | `pos_off` · `pos_wide` 0.35–0.65 · `pos_prod` 0.40–0.60 | `--box-pos-min/max`；off = 0.0–1.0 |
| **E4** | box_prefilter 阈值 | 在 E3 prod 下仍无效则可删 YAML 噪音 | stability/width/touches 单参 sweep | `sweep_chop_box_prefilter.py` |
| **E5** | **Grid 密度** | 2bps 下 dense 3L 优于 2L | `2L_1.1%` · `3L_0.33%` · `4L_0.25%` | `sweep_chop_grid_levels.py` + `--maker-fee-bps 2` |
| **E6** | Replenish | TP 后补挂 1 次最优 | unlimited · off · live(1) | 见 [`20260603_chop_grid_replenish_ablation/`](../20260603_chop_grid_replenish_ablation/) |
| **E7** | Joint promote | E1–E6 赢家组合 | 待定 | 四段 validate |

**E3 是回答「prefilter 有没有用」的核心实验。**  
**E2×E3 交叉（可选 E3b）：** `block_off + pos_prod` vs `block_on + pos_off` — 若二者等价，说明过度设计。

---

## E0 — Baseline 复现

```bash
OUT=results/chop_grid/experiments/stack_ablation_20260604/E0_baseline
CFG=config/strategies/chop_grid/research/calibrate_roll.default.yaml
SYM=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT
FWD=(--config "$CFG" --symbols "$SYM" --timeframe 2h --execution-timeframe 1min
     --initial-capital 10000 --no-maps)

python scripts/experiment_chop_grid_market_segment.py \
  --out-root "$OUT/recent_6m_oos" --segments recent_6m_oos -- "${FWD[@]}"
```

记录：`return_pct_timeline`、`n_trades`、`max_drawdown_portfolio`、segment 数。

---

## E2 — stable_box block ablation

```bash
BASE=results/chop_grid/experiments/stack_ablation_20260604/E2_stable_box

# prod: block stable box (exclude_box_prefilter=false → block_stable_box=true)
python scripts/experiment_chop_grid_market_segment.py \
  --out-root "$BASE/block_on" --segments recent_6m_oos -- \
  "${FWD[@]}" --block-stable-box

# 对照：不在 stable box bar 上额外 block（研究-only，非 live）
python scripts/experiment_chop_grid_market_segment.py \
  --out-root "$BASE/block_off" --segments recent_6m_oos -- \
  "${FWD[@]}" --no-block-stable-box
```

**判据：** `block_on` 若 total R ≤ `block_off` 且 maxDD 未改善 → 考虑 live 语义是否应改；若 R 明显更高 → 保留。

---

## E3 — prefilter box_pos ablation（核心）

```bash
BASE=results/chop_grid/experiments/stack_ablation_20260604/E3_box_pos

# A: 无 prefilter（全范围 0–1）
python scripts/experiment_chop_grid_market_segment.py \
  --out-root "$BASE/pos_off" --segments recent_6m_oos -- \
  "${FWD[@]}" --box-pos-min 0.0 --box-pos-max 1.0

# B: prod 0.40–0.60
python scripts/experiment_chop_grid_market_segment.py \
  --out-root "$BASE/pos_prod" --segments recent_6m_oos -- \
  "${FWD[@]}" --box-pos-min 0.40 --box-pos-max 0.60

# C: 宽一点 0.35–0.65（历史 multileg 候选）
python scripts/experiment_chop_grid_market_segment.py \
  --out-root "$BASE/pos_wide" --segments recent_6m_oos -- \
  "${FWD[@]}" --box-pos-min 0.35 --box-pos-max 0.65
```

**判据：**

- `pos_off` ≈ `pos_prod` → prefilter **可删**（过度设计）
- `pos_prod` 明显优于 off/wide → **保留** locked rules
- 额外看：segment 级 `entry` 是否在 box 边缘更多 forced exit

---

## E5 — Grid 密度（live 2bps）

在 **E3 赢家** 固定后跑（默认先固定 prod stack）：

```bash
python scripts/sweep_chop_grid_levels.py \
  --start 2025-10-01 --end 2026-03-31 \
  --out-dir results/chop_grid/experiments/stack_ablation_20260604/E5_density \
  --config-yaml config/strategies/chop_grid/research/calibrate_roll.default.yaml
```

脚本需支持 `--maker-fee-bps`（见 README）；对比 `dense_3L` vs `baseline_prod` 在 **live_binance_perp** 口径。

**判据：** dense 3L total R ↑、maxDD 不恶化、regime_exit 率可接受 → 实验分支写入 `execution.yaml`（不直接改 prod 直至 E7 四段）。

---

## 输出与 DECISION

每个 Ex 完成后在 [`DECISION.md`](DECISION.md) 填一行；全部完成后：

1. 列出 **保留 / 删除 / 合并** 的层  
2. 更新 `config/strategies/chop_grid/archetypes/*.yaml`（仅 E7 通过后）  
3. 链到 `20260603_chop_grid_oos_tune` 与 `levels_density_20260603` 本地 results

---

## 建议执行顺序

```
E0 → E3 (prefilter) → E2 (stable box) → E1 (regime 微调) → E4 → E5 (density @2bps) → E6 → E7
```

**优先 E3**：直接回答「prefilter.yaml 有没有用」。  
**E2 紧跟 E3**：判断是否与 box_pos 重复。  
**E5 与层 ablation 独立**：成本假设不同，但在 E3 结论后再跑 dense，避免 confound。
