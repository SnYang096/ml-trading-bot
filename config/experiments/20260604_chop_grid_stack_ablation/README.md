# chop_grid — 入场栈分层 ablation + dense grid（2026-06-04）

**问题：** chop 高 + box_pos 中间带 + stable box 禁入 + 宽 spacing 是否过度设计？`prefilter.yaml` 的 `box_pos_60` 规则是否真有增量？在 Binance ~2bps maker 下 dense 3L 是否应 promote？

**计划全文：** [`PLAN.md`](PLAN.md)  
**结论（跑完后填）：** [`DECISION.md`](DECISION.md)

## 实验一览

| ID | 主题 | 状态 |
|----|------|------|
| E0 | Baseline 复现（prod stack） | **done** +1.16% |
| E1 | Regime chop hysteresis | pending |
| E2 | stable_box block on/off | **done** — 与 prod 无差异 |
| **E3** | **prefilter box_pos** | **done** — prod 必要 |
| E4 | box_prefilter 阈值 | skip（prior sweep） |
| E5 | Grid 密度 2L vs dense 3L @ 2bps | **done** — dense +1.54pp |
| E6 | Replenish ablation | 见 replenish_ablation |
| E7 | 四段 joint promote | **done** — dense 四段全正，beat baseline@2bps |

## 快速开始（E0 + E3 smoke）

```bash
CFG=config/strategies/chop_grid/research/calibrate_roll.default.yaml
SYM=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT
FWD=(--config "$CFG" --symbols "$SYM" --timeframe 2h --execution-timeframe 1min
     --initial-capital 10000 --no-maps)
BASE=results/chop_grid/experiments/stack_ablation_20260604

# E0
python scripts/experiment_chop_grid_market_segment.py \
  --out-root "$BASE/E0_baseline" --segments recent_6m_oos -- "${FWD[@]}"

# E3 — prefilter 有没有用
python scripts/experiment_chop_grid_market_segment.py \
  --out-root "$BASE/E3_box_pos/pos_off" --segments recent_6m_oos -- \
  "${FWD[@]}" --box-pos-min 0.0 --box-pos-max 1.0

python scripts/experiment_chop_grid_market_segment.py \
  --out-root "$BASE/E3_box_pos/pos_prod" --segments recent_6m_oos -- \
  "${FWD[@]}" --box-pos-min 0.40 --box-pos-max 0.60
```

## 相关实验

| 目录 | 内容 |
|------|------|
| [`../20260603_chop_grid_oos_tune/`](../20260603_chop_grid_oos_tune/) | spacing / regime / box_pos promote 来源 |
| [`../20260603_chop_grid_exec_align/`](../20260603_chop_grid_exec_align/) | 1min exec 对齐 |
| [`../20260603_chop_grid_replenish_ablation/`](../20260603_chop_grid_replenish_ablation/) | replenish 0/1/unlimited |
| `results/chop_grid/experiments/levels_density_20260603/` | dense grid + fee 敏感性（gitignored） |

## Promote 准则

[`../LAYER_PROMOTION_CRITERIA.md`](../LAYER_PROMOTION_CRITERIA.md) — 四段 total R ↑、maxDD 不恶化、逻辑可解释。
