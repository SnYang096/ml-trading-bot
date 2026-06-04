# TPC 入场语义 + 执行对照（2026-06-04）

| 字段 | 值 |
|------|-----|
| 策略 | tpc |
| Grid | [`tpc_entry_semantic_grid.yaml`](tpc_entry_semantic_grid.yaml) |
| 变体树 | `scripts/research/prepare_tpc_entry_semantic_snapshots.py` → `config_experiments/tpc_*_strategies/` |

## 变体

| ID | 改动 |
|----|------|
| **E0_prod** | 现网 `config/strategies`（G1 gate + entry OR） |
| **S50_depth_gt50** | prefilter 增加 **`tpc_pullback_depth > 0.5`**（高语义深回踩） |
| **S51_depth_gt50_ema_near** | S50 + **`ema_1200_position >= -0.10`**；regime 死区缩至 \|pos\|>0.03；direction **inner_abs=-0.10**（允许略低于 EMA1200 做多） |
| E1_depth_ge15 | prefilter 增加 `tpc_pullback_depth >= 0.15` |
| E2_anti_chase | entry `combination_mode: and` + `bars_since_local_high >= 0.10` |
| E3_gate_pe | gate 增加 `path_efficiency_pct > 0.15` deny |
| E4_turbo_exec | execution 换 `turbo/20260424_191639` 快照 |

## 跑法

```bash
# 1) 生成变体树（首次或改 patch 后）
PYTHONPATH=src:scripts python scripts/research/prepare_tpc_entry_semantic_snapshots.py

# 2) canonical 三阶段 × 6 变体 + 全窗 E0/S50
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260604_tpc_entry_semantic_validate/tpc_entry_semantic_grid.yaml \
  --quiet-signal-logs
```

## 结果

- 分段：`results/tpc/experiments/entry_semantic_validate_20260604/<variant>/<segment>/`
- 全窗：`results/tpc/experiments/entry_semantic_validate_20260604/full/`

## 决策

见 [`DECISION.md`](DECISION.md)（跑完后填 Total R / maxDD / SOL 追高笔数）。

## 方法论笔记

讨论沉淀：[`docs/strategy/TPC语义约束与树标签对齐_CN.md`](../../docs/strategy/TPC语义约束与树标签对齐_CN.md)（语义 vs 统计、树 label 对齐、**regime 代理** 详解）。
