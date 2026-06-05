# TPC entry semantic validate — 决策记录

**实验 ID:** `tpc_entry_semantic_validate_20260604`  
**状态:** canonical 三阶段 **已完成**（21/21）；全窗 highcap **未跑**（见下文）  
**代码基线:** `8e977fd8`（direction band 对齐 + entry filter `direction: long|short` + E2 双边 anti-chase）

## 跑批说明

- 后台轮询任务被中断，**不影响** grid 主进程；segment_matrix 已全部写完 `capital_report.json`。
- `variant_grid` 在存在 `segment_matrix` 时**不会**合并顶层 `runs:`，因此 `full/E0_prod`、`full/S50`、`full/S51` 未执行。若需 SOL 全窗对照，请单独跑或修 grid 合并逻辑后再补 3 窗。

## 结果表（canonical 三阶段，6 币 highcap）

| 变体 | bear_2022 R | bull_2023_2024 R | recent R | **sum R** | **worst maxDD** | trades | vs E0 sum R |
|------|-------------|------------------|----------|-----------|-----------------|--------|-------------|
| **E0_prod** | 4.47 | 18.82 | 13.44 | **36.73** | -12.1% | 208 | — |
| **E2_anti_chase** | 9.08 | 25.08 | 18.89 | **53.05** | **-6.6%** | 130 | **+16.3** |
| E1_depth_ge15 | 13.44 | 19.51 | 13.83 | 46.79 | -9.6% | 194 | +10.1 |
| S50_depth_gt50 | 6.86 | 9.49 | -2.92 | 13.44 | -5.2% | 59 | -23.3 |
| S51_depth_gt50_ema_near | -1.95 | 7.77 | -4.43 | 1.38 | -7.2% | 477 | -35.4 |
| E3_gate_pe | 5.18 | -1.23 | -4.18 | -0.23 | -3.2% | 34 | -37.0 |
| E4_turbo_exec | 10.19 | 89.32 | 45.41 | 144.92 | -9.9% | 135 | +108.2† |

† E4 仅替换 execution（turbo 20260424 快照），**与入场语义实验不可直接比**；见执行层结论。

## Promote 检查（LAYER_PROMOTION_CRITERIA）

| 变体 | 总 R ↑（三阶段 sum） | maxDD 不恶化 | 可解释 | 判决 |
|------|---------------------|--------------|--------|------|
| **E2** | ✅ +16.3 R vs E0 | ✅ -6.6% vs -12.1% | ✅ 双边 `bars_since_local_*` 防贴极值追 | **推荐 promote → entry_filters** |
| E1 | ✅ +10.1 R | ✅ -9.6% vs -12.1% | ✅ depth≥0.15 挡 depth≈0 | **可选 promote → prefilter**；与 E2 未做组合 ablation |
| S50 | ❌ | ✅（但更浅因几乎无单） | ✅ | **拒绝** — 过度过滤 |
| S51 | ❌ | ≈ | ✅ | **拒绝** — bear/recent 负 R、笔数膨胀 |
| E3 | ❌ | ✅ | ✅ | **拒绝** — bull/recent 崩 |
| E4 | — | — | — | **执行层另议**，不入本次 entry 晋升 |

- [x] canonical 三阶段：**E2** 总 R ↑ 且 maxDD 改善
- [ ] 全窗 highcap：未跑（grid 未展开 `runs:`）
- [x] 逻辑可解释（机制 veto，非单段阈值挖矿）

## 结论与下一步

### 推荐写入生产（entry 层）

**E2 — `tpc_entry_anti_chase_strategies` 快照逻辑：**

- `combination_mode: and`（在现有 vol/delta OR 分支之上叠加）
- `direction: long` + `bars_since_local_high >= 0.10`
- `direction: short` + `bars_since_local_low >= 0.10`

三阶段均优于 E0，笔数更少、回撤更浅，符合「挡 SOL 贴极值追」假设。

### 可选（prefilter 层，单独 PR）

**E1 — `tpc_pullback_depth >= 0.15`**：三阶段 sum R 亦高于 E0，与 E2 机制互补；promote 前建议跑 **E1+E2 组合** 一格，避免重复过滤或过度稀疏。

### 不 promote

- **S50 / S51**：深回踩 + EMA 放宽未带来稳健提升；S51 交易数异常高。
- **E3**：高 path_efficiency deny 在 bull/recent 伤害过大。
- **E4**：仅证明 turbo execution 在同入场下 R 弹性大；若动 execution 需单独 execution ablation + DD 预算评审，**不与本次 entry 晋升混提**。

### 补跑（可选）

```bash
# 全窗 E0 / S50 / S51（当前 grid 未自动合并 runs，需手写三条或修 variant_grid）
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --strategy tpc --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --start-date 2022-01-01 --end-date 2026-04-01 \
  --strategies-root config/strategies \
  --capital-report results/tpc/experiments/entry_semantic_validate_20260604/full/E0_prod \
  --trades-csv results/tpc/experiments/entry_semantic_validate_20260604/full/E0_prod/event_trades_tpc.csv \
  --data-path data/parquet_data --no-kill-switch --quiet-signal-logs
```

（S50/S51 将 `--strategies-root` 换成对应 `config_experiments/*`。）

---

**产物目录:** `results/tpc/experiments/entry_semantic_validate_20260604/`  
**汇总脚本:** `scripts/research/summarize_entry_semantic_grid.py`
