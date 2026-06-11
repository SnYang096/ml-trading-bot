# TPC entry semantic — 全窗 trading map 对比（2026-06-04）

与 `20260604_tpc_entry_semantic_validate`（三阶段 segment grid）互补：本实验只跑 **2022-01-01 → 2026-04-01** 全窗 **trading map**，便于肉眼对比 SOL 红框等。

| 变体 | strategies_root | 说明 |
|------|-----------------|------|
| E0_prod | `config/strategies` | prod 基线 |
| E2a_or_anti_chase | `config/experiments/20260604_tpc_entry_semantic_trading_maps/variants/tpc_entry_e2a_or_anti_chase_strategies` | prod prefilter + `(vol OR delta) AND` bars_since |
| E1e2_band_or_anti | `config/experiments/20260604_tpc_entry_semantic_trading_maps/variants/tpc_entry_e1e2_band_or_anti_strategies` | `0.15≤depth≤0.85` + 同 E2a entry |
| S50_depth_gt50 | `config/experiments/20260604_tpc_entry_semantic_trading_maps/variants/tpc_semantic_depth_gt50_strategies` | `depth>0.5` 深回踩-only（对照 BPC） |

树由 `scripts/research/prepare_tpc_entry_semantic_snapshots.py` 生成（含 E2a / E1e2）。

## 跑法

```bash
export PYTHONPATH=src:scripts
bash config/experiments/20260604_tpc_entry_semantic_trading_maps/run_trading_maps.sh
```

产物：`results/tpc/maps/compare_entry_semantic_20260604/<variant>/trading_map_tpc_event.html`

## 状态

见 `results/tpc/maps/compare_entry_semantic_20260604/logs/`（`nohup.log` / `run_all.log`）。
