# docs/decisions — 已迁入 `config/experiments/`

决策文档不再集中在本目录，而是与**实验物料**放在同一实验文件夹，便于回溯。

## 文件分工（每个实验目录）

| 文件 | 用途 |
|------|------|
| `README.md` | **实验卡片**：一句话假设、yaml 物料清单、跑法命令、`results/` 路径、结论 TODO |
| `DECISION.md` 或 `*_experiment_*.md` | **决策全文**：H1–H4 假设表、变体表、双段回测结果、promote/reject 结论（跑完后填） |
| `rd_loop_*.yaml` / `*_grid.yaml` | 机器可读物料 |

**不要**在 README 里复制 DECISION 的长表；README 只链到本地决策文件。

## 迁移对照表

| 原路径 | 新路径 |
|--------|--------|
| `tpc_deep_pullback_hypothesis_2026.md` | [`config/experiments/20260530_tpc_deep_pullback/DECISION.md`](../config/experiments/20260530_tpc_deep_pullback/DECISION.md) |
| `tpc_regime_slope_signed_scan_20260527.md` | `config/experiments/20260527_tpc_regime_slope_signed/` |
| `tpc_slope_signed_calendar_20260527.md` | 同上 |
| `bpc_layer_validation_20260527.md` 等 BPC | `config/experiments/20260527_bpc_layer_validation/` |
| `bpc_entry_v2_experiment_20260527.md` | `config/experiments/20260527_bpc_entry_v2/` |
| `bpc_regime_ema_experiment_20260527.md` | `config/experiments/20260527_bpc_regime_ema/` |
| `chop_grid_proxy_validation_20260526.md` | `config/experiments/20260526_chop_grid_semantic_proxy/` |
| `tpc_validation_smoke_20260526.md` | `config/experiments/_smoke/` |
| `tpc_gate_vol_ABH_experiment_20260526.md` | `config/experiments/_smoke/` |
| `regime_thresholds/*.md` | `config/experiments/_cross/regime_thresholds/` |

`docs/strategy/` 中的历史链接仍指向旧路径，仅作历史记录；新实验请只写 `config/experiments/<实验目录>/`。

`scripts/_new_decision_doc.py` / `rd_loop` 的 `decision_doc.out` 建议写到对应实验目录（见各实验 `rd_loop_*.yaml`）。
