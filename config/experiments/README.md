# config/experiments — 按实验计划归档

每个**完整实验**一个目录：`config/experiments/<YYYYMMDD>_<strategy>_<topic>/`

| 内容 | 说明 |
|------|------|
| `README.md` | **实验卡片**：物料清单、跑法、`results/` 路径、链到决策文档（不写长表） |
| `DECISION.md` / `*_experiment_*.md` | **决策全文**：假设表、变体/回测结果、promote 结论（原 `docs/decisions/` 已迁入） |
| `rd_loop_*.yaml` | offline 扫描编排（mlbot research + 可选 variant_grid + **`tree_steps`** 树通道） |
| `*_grid.yaml` | event_backtest 变体网格 |

### `tree_steps`（树通道 fast_scalp 等）

| mode | 说明 |
|------|------|
| `prepare-only` | `train_strategy_pipeline.py --prepare-only` → features_labeled.parquet |
| `ic-prune` | `mlbot research ic-prune`（内核 [`src/research/stat_kernels/ic_prune.py`](../../src/research/stat_kernels/ic_prune.py)） |
| `train` | 浅树训练（同 mlbot train final） |
| `tau-scan` | `scripts/research/tree_holdout_tau_rr_scan.py` |
| `filter-predictions` | 按 symbol/split 切 predictions.parquet |

示例：`20260529_fast_scalp/rd_loop_fast_scalp_ic_plateau.yaml`（Phase 1）、`20260530_fast_scalp_alts_majors/rd_loop_fast_scalp_alts_majors.yaml`（Phase 2）。

**不在此目录：** 整棵策略变体树 → 仓库根 [`config_experiments/`](../config_experiments/)（与 `config/strategies` 对照）。

**跨实验校准：** [`_cross/regime_thresholds/`](_cross/regime_thresholds/)（regime τ 季度标定日志）。

**历史链接：** [`docs/decisions/README.md`](../docs/decisions/README.md) 为迁移索引；`docs/strategy/` 内旧 URL 不批量改。

## 跑法

```bash
# 完整 R&D loop（扫描 → 可选 grid → decision doc）
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/<dir>/rd_loop_*.yaml

# 仅因果 backtest
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/<dir>/*_grid.yaml --quiet-signal-logs
```

## 实验索引

| 目录 | 策略 | 主题 |
|------|------|------|
| [`20260526_chop_grid_semantic_proxy/`](20260526_chop_grid_semantic_proxy/) | chop_grid | 语义代理 baseline |
| [`20260527_bpc_layer_validation/`](20260527_bpc_layer_validation/) | bpc | 分层验证 + ABH gate |
| [`20260527_bpc_entry_v2/`](20260527_bpc_entry_v2/) | bpc | entry v2 |
| [`20260527_bpc_regime_ema/`](20260527_bpc_regime_ema/) | bpc | regime EMA grid |
| [`20260527_srb_entry_plateau/`](20260527_srb_entry_plateau/) | srb | entry plateau scan |
| [`20260527_tpc_regime_slope_signed/`](20260527_tpc_regime_slope_signed/) | tpc | regime slope 分符号 |
| [`20260528_me_compression_breakout/`](20260528_me_compression_breakout/) | me | 压缩突破分层 + no_box |
| [`20260528_me_direction/`](20260528_me_direction/) | me | direction 优化 |
| [`20260528_me_entry_filter/`](20260528_me_entry_filter/) | me | entry_filter / orderflow |
| [`20260528_me_gate_anti/`](20260528_me_gate_anti/) | me | gate anti |
| [`20260528_me_prefilter_v4/`](20260528_me_prefilter_v4/) | me | prefilter v4 漏斗 |
| [`20260528_me_prod_holdout/`](20260528_me_prod_holdout/) | me | prod holdout |
| [`20260528_tpc_me_trading_map/`](20260528_tpc_me_trading_map/) | tpc, me | 交易地图 bull/bear |
| [`20260529_fast_scalp/`](20260529_fast_scalp/) | fast_scalp | IC 剪枝 + pooled 训练（Phase 1） |
| [`20260530_fast_scalp_alts_majors/`](20260530_fast_scalp_alts_majors/) | fast_scalp_alts, fast_scalp_majors | alt/majors 拆分部署（Phase 2） |
| [`20260529_short_term_swing_ic_plateau/`](20260529_short_term_swing_ic_plateau/) | short_term_swing | IC plateau |
| [`20260529_tpc_direction_ema_align/`](20260529_tpc_direction_ema_align/) | tpc | direction EMA1200 对齐 + trail |
| [`20260529_tpc_gate_plateau/`](20260529_tpc_gate_plateau/) | tpc | gate plateau |
| [`20260530_tpc_deep_pullback/`](20260530_tpc_deep_pullback/) | tpc | 深回撤 + 吸收（H1–H4） |
| [`_smoke/`](_smoke/) | tpc | CI / 工具 smoke（非正式实验） |

## 新建实验 checklist

1. `mkdir config/experiments/<YYYYMMDD>_<strategy>_<topic>/`
2. 放入 `rd_loop_*.yaml` / `*_grid.yaml`；`variant_grid:` 用**项目根相对路径**
3. 写 `README.md`（假设、物料、跑法、`results/`、结论 TODO）
4. 变体策略树仍在 `config_experiments/<topic>_strategies/`
5. 在本表追加一行索引
