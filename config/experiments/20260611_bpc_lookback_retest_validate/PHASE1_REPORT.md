# BPC box×depth×lookback — Phase 1 报告（2026-06-04）

> 语义层设计（为何反追高不否定突破语义、回测怎么验收）：[`BPC_SEMANTICS.md`](BPC_SEMANTICS.md)

## 数据

| Parquet | lookback | 路径 |
|---------|----------|------|
| prod L=20 | 20 | `results/train_final/bpc/train_final_20260604_rd_rerun/bpc/features_labeled.parquet` |
| L=120 | 120 | `results/train_final/bpc/bpc_lb120_train_final_20260604_rd_rerun/bpc/` |
| L=240 | 240 | `results/train_final/bpc/bpc_lb240_train_final_20260604_rd_rerun/bpc/` |

- 特征集：`features_scan_phase1.yaml`
- 标的：BTCUSDT, SOLUSDT；240T
- 扫描：`results/rd_loop/bpc_box_pullback_20260611/quick_scan/report.html`

**注意**：scan parquet 上 `ema_1200_position` 全 0（轻量特征集未正确注入 EMA），L120/L240 **bull 子集 plateau 无效**。lookback 对比依赖 **Phase 3 因果 grid**，非 label 子集。

## Label scan 要点（`chop<=0.40`, n=3006）

| 扫描 | 结论 |
|------|------|
| `depth_floor` depth≥0.12 | \|z\|=**5.15**, succ +4pp — **支持下界 0.12** |
| `depth_plateau_L20` depth≤0.35 | \|z\|=3.81, succ **更差** — 过浅上界有害 |
| `anti_chase` box_pos_120≥0.85 | Δpp **−10.3**, \|z\|=**7.36** — **追高区 label 显著更差** |
| `box_scale` box_pos_120≥0.75 | Δpp −7.6, \|z\|=6.4 — 高位 box 不利 |
| `retest_band` depth 带 + box_breakout | **n=0** — L20 上组合不触发 |
| `breakout_up/down` | 负 Δpp — label 不支持作硬过滤 |

## Phase 2 定参（相对手拍参数）

| 参数 | 手拍 | 扫描后 |
|------|------|--------|
| lookback | 120/240 | **保留 B_L120 / B_L240**（因果验证） |
| depth 下界 | 0.12 | **保留 0.12** |
| depth 上界 | 0.55 | **保留 prod 0.55** |
| box 规则 | breakout≥0.5 | **改为 box_pos_120≤0.85**（反追高） |

已更新 [`config/experiments/20260611_bpc_lookback_retest_validate/variants/bpc_lb120_retest_strategies/bpc/archetypes/prefilter.yaml`](../../../config/experiments/20260611_bpc_lookback_retest_validate/variants/bpc_lb120_retest_strategies/bpc/archetypes/prefilter.yaml)。

## Phase 3

- 作废旧 `lookback_retest_20260611` partial runs
- 全量 4×3 grid + trading map
