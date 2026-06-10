# SRB SR时间周期对比实验：大周期 vs 小周期

| 字段 | 值 |
|------|-----|
| 目录 | `20260609_srb_sr_tf_compare/` |
| 日期 | 2026-06-09 |
| 策略 | srb |
| 目标 | 验证 SRB 是否应该用大周期 SR（L3 240-bar），而非小周期（L1 20-bar） |

## 假设

**H0**：SRB使用L3大周期SR（240-bar wide_sr）的prefilter效果不优于L1小周期SR（20-bar swing）。

**H1**：L3大周期SR能过滤更多假突破，提升meanR和胜率，代价是笔数减少。

**核心疑问**：
1. 当前prefilter用 `srb_l3_breakout_age_decay`（基于240-bar窗口）过滤，是否在过度约束？
2. 小周期SR（L1 20-bar）的突破信号是否已经足够分辨真假？
3. 如果只用L1+频谱，能否保留更多trades而不恶化meanR？

## 变量空间

| 变量 | 小周期 | 中周期 | 大周期（当前） | 更大周期 |
|------|--------|--------|----------------|----------|
| SR窗口 | L1: 20 bar | L2: 160 bar (POC) | L3: 240 bar | L4: 480 bar |
| anchor_shift | 1 | 8 | 12 | 24 |
| 对应特征组 | `l1_sr_upper/lower_px` | `sr_strength_max` + `dist_to_nearest_sr` | `wide_sr_swing_f` | 需构建 |
| prefilter规则 | sr_strength_max + spectrum | sr_strength_max + spectrum | sr_strength_max + spectrum + srb_l3_breakout_age_decay | sr_strength_max + spectrum + srb_l4_breakout_age_decay |

## 物料

- `rd_loop_srb_sr_tf_compare.yaml` — Phase 1扫描
- `srb_sr_tf_grid.yaml` — Phase 3事件回测
- `run_trading_maps.sh` — Phase 4语义检查

## 跑法

### Phase 1: IC扫描
```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260609_srb_sr_tf_compare/rd_loop_srb_sr_tf_compare.yaml
```

### Phase 3: 事件回测
```bash
python -m scripts.event_backtest \
  --variant-grid config/experiments/20260609_srb_sr_tf_compare/srb_sr_tf_grid.yaml
```

### Phase 4: Trading Maps
```bash
bash config/experiments/20260609_srb_sr_tf_compare/run_trading_maps.sh
```

## 判断标准（LAYER_PROMOTION_CRITERIA 三条杠）

1. ✅ IC lift vs baseline（Phase 1扫描报告）
2. ✅ Segment回测 Total R ≥ baseline（Phase 3 event_backtest）
3. ✅ Trading map语义正确（Phase 4）

## 结论

TODO（跑完后填写 promote / reject 与要点）。

## 关联

- 基线策略：`config/strategies/srb/`
- 参考实验：`20260527_srb_entry_plateau/`