# TPC Regime ADX — 完整实验

**实验 ID**: 20260610_tpc_regime_adx_phase1  
**假设**: ADX(50) 作为 regime 自适应退出指标优于 ema_1200_position

## 复现步骤

### Phase 0: 生成带 ADX 的特征数据

```bash
PYTHONPATH=src python config/experiments/20260610_tpc_regime_adx_phase1/augment_adx.py \
  --input results/train_final/tpc/train_final_20260604_rd_rerun/tpc/features_labeled.parquet \
  --output results/train_final/tpc/train_final_20260610_adx/tpc/features_labeled.parquet
```

### Phase 1: IC + label scan

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260610_tpc_regime_adx_phase1/rd_loop_tpc_regime_adx.yaml
```

输出：
- `results/rd_loop/tpc_regime_adx/quick_scan/adx_ic_decay.md`
- `results/rd_loop/tpc_regime_adx/quick_scan/adx50_threshold_ge.md`
- `results/rd_loop/tpc_regime_adx/quick_scan/regime_adx_conditions.md`
- `results/rd_loop/tpc_regime_adx/quick_scan/regime_candidates_compare.md`

### Phase 2: 定参 → DECISION.md

### Phase 3: Grid 回测 → `config_experiments/tpc_regime_adaptive_exit/grid.yaml` + E22

## 文件清单

| 文件 | 用途 |
|------|------|
| `augment_adx.py` | Phase 0: 追加 ADX 列到 parquet |
| `rd_loop_tpc_regime_adx.yaml` | Phase 1: 可复现的 label scan 配置 |
| `phase1_scan.json` | Phase 1 快速验证结果（单 BTC 手动跑） |
| `DECISION.md` | Phase 2: 定参决策 |
| `README.md` | 本文件 |
