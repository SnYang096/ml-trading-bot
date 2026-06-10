# TPC Regime ADX — 完整实验（Phase 0→1→2→3）

**实验 ID**: 20260610_tpc_regime_adx_phase1  
**假设**: ADX(50) 作为 regime 自适应退出指标优于 ema_1200_position
**状态**: Phase 2 完成（定参），Phase 3 待跑

## 文件清单（自包含）

| 文件 | Phase | 用途 |
|------|:-----:|------|
| `rd_loop_tpc_regime_adx.yaml` | 1 | 可复现的 label scan 配置 |
| `phase1_scan.json` | 1 | Phase 1 IC/plateau scan 结果 |
| `DECISION.md` | 2 | 定参决策（ADX(50)>25 作为 bull） |
| `phase2_grid.yaml` | 3 | variant grid 因果验证（E9/E21/E22） |
| `README.md` | — | 本文件 |

## 复现步骤

### Phase 0: 新增特征列到 FeatureStore

**关键**：加新特征时要指定 `--layer`（已有 layer 名），触发增量模式；不指定则 hash 变→全量重算。

```bash
# 1. 确认特征在策略 requested_features 中（不是 _shared 注册表）
#    编辑 config/strategies/tpc/features.yaml — 加 adx_f

# 2. 找到已有 layer
ls feature_store/features_tpc_120T_*

# 3. 增量构建（~30s，只算 adx_f 列）
mlbot feature-store build --no-docker \
  --config config/strategies/tpc \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --timeframe 120T \
  --start-date 2022-01-01 --end-date 2026-04-30 \
  --root feature_store \
  --layer features_tpc_120T_9506bdec50 \
  --warmup-months 12
```

### Phase 1: IC + label scan

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260610_tpc_regime_adx_phase1/rd_loop_tpc_regime_adx.yaml
```

产物：`results/rd_loop/tpc_regime_adx/quick_scan/`（IC decay + plateau + condition-set）

### Phase 2: 定参 → DECISION.md

从 Phase 1 报告：
- ADX(50) IC(20b)=0.043，分离度=82bps ✅
- 阈值 ADX>25 作为 bull，ADX≤20 作为 bear，20-25 死区回退 neutral

### Phase 3: Grid 回测

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260610_tpc_regime_adx_phase1/phase2_grid.yaml
```

对比：
- **E9_baseline**: trailing always on（基线）
- **E21_ema_018**: ema_1200_position>0.18 → structural exit
- **E22_adx50**: ADX(50)>25 + EMA1200>0.1 → structural exit

## 核心教训

1. **`--layer` 是强制约定**：加特征到 feature store 必须指定已有 layer，否则全量重算
2. **`_shared/features.yaml` 只是注册表**：实际触发计算的是策略自己的 `features.yaml` `requested_features`
3. **regime.yaml 是唯一分类源**：execution.yaml 只引用 bull/bear/neutral 标签，不重复判断
