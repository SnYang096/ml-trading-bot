# 多腿联合回测 — chop_grid + trend_scalp 同一账户 timeline (LiveEngine)

**Sizing：** `segment_dd_target: 0.072`, equity 10,000  
**Fee：** live archetypes（不依赖 calibrate_roll）  
**执行：** 1min 粒度，2h 信号，`ChopGridLiveEngine` + `DualAddTrendLiveEngine`

## 跑法

```bash
# 一键联合 timeline 回测（4 canonical segments）
python -m scripts.event_backtest \
  --variant-grid config/experiments/20260613_multileg_sizing_validate/variant_grid.yaml

# 或单段：
python scripts/backtest_multileg_timeline.py \
  --start 2025-12-01 --end 2026-05-31 \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --chop-config config/experiments/20260613_multileg_sizing_validate/variants/chop_prod/meta.yaml \
  --trend-config config/experiments/20260613_multileg_sizing_validate/variants/trend_prod/meta.yaml \
  --constitution-yaml live/highcap/config/constitution/constitution.yaml \
  --equity 10000 \
  --summary-json results/multileg_joint/manual/summary.json
```

## 结构

```
20260613_multileg_sizing_validate/
├── variant_grid.yaml     # 联合回测（engine: multileg_joint → timeline）
├── chop_grid.yaml        # chop 独立（engine: chop_grid）
├── trend_scalp.yaml      # trend 独立（engine: trend_scalp）
├── variants/
│   ├── chop_prod/        # → live/highcap chop_grid archetypes (symlink)
│   └── trend_prod/       # → live/highcap trend_scalp archetypes (symlink)
└── README.md
```

## 产物

```
results/multileg_joint/sizing_072_20260613_timeline/
├── preload.pkl              # 首段特征缓存（后续段复用）
├── timeline/{segment}/      # 每段 summary.json
└── joint/summary.json       # 四段汇总
```
