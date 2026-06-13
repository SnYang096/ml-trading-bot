# 多腿联合回测 — chop_grid + trend_scalp 同一账户

**Sizing：** `segment_dd_target: 0.072`, equity 10,000  
**Fee：** live archetypes（不依赖 calibrate_roll）  
**执行：** 1min 粒度，2h 信号

## 跑法

```bash
# 一键联合回测（chop + trend → 同账户模拟）
python -m scripts.event_backtest \
  --variant-grid config/experiments/20260613_multileg_sizing_validate/variant_grid.yaml

# 或单跑子策略：
python -m scripts.event_backtest \
  --variant-grid config/experiments/20260613_multileg_sizing_validate/chop_grid.yaml

python -m scripts.event_backtest \
  --variant-grid config/experiments/20260613_multileg_sizing_validate/trend_scalp.yaml
```

## 结构

```
20260613_multileg_sizing_validate/
├── variant_grid.yaml     # 联合回测（engine: multileg_joint）
├── chop_grid.yaml        # chop 独立（engine: chop_grid）
├── trend_scalp.yaml      # trend 独立（engine: trend_scalp）
├── variants/
│   ├── chop_prod/        # → live/highcap chop_grid archetypes (symlink)
│   └── trend_prod/       # → live/highcap trend_scalp archetypes (symlink)
└── README.md
```

## 产物

```
results/multileg_joint/sizing_072_20260613/
├── chop_grid/{segment}/     # chop 独立回测
├── trend_scalp/{segment}/   # trend 独立回测
└── joint/                   # sim_multileg_account.py 联合报告
```
