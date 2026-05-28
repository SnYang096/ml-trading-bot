# config/experiments — 按策略分子目录

避免根目录堆满 `rd_loop_*` / `*_grid.yaml`，新实验一律放进对应 slug 子目录。

| 目录 | 策略 / 用途 |
|------|-------------|
| `bpc/` | BPC layer validation、regime EMA、entry v2、ABH gate |
| `tpc/` | TPC regime slope、smoke grid |
| `me/` | CompressionBreakout（slug `me`）：regime 去 box、direction、prefilter |
| `srb/` | SRB entry-plateau rd_loop |
| `chop_grid/` | C 层 chop_grid 语义代理 grid |
| `smoke/` | 跨策略 CI / 文档 smoke |

**跑法示例**

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/me/rd_loop_me_compression_breakout.yaml

PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/me/me_regime_no_box_grid.yaml --quiet-signal-logs
```

**config_experiments/**（仓库根下）放 **整棵 strategies 实验树**（`{topic}_strategies/`），与生产 `config/strategies` 对照；新 ME 树建议 `config_experiments/me/<topic>_strategies/`。
