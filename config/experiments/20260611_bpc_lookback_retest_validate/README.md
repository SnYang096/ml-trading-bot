# BPC lookback + box-retest 验证（2026-06-11）

| 字段 | 值 |
|------|-----|
| 策略 | bpc |
| Grid | [`bpc_lookback_retest_grid.yaml`](bpc_lookback_retest_grid.yaml) |
| 变体树 | `config_experiments/bpc_lb*_strategies/`（静态 YAML，无 prepare 脚本） |

## 背景

Trading map 复盘：prod BPC 常在 **突破延续段/尖顶追高**，而非「压缩区突破 → 回测 → 再延续」。根因：`lookback_breakout=20`（≈1.7d）+ `depth<=0.55`（浅位/贴顶易过）+ 无 box 硬规则。

背景文档：[B系统入场语义与执行层周期错配_CN.md](../../docs/strategy/B系统入场语义与执行层周期错配_CN.md) §2.2

## 变体

| ID | lookback | prefilter 额外 |
|----|----------|----------------|
| **B0_prod** | 20 | prod 三锚点 + vol_compression |
| **B_L120** | 120（~10d） | 同上，仅拉长 soft_phase / bars_since |
| **B_L240** | 240（~20d） | 同上 |
| **B_L120_retest** | 120 | + `box_breakout_up>=0.5` + `depth>=0.12`（与 `<=0.55` 成带，反追高） |

实验树覆盖：`bpc_soft_phase_f.lookback_breakout` / `vol_ma_window` / `node_cache_version` + `bars_since_extreme_f.lookback`。

## 跑法

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260611_bpc_lookback_retest_validate/bpc_lookback_retest_grid.yaml \
  --quiet-signal-logs
```

全窗 trading map（胜出者，BTC/SOL）：

```bash
bash config/experiments/20260611_bpc_lookback_retest_validate/run_trading_maps.sh
```

## 结果

- 分段：`results/bpc/experiments/lookback_retest_20260611/<variant>/<segment>/`
- 地图：`results/bpc/maps/lookback_retest_20260611/`

## 决策

见 [`DECISION.md`](DECISION.md)。重点看 **bull_2023_2024** trading map 入场是否从「尖顶追高」移到压缩突破后的回测区。
