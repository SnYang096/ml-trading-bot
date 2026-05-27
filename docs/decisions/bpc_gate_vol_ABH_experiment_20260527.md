# BPC Gate 实验：A / B / H / no_breakout（vol gates + breakout 锚）

- **日期**: 2026-05-27
- **策略**: BPC（6 symbols）
- **工具**: `event_backtest --variant-grid`
- **Grid**: `config/experiments/bpc_abh_variant_grid.yaml`
- **索引**: `results/bpc/experiments/EXPERIMENT_INDEX.json`
- **决策**: **不 promote** — 维持生产 `config/strategies`（变体 A）

## 1. 变体

| ID | strategies_root | 改动 |
|---|---|---|
| **A** | `config/strategies` | 生产 baseline |
| **B** | `bpc_B_vol_off_strategies` | vol_persistence + vol_leverage_asymmetry **disabled** |
| **H** | `bpc_H_bull_vol_strategies` | vol deny 仅 `ema_1200_position > 0.10` |
| **no_breakout** | `bpc_no_breakout_strategies` | 去掉 `bpc_recent_breakout_strength>=0.4` 锚 |

## 2. Event backtest

### 2.1 2024-01-01 → 2025-01-01（calendar bull）

| 变体 | trades | totR | ret% | maxDD% |
|---|---:|---:|---:|---:|
| A | 27 | +16.85 | +7.02% | -5.93% |
| **B** | 29 | **+17.56** | **+7.73%** | -5.93% |
| H | 28 | +16.81 | +6.98% | -5.93% |
| no_breakout | 28 | +17.47 | +6.75% | -5.93% |

B / no_breakout 略优于 A（+0.7R / +0.6R），**笔数 ~27–29，统计力弱**；maxDD 四者相同量级。

### 2.2 2025-04-01 → 2026-04-01（recent）

| 变体 | trades | totR | ret% | maxDD% |
|---|---:|---:|---:|---:|
| **A** | 25 | **-1.22** | +2.74% | -4.96% |
| B | 17 | -4.18 | +2.15% | -4.96% |
| H | 14 | -5.59 | +0.36% | -4.96% |
| **no_breakout** | 25 | **-1.22** | +2.74% | -4.96% |

**近期窗全体 totR 为负**；B/H 更差（更少 trades、更负 totR）。**A 与 no_breakout 完全相同**（同 trades/totR）→ breakout 锚在近期未改变成交路径。

## 3. 结论（相对 TPC ABH）

| 问题 | TPC (20260526) | BPC (本实验) |
|---|---|---|
| Promote H？ | ✅ bull DD 保护，跨窗 Pareto | ❌ H recent 最差（-5.59R） |
| 关 vol (B)？ | recent totR 最高 | recent **更差** |
| breakout 锚 | — | label 反向；**R 上 bull 略好、recent 无差异** → **保留锚** |

**行动**：

1. **gate**：维持生产 vol gates 全开（不引入 TPC 式 H）。
2. **prefilter**：保留 `bpc_recent_breakout_strength`；label 扫描反向不单独作为删锚依据。
3. **策略层**：BPC 在 2025–2026 recent **弱于 TPC**（~25 笔 vs TPC ~170+），优先做 **BPC vs ME/SRB 组合对照** 或 PCM 权重，而非继续微调 gate。

## 4. 复现

```bash
cd /home/yin/trading/ml_trading_bot
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/bpc_abh_variant_grid.yaml --quiet-signal-logs
```
