# B+ 回测（lottery_backtest_bplus）

## 作用

把 **杠杆容量 parquet** 里的信号转成 **可数的交易序列 + 资金曲线**，用于回答「宏观门 ∧ 容量过滤之后，持有 H 根 bar 平仓，账面大概赚多少」，**不等于**实盘完整撮合。

## 入口

- **脚本**：`scripts/lottery_backtest_bplus.py`
- **默认配置**：`config/strategies/bad-candidates/lottery100/backtest_bplus.yaml`

```bash
python scripts/lottery_backtest_bplus.py --config config/strategies/bad-candidates/lottery100/backtest_bplus.yaml
```

产出目录（可由 YAML `output.directory` 改）：默认 `reports/lottery_bplus_default/`

| 文件 | 内容 |
|------|------|
| `trades.csv` | 每笔：进场/出场时间、`r_hold`、`eff_leverage`、`pnl_on_stake` |
| `equity_curve.csv` | 按平仓时间累加的 `equity` |
| `summary.json` / `summary.md` | 笔数、均值/方差、胜率、近似最大回撤 |

## PnL 口径（必读）

- **`pnl_on_stake`**：`eff_L * r_hold - funding_sum - fee_rt`，其中 `eff_L = min(名义杠杆, lmax_adj)`（默认打开 cap）。
- **单位**：按 **单位保证金 stake=1** 记账；数值可远大于 1（例如 ≈5 表示约 **500% ROE**，因高杠杆 × 正的持有期涨跌）。
- **不含**：强平路径模拟（除非 `--liquidate-if-over-capacity`）、盘口滑点分级、分批减仓。

## 信号含义

默认样本来自 **`lmax_adj ≥ min_lmax`**（例如 100）的全样本 **long**、`H=120`。这是「结构上曾允许极高杠杆」的时刻；**叠加** parquet 自带的 `bull_only`（若你用的是 v4 bull_only 文件）则由文件本身决定样本子集。

## 与彩票管线关系

B+ **不替代** Archetype 的 prefilter/gate/model；它只是 **Evidence 前的经济检验**。参数满意后再考虑接纸交易 / 薄 runner。
