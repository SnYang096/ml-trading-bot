# Multileg portfolio metrics — timeline canonical

实现：`scripts/pipeline/multileg_portfolio_metrics.py`

## Canonical return

1. 确定 `n_symbols` = trades 中 distinct symbol 数（至少 1）。
2. 按 `exit_time` 排序每笔 trade。
3. 组合增量 = `pnl_per_capital / n_symbols`。
4. 累加得 `cum_pnl_per_capital`；`return_pct` = 终值 × 100。

每币仍运行在独立 capital bucket；组合假设 **等权 notional**（总资金 = `initial_capital × n_symbols`，每币 `initial_capital`）。

## Legacy 对照

- **eq-mean：** 各 symbol 的 `sum(pnl_per_capital)` 再取均值 × 100。终值常与 timeline 相同，但 **忽略时间路径**。
- **pooled：** 全部 trade 直接相加 × 100，约放大 `n_symbols` 倍。

## Drawdown

- **`max_drawdown_portfolio`：** timeline `cum_pnl_per_capital` 相对 running peak 的最小值（归一化）。
- **`portfolio_cum_dd`（实验脚本）：** segment `end` 时间序上 pooled segment pnl 的 cum DD — 未按 symbol 权重，仅作 legacy 对照。

## Sharpe

- **`daily_sharpe`：** 对 timeline 的 **日度组合增量**（`portfolio_pnl_per_capital` 按日历 `1D` resample 求和）计算 Sharpe，年化因子 √365。
- 旧 chop_grid 用 pooled 日度 `pnl_per_capital` 求和，会高估约 `n_symbols` 倍波动/收益比。

## Capital report

`write_capital_report_from_trades(..., initial_capital=portfolio_total, n_symbols=n)`：

- `initial_capital` = 组合总 notional（per-symbol × n）
- 每笔 USD = `pnl_per_capital / n_symbols × initial_capital`
- `total_r` 应传 timeline 终值 `portfolio_pnl_per_capital_timeline`
