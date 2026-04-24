# 实验 3：因子 IC + Grid Search

两个独立入口：

## Part 1 - 因子 IC 分析 (`run_ic.py`)

对 12 个候选因子（不同 kind / lookback）× 3 个 horizon（1d / 3d / 7d）×
3 个变体（raw / 跨截面 z / 板块中性 z）计算 IC：

```bash
python -m src.cross_section.exp03_ic_and_grid.run_ic \
    --start 2023-01 --end 2026-03
```

产出：
- `ic_all.csv`：108 行全量 IC 表
- `ic_xs_z_h24_ranked.csv`：xs-z + 1d horizon 排序
- `quantile_fwd_returns.csv`：分位 forward return
- `q5_q1_spread.png`：所有因子的 Q5-Q1 spread bar chart
- `summary.md`：Top/Bottom 因子 + 解读

**用途**：决定哪些因子保留、哪些翻转符号、哪些丢弃。

## Part 2 - Grid Search (`run_grid.py`)

固定 default 因子组合，遍历：
- `hold_bars` ∈ {24, 72, 168, 336}（1d / 3d / 7d / 14d）
- `top_k` ∈ {3, 5, 8, 12}
- `sector_neutral` ∈ {off, on}

= **32 组**配置：

```bash
python -m src.cross_section.exp03_ic_and_grid.run_grid \
    --start 2023-01 --end 2026-03
```

产出：
- `grid_results.csv`：全部配置的指标表
- `top5_equity.png`：Net Sharpe 前 5 名的净值曲线对比
- `summary.md`：Top 10 表格 + Bottom 5

**用途**：定位最优"换手率 × 持仓数 × 中性化"组合。
