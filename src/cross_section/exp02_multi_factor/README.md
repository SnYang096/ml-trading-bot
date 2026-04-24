# 实验 2：多因子 + 板块中性 横截面 L/S

**目标**：基于 exp01 结论（简单 6 币种 XS 动量 Net Sharpe 0.75、回撤 35%），扩大币种池到 20+，
引入多因子合成并按板块中性化，目标把 Net Sharpe 推到 **>1.5**、回撤压到 **<15%**。

## 目录结构

```
exp02_multi_factor/
├── data_loader.py      # 价格(重采样tick) + funding rate 面板加载
├── sectors.py          # 板块定义 + 截面/板块内 z-score 中性化
├── factors.py          # 因子库：momentum / reversal / funding / low_vol / winsorize
├── backtester.py       # 多因子合成 -> 权重映射 -> L/S 回测 (gross/net)
├── run.py              # 入口脚本
└── README.md
```

## 默认配置

- **币种池**：65 个 USDT perp（见 `sectors.SECTOR_MAP`），按 `--min-coverage` 过滤不足期的币
- **因子组合**（`run.default_factor_specs`）：
  | 因子 | kind | lookback | weight |
  |---|---|---|---|
  | mom_7d | momentum | 168h | 1.0 |
  | mom_30d_skip1d | momentum skip-1d | 720h+24 | 0.5 |
  | reversal_1d | short-term reversal | 24h | 0.5 |
  | funding_3d | funding mean (取负) | 72h | 0.5 |
  | low_vol_7d | realized vol (取负) | 168h | 0.3 |
- **中性化**：板块内 z-score → 线性加权 → 整体 z-score
- **权重映射**：score-weighted（默认，按 |z|>0.5 分配）或 rank-based (`--top-k`/`--bottom-k`)
- **美元中性**：`sum(w)=0`, `sum(|w|)=1`
- **持仓**：`--hold-bars 24`（每日 rebalance）
- **成本**：`--fee-bps 5.0`（单边，按 |Δw| 计）

## 运行

```bash
# 默认：65 币种，2023-01 -> 2026-03，每天 rebalance
python -m src.cross_section.exp02_multi_factor.run

# 自定义：rank-based top-5/bottom-5
python -m src.cross_section.exp02_multi_factor.run \
    --top-k 5 --bottom-k 5 --start 2024-01 --end 2026-03

# 关闭板块中性（对照）
python -m src.cross_section.exp02_multi_factor.run \
    --no-sector-neutral
```

## 产出

`reports/cross_section/exp02/`:
- `prices.parquet` / `returns.parquet` / `funding.parquet`：对齐后的面板
- `equity.parquet` + `equity.png`：gross / net 净值曲线
- `weights.parquet`：每个 rebalance 时点的目标权重
- `sector_exposure.png`：各板块净暴露（应在 0 附近波动）
- `metrics.json`：全部绩效指标
- `summary.md`：结论与下一步建议

## 判读

- **Net Sharpe > 1.5 + MaxDD < 15%**：策略可进入小资金实盘验证
- **Net Sharpe 0.8–1.5**：方向正确，需调因子权重 / hold_bars / top_k
- **Gross vs Net Sharpe 差距 > 0.5**：换手过高，拉长 hold_bars 或降频
- **板块净暴露持续偏离 0**：中性化失效，检查 `SECTOR_MAP` 分类
