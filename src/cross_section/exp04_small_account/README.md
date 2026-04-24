# 实验 4：小资金版（1 万 USD）

**设计动机**：1 万美金账户 + 希望**少持仓、持仓更久、尽量不来回换**。

## 硬约束

| 项 | 值 |
|---|---|
| 账户规模 | $10,000 |
| 最大同时持仓 | **4 个币**（2 多 + 2 空） |
| 单腿名义 | ~$2,500 |
| 持仓周期 | **默认 14 天 rebalance**（可调） |
| 单腿止损 | 累计亏损 15% 强制平 |
| 费率 | 8 bps/side (taker 0.04% + 滑点 0.04%) |
| 候选池 | **20 个高流动性币**（无 meme、无新上） |

候选池（`config.LIQUID_POOL`）：
- L1 major/alt 共 16 个
- L2 2 个 (ARB, OP)
- DeFi 2 个 (UNI, LINK)

## 因子预设

| preset | 因子 | 说明 |
|---|---|---|
| `ic_top` (默认) | low_vol 14d(w=1.0) + low_vol 7d(0.6) + reversal 24h(0.6) | 基于 exp03 IC 分析的胜者 |
| `balanced` | low_vol + reversal + 小权重 mom + funding | 稳+动量分散 |
| `mom_only` | 7d + 14d momentum | 对照（exp03 发现动量 IC 差） |

## 运行

```bash
# 默认（ic_top 因子 + 14 天持仓 + 2L/2S）
python -m src.cross_section.exp04_small_account.run

# 更保守：7 天持仓 + 2L/2S + balanced 因子
python -m src.cross_section.exp04_small_account.run \
    --hold-bars 168 --factor-preset balanced \
    --outdir reports/cross_section/exp04_balanced_7d

# 对照组：momentum-only
python -m src.cross_section.exp04_small_account.run \
    --factor-preset mom_only --outdir reports/cross_section/exp04_mom
```

## 产出

`reports/cross_section/exp04/`:
- `equity.parquet` + `equity.png`：gross / net / BTC B&H 三条曲线
- `trades.parquet`：每次 rebalance 的持仓+止损信息
- `metrics.json`：全指标
- `summary.md`：含交易明细（每个币被持仓几次、被止损几次、平均名义 USD）

## 止损机制

逐根 K 线累计每腿 pnl（对做多腿 = +returns，做空腿 = -returns），
超过 `stop_loss_per_leg` 立刻关闭该腿（计平仓费），该腿仓位在本次 rebalance 周期剩余时间内为 0。
下次 rebalance 重新开仓。

## 下次迭代方向

- 若 Net Sharpe 仍 < 1：引入 **regime filter**（例如 BTC 30d vol 高于阈值时减仓/空仓）
- 若 stopped out 比例 > 30%：放宽止损或检查因子是否在当前市场失效
- 若 pct_feasible < 100%：单腿不足 $200，需增加账户规模或降低持仓数量
