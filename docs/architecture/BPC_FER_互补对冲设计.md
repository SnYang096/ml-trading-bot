# BPC vs FER 互补对冲设计

> 结论日期: 2026-02-22
> 数据基础: FER holdout 2024-05 ~ 2025-12, 600 trades, Sharpe(daily)=4.64

## 1. 核心结论：互补对冲，非正交

BPC 和 FER **不是正交**（无关联），而是**负相关互补** — 一个做多时另一个倾向做空。

两者捕捉的是**同一个市场事件的两面**：
- BPC 进场的趋势突破，如果失败了，就变成 FER 的入场信号
- FER 的反转成功后形成新趋势，又可能成为 BPC 的下一个信号

## 2. 方向偏向对比

| 市场环境 | BPC (趋势跟随) | FER (失败反转) | 净效果 |
|----------|----------------|----------------|--------|
| **牛市** | 以做多为主 (顺势突破) | 以做空为主 (抓 pump 失败) | 自然对冲 |
| **熊市** | 以做空为主 (顺势突破) | 以做多为主 (抓 dump 失败) | 自然对冲 |
| **震荡市** | 信号少 (无明确趋势) | 信号多 (反复失败) | FER 补位 |

## 3. FER 方向信号的结构性偏向

FER 的方向由 `fer_impulse_failure_direction` 和 `cvd_change_5_normalized` 决定（negate_sign 规则）。

训练数据 2023-01 ~ 2026-01 以**牛市**为主 (BTC 16k → 100k+)：

```
fer_impulse_failure_direction (Rule 1, 覆盖率 31.4%):
  上行失败 → 做空: 9,930 样本 (88.3%)   ← 牛市 pump-and-dump 更多
  下行失败 → 做多: 1,310 样本 (11.7%)
  比例: 7.6 : 1
```

**原因**：加密市场牛市中「急涨后回落」(pump-and-dump) 远多于「急跌后反弹」(crash-and-bounce)。

**关键特性**：方向规则是确定性的 (negate_sign)，不是模型学出来的 → **市场切换时方向会自然跟着翻转**。

## 4. PCM 仲裁的价值

| 维度 | 说明 |
|------|------|
| 冲突场景 | 同一 bar: BPC 说"趋势延续做多", FER 说"上行失败做空" |
| 仲裁逻辑 | FER `pcm_priority: 0` (最高优先级), 因为 FER 条件最严格 (prefilter 只放 ~8%) |
| 常态 | 大部分时候不冲突 — 在不同的 bar 上触发 |
| 组合效果 | 牛市 BPC 做多 + FER 做空 → 自然分散; 熊市反过来 |

## 5. 需关注的风险

- **Gate/Evidence 模型的 regime 偏向**: 在牛市数据上训练的 gate 模型可能学到「做空质量更高」的偏向。进入熊市后，gate 对做多信号的放行率可能偏低。下一次模型重训需用更均衡的数据验证。
- **单边行情风险**: 如果牛市一直单边上涨不回调，FER 信号会很少（prefilter 过不了），这时系统几乎只靠 BPC → 分散效果下降。

## 6. 数据验证 (FER holdout)

```
FER 最终回测:
  Trades: 600  (384/year)
  Mean R: 0.7631
  Win Rate: 64.50%
  Sharpe (daily): 4.64

Per-Symbol:
  ADAUSDT: 143 trades, Win 62.9%
  BNBUSDT: 105 trades, Win 63.8%
  BTCUSDT:  58 trades, Win 62.1%
  ETHUSDT:  82 trades, Win 65.9%
  SOLUSDT:  86 trades, Win 65.1%
  XRPUSDT: 126 trades, Win 66.7%
```

> 下一步: BPC + FER PCM 联合回测，验证实际对冲效果和冲突率
