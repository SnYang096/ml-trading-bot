# Multi-Alpha Holding & Fat-Tail 捕获架构

> **系统实现状态 (2026-03-05)**
>
> | 组件 | 状态 | 实现位置 |
> |------|------|--------|
> | EMA200 特征 | ✅ 已实现 | feature_dependencies.yaml `ema_200_value_f` |
> | BPC EMA200 regime filter | ✅ 已实现 | features_gate.yaml + features_evidence.yaml |
> | BPC structural exit (EMA200) | ✅ 已实现 | execution.yaml `structural_exit: ema200` |
> | BPC 灾难 trailing (7R) | ✅ 已实现 | execution.yaml `trail_r: 7.0` |
> | ME 紧 trailing (0.5R) | ✅ 已实现 | execution.yaml `trail_r: 0.5` |
> | Break-even lock | ✅ 已实现 | holding.yaml `breakeven_lock.trigger_r=1.0` |
> | Pyramiding ×5 | ✅ 已实现 | constitution.yaml `max_add_times=5` |
> | BPC/ME no TP | ✅ 已实现 | execution.yaml `take_profit.enabled=false` |
> | FER fixed TP | ✅ 已实现 | execution.yaml `target_r=3.0` |
> | time_stop 禁用 | ✅ 已实现 | execution.yaml `time_stop_bars=0` (BPC/ME) |
> | Partial TP | ❌ 明确排除 | break-even lock 在 1R 已保护，Partial TP 每笔 fat tail 收 34% 税 |
> | 三端一致性 | ✅ 已实现 | enforce_position() + 向量/事件回测 + 实盘 |

---

## 一、核心思想

> **不要试图预测 fat tail，而是设计持仓结构，让 fat tail 自己发生。**

Fat tail 的正确位置在 Holding / PCM，而不是 Entry。

```
Entry → BPC / ME (正常入场)
Holding → 捕捉 fat tail (止损固定, 盈利开放)
PCM → 放大 fat tail (pyramiding)
```

### 为什么不在 Entry 预测 fat tail

试图设计 `fat_tail_signal`（RSI 极端、funding 极端、OI 爆炸）= tail prediction，样本极少，容易过拟合。

真正的 fat tail 路径: `BPC → ME → 趋势加速 → 清算链 → 极端行情`

正确做法: 止损固定 1R，盈利不封顶 → fat tail 自然出现在收益分布中。

### Fat tail ≠ 风险增加 / 运气 / 过拟合

- **风险**: SL 仍然是 1R，不变。变化的只是盈利上限 = ∞
- **运气**: 短期可能，长期不是。Crypto 极端行情频率远高于正态分布，肥尾是市场结构
- **过拟合**: 你没有预测那笔 +15R，只是没有把它提前卖掉

---

## 二、Multi-Alpha Holding 核心架构

```
signal type → 对应 holding engine（而非所有信号 → 同一套 exit）
```

```
BPC signal → trend_hold    → EMA200 regime filter + EMA200 exit + 极宽灾难 trailing
ME  signal → momentum_hold → 紧 trailing (快进快出, 锁利润)
FER signal → fixed_tp      → 快速止盈 (反转行情持续时间短)
```

### 两种 Fat-Tail 范式

| 维度          | BPC (趋势型)               | ME (动量型)           |
| ------------- | -------------------------- | --------------------- |
| 抓的 fat-tail | 结构趋势: 缓涨 2-6 个月    | 动量爆发: 急涨 2-5 天 |
| 单笔典型 R    | 15-50R (稀少但单笔极大)    | 3-8R (频繁但每笔中等) |
| 盘整行为      | 穿越所有盘整, 直到结构破坏 | 第一次盘整就退出      |
| 利润回吐      | 多 (EMA200 离现价远)       | 少 (锁得紧)           |
| 交易频次      | 低                         | 高                    |

两者是互补关系，不是优劣关系。震荡上行(台阶式)紧 trailing 赢; 单边趋势(持续走) EMA200 赢。

---

## 三、BPC: EMA200 = Regime Filter + Exit (方案 A)

EMA200 在 BPC 中同时承担两个角色:

1. **Regime Filter (Gate 层)**: 只允许顺 EMA200 方向交易
2. **Structural Exit (Holding 层)**: 价格穿越 EMA200 = 趋势结构破坏

```
price > EMA200 → BPC 只做多
price < EMA200 → BPC 只做空
exit: close 穿越 EMA200
```

### 为什么 BPC 必须用方案 A

BPC 是 structure breakout 策略, compression breakout 成功率高度依赖 trend alignment:
- 在 EMA200 错误侧做 breakout = bear market rally, 大概率被打回
- EMA200 regime filter 直接过滤掉 "wrong direction fat tail" 交易
- 防止熊市中连续做多 breakout → 不断止损

### 排除的方案

- **方案 B (结构优先)**: BPC 决定方向, EMA200 只负责退出
  - 问题: EMA200 变成非常远的 stop, 风险不合理 (entry=100, EMA200=80 → 20% 回撤)
- **方案 C (EMA200 只做 regime filter)**: 不用 EMA200 退出, 用 ATR trailing
  - 问题: 紧 trailing 截断 fat tail (已验证)

---

## 四、ME: 无 EMA200 限制

ME 是 microstructure alpha (liquidation, orderflow imbalance, momentum burst), 可以逆 EMA200:

```
price < EMA200 + liquidation squeeze → ME 做多是合理的
```

ME 的 alpha 来源与趋势结构无关, 强制 regime filter 会错过合理交易。

---

## 五、Fat-Tail 捕获的持仓层次

```
Layer 1  Break-even lock (1R) → 无风险仓
Layer 2  Trend ride (trailing / EMA200 exit)
Layer 3  Pyramiding → 放大 fat tail
```

### Layer 1: Break-even Lock

```
盈利 1R → SL 移动到 entry → 最大亏损 = 0
```

剩余仓位变成 free ride，可以无限持有而不用担心亏损。

### Layer 2: Trend Ride

**BPC (trend_hold)**: EMA200 structural exit — 趋势结束才退出，穿越所有盘整

**ME (momentum_hold)**: 紧 trailing (0.5R) — 第一次盘整就退出，锁住利润

**FER**: fixed TP (3R) — 反转行情持续时间短，快速止盈

### Layer 3: Pyramiding

```
max_add_times = 5
require_locked_profit = true (加仓只在盈利后)
```

效果: 10R 趋势无 pyramiding → 10R 收益; 有 pyramiding → 16R+ 收益

Pyramiding 提高 Sharpe: downside risk 不变 (加仓只在盈利后), upside potential 增加。

---

## 六、Holding Profile 配置

### trend_hold (BPC)

```yaml
# 主退出: EMA200 结构性退出
structural_exit: ema200

# 灾难保护: 极宽 trailing, 仅防黑天鹅闪崩
trail_r: 7.0

# 保本锁: 1R
breakeven_trigger: 1.0

# 加仓: 允许 (max_add_times=5, require_locked_profit)
pyramiding: true
```

### momentum_hold (ME)

```yaml
# 主退出: 紧 trailing stop
activation_r: 1.0
trail_r: 0.5

# 保本锁: 1R
breakeven_trigger: 1.0

# 加仓: 允许
pyramiding: true
```

### fixed_tp (FER)

```yaml
# 主退出: 固定止盈
target_r: 3.0
no pyramiding
```

---

## 七、Slot 分区

每策略独占 slot, 互不竞争:

```yaml
per_strategy_limits:
  bpc:
    max_slots: 1
    holding_profile: trend_hold
  me:
    max_slots: 1
    holding_profile: momentum_hold
```

trend_hold 持仓时间可能数周; momentum_hold 持仓数小时到数天。
slot 分区确保 BPC 长期持仓不会阻塞 ME 交易。

---

## 八、三端一致性

两套 holding profile 通过各策略自有的 execution.yaml 配置,
在三端 (向量回测 / 事件回测 / 实盘) 共享 `enforce_position()` 统一执行。

```
enforce_position() 步骤:
  1. 初始止损
  2. 保本锁 (breakeven)
  3. HWM 更新
  3b. Structural exit (EMA200) — breakeven 之后, trailing 之前
  4. Trailing stop
  5. TP / Timeout
```

---

## 九、系统收益结构

```
BPC / ME → fat tail holding → 趋势单吃 fat tail
FER → fixed TP → 反转单提供稳定收益
```

成功趋势系统的典型统计: Top 5 trades 贡献 50%-80% 收益。这不是 bug，是 feature。

Crypto 特别适合: 杠杆清算 + 24h 交易 + 极端波动 → trend → liquidation cascade → fat tail。

---

## 十、设计原则

1. **Holding profile 数量控制**: 只做 trend_hold + momentum_hold + fixed_tp, 不膨胀
2. **Alpha 来源不变**: Holding 不是 Alpha 源, 是 Alpha 保全层
3. **独立优化**: BPC/ME/FER holding 可分别优化, 减少过拟合风险
4. **为 ML/RL 预留**: 未来新模型可指定 holding profile, 架构已支持
5. **EMA200 可调**: crypto 常用 EMA100/EMA150 (EMA200 较慢), 以后可优化
