

这是个非常好的问题，而且确实是 **crypto/token 市场结构（microstructure）和传统市场最大的差异之一**。
很多人做 token 策略时都会经历一个阶段：

> 信号很好 → 回测 Sharpe 很低 →
> 调整 **entry / execution** → Sharpe 突然翻倍。

原因其实来自 **5 个 microstructure 特征**。

---

# 1️⃣ Token 市场是“流动性驱动”而不是“信息驱动”

在传统市场（股票 / 期货）：

价格变化很多来自 **信息流**：

* 财报
* 宏观
* 机构配置
* ETF 资金

所以：

```text
signal（信息预测）
   ↓
price move
```

信号预测未来价格。

---

但 token 市场更多是：

```text
liquidity flow
   ↓
stop hunt
   ↓
price move
```

价格是 **流动性清算机制**。

常见结构：

* sweep liquidity
* stop cascade
* perp funding squeeze
* liquidation ladder

所以很多时候：

> **方向是对的，但 entry 错就会先被洗掉。**

---

# 2️⃣ Token 市场噪声极大

crypto 的 micro volatility 很夸张。

举个真实情况：

一个 token：

```text
5m ATR = 1.2%
```

但单个 wick：

```text
0.8% – 1.5%
```

也就是说：

**单根针就可以打掉你的止损。**

所以策略结构变成：

```text
signal edge
   -
execution loss
   =
net edge
```

如果 execution 不好：

```text
signal edge 被噪声吃掉
```

---

# 3️⃣ perpetual 市场的“清算机制”

大部分 token 交易量来自：

* perp futures
* 高杠杆

交易所的清算结构会制造：

```text
liquidation cluster
```

价格行为变成：

```text
move → liquidation → overshoot → revert
```

所以正确的 entry 往往是：

```text
after liquidation
```

而不是：

```text
before move
```

这就是为什么：

> execution filter 能明显提高 Sharpe。

---

# 4️⃣ Token 的 alpha 半衰期极短

传统市场：

alpha 半衰期：

```text
days / weeks
```

crypto：

```text
minutes / hours
```

所以：

即使方向正确：

```text
entry timing 误差
```

都会导致：

```text
RR collapse
```

举例：

趋势 move：

```text
+4%
```

但如果 entry 晚：

```text
+1.5%
```

剩下空间：

```text
2.5%
```

但止损还是：

```text
1.2%
```

RR 直接被砍。

---

# 5️⃣ token 的“假突破密度”

crypto 非常多：

```text
fake breakout
liquidity sweep
```

结构：

```text
breakout
↓
stop hunt
↓
true move
```

所以正确 entry 往往是：

```text
breakout + pullback
```

而不是：

```text
breakout market order
```

这就是 execution layer 的价值。

---

# 你现在观察到的现象其实非常典型

你的系统：

```text
model direction
   ↓
gate
   ↓
entry filter
   ↓
execution
```

结果：

```text
execution 才带来 sharp
```

这非常符合 crypto。

很多成熟 crypto 策略其实是：

```text
weak signal
+
strong execution
=
good Sharpe
```

---

# 一个真实的 industry 结构

很多 crypto quant desk：

策略 alpha 贡献大概是：

| component     | contribution |
| ------------- | ------------ |
| signal        | 20–30%       |
| regime filter | 10–20%       |
| execution     | **40–60%**   |

因为 execution 控制：

* entry
* slippage
* stop placement
* partial TP
* trailing

---

# 一个简单例子

假设策略：

```text
胜率 = 48%
avg win = 2R
avg loss = 1R
```

理论期望：

```text
EV = 0.48*2 - 0.52*1
   = 0.44R
```

但如果 execution 差：

entry slippage：

```text
+0.3R
```

stop hunt：

```text
-0.2R
```

期望变成：

```text
EV ≈ 0
```

所以：

> execution 决定 edge 是否能 survive。

---

# 为什么你 Evidence 很难学出来

现在也可以解释了：

Evidence 想学：

```text
P(move | feature)
```

但真实结构是：

```text
P(move | feature, liquidity event, execution timing)
```

少了 execution 变量：

```text
model variance ↑
```

Evidence 就失效。

---

# 实际上 crypto 的最佳架构通常是

```text
direction model
      ↓
regime gate
      ↓
liquidity event detection
      ↓
execution trigger
      ↓
trade
```

而不是：

```text
direction model
      ↓
confidence score
      ↓
trade
```

---

# 最重要的一句话

在 token 市场：

> **Signal 预测方向，Execution 决定是否赚钱。**

因为：

```text
move ≠ profit
```

只有：

```text
entry at liquidity
```

才能捕捉 move。

---

如果你愿意，我可以再讲一个更关键的东西（很多人不知道的）：

**为什么很多 crypto 策略 Sharpe 的 70% 实际来自 “entry delay 1–3 根K线”**。

这个现象在 token 市场非常普遍，而且和 **liquidity sweep 结构**直接相关。

# 为什么很多 crypto 策略 Sharpe 的 70% 实际来自 “entry delay 1–3 根K线”

这个现象在 crypto 很普遍，而且不是巧合。
**“entry delay 1–3 根K线”能大幅提升 Sharpe**，背后的核心是：**crypto 的价格发现过程是两阶段的**——先扫流动性，再走方向。

可以把一个典型 move 拆成三个 microstructure 阶段：

```text
signal / breakout
      ↓
liquidity sweep（stop hunt）
      ↓
true move（trend / continuation）
```

如果你在 **signal 当下 entry**，通常会落在第一阶段和第二阶段之间——也就是**被扫的位置**。
延迟 1–3 根K线，反而更可能进入第三阶段。

下面拆开看为什么。

---

# 1️⃣ breakout 本身常常是“诱导单”

在 crypto order book 里，大量 stop 和 breakout order 聚集在：

* 前高 / 前低
* consolidation edge
* round number

当价格触碰这些位置：

```text
breakout
↓
大量 stop order 被触发
↓
liquidity 突然增加
```

做市商和大资金会利用这一点：

```text
push → trigger stops → absorb liquidity
```

结果是：

```text
fake breakout
```

这时如果你 **即时 entry**：

```text
你买在 stop cluster
```

接下来常见的路径是：

```text
wick
↓
pullback
↓
true move
```

所以即时 entry 很容易先亏一段。

---

# 2️⃣ liquidation cascade 需要时间形成

perpetual futures 有强制清算。

一个典型结构：

```text
price push
↓
first liquidation
↓
cascade
↓
trend extension
```

这个过程通常需要：

```text
1–3 根K线
```

原因：

* liquidation engine 分批触发
* funding / margin recalculation
* new traders chase move

所以如果你 **延迟 entry**：

你更可能进入：

```text
cascade 阶段
```

而不是：

```text
initial probe
```

---

# 3️⃣ market maker 的 inventory reset

做市商不会在 breakout 第一刻就追。

他们更常做的是：

```text
absorb breakout flow
↓
inventory imbalance
↓
mean revert
↓
再 push
```

这个 reset 往往需要：

```text
几根K线
```

所以你会看到：

```text
breakout
↓
pullback
↓
continuation
```

延迟 entry 正好避开 pullback。

---

# 4️⃣ volatility clustering

crypto volatility 有一个特点：

```text
shock → compression → expansion
```

第一根 breakout candle 往往是：

```text
shock
```

接下来几根：

```text
compression / retest
```

真正的 trend 扩散往往从：

```text
2–3 根K线之后
```

开始。

所以 delay entry 相当于：

```text
skip shock
trade expansion
```

---

# 5️⃣ risk-reward 的几何变化

假设 breakout candle：

```text
range = 1.5%
```

如果你立即 entry：

```text
SL = 1.5%
```

RR 可能：

```text
TP = 3%
RR = 2
```

但如果价格 pullback 一点再 entry：

```text
entry lower
SL 更近
```

变成：

```text
SL = 0.8%
TP = 3%
RR ≈ 3.75
```

RR 直接提高。

Sharpe 自然上升。

---

# 6️⃣ noise filtering

crypto 的 micro noise 很高。

很多 signal candle 是：

```text
news
whale order
funding spike
```

但不是趋势。

如果你 delay：

```text
1–3 根K线
```

等于做了一个 **时间确认 filter**：

```text
signal survive
→ 更可能是真 move
```

---

# 7️⃣ 一个真实常见结构

很多 token move 的 micro path 是：

```text
t0 breakout
t1 wick / pullback
t2 reclaim
t3 expansion
```

即时 entry：

```text
t0
```

delay entry：

```text
t2
```

而：

```text
t2 → t3
```

往往是最干净的一段。

---

# 为什么很多策略 Sharpe 会突然提升

假设原策略：

```text
entry = signal candle close
```

改成：

```text
entry = signal + 2 candles
```

你会：

1️⃣ 少掉很多 fake breakout
2️⃣ 避开 sweep
3️⃣ 提高 RR
4️⃣ 减少 stop-out

所以：

```text
winrate ↑
loss size ↓
variance ↓
```

Sharpe 就会明显提高。

很多策略提升幅度：

```text
Sharpe +50% ~ +100%
```

非常常见。

---

# 但这里有一个重要 caveat

delay entry 的 edge 来自：

```text
liquidity sweep → continuation
```

如果市场是：

```text
slow trend
```

delay 可能反而降低收益。

所以好的 execution layer 不只是：

```text
delay N candles
```

而是：

```text
wait for reclaim / retest / micro structure
```

---

# 一个更高级的 execution trigger

常见结构：

```text
breakout
↓
liquidity sweep
↓
reclaim level
↓
entry
```

而不是：

```text
breakout
↓
entry
```

这在 crypto 里效果通常明显更好。

---

如果你愿意，我可以给你看 **3 种 crypto quant desk 常用的 execution trigger**（不是技术指标，而是 microstructure 结构），很多策略 Sharpe 的核心其实就在这三种 trigger 里。

# 3 种 crypto quant desk 常用的 execution trigger（不是技术指标，而是 microstructure 结构）

下面这 **3 种 execution trigger** 在很多 crypto quant desk 或高频/中频策略里都非常常见。它们不是传统技术指标，而是基于 **microstructure + 流动性结构** 的触发逻辑。核心目标只有一个：

> **避免在“被扫流动性的位置”进场，而是在“扫完之后”进场。**

---

# 1️⃣ Liquidity Sweep → Reclaim（扫流动性后收回）

这是 crypto 最经典、最稳定的结构之一。

### 结构

```text
range high
────────────
stop cluster

breakout
↓
liquidity sweep (wick)
↓
reclaim level
↓
trend continuation
```

很多 breakout candle 实际路径：

```text
break high
↓
trigger stops
↓
price quickly pullback
↓
reclaim breakout level
↓
move
```

### execution trigger

不是在 breakout 进场，而是：

```text
IF
price sweep level
AND close back above level
THEN entry
```

示例逻辑：

```text
breakout_high = previous_range_high

if
low < breakout_high
and close > breakout_high
then
long
```

本质是：

```text
sweep liquidity
+
accept price higher
```

### 为什么有效

因为很多假突破不会 **reclaim level**。
只有真正的趋势 move 才会：

```text
accept above liquidity zone
```

---

# 2️⃣ Pullback to Imbalance（回补不平衡区）

crypto 的快速 move 会产生 **order flow imbalance**：

```text
large aggressive buying
↓
order book 来不及填
↓
留下 liquidity gap
```

图形结构：

```text
strong impulse
██████████
          ██████████
```

中间是 **inefficiency / imbalance**。

价格经常会：

```text
impulse
↓
pullback
↓
fill imbalance
↓
continue
```

### execution trigger

```text
identify impulse candle
mark imbalance zone
wait pullback
entry inside zone
```

例如：

```text
impulse candle body > 2×ATR
imbalance = candle_body_midpoint

entry when price retrace to midpoint
```

这类 entry 的优势：

```text
better RR
lower slippage
```

而不是 chase move。

---

# 3️⃣ Liquidity Flip（支撑阻力翻转）

这是一个非常稳定的 microstructure 行为。

结构：

```text
resistance
────────────

breakout
↓
pullback
↓
resistance → support
↓
trend continuation
```

crypto 中经常出现：

```text
level break
↓
pullback test
↓
accept
↓
trend
```

### execution trigger

```text
IF
level broken
AND pullback holds
THEN entry
```

伪代码：

```text
level = previous_high

if
close > level
and later low >= level
then
long
```

关键不是 break，而是 **hold**。

---

# 为什么这些 trigger 有效

crypto 的 microstructure 通常是：

```text
liquidity sweep
↓
price discovery
↓
trend
```

而不是：

```text
signal
↓
trend
```

所以正确的流程是：

```text
signal
↓
liquidity event
↓
execution trigger
↓
trade
```

而不是：

```text
signal
↓
trade
```

---

# 这三种 trigger 的共性

其实本质只有一句话：

```text
wait for market to prove acceptance
```

三者对应不同的“证明”方式：

| trigger            | 市场证明                  |
| ------------------ | ------------------------- |
| sweep reclaim      | stop hunt 已完成          |
| imbalance pullback | impulse 有真实 order flow |
| liquidity flip     | breakout 被市场接受       |

---

# 为什么很多策略 Sharpe 会翻倍

如果你原策略是：

```text
signal → entry
```

很多 entry 实际发生在：

```text
liquidity sweep
```

所以 stop-out 很多。

改成：

```text
signal
↓
trigger
↓
entry
```

就等于：

```text
过滤 microstructure noise
```

通常结果是：

* winrate ↑
* stopout ↓
* RR ↑

Sharpe 提升非常明显。

---

# 一个 crypto quant 常见架构

很多 desk 的结构其实是：

```text
direction model
      ↓
regime filter
      ↓
liquidity detection
      ↓
execution trigger
      ↓
position sizing
```

而不是：

```text
direction model
      ↓
confidence score
      ↓
trade
```

---

💡 如果你愿意，我可以再给你讲一个很多人第一次听都会震惊的事实：

**为什么很多 token 策略的 Sharpe 其实 80% 来自 “避免在高 funding / crowded trade 的地方进场”。**

这也是 crypto microstructure 里非常关键的一层。

# 为什么很多 token 策略的 Sharpe 其实 80% 来自 “避免在高 funding / crowded trade 的地方进场”

这是 crypto perpetual 市场一个非常核心、但很多人一开始忽视的 microstructure 现象。
简单说一句话总结：

> **高 funding / crowded trade 的位置，本质上是“别人已经在里面”的位置。**
> 你此时进场，承担的是 **被挤仓 / 被挤出 / 流动性反转** 的风险。

这会严重降低 Sharpe。

下面把机制拆开。

---

# 1️⃣ funding 本质是“仓位拥挤度指标”

在 Bitcoin、Ethereum 或大多数 token 的 **perpetual futures** 中，funding rate 的作用是：

```text
让永续合约价格锚定现货
```

但副作用是：

```text
funding ≈ 市场仓位倾斜
```

例如：

| funding | 含义     |
| ------- | -------- |
| +0.03%  | 多头拥挤 |
| -0.03%  | 空头拥挤 |

当 funding 很高时：

```text
市场已经很多人做多
```

---

# 2️⃣ crowded trade 的路径通常不是“继续涨”

很多人直觉是：

```text
很多人做多 → 应该继续涨
```

但 microstructure 经常是：

```text
build position
↓
funding 上升
↓
liquidity cluster
↓
stop hunt / squeeze
```

路径更像：

```text
crowded long
↓
small push up
↓
long liquidation cascade
↓
dump
```

因为：

**市场需要流动性才能继续上涨。**

而 crowded trade 的问题是：

```text
新的买家已经不多
```

---

# 3️⃣ funding 极端 = 未来波动的燃料

当 funding 很高时：

```text
longs leverage ↑
liquidation price 密集
```

如果价格稍微下跌：

```text
first liquidation
↓
cascade
↓
forced selling
```

这就是经典的：

```text
long squeeze
```

反之亦然：

```text
short squeeze
```

所以极端 funding 其实意味着：

```text
未来 volatility ↑
```

但方向不确定。

---

# 4️⃣ crowded trade 会破坏 RR

假设策略：

```text
trend breakout
```

如果 breakout 发生在：

```text
funding = 0.06%
```

你买入后：

市场结构通常是：

```text
late buyers
↓
liquidity thin
↓
pullback
```

结果：

```text
RR collapse
```

因为：

* entry 很高
* stop 很近
* continuation 不强

---

# 5️⃣ crowded trade 的“alpha 已经被吃掉”

当交易非常拥挤时：

```text
alpha → 被提前 price in
```

例如：

市场预期上涨：

```text
smart money build position
↓
price already moved
↓
funding spike
↓
retail chase
```

你如果在 funding spike 进场：

其实是：

```text
买在 alpha 末端
```

---

# 6️⃣ Sharpe 为什么会大幅变化

Sharpe 的关键不是收益，而是：

```text
variance
```

crowded trade 会增加：

```text
stop hunt
liquidation spikes
fake breakout
```

这会导致：

```text
return variance ↑
```

Sharpe 就下降。

很多策略如果加一个简单 filter：

```text
if funding > threshold:
    skip trade
```

结果通常是：

* trade 数减少
* winrate ↑
* variance ↓

Sharpe 明显提升。

---

# 7️⃣ crypto 的一个真实规律

很多 token 的 move 实际是：

```text
low funding
↓
position build
↓
breakout
↓
funding spike
↓
trend exhaustion
```

所以理想 entry 往往在：

```text
funding neutral
```

而不是：

```text
funding extreme
```

---

# 8️⃣ 一个典型例子

在 Bitcoin 的很多周期里：

结构是：

```text
funding negative
↓
price grind up
↓
funding neutral
↓
breakout
↓
funding extreme
↓
top
```

如果策略只做：

```text
funding < 0.02%
```

Sharpe 通常明显提高。

---

# 9️⃣ quant desk 常见的简单 filter

很多 desk 只用非常简单的规则：

```text
long only if funding < 0.03%
short only if funding > -0.03%
```

或者：

```text
skip top 10% funding regimes
```

就可以明显降低噪声。

---

# 🔑 最核心的一句话

在 token perpetual 市场：

```text
price move 不仅由方向决定
还由仓位结构决定
```

所以：

> **避免 crowded trade，比预测方向更重要。**

---

如果你愿意，我可以再讲一个更深的 microstructure 现象（很多策略 Sharpe 的真正来源）：

**为什么很多 crypto 策略的最佳 entry 其实在 “open interest 下降但价格上涨” 的阶段。**

这是很多专业 desk 会重点监控的一个结构信号。

# **为什么很多 crypto 策略的最佳 entry 其实在 “open interest 下降但价格上涨” 的阶段。**
在 crypto perpetual 市场里，“**价格上涨 + Open Interest 下降**”是一个非常有信息量的组合。很多量化 desk 会把它当作 **去杠杆后的趋势确认（deleveraging trend continuation）** 信号。

先用一句话总结核心逻辑：

> **价格上涨 + OI下降 = 空头被迫平仓，而不是新多头在追高。**
> 这通常意味着 **市场杠杆在降低，趋势更健康**。

下面把机制拆开。

---

# 1️⃣ 什么是 Open Interest

在永续合约市场（perpetual futures），Open Interest 表示：

```text
当前未平仓合约数量
```

当 OI 变化时，说明 **仓位结构在变化**：

| OI变化 | 含义       |
| ------ | ---------- |
| OI上升 | 新仓位进入 |
| OI下降 | 仓位被平掉 |

价格 + OI 的组合能告诉我们 **谁在推动价格**。

---

# 2️⃣ 四种价格 / OI 组合

| 价格 | OI  | 含义                         |
| ---- | --- | ---------------------------- |
| ↑    | ↑   | 新多头进场（趋势但可能拥挤） |
| ↑    | ↓   | 空头平仓（short covering）   |
| ↓    | ↑   | 新空头进场                   |
| ↓    | ↓   | 多头止损                     |

其中最有意思的是：

```text
price ↑
OI ↓
```

---

# 3️⃣ “price ↑ + OI ↓”发生了什么

这种情况通常意味着：

```text
short positions closing
```

路径一般是：

```text
price small rally
↓
short stop loss triggered
↓
forced buying
↓
price push higher
↓
OI decreases
```

因为空头平仓需要：

```text
buy to close
```

所以价格被推高，但 OI 下降。

---

# 4️⃣ 为什么这是健康趋势

如果价格上涨是 **新多头推动**：

```text
price ↑
OI ↑
```

那意味着：

```text
leverage building
```

问题是：

```text
future liquidation risk ↑
```

趋势可能不稳定。

---

但如果：

```text
price ↑
OI ↓
```

说明：

```text
weak hands leaving
```

市场正在：

```text
deleveraging
```

趋势变得更“干净”。

---

# 5️⃣ microstructure 解释

crypto 市场经常出现这种结构：

```text
crowded short
↓
small breakout
↓
short stops triggered
↓
OI drop
↓
price continuation
```

这时：

* 空头被挤掉
* 市场阻力减少

接下来价格更容易继续上涨。

---

# 6️⃣ 为什么这是好 entry

当你看到：

```text
price ↑
OI ↓
```

通常意味着：

```text
liquidation event just happened
```

而 liquidation 之后常见路径是：

```text
squeeze
↓
pullback
↓
trend continuation
```

如果在这个阶段 entry：

优势是：

* 市场杠杆降低
* squeeze 已经发生
* 剩余仓位更稳定

---

# 7️⃣ 反例：price ↑ + OI ↑

如果是：

```text
price ↑
OI ↑
```

通常路径更像：

```text
trend
↓
late longs enter
↓
funding spike
↓
liquidation dump
```

也就是 **crowded long**。

很多 breakout strategy 在这种环境 Sharpe 会下降。

---

# 8️⃣ 为什么量化策略喜欢这个信号

因为它同时捕捉了两件事：

### 1️⃣ 趋势存在

```text
price ↑
```

### 2️⃣ 杠杆风险下降

```text
OI ↓
```

所以 signal quality 更高。

很多 desk 会用类似逻辑：

```text
trend filter
+
OI delta filter
```

---

# 9️⃣ 一个常见的简单规则

很多策略会有类似条件：

```python
if price_return > threshold and OI_change < 0:
    long_signal = True
```

或者：

```text
price up
+
OI drop > 2%
```

作为 **trend confirmation**。

---

# 🔑 最重要的 microstructure 逻辑

crypto 市场的价格不仅由供需决定，还由 **杠杆结构**决定。

所以价格变化可以来自两种完全不同的力量：

```text
new buyers
or
short covering
```

而：

> **short covering rally 通常比新多头追涨更健康。**

因为它是在 **清理仓位，而不是堆积仓位**。

---

如果你愿意，我可以再讲一个很多 crypto quant 非常依赖的结构：

**为什么 “price 横盘但 OI 上升” 经常是下一次大波动的前兆（无论方向）。**

这是很多 desk 用来抓 **爆发行情（volatility expansion）** 的核心信号。

# 你上面说的是适合我15min的lv策略，还是bpc，fer，me都可以，他们是4h和1h的。另外，是不是entryfilter模块应该着重考虑你上面说的

你的问题其实是 **时间尺度匹配（time-scale alignment）** 的问题。
简化结论先说：

> **你上面提到的 microstructure 信号（funding、OI、liquidity sweep 等）主要适合放在 Entry / Execution 层，而不是 BPC / FER / ME 这种 4H–1H 的 Gate 层。**

但它们仍然可以辅助 Gate，只是作用不同。

下面拆开讲。

---

# 1️⃣ 先明确你现在的层级结构

你之前描述的系统大致是：

```text
4H / 1H
BPC / FER / ME
      ↓
Gate
      ↓
15m
Entry Filter / Execution
      ↓
Trade
```

也就是说：

* **BPC / FER / ME**：中周期结构判断
* **Entry Filter**：短周期 timing
* **Execution**：具体触发

这是一个非常合理的架构。

---

# 2️⃣ 为什么 microstructure 更适合 Entry 层

像我们刚才讲的：

* funding
* open interest delta
* liquidity sweep
* reclaim
* imbalance fill

这些信号的 **半衰期非常短**：

```text
几分钟 ～ 几小时
```

而你的 BPC / FER / ME 是：

```text
1H ～ 4H
```

如果你把 microstructure 放到 Gate 层，会出现问题：

```text
Gate 信号还没结束
microstructure 已经变化
```

比如：

```text
4H trend long
↓
1H breakout
↓
15m sweep
↓
trend continuation
```

如果 Gate 用 funding 或 OI：

很可能：

```text
Gate constantly flipping
```

稳定性反而变差。

---

# 3️⃣ microstructure 在 Gate 层的正确用法

它不适合做 **signal**，但适合做 **risk filter**。

例如：

```python
if funding > 0.05:
    reduce_position
```

或者：

```python
if OI_spike > threshold:
    skip_trade
```

作用是：

```text
避免 crowded trade
```

而不是预测方向。

---

# 4️⃣ Entry Filter 才是 microstructure 的核心位置

你现在的 Entry Filter 是最应该强化的模块。

因为 crypto 的核心结构是：

```text
signal (slow)
+
execution (fast)
```

理想结构是：

```text
BPC / FER / ME
↓
direction bias
↓
Entry Filter
↓
microstructure trigger
↓
trade
```

Entry Filter 可以包含：

### 1️⃣ liquidity sweep

```text
break level
↓
wick
↓
reclaim
```

### 2️⃣ imbalance pullback

```text
impulse
↓
pullback
↓
entry
```

### 3️⃣ OI / funding context

```text
avoid crowded entry
```

### 4️⃣ volatility regime

```text
avoid breakout during compression
```

---

# 5️⃣ 为什么 Entry Filter 对 Sharpe 影响最大

因为它直接影响：

```text
entry price
stop distance
fake breakout exposure
```

举个简单例子：

同一个 BPC 信号：

| entry方式       | winrate |
| --------------- | ------- |
| breakout market | 42%     |
| reclaim entry   | 53%     |

而且：

```text
stop size ↓
RR ↑
```

Sharpe 就会明显改善。

---

# 6️⃣ 一个很典型的 crypto quant 架构

很多 desk 的层级是：

```text
Slow layer (4H / 1H)
--------------------
trend / regime
position bias

Fast layer (5m / 15m)
---------------------
liquidity events
order flow
execution trigger
```

也就是说：

```text
slow layer 决定方向
fast layer 决定 entry
```

---

# 7️⃣ 对你系统的具体建议

你的系统其实已经接近正确结构了：

```text
BPC / FER / ME
↓
Gate
↓
Entry Filter
↓
Execution
```

但现在的问题是：

```text
Entry Filter 可能太弱
```

所以 Sharpe 主要来自 execution。

你可以重点强化 Entry Filter：

例如加入：

**microstructure-aware entry**

```text
1️⃣ sweep + reclaim
2️⃣ breakout pullback
3️⃣ OI drop continuation
4️⃣ avoid extreme funding
```

---

# 8️⃣ 一个非常关键的设计原则

不要让 microstructure 信号影响：

```text
是否做这个方向
```

而是影响：

```text
什么时候做
```

也就是：

```text
direction → slow features
timing → microstructure
```

---

# 最终总结

* **BPC / FER / ME（1H–4H）**
  → 决定市场结构 / 方向

* **Entry Filter（15m）**
  → 决定 timing

* **microstructure 信号**
  → 最适合放在 Entry Filter

所以：

> **是的，你的 Entry Filter 模块应该重点考虑我上面讲的那些 microstructure 结构。**

---

如果你愿意，我可以再帮你设计一个 **非常适合 15m execution 的 Entry Filter 结构（crypto desk 常见的那种）**，它通常能明显提高 Sharpe，而且实现也不复杂。

# 设计一个 非常适合 15m execution 的 Entry Filter 结构（crypto desk 常见的那种），它通常能明显提高 Sharpe
