# sc-shallow

结论先行（不绕）

按你现在的 BPC 定义：
SC ≠ BPC，小回撤 SC 并没有被 BPC 覆盖。

## 一、我们先“回忆”你前面真实的 BPC 定义（不是抽象 SC）

你前面给过非常关键的信息（我按你原话语义复述）：

BPC entry filters 都要求
was_in_pullback == 1

而且 base filter 是
pullback_depth >= 0.55

候选池大概是 ~1200 个 pullback bar

深回撤 + 质量（bb_compression / wpt / cvd 等）

这意味着什么？
这不是泛指 pullback。

👉 这是一个非常明确的定义：

BPC =
已发生 breakout →
明确的、可度量的、较深的回撤 →
再继续

## 二、小回撤 SC 在“语义上”和 BPC 是不同的东西

现在我们把你担心的那个「小回撤」摊开说。

小回撤 SC 的典型特征是：

pullback_depth 很小（比如 0.15～0.35）

甚至只是：

1–2 根 inside bar

微弱的 time-based pullback

价格结构没破

动能没真正释放

👉 它的核心假设是：
“强趋势不需要深回撤”

而你现在的 BPC 假设是：

“趋势延续前，需要一次可验证的换手 / 压缩”

这两个不是一个假设。

# 小回撤sc是不是只需要改变bpc定义就能做 重新跑一下流程即可，怎么命名它呢，如果他和me，bpc同时出现 ，优先me ，再是它 ，然后bpc吗

先给结论（很重要）

小回撤 SC 不是“稍微改一下 BPC 定义就完事了”的东西，
但它也不需要一整套全新流程。

更准确地说是：

它是“BPC 的兄弟 archetype”，
不是 BPC 的参数变体。

所以：

❌ 不建议叫 BPC-lite / BPC-v2

✅ 应该是 独立命名、独立 Gate、共享部分基础设施

## 一、为什么「只改 BPC 定义」会很危险（这是关键）

你现在的 BPC 定义，核心不是某个阈值，而是世界假设：

你当前 BPC 的隐含假设是：

breakout 已发生

需要一次明确换手

换手体现在：

深回撤

波动压缩

小波动能未释放

然后再 continuation

## 👉 “需要换手”是 BPC 的灵魂。

小回撤 SC 的假设恰好相反

小回撤 SC 的世界观是：

breakout 已发生

趋势强到不需要换手

pullback 是：

time-based

shallow

often overlapping bars

continuation 靠的是：

持仓者不卖

追单者继续

## 👉 它赚的是 “没给你上车机会”的钱。

如果你只是：

把 pullback_depth >= 0.55
改成 >= 0.25

那你做的是：

用“需要换手”的 Gate，
去交易“不发生换手”的世界

这不是扩展，是语义错配，
这才是真正的 archetype 塌缩。

## 二、那正确的做法是什么？（很克制）
✅ 正确做法：

小回撤 SC = 新 archetype，但复用 60% 管线

你可以复用的部分包括：

breakout 检测

trend context

execution / risk / pcm

system safety（流动性、反身性）

你必须独立定义的只有三块：

Pullback 定义

Gate（反证）

失败模式

## 三、小回撤 SC 的 pullback 定义，和 BPC 有“本质差异”

我不给你公式，只给你原则（因为你已经会实现）：

BPC pullback 是：

price retrace-based

depth-driven

volume / vol compression driven

小回撤 SC pullback 应该是：

time-based / overlap-based

retrace 很小，甚至 0

重点看：

overlap ratio

progress slowdown

没有对向 CVD

一句话：

BPC 看“回来了多少”，
小回撤 SC 看“走慢了多久”。


## 四、命名问题：千万别把它叫成 BPC 的变体

命名不是形式问题，是防止你未来自我欺骗。

我给你几个不会塌缩的命名建议：

推荐命名（按语义强度）

SC_shallow

TrendPause

LowRetraceSC

TimePullbackSC

不推荐：

BPC_light

BPC_shallow

BPC_v2

👉 后者会逼着你复用 BPC Gate，这是陷阱。


## 五、你提出的优先级顺序——是否合理？

你说的是：

如果 ME / 小回撤 SC / BPC 同时出现
👉 优先 ME → 再小回撤 SC → 再 BPC

这个顺序是完全合理的，而且非常干净

为什么？

1️⃣ ME > SC_shallow

因为：

一旦是 ME，

continuation 已经不是“结构延续”，而是“加速”

2️⃣ SC_shallow > BPC

因为：

在同一根 bar 上：

若 shallow 成立

说明 市场明确拒绝深回撤

那 BPC 的世界假设已被否定

👉 这是逻辑否定，不是收益偏好。


## Archetype 的边界，
不是由参数决定的，
而是由“失败叙事是否不同”决定的。

BPC 的失败是：

换手不成立

压缩是假象

小回撤 SC 的失败是：

动能枯竭

持仓者开始卖

这已经是两个世界了。

# 我感觉bpc scshallow me是趋势的三个阶段 多一个更能跟上趋势 是吧 但我slot就两个 应该也不会扩大很多风险 加是不是比不加更能抓住机会 我的整个pipeline是可以复用的

给结论

BPC / SC-shallow / ME 本来就是同一条趋势在不同“能量状态”的三个切片。

在你只有两个 slot、用强 Gate、pipeline 可复用的前提下：
👉 加 SC-shallow 的系统风险 < 不加它的机会成本。

而且重点是：

你不是“多做一种交易”，
你是在补全一条趋势生命周期。


## 一、你现在的三段，其实已经天然排好了序

你直觉说它们是三个阶段，这个不是感觉，是市场机制决定的。

可以这样看：

趋势能量低 → 中 → 高
────────────────────
BPC       → SC-shallow → ME
（换手）     （不换手）     （加速）

1️⃣ BPC ——「趋势刚被重新分配」

需要换手

市场在问：
“之前那批人走了吗？新的接得住吗？”

节奏慢、结构感强

2️⃣ SC-shallow ——「趋势被确认但还没爆」

不再给深回撤

价格走，但走得有点憋

这是：

最容易错过

也是最“该有”的一段

3️⃣ ME ——「钱已经踩油门」

不纠结结构

动能 + 波动率说话

再不进就追不上了

👉 这三段是一条连续谱，不是三个 unrelated idea。


## 二、slot 只有两个，反而是你的优势

你说你 slot 只有两个，这一点非常重要。

这意味着：

你 不可能同时持有：

BPC

SC-shallow

ME

系统被迫做选择

这天然就变成了一个简化 router，而且是规则型、可解释的。

你现在的隐含 router 已经是：

if ME → 用 ME
else if SC-shallow → 用它
else if BPC → 用 BPC
else → 不做


👉 slot 限制 = 风险上限器，不是负担。

三、加 SC-shallow 会不会扩大风险？

关键问题来了。

答案是：不会显著扩大，前提你已经满足了

你已经满足的条件：

✅ 强 Gate（反证法）

✅ Archetype 正交（失败叙事不同）

✅ pipeline 复用

✅ slot 有上限

✅ 有优先级（不会叠加）

在这种情况下，加一个 archetype 的风险来源只剩一个：

它会不会在“本来该不交易”的时候误触发？

而 SC-shallow 的触发区域是：

BPC 已经 deny（没给深回撤）

ME 尚未成立（没加速）

👉 这是你现在系统的盲区。

## 四、不加它，你在结构上是“有洞的”

你现在的系统，其实长这样（诚实一点）：

趋势 → 有深回撤？──是──▶ BPC
        │
        否
        │
        有加速？──是──▶ ME
        │
        否
        │
       空白（你现在在这不交易）


而市场非常爱在这个“否 / 否”的区域里：

价格慢慢推

流动性不枯

BB 不压

所有人都在等回撤，但它就是不给

👉 这不是噪音区，这是健康趋势的中段。

## 最后一句定锚

你不是在增加复杂度，
你是在把趋势这件事“做完整”。