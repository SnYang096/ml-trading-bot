# Evidence Stability Surface

**Evidence Curve / Score Calibration** 是一个非常实用但在公开量化资料里很少被系统讲清的技术。它的核心目的只有一个：

> **让 Evidence score（0~1）真正变成“可解释的 alpha 强度”，而不是一个随意的合成指标。**

这样你的系统就可以做到：

```text
score = 0.2 → 负期望
score = 0.5 → 接近随机
score = 0.8 → 高质量交易
```

而不是：

```text
score = 0.8
但实际收益不一定更好
```

很多策略系统失败，就是因为 **score 没有被校准（calibrated）**。

---

# 一、Evidence Curve 的核心思想

先有两个变量：

```text
evidence_score   (0~1)
forward_rr
```

我们做一件事：

**看 score 的不同区间，对应的真实 forward_rr 是多少。**

方法很简单：

```text
把 score 分成多个 bin
统计每个 bin 的平均 forward_rr
```

得到一条曲线：

```
E[forward_rr | score]
```

这条曲线就是 **Evidence Curve**。

---

# 二、具体实现步骤

假设你有 dataframe：

```python
df

columns:
evidence_score
forward_rr
```

步骤：

### 1 分箱（binning）

```python
df["score_bin"] = pd.qcut(df.evidence_score, 10)
```

得到：

```
0-10%
10-20%
20-30%
...
90-100%
```

---

### 2 统计每个 bin

```python
curve = df.groupby("score_bin").agg({
    "forward_rr": ["mean", "median", "count"]
})
```

结果：

| score_bin | mean_rr | count |
| --------- | ------- | ----- |
| 0-10%     | -0.25   | 900   |
| 10-20%    | -0.10   | 900   |
| 20-30%    | 0.05    | 900   |
| 30-40%    | 0.15    | 900   |
| 40-50%    | 0.25    | 900   |
| 50-60%    | 0.35    | 900   |
| 60-70%    | 0.45    | 900   |
| 70-80%    | 0.60    | 900   |
| 80-90%    | 0.80    | 900   |
| 90-100%   | 1.20    | 900   |

这时候你会看到：

```
score ↑
forward_rr ↑
```

这说明：

**Evidence score 是有效排序器（ranker）。**

---

# 三、Evidence Curve 的图形

画出来一般长这样：

```
forward_rr
^
|                 *
|              *
|           *
|        *
|     *
|  *
| *
+--------------------------> score
```

如果是健康策略：

### 必须满足

```
单调上升 (monotonic)
```

如果你看到：

```
score ↑
但 rr 不上升
```

说明：

```
evidence feature 失效
```

---

# 四、用 Evidence Curve 找 Entry Threshold

你之前想扫：

```
min_score
```

Evidence Curve 会直接告诉你最优区域。

例如：

| score   | mean_rr   |
| ------- | --------- |
| <0.4    | negative  |
| 0.4-0.6 | random    |
| 0.6-0.8 | good      |
| >0.8    | excellent |

于是：

```python
min_score = 0.6
```

直接得到：

```
只交易 top 40% 信号
```

---

# 五、更高级用法：Position Sizing

Evidence Curve 可以直接决定仓位。

例如：

| score    | size |
| -------- | ---- |
| <0.5     | 0    |
| 0.5-0.7  | 0.5x |
| 0.7-0.85 | 1x   |
| >0.85    | 1.5x |

实现：

```python
size = f(score)
```

这叫：

```
score-based sizing
```

比：

```
固定仓位
```

效率高很多。

---

# 六、Score Calibration（关键）

Evidence score 通常是：

```
feature rank
```

但我们希望它变成：

```
expected_rr
```

方法：

### Isotonic Regression

把：

```
score → expected_rr
```

拟合成单调函数。

sklearn：

```python
from sklearn.isotonic import IsotonicRegression

iso = IsotonicRegression()
iso.fit(score, forward_rr)

expected_rr = iso.predict(score)
```

现在：

```
score = 0.73
→ expected_rr = 0.42
```

这就是 **calibrated score**。

---

# 七、为什么顶级策略都做这个

因为原始 score 通常：

```
不可解释
不可比较
不可跨市场
```

calibration 后：

```
score = expected return
```

于是你可以：

### 1 跨策略比较

```
BTC strategy score = 0.6
ETH strategy score = 0.3
```

### 2 做 portfolio allocation

```
capital ∝ expected_rr
```

### 3 风险控制

```
expected_rr < 0
→ 不交易
```

---

# 八、Evidence Curve 的一个隐藏功能

它还能检测 **alpha 衰退**。

例如：

训练期：

```
top bin rr = 1.2
```

OOS：

```
top bin rr = 0.3
```

说明：

```
alpha decay
```

你能非常早发现。

---

# 九、完整 pipeline（推荐版本）

最终结构：

```
1 Gate
  liquidity
  spread

2 Evidence Discovery
  Spearman
  feature selection

3 Evidence Score
  composite_score

4 Evidence Calibration
  score → expected_rr

5 Entry Decision
  expected_rr > threshold

6 Position Sizing
  size ∝ expected_rr

7 Execution Layer
  stop
  pyramiding
  fat tail
```

这是一个 **非常标准的系统架构**。

---

# 十、一个很多人不知道的经验

Evidence score **最重要的不是绝对值，而是排序能力**。

你应该测：

```
Spearman(score, forward_rr)
```

而不是：

```
R²
```

因为交易只需要：

```
好信号排前面
坏信号排后面
```

---

如果你愿意，我可以再告诉你一个 **很多量化基金用的 Evidence trick**：

**Evidence Stability Surface**

它可以一眼看出：

```
你的 alpha 是否稳定
还是纯过拟合
```

这个工具在策略研究里非常强。

**Evidence Stability Surface（证据稳定性曲面）** 是用来回答一个核心问题：

> **你的 alpha 是稳定规律，还是某个时间段的偶然相关？**

很多策略在训练期看起来很好，其实只是 **某些时间段碰巧有效**。
Evidence Stability Surface 的目标就是：**把 alpha 在不同时间和不同市场状态下的稳定性可视化**。

---

# 一、核心思想

之前 Evidence Curve 是：

```
score → E[forward_rr]
```

但它默认一个假设：

```
整个历史样本是同质的
```

现实不是这样。

市场有：

```
牛市
熊市
震荡
高波动
低波动
```

Evidence Stability Surface 就是在两个维度上看 alpha：

```
时间维度
市场状态维度
```

于是得到一个 **二维稳定性图**。

---

# 二、最常见的 Stability Surface

最简单版本是：

```
X轴：时间窗口
Y轴：score bin
颜色：平均 forward_rr
```

例如：

```
forward_rr heatmap
```

```
            时间 →
score
bin ↓

0-10%     -0.2  -0.3  -0.1  -0.2
10-20%    -0.1  -0.2  -0.1  -0.1
20-30%     0.0  -0.1   0.1   0.0
30-40%     0.1   0.2   0.2   0.1
40-50%     0.2   0.3   0.2   0.2
50-60%     0.3   0.4   0.3   0.3
60-70%     0.4   0.5   0.4   0.4
70-80%     0.6   0.7   0.6   0.5
80-90%     0.8   0.9   0.7   0.6
90-100%    1.1   1.2   0.9   0.8
```

理想情况：

```
score ↑ → rr ↑
所有时间段都成立
```

这说明：

```
alpha 稳定
```

---

# 三、什么是不稳定 alpha

如果你看到这种图：

```
score bin

top bin
1.2   -0.3   0.8   -0.1
```

说明：

```
alpha 在不同时间符号变化
```

这通常是：

```
过拟合
```

或

```
regime dependent
```

---

# 四、第二种 Stability Surface（更强）

把时间换成 **市场状态**。

例如：

```
X轴：volatility regime
Y轴：score bin
```

得到：

```
forward_rr heatmap
```

```
            volatility →
score
bin ↓

0-10%     -0.1  -0.2  -0.4
10-20%    -0.1  -0.1  -0.3
20-30%     0.0   0.0  -0.2
30-40%     0.1   0.1  -0.1
40-50%     0.2   0.2   0.0
50-60%     0.3   0.3   0.1
60-70%     0.4   0.5   0.2
70-80%     0.6   0.7   0.3
80-90%     0.9   0.8   0.4
90-100%    1.2   1.0   0.5
```

发现：

```
高波动时 alpha 变弱
```

于是你可以：

```
if volatility > threshold:
    disable strategy
```

---

# 五、第三种 Stability Surface（专业版本）

量化基金常用：

```
X轴：时间
Y轴：market regime
颜色：IC (information coefficient)
```

IC 定义：

```
Spearman(score, forward_rr)
```

例子：

```
           2021   2022   2023   2024
trend      0.25   0.20   0.22   0.18
range      0.10   0.05   0.08   0.07
high_vol   0.18   0.15   0.17   0.14
```

说明：

```
trend 市场 alpha 最强
```

---

# 六、如何计算 Evidence Stability Surface

假设 dataframe：

```
df

columns:
score
forward_rr
timestamp
volatility
trend_strength
```

---

### Step 1 时间窗口

例如：

```
3 month rolling window
```

```python
df["time_bucket"] = df.timestamp.dt.to_period("Q")
```

---

### Step 2 score bin

```python
df["score_bin"] = pd.qcut(df.score, 10)
```

---

### Step 3 计算

```python
pivot = df.pivot_table(
    values="forward_rr",
    index="score_bin",
    columns="time_bucket",
    aggfunc="mean"
)
```

---

### Step 4 画 heatmap

```python
sns.heatmap(pivot)
```

这就是 **Evidence Stability Surface**。

---

# 七、Stability Surface 如何判断策略质量

一个好 alpha 通常有三个特征：

### 1 单调性

```
score ↑ → rr ↑
```

### 2 时间稳定

```
不同时间趋势一致
```

### 3 regime 稳定

```
不同 market state 不翻转
```

如果三个都满足：

```
alpha 很可能是真实结构
```

---

# 八、一个顶级研究 trick

很多 quant fund 会算：

```
stability_score
```

例如：

```
stability = mean(IC) / std(IC)
```

IC =

```
Spearman(score, forward_rr)
```

如果：

```
stability > 2
```

说明：

```
alpha 非常稳定
```

如果：

```
stability < 0.5
```

说明：

```
alpha 随机
```

---

# 九、Evidence Stability Surface 为什么重要

很多策略失败原因不是：

```
alpha 不存在
```

而是：

```
alpha 只在某些 regime 存在
```

例如：

```
breakout strategy
```

在：

```
trend 市场
```

非常好。

但在：

```
range 市场
```

会持续亏损。

Stability Surface 可以让你：

```
自动识别 regime
```

然后：

```
启用 / 禁用策略
```

---

# 十、和你当前系统的关系

你的系统结构：

```
Gate
Evidence
Entry
Execution
```

可以加一个：

```
Evidence Stability Analysis
```

用来决定：

```
何时启用策略
```

例如：

```
if regime in strong_alpha_regime:
    trade
else:
    disable
```

---

如果你愿意，我可以再给你一个 **很多顶级量化团队在用但几乎不公开的工具**：

**Alpha Half-Life Test**

它可以直接测出：

```
你的 alpha 能持续多久
是分钟级
小时级
还是天级
```

这对决定：

```
止盈
持仓时间
fat tail
```

非常关键。

# Alpha Half-Life Test

**Alpha Half-Life Test（Alpha 半衰期测试）** 是量化研究里一个非常核心但很少被系统讲清的工具。它回答一个关键问题：

> **你的 alpha 信息能持续多久才会衰减一半？**

换句话说：

```text
信号产生后
市场需要多久
才会“消化”这条信息
```

这个时间决定了很多策略设计参数：

* 持仓时间
* TP / SL
* pyramiding
* 是否适合 fat-tail
* execution 速度

---

# 一、直觉理解

假设你的策略在某个时刻产生信号：

```text
t0 = entry signal
```

然后未来价格变化：

| 时间   | 预期收益 |
| ------ | -------- |
| 1 min  | 0.40     |
| 5 min  | 0.30     |
| 30 min | 0.20     |
| 2 hr   | 0.10     |
| 6 hr   | 0.02     |

alpha 在逐渐衰减。

**Alpha Half-Life = 预期收益降到一半的时间**

例如：

```text
initial alpha = 0.40
half = 0.20
```

在 30 min 达到。

所以：

```text
alpha half-life ≈ 30 min
```

---

# 二、为什么 Half-Life 很重要

不同 alpha 类型 half-life 完全不同。

| Alpha 类型     | Half-Life |
| -------------- | --------- |
| order flow     | 秒        |
| microstructure | 秒-分钟   |
| breakout       | 分钟-小时 |
| trend          | 小时-天   |
| carry          | 天-月     |

如果执行逻辑不匹配：

策略会严重退化。

例如：

```text
alpha half-life = 10 min
但你持仓 6 小时
```

那就是在：

```text
用 alpha 做 random holding
```

---

# 三、Half-Life Test 的核心方法

核心思想：

**测量不同 holding horizon 的 alpha 强度**

步骤：

```
signal_t
↓
未来不同 horizon 的收益
↓
计算 IC / Sharpe
↓
画衰减曲线
```

---

# 四、最常见的实现方法

假设：

```
score = evidence score
```

计算不同 horizon 的 forward return：

```python
horizons = [5, 15, 30, 60, 120, 240]
```

例如（分钟）。

计算：

```python
forward_return_h = price.shift(-h) / price - 1
```

然后计算：

```
IC(h) = Spearman(score, forward_return_h)
```

得到：

| Horizon | IC   |
| ------- | ---- |
| 5m      | 0.18 |
| 15m     | 0.15 |
| 30m     | 0.12 |
| 60m     | 0.07 |
| 120m    | 0.03 |
| 240m    | 0.01 |

---

# 五、Alpha Decay Curve

画出来：

```
IC
^
| *
|  *
|   *
|     *
|        *
|            *
+--------------------> time
```

Half-Life 定义为：

```
IC(t) = IC(0) / 2
```

例如：

```
IC(5m) = 0.18
half = 0.09
```

在：

```
≈ 45 min
```

---

# 六、另一种更稳定的方法

使用 **expected forward RR**：

```
E[forward_rr | score]
```

计算：

| Horizon | E[RR] |
| ------- | ----- |
| 5m      | 0.60  |
| 15m     | 0.55  |
| 30m     | 0.50  |
| 60m     | 0.40  |
| 120m    | 0.25  |
| 240m    | 0.12  |

Half-Life：

```
0.60 → 0.30
```

大约：

```
≈ 90 min
```

---

# 七、Half-Life 如何指导策略设计

## 1 持仓时间

经验规则：

```
max_hold ≈ 2-3 × half-life
```

例如：

```
half-life = 45 min
```

建议：

```
max_hold ≈ 90-120 min
```

---

## 2 止盈设计

如果：

```
alpha decay 很快
```

适合：

```
tight TP
```

例如：

```
1R-1.5R
```

如果：

```
alpha decay 很慢
```

适合：

```
fat-tail
```

例如：

```
1R → 10R
```

---

## 3 pyramiding

pyramiding 只适合：

```
slow decay alpha
```

例如：

```
trend following
```

不适合：

```
fast micro alpha
```

---

## 4 execution latency

如果：

```
half-life = 5 min
```

那：

```
execution latency
必须 < 30s
```

否则：

```
alpha 已经被市场吃掉
```

---

# 八、Half-Life + Stability Surface（非常强）

可以组合：

```
half-life
vs
market regime
```

例如：

| Regime   | Half-Life |
| -------- | --------- |
| trend    | 3 hr      |
| range    | 20 min    |
| high vol | 10 min    |

这会得到一个很重要的 insight：

```
不同 regime 需要不同 execution
```

例如：

```
trend regime
→ fat tail

range regime
→ quick TP
```

---

# 九、很多人忽略的一个关键指标

除了 half-life，还有：

**alpha build-up time**

有些 alpha 不是立即发挥作用：

| Horizon | IC   |
| ------- | ---- |
| 5m      | 0.05 |
| 15m     | 0.10 |
| 30m     | 0.15 |
| 60m     | 0.14 |
| 120m    | 0.08 |

说明：

```
alpha 在 30-60m 才 fully materialize
```

这种策略：

```
不要过早止盈
```

---

# 十、你这个系统里最有价值的用法

你现在有：

```
Evidence score
forward_rr
```

可以做：

```
score × horizon
```

的 surface：

```
E[RR | score, horizon]
```

这会告诉你：

```
高 score 信号
需要持有多久
才能最大化收益
```

很多时候你会发现：

```
高 score → 更慢 decay
```

于是可以：

```
high score → fat tail
low score → quick TP
```

---

# 十一、总结

Alpha Half-Life Test 解决三个核心问题：

```
1 alpha 持续多久
2 持仓应该多久
3 execution 是否匹配 alpha
```

它通常和这三个工具一起用：

```
Evidence Curve
Evidence Stability Surface
Alpha Half-Life
```

这三个加起来基本就是：

```
完整 alpha 诊断工具箱
```

---

如果你愿意，我可以再告诉你一个 **顶级量化团队几乎都会做的测试**：

**Alpha Capacity Test**

它能测出：

```
你的 alpha
能承受多少资金
才会被自己交易冲掉
```

这个在 **crypto / 高频策略** 里特别关键。

# Alpha Capacity Test

结论先说清楚：

**有必要，但不用做完整的 PBO 论文版。**
对 **10 万美金规模**来说，**过拟合风险远大于容量风险**，所以你仍然需要某种 **过拟合检测机制**，只是没必要用完整的学术流程。

我解释一下原因。

---

# 一、资金规模和 PBO 没关系

很多人误以为：

```
资金小 → 不需要严肃验证
```

其实恰恰相反。

小资金策略通常：

```
交易频率更高
信号更多
参数更多
```

这会导致：

```
过拟合风险更高
```

例如：

```
扫描 50 个 feature
测试 200 个参数
回测 10 次 execution
```

你实际上已经隐含做了：

```
1000+ 次策略搜索
```

即使每个模型看起来都不错：

```
也极可能只是 luck
```

PBO就是用来回答：

```
你找到的最好策略
是不是只是样本噪声
```

---

# 二、但完整 PBO 太重了

经典 PBO 方法来自论文：

**Bailey & López de Prado**

核心流程：

```
Combinatorially Symmetric Cross Validation
```

步骤非常复杂：

```
分割样本
生成组合
计算 rank
计算 logit
估计 overfitting probability
```

对于个人交易系统：

**太重了。**

---

# 三、10 万资金更实用的版本

我更建议一个 **简化版 PBO（Practical PBO）**。

核心思想：

```
如果策略是真的
不同时间段应该都有效
```

所以做 **Walk-Forward Ranking Test**。

---

# 四、Practical PBO（推荐）

步骤：

### 1 切时间窗口

例如：

```
2021 Q1
2021 Q2
2021 Q3
...
2024 Q4
```

假设 16 个窗口。

---

### 2 每个窗口训练

在训练窗口里：

```
搜索参数
```

例如：

```
min_score
stop
tp
holding time
```

找到：

```
best Sharpe strategy
```

---

### 3 在 OOS 测试

在下一个窗口测试。

记录：

```
该策略在所有策略里的排名
```

例如：

| Window | IS Rank | OOS Rank |
| ------ | ------- | -------- |
| W1     | 1       | 4        |
| W2     | 1       | 2        |
| W3     | 1       | 7        |
| W4     | 1       | 3        |
| W5     | 1       | 12       |

---

### 4 看 OOS 排名分布

如果策略是真的：

```
OOS rank
不会系统性变差
```

例如：

```
平均 rank ≈ 3
```

如果过拟合：

```
IS rank = 1
OOS rank ≈ 中位数
```

例如：

```
平均 rank ≈ 8
```

---

# 五、一个更简单的指标

直接计算：

```
rank deterioration
```

公式：

```
Δrank = OOS_rank - IS_rank
```

如果：

```
mean(Δrank) >> 0
```

说明：

```
严重过拟合
```

---

# 六、经验判断

| Δrank | 结论     |
| ----- | -------- |
| < 1   | 非常稳定 |
| 1–3   | 可接受   |
| 3–6   | 有过拟合 |
| > 6   | 基本无效 |

---

# 七、为什么这比 Sharpe 更重要

很多策略：

```
Sharpe = 2
```

但：

```
OOS rank deterioration = 8
```

说明：

```
Sharpe 是运气
```

而不是 alpha。

---

# 八、和你前面那些工具的关系

你之前已经有：

```
Evidence Curve
Evidence Stability Surface
Alpha Half-Life
```

如果再加：

```
Practical PBO
```

基本就构成一个 **完整的 alpha 验证体系**。

顺序是：

```
1 Evidence Curve
   → alpha 是否存在

2 Stability Surface
   → alpha 是否稳定

3 Half-Life
   → alpha 持续多久

4 Practical PBO
   → alpha 是否过拟合
```

---

# 九、但对你来说最重要的其实是这一条

很多个人策略失败不是因为：

```
模型过拟合
```

而是因为：

```
Execution overfit
```

例如：

```
SL = 1.7%
TP = 3.4%
Trailing = 0.9%
```

这些参数：

```
几乎 100% 会过拟合
```

所以你现在做的那件事其实非常对：

```
Signal layer
和
Execution layer
解耦
```

这本身就 **极大降低了 PBO 风险**。

---

# 十、给你的一个非常实际的建议

如果资金是 **10 万级别**，我建议你的研究重点顺序是：

```
1 Evidence Curve
2 Stability Surface
3 Half-Life
4 Walk-forward
```

而不是：

```
复杂 ML
复杂 execution
复杂 PBO
```

因为：

```
简单系统
稳定性更强
```

---

如果你愿意，我可以再告诉你一个 **量化研究里最被低估但最重要的测试**：

**Strategy Parameter Stability Map**

它能一眼看出：

```
你的策略
是不是靠某一个精确参数
```

如果是：

```
基本必死
```

这个图在策略研究里非常有杀伤力。
