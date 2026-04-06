👉 dashboard、压力测试、原语训练，其实是同一套世界观的三个接口。

一、Live Dashboard：只盯 5 个数（多一个都是噪声）

原则先说清楚一句话：
Dashboard 不是给你“做决定”的，是给你“阻止你犯蠢”的。

❌ Dashboard 不应该显示

信号强度

单笔盈亏

指标曲线

“今天赚不赚”

✅ Dashboard 只回答 5 个问题：

系统现在是不是在“健康博弈区”？

✅ 你真正需要盯的 5 个数（工业级）
① Tradeability Score（0–1）

你现在是否应该参与市场？

tradeability_score


来源：

Hurst

Spectrum entropy

Hilbert phase stability

WPT energy concentration

解读规则：

< 0.4 → 今天市场不配合（别硬上）

0.4 ~ 0.7 → 轻仓、只做最强 archetype

> 0.7 → 正常运行

👉 这是 dashboard 的“红绿灯”

② Archetype Entropy（结构清晰度）

市场现在像不像“一个确定的博弈”？

H(archetype | last N bars)


低熵：结构稳定（好）

高熵：来回切换（极易被打脸）

经验阈值：

> 0.8 → 强制降速

> 1.0 → freeze

👉 这是“结构混乱报警器”

③ Execution Failure Rate（不是盈亏）

你是不是“一进去就错”？

mean(MAE / ATR, rolling 20 trades)


不看 PnL

不看胜率

只看：入场是否合理

解读：

> 0.6 → 执行层正在失效

连续升高 → archetype 已不适用

👉 这是“因果失效报警器”

④ OOD Drift Score（市场变没变）

现在的市场，还是你“认识的那个市场”吗？

feature_drift_score


计算：

fp_scene

trade_cluster_semantic

liquidity_void_scene
与训练分布的距离（Wasserstein / KL）

👉 这是“世界是否换了一套规则”的检测器

⑤ Risk Utilization Ratio（你暴露了多少风险）
current_risk / allowed_risk


不是仓位，不是杠杆，是结构性风险暴露。

> 0.7 → 禁止新单

> 0.9 → 只能减仓

👉 这是最后一道保险丝

📌 总结一句话

如果这 5 个数都没报警，
你根本不需要看别的。

---

二、增强版（同一套世界观的“可落地版”）

> 依然坚持“只看少数关键数”，只是把它们变成 **可执行/可回测/可审计** 的接口输出。

✅ 建议增强为这 5 个（和基础版一一对应）：

1) OOD Score（全局，0~1）
   - 解释：当前市场有多“分布外”
   - 用法：超过阈值 -> 触发 kill / 降速；低于更严格阈值 -> 分阶段 revive

2) Top Archetype Survival Prob（0~1）
   - 解释：在当前世界里，最可能“活得久”的 archetype 的生存概率（来自 Survival Head）
   - 用法：用于选择 archetype / 给 size cap 上限

3) Active Archetype + Router Confidence
   - 解释：当前实际启用的 archetype（TC/TE/FR/ET）以及 Router 对当前 regime 的置信度
   - 用法：解释“今天在做什么”，而不是“今天赚不赚”

4) Size Cap（当前）
   - 解释：当前允许的最大风险暴露上限（来自 OOD/Survival/Constitution 的合成）
   - 用法：这是“最后一道保险丝”更细的版本

5) Kill-switch State（运行态）
   - 解释：是否处于 freeze / degrade / revive phase1/2/3
   - 用法：让系统行为可审计（为什么不下单 / 为什么只小仓位）