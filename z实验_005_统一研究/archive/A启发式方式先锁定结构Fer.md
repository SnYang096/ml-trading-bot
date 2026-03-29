# 1. 有没有可能1h更好，我把bpc，me，fer都改成1h级别 2. 对fer，fer_signed_efficiency 能让开仓靠近支撑阻力吗，为什么6,3配置靠近sr，12,6和me类似，直接跟着波动率爆炸方向走了，是否应该用 fer_signed_efficiency  sr_strength_max_f 等限定开仓在sr附近更合理，这样有可能在12,6配置下也能找到合适的参数？

这两个问题都很关键，结论先说：

“全部改 1H”不建议一次性做
FER 用 fer_signed_efficiency + SR 位置约束 是对的，确实有机会让 12,6 更稳
1) 1H 会不会更好？
有可能对 FER 更好，但不建议把 BPC/ME/FER 一起全改 1H。
原因是 1H 不是“更多样本”这么简单，而是策略语义变化：

BPC：偏趋势/结构，4H 往往更干净；1H 容易噪音化
FER：偏失败反转与微结构，1H 可能更有信息密度
ME：中间态，得看执行与标签稳定性
更好的做法是：分策略做 timeframe A/B，先从 FER 开 1H 分支，不动 BPC。

2) fer_signed_efficiency 能否让开仓靠近 SR？
fer_signed_efficiency 本身更像“失败效率/方向语义”，不是位置特征。
“是否靠近支撑阻力”更直接由这些特征决定：

sr_strength_max
dist_to_nearest_sr（如果在该特征组输出里）
你观察到 12,6 更像跟随波动率爆炸方向，通常是因为：

prefilter 在长窗口下被稀释/空化；
gate fallback 更偏波动/尾部特征，缺少“必须靠近 SR”的语义锚。
所以你提的方案是对的：
把 FER 语义锚定为“失败效率 + SR 附近”，能提升跨 regime 稳定性。

可执行的约束建议（FER）
在 prefilter 阶段引入“必选约束”（可 locked）：

sr_strength_max <= T1（要求接近关键位）
dist_to_nearest_sr <= T2（如果可用）
fer_signed_efficiency 在合理区间（保留你现在的失败语义带）
这样可以减少“远离 SR 的追涨杀跌型信号”，对 12,6 特别有帮助。

如果你愿意，我下一步可以给你一份最小实验矩阵（FER: 4H vs 1H、6/3 vs 12/6、是否加 SR 锁定）和清晰的判定标准，直接照跑即可。

# sr_strength_max  dist_to_nearest_sr  fer_signed_efficiency 可以启发式的先人为定义吗？他们是归一化的pct吗？

可以，完全可以先启发式人为定义，而且我建议先这么做。

这三个量纲分别是：

sr_strength_max
是归一化强度，范围约 [0,1]（不是 pct 排名）
dist_to_nearest_sr
是ATR 归一化后的距离倍数（大致 [-3,3] 常见），不是 0~1 百分位
绝对值越小越靠近 SR
fer_signed_efficiency
是 ΔPrice / ΔCVD 的有符号原值（不是 pct）
对应的百分位版本是 fer_signed_efficiency_pct（0~1）
给你一组 FER-short 可先试的启发式约束（prefilter）：

rules:
  - feature: dist_to_nearest_sr
    operator: "<="
    value: 1.2
  - feature: dist_to_nearest_sr
    operator: ">="
    value: -1.2
  - feature: sr_strength_max
    operator: ">="
    value: 0.55
  - feature: fer_signed_efficiency
    operator: "<="
    value: -0.00005
含义：

必须在 SR 附近（|dist_to_nearest_sr| <= 1.2 ATR）
SR 强度不能太弱（>= 0.55）
失败效率偏负（更贴近“失败后反转”语义）
如果你担心阈值太硬，先放宽到 ±1.5、sr_strength_max>=0.45，再收紧。
这类“先人为锚定语义，再让统计法微调”的方式，比纯自动搜索在 12,6 下更抗 regime shift。

# 如果我认为加载@config/strategies/fer-short/archetypes/prefilter.yaml 里面，会被自动化工具覆盖吗，还是有lock机制

会被覆盖，默认是会的。

你这个文件 config/strategies/fer-short/archetypes/prefilter.yaml 在 pipeline 里属于自动产物：

analyze_archetype_feature_stratification.py --promote 会重写它
后续 ADOPT 也会把实验目录里的 prefilter.yaml 复制回生产目录
所以如果你手工改了但不加保护，下一轮大概率被覆盖。

有 lock 机制，但要注意当前实现语义：

你可以在生产 prefilter.yaml 规则上加 locked: true
ADOPT 前有“语义锁定校验”：会检查实验产物是否仍保留这些 locked 规则的核心特征，不满足会拒绝采纳
但这不是“文件级永不覆盖”，而是“采纳门禁保护语义不丢”
所以它是软保护（校验），不是硬保护（完全不改写）。

实操建议：

你要先启发式锁结构时，在 prefilter.yaml 里给关键规则加 locked: true + lock_reason
然后跑 pipeline，若新实验丢掉这些语义，会在 adopt 阶段被挡住。
