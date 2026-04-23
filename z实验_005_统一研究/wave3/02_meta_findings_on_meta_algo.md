# 02 — 方法论核心结论

**本文是 Wave 3 最重要的产出**。下次任何人（包括你自己）想动 meta-algo、加 K-fold、改 label 语义、或者又觉得"慢管线应该能自动选特征"之前，先读这份。

---

## 1. 自由度 vs 样本：一切过拟合的源头

### 公式
```
slow 模式自由度 ≈ 候选特征数 × 分位数档数 × scoring 方法数 × rolling 窗口数
                ≈ 50 × 20 × 4 × 12
                ≈ 48,000 种潜在规则组合
```

### 数据侧
- 2H 频率，12 个月训练窗 ≈ 4,320 bar
- 去重（去掉 consecutive 相关 bar）后独立样本更少
- 每个分位数阈值估计需要至少 50 个样本才有统计意义
- 理论上能稳定支撑的规则数 ≈ 4,320 / 50 ≈ **86 条独立规则**

### 结论
**48,000 种候选 vs 86 条可稳定支撑** —— 差了 3 个数量级。任何让 meta-algo 在这个比值下"自动选最优"的机制都是过拟合。

### Wave 1+2 做了什么
封堵了若干自由度维度：

| Wave | 改动 | 削减的自由度 |
|---|---|---|
| Wave 1 | range-width guard (`min_range_width_sigma`) | 砍掉 "0.15 ≤ x ≤ 0.16" 这类过窄规则 |
| Wave 1 | `require_positive_effect` | 要求 Allow mean_rr > Deny mean_rr |
| Wave 1 | SHAP cutoff date | 防止特征选择用到 test 期 |
| Wave 2-A | [`features_gate.yaml`](../../config/strategies/) 白名单 | Gate 只能从 risk/volatility 特征里挑 |
| Wave 2-E | 单一 scoring method | 去掉 method-shopping 的自由度 |
| Top-N cap | `max_hard_gates=2, max_system_safety=2` | 每层规则数上限 |

**效果**：Wave 1+2 之后慢管线不再产出离谱规则（如窄区间、反主战场），但仍没有让 slow 可靠到能自动落地。Wave 3 想靠 label 改造从另一维度压缩自由度 → 证伪。

## 2. 核心结论：**任何留给 meta-algo 的决策维度都会过拟合**

### Wave 3 给出的证据
- Step 1（label）：表面看是 NO-OP，但实质是 "我们以为封堵了这个自由度，其实 scoring 根本不消费它"—— 说明我们对 meta-algo 的控制力有盲区
- Step 2（label）：收紧一种语义（tail-only）反而在另一个维度松动（Gate 阈值放宽）—— 自由度不是 isolated 的，压一头鼓一头

### 推论
如果把 slow 模式定位成 **decision mode（自动落地）**，必须把所有自由度维度全部封堵到 0 —— 这等价于 "完全锁死特征集 + 锁死 scoring + 锁死阈值"，那就不是 slow 模式了，就是 fast 模式。

所以 slow 模式 **只能做 candidate mode（候选发现）** —— 把自由度当成"多维度画像"的**资产**，而不是"自动决策"的**工具**。

详见 [04_slow_mode_redesign_candidate_discovery.md](04_slow_mode_redesign_candidate_discovery.md)。

## 3. Purged K-fold 为什么不适用金融生产

### Lopez de Prado 原意
Purged K-fold 解决两个泄漏：
1. **Label horizon leakage**：你的 label `y_t` 用了 `[t, t+h]` 的未来 bar → train 必须去掉 `[t-h, t]` 的样本
2. **Embargo**：val 之后 `[val_end, val_end+h]` 也要从 train 去掉

但它 **不解决** 第三种泄漏：

### 时序结构泄漏（他自己承认的局限）
- 金融时序有 regime clustering、vol clustering、trending persistence
- K-fold 允许 "用 Q2~Q4 训练预测 Q1" 这类 non-causal 切法
- 即使 label 不直接泄漏，训练集"见过 Q2 是趋势期"这个事实就偷看了未来 regime 属性
- 生产时不可能有这种信息

Lopez de Prado 在 *Advances in Financial Machine Learning* 结尾也承认：K-fold 适合**稳定性感知**（"这个规则在不同历史段都 ok 吗"），**不适合替代 walk-forward 做生产决策**。

### 对我们的场景
- 生产时我们只能用过去预测未来（causal walk-forward）
- K-fold 带来的 "K 次平均降方差" 收益，和 "偷看未来" 代价不对等
- 老管线 9mo causal Holdout val 已经够用（见 §4）
- 慢管线 rolling 本身就是 causal walk-forward，结构上已经比 K-fold 优

### 结论
**Wave 3 Step 5 永久搁置**。未来若要做"稳定性感知"，用严格 causal 版本的 "多窗口 walk-forward + 共识"，不用 K-fold。

## 4. 老管线 9mo causal Holdout val 足够

### 为什么"看过去够了"
- 老管线选出的特征是 "**在决策时点之前的数据上稳定的特征**"
- 它不对未来做承诺
- 未来 regime shift 到来时：
  - 轻度 shift → 快管线调阈值能跟上
  - 重度 shift → **策略本身失效**，应暂停/退役，**不是换特征能救**
- 周期匹配：老管线季度/半年 rebuild ≈ regime shift 的典型尺度

### 和 K-fold 的对比
| 维度 | 老管线 9mo Holdout | Purged K-fold |
|---|---|---|
| 样本利用 | 9mo 一次裁决 | K × 3mo 轮转平均 |
| Causality | ✅ 严格 | ❌ 非严格 |
| 和实盘一致 | ✅ | ❌ |
| 实现复杂度 | 低 | 中（purge zone + embargo） |
| 评估方差 | 中（1 次抽样） | 低（K 次平均） |
| 生产决策适用 | ✅ | ⚠️ 有争议 |

**选老管线 Holdout**，用"定期 rebuild"取代"内部 K 折平均"。

## 5. Method-shopping 的反转：毒药 or 资产取决于模式

### Decision mode 下是毒药（Wave 2-E 正确）
```python
# Wave 2-E 之前的逻辑（错误）
best_method = argmax over methods of val_score(method)
apply(rules_from(best_method))
```
问题：同一 val 被用来
- 挑 method（自由度 × method 数）
- 挑 rule（自由度 × rule 数）
- 裁决 winner

三层自由度叠加 → 典型的 "多重假设检验挑冠军" → 赢家是运气好的那个。

**Wave 2-E 固定单方法**（Prefilter=`distribution_ks`、EF=`upside_positive_rate_ratio`）是对的，砍掉 method 选择这一层自由度。

### Candidate mode 下是资产（未来方向）
```python
# 候选发现模式
for method in [distribution_ks, mean_effect, upside_positive_rate_ratio, tail_bad_rate_ratio]:
    candidates[method] = run_meta_algo(method)
# 不挑 winner，产出 per-method 并列报告 + 共识矩阵
```

不同 method 从不同角度画像：
- `distribution_ks`：分层能力
- `mean_effect`：正向 alpha
- `upside_positive_rate_ratio`：胜率型
- `tail_bad_rate_ratio`：风险过滤

**4 个 method 都选到 → 高置信度；只有 1 个选到 → 怀疑 overfit**。
共识矩阵的信息量 > 单方法 winner 的信息量。

具体设计见 [04_slow_mode_redesign_candidate_discovery.md](04_slow_mode_redesign_candidate_discovery.md)。

## 6. Label 语义不能自动化

### Wave 3 给出的直接证据
- `rr >= -0.8`（原）：学 "避开极端负 RR"
- `rr >= 0`（Step 1）：学 "正收益"
- tail-only q=0.05（Step 2）：学 "只 deny 极端不利"

三种 label 对应三种策略哲学，没有哪个天然 "对"。Step 2 FAIL 证明 **随便改 label 会崩盘**。

### 原则
Label 语义必须由人根据策略核心 invariant 选定：
- BPC 核心 invariant = "趋势中的 pullback + breakout + recovery 三段结构"
- 理论上最准的 label 应该是 "**这笔交易是否吃到三段结构的完整价值**"，而不是"forward RR 阈值"
- 这种 label 需要**人做语义工程**（pattern detection、segment labeling），不是 meta-algo 能自动发现的

### 实用建议
- 保持 `rr >= -0.8` 作为默认 label（baseline 语义：避开极端尾损）
- 不再试图通过改 label 来 "修复" 策略退化
- 策略退化要从 **execution 层（SL/TP/加仓）** 和 **Prefilter locked 阈值** 两个方向诊断

## 一句话总结

> **slow 模式的自由度 > 样本承载力 → 只能做候选发现器，不能做决策者**。
> K-fold 不能救它（看未来）、method-shopping 反转用法更好（共识 > 挑冠军）、label 语义必须人定。
> 老管线定骨架 + 快管线跟流动 + 慢管线发现候选 —— 这是唯一稳健的分工。

分工细节见 [03_three_pipeline_role_division.md](03_three_pipeline_role_division.md)。
