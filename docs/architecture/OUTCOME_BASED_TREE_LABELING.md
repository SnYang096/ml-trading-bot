# Outcome-Based Tree Labeling 方案

> **核心哲学**：用不带偏见的 Outcome-Based 标签发现真实有效条件，抽取可解释规则，只把"结构上说得通、统计上站得住"的部分写回 archetypes 作为 veto/guardrail。

---

## 一、问题背景：为什么传统 Label 方案有循环论证风险

### 传统做法（存在偏差）

```
规则 → 筛选样本 → 打标签 → 训练 → 验证规则
     ↑_____________________↓
          （自己证明自己对）
```

### 本方案（反证驱动）

```
全样本（无预设）
     ↓
假设做多/做空 → forward_rr（纯未来结果）
     ↓
浅树学习 → 导出"负规则"（系统性亏损区）
     ↓
映射到 archetype gate → veto / 禁区
```

---

## 二、数据与标签设计

### 2.1 数据集构造（关键）

```python
# Long Dataset：所有样本假设做多
y_long = forward_rr_long

# Short Dataset：所有样本假设做空
y_short = forward_rr_short
```

**铁律**：
- ❌ 不预筛 direction
- ❌ 不套 archetype gate
- ✅ 全样本假设「如果我做了，会怎样」

### 2.2 forward_rr 定义（路径极值型）

```python
def compute_forward_rr_label(
    df: pd.DataFrame,
    direction: str,  # 'long' or 'short'
    horizon: int = 50,
    sl_r: float = 1.0,  # 止损基准单位（如 1 ATR）
    atr_col: str = 'atr',
) -> pd.Series:
    """
    纯 Outcome-Based 标签：不依赖任何当前特征筛选
    """
    labels = np.full(len(df), np.nan)
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    atr = df[atr_col].values
    
    for i in range(len(df) - horizon):
        entry_price = close[i]
        risk_unit = atr[i] * sl_r
        
        if risk_unit <= 0:
            continue
        
        future_high = high[i+1 : i+1+horizon]
        future_low = low[i+1 : i+1+horizon]
        
        if direction == 'long':
            mfe = np.max(future_high) - entry_price
            mae = entry_price - np.min(future_low)
            labels[i] = (mfe - mae) / risk_unit
        else:  # short
            mfe = entry_price - np.min(future_low)
            mae = np.max(future_high) - entry_price
            labels[i] = (mfe - mae) / risk_unit
    
    return pd.Series(labels, index=df.index, name=f'forward_rr_{direction}')
```

### 2.3 label_meta 元信息（强制规范）

```yaml
label_meta:
  rr_type: path_extreme        # vs. barrier_based / time_exit
  horizon: 50
  exit_model: none             # vs. tp_sl_barrier / trailing
  note: "用于禁区挖掘，非实盘 RR"
```

**重要**：`path_extreme` 类型的 RR 只用于"禁区识别"，不能直接用于实盘执行评估。

---

## 三、树模型配置（为规则导出设计）

### 3.1 模型设定

```yaml
model:
  type: lightgbm
  objective: regression
  max_depth: 3          # 绝不超过 4
  num_leaves: 8
  min_data_in_leaf: 500 # 强制统计稳定
  learning_rate: 0.05
  n_estimators: 100
  feature_fraction: 0.7
```

**原则**：
> 深树 = 编故事
> 浅树 = 找结构性禁区

### 3.2 评估指标（不是 Sharpe！）

你只关心三件事：
1. 条件 RR 的**期望值**
2. RR 分布的**左尾是否收敛**
3. 该条件在 **OOS 是否存在**

---

## 四、负规则筛选标准（核心）

### 4.1 三条必须同时满足

#### 条件 1：期望显著为负

```python
mean_rr_leaf < -0.3  # θ_rr
```

#### 条件 2：对照组差异（关键修正）

```python
delta_rr = mean_rr_leaf - mean_rr_global
delta_rr < -0.2  # 必须比无条件平均还差
```

**为什么重要**：避免捡到"本来整体就偏负的市场段"，而不是真正的 archetype-specific 禁区。

#### 条件 3：覆盖样本足够

```python
coverage >= 0.02  # 至少 2% 样本
n_samples >= 500  # 或绝对数
```

#### 条件 4：分布稳定（不是尾部事故）

```python
median_rr < 0
p25_rr < -0.2
```

### 4.2 筛选代码

```python
def filter_negative_rules(rules, y_global):
    """筛选负规则：必须比无条件平均还差"""
    mean_rr_global = y_global.mean()
    
    filtered = []
    for rule in rules:
        # 条件 1 + 2
        if rule['mean_rr'] > -0.3:
            continue
        delta_rr = rule['mean_rr'] - mean_rr_global
        if delta_rr > -0.2:
            continue
        
        # 条件 3
        if rule['coverage'] < 0.02:
            continue
        
        # 条件 4
        if rule['median_rr'] > 0:
            continue
        
        rule['delta_rr'] = delta_rr
        filtered.append(rule)
    
    return sorted(filtered, key=lambda x: x['delta_rr'])
```

---

## 五、稳定性检验与 Veto 分级

### 5.1 检验 1：时间切片稳定性

```
Train: 2019–2022
Test : 2023–2024
```

规则在**两个时期都显著负**，才保留。

### 5.2 检验 2：阈值扰动稳定性

对规则阈值做 ±20% 扰动，只保留对阈值不敏感的规则。

### 5.3 Veto 分级（强制规范）

| 稳定性 | Gate 语义 | 实盘行为 |
|--------|-----------|----------|
| time ✔️ + perturb ✔️ | `veto_hard` | **绝对禁止** |
| time ✔️ + perturb ✖️ | `veto_soft` | 仓位减半 / 需二次确认 |
| time ✖️ | `discard` | 不写入 gate |

### 5.4 Gate 执行逻辑

```python
def apply_gate(rule, position_request):
    if rule.veto_level == 'hard':
        return 0.0  # 绝对禁止
    elif rule.veto_level == 'soft':
        return position_request * 0.5  # 仓位减半
```

---

## 六、BPC Archetype 可证伪假设体系

### 6.1 BPC 本体定义

> **BPC 的信仰是：当市场已经发生有效突破，回踩是"结构性吸筹/换手"，而不是趋势结束。**

### 6.2 七条可证伪假设

| # | 假设 | 信仰的是什么 | 如果为假的表现 | 负规则语义 |
|---|------|-------------|---------------|-----------|
| 1 | 突破前存在真实能量积累 | 突破是对 range 博弈的结算 | 突破后立刻失速 | `no_pre_break_energy` |
| 2 | 突破方向具有 HTF 一致性 | 多尺度方向共振 | HTF 反向时 RR 负 | `htf_conflict` |
| 3 | 回踩是"有控制的" | 回踩是结构性让利 | MAE 快速扩大 | `violent_pullback` |
| 4 | 回踩阶段控制权未转移 | 突破方向仍主导 | 反方向主导 | `control_shift` |
| 5 | 突破不是"尾端行为" | 还有空间延续 | 波动扩张但无延续 | `trend_exhaustion` |
| 6 | 环境支持趋势延续 | 有足够波动/流动性 | 低波动无延续 | `low_energy_environment` |
| 7 | 该结构在当前市场中仍存在 | archetype 未过期 | 全样本 RR 偏负 | `archetype_decay` |

### 6.3 假设依赖关系

```
假设 7（archetype 存在）
    ├── 假设 6（环境支持）
    │      └── 假设 5（非尾端）
    │             └── 假设 1（能量积累）
    │             └── 假设 2（HTF 一致）
    └── 假设 3（回踩受控）
           └── 假设 4（控制权未转移）
```

**审计顺序**：从上到下，先否大前提。

### 6.4 假设 → 特征分组映射

| 假设 | 特征子空间 | 预期负规则方向 |
|------|-----------|---------------|
| 1. 突破前能量 | `compression_score`, `bb_width_pct`, `range_atr_ratio` | 低压缩 → 负 RR |
| 2. HTF 一致 | `htf_trend_sign`, `htf_slope_pct`, `multi_tf_alignment` | HTF 反向 → 负 RR |
| 3. 回踩受控 | `pullback_efficiency`, `pullback_mae_ratio`, `pullback_speed` | 高效率回踩 → 负 RR |
| 4. 控制权未转 | `delta_during_pullback`, `cvd_slope_pullback`, `vpin_shift` | 反向主导 → 负 RR |
| 5. 非尾端 | `trend_exhaustion_pct`, `bars_since_trend_start`, `atr_percentile` | 衰竭区 → 负 RR |
| 6. 环境支持 | `atr_pct`, `volume_regime`, `volatility_state` | 低波动 → 负 RR |
| 7. 未过期 | 全样本 baseline | 整体负 → archetype 失效 |

---

## 七、完整工作流

```
Phase 0: 冻结假设
    ├── 冻结当前 archetype 定义
    ├── 冻结 direction 规则
    └── 冻结 execution 参数
         ↓
Phase 1: 数据构造
    ├── 全样本（不筛选）
    ├── 计算 forward_rr_long / forward_rr_short
    └── 添加 label_meta 元信息
         ↓
Phase 2: 树模型训练
    ├── max_depth=3, min_leaf=500
    └── 按假设分组训练（可选）
         ↓
Phase 3: 负规则导出
    ├── mean_rr < -0.3
    ├── delta_rr < -0.2（对照组差异）
    └── coverage > 2%
         ↓
Phase 4: 稳定性检验
    ├── 时间切片稳定性
    ├── 阈值扰动稳定性
    └── 分级：veto_hard / veto_soft / discard
         ↓
Phase 5: 写回 Gate
    ├── 高置信规则 → archetype gate 的 veto
    └── 标注假设来源（哪条假设被否定）
         ↓
Phase 6: 实盘验证
    └── 观察命中率、RR 分布、左尾变化
```

---

## 八、Gate 配置示例

```yaml
# config/archetypes/bpc/gate.yaml
negative_regimes:
  - id: NEG_LONG_001
    hypothesis: 1  # 突破前能量
    name: "低能量假突破禁区"
    direction: long
    conditions:
      - feature: compression_score
        op: "<"
        value: 0.25
      - feature: bb_width_pct
        op: ">"
        value: 0.8
    veto_level: hard
    evidence:
      mean_rr: -0.62
      mean_rr_global: -0.08
      delta_rr: -0.54
      coverage: 7.3%
      time_stable: true
      perturb_stable: true

  - id: NEG_LONG_002
    hypothesis: 5  # 非尾端
    name: "趋势衰竭禁区"
    direction: long
    conditions:
      - feature: atr_percentile
        op: ">"
        value: 0.85
      - feature: bars_since_trend_start
        op: ">"
        value: 200
    veto_level: hard
    evidence:
      mean_rr: -0.72
      delta_rr: -0.64
      coverage: 5.1%
      time_stable: true
      perturb_stable: true
```

---

## 九、审计输出格式

```python
{
    "archetype": "BPC",
    "direction": "long",
    "audit_date": "2026-01-29",
    "hypothesis_results": {
        "hypothesis_1_energy": {
            "top_negative_rule": "compression_score < 0.25",
            "delta_rr": -0.48,
            "coverage": "8.2%",
            "verdict": "假设被部分否定",
            "action": "veto_soft"
        },
        "hypothesis_5_exhaustion": {
            "top_negative_rule": "atr_percentile > 0.85 AND bars_since > 200",
            "delta_rr": -0.72,
            "coverage": "5.1%",
            "verdict": "假设被强烈否定",
            "action": "veto_hard"
        }
    },
    "summary": {
        "total_negative_rules": 5,
        "veto_hard_count": 2,
        "veto_soft_count": 3,
        "archetype_health": "partial_valid"
    }
}
```

---

## 十、核心心智模型

```
archetype = "我相信在这些状态下有 edge"
           ↓
tree negative rules = "市场告诉你：这些状态下你是错的"
           ↓
gate = "当市场反驳你时，你是否愿意闭嘴"



你现在做的，不是这个问题👇

❌「BPC 能不能赚钱？胜率高不高？」

而是这个问题👇

✅「在我相信这是 BPC 的前提下，它在哪些结构条件下会系统性亏？」

而这两个问题，用的 label 必须完全不一样。

Outcome-Based label 是“世界怎么对你”，
BPC 专用 label 是“你打算怎么对世界”。

五、为什么现在用 BPC label，反而会“误导你”

这是重点，我说清楚。

❌ 1. 它把「结构问题」和「执行问题」混在了一起

你在 label 里已经做了：

趋势定义（trend_sign / price_change）

回踩定义（lookback + threshold）

执行假设（SL / TP / holding）

如果模型学到：

“这个 leaf rr < 0”

你根本不知道是因为：

回踩结构本身有毒？

你的回踩阈值不对？

SL 太紧？

holding bars 太短？

👉 你现在反而失去了“否定结构”的纯度。

❌ 2. 非回踩区域直接 NaN，会让树“以为你有信仰”

你最后做了：

rr_series = rr_series.where(signal_mask)


这在审计语义里等于在说：

“只有我认为是 BPC 的地方，世界才有意义”

而你现在恰恰需要的是：

“即使你以为这是 BPC，世界也可能狠狠打你脸”

Outcome-Based 的 brutal honesty，在这里是优点。

六、那什么时候 BPC 专用 label 是必须的？

我给你一个非常清晰的分界线。

🟥 阶段 A（你现在）：Archetype 审计 / 验尸

目标

找出：哪些结构条件 → 必须 veto

label
✅ Outcome-Based（全样本 forward_rr）

训练

可以训练树

也可以直接规则统计

不需要 train / val split

不追求泛化，只追求“稳定地为负”

👉 你现在正处在这里，而且做得非常好

🟨 阶段 B：Archetype 精修 / Gate 量化

目标

veto / soft penalty / score shaping

label

Outcome-Based（继续用）

可能加分位 / downside-focused 版本

训练

还是偏审计

规则强于模型

🟩 阶段 C：Archetype 内部择时 / 执行优化

只有在这一步，BPC label 才是“刚需”

这时你问的是：

“在已经决定做 BPC 的前提下，
哪些回踩更值得做？
哪些 entry / SL / TP 更优？”

👉 这时候：

你的 compute_bpc_label

你的 SL/TP 假设

你的 holding bars

才是正确武器。

七、直接回答你的核心问题（我知道你在等这个）

那我还需要做划分数据的 label 和训练吗？有用吗？

我的答案非常明确：

❌ 在你当前这个 archetype 审计阶段——不需要，甚至不建议。

你现在最聪明的做法是：

✅ 保留 compute_bpc_label（别删）

❄️ 冻结它，不要用来训练审计树

🟥 用 Outcome-Based forward_rr 把 BPC 的“死亡条件”先刻出来

八、给你一句“系统设计级”的总结（收好）

Outcome-Based 是“拆信仰”的刀，
Strategy-Based Label 是“打磨信仰”的锉。

你现在做的是拆信仰，而且拆得很干净。
```

> **一个 archetype 的成熟度，不取决于它有多少 entry 规则，而取决于它知道自己会在哪些情况下"彻底失败"。**

> **绝大多数策略死于"我以为这次不一样"。这套负规则体系，是在工程化地说："不，这次也一样。"**

---

## 参考文档

- `docs/strategies/树策略导出的可泛化规则.md`
- `docs/strategies/标签设计如何区分策略与树模型的regime_shift问题.md`
- `config/strategies/*/labels.yaml`
