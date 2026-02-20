# Gate 语义特征实验报告：Archetype 语义 vs extreme_rr 预测力

> 更新日期：2025-02-19
> 标签定义：`failure_rr_extreme = forward_rr < -0.8R`，`forward_rr = (MFE-MAE)/ATR`（per-symbol 计算）

## 1. 背景与问题

当前 Gate 模型在全量数据上训练 extreme_rr / no_opportunity 标签。

| 策略 | Gate 训练 CV | Gate 效果 | 优化结果 |
|---|---|---|---|
| BPC | 0.079 | 856 allow / 5354 veto | 3 rules, 3 evidence |
| ME | 0.046 | 9529 allow / 16445 veto | 1 frozen rule, 0 evidence |
| FER | 0.045 | 6210 allow / 0 veto (全放行) | 0 rules, 0 evidence |

**核心问题**：为什么 BPC 能找到 Gate，而 FER/ME 找不到？

## 2. 实验方法

对每个策略的 archetype 语义特征(`bpc_*/fer_*/me_*`)，按阈值(P80/P90/P95)切分数据，
比较「有语义信号」vs「无语义信号」的：
- extreme_rr rate (bad rate, 越低越好)
- median forward_rr (越高越好)

如果某特征在极端阈值能拉开好坏差距 → 说明该语义特征有预测力，Gate 可以利用。

## 3. Feature Store 现状

| Feature Store | Timeframe | Cols | archetype 特征数 |
|---|---|---|---|
| `features_792208f36f` | 4H | 300 | 38个 `bpc_*` |
| `features_57ee22ea09` | 4H | 274 | 12个 `fer_*` |
| `features_2662e6b519` | 1H | 293 | 18个 `me_*` |

---

## 4. BPC 分析（已能工作，对标参考）

**5910 rows, 6 symbols, extreme_rr rate = 48.3%, median RR = -0.50**

### 4.1 高端信号（强语义 = 低 bad rate）

| 特征 | 阈值 | n | bad rate | vs 其余 | medRR | vs 其余 |
|---|---|---|---|---|---|---|
| `bpc_impulse_return_atr` | P90(>0.837) | 591 | **36.7%** | 49.6% | **+1.8** | -0.7 |
| `bpc_impulse_return_atr` | P80(>0.558) | 1182 | **38.0%** | 50.9% | **+1.7** | -0.9 |
| `bpc_score_continuation` | P90(>0.554) | 591 | 43.8% | 48.8% | +0.2 | -0.6 |
| `bpc_vol_breakout_confirm` | P80(>0.944) | 1182 | 44.8% | 49.2% | -0.2 | -0.7 |

### 4.2 低端信号（absence = bad）

| 特征 | 阈值 | n | bad rate | vs 其余 | medRR | vs 其余 |
|---|---|---|---|---|---|---|
| **`bpc_vol_breakout_confirm`** | P5(<=0.227) | 296 | **59.1%** | 47.7% | **-2.5** | -0.4 |
| `bpc_pullback_quality` | P5(<=0.061) | 296 | **55.1%** | 47.9% | -1.5 | -0.4 |
| `bpc_recovery_strength` | P5(<=0.120) | 296 | **54.1%** | 48.0% | -1.5 | -0.4 |
| `bpc_vol_breakout_confirm` | P10(<=0.274) | 591 | **55.5%** | 47.5% | -1.9 | -0.4 |
| `bpc_bb_compression` | P5(<=0.093) | 300 | **32.7%** | 49.1% | **+2.7** | -0.7 |

### 4.3 反信号（高 = 坏）

| 特征 | 阈值 | n | bad rate | vs 其余 | 语义解释 |
|---|---|---|---|---|---|
| `bpc_bb_compression` | P95(>0.900) | 294 | 54.4% | 48.0% | 极高压缩 = 还没突破 |
| `bpc_score_continuation` | P95(>0.750) | 211 | 52.6% | 48.1% | 过度延续 = 筋疲力竭 |

### 4.4 BPC 结论

**BPC Gate 有效的原因：38 个特征，两端都有强信号。**
- 正面(high=good): `impulse_return_atr` 12%绝对差异
- 反面(absence=bad): `vol_breakout_confirm` P5 有 11%绝对差异
- 组合空间大: 38特征×多阈值 → Gate 学到有效规则

---

## 5. FER 分析（全放行，核心问题）

**6210 rows (after dropna), 6 symbols, extreme_rr rate = 48.3%, median RR = -0.50 (4H)**

### 5.1 "失败检测"类特征 — 无预测力

| 特征 | 阈值 | bad rate(有信号) | bad rate(无信号) | 差异 |
|---|---|---|---|---|
| `fer_impulse_failure_score` | >0 | ~48% | ~48% | 无 |
| `fer_momentum_efficiency_decay` | >0 | ~48% | ~48% | 无 |
| `fer_efficiency_flip_score` | P90 | ~48% | ~48% | 无 |
| `fer_signed_efficiency` | P95 | 43.9% | 48.5% | 轻微 |

**语义解释**："冲击失败了" ≠ "反转机会来了"。失败是市场常态，不构成 alpha。

### 5.2 "被套"类特征 — 唯一有效信号

| 特征 | 阈值 | n | bad rate | vs 其余 | medRR | vs 其余 |
|---|---|---|---|---|---|---|
| **`fer_trapped_longs_score`** | P90(>4.615) | 537 | **37.4%** | 49.4% | **+1.84** | -0.72 |
| `fer_trapped_longs_score` | P80(>3.204) | 1073 | 44.2% | 49.2% | +0.21 | -0.70 |
| `fer_trapped_shorts_score` | P80(>3.860) | 1073 | 44.8% | 49.1% | -0.04 | -0.64 |

### 5.3 组合预筛

```
COMBO: fer_trapped_longs_score > P90(4.615) OR fer_trapped_shorts_score > P80(3.860)
  覆盖率: 27.2% (1610 / 5910 rows)
  bad rate: 42.4% vs 50.5% (8.1% 绝对差异)
  median RR: +0.55 vs -0.88
```

### 5.4 FER 关键洞察

**"有人被套住了" = 反转燃料（FER 唯一有效语义）**
- `fer_trapped_longs_score`（多头被套）= 空头反转的燃料
- `fer_trapped_shorts_score`（空头被套）= 多头反转的燃料
- 其他所有 "failure/decay/flip" 特征：在所有阈值上几乎零预测力

**FER Gate 全放行的原因**：12个特征中只有 2 个(trapped)有信号。Gate 模型面对 10 个噪声 + 2 个微弱信号 + 200 个通用特征 → 无法学到任何东西。

### 5.5 FER 建议方案

**用 trapped score 做 Gate 训练数据预筛：**
```yaml
# 只在"有人被套"的时段训练 FER Gate
data_filter:
  condition: OR
  rules:
    - feature: fer_trapped_longs_score
      operator: ">"
      threshold: 4.615   # P90
    - feature: fer_trapped_shorts_score
      operator: ">"
      threshold: 3.860   # P80
```

预期效果：从 6210 行压到 ~1610 行(27%)，bad rate 从 48.3% 降到 42.4%，
让 Gate 在"有反转燃料"的子集上学习，排除"什么都没发生"的噪声。

---

## 6. ME 分析（Gate 弱，1 frozen rule）

**25674 rows (after dropna), 6 symbols, extreme_rr rate = 44.8%, median RR = +0.01 (1H)**

### 6.1 关键发现：ME 大多数语义特征是反信号（高 = 坏）

| 特征 | 阈值 | n | bad rate | vs 其余 | medRR | vs 其余 |
|---|---|---|---|---|---|---|
| **`me_accel_5k`** | P5(<-0.531) | 1284 | **55.0%** | 44.2% | **-1.65** | +0.09 |
| `me_vol_divergence` | P95(>0.529) | 1284 | **52.1%** | 44.4% | -1.17 | +0.06 |
| `me_volume_surge` | P95(>0.950) | 1195 | 51.1% | 44.5% | -1.01 | +0.05 |
| `me_cvd_strength` | P95(>0.924) | 1284 | 49.5% | 44.5% | -0.67 | +0.04 |
| `me_volume_accel` | P90(>0.900) | 2434 | 49.9% | 44.2% | -0.79 | +0.09 |
| `me_accel_persistence` | P80(>0.600) | 1928 | 47.7% | 44.5% | -0.49 | +0.04 |

### 6.2 低 ATR = 安全区

| 特征 | 阈值 | n | bad rate | vs 其余 | medRR | vs 其余 |
|---|---|---|---|---|---|---|
| `me_atr_pct` | P5(<=0.010) | 2103 | 41.5% | 45.1% | +0.59 | -0.04 |
| `me_false_expansion > 0` | binary | 4970 | 42.4% | 45.3% | +0.35 | -0.08 |

### 6.3 组合测试

| 组合 | n | bad rate | vs 其余 | medRR | vs 其余 |
|---|---|---|---|---|---|
| `accel_5k<P10 AND vol_accel>P80` | 890 | **53.3%** | 44.5% | -1.49 | +0.04 |
| `vol_surge>P80 AND cvd>P80` | 1964 | **51.9%** | 44.2% | -1.05 | +0.10 |
| `vol_div>P80 AND vol_surge>P80` | 1379 | **51.2%** | 44.4% | -0.96 | +0.05 |
| `accel_5k<P5` (单特征) | 1284 | **55.0%** | 44.2% | -1.65 | +0.09 |
| `vol_div>P95` (单特征) | 1284 | 52.1% | 44.4% | -1.17 | +0.06 |

### 6.4 废特征

| 特征 | 问题 |
|---|---|
| `me_delta_net_flow` | **全为 0**（废特征，需修复或移除） |
| `me_flow_exhaustion` | 仅 2.6% non-zero（样本不足） |
| `me_regime_suitable` | 26% non-zero, bad=43.6% vs 45.2%（差异极小，几乎无用） |

### 6.5 ME 关键洞察

**ME 的语义悖论："强动量扩张" = 即将耗竭（反信号）**
- 高 CVD strength / volume_surge / volume_accel / vol_regime → 全部是 **更差的** 结局
- 最强反信号：`me_accel_5k < P5`（急剧减速）→ 55% bad rate, medRR = -1.65
- 语义解释：
  - "动量在加速" → 市场可能已经走完（momentum exhaustion）
  - "急剧减速" → 趋势终结的前兆
- 唯一微弱正信号：`me_false_expansion > 0`（标记为假扩张的反而稍好，反直觉）

**ME Gate 弱的原因**：18个特征中，大部分是反信号（Gate 难以学习"高值=坏"的模式），
加上特征间高度共线（volume_surge/volume_accel/cvd_strength 相似），组合空间有限。

### 6.6 ME 建议方案

ME 不需要预筛训练数据。ME 的问题是特征本身需要加强或修正：

1. **修复废特征**：`me_delta_net_flow` 全为 0，需要排查计算逻辑
2. **构造"减速"特征**：`me_accel_5k < P5` 是最强信号，考虑作为 hard_gate 候选
3. **反转特征方向**：把 "high exhaustion = deny" 显式编码到 gate 候选规则中

---

## 7. 总结对比

| 策略 | 特征数 | 有效特征 | 最强信号 | Gate状态 | 根因 |
|---|---|---|---|---|---|
| BPC | 38 | 5+ 双端有效 | impulse_return_atr P90: 12%差异 | 正常工作 | 特征多+强 |
| FER | 12 | **仅 trapped (2个)** | trapped_longs P90: 12%差异 | 全放行 | 10个噪声特征淹没 |
| ME | 18 | 多个反信号 | accel_5k P5: 11%差异(反向) | 极弱 | 反信号+共线 |

## 8. 行动方案

### FER: 用 trapped score 预筛 Gate 训练数据

```
条件: fer_trapped_longs_score > 4.615 (P90) OR fer_trapped_shorts_score > 3.860 (P80)
效果: 数据量 6210 → ~1610 (27%), bad rate 48.3% → 42.4%
语义: "只在有人被套（反转燃料存在）的时段训练 Gate"
```

### ME: 特征修复 + hard_gate 候选

```
1. 修复 me_delta_net_flow (当前全为0)
2. 添加 hard_gate 候选: me_accel_5k > -0.531 (排除急剧减速时段, 5%过滤)
3. 考虑添加: me_vol_divergence < 0.529 (排除极端价量背离, 5%过滤)
```

### BPC: 无需改动

### LV: 15min 天然正交，待训练完成后评估
