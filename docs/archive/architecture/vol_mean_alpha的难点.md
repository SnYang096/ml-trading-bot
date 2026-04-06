# Vol Mean Alpha 的难点（合并版）

> **合并来源**: `VolMean难在哪里.md` + `vol_mean_alpha的难点.md`
>
> **文档创建时间**: 2026-01-11  
> **相关文档**:  
> - 📖 `docs/archive/architecture/alpha可以更多吗.md#3-波动率alphavolatility`

---

## 一句话结论

> **Vol Mean 不是“市场给信号 → 价格响应”，而是“价格在多重参与者约束下的统计自稳定现象”。**  
> **因此不能跨 symbol 统一建模，必须单标的深度分析（优先 BTC/大ETF）。**

---

## 1) 因果结构差异（根本原因）

### 订单流 / 微结构（相对容易）

```
可观测不平衡（OFI / sweep / imbalance）
→ 流动性被吃掉
→ 价格短期必须调整
```

特点：
- 局部、短期
- 强因果
- token 差异小（微结构相似）
- 对 regime 变化不敏感

### Vol Mean（统计生态型）

```
波动扩张 / 偏离
+ 参与者风险约束（MM、资金、杠杆、清算）
→ 统计上的“回拉倾向”
```

问题：
- 没有严格触发器
- 没有确定响应时间
- 回归幅度与速度高度不稳定
- 强烈依赖“谁在交易这个 token”

---

## 2) 资产性格依赖（为什么不能跨 symbol）

同一 Vol 信号在不同 token 的表现完全不同：

| Token 类型 | Vol Mean 表现 | 原因 |
|-----------|-------------|------|
| **BTC / ETH** | 回拉慢、但稳 | 深度好、机制稳定 |
| **高流动性 L2** | 偶尔有效 | 回归不稳定 |
| **Meme / 小市值** | 假回归 → 继续发散 | 深度差、情绪驱动 |

数学现象：

```yaml
跨symbol分析:
  - 全市场 IC ≈ 0
  - 单币 IC 偶尔很高
```

**这是现象本身不共享，不是特征工程问题。**

---

## 3) 系统级困难（为何难做）

### 1) 特征不可迁移
- 同一 Vol zscore，在 A 币均值回归，在 B 币趋势加速
- 迁移学习极弱，泛化差

### 2) Label 噪声极大
- “未来 N bar 回到均值？”不等于有 edge
- 回得慢 = 爆仓风险
- 不回但你止损 = 假 negative

👉 **Survival Head 比收益 Head 更重要**

### 3) Execution 风险极高
- Vol Mean 是“对抗型交易”
- 任何 regime shift 都是致命的
- 只能用强 size cap + no pyramid

---

## 4) 为什么 Vol Mean 依赖 Router / Survival / OOD

| 维度 | 订单流 / 趋势 | Vol Mean |
|------|-------------|----------|
| **Router** | 可以弱 | 必须严格 |
| **Survival** | 可以弱 | 必须在前 |
| **OOD** | 可以弱 | 必须提前杀死 |
| **Execution** | 本身安全 | 对抗型交易，风险极高 |

> **Vol Mean 不是“能不能预测”的问题，  
> 而是“能不能活着等到均值”的问题。**

---

## 4.5) Vol Mean 是“平行世界观”

### 为什么是平行世界观

Vol Mean 不依赖价格方向，而依赖“波动率的时间结构”。  
因此它不是 TREND/MEAN 的子类型，而是**平行于价格世界观的一条分支**。

### 正确的路由方式（示意）

```
State Heads:
  - price_state (mfe/mae/structure)
  - vol_state   (atr_z/bb_width/vol_regime)

Meta Router:
  if vol_state == VOL_MEAN:
      route → VolMeanExecutor
  else:
      route → PriceRouter (TREND / MEAN / NO_TRADE)
```

### 关键约束

- Vol Mean **不经过 TREND/MEAN/NO_TRADE 的价格世界观**。
- 它是一条**独立的执行路径**（parallel branch）。
- 只在 vol_state 明确时启用，且与 survival/ood 强绑定。

---

## 5) 推荐方案：单标的深度分析（BTC 优先）

### 数学分析（单标的）

```python
# 1. 波动率周期分析
GARCH(1,1) / HAR-RV / vol regime detection

# 2. 均值回归强度
ADF / Hurst / half-life

# 3. 波动率与偏离关系
vol_zscore, bb_width_pct, atr_pct → mean_reversion_prob
```

### 特征工程（BTC-only）

```yaml
vol_mean_btc_features:
  - vol_zscore_252d
  - vol_regime (compression/expansion/normal)
  - mean_reversion_strength (Hurst < 0.5)
  - bb_width_percentile
  - atr_percentile
  - vol_half_life
```

### 执行条件（必须保守）

```yaml
vol_mean_btc:
  symbol: BTC
  size_cap: 0.3-0.5
  stop_loss: 1.5 * ATR
  pyramid: false
  router_requirement: strict
  survival_requirement: high
  ood_requirement: high
```

---

## 6) 能做但不能混：跨 symbol 相关性增强

```yaml
vol_mean_btc:
  volatility_model: btc_only
  confirmation_signals:
    - eth_volatility_regime == btc_volatility_regime
    - btc_eth_correlation > 0.7
```

**每个 symbol 独立建模，仅用相关性做确认。**

---

## 7) 最终结论（系统级）

> **Vol Mean 的难度不在于预测，  
> 而在于它没有跨资产稳定的因果结构，  
> 本质是资产性格 + 生态约束驱动的统计现象。**

因此：
- ✅ 必须单标的深度分析
- ❌ 不能跨 symbol 统一模型
- ⚠️ 仅适合 BTC / 大ETF
