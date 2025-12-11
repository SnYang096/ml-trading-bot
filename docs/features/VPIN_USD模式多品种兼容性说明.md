# VPIN USD 模式多品种兼容性说明

## 核心原理

USD bucket_volume 方案的核心是：**使用 USD 价值而不是数量作为切片单位**。

### 传统方式（按数量）

```
BTC:  bucket_volume = 100 BTC
ETH:  bucket_volume = 5000 ETH
ADA:  bucket_volume = 1000000 ADA
```

**问题：**
- 不同品种的 bucket_volume 不可比较
- BTC 的 100 BTC 和 ETH 的 5000 ETH 的 USD 价值差异巨大
- VPIN 值无法跨品种比较

### USD 方式（按价值）

```
所有品种: bucket_volume_usd = 1,000,000 USD

BTC:  1,000,000 USD ÷ 50,000 USD/BTC = 20 BTC 一个桶
ETH:  1,000,000 USD ÷ 3,000 USD/ETH = 333 ETH 一个桶
ADA:  1,000,000 USD ÷ 1 USD/ADA = 1,000,000 ADA 一个桶
```

**优势：**
- ✅ 所有品种使用相同的 USD bucket_volume
- ✅ VPIN 值天然可比较（都基于相同的 USD 价值）
- ✅ 模型可以学习跨品种的通用订单流模式

## 为什么能处理不同价格的币？

### 计算过程

对于每个 tick：
```python
# 每个 tick 的 USD 价值
tick_value_usd = price × volume

# 按 USD 价值累积，达到 bucket_volume_usd 时形成一个桶
cumulative_value_usd += tick_value_usd

if cumulative_value_usd >= bucket_volume_usd:
    # 形成一个桶，计算 VPIN
    imbalance = abs(buy_value_usd - sell_value_usd)
    vpin = imbalance / bucket_volume_usd
```

### 示例对比

假设 `bucket_volume_usd = 1,000,000 USD`：

| 品种 | 价格 | 一个桶的数量 | 说明 |
|------|------|-------------|------|
| BTC  | 50,000 USD | 20 BTC | 高价值币，数量少 |
| ETH  | 3,000 USD | 333 ETH | 中等价值币 |
| ADA  | 1 USD | 1,000,000 ADA | 低价值币，数量多 |
| DOGE | 0.1 USD | 10,000,000 DOGE | 极低价值币，数量非常多 |

**关键点：**
- ✅ 无论价格高低，每个桶的 USD 价值都是 1,000,000 USD
- ✅ VPIN 值（0-1 范围）基于相同的 USD 价值，天然可比较
- ✅ 价格低的币（如 ADA）会自动用更多数量填满一个桶，但 USD 价值相同

## 实际计算示例

### BTC 示例

```python
# Tick 数据
price = 50,000 USD
volume = 0.5 BTC
side = 1 (buy)

# USD 价值
tick_value_usd = 50,000 × 0.5 = 25,000 USD

# 累积到 1,000,000 USD 需要：
# 1,000,000 ÷ 25,000 = 40 个这样的 tick
```

### ADA 示例

```python
# Tick 数据
price = 1 USD
volume = 10,000 ADA
side = 1 (buy)

# USD 价值
tick_value_usd = 1 × 10,000 = 10,000 USD

# 累积到 1,000,000 USD 需要：
# 1,000,000 ÷ 10,000 = 100 个这样的 tick
```

**结论：**
- ✅ BTC 和 ADA 虽然数量差异巨大，但 USD 价值相同
- ✅ 每个桶的 USD 价值都是 1,000,000 USD
- ✅ VPIN 值（imbalance / bucket_volume_usd）天然可比较

## 潜在问题和解决方案

### 问题 1: 价格波动影响

**问题：**
- 如果价格在计算过程中波动，同一个桶内的 USD 价值可能略有变化
- 例如：BTC 从 50,000 涨到 51,000，同一个桶的 USD 价值会略高

**影响：**
- ⚠️ 影响很小：每个桶的时间跨度通常很短（几分钟到几小时）
- ⚠️ 价格波动在桶内通常 < 5%，影响可忽略
- ✅ 使用 tick 级别的实时价格计算，已经是最准确的

### 问题 2: 极低价格币的精度

**问题：**
- 如果价格极低（如 0.0001 USD），需要非常大的数量才能填满一个桶
- 可能导致计算精度问题

**解决方案：**
```python
# 代码中已经处理了精度问题
if filled_value >= target_bucket - 1e-9:  # 使用容差避免浮点误差
    # 形成一个桶
```

### 问题 3: 不同币的流动性差异

**问题：**
- BTC 和 ADA 的流动性差异很大
- BTC 可能几分钟就填满一个桶，ADA 可能需要更长时间

**影响：**
- ⚠️ 这是正常的：不同币的流动性不同
- ✅ VPIN 值仍然可比较（都基于相同的 USD 价值）
- ✅ 模型可以通过其他特征（如 volume、tick_count）学习流动性差异

## 代码实现验证

### 关键代码片段

```python
# src/data_tools/tick_loader.py
def _compute_vpin_buckets_for_month(
    path: Path,
    bucket_volume: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    bucket_volume_usd: Optional[float] = None,
):
    # 如果使用 USD bucket_volume，计算每个 tick 的 USD 价值
    if bucket_volume_usd is not None:
        prices = df["price"].astype(float).to_numpy()
        values_usd = prices * volumes  # ✅ 每个 tick 的 USD 价值
        target_bucket = bucket_volume_usd
    else:
        values_usd = None
        target_bucket = bucket_volume
    
    # 按 USD 价值累积
    for i, (ts, vol, side) in enumerate(zip(timestamps, volumes, sides)):
        if bucket_volume_usd is not None:
            tick_value = values_usd[i]  # ✅ 使用 USD 价值
        else:
            tick_value = vol  # 传统方式：使用数量
        
        # 累积到 target_bucket
        remaining = tick_value
        while remaining > 0:
            space_left = target_bucket - filled_value
            trade_value = min(remaining, space_left)
            # ...
```

**验证点：**
- ✅ 所有品种都使用相同的 `bucket_volume_usd`
- ✅ 每个 tick 的 USD 价值 = price × volume
- ✅ 按 USD 价值累积，达到 `bucket_volume_usd` 时形成一个桶
- ✅ VPIN 值 = imbalance / bucket_volume_usd，天然归一化到 [0, 1]

## 使用建议

### 1. 选择合适的 bucket_volume_usd

**推荐值：**
- **小币种/低流动性**：500,000 USD（更敏感）
- **主流币种（BTC/ETH）**：1,000,000 USD（平衡）
- **高流动性币种**：2,000,000 USD（更稳定）

**选择原则：**
- 确保每天生成足够多的桶（建议 20-100 个/天）
- 避免桶太大（信号滞后）或太小（噪声大）

### 2. 多品种训练配置

```yaml
# config/feature_dependencies.yaml
vpin_features:
  compute_params:
    vpin_n_buckets: 50
    vpin_adaptive: false  # 禁用自适应（使用 USD 模式）
    vpin_bucket_volume_usd: 1000000.0  # 固定 USD bucket_volume
```

### 3. 验证跨品种一致性

```python
# 检查不同品种的 VPIN 分布
btc_vpin = df[df['symbol'] == 'BTC']['vpin']
eth_vpin = df[df['symbol'] == 'ETH']['vpin']
ada_vpin = df[df['symbol'] == 'ADA']['vpin']

# 应该看到相似的分布（都基于相同的 USD bucket_volume）
print(f"BTC VPIN mean: {btc_vpin.mean():.4f}")
print(f"ETH VPIN mean: {eth_vpin.mean():.4f}")
print(f"ADA VPIN mean: {ada_vpin.mean():.4f}")
```

## 总结

✅ **USD bucket_volume 方案可以处理所有价格的币种**，包括：
- 高价值币（BTC, ETH）
- 中等价值币（BNB, SOL）
- 低价值币（ADA, DOGE）
- 极低价值币（SHIB 等）

**核心原因：**
1. 使用 USD 价值而不是数量作为切片单位
2. 所有品种使用相同的 USD bucket_volume
3. VPIN 值基于相同的 USD 价值，天然可比较
4. 价格低的币自动用更多数量填满一个桶，但 USD 价值相同

**唯一要求：**
- ✅ tick 数据必须包含 `price` 列（用于计算 USD 价值）
- ✅ 这是标准要求，所有 tick 数据都应该有价格信息

