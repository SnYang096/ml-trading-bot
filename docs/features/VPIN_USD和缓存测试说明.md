# VPIN USD bucket_volume 和缓存功能测试说明

## 测试文件

### 1. `tests/test_vpin_usd_bucket_and_cache.py` - 新增测试文件

**测试内容：**

#### TestVPINUSDBucketVolume（USD bucket_volume 计算测试）
- ✅ `test_usd_bucket_volume_basic` - 测试 USD bucket_volume 基本计算
- ✅ `test_usd_bucket_volume_multi_asset_comparison` - 测试多品种（BTC/ETH/ADA）使用相同 USD bucket_volume 的可比性
- ✅ `test_usd_vs_traditional_bucket_volume` - 测试 USD 模式 vs 传统模式对比

#### TestVPINUSDCache（USD 模式缓存测试）
- ✅ `test_usd_cache_key_generation` - 测试 USD 模式缓存键生成
- ✅ `test_usd_cache_save_and_load` - 测试 USD 模式缓存保存和加载
- ✅ `test_usd_cache_incremental_computation` - 测试 USD 模式增量计算（缓存命中）

#### TestVPINUSDIntegration（集成测试）
- ✅ `test_extract_order_flow_features_with_usd_bucket` - 测试完整流程（extract_order_flow_features 使用 USD bucket_volume）

### 2. `tests/test_monthly_cache.py` - 更新现有测试

**新增测试：**
- ✅ `test_vpin_usd_cache_key_generation` - 测试 USD 模式缓存键生成（在 TestVPINMonthlyCache 类中）

## 测试覆盖范围

### 单元测试

1. **USD bucket_volume 计算**
   - ✅ 基本计算正确性
   - ✅ VPIN 值范围验证（[0, 1]）
   - ✅ 多品种兼容性（BTC/ETH/ADA）
   - ✅ USD 模式 vs 传统模式对比

2. **缓存功能**
   - ✅ 缓存键生成（传统模式 vs USD 模式）
   - ✅ 缓存保存和加载
   - ✅ 缓存数据一致性验证
   - ✅ 增量计算（缓存命中）

3. **集成测试**
   - ✅ extract_order_flow_features 使用 USD bucket_volume
   - ✅ 完整流程验证

## 运行测试

### 运行所有 USD bucket_volume 和缓存测试

```bash
# 运行新测试文件
pytest tests/test_vpin_usd_bucket_and_cache.py -v

# 运行所有 VPIN 相关测试
pytest tests/test_vpin*.py -v

# 运行按月缓存测试（包括 USD 模式）
pytest tests/test_monthly_cache.py::TestVPINMonthlyCache -v
```

### 运行特定测试

```bash
# 测试 USD bucket_volume 基本计算
pytest tests/test_vpin_usd_bucket_and_cache.py::TestVPINUSDBucketVolume::test_usd_bucket_volume_basic -v

# 测试多品种可比性
pytest tests/test_vpin_usd_bucket_and_cache.py::TestVPINUSDBucketVolume::test_usd_bucket_volume_multi_asset_comparison -v

# 测试 USD 模式缓存
pytest tests/test_vpin_usd_bucket_and_cache.py::TestVPINUSDCache::test_usd_cache_save_and_load -v
```

### 直接运行（不使用 pytest）

```bash
python tests/test_vpin_usd_bucket_and_cache.py
```

## 测试数据

测试使用模拟数据：

- **BTC**: 价格 ~50,000 USD，数量 0.1-5.0 BTC
- **ETH**: 价格 ~3,000 USD，数量 1.0-50.0 ETH
- **ADA**: 价格 ~1.0 USD，数量 1,000-10,000 ADA

所有测试数据都包含：
- `timestamp` - 时间戳
- `price` - 价格（必需，用于计算 USD 价值）
- `volume` - 成交量
- `side` - 买卖方向（1/-1）

## 验证点

### USD bucket_volume 验证

1. **计算正确性**
   - ✅ 每个 tick 的 USD 价值 = price × volume
   - ✅ 按 USD 价值累积，达到 bucket_volume_usd 时形成一个桶
   - ✅ VPIN 值 = imbalance / bucket_volume_usd，归一化到 [0, 1]

2. **多品种可比性**
   - ✅ 所有品种使用相同的 USD bucket_volume
   - ✅ VPIN 值分布相似（都基于相同的 USD 价值）
   - ✅ 价格低的币（如 ADA）自动用更多数量填满一个桶，但 USD 价值相同

3. **与传统模式对比**
   - ✅ USD 模式和传统模式都能生成有效结果
   - ✅ VPIN 值都在 [0, 1] 范围内

### 缓存功能验证

1. **缓存键生成**
   - ✅ 相同参数生成相同的缓存键
   - ✅ 不同参数（bucket_volume_usd）生成不同的缓存键
   - ✅ USD 模式缓存键包含 "usd" 标识

2. **缓存保存和加载**
   - ✅ 缓存文件正确保存
   - ✅ 缓存数据正确加载
   - ✅ 加载的数据与原始数据一致

3. **增量计算**
   - ✅ 第一次计算时创建缓存
   - ✅ 第二次计算时使用缓存
   - ✅ 使用缓存的结果与原始计算结果一致

## 注意事项

1. **测试数据要求**
   - 所有 tick 数据必须包含 `price` 列（用于计算 USD 价值）
   - 这是标准要求，所有 tick 数据都应该有价格信息

2. **缓存目录**
   - 测试使用临时目录，测试结束后自动清理
   - 实际使用时，缓存目录应持久化

3. **性能考虑**
   - USD 模式需要计算 price × volume，可能略慢于传统模式
   - 但差异很小，可以忽略

## 后续改进

1. **更多测试场景**
   - 测试极低价格币（如 DOGE）
   - 测试价格波动对 USD 价值的影响
   - 测试边界情况（空数据、单 tick 等）

2. **性能测试**
   - 对比 USD 模式 vs 传统模式的性能
   - 测试缓存带来的性能提升

3. **集成测试**
   - 测试完整训练流程（使用 USD bucket_volume）
   - 测试多品种联合训练

