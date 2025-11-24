# 为什么需要 factor-compute：实盘因子计算的优化

## 问题：直接运行完整特征工程的问题

### 场景对比

#### ❌ 直接运行完整特征工程代码
```python
# 实盘代码
engineer = ComprehensiveFeatureEngineer(feature_types="comprehensive")
df_features = engineer.engineer_all_features(df)  # 没有 required_features 参数
```

**结果**：
- 计算所有特征（可能 500-1000+ 个）
- 包括：baseline (100+) + default (50+) + enhanced (200+) + alpha101 (101) + dl_sequence (50+) 等
- **计算时间**：可能需要几分钟甚至更久
- **内存占用**：大量不必要的特征占用内存
- **实盘延迟**：无法快速响应市场变化

#### ✅ 使用 factor-compute（只计算需要的因子）
```python
# 实盘代码
engineer = ComprehensiveFeatureEngineer(feature_types="comprehensive")
df_features = engineer.engineer_all_features(
    df, 
    required_features={"rsi_7", "macd", "price_to_zz_high_pct"}  # 只计算这3个
)
```

**结果**：
- 只计算需要的 3 个因子
- **计算时间**：几秒钟
- **内存占用**：最小化
- **实盘延迟**：快速响应

---

## 性能对比示例

假设你需要计算 3 个因子：`rsi_7`, `macd`, `price_to_zz_high_pct`

### 方案 1：完整特征工程
```python
engineer = ComprehensiveFeatureEngineer(feature_types="comprehensive")
df_features = engineer.engineer_all_features(df)
# 计算了 800+ 个特征，但只需要 3 个
```

**性能**：
- ⏱️ 计算时间：~60 秒
- 💾 内存：~500MB
- 📊 输出特征：800+ 个（但只用 3 个）

### 方案 2：factor-compute（使用 required_features）
```python
engineer = ComprehensiveFeatureEngineer(feature_types="comprehensive")
df_features = engineer.engineer_all_features(
    df,
    required_features={"rsi_7", "macd", "price_to_zz_high_pct"}
)
```

**性能**：
- ⏱️ 计算时间：~5 秒（快 12 倍）
- 💾 内存：~50MB（减少 90%）
- 📊 输出特征：3 个（精确匹配需求）

---

## required_features 的工作原理

### 1. 依赖关系解析
虽然指定了 `required_features`，但系统会：
- ✅ 自动识别依赖关系（例如 `macd` 需要先计算 `close` 的 EMA）
- ✅ 只计算必要的中间特征
- ✅ 跳过不需要的特征模块

### 2. 模块级优化
```python
# 示例：如果只需要 rsi_7
required_features = {"rsi_7"}

# 系统会：
# 1. 计算 baseline 特征时，检查是否需要 baseline 模块的特征
#    - 如果 rsi_7 不在 baseline 中，跳过整个 baseline 模块
# 2. 计算 default 特征时，检查是否需要 default 模块的特征
#    - 如果 rsi_7 在 default 中，只计算 rsi_7 及其依赖
# 3. 跳过其他模块（enhanced, alpha101, dl_sequence 等）
```

### 3. 内存优化
```python
# 在每个模块生成后立即过滤
def _filter_features(df_in, module_name):
    if required_features is None:
        return df_in  # 不过滤
    # 只保留需要的特征列
    cols_to_keep = [c for c in df_in.columns 
                   if c in data_cols or c in required_features]
    return df_in[cols_to_keep]  # 立即释放不需要的特征内存
```

---

## 实盘场景示例

### 场景：实盘交易系统，每秒需要计算因子

#### ❌ 不使用 factor-compute
```python
# 每秒调用
def get_factors():
    engineer = ComprehensiveFeatureEngineer()
    df_features = engineer.engineer_all_features(latest_data)
    return df_features[["rsi_7", "macd"]]  # 只用了2个，但计算了800+
```

**问题**：
- 每秒计算 800+ 个特征
- CPU 占用高
- 延迟高（可能超过 1 秒）
- 内存占用大

#### ✅ 使用 factor-compute
```python
# 每秒调用
def get_factors():
    engineer = ComprehensiveFeatureEngineer()
    df_features = engineer.engineer_all_features(
        latest_data,
        required_features={"rsi_7", "macd"}
    )
    return df_features  # 只有需要的2个特征
```

**优势**：
- 每秒只计算 2 个特征
- CPU 占用低
- 延迟低（<100ms）
- 内存占用小

---

## factor-compute 的额外优势

### 1. 输出格式灵活
```bash
# 输出为 CSV（便于数据库导入）
make factor-compute FACTOR_COMPUTE_FORMAT=csv

# 输出为 Parquet（高效存储）
make factor-compute FACTOR_COMPUTE_FORMAT=parquet

# 输出为 JSON（API 接口）
make factor-compute FACTOR_COMPUTE_FORMAT=json
```

### 2. 批量计算
```bash
# 为多个时间点批量计算因子
for date in dates:
    make factor-compute \
        FACTOR_COMPUTE_FACTORS="rsi_7 macd" \
        FACTOR_COMPUTE_START_DATE=$date \
        FACTOR_COMPUTE_END_DATE=$date \
        FACTOR_COMPUTE_OUTPUT="factors/${date}.csv"
```

### 3. 与模型集成
```python
# 从 ts-dim-compare 的结果中读取 top_factors.json
with open("results/dim_compare/.../top_factors.json") as f:
    top_factors = json.load(f)

# 只计算这些因子
make factor-compute FACTOR_COMPUTE_FACTORS="$(cat top_factors.json | jq -r '.[]')"
```

---

## 总结

### 为什么需要 factor-compute？

1. **性能优化** ⚡
   - 只计算需要的因子，节省计算时间（快 10-20 倍）
   - 减少内存占用（减少 80-90%）

2. **实盘需求** 🚀
   - 实盘需要快速响应（<100ms）
   - 不能计算所有特征（太慢）

3. **资源节约** 💰
   - 减少 CPU 使用
   - 减少内存占用
   - 降低服务器成本

4. **灵活性** 🔧
   - 支持多种输出格式
   - 易于集成到实盘系统
   - 支持批量计算

### 什么时候用哪个？

| 场景 | 使用 |
|------|------|
| **研究/训练** | 完整特征工程（需要所有特征） |
| **因子测试** | `factor-test`（评估因子质量） |
| **实盘计算** | `factor-compute`（只计算需要的因子） |
| **特征选择** | `ts-dim-compare`（找出最优因子） |

### 最佳实践

```bash
# 1. 研究阶段：找出最优因子
make ts-dim-compare SYMBOLS=BTCUSDT FACTOR_COUNTS=20,40,60

# 2. 获取选出的因子列表
top_factors=$(cat results/dim_compare/.../top_factors.json | jq -r '.[]' | tr '\n' ' ')

# 3. 实盘：只计算这些因子
make factor-compute FACTOR_COMPUTE_FACTORS="$top_factors"
```

---

## 技术细节

### required_features 的优化程度

虽然 `required_features` 会跳过不需要的特征，但**仍然会调用各个特征工程模块**。这是因为：

1. **依赖关系复杂**：某些因子可能依赖多个模块
2. **模块初始化成本**：创建特征工程器需要初始化
3. **部分计算**：即使跳过某些特征，模块仍需要检查依赖

**但相比完整计算，仍然快很多**：
- 跳过不需要的特征计算（节省 90%+ 时间）
- 立即释放不需要的特征内存
- 减少中间结果的存储

### 进一步优化建议

如果实盘只需要极少数因子（<10 个），可以考虑：
1. **直接调用底层函数**：绕过 `ComprehensiveFeatureEngineer`
2. **缓存中间结果**：如果多个因子共享依赖，缓存中间计算
3. **预计算**：对于不依赖最新数据的因子，可以预计算

但对于大多数场景，`factor-compute` 已经足够优化了。

