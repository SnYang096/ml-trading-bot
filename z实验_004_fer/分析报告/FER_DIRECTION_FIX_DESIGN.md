# FER方向逻辑修复设计

**问题**: FER当前配置为固定做多(`direction: long`)，与反转策略语义不符

---

## 1. 问题诊断

### 当前配置错误

| 策略 | 当前direction | 语义 | 是否正确 |
|------|--------------|------|---------|
| BPC | `long` | 突破回踩延续（做多） | ✅ 正确 |
| ME | `long` | 动量扩张突破（做多） | ⚠️ 应该动态 |
| FER | `long` | **单边失败反转** | ❌ **错误** |

### FER核心语义

引用 `z实验_004_fer/fer语义.md`:

> **资金强度没有下降，但价格推进已经死亡。**  
> 单边博弈失败 → 反向清算

**关键特征**:
1. 多头失败 → 做空清算
2. 空头失败 → 做多清算
3. **方向应该是失败方向的反向**

---

## 2. 方向决定逻辑设计

### 方案1: 基于CVD方向（推荐）

**逻辑**:
```python
# CVD正值 = 多头aggressor活跃但价格不涨 → 多头失败 → 做空
# CVD负值 = 空头aggressor活跃但价格不跌 → 空头失败 → 做多

if 'cvd_change_5_normalized' in df.columns:
    # CVD正=做空，CVD负=做多
    direction = -np.sign(df['cvd_change_5_normalized'])
    # 处理CVD=0的情况（默认做多）
    direction = direction.replace(0, 1)
else:
    # 降级方案：固定做多（当前行为）
    direction = 1.0
```

**优点**:
- 符合FER语义（CVD强但价格推进失败）
- Evidence特征中CVD排名第1（重要性157,662）
- 已在predictions中存在

### 方案2: 基于RSI方向

**逻辑**:
```python
# RSI > 70 = 超买 → 做空反转
# RSI < 30 = 超卖 → 做多反转

if 'rsi' in df.columns:
    direction = np.where(df['rsi'] > 50, -1, 1)  # RSI>50做空，<50做多
else:
    direction = 1.0
```

**优点**:
- RSI是Evidence特征第2重要（147,510）
- 超买/超卖信号明确

**缺点**:
- 不如CVD直接反映"失败"特征

### 方案3: 基于BB位置

**逻辑**:
```python
# bb_position > 0.8 = 接近上轨 → 做空
# bb_position < 0.2 = 接近下轨 → 做多

if 'bb_position' in df.columns:
    direction = np.where(df['bb_position'] > 0.5, -1, 1)
else:
    direction = 1.0
```

---

## 3. 实施方案对比

| 方案 | 语义准确性 | 数据可用性 | 实施难度 | 推荐度 |
|------|-----------|-----------|---------|-------|
| **CVD方向** | ⭐⭐⭐⭐⭐ | ✅ 已有 | 简单 | **强烈推荐** |
| RSI方向 | ⭐⭐⭐ | ✅ 已有 | 简单 | 备选 |
| BB位置 | ⭐⭐ | ✅ 已有 | 简单 | 备选 |

---

## 4. 代码实施位置

### 修改点1: 标签配置

**文件**: `config/strategies/fer/labels_return_tree.yaml`

```yaml
label_generator:
  params:
    direction: dynamic  # 改为dynamic，由代码决定
    # 或者删除direction参数
```

### 修改点2: Predictions后处理

**文件**: 新建 `scripts/fix_fer_direction.py`

```python
def add_fer_direction(df):
    """为FER添加动态方向列"""
    # 方案1: 基于CVD
    if 'cvd_change_5_normalized' in df.columns:
        direction = -np.sign(df['cvd_change_5_normalized'])
        direction = direction.replace(0, 1)
    else:
        # 降级：固定做多
        direction = 1.0
    
    df['entry_direction'] = direction
    return df
```

### 修改点3: 回测脚本

**文件**: `scripts/backtest_execution_layer.py`

在 `_detect_direction_col` 函数后添加FER专属处理：

```python
# 检测方向列后
dir_col = _detect_direction_col(df, arch_name)

# FER专属：动态计算方向
if arch_name.lower() == 'fer':
    if 'cvd_change_5_normalized' in df.columns:
        direction = -np.sign(df['cvd_change_5_normalized'])
        direction = direction.replace(0, 1)
        df['entry_direction'] = direction
    elif dir_col:
        df['entry_direction'] = df[dir_col]
    else:
        print(f"⚠️  FER: 无CVD列，使用固定做多")
        df['entry_direction'] = 1.0
else:
    # 其他策略保持原逻辑
    df['entry_direction'] = df[dir_col]
```

---

## 5. ME方向逻辑优化

ME当前也是固定做多，应该改为：

### 方案: 基于突破方向

**语义**: ME是压缩后扩张突破，方向应该由突破方向决定

**实施**:
```python
if arch_name.lower() == 'me':
    # 方案1: 如果有breakout_sign列
    if 'breakout_sign' in df.columns:
        df['entry_direction'] = df['breakout_sign']
    # 方案2: 基于价格相对SR位置
    elif 'dist_to_nearest_sr' in df.columns:
        # 价格高于SR→做多，低于SR→做空
        df['entry_direction'] = np.sign(df['dist_to_nearest_sr'])
    # 方案3: 降级为固定做多
    else:
        df['entry_direction'] = 1.0
```

---

## 6. 验证计划

### 步骤1: 修复FER direction
1. 修改FER predictions，添加基于CVD的动态方向
2. 重新运行PCM回测
3. 对比胜率/Sharpe变化

### 步骤2: 评估ME direction
1. 检查ME是否有breakout_sign列
2. 如果没有，评估是否需要重新训练
3. 短期可保持做多，长期需修复

### 步骤3: 完整PCM验证
1. 三策略都修复后重新回测
2. 检查冲突信号处理是否合理
3. 验证各策略胜率/R是否改善

---

## 7. 预期影响

### FER修复后

| 指标 | 修复前 | 预期修复后 | 原因 |
|------|-------|-----------|------|
| 胜率 | 55.5% | **45-50%** | 反转策略本身胜率偏低 |
| 平均R | 0.4525 | **0.6-0.8** | 反转成功时R更高 |
| 交易数 | 10,494 | **约一半** | 只在明确失败时入场 |

### ME修复后

| 指标 | 修复前 | 预期修复后 | 原因 |
|------|-------|-----------|------|
| 胜率 | 82.3% | **60-65%** | 不再只做多，双向交易 |
| 交易数 | 158 | **250-300** | 增加空头机会 |

---

## 8. 立即行动

1. **等待ME新训练完成**（监控脚本已启动）
2. **修复FER direction**（基于CVD）
3. **重新运行PCM回测**
4. **对比修复前后的KPI**

---

**结论**: FER和ME的direction逻辑需要修复，当前固定做多与策略语义不符。优先修复FER（基于CVD反向），ME可暂时保持做多（等新训练结果）。
