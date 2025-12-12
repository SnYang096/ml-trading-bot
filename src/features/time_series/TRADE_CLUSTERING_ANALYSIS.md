# Trade Clustering 分析与优化建议

## 一、当前实现的问题

### 1. 缓存键包含 start/end 时间（类似 VPIN 之前的问题）

**问题**：
- 当前缓存键：`trade_clustering_monthly_{month_str}_{window_size}_{start.isoformat()}_{end.isoformat()}`
- 即使计算同一个月的 trade clustering，如果时间窗口不同，也会重新计算
- 导致缓存无法复用，浪费计算资源

**影响**：
- 计算 1~6 月后，再计算 3~6 月，3~6 月会重新计算（因为时间窗口不同）
- 缓存命中率低，内存占用高

### 2. 状态连续性处理

**当前实现**：
- Trade clustering 需要维护一个滑动窗口状态（`window_runs`），这个状态是跨月的
- 每个月的计算会传入上个月的状态（`initial_state`）
- 但每个月的 `final_state` 应该是固定的（只取决于该月数据），不依赖于前一个月的状态

**潜在问题**：
- 如果 `initial_state` 不同，计算出的 `final_state` 可能不同（因为滑动窗口的初始状态不同）
- 但理论上，如果窗口足够大，`final_state` 应该收敛到固定值

## 二、Trade Clustering 能否找出冰山单子？

### 1. 冰山单子的特征

冰山单子（Iceberg Order）的特征：
- **连续小额同向交易**：大单被拆分成多个小单，连续执行
- **隐藏真实意图**：避免暴露大单，减少市场冲击
- **时序模式**：连续多笔都是 buy（或 sell），但每笔量都不大

### 2. Trade Clustering 的捕捉能力

**Trade Clustering 可以捕捉的特征**：
- ✅ **连续同向交易**：`max_buy_run` / `max_sell_run` 可以捕捉连续同向交易的模式
- ✅ **聚集性**：`avg_buy_run` / `avg_sell_run` 可以衡量平均连续同向交易长度
- ✅ **不平衡性**：`imbalance_ratio` 可以衡量买卖方向的不平衡
- ✅ **方向熵**：`directional_entropy` 可以衡量交易方向的随机性（冰山单子会导致熵降低）

**Trade Clustering 的局限性**：
- ❌ **不关注交易量**：只关注交易方向（side），不关注每笔交易的量
- ❌ **不关注价格变化**：不关注价格是否在移动
- ❌ **不能直接识别**：只能捕捉模式，不能直接判断是否为冰山单子

### 3. 如何识别冰山单子？

要识别冰山单子，需要结合多个特征：

1. **Trade Clustering 特征**：
   - `max_buy_run` / `max_sell_run` 高（连续同向交易多）
   - `directional_entropy` 低（方向随机性低，表明有系统性行为）

2. **交易量特征**：
   - 每笔交易量小但稳定
   - 总交易量累积较大

3. **价格特征**：
   - 价格变化小（冰山单子通常不会大幅推动价格）
   - 买卖价差稳定

4. **时间特征**：
   - 交易间隔规律（冰山单子通常按固定间隔执行）

### 4. 建议的优化方向

如果要更好地识别冰山单子，可以考虑：

1. **增强 Trade Clustering**：
   - 添加基于交易量的加权 clustering（volume-weighted clustering）
   - 添加基于价格的 clustering（价格变化小的连续交易）

2. **结合其他特征**：
   - 结合 VPIN（捕捉订单流不平衡）
   - 结合交易量分布（捕捉小额交易的累积）
   - 结合价格变化（捕捉价格稳定性）

3. **专门的冰山单子检测**：
   - 使用机器学习模型，结合多个特征
   - 使用规则引擎，定义冰山单子的特征组合

## 三、优化建议

### 1. 缓存键优化（类似 VPIN）

**建议**：
- 移除 `start` 和 `end` 时间，只保留月份和 `window_size`
- 按月完整计算并缓存，不同时间窗口可以复用同一月份的缓存
- 最终结果按 `start/end` 裁剪

**实现**：
```python
def _get_monthly_trade_clustering_cache_key(
    file_path: str,
    window_size: int,
    initial_state: Optional[Dict[str, Any]] = None,  # 类似 VPIN 的 prev_bucket_state
) -> str:
    """生成按月 Trade Clustering 缓存的键（不包含 start/end）"""
    path = Path(file_path)
    month_str = path.stem.split("_")[-1] if "_" in path.stem else path.stem
    key_str = f"trade_clustering_monthly_{month_str}_{window_size}"
    
    # 如果 initial_state 不为空，将其信息加入缓存键
    if initial_state is not None:
        state_str = json.dumps({
            "current_run_side": initial_state.get("current_run_side"),
            "current_run_length": round(initial_state.get("current_run_length", 0), 6),
            # ... 其他状态信息
        }, sort_keys=True)
        key_str = f"{key_str}_state_{state_str}"
    
    return hashlib.md5(key_str.encode()).hexdigest()
```

### 2. 流式处理优化（类似 VPIN）

**建议**：
- 实现半流式处理：每次只加载当前月+前一个月的数据
- 维护跨月连续性状态，确保 Trade Clustering 计算的正确性
- 支持自动从前一个月加载状态（如果前一个月存在）

### 3. 标准缓存 vs 状态缓存

**建议**：
- 标准缓存：只存储 `final_state`（不存储中间结果），节省存储空间
- 状态缓存：存储完整结果（DataFrame + final_state），用于加速带上下文的查询

## 四、总结

1. **Trade Clustering 可以捕捉冰山单子的特征**，但不能直接识别冰山单子
2. **需要结合其他特征**（交易量、价格变化等）才能更好地识别冰山单子
3. **缓存策略需要优化**，类似 VPIN 的优化方案
4. **流式处理需要优化**，支持半流式处理，减少内存占用

