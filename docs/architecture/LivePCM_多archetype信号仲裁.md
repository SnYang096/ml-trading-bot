# LivePCM — 多 Archetype 信号仲裁层

## 1. 定位

LivePCM（Live Portfolio Control Manager）是实盘链路中的 **信号仲裁层**，
位于 `OrderFlowListener` 和各个策略引擎（BPC / ME / …）之间。

```
OrderFlowListener._handle_features(features, symbol)
  └→ LivePCM.decide(features, symbol)          ← 仲裁层
       ├─ BPCLiveStrategy.decide() → List[TradeIntent]
       ├─ MELiveStrategy.decide()  → List[TradeIntent]   (未来)
       ├─ 合并候选信号
       ├─ 固定优先级 + Evidence 排序 → 选最优
       ├─ Slot 检查 → 允许 / 拒绝
       └→ 返回 List[TradeIntent]（0 或 1 个）
```

**设计原则**：  
- 单策略时行为等价于直接挂 BPCLiveStrategy（零额外开销）
- 实现 `decide(*, features, symbol, bars=None)` 接口，对 OrderFlowListener 完全透明
- 不侵入已有策略代码——通过注册机制组合

## 2. 仲裁算法：固定优先级 + Evidence

### 2.1 设计决策

早期版本使用 AOS（Edge × Evidence）排序，但 Edge 来自历史统计，
**历史有效 ≠ 未来有效**，引入额外不确定性。

简化方案：
- **不同 archetype** → 按条件严格性固定排序（越严格越优先）
- **同 archetype 跨 symbol** → 比 Evidence Score（同 archetype 的 evidence 可比）

### 2.2 默认优先级

```
Reversal > ME > BPC
```

| 优先级 | Archetype | 条件严格性 | 说明 |
|--------|-----------|------------|------|
| 0（最高） | Reversal | 最严格 | 需同时满足趋势耗竭 + 反转确认 |
| 1 | ME | 中等 | 需确认动量扩张 |
| 2（最低） | BPC | 最宽松 | 布林带压缩突破，最常见 |

### 2.3 排序键

```python
sort_key = (priority_rank, -evidence)
```

- `priority_rank`：越小越优先（Reversal=0, ME=1, BPC=2）
- `evidence`：同优先级时，Evidence Score 越大越优先
- `confidence=None` → 默认 0.5
- 未知 archetype（不在优先级列表中）→ 排到最后

### 2.4 示例

| 信号 | Priority | Evidence | sort_key | 结果 |
|------|----------|----------|----------|------|
| BPC (confidence=1.0) | 2 | 1.0 | (2, -1.0) | |
| ME (confidence=0.6) | 1 | 0.6 | (1, -0.6) | |
| Reversal (confidence=0.5) | 0 | 0.5 | **(0, -0.5)** | **胜出** |

同 archetype 示例：

| 信号 | Priority | Evidence | sort_key | 结果 |
|------|----------|----------|----------|------|
| BPC-BTCUSDT (confidence=0.7) | 2 | 0.7 | (2, -0.7) | |
| BPC-ETHUSDT (confidence=0.9) | 2 | 0.9 | **(2, -0.9)** | **胜出** |

## 3. Slot 控制

可选配置 `get_open_slot_count` 回调：
- 提供时：当前已占用 slot >= max_slots → 拒绝新信号
- 不提供时：不做跨 symbol slot 限制（依赖下游 PositionManager）

## 4. 决策流程（伪代码）

```python
def decide(features, symbol):
    # 1. 收集所有策略的候选信号
    all_intents = []
    for strategy in registered_strategies:
        intents = strategy.decide(features, symbol)
        all_intents.extend(intents)

    if len(all_intents) == 0:
        return []

    # 2. 快速路径：单候选直接返回
    if len(all_intents) == 1:
        return check_slot(all_intents[0])

    # 3. 固定优先级 + Evidence 排序
    def sort_key(intent):
        rank = priority_list.index(intent.archetype)  # 越小越优先
        evidence = intent.confidence or 0.5
        return (rank, -evidence)

    best = min(all_intents, key=sort_key)

    # 4. Slot 检查
    return check_slot(best)
```

## 5. 文件位置

| 文件 | 说明 |
|------|------|
| `src/time_series_model/portfolio/live_pcm.py` | LivePCM 实现 |
| `scripts/run_live.py` → `_setup_bpc()` | 接入点：构造 LivePCM 并注入给 listener |
| `tests/unit/test_live_pcm.py` | 单元测试（24 个 case） |
| `tests/unit/test_live_pcm_smoke.py` | 冒烟测试（3 个 case） |

## 6. 扩展 ME 时的操作步骤

```python
# 1. 实现 MELiveStrategy（同 BPCLiveStrategy 接口）
me = MELiveStrategy(...)
me.load_configs()

# 2. 注册进 PCM
pcm.register("me", me)

# 3. （可选）如需自定义优先级顺序：
# pcm = LivePCM(archetype_priority=["Reversal", "ME", "BPC"])
# 默认已是 Reversal > ME > BPC，无需额外配置

# 完成。无需改动 run_live.py 的其他代码。
```

## 7. 测试覆盖

### 单元测试（24 个）

| 场景 | 测试 |
|------|------|
| 单策略透传 | `test_single_strategy_passthrough` |
| 无信号返回空 | `test_single_strategy_no_signal` |
| 无注册策略返回空 | `test_no_registered_strategy` |
| Reversal 胜 ME | `test_reversal_beats_me` |
| ME 胜 BPC | `test_me_beats_bpc` |
| Reversal 胜 BPC | `test_reversal_beats_bpc` |
| 三者同时触发 Reversal 胜 | `test_all_three_reversal_wins` |
| 同优先级比 Evidence | `test_same_priority_compare_evidence` |
| 一个策略出信号一个静默 | `test_one_strategy_fires_one_silent` |
| confidence=None 默认 0.5 | `test_confidence_none_defaults_to_half` |
| 自定义优先级顺序 | `test_custom_priority_order` |
| 未知 archetype 排最后 | `test_unknown_archetype_lowest_priority` |
| Slot 满拒绝 | `test_slot_full_rejects_single` |
| Slot 有空位允许 | `test_slot_available_allows` |
| 无 slot 回调不限制 | `test_no_slot_callback_always_allows` |
| Slot 满拒绝多策略胜者 | `test_slot_full_rejects_multi_strategy_winner` |
| 策略异常不崩溃 | `test_strategy_error_does_not_crash` |
| 所有策略异常返回空 | `test_all_strategies_error_returns_empty` |
| 注册/注销 | `test_register_unregister` |
| quantiles 透传 | `test_set_quantiles_transparent` |
| quantiles_from_df 透传 | `test_set_quantiles_from_df_transparent` |
| load_all_configs 透传 | `test_load_all_configs` |
| archetype_priority 属性 | `test_archetype_priority_property` |
| 默认优先级 | `test_default_priority` |

### 冒烟测试（3 个）

| 场景 | 测试 |
|------|------|
| _setup_bpc 返回 LivePCM | `test_setup_bpc_returns_live_pcm` |
| quantiles 透传 | `test_live_pcm_quantiles_passthrough` |
| decide 委托给 BPC | `test_live_pcm_decide_delegates_to_bpc` |
