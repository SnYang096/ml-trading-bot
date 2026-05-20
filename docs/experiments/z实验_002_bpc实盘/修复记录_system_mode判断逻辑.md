# 🔧 System Mode 判断逻辑修复记录

## 问题描述

**日期**: 2026-02-12

### 发现的 Bug

系统在判断启动模式时，只检查 `ticks_1min` 数据，**完全忽略了 `features_4h`**：

```python
# ❌ 原有逻辑（第 96-106 行）
ticks_1min = warmup_data.get("ticks_1min", pd.DataFrame())

if len(ticks_1min) == 0:
    return ModeDecision(
        mode=SystemMode.OFFLINE,
        reason="No warmup data available",
        bar_count=0,
        data_coverage_hours=0.0,
    )
```

### 问题影响

- ✅ **Feature Store 数据正常**：成功加载 109 条 4h 特征（= 436 小时覆盖）
- ❌ **判断逻辑缺陷**：因为 `ticks_1min` 为空，直接判定为 `OFFLINE`
- ❌ **结果**：明明有充足的特征数据，却无法启动实盘系统

---

## 修复方案

### 核心思路

让 `decide_mode` 方法能够：
1. **优先使用** `ticks_1min` 计算数据覆盖（原有逻辑）
2. **回退机制**：如果 `ticks_1min` 为空，从 `features_4h` 推算数据覆盖
3. **等效计算**：109 条 4h bar × 240 分钟 = 26,160 等效 1min bars

### 修复后的逻辑

```python
# ✅ 修复后的逻辑（第 95-145 行）
ticks_1min = warmup_data.get("ticks_1min", pd.DataFrame())
features_4h = warmup_data.get("features_4h", pd.DataFrame())

# 优先使用 ticks_1min，如果为空则尝试从 features_4h 推算
if len(ticks_1min) == 0 and len(features_4h) == 0:
    return ModeDecision(
        mode=SystemMode.OFFLINE,
        reason="No warmup data available (no ticks_1min or features_4h)",
        bar_count=0,
        data_coverage_hours=0.0,
    )

# 如果没有 ticks_1min 但有 features_4h，从 features_4h 推算数据覆盖
if len(ticks_1min) == 0 and len(features_4h) > 0:
    logger.info(
        f"📊 No ticks_1min data, using features_4h to estimate coverage: "
        f"{len(features_4h)} 4h bars"
    )
    # 109 条 4h 特征 = 109 × 240 分钟 = 26,160 分钟的覆盖
    bar_count = len(features_4h) * 240  # 每个 4h bar = 240 个 1min bar
    
    # 从 features_4h 的 timestamp 列计算覆盖时长
    if "timestamp" in features_4h.columns:
        first_ts = pd.to_datetime(features_4h["timestamp"].iloc[0])
        last_ts = pd.to_datetime(features_4h["timestamp"].iloc[-1])
        coverage_hours = (last_ts - first_ts).total_seconds() / 3600
    else:
        coverage_hours = len(features_4h) * 4
    
    # 使用 features_4h 时不检查分钟级缺口（粒度不同）
    missing_periods = []
    has_large_gap = False
else:
    # 使用 ticks_1min 计算（原有逻辑）
    bar_count = len(ticks_1min)
    # ... 计算 coverage_hours 和 missing_periods
```

### 判断阈值（不变）

- **NORMAL 模式**：≥ 240 bars（4 小时）且无大缺口
- **DEGRADED 模式**：120-240 bars（2-4 小时）或有大缺口
- **OFFLINE 模式**：< 120 bars（2 小时）

---

## 验证结果

### 测试脚本

创建了 [`test_system_mode_fix.py`](file:///home/yin/trading/ml_trading_bot/docs/z实验_002_bpc实盘/test_system_mode_fix.py) 验证 4 个场景：

1. ✅ **无任何数据** → `OFFLINE`
2. ✅ **只有 features_4h (109条)** → `NORMAL` (26,160 bars, 432h)
3. ✅ **features_4h 不足 (20条)** → `NORMAL` (4,800 bars, 76h)
4. ✅ **同时有 ticks_1min 和 features_4h** → 优先使用 `ticks_1min`

### 测试输出

```
============================================================
测试场景 2: 只有 features_4h (109条 = 436小时)
============================================================
✓ 判定结果: NORMAL
  原因: Data complete: 26160 bars, 432.00h coverage
  等效 bars: 26160 (109 × 240 = 26,160)
  覆盖时长: 432.00 小时

============================================================
✅ 所有测试通过！
============================================================
```

---

## 修复文件清单

| 文件路径 | 修改内容 | 行数变化 |
|---------|---------|---------|
| [`src/live_data_stream/system_mode.py`](file:///home/yin/trading/ml_trading_bot/src/live_data_stream/system_mode.py) | 添加 `features_4h` 回退逻辑 | +41, -17 |
| [`docs/z实验_002_bpc实盘/实盘启动命令.md`](file:///home/yin/trading/ml_trading_bot/docs/z实验_002_bpc实盘/实盘启动命令.md) | 更新启动流程说明 | +9 |
| [`docs/z实验_002_bpc实盘/test_system_mode_fix.py`](file:///home/yin/trading/ml_trading_bot/docs/z实验_002_bpc实盘/test_system_mode_fix.py) | 新增测试脚本 | +128 |

---

## 下一步

> **策略B 更新（2026-02-12）**：已采用方案 3（Copy Ticks + 4h Wait），live 不再依赖 Feature Store。
> `features_4h` 回退逻辑作为历史修复记录保留，但实际运行时不会再触发（warmup 数据全部来自历史 ticks）。

1. **准备 warmup 数据**：执行 `bash live/scripts/prepare_warmup_ticks.sh highcap 6`，下载 6 个月历史 ticks
2. **启动实盘**：执行 `./live/scripts/start_live.sh highcap`
3. **等待 4h**：系统自动从 OFFLINE → DEGRADED → NORMAL，基于实时数据累积量判定
4. **监控运行**：验证 WebSocket 连接正常，特征计算基于 ticks/bars 实时重算

---

## 技术细节

### 为什么使用 240 倍数？

- **1 个 4h bar** = 4 小时 = 4 × 60 分钟 = **240 分钟**
- 系统的阈值以 1min bars 为单位：
  - NORMAL ≥ 240 bars = 4 小时
  - DEGRADED ≥ 120 bars = 2 小时
- 因此，109 条 4h 特征 = **109 × 240 = 26,160 等效 1min bars** >> 240 阈值

### 为什么不检查 features_4h 的缺口？

- `_detect_gaps` 方法检查相邻时间戳的**分钟级差异**
- 对于 4h 粒度的数据，相邻时间戳间隔本身就是 240 分钟
- 检查缺口会产生误报，因此在使用 `features_4h` 时跳过缺口检测

---

## 参考

- Bug 发现者：用户反馈
- 修复日期：2026-02-12
- 相关 Issue：Feature Store 数据加载正常但系统判定为 OFFLINE
