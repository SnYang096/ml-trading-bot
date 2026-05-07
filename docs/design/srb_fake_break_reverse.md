# SRB 假突破反手（Fake-Break Reverse）— 归档

> **状态：DEPRECATED（2026-04-20）**
>
> SRB 已回归纯粹的"真突破 → 顺势延续"。假破反手语义迁移至：
> - **FBF**（`config/strategies/fbf/`）：机器学习式事前预判假破反手。
> - **策略 X**（`docs/design/strategy_x_hub_rebound.md`）：中枢砸盘 + 低点反弹（规划中）。
>
> 废弃原因（实证）：`mode=reverse` 与 `mode=re_entry` 两组 rolling sim 都**显著跑输**
> 纯突破基线（BASE total_R ≈ 249 vs reverse/re_entry 版本；详见
> `docs/z实验_005_统一研究/` 下 20260419+ 报告）。核心问题：
> 1. 执行侧锯齿：confirm_k 无论 1/2/3 都会被碎震撑阻洗出。
> 2. 占用 slot：即使单笔 meanR 不差，"反手侦查"期间会挤掉正面突破机会。
> 3. 链式反转 bug 的发现也说明这个机制的状态管理过于复杂，复杂度与收益不匹配。
>
> 结论：反手语义必须由**独立**的策略（FBF / 策略 X）负责，不再作为 SRB 的补丁。
>
> 下文保留原设计供历史参考。

---

## 原版内容（历史）

> 原始版本: v3 | 引入 `mode: reverse | re_entry` 双路径

---

## 0. 两种 mode 并存

| mode | 触发后方向 | 语义 | 默认 |
|------|-----------|------|------|
| `reverse` | **反向** | 突破失败，反向才是真趋势（如 LONG 突破 resistance 被 SL 后，价格站稳 resistance 下方 → 做 SHORT） | ✅ 默认 |
| `re_entry` | **同向** | 主力砸坑扫损后价格回 SR 正确侧 → 再做原方向（抗洗盘） | 可切换 |

两种语义对应不同的市场画像：
- **reverse** 更符合"假突破"命名：突破被拒绝 = 方向错了，直接翻仓。
- **re_entry** 更符合"stop hunt 爬坑"：主力做了假动作扫损，真趋势仍在原方向。

二者**不能同时触发**（同一 candidate 只会用一次），按配置 `fake_break_reverse.mode` 选择。

## 0.0 原始意图（re_entry 模式背景）

> "我本来的意图是抓到真正的大方向、有利润的方向。
> 趋势启动前，主力有的时候会反向砸一个大坑——这个策略要有从坑里面爬出来的能力。"

上面这段话对应的是 **`mode=re_entry`** 语义。典型模式：主力在关键支撑下方制造恐慌
（stop hunt / liquidity grab），扫完止损后迅速拉升——如果策略只做"突破延续"，被扫损
后就死了；有再入场能力则能在坑里翻身，抓住真正的大方向。

## 0.1 reverse 模式背景

2026-04 实验阶段经验：SRB 强势突破成功率不足时，被 SL 后**继续反向走**的情况比
"回到 SR 正确侧"的"爬坑"更常见。`mode=reverse` 直接把失败转化为反向机会，
与 fake-break 的字面语义对齐，是当前 SRB 在多品种回测中的首选路径。

---

## 0.1 Shadow Report Findings（run `20260416_145229`）

基于 `results/srb/slow-rolling-sim/_rolling_sim/20260416_145229/srb_reverse_shadow_candidates.csv`
对 63 个 `srb_reverse_expired` 候选做了 10-bar / `confirm_k=3` 的只读回放：

| 语义锚点 | would_trigger |
|----------|---------------|
| `entry_price` | `0 / 63` |
| `true_sr_level`（用 `swing_sr_levels()` 重算） | `0 / 63` |
| `stop_hunt_extreme` | `61 / 63` |

结论：

1. **问题不只是 `entry_price` 过严。**
   即便改成“真实 SR level”，在当前 `10 bars + 3 confirms` 的确认规则下也依然是 `0 / 63`。
2. **`stop_hunt_extreme` 又过于宽松。**
   几乎所有候选都能触发，说明它更适合作为诊断字段或辅助条件，不能单独充当主锚点。
3. **优先要改的是“确认时机/确认规则”，其次才是锚点。**
   当前实现要求“止损后在 10 根 2H bar 内，连续 3 根收回到锚点正确侧”，
   这更像“趋势重新确立”而不是“坑里爬出来”的早期恢复信号，显著偏晚。

因此，下一版语义不应再是：

```text
止损后 -> 等待价格重新站回 entry_price / SR level 正确侧 3 根 bar
```

而应转为：

```text
止损后 -> 先识别 stop-hunt extreme / reclaim 事件
      -> 再用真实 SR level 作为方向确认或二次过滤
```

---

## 0.2 SR Level 来源盘点与选型

### 可用 SR 价位来源

| # | 来源 | 代码 / 特征名 | 输出 | 性质 | 在 feature store 中 | event_backtest 可用性 |
|---|------|---------------|------|------|--------------------|-----------------------|
| A | **Swing range** | `swing_sr_levels()` in `srb_regime.py` | `srb_sr_support` (min low), `srb_sr_resistance` (max high) | lookback 根 K 线的极值范围，**实际价格** | 否 — 运行时注入 | 已注入到 `primary_features`（`sr_structural_exit.enabled` 或 add policy 开启时） |
| B | **Volume-profile POC/HAL** | `poc_hal_features_f` → `sr_strength_max_f` | `poc`, `hal_high`, `hal_low`（归一化）→ `sr_strength_max`, `dist_to_nearest_sr`, `direction_to_nearest_sr` | 160-bar 滚动 volume profile 的 POC 和 HAL 边界，归一化为 `(level - close) / ATR` | `sr_strength_max` / `dist_to_nearest_sr` / `direction_to_nearest_sr` 在 store；`poc` / `hal_*` 是中间列，可能不在最终 parquet | `dist_to_nearest_sr` + `direction_to_nearest_sr` 可用 → 可反算近似价格 |
| C | **Footprint bar-level** | `compute_kline_footprint_features` | `fp_poc`, `fp_hvn`, `fp_lvn`, `fp_vah`, `fp_val` | **单根 K 线**的 tick-level volume profile 价位 | 是 | 直接从特征行读取 |
| D | **VWAP** | `vwap_position_f`, `macro_tp_vwap_1200_position` | 距 VWAP 的偏离度 | 动态均值回归锚点 | 是 | 可用（距离，非 SR 水平位） |

### 选型决策

**`true_sr_level` 用于反手二次确认 → 选 A (swing range)，辅以 B (nearest SR distance) 做质量过滤。**

理由：

1. **语义匹配**：反手确认需要的是"价格是否重新站回被突破的那个结构区间"。
   `swing_sr_levels()` 的 min(low) / max(high) 正是最近的结构区间边界——
   这和原 Bot `calc_break` 用 SR level 穿越做确认/否决是同一语义。

2. **因果性**：`swing_sr_levels()` 只看 `ts` 前的已收盘 bar，不存在未来信息泄漏。

3. **入场时冻结**：`true_sr_level` 应在**入场时冻结**（记录入场那根 bar 的 swing range），
   而不是止损后重算。原因：
   - 入场时的 SR 才是"被突破的结构"；止损后的 SR 已经包含了止损 bar 本身，语义漂移。
   - Shadow report 发现止损后重算的 `true_sr_level` 偏移太远（因为止损 bar 的极值被纳入了 swing range）。

4. **方向映射**：
   - 入场做多 → `true_sr_level = srb_sr_support`（支撑位 = 被突破的底部结构）
   - 入场做空 → `true_sr_level = srb_sr_resistance`（阻力位 = 被突破的顶部结构）
   - 反手确认 = 价格重新站回该 level 的"正确侧"

5. **质量过滤（Phase 2）**：`sr_strength_max` 可作为"这个 SR level 够不够强"的门控 —
   弱 SR 位不值得反手。Phase 1 暂不引入，先拿基线。

**不选 C (footprint bar-level) 的原因**：
`fp_poc` 是单根 K 线粒度，代表的是一根 2H bar 内的成交聚集区，而不是多 bar 的结构价位。
它更适合做微观入场优化（Phase 2 调 entry price），而不是"被突破的结构"的代理。

**不选 D (VWAP) 的原因**：
VWAP 是动态均值回归锚点，它的意义是"公允价值"而不是"结构边界"，
和"假突破反手"需要的"被突破/被恢复的结构位"语义不匹配。

### 入场时的记录方式

```python
# open_position() 之后（SRB only），将 true_sr_level 冻结到 pos 字典：
if archetype == "srb":
    side = pos["side"].upper()
    if side in ("LONG", "BUY"):
        pos["_srb_true_sr_level"] = float(features.get("srb_sr_support", entry_price))
    else:
        pos["_srb_true_sr_level"] = float(features.get("srb_sr_resistance", entry_price))
```

如果 `srb_sr_support` / `srb_sr_resistance` 尚未注入（`sr_structural_exit.enabled = false` 且
add policy 也不触发 regime 计算），则需要在 `maybe_inject_srb_experiment_features` 里
增加 `need_sr = True` 的条件（当 `fake_break_reverse.enabled` 时也要算 SR）。
这样 `swing_sr_levels()` 的结果就会在入场前就绑定到 `primary_features` 中。

### 窄窗噪声与宽窗锚点 fallback（Phase 2）

当 `fake_break_reverse.true_sr_wide_fallback_atr` 设为 **N > 0** 时：

- **做多**：若 `|entry_px − srb_sr_support| < N × ATR`，则 `_srb_true_sr_level` 改用 `srb_sr_support_wide`
  （窄窗太近视为噪声；“二期”结构用宽窗 swing）。
- **做空**：若 `|entry_px − srb_sr_resistance| < N × ATR`，则改用 `srb_sr_resistance_wide`。

实现：`pick_srb_true_sr_level()`（`src/time_series_model/live/srb_regime.py`），事件回测在 `open_position` 后与 live `TradeIntent.strategy_specific["srb_true_sr_level"]` / `build_position_dict` 共用同一公式。

### 宽窗入场屏蔽 `sr_wide_entry_guard`

顶层配置（`config/strategies/srb/archetypes/execution.yaml`）：

- `enabled`: 是否启用。
- `min_distance_atr`: 现价到**反向**宽窗 SR 的最小间隔（倍数 × 当前 primary ATR）；低于则拒绝**新开仓**。
- `apply_to_new_only`: 语义由执行层保证（事件回测仅拦 PCM 新开仓；live `decide()` 信号路径等价于新开仓）。

逻辑：`should_reject_srb_wide_entry()` — LONG 看上方 `srb_sr_resistance_wide`，SHORT 看下方 `srb_sr_support_wide`。

### `stop_hunt_extreme` 的计算

止损后，在 `_reverse_candidate` 中记录局部极值：

```python
# 止损 bar 附近 ± stop_hunt_window_bars 范围内的极值
# 使用 PositionSimulator 持有的 1min bar 滑动窗口不合适（粒度太细，状态复杂）
# 改用：止损时的 entry_bar (primary TF) 的 high/low + ATR buffer
#
# 做多被止损 → stop_hunt_extreme = sl_price - buffer (止损 bar 的最低点估计)
# 做空被止损 → stop_hunt_extreme = sl_price + buffer (止损 bar 的最高点估计)
#
# buffer = atr_at_entry * stop_hunt_buffer_atr (default 0.3)
#
# Phase 2 可改为在 check_srb_reverse 里用后续 bar 的实际 low/high 动态更新 extreme。
```

---

## 1. 原始 Bot 复盘

### 1.1 核心状态机（`calc_break` L524-684）

```
价格穿越 SR 关键位
  → 候选（_recent_breakouts）
  → 每 bar 检查：
      price 维持在 level 正确侧 → confirm_count++
      price 回到 level 错误侧   → fail_count++
      confirm_count >= confirm_k → 真突破 → confirmed_break_long/short 开仓
      fail_count >= confirm_k    → 假突破 → fake_break_reverse_long/short 反手
      超过 fake_lookahead bars   → 过期清除
```

### 1.2 门控条件

1. **时间窗**：`(当前bar - 候选bar) <= fake_lookahead`（默认 10 根 K）
2. **无持仓**：当前无仓才开反手
3. **订单流确认**：`_orderflow_confirm(long=not is_long)` — 反方向的 CVD/taker/VWAP/volume 加权分 ≥ 动态阈值

### 1.3 原 Bot 的优点

- **方向完全由 SR 穿越决定**，不依赖趋势指标——SR 本身就是"市场共识价位"
- **confirm_k 机制**同时覆盖正向（真突破）和反向（假突破），一套逻辑两用
- **止损基于结构价位**（刚才的 high/low + ATR buffer），语义清晰
- **订单流多维评分**在反手前做了方向确认

### 1.4 原 Bot 的缺陷

| 问题 | 描述 |
|------|------|
| **特征/阈值全部硬编码** | `confirm_k=3`, `fake_lookahead=10`, `wick_ratio=2.0` 等全写死，不同行情统一参数 |
| **无冷却期** | 同一 SR 位可以反复 fake→reverse→fake→reverse |
| **影线加速器过于敏感** | 2H K 线一根长影线就 mark fake，消息面噪声大 |
| **不区分结构强度** | 弱 SR 位和强 SR 位用同一套逻辑 |

---

## 2. SR 天生适应力强吗？需要调特征和阈值吗？

### 2.1 SR 价位本身是自适应的

SR = "市场历史上在此价位反复博弈"。随着新 swing high/low 出现，SR 位自动更新。
这比 EMA/MACD 等趋势指标更"原生自适应"——它反映的是结构，不是平滑。

### 2.2 但"怎么交易 SR"不是自适应的

SR 位虽然自适应，围绕它的**交易参数**却不是：

| 参数 | 在不同行情下的最优值差异 | 是否需要 ML 调 |
|------|------------------------|---------------|
| `confirm_k`（确认根数） | 高波动需要更多根确认，低波动 2-3 根就够 | **是** — ATR percentile 可以驱动 |
| `fake_lookahead`（等待窗口） | 压缩市场假突破回收快（5 bar），趋势市场回踩慢（15 bar） | **是** — 压缩度可驱动 |
| 止损宽度 | 高 ATR 需要更宽，低 ATR 可以更紧 | **是** — 已有 `initial_r` 机制 |
| 是否开仓 / 是否反手 | 结构强度高的 SR 位假突破信号更可信 | **是** — `sr_strength_max` 已在 prefilter |
| 反手后目标 | 压缩区间反手目标是区间另一侧；趋势启动反手目标是延续 | **是** — 需要不同的退出策略 |

### 2.3 结论：ML 框架对 SR 的增强点

**不是改变 SR 的核心逻辑（穿越 → 确认/否决 → 开仓/反手），
而是让原来硬编码的"旋钮"随行情自动调节。**

原 Bot 的 `confirm_k=3` 在所有行情下都是 3——ML 框架可以做到：
- 高波动 / 低 sr_strength 时自动放宽到 5
- 低波动 / 高 sr_strength 时收紧到 2
- 主力扫损后的特征签名（volume spike + CVD 反转 + 价格急速回归）被 gate/prefilter 识别后，
  自动降低反手门槛

**这就是 ML 对 SR 最大的增强——不是"发明新策略"，而是"让原来的好策略在不同行情下都用最优参数"。**

---

## 3. 架构分析：为什么能搬

### 3.1 event_backtest 就是 bar-by-bar

`event_backtest.py` 的 `PositionSimulator` 逐 bar 遍历：检查信号、开仓、管理持仓、止损/止盈。
这与原 Bot 的 `next()` 循环在语义上是对齐的。

**需要增加的状态**：
- `_reverse_candidate`: 当 SRB 仓被止损时，记录 `{sr_level, original_side, sl_bar, sl_price}`
- 后续每 bar 检查是否满足反手条件

### 3.2 方向来源应改为 SR 穿越驱动

**现状**：SRB 方向主要来自 `archetypes/direction.yaml` 与研究管线里的 `direction_tuning`（旧 YAML 占位键 `direction_stack` 已移除且无代码消费）。当规则等价为仅靠 ema200/MACD 给出 LONG-only 时，假突破下方后想做 SHORT 反手就会被禁。

**目标**：SRB 的方向应由 **SR 穿越事件** 驱动，而非趋势指标：
- 价格向上穿越阻力 → LONG 候选
- 价格向下穿越支撑 → SHORT 候选
- 确认 → 同向开仓；否决 → 反向开仓

趋势指标（ema200/MACD）不应决定方向，但可以作为**权重/置信度调节器**：
- 顺大势的突破/反手 → 正常 size
- 逆大势的突破/反手 → 缩 size 或增加确认根数

> 这个改动范围较大（需要改 direction.yaml 和 event_backtest 的信号消费逻辑），
> 可以分阶段做：Phase 1 先在 event_backtest 内部实现反手（不改模型出信号），
> Phase 2 再把方向来源改为 SR 穿越。

### 3.3 止损应 follow 结构价位

**现状**：SRB 止损 = `initial_r × ATR`，被 `guardrail_clip` 裁剪。

**目标**：止损基于"刚才的 swing high/low + buffer"，与原 Bot 的 `_structure_stop` 对齐。
好处：
- 止损有结构意义（"如果价格回到那个 swing 点之上/下，说明我的判断错了"）
- 被结构止损扫出后，如果价格又回来了，反手的语义更清晰

> 同样可以分阶段：Phase 1 先用现有 ATR 止损做反手 MVP，Phase 2 再改结构止损。

---

## 4. 实现设计（修正版）

### 4.1 状态机

在 `PositionSimulator` 中增加反手候选状态，但**不再把 `entry_price` 当作唯一核心锚点**。

```
正常流程：
  模型出信号 → gate/prefilter → 开仓 → 持仓管理 → 止损或止盈

反手扩展：
  SRB 仓被止损（exit_reason = "sl"）
    → 记录 reverse_candidate = {
          sr_level:    原始突破使用的真实 SR level（需在入场时记录）,
          entry_price: 原始入场价（仅诊断，不作主锚点）,
          stop_hunt_extreme: 止损 bar 附近形成的局部极值,
          original_side: 止损仓的方向,
          sl_bar:       止损 bar 索引,
          sl_price:     止损价格,
          used:         False
      }
    → 后续每 bar（在正常信号检查之前）：
        if bar - sl_bar > fake_lookahead:
            reverse_candidate = None  # 过期
        elif 先出现 reclaim 事件：
              （原来做多被扫 → 价格脱离 stop_hunt_low，重新收回坑内）
              （原来做空被扫 → 价格脱离 stop_hunt_high，重新收回坑内）
            且 reclaim_count >= reclaim_k:
                → 标记 recover_stage = true
        elif recover_stage == true
             且 价格重新站回真实 SR level 的正确侧
             且 confirm_count >= confirm_k:
                → 反手开仓（方向 = original_side）
                → reverse_candidate.used = True
```

**关键洞察**：反手的方向 = **原来的方向**（不是相反方向）。
因为场景是：主力砸坑（假突破下方扫止损）→ 价格回升 → 证明原来做多是对的 →
重新做多（而非做空）。这和原 Bot 的 `fake_down_break_reverse_long` 语义一致。

### 4.2 确认逻辑（两阶段，含 SR 来源）

```python
def check_srb_reverse(self, current_price, current_bar_count):
    cand = self._reverse_candidate
    pol  = self._srb_reverse_policy or {}

    bars_since = current_bar_count - cand["sl_bar"]
    if bars_since > pol.get("fake_lookahead", 10):
        expire; return None

    reclaim_k = pol.get("reclaim_k", 1)
    confirm_k = pol.get("confirm_k", 2)
    extreme   = cand["stop_hunt_extreme"]   # sl_price +/- buffer
    true_sr   = cand["true_sr_level"]       # 入场时冻结的 swing SR (来源 A)
    is_long   = cand["original_side"] in ("LONG", "BUY")

    # Stage 1: reclaim — 价格脱离 stop-hunt extreme
    if not cand.get("recover_stage"):
        reclaimed = (current_price > extreme) if is_long else (current_price < extreme)
        cand["reclaim_count"] = (cand.get("reclaim_count", 0) + 1) if reclaimed else 0
        if cand["reclaim_count"] >= reclaim_k:
            cand["recover_stage"] = True

    # Stage 2: confirm — 站回 true SR level 正确侧
    if cand.get("recover_stage"):
        confirmed = (current_price > true_sr) if is_long else (current_price < true_sr)
        cand["confirm_count"] = (cand.get("confirm_count", 0) + 1) if confirmed else 0
        if cand["confirm_count"] >= confirm_k:
            -> reverse open (original direction)
```

**SR 来源**（详见 §0.2）：
- `true_sr_level` = 入场时冻结的 `srb_sr_support`（LONG）或 `srb_sr_resistance`（SHORT），
  来自 `swing_sr_levels(lookback=20)` — 最近 20 根 bar 的 min(low) / max(high)。
- `stop_hunt_extreme` = `sl_price +/- atr * stop_hunt_buffer_atr`，
  简单估计止损附近的局部极值（Phase 2 可用实际 bar low/high 替代）。

### 4.3 为什么要两阶段确认

shadow report 已经证明：

- 直接拿 `entry_price`/`true_sr_level` 做 10-bar reclaim，**太慢**，63 个候选一个都回不来。
- 直接拿 `stop_hunt_extreme` 做 trigger，**太松**，61/63 都会开。

所以更合理的语义是：

1. `stop_hunt_extreme` 负责识别“坑里开始爬出来了”；
2. `true_sr_level` 负责确认“回到正确趋势侧了”；
3. `entry_price` 只作为诊断和比较字段保留。

### 4.4 YAML 配置

```yaml
# execution.yaml → fake_break_reverse 块
fake_break_reverse:
  enabled: true
  reclaim_k: 1                  # Stage 1: 脱离 stop-hunt extreme 的最小连续确认
  confirm_k: 2                  # Stage 2: 回到真实 SR 正确侧后的二次确认
  fake_lookahead: 10            # 止损后最多等多少 primary bar
  stop_hunt_buffer_atr: 0.3     # extreme = sl_price ± atr * buffer（估计止损附近的坑底）
  max_reverse_per_level: 1      # 同一 SR 位只反手 1 次
  cooldown_bars: 10             # 反手开仓后冷却期
```

下一版优先验证：

- `reclaim_k` 是否需要 1 或 2；
- `confirm_k` 是否需要从 3 下调到 1~2；
- `true_sr_level` 是否应该在入场时冻结，而不是在止损后重算。

### 4.5 与现有加仓的关系

```
加仓（tag 2）：  持仓有浮盈 → 趋势确认后加码（顺势放大）
反手（本 Phase）：持仓被止损 → 判断是假动作后重新进场（从坑里爬出来）

两者不冲突：先反手建仓 → 如果走出利润 → 可以继续加仓
```

### 4.6 退出机制

反手仓与正常 SRB 仓**完全一致**的止损/trailing 参数。
不缩 size、不收紧止损——Phase 1 先让它和正常仓一样跑，看数据再说。

---

## 5. ML 框架的增强点（Phase 2 展望）

原 Bot 的 `confirm_k=3`, `fake_lookahead=10` 是写死的。
ML 框架可以让这些参数根据行情自适应：

### 5.1 可调参数 × 驱动特征

| 写死参数 | ML 驱动特征 | 调节逻辑 |
|----------|------------|---------|
| `confirm_k` | `atr_percentile`, `sr_strength_max` | 高波动/弱SR → 增加确认根数；低波动/强SR → 减少 |
| `fake_lookahead` | `bpc_volume_compression_pct`, `bb_width_normalized_pct` | 高压缩 → 缩短（假突破回收快）；低压缩 → 延长 |
| 是否允许反手 | `sr_strength_max`, gate score | 弱 SR 位（strength < 0.5）→ 不反手 |
| 反手止损位 | swing high/low + ATR buffer | Phase 2 改为结构止损 |
| 反手 size | 大势顺逆判断（ema1200_position） | 逆大势 → 缩 size（Phase 2） |

### 5.2 gate/prefilter 的角色

现有 prefilter 已经在过滤 `sr_strength_max >= 0.42`。
这个阈值本身就是在保护"只在有意义的 SR 位交易"——反手也应受此保护。

gate 可以增加"反手专用规则"：
- `fake_break_reverse` 意图额外要求 `volume_spike`（止损后出现放量 = 主力吸筹信号）
- `cvd_divergence`（价格新低但 CVD 不新低 = 卖压衰竭）

这些都是 ML 可以在 rolling 中自动调阈值的特征。

### 5.3 方向来源改造（Phase 2）

```yaml
# direction.yaml — 改为 SR 穿越驱动
direction_rules:
  - id: srb_sr_crossing
    method: sr_level_crossing
    description: "SR 穿越决定方向；趋势指标仅作置信度调节"
    sr_source: swing_levels     # 或 volume_profile_poc
    confidence_modifiers:
      - feature: ema_1200_position
        effect: size_scaling    # 顺势 1.0x，逆势 0.7x
```

### 5.4 结构止损改造（Phase 2）

```yaml
# execution.yaml — 改为结构止损
stop_loss:
  type: structural            # 替代 trailing
  primary: swing_level        # 止损 = 最近反向 swing + buffer
  buffer_atr: 0.3
  fallback: atr_multiple      # swing 不可用时退化为 ATR
  fallback_initial_r: 6.0
```

---

## 6. 实现路线

### Phase 1：反手 MVP（本次实现）

| 步骤 | 描述 |
|------|------|
| 6.1 | `execution.yaml` 增加 `fake_break_reverse` 配置块 |
| 6.2 | `event_backtest.py` `PositionSimulator` 增加 `_reverse_candidate` 状态和检查逻辑 |
| 6.3 | 反手开仓的 tag = `fake_break_reverse`，在 trade CSV 中可区分 |
| 6.4 | funnel 增加 `reverse_attempt` / `reverse_opened` / `reverse_expired` 计数 |
| 6.5 | 单测 |
| 6.6 | rolling 验证 |

### Phase 2：ML 自适应参数

| 步骤 | 描述 |
|------|------|
| 6.7 | `confirm_k` / `fake_lookahead` 由特征驱动（`srb_regime.py` 函数） |
| 6.8 | 反手专用 gate 规则（volume spike + CVD divergence） |
| 6.9 | 方向来源改为 SR 穿越 |
| 6.10 | 止损改为结构价位 |

---

## 7. 防打脸机制（精简版，Phase 1）

| 机制 | 参数 |
|------|------|
| 同一 SR 位只反手 1 次 | `max_reverse_per_level: 1` |
| 反手后冷却 | `cooldown_bars: 10` |
| 时间窗口 | `fake_lookahead: 10` — 超过 10 bar 不再反手 |
| 必须确认 | `confirm_k: 3` — 价格连续 3 bar 回到 SR 正确侧 |

Phase 1 **不做**的事：size 缩减、止损收紧、regime 限制。
先拿到干净的基线数据，再决定是否需要这些安全网。

---

## 附录：原始 Bot 代码索引

| 函数 | 行号 | 作用 |
|------|------|------|
| `calc_break` | L524-684 | 突破确认 / fake 判定 / 反手开仓 |
| `wick_fake_break_check_for_last_candidate` | L686-736 | 影线预判假突破 |
| `_orderflow_confirm` | L161-271 | 多维订单流评分门控 |
| `find_candidate_break` | L760-826 | SR 穿越检测 |
| `_enter_long` / `_enter_short` | L860-975 | 下单（tag 区分 confirmed / fake_reverse） |
| `execute_position_management_strategies` | L415-522 | 回踩加仓 / 时间止损 / trailing |
