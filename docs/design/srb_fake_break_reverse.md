# SRB 假突破反手（Fake-Break Reverse）设计文档

> 版本: v2 | 基于 tag `2` (f4427e1)
> 对照: `docs/archive/rule_based_strategies/sr_breakout_bot.py`

---

## 0. 核心意图（来自原作者）

> "我本来的意图是抓到真正的大方向、有利润的方向。
> 趋势启动前，主力有的时候会反向砸一个大坑——这个策略要有从坑里面爬出来的能力。"

**反手不是"震荡区间博弈回归"，而是"识别主力假动作，跟上真正的趋势启动"。**
典型模式：主力在关键支撑下方制造恐慌（stop hunt / liquidity grab），扫完止损后迅速拉升——
如果策略只做"突破延续"，被扫损后就死了；有反手能力则能在坑里翻身，抓住真正的大方向。

这改变了整个设计的出发点——反手不应只限于"低 ADX 震荡区"，
它在**任何 regime** 都可能发生，关键是判断"这次止损是不是主力做的假动作"。

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

**现状**：SRB 方向由 `direction_stack (ema200/MACD)` 决定 — 这限制了反手能力，
因为如果 ema200 说 LONG-only，假突破下方后想做 SHORT 反手就被禁了。

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

## 4. 实现设计（Phase 1：反手 MVP）

### 4.1 状态机

在 `PositionSimulator` 中增加反手候选状态：

```
正常流程：
  模型出信号 → gate/prefilter → 开仓 → 持仓管理 → 止损或止盈

反手扩展：
  SRB 仓被止损（exit_reason = "sl"）
    → 记录 reverse_candidate = {
          sr_level:    止损仓的 entry_price（近似 SR 位）,
          original_side: 止损仓的方向,
          sl_bar:       止损 bar 索引,
          sl_price:     止损价格,
          used:         False
      }
    → 后续每 bar（在正常信号检查之前）：
        if bar - sl_bar > fake_lookahead:
            reverse_candidate = None  # 过期
        elif 价格回到 sr_level 的"正确侧"
              （原来做多被扫 → 价格回到 sr_level 之上 = 确认假突破后反弹）
              （原来做空被扫 → 价格回到 sr_level 之下 = 确认假突破后下跌）
            且 confirm_count >= confirm_k:
                → 反手开仓（方向 = original_side，因为"假突破"意味着原方向才是对的）
                → reverse_candidate.used = True
```

**关键洞察**：反手的方向 = **原来的方向**（不是相反方向）。
因为场景是：主力砸坑（假突破下方扫止损）→ 价格回升 → 证明原来做多是对的 →
重新做多（而非做空）。这和原 Bot 的 `fake_down_break_reverse_long` 语义一致。

### 4.2 确认逻辑

```python
# 在 PositionSimulator.step() 里，检查 reverse_candidate
if self._reverse_candidate and not self._reverse_candidate['used']:
    cand = self._reverse_candidate
    bars_since = current_bar - cand['sl_bar']

    if bars_since > fake_lookahead:
        self._reverse_candidate = None  # 过期
    else:
        sr_level = cand['sr_level']
        if cand['original_side'] == 'LONG':
            # 原来做多被扫，如果价格回到 sr_level 之上 → 确认假突破
            if current_price > sr_level:
                cand['confirm_count'] = cand.get('confirm_count', 0) + 1
            else:
                cand['confirm_count'] = 0  # 重置
        else:  # SHORT
            if current_price < sr_level:
                cand['confirm_count'] = cand.get('confirm_count', 0) + 1
            else:
                cand['confirm_count'] = 0

        if cand['confirm_count'] >= confirm_k:
            # 反手开仓：方向 = original_side（重新做原来的方向）
            open_position(side=cand['original_side'], tag='fake_break_reverse')
            cand['used'] = True
```

### 4.3 YAML 配置

```yaml
# execution.yaml → fake_break_reverse 块
fake_break_reverse:
  enabled: true
  confirm_k: 3            # 价格回到 SR 正确侧后需确认的 bar 数
  fake_lookahead: 10       # 止损后最多等多少 bar
  max_reverse_per_level: 1 # 同一 SR 位只反手 1 次
  cooldown_bars: 10        # 反手开仓后冷却期（不再对下一次止损立即反手）
```

Phase 1 中**不做**：size 缩减、止损收紧、regime 限制——先跑出基线数据。

### 4.4 与现有加仓的关系

```
加仓（tag 2）：  持仓有浮盈 → 趋势确认后加码（顺势放大）
反手（本 Phase）：持仓被止损 → 判断是假动作后重新进场（从坑里爬出来）

两者不冲突：先反手建仓 → 如果走出利润 → 可以继续加仓
```

### 4.5 退出机制

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
