# SRB SR-cross 状态机 设计文档

> 版本: v1 | 对应标签: `srb-cross-sm`（2026-04-19）
> 对照 legacy: `docs/archive/rule_based_strategies/sr_breakout_bot.py`
> 取代文档: [`srb_fake_break_reverse.md`](./srb_fake_break_reverse.md)（已 deprecated）
>
> ## ⚠️ 2026-04-20 状态更新
>
> SRB 已回归**纯规则突破**（generic_live_strategy + prefilter/gate），本状态机
> **未接入事件回测 / live 路径**。其 `fake → 反向` 路径（`CrossDecision.status == "fake"`）
> 不再由 SRB 消费；保留模块作为：
> 1. **FBF 实验**：fake 检测可作为 ML 的规则 baseline。
> 2. **策略 X**（[`strategy_x_hub_rebound.md`](./strategy_x_hub_rebound.md)）：中枢破位确认可复用同样的 state machine 骨架。
>
> 下文保留原有设计以备复用。

---

## 0. 为什么要换掉"SL-后两段确认反手"

此前 SRB 的反手路径是：**先被扫损 → 在 stop-hunt 坑里登记 `_reverse_candidate` →
两段确认（reclaim extreme → confirm back across SR）→ 反向再开仓**。
在 `slow` 2h / 16 个月 rolling 回测上，观察到三个稳定问题：

1. **反手触发率长期≈0**：`shadow report (20260416)` 显示 63 个 `srb_reverse_expired`
   候选里用各种锚点回放 `would_trigger = 0/63`。问题不在阈值，而是
   **"等 SL 扫掉"本身晚于趋势启动**，等真趋势确认时市场已走出 3–5 ATR。
2. **仓位已平后再反手 → 成本双倍**：一次 `-1R` 的 SL 损失后再进反向，
   在 SOL/XRP 这种频繁假破的 symbol 上 meanR 会被反手首仓 ATR 放大拖到 ≤ -1。
3. **语义 leakage**：`fake_break_reverse` 块里被塞进了 `true_sr_wide_fallback_atr`、
   `stop_hunt_buffer_atr` 等跨概念参数，实际只有 `confirm_k / fake_lookahead` 被读。

新架构把反手判定**前移到"cross 事件发生后的 K 根 primary bar 内"**：
每根 bar 都在评估"这次穿越是真突破还是假突破"，真 → 顺势开，假 → 立即反向开，
**不再等 SL**。设计动机（"主力砸坑 → 爬坑 → 跟真正的大方向"）整体保留，
但实现语义改为**cross-centric + bar-level 二元分支**。

---

## 1. 设计目标（What good looks like）

| 目标 | 度量 |
|------|------|
| 反手不再依赖 SL 先触发 | 0 个 `_reverse_candidate`；所有反手都带 `is_fake_reverse=True` 从 `GenericLiveStrategy.decide()` 出来 |
| 真突破顺势跟得上 | 每根 bar 只要 `confirm_k` 根连续同侧即开；SR 的窄窗优先 |
| 假破反手 meanR 不再 ≤ -1 | SOL/XRP `exit_reason × is_reverse` 对比表 meanR 抬起（见 rolling 实验报告） |
| 保持 prefilter 不变，仍可被 regime / risk budget 盖住 | `decide()` 仍走 prefilter；仅 `fake` 分支跳过 gate/entry_filter（语义已被状态机确认） |
| 代码无 pandas/IO 依赖，易测 | 纯 dataclass + pure function，14 个 unit test 覆盖所有分支 |

---

## 2. 架构概览

```
         primary TF bar 到达
                │
                ▼
  ┌──────────────────────────────────────────────┐
  │   GenericLiveStrategy.decide()               │
  │   (SRB archetype only)                       │
  │                                              │
  │   ① _advance_srb_cross(symbol, features)     │─── 读 close/open/high/low/
  │       → update_cross_state()                 │    volume/volume_ma/
  │       → CrossDecision{status,side,level}     │    sr_{support,resistance}
  │                                              │
  │   ② 按 status 分支：                         │
  │     idle/pending/expired → return []         │
  │     confirmed            → side 覆盖, rule=  │
  │                            srb_cross_conf    │
  │     fake                 → side 覆盖, rule=  │
  │                            srb_cross_fake,   │
  │                            is_fake_reverse=1 │
  │                            跳过 gate / entry │
  │                                              │
  │   ③ 仍走 prefilter + (非 fake 时) gate/entry │
  └──────────────────────────────────────────────┘
                │
                ▼ TradeIntent (strategy_specific={is_fake_reverse, srb_cross_level})
  ┌──────────────────────────────────────────────┐
  │   event_backtest PositionSimulator           │
  │   · is_fake_reverse → 跳过 sr_wide_entry_guard│
  │   · is_fake_reverse → _srb_pos._is_reverse=T │
  │   · try_add_position: breakout_side_only     │
  │   · enforce_position: trailing scale_with_atr│
  └──────────────────────────────────────────────┘
```

核心模块：[`src/time_series_model/live/srb_cross_state_machine.py`](../../src/time_series_model/live/srb_cross_state_machine.py)

---

## 3. 状态机语义

### 3.1 数据结构

```python
@dataclass(frozen=True)
class CrossConfig:
    enabled: bool = True
    confirm_k: int = 3             # 连续 K 根 close 同侧 → confirmed
    fake_lookahead: int = 10       # 候选最多活多少根 bar
    wick_ratio_threshold: float = 2.0   # wick-prior 影线/body 阈值
    low_vol_ratio: float = 0.8     # volume < ratio * volume_ma 视为量能不足
    cooldown_bars: int = 10        # confirmed/fake/expired 后冷却 bar 数
    max_reverse_per_level: int = 1 # 统计用，MVP 未强制

@dataclass
class CrossCandidate:
    direction: str       # 'up' | 'down'
    level: float         # SR level（穿越时锁定）
    bar0: int            # 起候选的 bar_index
    confirm_count: int   # 连续同侧根数（bar0 起始计 1）
    fail_count: int      # 连续反侧根数
    fake_stage: bool     # wick+低量 prior 已触发
    fake_stage_count: int

@dataclass
class CrossDecision:
    status: str   # 'idle' | 'pending' | 'confirmed' | 'fake' | 'expired'
    side:  Optional[str]   # confirmed/fake 时的入场方向（LONG/SHORT）
    level: Optional[float]
```

### 3.2 状态转移

每根 primary bar 调用一次 `update_cross_state()`（pure function）：

```
  候选=None
     │   has_position or bar_index < cooldown_until_bar
     ├───────────────────► status = idle
     │
     │ detect_cross(close_prev, close_curr, support, resistance)
     │   返回 None
     ├───────────────────► status = idle
     │
     │   返回 (direction, level) → 起候选（confirm_count=1）
     ▼
  候选=CrossCandidate(direction, level, bar0=bar_idx)
     │
     │ 本根 close 与 level 比较：
     │   同侧  → confirm_count += 1；fail_count=0；撤销 fake_stage
     │   反侧  → fail_count += 1；fake_stage 存在则 fake_stage_count += 1
     │
     │ wick_fake_prior：
     │   up 方向 & upper_wick≥R·body & volume<L·vma  → fake_stage=True
     │   down 方向类似
     │
     │ 判定优先级：
     │   ① confirm_count  ≥ confirm_k → status=confirmed, side=(up?LONG:SHORT)
     │   ② fake_stage 且 fake_stage_count ≥ confirm_k → status=fake
     │   ③ fail_count    ≥ confirm_k → status=fake (confirmed 反侧)
     │   ④ bar_index - bar0 > fake_lookahead → status=expired
     │   ⑤ 否则 status=pending（保留候选）
     │
     ▼
  候选=None（confirmed/fake/expired 清空；pending 保留推进）
```

### 3.3 `detect_cross` 的边界规则

- **up**：`close_prev ≤ resistance < close_curr` —— 必须真的穿越（不是 touch）
- **down**：`close_curr < support ≤ close_prev`
- **优先级**：resistance > support（up 优先 down），与 legacy
  `find_candidate_break` 的 swing 结构保持一致（MVP 不做 zigzag/POC）。
- **空值**：任一 SR level 或 close 不是有限实数 → 直接返回 None。

### 3.4 为什么 fake 也是 "K 根同侧"（不是 1 根）

早期 legacy 在"首根就反手"被 stop-hunt 快速 V 反抽死过多次（记录于
`docs/archive/rule_based_strategies/sr_breakout_bot.py` 的注释）。本实现要求
**fake_stage_count 或 fail_count 达到 confirm_k 才 fake**，使 confirm 与 fake
在路径长度上对称——这也是 "wick-prior 是入场条件而非触发器" 的原因：
wick-prior 只是让候选更快进入 fake 轨道，但最终还是要 K 根反侧 close 收盘。

### 3.5 cooldown 与去重

状态机本身只维护 candidate + cooldown_until_bar；`cooldown_until_bar` 由调用方
（`GenericLiveStrategy._advance_srb_cross` / `event_backtest`）在收到
`confirmed / fake / expired` 后设为 `bar_index + cfg.cooldown_bars`。
`max_reverse_per_level` 目前仅用于日志统计，不在状态机内强制——
产生同一 level 连续 fake 的场景非常罕见，且 cooldown 已经覆盖。

---

## 4. 与系统其它组件的接线

### 4.1 `GenericLiveStrategy.decide()`

- 在 SRB archetype 的最前面调用 `_advance_srb_cross`（比 prefilter 更早），
  确保状态机在"被 prefilter 挡住的 bar"上也能正确累积 confirm/fail 计数。
- 根据 `CrossDecision.status`：
  - `idle / pending / expired`：直接 return `[]`（不出 intent）。
  - `confirmed`：用 `decision.side` 覆盖方向，`rule_id=srb_cross_confirmed`，
    `strategy_specific.srb_cross_level=decision.level`；正常走
    prefilter → gate → entry_filter → execution。
  - `fake`：用 `decision.side` 覆盖方向，`rule_id=srb_cross_fake_reverse`，
    `strategy_specific.is_fake_reverse=True`；**跳过 gate 与 entry_filter**
    （状态机本身已经是确认层），仍走 prefilter + execution。
    `execution_tags` 中追加 `srb_cross_fake_reverse`。

### 4.2 `event_backtest.PositionSimulator`

- **删除**：`_srb_reverse_policy / _reverse_candidate / _reverse_cooldown_until_bar /
  _last_reverse_status / check_srb_reverse()` —— 旧两段确认路径完全下线。
- **保留**：`_primary_bar_count`（cooldown 计算依赖）。
- **`open_position()`**：
  - 读 `TradeIntent.execution_profile.strategy_specific.is_fake_reverse`。
  - 若为真 → `_srb_pos["_is_reverse"] = True`；
    **`sr_wide_entry_guard` 被跳过**（状态机已给出确认的反手方向，
    再拒绝会产生"确认但不敢开"的语义矛盾）。
  - `pick_srb_true_sr_level` 的宽窗 fallback 阈值从
    `true_sr_level.wide_fallback_atr` 读（默认 2.0）；
    L3 大级别 SR 来源于统一特征 `wide_sr_swing_f`
    （`wide_sr_upper_px` / `wide_sr_lower_px`）。
- **`try_add_position()`**：加 `breakout_side_only` 守卫——
  只有当前价距 `_srb_true_sr_level` 在突破方向侧且 ≥
  `breakout_side_tolerance_atr × ATR` 才允许加仓；在支撑/阻力之间来回震荡时
  不再触发 ATR ladder。
- **`enforce_position()`**（via `position_logic.py`）：trailing 改用
  `scale_with_primary_atr=True` + `min_trail_atr_ratio=0.5`，即
  `trail_base_atr = max(current_primary_atr, entry_atr * 0.5)`；波动下降时
  trailing 同步收紧，避免 "ATR 爆炸那根锁死一个很宽的 trail"。

### 4.3 执行画像

反手 intent 的执行画像**与正向 SRB 完全一致**：同样的 initial_r / activation_r /
trailing / ladder，唯一差异是方向与 `_is_reverse` 标记。这是刻意的——
反手本身是"真趋势启动"，没有理由用更小仓位或更紧 SL。
（FBF-ML 仍并行存在；同一根 bar 可能双开，由 portfolio 层仓位上限兜底。）

---

## 5. 配置（`config/strategies/srb/archetypes/execution.yaml`）

```yaml
sr_cross_state_machine:
  enabled: true
  confirm_k: 3                  # 连续 K 根 close 同侧 → confirmed / fake
  fake_lookahead: 10            # 候选最多存活多少根 primary bar
  wick_ratio_threshold: 2.0     # 影线/body ≥ threshold → 长影线
  low_vol_ratio: 0.8            # volume < ratio × volume_ma → 量能不足
  cooldown_bars: 10             # confirmed/fake/expired 后冷却多少根 bar
  max_reverse_per_level: 1      # 目前只做日志统计

true_sr_level:
  wide_fallback_atr: 2.0        # 窄窗 SR 距入场 < N×ATR → 用 L3 大级别 SR (wide_sr_swing_f) 作 fallback

stop_loss:
  trailing:
    scale_with_primary_atr: true   # 随 current_atr 双向缩放
    min_trail_atr_ratio: 0.5       # 相对 entry_atr 的最小地板
    activation_r: 5.0              # 原 6.0 → 5.0，略微提前激活 trailing

srb_add_position_policy:
  allow_regime_buckets:
    - high_adx_low_er
    - high_adx_high_er
    - low_adx_high_er             # 新增：结构启动期常处于 low_adx_high_er
  breakout_side_only: true
  breakout_side_tolerance_atr: 0.25
```

prefilter 同步调整（`config/strategies/srb/archetypes/prefilter.yaml`）：

```yaml
# 删除
# - spectrum_price_high_freq_ratio <= ...
# - tpc_score_continuation >= ...
# 新增
- bb_width_normalized_pct <= 0.65   # 结构性 SR 多发于"压缩→释放"而非极端波动
- trend_r2_20 <= 0.65               # 极强趋势段里"穿越 SR"本身不具有语义上的反手空间
```

---

## 6. 验证计划

- **单元测试**：`tests/unit/test_srb_cross_state_machine.py` 覆盖
  `detect_cross`（up/down/none）、`update_cross_state` 的
  confirmed / fake-via-fail / fake-via-wick-prior / expired / has-position-idle /
  cooldown-idle —— 共 14 条 case，均为 pure function，无 pandas 依赖。
- **回归测试**：`tests/unit/test_srb_regime.py` 中旧 `check_srb_reverse` 家族
  已被整体移除（代码路径不存在）；其它 SRB 相关 case（featurelist / true_sr /
  structural exit 等）保留并通过。
- **Rolling 回测**：`scripts/auto_research_pipeline.py --config
  config/prod_train_pipeline_2h_slow_srb_only.yaml` 跑 16 个月全量 rolling，
  产物落 `results/srb/research_roll.features_on/_rolling_sim/<ts>`；对比维度：
  `symbol × side × exit_reason × is_reverse` 的 trade count / meanR /
  hit_rate，重点看 SOL/XRP 的 reverse meanR 是否从 -1.0 抬起。
- **实验报告**：`docs/z实验_005_统一研究/SRB_cross_state_machine_20260419.md`。

---

## 7. 已知限制 / 后续

- **MVP 不做 zigzag/POC**：`detect_cross` 只用 `sr_support_level /
  sr_resistance_level` 这两个主特征；后续可把 L2 的 POC 层接进来。
- **`max_reverse_per_level` 未强制**：目前 cooldown 已足够去重；真正要强制需要
  额外维护 `level → last_seen_bar` 表，放到 live state。
- **confirm_k 与 fake_lookahead 对称性**：`confirm_k=3, fake_lookahead=10`
  意味着最多 3 根反侧 + 最多 10 根 pending；这是一个比较宽松的设定，
  观察 SOL/XRP 的 confirmed_rate / expired_rate 之后可再收紧。
- **与 FBF-ML 的关系**：两者可能同一根 bar 同时出 fake-reverse 信号——
  这是刻意保留的（路由 diversification），由 portfolio 层仓位上限兜底。

---

## 附：入口索引

- 状态机实现：`src/time_series_model/live/srb_cross_state_machine.py`
- live 接线：`src/time_series_model/live/generic_live_strategy.py::_advance_srb_cross`
- 回测接线：`scripts/event_backtest.py`（`PositionSimulator.open_position` /
  `try_add_position` / `enforce_position` 周边）
- 仓位字段：`src/time_series_model/live/position_logic.py::build_position_dict` /
  `enforce_position`
- SRB 主 regime 逻辑：`src/time_series_model/live/srb_regime.py`
- 旧文档（deprecated）：[`srb_fake_break_reverse.md`](./srb_fake_break_reverse.md)
