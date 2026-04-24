# 策略 X：中枢砸盘 + 低点反弹（Hub-Break-Rebound）

> **状态（2026-04-21）**：**已废弃**。与 RMR 一并经数据统计无优势，不再作为产品方向；下文保留为历史规格。
>
> **2026-04-20 裁决（背景）**：strict 离线原型触发极稀、放宽即转负；**FBF + 近 SR 过滤** 覆盖相近语义更省复杂度。详见 [`docs/z实验_005_统一研究/FBF_RMR_HubRebound_verdict_20260420.md`](../z实验_005_统一研究/FBF_RMR_HubRebound_verdict_20260420.md)。
>
> **归档入口**：[hub_rebound_incubation.md](../architecture/strategies/hub_rebound_incubation.md)
>
> **历史 Spec Draft（2026-04-20）**：
>
> 目的：孵化一条与 SRB / FBF 并列、**语义干净**的独立交易边 —— 中枢整理 → 决定性向下击穿 → 低点反弹站回的三要素，抓长周期低点做 LONG。镜像可做 SHORT 顶部。

---

## 0. 动机与语义定位

### 0.1 交易边的直觉

> "一个中枢，往下砸盘，最后有反弹上来 —— 这个开仓的点位，会是一个长期低点，非常有优势。"

本质是"**假破底 + 反弹确认**"：中枢已经形成有结构的支撑，向下决定性击穿一般伴随砸盘式流动性清洗（stop hunt / 恐慌抛售），若价格能**快速**站回支撑内侧，说明砸盘被吸收 —— 这种点位在历史上常对应中期甚至长期低点。

### 0.2 与 SRB / FBF 的边界

| 策略 | 交易逻辑 | 行情假设 | 方向 | 触发机制 |
|------|---------|---------|------|---------|
| **SRB** | 真突破 → 顺势延续 | 趋势启动 | 与突破同向 | 规则（sr_cross_state_machine） |
| **FBF** | 事前预判假破 → 反向 | 区间边缘假突破 | 与突破反向 | ML（labels/model） |
| **策略 X** | 中枢砸盘 → 反弹站回 → 做 LONG（镜像 SHORT） | 中枢末端恐慌清洗 | 与砸盘反向 | 规则（三要素状态机） |

三者在"是否预判 / 是否需要反手 / 在 regime 上的偏好"三个维度上互斥：
- SRB 偏好高 ADX + 低 ER 的单边环境。
- FBF 偏好高频假突破环境（ML 学习出来）。
- **策略 X** 偏好长期横盘中枢被一次性打穿后的恐慌释放（ATR 暴涨、成交量尖峰）。

三者可以在同一品种同一时刻都存在信号，由组合层通过 slot gate / exposure cap 调度，不互相覆盖。

---

## 1. 三要素精确定义

### 1.1 中枢（Hub / Consolidation）

用于判定"一段时间内价格在狭窄带宽内反复震荡，累积了有结构的支撑压力"。

候选维度（组合式判定，写 spec 层时列全，实现时挑子集做 MVP）：

| 维度 | 建议特征 | 阈值语义 |
|------|---------|---------|
| 振幅 | `bb_width_normalized_pct` | ≤ P35（压缩） |
| 持续时间 | 满足压缩条件的连续 bar 数 | ≥ `hub_min_bars`（例如 2h 周期下 40 根 ≈ 3.3 天） |
| 方向性 | `trend_r2_20` / `path_efficiency` | 非强趋势（R² ≤ 0.35） |
| ADX | `adx14` | ≤ 20（无方向） |
| SR 稳定性 | `srb_sr_support_wide` 的标准差 / ATR | ≤ 0.3（支撑位稳定） |

**关键参数**：`hub_support_level = srb_sr_support_wide`（中枢下沿）；`hub_resistance_level = srb_sr_resistance_wide`（中枢上沿）。

**落地要求**：在信号 bar t 前，至少 `hub_min_bars` 根 bar 同时满足上述维度的最小子集（如 `bb_width` + 时长 + 非强趋势）。

### 1.2 破位（Break-Down）

中枢下沿被**决定性**向下击穿 —— 区别于一次性插针。判定信号：

| 维度 | 条件 | 语义 |
|------|------|------|
| 收盘破 | `close < hub_support_level - break_buffer_atr × ATR` | 不是长影线假破 |
| 动能 | `abs(close - hub_support_level) ≥ break_min_magnitude_atr × ATR` | 破位幅度够，排除贴线波动 |
| 成交量（可选） | `volume_ratio ≥ break_min_volume_ratio` | 伴随放量（恐慌/清洗） |
| 距离 | `bars_since_hub_end ≤ break_max_bars`（例如 5） | 破位紧跟中枢末端，防陈旧 |

**状态记录**：确认破位时冻结 `break_low_extreme = low`（后续每根 bar 更新为破位期间的最低点）、`break_low_bar = t_break`。

### 1.3 反弹（Rebound / Reclaim）

破位后价格重新站回中枢下沿内侧 —— 砸盘被吸收的确认。

| 维度 | 条件 | 语义 |
|------|------|------|
| 站回 | `close > hub_support_level + rebound_buffer_atr × ATR` | 不是在破位线附近犹豫 |
| 连续 | 满足站回条件连续 `rebound_confirm_k` 根 bar | 过滤假反弹 |
| 时效 | `bars_since_break ≤ rebound_window_bars`（例如 8） | 必须快速反弹，超时则放弃 |
| 低点已过 | 反弹 bar 的 `low > break_low_extreme` 至少一段时间 | 底已筑好 |

---

## 2. 入场 / 止损 / 仓位 / 退出

### 2.1 入场

- **方向**：LONG（镜像 SHORT 做中枢顶部假突破后的回落）。
- **时点**：满足反弹三条件的第一根 bar 收盘后。
- **价格**：当前 bar 收盘价（或下一根 bar 开盘，由执行层决定）。
- **size**：evidence_score 由三要素强度组合（例如 `1.0 × hub_score × break_score × rebound_score`），再经 `rr / stop_pct` sizing。

### 2.2 止损（结构化）

- **SL 锚点**：`break_low_extreme`（破位期间的最低点）。这是最自然的失效位 —— 跌破破位低点意味着反弹无效、中枢彻底失守。
- **SL 价**：`SL = break_low_extreme - sl_buffer_atr × ATR`（例如 0.3 ATR 缓冲）。
- **兜底**：若 `(entry - SL) < min_distance_atr × ATR`（极端压缩），用 `initial_r × ATR` 兜底防被即时洗出。
- **SL 距离宽时**：sizing 自然缩小仓位（`risk / stop_pct`），不做 clip。

### 2.3 加仓

- **触发**：浮盈阶梯（`float_r_ladder_only`），与 ME / SRB 对齐。
- **限制**：加仓方向只有 LONG，不反向加仓；加仓 SL 继承母仓。

### 2.4 退出

三档次并行：

1. **SL / trailing**：按 `expand_with_primary_atr = true` 的 trailing 带宽。
2. **结构化 TP**：首个 TP 锚定 `hub_resistance_level`（中枢上沿）；若突破中枢上沿则切入 SRB 语义（由组合层或信号层转交）。
3. **时间止损**：`max_holding_bars`（例如 2h × 120 = 10 天）防反弹后陷入新的小幅震荡。

---

## 3. 特征工程与数据依赖

新增特征（feature_store 层）：
- `hub_start_bar` / `hub_end_bar` / `hub_min_bars_count`（中枢时长）
- `hub_bb_width_min`（中枢内最窄 BB 宽度）
- `hub_support_level` / `hub_resistance_level`（锚定 wide swing SR）
- `break_low_extreme` / `break_low_bar`（每根 bar 滚动更新）
- `rebound_bars_since_break` / `rebound_confirmed`

状态机实现：参考 `src/time_series_model/live/srb_cross_state_machine.py` 的形式，写一个独立 `hub_rebound_state_machine.py`，三状态：
- `IDLE` → 中枢条件满足 → `HUB_READY`
- `HUB_READY` → 破位条件满足 → `BROKEN`（冻结 break_low）
- `BROKEN` → 反弹确认满足 → emit signal；回落到 `break_low_extreme` 以下 → `INVALIDATED`（丢弃）

---

## 4. 配置落点（规划）

```
config/strategies/hub_rebound/      # 代号 hr 或 strategy_x，留给实现时决定
  meta.yaml                         # 与 SRB/FBF 并列的独立 archetype
  archetypes/
    execution.yaml                  # trailing / structural_sl / ladder（复用模板）
    prefilter.yaml                  # regime = low_adx_low_er 白名单 + 压缩阈值
  features.yaml                     # 中枢/破位/反弹三类特征
  evidence_candidates.yaml          # hub_score / break_score / rebound_score
```

---

## 5. MVP 路线（建议顺序）

1. **阶段 0**：本 spec（当前）。与 SRB/FBF 语义对照表入 `docs/z实验_005_统一研究/`。
2. **阶段 1**：离线特征工程 + `hub_rebound_state_machine.py` 规则实现，产出 signal CSV 但**不入回测**。
3. **阶段 2**：人工 inspect N 个历史信号，确认语义对齐（中枢是否"像"、破位是否"决定性"、反弹是否"干净"）。
4. **阶段 3**：小仓 backtest 跑单品种（建议 BTC）月级 rolling，评估 hit_rate / meanR / max_drawdown。
5. **阶段 4**：多品种 + 与 SRB/FBF 同桌跑 portfolio 回测，确认不互相挤占收益。

---

## 6. 开放问题（实现前需敲定）

1. **镜像方向**：中枢顶部假破反弹 SHORT 是否做？直觉上做多头"恐慌清洗"效果更强（合约持仓结构偏多），空头对称未必有效。建议 MVP 只做 LONG。
2. **中枢判定的时间尺度**：2h 周期下 40 根 ≈ 3.3 天，是否够长？是否需要日线确认？
3. **与 SRB 的仓位协调**：当策略 X 与 SRB 同品种反向时由谁先开仓？是否允许同时持仓？建议由组合层 `exposure_cap` 控制而非策略内部。
4. **成交量依赖**：是否必须要求破位放量？加密合约里 volume 噪声大，可做可选项。
