# B 系统：入场语义窗口 vs 执行层周期错配

> 记录日期：2026-06-08  
> 背景：S50 trading map 复盘 — `tpc_pullback_depth > 0.5` 落在上涨途中 **小震荡**，错过用户关心的 **大级别深回调（≈15–20%+）**。  
> 交叉引用：[TPC语义约束与树标签对齐_CN.md](TPC语义约束与树标签对齐_CN.md)、[B系统.md](B系统.md)

---

## 1. 执行层（BPC / TPC / ME / SRB 共用）

四策略均在 **120T（2h）** 上研发；`execution.yaml` 口径高度一致：

| 参数 | 典型值 | 日历语义（120T） |
|------|--------|------------------|
| `structural_exit: ema1200` | span=1200 | ≈ **100 天** 宏观趋势出场 |
| `initial_r` | 4.0–4.5 ATR | 宽止损，允许持 **数周～数月** |
| `trail_r` / `activation_r` | 6.0 / 3.5–6.0 | 远追踪 |
| `breakeven trigger_r` | 10.0 | 极远保本 |
| 加仓 ladder | 2R / 4R | 趋势腿复利语义 |

**结论**：执行层统一是 **大周期 swing / beta 腿**，不是短线 scalping。

---

## 2. 入场语义主窗口（按策略）

换算：`N bar × 2h`（仅 120T 主周期）。

### 2.1 TPC — **错配严重（与「大回调」目标）**

| 特征 / 层 | 参数 | 日历 | 说明 |
|-----------|------|------|------|
| `tpc_soft_phase_f` | `lookback_breakout: 20` | ≈ **1.7 天** | `tpc_pullback_depth` 分母为近 20 bar 高低 **区间宽度** |
| `bars_since_extreme_f` | `lookback: 20` | ≈ **1.7 天** | anti-chase |
| `ema_1200_position_f` | span=1200 | ≈ **100 天** | regime / direction 宏观锚 |
| prod prefilter | `depth <= 0.85` | — | 仅上界，无「多深才进」下界 |
| S50 | `depth > 0.5` | — | **20 bar 区间下半段**，≠ 价格回撤 20% |

**机制问题**：

1. `pullback_depth = (rolling_high - close) / (rolling_high - rolling_low)` 是 **区间位置分位**，不是 **从 swing high 跌了多少 %**。
2. 大回调持续数周时，20 bar 的 `rolling_high` 随价格下跌 **一起下移**，depth 反而不一定升高。
3. 结果：在上涨 **小回踩** 触发，在图上的 **宏观深箭头** 处常不触发或已错过最佳区。

**判定**：**短周期入场语义 + 长周期执行** — 若目标是「牛市大回调再买」，当前 TPC 特征 **语义不对齐**。

---

### 2.2 BPC — **与 TPC 同源，且更偏短周期延续腿**

| 特征 / 层 | 参数 | 日历 | 说明 |
|-----------|------|------|------|
| `bpc_soft_phase_f` | `lookback_breakout: 20` | ≈ **1.7 天** | 与 TPC 共用 `_compute_soft_phase_core` |
| prefilter | `bpc_recent_breakout_strength >= 0.4` | 20 bar 内突破 | 必须 **近期** Donchian 突破 |
| prefilter | `bpc_pullback_depth <= 0.55` | 20 bar 内 | 明确要 **浅回踩**，排斥深回调 |
| 辅助 | `bpc_pullback_duration` / `impulse` lookback=20 | ≈ 1.7 天 | |

**判定**：**错配同类** — 设计语义是「近 2 天突破 → 浅回踩 → 延续」，不是宏观 dip buying。与 ema1200 执行并用时，仍是 **短博弈入场 + 长持仓**。

---

### 2.3 ME — **分层 intentional，非同一类 bug**

| 层 | 特征 | 窗口 | 日历 |
|----|------|------|------|
| Prefilter | `atr_percentile` | 540 bar | ≈ **45 天** |
| Prefilter | `recent_compression_decay` | 540 bar | ≈ **45 天** |
| Direction | `me_accel_5k` | 3 vs 8 bar | **6h vs 16h** |
| Direction | `me_accel_2k` | 2 bar | **4h** |
| Regime | `ema_1200_position` | 1200 bar | ≈ **100 天** |

**判定**：**长 setup（45d 压缩）+ 短 trigger（几小时加速度）** 是 CompressionBreakout 本体。与执行层仍有张力，但 **不是**「误以为买大回撤、实际买小震荡」。

---

### 2.4 SRB — **多尺度分层，错配最轻**

| 层级 | 窗口（120T） | 日历 | 用途 |
|------|-------------|------|------|
| L1 | 20 bar | ≈ 1.7 天 | 破位 / 结构化 SL |
| L2 | 160 bar (POC/HAL) | ≈ **13 天** | `sr_strength_max` prefilter |
| L3 | 240 bar (`wide_sr_swing`) | ≈ **20 天** | 大级别 SR |
| 新鲜度 | `max_age_bars: 24` | ≈ **2 天** | 只做刚突破，不追尾 |
| Entry | `efficiency_window: 20` | ≈ 1.7 天 | 订单流确认 |

**判定**：prefilter 已用 L2/L3；短窗用于 **突破后 2 天新鲜腿** 是 archetype 意图。若要「大牛市深回调」，**不应改 SRB**，应改 TPC/BPC 分工。

---

## 3. 总览表

| 策略 | 入场主窗口 | 执行层 | 「短博弈 + 长执行」错配？ |
|------|------------|--------|---------------------------|
| **TPC** | 20 bar depth / anti-chase | ema1200 ~100d | **是**（抓大回调时） |
| **BPC** | 20 bar 突破 + 浅回踩 | ema1200 ~100d | **是** |
| **ME** | 45d setup + 4h trigger | ema1200 ~100d | 部分（设计如此） |
| **SRB** | 13–20d 结构 + 2d 新鲜突破 | ema1200 ~100d | **否**（intentional） |

---

## 4. 实验 A 备忘（S50 × BPC PCM，2026-06-07）

全窗 2022→2026，6 highcap：

| 变体 | Total R | Max DD |
|------|---------|--------|
| pcm_prod_baseline | 34.24 | -11.4% |
| pcm_s50_tpc_heavy（最优 S50） | 29.04 | -8.7% |

S50 降 DD 但牺牲 ~5R；根因仍是 **TPC 信号稀少 + 语义偏短窗**，非 PCM 配置 alone 可解。

---

## 5. TPC 改法：如何抓「大周期回调」

> **原则**：保持 **120T** 与现有 execution（ema1200 / 宽 trail）；**改入场语义**，不先改 timeframe。

### 5.1 不推荐优先做的

| 做法 | 原因 |
|------|------|
| 120T → 240T/480T | 整条管线重标定；`lookback_breakout` 用参数即可拉长日历窗口 |
| 仅调高 S50 `depth > 0.5` | 在 20 bar 上 depth 仍非「价格回撤 %」 |
| 只加 constitution 提仓 | 不改变入场位置，3× 加在错误 bar 上 |

### 5.2 推荐路线（分阶段）

#### Phase-1：新特征 — **宏观回撤百分比**（首选）

新增 `tpc_macro_pullback_pct`（命名可讨论），与现有 `tpc_pullback_depth` **并存**：

```text
# 多头（trend_sign 来自 ema_1200_position sign）
drawdown_pct = (roll_high_N - close) / roll_high_N

# N 候选（120T bar 数 → 日历）
N=120  → ~10 天
N=240  → ~20 天
N=480  → ~40 天
```

| 属性 | `tpc_pullback_depth`（现） | `tpc_macro_pullback_pct`（新） |
|------|---------------------------|-------------------------------|
| 分母 | 近 N bar 高低 **区间宽** | 近 N bar **最高价** |
| 大回调中 rolling_high | 随跌下移，depth 失真 | 高位锚定更久，pct 更稳 |
| 与用户「20% 回调」 | 不对应 | 可直接阈值 `>= 0.15 / 0.20` |

**建议 prefilter 变体**（bull regime 不变）：

```yaml
# P15 — 中等宏观回踩
- feature: tpc_macro_pullback_pct
  operator: '>='
  value: 0.15

# P20 — 深宏观回踩（用户主目标）
- feature: tpc_macro_pullback_pct
  operator: '>='
  value: 0.20
```

配合 **宏观仍在多头结构**（避免熊市接刀）：

```yaml
- feature: ema_1200_slope_10
  operator: '>='
  value: 0.0        # 宏观趋势未明显转负
- feature: ema_1200_position
  operator: '>='
  value: -0.12      # 允许深回踩略破均线，但非无限深熊
```

#### Phase-2：拉长 `lookback_breakout`（改旧 depth 语义）

在 `feature_dependencies.yaml` 或实验树覆盖 `tpc_soft_phase_f.compute_params`：

| 变体 ID | lookback_breakout | 日历 | 备注 |
|---------|-------------------|------|------|
| L60 | 60 | ~5d | 最小可行拉长 |
| L120 | 120 | ~10d | 与 P15 窗口对齐 |
| L240 | 240 | ~20d | 与 P20 / wide_sr 同量级 |

须 **同步** `bars_since_extreme_f.lookback` 与 `normalize_window`，并重扫 `depth` 阈值（L120 上 `>0.5` ≠ 20 bar 上 `>0.5`）。

`node_cache_version` 必须 bump，避免 FeatureStore 吃到旧 20 bar 缓存。

#### Phase-3：entry 层防「抄底刀」

大回调入场需防 falling knife（与浅回踩不同）：

| 规则 | 作用 |
|------|------|
| `tpc_vol_pullback_confirm` 或 CVD absorption | 卖压衰竭 |
| `path_efficiency_pct` 下界 | 排除单边崩 |
| `bars_since_local_high` 用 **长 lookback** | 确认离前高已有一段时间 |

prod entry OR 可保留；在 **P15/P20 prefilter 子空间** 上再跑 E2 anti-chase 组合 grid。

#### Phase-4：执行层微调（验证后再动）

宏观 dip 止损距离常更大：

- 可试 `initial_r: 5–6` 或略放宽 `max_stop_pct`
- **暂不**改 `structural_exit: ema1200`（与「大周期 beta 腿」一致）
- 深回踩腿仓位：constitution `tpc.max_risk_per_trade` 或 experiment B `size_multiplier` 仅在 **P20 子集** 启用

### 5.3 建议实验矩阵（下一批）

| ID | 改动 | 假设 |
|----|------|------|
| **P15** | `macro_pullback_pct >= 0.15`, N=240 | 抓 10–20d 级回踩 |
| **P20** | `>= 0.20`, N=240 | 用户主目标 |
| **P15_E2** | P15 + entry anti-chase | 减少贴底追 |
| **L120_S50** | lookback=120 + depth>0.5 重标定 | 旧特征拉长是否够用 |
| **P20_3x** | P20 + execution size_multiplier 3 | 实验 B 在正确语义上复测 |

每个变体：**segment grid + trading map（BTC/SOL）**，对照 E0_prod 与 S50。

### 5.4 实现落点（工程）

| 项 | 路径 |
|----|------|
| 新特征函数 | `src/features/time_series/bpc_features.py` 或 `tpc_macro_features.py` |
| 注册 | `config/feature_dependencies.yaml` → `tpc_macro_pullback_pct_f` |
| TPC 请求列 | `config/strategies/tpc/features.yaml` |
| 实验树 | `config_experiments/tpc_macro_replace_*_strategies/`（静态 prefilter YAML） |
| Grid（macro 替代 depth） | [`config/experiments/20260610_tpc_macro_pullback_replace/`](../../config/experiments/20260610_tpc_macro_pullback_replace/) |
| 历史扫描 | `scripts/research/scan_tpc_pullback_lookback.py` → `results/tpc/research/macro_pullback_scan_*` |

---

## 6. BPC 后续（TPC 验证后）

若 TPC 宏观回踩腿有效，BPC 保持 **浅回踩短窗** 分工即可，**不必**把 BPC `pullback_depth <= 0.55` 改成深回踩。PCM：`bpc` 延续腿 + `tpc` 宏观 dip 腿。

---

## 7. 交叉引用

- [TPC语义约束与树标签对齐_CN.md](TPC语义约束与树标签对齐_CN.md) §2–3 depth 与 bars_since
- `config/experiments/20260607_tpc_s50_pcm_leverage/README.md` — PCM / 杠杆实验
- `src/features/time_series/bpc_features.py` — `_compute_soft_phase_core`, `lookback_breakout`
- `config/feature_dependencies.yaml` — `tpc_soft_phase_f`, `bars_since_extreme_f`
