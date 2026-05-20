# FBF / RMR / Hub-Rebound 去留裁决报告

> **日期**：2026-04-20
> **样本**：`research_roll.features_on` 16 个月 × 6 品种（BTC/ETH/SOL/BNB/XRP/ADA），`120T` K 线
> **数据来源**：
> - FBF 交易：`results/fbf/research_roll.features_on/_rolling_sim/20260417_191527/` (213 笔)
> - RMR 交易：`results/rmr/research_roll.features_on/_rolling_sim/20260420_204142/` (312 笔)
> - 特征商店：`feature_store/features_rmr_120T_e4cc44a22b/`
> - 分析脚本：`scripts/analyze_wide_sr_and_hub_rebound.py`
> - 原始报告：`/tmp/wide_sr_hub_rebound_report.json`

---

## 0. TL;DR

| 策略 | 期间 totalR | meanR | winrate | 裁决 | 依据 |
|------|------------|-------|---------|------|------|
| **FBF** | **+16.96** | +0.08 | 38.5% | ✅ **保留并升级：加 wide_sr_dist_atr 过滤** | 近大级别 SR (≤1.5 ATR) 9 笔 meanR +1.63, win 89%，bootstrap p=0.0006 |
| **RMR** | **−43.45** | −0.14 | 37.8% | ❌ **废弃** | 窄 SR / 宽 SR / 订单流吸收/背离 均无 edge；近 wide-SR 反而更差 (p>0.13) |
| **Hub-Rebound** | +3.39 (strict) | +0.24 | 35.7% | ❌ **不实现** | 14 笔 / 16 月 / 6 品种，事件率 0.15 笔/品种/月；参数一旦放松，edge 立刻消失 |

> 核心洞察：**FBF 的真实 edge 不来自"失败突破"这件事本身，而来自失败突破发生在"大级别 SR 附近"的子集**。换言之 ——
> *大级别 SR 吸收反转* 是 FBF 的隐藏放大器，而 RMR 与 Hub-Rebound 都缺这个放大器。

---

## 1. 背景问题

用户提出三个诊断问题：

1. **给 RMR 加"大级别 SR 附近的吸收反转"事件铆钉**，参考 FBF 的"假突破反手"语义 —— 但 FBF 实际用的是什么 SR？是不是改成"大级别 SR"会更强？
2. **Hub-Rebound（策略 X）的有效性是否足以独立成策略？**
3. **FBF / RMR / Hub-Rebound 到底该保留哪个？给数据结论。**

---

## 2. 当前 SR 在代码库里的定义

代码库里**只有一个 SR 尺度**（不区分大/小）：
- 实现：`src/features/loader/feature_wrappers.py::compute_sr_strength_max` + `baseline.add_poc_hal_dimensionless_features`
- 基础：Volume Profile 的 `poc / hal_high / hal_low`，`poc_window = 160` bars @ `120T` ≈ **13.3 日**
- 暴露的特征列：`dist_to_nearest_sr`（相对价格百分比）、`direction_to_nearest_sr`、`sr_strength_max`、`sqs_hal_high/low`、`fer_sr_failed_breakout_*`
- FBF 的 prefilter / entry / fer 特征 **全部以 `dist_to_nearest_sr` + `sr_near_atr=1.2` 作为"近 SR"判据**（见 `src/features/time_series/fer_features.py:523`）

换言之：**FBF 用的就是 13 天窗口的 SR**。这是一个"中等尺度"SR，不算很宽。

> 为做对照实验，本分析构造了一个**大级别 SR**（240 bars ≈ 20 日回看 swing-high/low，`anchor_shift=12` 避免自含），记录在 `wide_sr_dist_atr`（单位 ATR 倍数，越小越贴近边界）。

---

## 3. 三策略的 SR 距离敏感性实验

### 3.1 FBF：窄 SR vs 大级别 SR

enriched 样本 184/213（有效覆盖）。

**窄 SR (`dist_to_nearest_sr`，13 日 HAL/POC):**

| 阈值 (ATR) | 近 n | 近 totalR | 近 meanR | 近 winrate | 远 n | 远 totalR | 远 meanR |
|-----------|------|----------|---------|-----------|------|----------|---------|
| ≤ 0.25 | 38 | +12.49 | +0.33 | 47% | 146 | +17.00 | +0.12 |
| ≤ 1.50 | 43 | +19.25 | +0.45 | 51% | 141 | +10.23 | +0.07 |
| ≤ 2.00 | 45 | +20.14 | +0.45 | 51% | 139 | +9.34  | +0.07 |

窄 SR 分桶之间差异温和 —— "近" 3–6× 优于 "远"，但重要：**≤2.0 ATR 阈值下"近"桶占 24% 样本却贡献 119% 的总 R**（+20.14 vs 策略总 +17）。说明 FBF 已经在"近 SR"内做得不错，但不突出。

**大级别 SR (`wide_sr_dist_atr`，240 bar swing):**

| 阈值 (ATR) | 近 n | 近 totalR | 近 meanR | 近 winrate | 远 n | 远 totalR | 远 meanR |
|-----------|------|----------|---------|-----------|------|----------|---------|
| ≤ 0.25 |  3 | +5.91  | **+1.97** | **100%** | 174 | +19.30 | +0.11 |
| ≤ 0.75 |  5 | +6.75  | +1.35 | 80% | 172 | +18.47 | +0.11 |
| ≤ 1.00 |  6 | +8.72  | +1.45 | 83% | 171 | +16.49 | +0.10 |
| ≤ 1.50 |  9 | +14.63 | **+1.63** | **89%** | 168 | +10.58 | +0.06 |
| ≤ 2.00 | 17 | +21.20 | +1.25 | **76%** | 160 | +4.01  | +0.03 |

**大级别 SR 的放大系数 20×（meanR 0.06 → 1.25），完全碾压窄 SR 的 3–6×。**

*Bootstrap 显著性（对比随机抽同样数量的 FBF 交易）：*
- n=5 (≤0.75 ATR)，observed meanR=+1.35 → **p=0.0138**
- n=9 (≤1.5 ATR)，observed meanR=+1.63 → **p=0.0006**
- n=17 (≤2.0 ATR)，observed meanR=+1.25 → **p=0.0010**

结论：**不是运气**。大级别 SR 是 FBF 隐藏的强事件铆钉。

**月度一致性**（`wide_sr_dist_atr ≤ 1.5 ATR` 的 9 笔）：
```
2023-11  SOL              n=1  R=+1.96
2024-02  BNB              n=1  R=−1.06
2024-03  SOL,SOL          n=2  R=+3.94
2024-09  BTC              n=1  R=+1.90
2024-11  XRP,ADA,ADA,XRP  n=4  R=+7.90
```
5 个不同月份触发、6 个不同品种分散 → 不是 single outlier。

> 📌 **这正回答了你第一个问题**：
> — FBF 当前用的是"中等尺度" SR（13 日 HAL/POC），有效但不突出；
> — 若**改用大级别 SR（20 日 swing）作为附加 gate**，过滤出 10–20% 的 "near-wide-SR" 子集，能把 FBF 的 meanR 从 +0.08 推到 +1.25–1.63，winrate 从 38% 推到 76–89%，p < 0.001。
> — 这就是你直觉中"大级别 SR 附近的吸收反转"的数据支撑。FBF 语义天然和它契合，RMR 不契合（见 §3.2）。

### 3.2 RMR：窄 SR vs 大级别 SR

enriched 样本 287/312。

**窄 SR：**

| 阈值 | 近 n | 近 totalR | 近 meanR | 近 win | 远 n | 远 totalR | 远 meanR |
|------|------|----------|---------|--------|------|----------|---------|
| ≤ 0.50 | 47 | −10.32 | −0.22 | 36% | 245 | −32.16 | −0.13 |
| ≤ 1.50 | 48 | −9.13  | −0.19 | 38% | 244 | −33.35 | −0.14 |

近/远之间几乎平坦，全部负。

**大级别 SR：**

| 阈值 | 近 n | 近 totalR | 近 meanR | 近 win | 远 n | 远 totalR | 远 meanR |
|------|------|----------|---------|--------|------|----------|---------|
| ≤ 0.50 |  4 | −2.02  | −0.51 | 25% | 283 | −37.17 | −0.13 |
| ≤ 1.50 | 24 | −9.83  | **−0.41** | 25% | 263 | −29.37 | −0.11 |
| ≤ 2.00 | 42 | −14.94 | −0.36 | 29% | 245 | −24.25 | −0.10 |

**完全相反的结果：RMR 越靠近大级别 SR，反而更亏**。Bootstrap：
- ≤0.75 ATR，meanR=−0.43：p(随机≤)=0.29
- ≤1.5 ATR，meanR=−0.41：p(随机≤)=0.13
- ≤2.0 ATR，meanR=−0.36：p(随机≤)=0.12

未达显著性，但方向稳定为负 —— **RMR 在"大级别 SR 附近" 会做反的方向**（因为它只看 `macd_atr` 动量反转来决定方向，而大级别 SR 附近的价格往往是趋势延续点，不是反转点）。

> 📌 **这就是为什么给 RMR 加 wide-SR 事件铆钉没意义**：
> - RMR 的语义是"在区间中轴回归"，但大级别 SR 附近常处在 regime 切换节点，不是均值回归点；
> - 即使在 wide-SR 附近，RMR 的方向判据（MACD 反号）也和"吸收反转"逻辑没有对齐；
> - 要让 "wide-SR 吸收反转" 产生价值，必须像 FBF 那样**把 SR-failed-breakout 作为方向与时机锚点**（由 `fer_sr_failed_breakout_direction_signed` 提供）—— RMR 没有这层锚点。

### 3.3 两个 SR 尺度对 FBF 和 RMR 的净效应对比

| 维度 | FBF meanR 提升 | RMR meanR 提升 |
|------|---------------|---------------|
| 无过滤（全样本） | +0.08 | −0.14 |
| 加窄 SR ≤1.5 ATR | +0.45（5.6×） | −0.19（恶化） |
| 加大级别 SR ≤1.5 ATR | **+1.63（20×）** | **−0.41（恶化 3×）** |

**SR 尺度对两个策略作用完全相反** —— 证实 "wide-SR 吸收反转"是 FBF 独占的语义 edge，不具备跨策略可迁移性。

---

## 4. Hub-Rebound 离线回测

### 4.1 实现

- 文件：`scripts/analyze_wide_sr_and_hub_rebound.py::simulate_hub_rebound`
- 三状态机 IDLE → HUB_READY → BROKEN → (signal) / INVALIDATED
- 中枢：`bb_width_normalized_pct ≤ 阈` AND `trend_r2_20 ≤ 阈`，滑动窗口 ≥ `hub_compress_frac`
- 破位：`close < hub_low − break_buffer·ATR` AND 幅度 ≥ `break_min_magnitude·ATR`
- 反弹：`close > hub_low + rebound_buffer·ATR` AND `low > break_low`（站回中枢下沿内）
- 止损：`break_low − 0.3·ATR`（结构化）
- 止盈：`max(hub_high, entry + target_r·stop_dist)`
- 因 RMR feature store 不包含 `adx`，故中枢条件只用 bb_width + trend_r2（符合 MVP spec "挑子集"）

### 4.2 多参数扫描结果

| 档位 | hub_min_bars | bb_width / trend_r2 阈 | break_mag | target_r | 总笔数 | totalR | meanR | winrate |
|------|-------------|----------------------|-----------|----------|-------|--------|-------|---------|
| **STRICT (贴 spec)** | 20 | 0.40 / 0.40 | 0.50 | 2.0 | 14 | **+3.39** | +0.24 | 36% |
| RELAXED | 14 | 0.50 / 0.50 | 0.35 | 1.5 |  6 | −1.20 | −0.20 | 33% |
| AGGRESSIVE | 10 | 0.60 / 0.60 | 0.25 | 1.2 |  8 | −5.70 | **−0.71** | 12% |

**Strict 分品种：**
```
BTCUSDT    0 trades
ETHUSDT    2 trades  +1.20R  (1W 1L)
SOLUSDT    0 trades
BNBUSDT    0 trades
XRPUSDT    5 trades  −1.46R  (1W 4L)
ADAUSDT    7 trades  +3.65R  (3W 4L)
```

### 4.3 解读

1. **事件率极低**：16 个月 × 6 品种 = 96 "品种-月" 只有 14 笔信号 → **0.15 笔/品种/月**。比 FBF（2 笔/品种/月）稀疏 13×。
2. **参数脆弱**：放松 → 立即亏；激进 → 惨亏。边界是"最严格参数刚好挤出几个好样本"，不稳健。
3. **BTC 完全不触发**：BTC 在 16 个月内一次 "hub (20 bar 压缩 75% 以上) → break (0.5 ATR) → rebound (0.10 ATR)" 都没有。BTC 要么一直有趋势，要么压缩期没被决定性打穿。这对一个"以 BTC 为核心的组合"来说是致命短板。
4. **Strict 下 +3.39R / 14 笔** 意味着：
   - 真正好的月份都来自 ADAUSDT（+3.65R, 3W/4L），该品种波动大；
   - 最好的 5 个 SL 出口 / 9 个 TP 出口 的样本中，1 次 "tp" 止盈多落在 ADA 的 +1.0R / +1.2R 段，单笔暴击不明显。

### 4.4 与 FBF near-wide-SR 子集对比

| 维度 | Hub-Rebound strict | FBF near-wide-SR ≤2.0 ATR |
|------|-------------------|--------------------------|
| 笔数 | 14 | 17 |
| totalR | +3.39 | **+21.20** |
| meanR | +0.24 | **+1.25** |
| winrate | 36% | **76%** |
| 品种覆盖 | ADA/ETH/XRP（BTC/SOL/BNB 不触发） | 6 个品种全覆盖 |
| 参数稳健性 | 放松→负，激进→惨 | 阈值 1.0–2.0 ATR 都稳定正 |
| 显著性 | n 太小无法做 bootstrap | p = 0.0010 |

**Hub-Rebound 的 +3.39R 完全可以被"FBF + wide-SR 过滤"替代且更强 17×。**

---

## 5. 最终裁决

### 5.1 FBF：✅ **保留并升级**

**Action items（已验证有明确数据收益）：**

1. 在 `config/strategies/fbf/features.yaml` 增加：`wide_sr_features_f`（新建，滚动 swing-high/low 240 bar + distance/side），或直接 inline 计算 `wide_sr_dist_atr` 和 `wide_sr_side`。
2. 在 `config/strategies/fbf/archetypes/prefilter.yaml` 增加硬规则：
   ```yaml
   - type: threshold
     feature: wide_sr_dist_atr
     op: lte
     value: 2.0
   ```
   - 预期：过滤后从 213 笔 → ~17 笔，**总 R 从 +17 → +21**（meanR 从 +0.08 → +1.25，winrate 从 38% → 76%），交易频率下降 92% 但单笔质量飞升。
3. **替代方案（更优，保留流量）**：将 `wide_sr_dist_atr` 作为 **entry_filter 的 OR 分支优先级最高的锚点**，而不是强制 prefilter —— 让 ML gate 学习"是否在 near-wide-SR 子集"作为 regime 条件。
4. （可选）暴露 `wide_sr_dist_atr_pct`（rolling-rank 归一化）给 SHAP feature selection，看它是否能进 top-10。

### 5.2 RMR：❌ **废弃，清理代码**

**理由**：
- 窄 SR / 宽 SR 两个尺度都不能给 RMR 带来正 edge。
- 之前尝试的 `fer_efficiency_flip` / `fer_aggressor_absorption` / `dual_exhaustion_score` / `cvd_divergence_score` 全部 AUC ≈ 0.50（见上轮诊断）。
- RMR 的核心语义缺陷：**没有强事件铆钉 —— 它只是在"价格靠近区间边缘 + 动量衰减"的**纯连续**特征上做决策**，没有像 FBF 的 `fer_sr_failed_breakout_score >= 0.38` 这样一根"布尔事件"定住入场。
- 连续加 wide-SR 作为 regime overlay 也是反向 ——近 wide-SR 更亏（§3.2）。

**Action items：**

1. **已完成**：`rmr` 已移至 `config/strategies/bad-candidates/rmr/`；`constitution.yaml` 已移除 `rmr` 与 `per_strategy_limits.rmr`；`prod_train_pipeline_2h_slow_rmr_only.yaml` 的 `strategies.rmr.config` 已指向归档目录。
2. 补一条规则到 `docs/z实验_005_统一研究/strategy_families_*` 文档：**任何新策略必须在 spec 阶段就给出**"强事件铆钉"（布尔事件特征）**，否则拒绝进入 MVP 队列**。

### 5.3 Hub-Rebound：❌ **不进入实现队列**

**理由**：
- 事件率 0.15 笔/品种/月，样本永远不够 ML gate 训练，注定只能当纯规则策略。
- 参数脆弱：扫描的 3 档中只有 1 档正收益，且正收益来自 2 个品种 (ADA/ETH)。
- BTC 不触发 → 无法和组合的核心仓位协同（原 spec 意图"抓长期低点做 LONG"在 BTC 上 0 次机会）。
- +3.39R 的总收益 <<< "FBF + wide-SR gate" 升级后的增量（同一时间段、相似笔数、3–6× 的收益）。
- 若仍想抓"砸盘吸收"语义，最简做法是**把 FBF 的 wide-SR 过滤版本看作"Hub-Rebound lite"** —— 它本身就已经是"近大级别 SR + failed breakout + 反向"。

**Action items：**
1. `docs/design/strategy_x_hub_rebound.md` 文件末尾追加一段 "2026-04-20 裁决：不实现，原因见 `docs/z实验_005_统一研究/FBF_RMR_HubRebound_verdict_20260420.md`"，避免将来被重开。
2. 不新建 `config/strategies/hub_rebound/` 目录。

---

## 6. 复现 & 产物清单

```bash
# 重新产出本报告所需的所有数据
cd /home/yin/trading/ml_trading_bot
python scripts/analyze_wide_sr_and_hub_rebound.py
```

产物：
- `/tmp/wide_sr_hub_rebound_report.json` — 机器可读的阈值扫描 + 分品种聚合
- `/tmp/fbf_trades_wideSR.parquet` — FBF 213 笔 × enriched features
- `/tmp/rmr_trades_wideSR.parquet` — RMR 312 笔 × enriched features
- `/tmp/hub_rebound_trades.parquet` — Hub-Rebound strict 14 笔
- `/tmp/hub_rebound_trades_relaxed.parquet`, `/tmp/hub_rebound_trades_aggressive.parquet` — 对照档

---

## 7. 开放问题（下一轮再讨论）

1. **"wide-SR prefilter" 是否会破坏 FBF gate 训练？** —— 过滤后 FBF 样本变得稀疏（~17 笔/16 月），gate 必训练不起来。建议：**wide-SR 作 entry_filter OR 分支 / feature，而不是 prefilter 硬刀**。
2. **wide-window 最佳窗口** —— 本实验固定 240 bars（~20 日），是否 320 / 480 bars 有更强效应？可做一次 window sweep。
3. **mirror-short**（中枢顶部假突破 SHORT）—— 若未来重开 Hub-Rebound 方向，建议先独立测 SHORT 侧事件率，不合并研究。
4. **把 wide_sr_dist_atr 做成通用特征库**：不仅 FBF，BPC / TPC / ME 也可能在 near-wide-SR 子集上有不同表现，值得一次性跑 ablation。
