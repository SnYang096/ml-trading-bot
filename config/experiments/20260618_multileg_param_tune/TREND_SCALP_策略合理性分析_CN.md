# trend_scalp 策略合理性分析

> **日期**：2026-06-18  
> **范围**：`trend_scalp`（dual_add_trend）研究栈 + 2026-06-18 回测结果  
> **关联**：[`TREND_LOSER_TIMEOUT_优化说明_CN.md`](TREND_LOSER_TIMEOUT_优化说明_CN.md)、[`config/strategies/trend_scalp/README.md`](../../strategies/trend_scalp/README.md)

---

## 0. 一句话结论

**策略在机制上是合理的**：它只在「有方向、低震荡、非稳定箱体」的时段交易，用有上限的顺势加仓 + basket 止盈赚趋势延续；**不是**全天候策略，也**不该**在 chop/box 里跑。

**2026-06-18 回测的关键发现**：prod 栈在 `2h` 信号 + `1min` 执行时，`max_loser_hold_bars=24` 被当成 24 分钟而非 24×2h，导致 72% 成交是必亏的 `loser_timeout` 循环——这是**执行尺度 bug**，不是特征或 regime 判断失效。修复 hold 缩放后，四段 OOS 全正。

**z 分数**：在 **locked archetype 入场逻辑里未使用**；仅作为 Phase 1 研究候选特征，当前无 promote 证据。

---

## 1. 特征为什么这样设计？

### 1.1 设计原则：按职责分层

trend_scalp 的特征不是「一个大模型打分」，而是 **三层分工**：

| 层级 | 特征族 | 职责 | 是否 locked |
|------|--------|------|-------------|
| **Regime 门** | `trend_confidence`、`bpc_semantic_chop`、`box_structure` | 决定「这段能不能做趋势加仓」 | ✅ 是 |
| **执行参数** | `atr14`、`atr_percentile`、波动 regime | 决定加仓间距、止盈距离 | ✅ 间接 |
| **研究候选** | Hurst、VPIN、trade_cluster z-score、CVD 等 | Phase 1 扫描，找额外 gate | ❌ 否 |

声明见 `config/strategies/trend_scalp/features.yaml`：

```yaml
# Experimental trend/order-flow confirmation candidates. These are consumed
# by research selectors only; archetypes keep the locked trend anchor.
```

### 1.2 核心门控特征

#### `trend_confidence` — 有没有一致方向

计算（`compute_trend_confidence_from_series`）：

```
对 horizons = 3, 5, 10 根信号 bar 的收益率取符号
trend_confidence = mean(|sign|) × |mean(sign)|
trend_direction    = sign(mean(sign)) → UP / DOWN
```

含义：
- 多周期方向**一致** → confidence 高
- 方向**打架**（有的涨有的跌）→ confidence 低

这不是预测未来，而是回答：**当前这几根 bar 是否在走单边**。与策略「顺势加仓」直接对齐。

locked 阈值（`archetypes/regime.yaml`）：
- 入场：`trend_confidence ≥ 0.7`（研究 README 写 0.80，以 effective config 为准）
- 持仓：`≥ 0.4` 才继续持有
- 低于 exit → regime 段结束

#### `bpc_semantic_chop` — 是不是震荡区

来自 BPC soft phase 核心（`bpc_features.py`）：

```
semantic_chop = clip(bb_compression × (1 - direction_confidence) × 2, 0, 1)
```

- **BB 收窄** + **多周期方向不一致** → chop 高
- 与 ME、TPC 等同构公式，保证跨策略可比

locked 阈值：
- 入场：`semantic_chop ≤ 0.25`
- 持仓：`≤ 0.40`

含义：**带宽压缩且来回扫** 的时段不做趋势加仓——这类时段交给 `chop_grid`。

#### `box_prefilter` — 是不是稳定箱体

来自 `box_structure_f`（120 bar 窗口）：

```
box_ok = stability ≥ 0.85
      ∧ width ∈ [4%, 30%]
      ∧ touches ≥ 5
      ∧ 内部 chop ≥ 0.40
```

默认 `exclude_box_prefilter: true` → **稳定 box 段直接禁止 trend 入场**。

箱体边界反转交给 CRF/BPC 类策略；trend_scalp 要的是 **突破后的延续**，不是箱内来回。

#### ATR 族 — 执行尺度，不是方向

- `atr_f` / `atr_percentile_f`：加仓间距 `0.75×ATR`、止盈 `max(0.6×ATR, 0.12%)`
- 保证止盈 **fee-aware**（`fee_buffer + net_target`），避免「毛利不够手续费」

### 1.3 为什么不是一套特征打天下？

ABC 多腿系统刻意 **策略分 regime**：

| 市场形态 | 合适策略 | 核心机制 |
|----------|----------|----------|
| 无方向震荡 | chop_grid | 对称网格收割回摆 |
| 稳定箱体 | CRF / BPC | 边界反转、突破回踩 |
| **有方向趋势** | **trend_scalp** | 顺势有限加仓 + basket TP |

若用同一套特征覆盖所有行情，要么过拟合某一态，要么在错误态强行交易。trend_scalp 的特征设计是 **窄而深**：只回答「这段像不像可加仓的趋势」。

---

## 2. 为什么能判断出「xx 区域」？

这里的「区域」= **regime 段（segment）**，不是人工标注的牛熊市标签。

### 2.1 段发现算法（hysteresis）

回测入口 `diagnose_dual_add_trend.run()`：

```python
# 趋势模式
entry = (trend_confidence >= trend_min) & (chop <= exit_chop_min)
hold  = (trend_confidence >= trend_exit_min) & (chop <= chop_min)

# 可选：prefilter 规则、box 排除
entry &= rule_mask
entry &= ~box_prefilter

segs = _hysteresis_segments(entry, hold, min_len=6, max_len=120)
```

**迟滞（hysteresis）** 避免阈值附近抖动：
- 入场门槛 **严于** 持仓门槛（trend 0.7 vs 0.4，chop 0.25 vs 0.40）
- 一旦进入段，除非 trend 跌破或 chop 升高，否则不轻易退出

一段 = 连续满足 hold 条件的 2h bar 序列（6～120 根，即 12h～10 天）。

### 2.2 三种「区域」的对应关系

| 你说的区域 | 特征表现 | 策略行为 |
|------------|----------|----------|
| **趋势区** | trend_conf 高、chop 低、非 box | ✅ 开段、顺势加仓 |
| **震荡区** | chop 高、direction 分歧 | ❌ 不开段（chop_grid 的地盘） |
| **稳定箱体** | box_prefilter=true | ❌ 不开段（CRF 的地盘） |
| **趋势末端/拉伸** | extension 高（研究用） | 当前 locked 未直接用；可通过 chop/flip 间接退出 |

### 2.3 与「牛/熊/最近」标签的关系

`config/market_segment.yaml` 的四段（bear_2022、bull_2023_2024、recent_range_to_bear、recent_6m_oos）是 **事后切窗**，用于检验策略在不同宏观环境下的表现。

策略 **不会** 读取「现在是牛市」这类标签；它只在每个 bar 上算 trend/chop/box。  
四段 OOS 全正，说明：**在各宏观窗内，仍能找到足够多的「局部趋势段」**，而不是靠单一宏观假设。

---

## 3. 策略从根本上是否合理？

### 3.1 经济逻辑（合理之处）

1. **有边界的加仓**：`max_adds_per_side=3`、`max_gross=4`、`max_net=2` — 不是无限马丁
2. **顺势而非双向网格**：`initial_legs: TREND`，只开当前方向；flip 时 `close_offside_all`
3. **fee-aware 止盈**：basket TP 必须覆盖往返费用 + 最小净利
4. **regime 失效即走**：`force_exit_on_regime_loss: true`，不在 chop 里死扛
5. **与系统分工一致**：趋势段赚延续，震荡段交给 chop_grid

消融（README）：`max_adds=0` → `max_adds≥1` 在同一窗口显著提升收益，说明 **加仓在趋势段有边际贡献**，不是纯随机换手。

### 3.2 机制风险（需正视）

| 风险 | 说明 | 当前缓解 |
|------|------|----------|
| 趋势假突破 | 入场后快速反转 | flip 平仓 + regime 退出 |
| 高 beta 币主导 PnL | SOL/XRP 贡献大于 BTC | 五币等权资本桶；分币看 segment_by_symbol |
| 路径依赖 | 加仓间距、TP、flip 顺序影响大 | 固定 archetype + 分段 OOS |
| 费用敏感 | 全周期约 35% gross 被费吃掉 | 20bps 压力测试仍为正（hold_scaled） |
| **执行尺度错误** | 24@1min ≈ 24min 砍亏腿 | `--scale-max-loser-hold-to-signal` → 2880 |

### 3.3 与 baseline 对比后的判断

| 维度 | baseline（24@1min） | hold_scaled（2880@1min） |
|------|---------------------|--------------------------|
| 全窗 4bps | -14.8%，72% loser_timeout | **+145.8%**，0% timeout |
| 逻辑 | 24 分钟砍腿 → 必亏循环 | ~48h 与 2h 信号语义一致 |
| 可解释性 | 差（像策略坏了） | 差的是 **执行 bug**，不是 regime 特征 |

**结论**：策略 **在正确的 hold 语义下** 机制合理；baseline 的惨状主要是执行配置错误，不能用来否定「趋势加仓」本身。

### 3.4 尚未证明的部分

- [ ] live 引擎与 diagnose 的 `loser_timeout` / hold 缩放 **未对齐**
- [ ] constitution 仍 **禁用** trend_scalp 实盘
- [ ] 无 funding、强平、真实 fill 的完整账户曲线
- [ ] `return_pct_timeline` 是 **资本归一化累加**，不是带杠杆的 CAGR

---

## 4. 回测步骤是否正确？分几步？

### 4.1 标准回测流水线（研究栈）

```
Phase 0  数据准备
         └─ 1min parquet → resample 2h 信号 bar
         └─ build_features()：ATR、semantic_chop、box_structure、…
         └─ compute_trend_confidence_from_series()

Phase 1  Regime 段发现
         └─ entry/hold 迟滞掩码 + box 排除 + prefilter
         └─ _hysteresis_segments(min=6, max=120)
         └─ constitution 日开段上限过滤

Phase 2  段内 inventory 模拟（每段独立）
         └─ 冻结段首 close / ATR 作为 anchor
         └─ 若 execution_timeframe=1min：子 bar 回放 merge 信号特征
         └─ simulate_dual_add_segment()：
              开 TREND 腿 → 顺势 touch 加仓 → basket TP
              → flip 清逆势腿 → regime 结束强平
              → loser_timeout（hold 超限且亏损）

Phase 3  汇总
         └─ dual_add_trades.csv / dual_add_segments.csv
         └─ summary.csv（return_pct_timeline、胜率、timeout 率等）
         └─ capital_report.html

Phase 4  分段 OOS（可选）
         └─ experiment_trend_scalp_market_segment.py
         └─ 对 market_segment.yaml 四段各跑 Phase 0–3
         └─ segment_summary.csv 汇总

Phase 5  压力 / 参数网格（可选）
         └─ run_multileg_param_tune.py（20bps A/B、hold/reseed 网格）
```

### 4.2 步骤正确性评估

| 环节 | 是否正确 | 说明 |
|------|----------|------|
| 特征与段发现同源 | ✅ | 同一条 `build_features` 链路 |
| 2h 信号 + 1min 执行 | ✅ | `merge_signal_features_onto_execution_bars` 前视安全 |
| 费用扣减 | ✅ | 每笔 `2×fee_bps` round-trip + entry/add slippage |
| hold 缩放 | ⚠️ **prod 默认未开** | 须 `--scale-max-loser-hold-to-signal` 或写 archetype |
| Backtrader 交叉验证 | ✅ | 2026-06-02：与 diagnose 偏差 <2%（见 segment_validate DECISION） |
| funding | ❌ 未建模 | 高费率场景可能偏乐观 |
| 与 live 一致 | ⚠️ | live 无 loser_timeout；constitution 禁 trend |

### 4.3 常见误读

| 误读 | 正确口径 |
|------|----------|
| `return_pct_pooled` 五币相加 | 用 **`return_pct_timeline`** 作组合收益 |
| 段数少 = 策略没交易 | chop/box 段本就不该有 trend 段 |
| baseline 负 = 特征无效 | 先看 `loser_timeout_rate` 和 `resolved_max_loser_hold_bars` |
| 四段全正 = 全天候 | 只说明 **各宏观窗内仍有趋势段可赚**，不是震荡里也赚 |

---

## 5. 回测是否适配牛/熊/最近/震荡？

### 5.1 四段 OOS 结果（hold_scaled，fee=4bps）

| 段 | 窗 | return_pct_timeline | trade_win_rate | loser_timeout |
|----|-----|--------------------:|---------------:|--------------:|
| bear_2022 | 2022-01 → 2023-11 | **+106.8%** | 84.4% | 0% |
| bull_2023_2024 | 2023-06 → 2025-01 | **+79.5%** | 83.6% | 0% |
| recent_range_to_bear | 2025-01 → 2026-05 | **+89.0%** | 85.2% | 0% |
| **recent_6m_oos** | 2025-12 → 2026-05 | **+28.9%** | 84.2% | 0% |

20bps 压力下四段仍为正（recent_6m **+10.0%**）。

### 5.2 如何理解「适配」

**适配 ≠ 每个 bar 都交易。**

- **牛市/熊市**：宏观方向不同，但局部仍有 trend_conf 高、chop 低的 **可交易子段**；四段全正支持这一点
- **recent_6m**：promote 门禁段；+28.9%（4bps）说明 **当前 regime 下仍有 edge**
- **纯震荡**：策略 **主动回避**（chop>0.25 不入、chop>0.40 退出）；这不是失败，是设计分工

与 chop_grid 对照（同窗 2024-01～2026-05）：
- chop_grid：~104 笔，+1.2% — 稀疏、低换手
- trend baseline：3.5 万笔，-14.8% — hold bug 导致假高频
- trend hold_scaled：1.08 万笔，+145.8% — 合理频率下的趋势段收益

### 5.3 未覆盖的场景

- 极端单边闪崩（灾难止损为研究项，live catastrophic stop 未完全验证）
- 低流动性币、上新币
- 多策略同时占槽的 constitution 联合回测（`backtest_multileg_timeline` 待跑）

---

## 6. z 分数是否有效？

### 6.1 定义（trade_cluster 族）

`utils_order_flow_features.py`：

```python
z = (x - rolling_mean(x, w)) / (rolling_std(x, w) + ε)
# w ∈ {20, 50} bar
```

应用于：`imbalance_ratio`、`net_runs`、`max_buy_run`、`max_sell_run` 等 **逐笔聚类微观结构** 指标。

### 6.2 在 trend_scalp 中的实际地位

| 问题 | 答案 |
|------|------|
| locked 入场用 z 分数吗？ | **否** — regime 只用 trend_confidence + semantic_chop + box |
| features.yaml 为何列出？ | Phase 1 `mlbot research scan` 候选，寻额外 gate |
| 有 promote 到 archetype 吗？ | **无** — prefilter.yaml `rules: []` |
| 与 TPC 实验关系？ | TPC 扫过同类 z 特征；trend_scalp 未绑定结论 |

### 6.3 有效性判断（研究视角）

**潜在价值**：
- z 分数捕捉 **相对异常的订单流聚类**（如突然一边倒的 run），可能过滤「趋势特征满足但微观结构恶化」的坏样本

**当前局限**：
1. **未过 Phase 3 分段回测门禁** — 无「加 z gate 后四段 OOS 提升」的 locked 证据
2. **与 trend_confidence 部分重叠** — 大行情时 imbalance z 常与价格方向同向，增量信息需 plateau 扫描证明
3. **窗口 20/50 bar@2h** = 40h～100h，与 trend horizons 3/5/10 尺度不同，需避免手调阈值
4. diagnose 默认路径 **不依赖** z 列；FeatureStore 缺列时不会自动失败 — 扫描与回测须同一 feature surface

**结论**：z 分数是 **合理的候选特征**，但在 trend_scalp **当前 prod 栈中未证明有效**；不能把它当作「策略能识别区域」的原因。区域识别来自 **trend_confidence + semantic_chop + box_prefilter**。

若要做有效性验证，应按 R&D 流程：

```bash
# Phase 1：plateau / pair-scan on parquet
mlbot research scan --hypothesis-yaml config/experiments/.../rd_loop_*.yaml

# Phase 3：入选规则写入 prefilter 后，重跑四段
python scripts/experiment_trend_scalp_market_segment.py ...
```

---

## 7. 综合判决表

| 问题 | 判决 |
|------|------|
| 特征设计是否合理？ | ✅ 分层清晰，regime 门与执行参数分离 |
| 能否识别趋势/震荡/箱体？ | ✅ 通过 trend/chop/box 迟滞段；非 z 分数 |
| 策略机制是否合理？ | ✅ 有限顺势加仓 + fee-aware TP；⚠️ 依赖正确 hold 语义 |
| 回测步骤是否正确？ | ✅ 主链路正确；⚠️ prod 默认 hold 缩放缺失；❌ 无 funding |
| 分几步？ | Phase 0 特征 → 1 段发现 → 2 段内模拟 → 3 汇总 → 4 OOS → 5 压力 |
| 是否适配牛熊最近？ | ✅ 四段 OOS 全正（hold_scaled）；❌ 不适配纯震荡（by design） |
| z 分数是否有效？ | ⬜ 研究候选，**未 promote**；locked 逻辑不依赖 |

---

## 8. 建议的下一步（按优先级）

1. **执行对齐**：prod archetype 或 `dual_add_backtest` 默认 `scale_max_loser_hold_to_signal: true`
2. **live 对齐**：`dual_add_trend_live_engine` 实现 loser_timeout + hold 缩放
3. **联合回测**：`backtest_multileg_timeline`（chop + trend 占槽），验证真实组合频率
4. **z 分数**：若要上，走 Phase 1 scan → DECISION.md 定 τ → Phase 3 四段验证
5. **promote 门禁**：constitution 重评 + recent_6m rolling 监控

---

## 9. 复现与结果路径

| 内容 | 路径 |
|------|------|
| 全窗 hold_scaled | `results/trend_scalp/hold_scaled_validate/` |
| OOS 4bps | `results/trend_scalp/oos_segment_20260618/` |
| OOS 20bps | `results/trend_scalp/oos_segment_20bps_20260618/` |
| 20bps A/B | `results/trend_scalp/stress_20bps_20260618/comparison.csv` |
| 市场分段定义 | `config/market_segment.yaml` |
| 策略 archetype | `config/strategies/trend_scalp/archetypes/` |
| loser_timeout 说明 | `TREND_LOSER_TIMEOUT_优化说明_CN.md` |
