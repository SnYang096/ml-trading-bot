# SRB L3 Dynamic Trailing & Excursion 诊断（2026-04-22）

## 1. 动机

来自用户观察：
1. SRB 抓住趋势后**退出太早**（2024-11 XRP 大涨 +0.95 / +2.28 就走了）。
2. 在震荡区（2024-01 XRP、2024-06 XRP）**开仓→加仓→全部 SL**，−27R 级的灾难。

问题：这些是不可避免的 false-breakout？还是 trailing 逻辑问题可以修？

## 2. 方法

`scripts/simulate_srb_l3_trailing.py` 对 `reports/srb_break_level_attribution_v2_alltrades_trades.parquet`
的 106 首单逐 bar 回放（feature store `features_srb_120T_5643a66b47`）：

- **Part 1 — Excursion 诊断**：对每笔 trade 计算 MFE / MAE / captured_pct / 反向 L3 最近距离。
- **Part 2 — Trailing 扫参**：simulator 带 5 个旋钮
  - `activation_r`：trailing 激活门槛（MFE ≥ N R 激活）
  - `m_far`：远离反向 L3 时 trailing = M × ATR
  - `m_near`：接近反向 L3 时 trailing = M × ATR（`m_near == m_far` = 无 L3 动态）
  - `thr_l3_atr`：反向 L3 距离阈值
  - `breakeven_lock_r`：MFE ≥ N R 时把 SL 抬到 entry（0 = 关）

> **Caveat**：simulator 是 naive 模型，**没有**还原生产的 structural_opposite_sr stop、add_position、reverse；所以 alt 的绝对数值存在偏差，只用 **delta / 方向 / 量级** 做判断。

## 3. Part 1 发现：leftover profit 金矿

| exit_reason | n | avg_pnl | avg_mfe | **leftover** |
|-------------|---|--------:|--------:|---:|
| sl          | 53 | −1.02 | +0.55 | **+1.57** |
| trailing_sl | 50 | +1.72 | +2.84 | **+1.12** |

两个关键信号：
- **SL 亏损单**：平均曾经盈利 +0.55R 才被打回 SL。轻度 breakeven 能救一大批。
- **Trailing 盈利单**：平均留了 +1.12R（60% captured pct）。窄 trailing 切得太早。

## 4. Part 2 发现：activation_r 是主矛盾，L3 dynamic 是加成

106 首单 ORIGINAL：totalR = +30.72, meanR = +0.29, win = 0.453。

### 按 (activation_r, breakeven) 分组，比较同组最优 fixed 与最优 L3-dynamic：

| activation_r | breakeven | fixed best Δ | dynamic best Δ | **dyn 增量** |
|-------------:|----------:|-------------:|---------------:|-------------:|
| 1.0 | 0   | +29.7 (m=5) | +32.8 (7/5)  | **+3.1** |
| 2.0 | 0   | +19.6 (m=5) | +24.2 (7/5)  | **+4.6** |
| 3.0 | 0   | +20.3 (m=5) | +24.4 (7/5)  | **+4.1** |
| 6.0 | 0   | +20.4 (m=5) | +19.3 (7/5)  | **−1.1** |
| 1.0 | 1.0 | +7.6  | +12.7 | +5.1 |
| 2.0 | 1.0 | +3.8  | +9.3  | +5.5 |
| 6.0 | 1.0 | +2.5  | +2.4  | −0.2 |

### 结论

1. **主贡献 ≈ +20~30R** 来自 `activation_r: 6.0 → 1.0~3.0`（让 trailing 更早激活）。
2. **L3 dynamic 在低 activation 时额外 +3~5R（10-20%）**，`m_far=7, m_near=5, thr=2.0` 稳定最优。
3. **L3 dynamic 在生产默认 activation=6 时无效甚至反向** —— trailing 根本没激活，动态阈值没机会起作用。
4. **`breakeven_lock_r=1.0` 全组为负**（提早保本 = 被 1R 级正常回调洗出）。`breakeven=2.0` ≈ OFF。

### 最优配置

```yaml
# 建议实验参数（基于 naive simulator，需配 rolling_sim 验证）
stop_loss:
  trailing:
    enabled: true
    activation_r: 1.0          # 6.0 -> 1.0: MFE 1R 就激活
    # 动态 trail_r，由反向 L3 距离切换（语义与用户建议一致）
    trail_r_far: 7.0           # 远离反向 L3（distance >= thr_atr）
    trail_r_near: 5.0          # 接近反向 L3
    l3_near_threshold_atr: 2.0
  breakeven:
    enabled: false             # 保持关闭（1R 过早，2R 等效关）
```

Sim 结果：**+30.7 → +62.1R（+33R / 翻倍）**，胜率 0.453 → 0.535，81 改善 / 18 变差。

## 5. Part 3：XRP 用户例子复盘

`scripts/srb_l3_trail_xrp_detail.py` 用最优配置跑 XRP 首单：

| entry_time | side | orig | alt | Δ | 说明 |
|---|---|--:|--:|--:|---|
| **2024-01-03 16:00** | **SHORT** | **−1.01** | **+0.68** | **+1.69** | 你的"震荡区灾难"首单。低 activation + L3 trail 救回。|
| 2024-06-12 20:00 | SHORT | −1.02 | −0.19 | +0.84 | 类似场景，SL 损失收窄 80%。 |
| **2024-11-15 18:00** | **LONG** | **+0.95** | **+2.43** | **+1.48** | 你的"大涨拿不住"首单。低 activation 多拿 1.5R。 |
| 2024-11-29 18:00 | LONG | +2.28 | NaN | — | feature store 边界（只到 2024-11-30）。 |
| 2024-07-13 22:00 | LONG | −1.00 | +0.41 | +1.42 | 原来直接 SL，alt 锁到 +0.4R。 |
| 2024-03-20 00:00 | SHORT | +0.22 | +0.89 | +0.67 | 改善。 |
| 2023-10-09 14:00 | SHORT | +0.01 | +0.43 | +0.43 | 改善。 |
| 2023-11-20 00:00 | LONG | −1.02 | −1.00 | +0.02 | wide_dist=11 即时反转。**真失败信号，不可救**。 |
| 2024-05-20 04:00 | SHORT | −1.03 | −1.00 | +0.03 | 7 bars 即 SL。**真失败**。|

**XRP 首单 totalR：orig = −2.74R → alt = −2.41R**（改动有限，因为 XRP 很多 trades 是 MFE 极小的真失败）。

**但注意**：**XRP 2024-01 灾难的 −27R 中 −24R 来自 3 个 add_position**，simulator 只跑首单，没覆盖 adds。

关于"是否可避免"：
- **"大涨拿不住"（2024-11 类）：可避免**。activation_r 降低能多拿 50-100% MFE。
- **首单震荡区 SL（2024-01-03、2024-06-12 首单）：部分可避免**。降 activation 能把 −1R 首单变成 +0.4~0.7R。
- **2024-01 adds 的 −24R 灾难：需要独立修**（与本次 trailing 无关）。
  - 数据：add 1 at wide_dist=7.89, add 2-3 at wide_dist≈10 → 深度 in-profit 时加仓，**但 mother SL 没有随 MFE 抬到 breakeven**，反转时 3 笔全打回原始 SL。
  - 对症的修：MFE 触及 N R 后 **母仓 SL 抬到 breakeven 或半程锁定**，adds 共享。
- **wide_dist=11 的真失败（2023-11-20 LONG / 2024-05-20 SHORT）：不可避免**。

## 6. 下一步

### A. 实现 trailing 重构（主路径，期望 +30R）
- `config/strategies/srb/archetypes/execution.yaml`:
  - `trailing.activation_r`: 6.0 → **1.0**（或 2.0 保守）
  - 新增 `trailing.trail_r_far = 7.0`, `trailing.trail_r_near = 5.0`, `trailing.l3_near_threshold_atr = 2.0`
- `src/time_series_model/live/position_logic.py`（或对应的 trailing 实现）：读取反向 L3 (`wide_sr_upper_px` / `wide_sr_lower_px`)，按距离切换 `trail_r`
- 跑 rolling_sim 验证

### B. 母仓 SL → breakeven 锁（独立改动，针对 XRP 2024-01 灾难）
- 需要先诊断：母仓 MFE 多大时 adds 被堆起来？
- 改动点：position tracker / position_logic
- 语义：MFE ≥ `mother_breakeven_at_r`（候选 3.0）时把母仓 SL 抬到 entry；adds 共享
- 这 block 不在 trailing 重构内，独立实验

### C. 不做的方向（已否决）
- ~~"震荡区 regime filter 在 entry 层拒单"~~（上次诊断：误杀很多有效 compression breakout）
- ~~breakeven_lock_r = 0.5 / 1.0~~（定量灾难，−18~28R）

## 7. 相关产出

- `reports/srb_l3_dynamic_trailing.json`：全扫参结果
- `reports/srb_trade_excursions.parquet`：per-trade MFE/MAE/captured_pct
- `scripts/simulate_srb_l3_trailing.py`：主 simulator
- `scripts/srb_l3_trail_xrp_detail.py`：XRP 细节对比

---

## 8. 对话沉淀（2026-04-22 后续讨论）

### 8.1 核心论断："低波动区开仓是 alpha，执行层非对称才是正解"

完整论证链：

1. **数据事实**：入场瞬间 `trend_r2_20 ≤ 0.1`（低波动 / 压缩）的 SRB 首单 meanR **+1.45**、win **0.571** ——
   显著正 alpha，不是噪声。
2. **入场时刻的不可分性**：SR 突破的那一瞬，真趋势启动和假突破噪声在 `trend_r2 / bb_width /
   wide_sr_side` 上**同分布**。典型的"压缩 → 扩张"过渡状态。入场侧 regime filter 会把好坏一起砍掉。
   - 反例：2024-01-03 XRP SHORT（灾难）与 2024-11-15 XRP LONG（大涨）在入场时 `r2 / wide_sr_dist /
     bb_width` 都处在同一个压缩 → 突破的窗口上，外观几乎不可区分。
3. **Hold 期间的可分性**：hold 几根 bar 后，两类形态就开始分离（MFE 继续扩张 vs MFE 卡在 0.5-1R 掉头）。
   → **只有执行层能利用这个事后信息**。

**结论**：SRB 入场侧只保留"**结构几何硬约束**"（`sr_wide_entry_guard`：反手空间 ≥ 2 ATR），不做"形态预测"。
其他所有分辨真假突破的工作交给执行层非对称管理。

### 8.2 执行层三把锁

| 工具 | 治什么 | 数据 leverage |
|---|---|---|
| **Trailing 更快激活 + L3 动态** | 事后发现是真趋势 → 立刻锁 MFE，不让 MFE +2R 回 0 | Sim +33R / XRP 2024-11 从 +0.95R → +2.43R |
| **母仓 MFE ≥ 3R 后 SL 抬到 breakeven，adds 共享** | 事后发现是大趋势 → 即便反转也不亏 | XRP 2024-01 mother MFE≈10R 时锁 → adds 反转 0R 出场（原 −24R）|
| **Add 执行层形态筛选**（新机制） | 事后看加仓时价格根本在震荡 → 不堆 adds | 需 sim 扫参量化 |

### 8.3 Add 筛选机制的状态澄清

- 现有 `srb_add_position_policy` 已有 **regime bucket + volume_compression** 两个 gate（[scripts/event_backtest.py:956-961](scripts/event_backtest.py)）。
- `float_r_ladder_only` 触发器**主动 bypass** `validate_add_position_trigger` 的特征筛选
  （[src/time_series_model/core/constitution/add_position_rules.py:253-258](src/time_series_model/core/constitution/add_position_rules.py)，
  注释："此处不附加特征条件"）。
- **"加仓时价格根本在震荡"属于 bypass 留下的空档**，需要在 `srb_add_position_policy` 下新增
  `post_hoc_shape_gate` 子树专门处理，不改 constitution 层。

### 8.4 落地路线（本次新增 Phase A-D）

```
Phase A: Trailing 重构（activation_r=1 + L3 dynamic）—— 主升路径，期望 +33R
    ↓
Phase B: 母仓 MFE breakeven lock —— 独立修，治 adds 堆叠灾难
    ↓
Phase C: 验证 sr_wide_entry_guard 是否在 2024-01-03 类 case 真的拦住
    ↓
Phase D: Add 形态筛选（新）—— retrace_guard / recent_momentum / r2_gate / wide_sr_expansion
         4 子门 all default-off，simulator 逐门扫参后决定
```

详见 [.cursor/plans/srb_execution_refactor_a-d_*.plan.md](../.cursor/plans) 执行计划。

---

## Rolling_sim 终审（2026-04-22，16 月全量）

Run: `results/srb/research_roll.features_on/_rolling_sim/20260422_143930`（NEW = Phase A+B+D momentum enabled）
Baseline: `results/srb/research_roll.features_on/_rolling_sim/20260421_222624`（OLD = 改动前）

| 指标 | NEW | OLD | Δ |
|---|---:|---:|---:|
| totR | +133.4 | +205.5 | **−72.1** |
| mother_r | 25.9 | 30.7 | −4.8 |
| add_r | 107.5 | 174.7 | −67.2 |

### 诊断

- **Phase A `activation_r: 6→1` 是主要元凶**：simulator 扫参预测 +33R，实跑 −72R。
  - 2023-11（BTC 牛市起点）：totR +115.2 → +36.7（**−78.5R**），NEW 仅 3 笔 add_trades，OLD 9 笔；后续月加仓都被提前打掉。
  - 2024-03（ETH/BTC 大趋势月）：add 平均 R 从 +14.12 掉到 +11.36 × 8 ≈ −22R。
  - 原因：激进激活会连带收紧 add leg 的 SL，并在趋势中段被小回撤打掉 → 后续月 `add_position_ok` 骤降。**扫参只用母仓 trade，漏掉了 add leg 连锁反应**。

- **Phase B `mother_breakeven` 是干净正贡献**：
  - 2024-06 XRP：OLD −7.99R（3 adds 拖垮）→ NEW +0.55R（**+8R**）
  - 2024-08 XRP：OLD −13.87R → NEW −2.02R（**+12R**）
  - 2024-11：OLD −10.6 → NEW +28.8（**+39R**）反转月母仓被锁 breakeven，adds 连带保护
  - 机制：不筛加仓，而是锁住"已确认大趋势"的全仓下行。对 MFE < 3R 就失败的灾难（XRP 2024-01-18 entry −34R）**无能为力**——那种是"母仓早死 + adds 错堆"问题。

- **Phase D `recent_momentum` 扫参 +2.87R → 实盘 0 次触发**：
  - rolling_sim 全 16 月 `shape_gate_*` 拒绝计数 = 0。
  - 扫参里 14/149 ≈ 9% 拒绝率在 rolling_sim 落到 0，因为加仓触发时机不完全一致。
  - 其他 3 个门（retrace / r2 / wide_exp）扫参里本身就是 −17 ~ −67R 灾难，更不能开。

### 数据结论（对用户"要不要加加仓形态限制"）

**不加**。扫参 reports/srb_add_shape_gate_sweep.json 清晰显示：
- `retrace ≥ 0.7`：**−17 ~ −22R**（误伤 MFE 回撤大的趋势初期加仓）
- `trend_r2_20 ≥ 0.3-0.5`：**−37 ~ −53R**（低 r2 低波动启动段正是 SRB 真正 alpha）
- `wide_sr_expansion ≥ 0.5-2.0`：**−67R**（拦掉 107-124 个加仓，灾难）
- `recent_momentum`：扫参 +2-3R / 实盘 0 次触发 → 可配置保留但默认关

SRB 加仓的 alpha 正好在"形态不漂亮"的时刻（低动量/大回撤/接近反向 SR），跟"低波动开仓是 alpha"同理 —— 事后形态门本质是**在 alpha 源头做 survivorship bias 筛选**。

### 最终保留配置

- **Phase A 回退**：`activation_r: 1.0 → 6.0`（主 trailing + regime_execution.buckets 全部）
- **Phase A 保留**：`trail_r_far=7.0 / trail_r_near=5.0 / l3_near_threshold_atr=2.0`（非对称 L3 逻辑本身没错，在 MFE ≥ 6R 真激活时才有效）
- **Phase B 保留**：`mother_breakeven.enabled=true, trigger_r=3.0, lock_level_r=0.0`
- **Phase D 全关**：4 个 shape gate 全部 `enabled: false`，配置项保留供后续实验
- **Phase C**：纯诊断无代码改动，`sr_wide_entry_guard` 已确认语义正确（XRP 2024-01-03 rev_dist=-0.45 意味价已越过 L3，guard 不该拦）

预期最终 rolling_sim：totR ≈ baseline +205R + Phase B 净贡献 (~+50R，2024-06/08/11 XRP 叠加) ≈ **+250~280R**。

