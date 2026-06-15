# T5 — 订单墙 / 清算簇：设计稿（Phase 0）

> **状态**：设计稿（2026-06-14）  
> **路线图**：[产品路线图_TODO优先级_CN.md](产品路线图_TODO优先级_CN.md) §T5（**P1 当前主攻**）  
> **R&D 纪律**：[config/experiments/README.md](../../config/experiments/README.md) — **Phase 1 scan → DECISION → 再 grid/live**

---

## 0. 符号说明与讨论纪要（2026-06-14）

### 0.1 T5α / T5β 是什么？（≠ A 层 beta）

| 记号 | 读法 | 含义 | 数据 |
|------|------|------|------|
| **T5** | 路线图 backlog ID | 订单墙 + 清算簇 **总课题** | — |
| **T5α** | T5-**wall** / 挂单墙线 | Resting orderbook 大墙、突破/拒绝 | **L2 depth**（§15，未接） |
| **T5β** | T5-**liq** / 清算代理线 | OI 急变、费率极端、拥挤度 → 级联/耗尽 **代理** | **已有** OI/Funding 列（§12.1） |

**α / β 只是两条子假设的编号**，与 ABC 架构里「A 层 beta 容器」**无关**。若易混淆，口头可说 **「墙线 / 清算线」**。

两条线 **分开 scan、分开 DECISION**；过关后 **优先嵌进 TPC**（gate / entry / execution），**不**默认新开 OWB 策略（§2.3、§6.1）。

### 0.2 战略取舍（已写入路线图）

| 项目 | 决定 |
|------|------|
| T1 `rolling_trend` live | ❌ 放弃 |
| A 层 `profit_satellite` live | ❌ 放弃 |
| **当前主攻** | **T5** Phase 1A（T5β scan）→ 嵌 TPC |
| BPC live | 不优先；独立 edge 弱于 TPC，仅作以后「TPC 加仓 leg」候选 |

### 0.3 TPC 怎么用 T5？（不是新策略配对）

T5 特征挂在 **同一笔 TPC** 的 archetype 层上，不是 chop_grid+trend_scalp 式双账户：

| 层 | 文件 | T5 用法 |
|----|------|---------|
| gate | `gate.yaml` | 极端清算应力 → **deny 新开**（如 `oi_flow_zscore > 3`） |
| entry_filters | `entry_filters.yaml` | 墙被吃 / 吸收确认 → **allow 进场** |
| execution | `execution.yaml` | 前方卖盘墙 → 收紧 trailing；止损锚墙外（**Scan C 后再验**） |

**实验顺序**：入场 Scan A/B → 改 gate/entry → 持仓路径 Scan C → 最后才动止损/trailing。

### 0.4 币安图里的「1000」与墙体量（易混）

| 概念 | 含义 | 跨币用法 |
|------|------|----------|
| App 步长 **1000** | **价格**每 1000 **USDT** 一档（65k/66k） | 仅 BTC 高价区合理；**不能**全币统一 |
| 档上 **738 BTC** | 该价位 **挂单张数**（墙体量） | BTC ~$50M+ 才算「大墙」量级参考 |
| **1000 BTC** | 体量直觉（非分档单位） | **不能**套到 ETH（应用 USD 或 %OI） |

默认：**价格档** = `mid × 0.5%` 或 `0.25×ATR`；**墙阈值** = USD 名义 / OI 占比 / z-score（§15.2）。

### 0.5 T5 vs BPC：先做谁？

**先做 T5β scan**（现有列、服务 live TPC）。BPC 不抢资源：highcap 仅 `tpc` live，BPC PF 回测弱于 TPC。

---

## 1. 动机（来自你的观察）

两张典型图：

| 图 | 含义 |
|----|------|
| **币安 App 委托订单（步长 1000）** | **价格轴**按 **1000 USDT 一档** 聚合（64k/65k/66k）；**数量**是该档挂单 **BTC 张数**（如 738 BTC），不是「1000 BTC 一桶」 |
| **Bookmap 热力** | 价格 × 时间；亮色带 = 某价位 **resting liquidity** 随时间 persistence |

原始想法：**找到「大挂单墙」所在价位 → 以此为结构做 swing 入场**。

必须正视的三个难点：

1. **撤单 / spoof**：墙可能是假流动性，快照一瞬间很大，几秒后消失。  
2. **数据平面**：当前 B 层 live 主路径是 **120T（2h）+ feature bus OHLC/订单流成交**，**没有** L2 深度墙的稳定特征列。  
3. **与现有策略边界**：语义上接近 **SRB（关键位突破）** 和 **LV（清算级联/反转）**，不宜再开一条「平行 TrendSwing 家族」而不做 scan。

---

## 2. 核心建议（先读结论）

### 2.1 不要先做「chop_grid + trend_scalp」式配对

C 层配对能成立，因为：

- **edge 相反**（震荡网格 vs 趋势段）  
- **同一账户、同一 symbol 互斥**（宪法强制二选一）  
- **regime 开关清晰**（`semantic_chop` vs `trend_confidence`）

订单墙 / 清算 **不是** 这种关系。它更像是：

- **入场时机 / 结构确认**（「在这个价位有没有真墙、有没有被吃掉、有没有清算级联」）  
- 与 TPC 是 **同 regime（趋势 swing）** 下的 **信号层**，不是另一套 C 层库存引擎。

因此：**不要** 新建「TPC 账户 + OWB 账户」双系统并行；也 **不要** 复制 multi_leg 的 segment 配对模型。

### 2.2 也不要先「多开几个 TrendSwing」（SRB / BPC / ME 齐上）

[B系统.md](B系统.md) 已有共识：

- **Live 主攻 TPC**；BPC/ME/SRB 为研究候选  
- 多 archetype 平行 = **同向暴露叠加**，PCM 虽 `max_trend_slots_per_symbol: 1`，但轮换赢家仍可能全是同 beta  
- SRB 与「订单墙突破」**语义最近** — 应先验证是 **补强 SRB/TPC**，还是 **独立 edge**，而不是再堆 ME/BPC

**结论**：T5 资源用于 **假设验证 + 最小接入**；**不是** 扩 archetype 数量竞赛。

### 2.3 推荐路径（三阶段）

```text
Phase 1A（立刻，2h parquet）
  衍生品清算代理 scan：oi_z / funding / liquidation 相关列
  → 回答「清算应力」对 forward RR 有没有 lift

Phase 1B（并行设计，可后做实现）
  挂单墙特征规格 + L2 数据采集可行性
  → 回答「resting wall」是否比 SRB 的 L3 结构多解释力

Phase 2（仅当 1A/1B 有 plateau）
  优先接入形态（二选一，看 scan）：
    A) TPC gate / entry_filter 扩展   ← 默认首选
    B) SRB entry_filter 扩展          ← 若 lift 集中在「突破关键位」子集
  仅当 lift 高且与 TPC 低相关 → 才立项独立 archetype OWB

Phase 3
  event_backtest variant-grid + trading map 语义检查 → promote 门槛见 LAYER_PROMOTION_CRITERIA
```

| 选项 | 建议 |
|------|------|
| TPC + T5 配对（新账户/新 runner） | ❌ Phase 2 前不做 |
| 新建 OWB 与 SRB/BPC/ME 并列 | ❌ scan 未过关不做 |
| T5 → TPC/SRB 的 **gate 或 entry_filter** | ✅ scan 过关后的默认落地 |
| LV cascade/reversal 独立策略（分钟级） | ⏸ 需 tick 回放；先用 2h 代理 scan |

---

## 3. 假设拆分：两道完全不同的信号

不要把「订单墙」和「清算」混成一个策略名。

### 3.1 T5α — Resting Order Wall（挂单墙）

| 项 | 定义 |
|----|------|
| **观测** | L2 limit depth 在价格档 $P$ 的累计挂单量 $D_P$（**base 资产张数** 或 **USD 名义**） |
| **价格分档** | **自适应**（见 §15.2）；币安 UI 的「1000」= **1000 USDT 价格步长**（仅 BTC 高价区合理） |
| **墙体量阈值** | 「大墙」用 **USD 名义** 或 **相对 OI 的分位/z-score**；BTC 上 **~1000 BTC** ≈ 量级参考，**不能**套到 ETH/小币 |
| **方向语义** | **吃墙突破**（break & hold）vs **贴墙反弹**（reject）— **必须先 scan 定主假设** |
| **与 SRB** | SRB 用 **K 线推导的 L3 wide SR**；墙用 **交易所真实挂单** — 相关但不等价 |
| **主要风险** | spoof、撤单、仅快照无持续 |

**防 spoof 最小规则（设计，非最终 τ）**：

```text
wall_persist_min_sec   = 300      # 墙在档上持续 ≥5min
wall_cancel_rate_max   = 0.5      # 窗口内撤单量 / 峰值深度
wall_min_fill_ratio    = 0.05     # 价格触及后有一定成交消耗
wall_size_zscore_min   = 2.0      # 相对同档历史深度的 z-score
```

### 3.2 T5β — Forced Liquidation Cluster（清算簇）

| 项 | 定义 |
|----|------|
| **观测** | 强平成交 burst、OI 急变、费率极端、wick + 成交失衡 |
| **仓库锚点** | 见 §12 **T5β 已有列**；`liquidation_cluster_score` **未实现**；`bad-candidates/lv_liquidation_{cascade,reversal}` 为占位 |
| **时间尺度** | 真 LV 偏 **分钟～小时**；Phase 1 允许 **2h 代理**（与 TPC 同 bar） |
| **方向语义** | **级联跟随**（cascade）vs **耗尽反转**（reversal）— 两个子假设，分开 scan |
| **主要风险** | 二段级联、滑点、标签与执行周期错配 |

### 3.3 与 SRB / LV / FBF 对照

| 策略族 | 问的问题 | 数据 | 与 T5 关系 |
|--------|----------|------|------------|
| **SRB** | 结构关键位 **突破后** 能否延续？ | K 线 L2/L3 SR | T5α 墙突破 ≈ SRB 的 **微观确认层** |
| **LV cascade** | 强平后 **顺势** 还有一波？ | tick / 强平流 | T5β 子假设；placeholder 在 bad-candidates |
| **LV reversal** | 强平 **耗尽** 后反转？ | 同上 | T5β 子假设；与 FBF（假突破）部分重叠 |
| **FBF** | 假突破失败 → 反向短打 | 结构边界 | 与墙 **拒绝** 语义近，但 FBF 已归档待解释 fast/slow 差 |
| **TPC** | 大趋势内 **回调不破** 再买 | 2h 结构+订单流 | T5 宜作 **filter**，不宜另起炉灶 |

---

## 4. 数据架构

### 4.1 现状（能立刻 scan 的）

| 数据 | 路径/模块 | 用于 |
|------|-----------|------|
| 2h labeled parquet | `mlbot train final --prepare-only` | Phase 1A |
| OI / funding 列 | `oi_features_f`、`funding_rate_features_f`（§12、§14） | T5β |
| 订单流成交 | VPIN、footprint、trade cluster | 墙被「吃掉」的 **成交侧** 弱 proxy |
| **无** L2 深度历史 | — | T5α 全量 scan **blocked**（§15） |

### 4.2 T5α 需要新增（Phase 1B）

见 **§15 Orderbook 墙：采集与聚合**（价格分档 + 墙体量阈值 + 防 spoof）。

### 4.3 标签（Phase 1 scan）

与 B 层统一，优先复用现有 forward RR / `success_no_rr_extreme` 类标签；子样本条件写在 `condition-set` 里，不急着新建标签引擎。

| 子假设 | 条件示例 | 看 lift 的 label |
|--------|----------|------------------|
| 清算应力 | `oi_flow_zscore >= 2` | forward RR @ 20bar |
| 费率极端 | `funding_rate_abs_zscore_50 >= 2` | 同上 |
| 墙近端（若有列） | `wall_nearest_dist_atr <= 1` | 同上 |
| cascade | `liquidation_burst` + trend 同向 | 同上 |
| reversal | `liquidation_burst` + wick 超 ATR | 同上 |

---

## 5. Phase 1 实验计划（可执行）

### 5.1 目录

```text
config/experiments/20260614_t5_liquidation_wall_scan/
  README.md
  DECISION.md
  rd_loop_t5_phase1.yaml
  quick_scan/          # mlbot research scan 输出
```

### 5.2 Scan 矩阵（先 β 后 α）

| ID | 工具 | 条件 / 特征 | 目的 |
|----|------|-------------|------|
| S1 | `condition-set` | `oi_flow_zscore >= {1.5,2,2.5}` | 清算应力分层 |
| S2 | `condition-set` | `funding_rate_abs_zscore_50 >= 2` | 费率极端 |
| S3 | `feature-plateau` | `funding_oi_crowding_score`, `oi_exhaustion_score` | 连续特征平台 |
| S4 | `ic` | 上列 + OI 变化族 | 与 label 秩相关 |
| S5 | `condition-set` | TPC prefilter 成立 **且** S1/S2 | **TPC 条件增益**（是否值得做 gate） |
| S6 | `condition-set` | SRB 结构成立 **且** S1/S2 | **SRB 条件增益** |

**通过门槛（草案，人审写在 DECISION.md）**：

- canonical segment 上 **稳定正 lift**（非单年偶然）  
- 样本量够（条件集 n 不低于同目录其它 B 实验）  
- plateau 可解释（不是极窄 τ 尖峰）  
- S5/S6 至少一侧显示 **边际增益** — 否则不接入 live

### 5.3 命令模板

```bash
# 0. parquet（若尚无近期 features_labeled）
mlbot train final --prepare-only --strategy tpc ...

# 1. 清算应力
mlbot research scan condition-set \
  --features-parquet results/<run>/features_labeled.parquet \
  --label <forward_rr_col> \
  --condition "oi_z: oi_flow_zscore>=2.0" \
  --output config/experiments/20260614_t5_liquidation_wall_scan/quick_scan/oi_z.md

# 2. 批量（rd_loop）
python -m scripts.rd_loop --hypothesis-yaml config/experiments/20260614_t5_liquidation_wall_scan/rd_loop_t5_phase1.yaml
```

---

## 6. Phase 2 接入形态（preview，scan 前勿实现）

### 6.1 默认：TPC 的 gate / entry_filter 扩展

适合当 S5 显示：**同一 TPC 语义下，清算/墙代理改善 entry 质量**。

**TPC 四层里 T5 能挂哪里**（不是新 OWB 策略，是 **同一笔 TPC 交易** 上的修饰）：

```text
prefilter（慢结构，已 locked）
  tpc_pullback_depth 等 — T5 一般不动这层（除非证明「深回调+清算」是全新 regime）

direction（多空）
  MACD × EMA1200 — 通常不动

gate（硬否决：这根 bar 不该做 trend）
  例：oi_flow_zscore > 3 → deny（极端清算乱流，不追回调）
  例：semantic_chop 已在这里 — T5 清算应力 veto 语义上最接近 gate

entry_filters（时机确认：prefilter+方向过了，还要不要在这根 bar 进）
  例：墙被吃掉比例 wall_eaten_ratio > 0.1 → allow
  例：回踩段 delta 吸收 — 现有 tpc_deep_pullback_vol_confirm 的「订单流同伴」
  组合：combination_mode: or / and（与现网 entry_filters.yaml 一致）

execution（持仓中：止损 / trailing / 结构出场）
  现网：take_profit.enabled=false；出场靠 ema1200 structural + trailing + breakeven
  T5 若用在执行层，不是经典「止盈价」，而是：
    · initial_r / 结构止损锚在墙外侧（像 SRB structural_sl）
    · 前方有卖盘墙 → 提前收紧 trailing 或触发减仓
    · 持仓中再爆清算级联 → 加速 trailing / 强制减仓（持仓路径特征）
```

**现网锚点**：

| 层 | 文件 | 今天长什么样 |
|----|------|----------------|
| gate | `tpc/archetypes/gate.yaml` | 仅 `tpc_semantic_chop > 0.4` deny |
| entry | `tpc/archetypes/entry_filters.yaml` | `tpc_vol_pullback_confirm >= 0.45`（OR） |
| execution | `tpc/archetypes/execution.yaml` | `initial_r: 4`、breakeven@6ATR、trailing@3.5R、`structural_exit: ema1200` |

```yaml
# 示意 — config/strategies/tpc/archetypes/gate.yaml
- id: t5_liq_stress_veto
  feature: oi_flow_zscore
  operator: ">"
  threshold: 3.0
  action: deny   # 极端清算应力下禁止新开（或仅 deny add）

# 示意 — entry_filters.yaml
- id: t5_wall_break_confirm
  feature: wall_eaten_ratio_1h   # Phase 1B 后有列再启用
  operator: ">"
  threshold: 0.1
```

**优点**：不增加 PCM archetype、不新账户、符合「先把 TPC 跑稳」。

#### 6.1.1 实验顺序：先 gate/entry，再 execution（止损/trailing）

| 阶段 | 回答的问题 | 工具 | 为何这个顺序 |
|------|------------|------|----------------|
| **Scan A** | 入场 bar 上 T5 特征对 **forward RR** 有无 lift？ | `condition-set` / `ic` on 全样本 | 最便宜；清算/墙信息在 **决策 bar** 最完整 |
| **Scan B** | lift 是否集中在 **TPC 子样本**（pullback 已成立）？ | S5：`tpc_pullback 成立 AND oi_z>=2` | 决定改 gate 还是 entry |
| **Scan C** | 持仓路径：接近墙 / 级联时 **MAE、MFE、是否该早退**？ | excursion profile / 分段 label | 才回答「止损止盈要不要用 T5」 |
| **Grid** | 过关后 ablation：仅 gate / 仅 entry / 仅 execution | event_backtest variant-grid | 三层 **分开 promote**，避免混因 |

**止损/止盈要不要用 T5？**

- TPC **没有固定 take_profit**；「止盈」= trailing + ema1200 结构出场。  
- T5 在执行层更自然的用法是：  
  1. **止损锚定**：多单止损放在 **买盘墙下方**（墙破则叙事失败）— 需墙特征 + 回测 structural_sl 路径  
  2. **trailing 调节**：价格距 **前方卖盘墙** < 1 ATR → `trail_r` 收紧（防撞墙回落）  
  3. **持仓 veto**：持仓中 `oi_z` 再飙升 → 视为级联续作，**加速出场**（不是入场 gate 的重复）  

**建议**：Phase 1 **默认只验证 gate + entry**；仅当 Scan A/B 显示入场 lift 弱、但 Scan C 显示 **持仓路径** 有区分度时，才开 execution 实验（否则 execution 改动面大、归因难）。

**OWB 独立策略**：仅当「全样本 lift 高 + TPC 子样本 lift 弱 + 与 TPC 触发重叠低」— 三道都不过就 **不做 OWB**。

### 6.2 备选：SRB entry 补强

适合当 lift **集中在突破位**，且 SRB 子样本比 TPC 更干净。

SRB 已有 `srb_sr_success_breakout_score` 在 entry；墙特征可作为 **第二条 entry OR 规则** 或 **2b 确认分数**。

### 6.3 仅当独立 edge 成立：新 archetype `owb`（Order Wall Break）

立项条件（同时满足）：

1. 条件集 lift 在 **非 TPC、非 SRB** 子样本仍显著  
2. 与 TPC 同时触发率 < 30%（避免双倍暴露）  
3. Phase 3 segment backtest + trading map 语义通过  

否则 **owb 不进入** `enabled_archetypes`。

---

## 7. 与 chop / trend_scalp / TPC 配对的对比表

| 维度 | chop + trend_scalp（C） | T5 + TPC（勿照搬） |
|------|-------------------------|-------------------|
| 账户 | 同一 multi_leg 账户 | B trend 账户 |
| 关系 | regime **互斥** | 同 regime **叠加风险** |
| 引擎 | 多腿库存 / 网格 | 单仓 PCM + trailing |
| 正确做法 | 宪法二选一 | **filter / gate 或** 单 archetype |
| 错误做法 | — | 双 runner 并行追墙 |

---

## 8. 显式不做

| 项 | 原因 |
|----|------|
| 未 scan 直接 live | 违反 experiments workflow |
| 用 24h 全市场涨幅类逻辑冒充墙 | 与 T5 无关（已放弃的 profit_satellite 路子） |
| 先扩 BPC+ME+SRB 再管 T5 | 分散假设验证 |
| T5 独立账户 + 独立杠杆 | 运维与 T2 调仓复杂度 ↑ |
| 分钟级 LV live 无 tick 回放 | `lv_liquidation_*` README 已写明 |
| 把 spoof 墙当硬结构止损唯一依据 | 墙消失 → 结构失效太快 |

---

## 9. 开放问题（需你拍板）

1. **T5α 主语义**：先做 **吃墙突破（with cascade）** 还是 **贴墙反弹**？（建议 scan 两个方向各自的 forward path，不要先赌）  
2. **价格分档**：默认 **mid × 0.5%** 或 **0.25×ATR**（§15）；币安 UI「1000 USDT 步长」仅作 BTC 对照  
3. **Phase 1B 数据**：是否加 **depth WS/定时 snapshot**（§15.1）  
4. **PCM**：宪法已 enable `tpc/bpc/me/srb`；T5 过关后是否 **只改 tpc yaml**，暂不动 `enabled_archetypes` 扩员？

---

## 10. 一句话结论

> **T5 不是新 C 层配对，也不是先多开 SRB/BPC/ME。**  
> 先把 **清算应力（2h 现有列）** 和 **挂单墙（需 L2）** 拆成两条假设做 Phase 1 scan；过关后 **优先嵌进 TPC 或 SRB 的 gate/entry**，只有证明 **独立 edge** 才新 archetype。  
> 你图的 65k/66k 大墙，本质是 **T5α**；撤单问题靠 **持久度 + 成交消耗** 过滤，单靠快照不够。

---

## 11. 配置锚点

| 用途 | 路径 |
|------|------|
| 实验目录 | `config/experiments/20260614_t5_liquidation_wall_scan/`（待建） |
| LV 占位 | `config/strategies/bad-candidates/lv_liquidation_{cascade,reversal}/` |
| SRB 突破 | `config/strategies/srb/` |
| TPC live | `config/strategies/tpc/archetypes/` |
| PCM 优先级参考 | `src/time_series_model/portfolio/live_pcm.py`（历史 LV>FER>ME>BPC，**以 constitution 为准**） |
| OI/Funding 刷新 | `scripts/refresh_funding_oi_data.py` |
| 数据审计 | `scripts/audit_funding_oi_coverage.py` |

---

## 12. T5 特征全集（列名对照）

设计稿口语名 ≠ parquet 列名；**scan 必须用下列实际列名**。

### 12.1 T5β — 已有（`tpc/features.yaml` 已请求，prepare-only 可产出）

| 分组 | parquet 列名 | 模块 | T5 语义 |
|------|--------------|------|---------|
| OI 存量 | `oi_usd`, `oi_zscore` | `open_interest_features.py` | 拥挤 / 杠杆环境 |
| OI 流量 | `oi_change_pct`, **`oi_flow_zscore`** | 同上 | ΔOI 异常（文档旧名 `oi_change_zscore` → **此列**） |
| OI 方向 | `oi_delta_price_sign` | 同上 | 价-OI 同向/背离 |
| OI 场景 | `oi_compression_score`, `oi_ignition_score`, `oi_absorption_score`, `oi_exhaustion_score`, `oi_trend_divergence_score` | `oi_scene_semantic_scores` | 级联/耗尽叙事 |
| 费率 | `funding_rate`, `funding_rate_abs`, `funding_rate_change_1`, `funding_rate_zscore_50`, **`funding_rate_abs_zscore_50`** | `funding_rate_features.py` | 极端费率（旧名 `funding_rate_extreme` → **此列**） |
| 拥挤交互 | **`funding_oi_crowding_score`** | `utils_interaction_features.py` | 高 OI × 高费率 |
| 订单流（弱 proxy） | `bpc_pullback_delta_absorption`, VPIN/CVD 等 | 订单流族 | 吸收/墙被吃 |

### 12.2 T5β — 未实现（Phase 1A **不要**写进 condition 直到有列）

| 列名 | 状态 |
|------|------|
| `liquidation_cluster_score` | 仅 A2 设计稿；无 Binance 2h 注册特征 |
| 强平 burst 序列 | 需 aggTrades / 第三方 |

### 12.3 T5α — 未实现（需 orderbook，§15）

| 规划列名 | 含义 |
|----------|------|
| `wall_bid_notional_usd_max` | 买盘侧最大墙 USD |
| `wall_ask_notional_usd_max` | 卖盘侧最大墙 USD |
| `wall_nearest_dist_atr` | 最近大墙距 mid（ATR 单位） |
| `wall_persist_sec` | 最大墙持续秒数 |
| `wall_eaten_ratio_1h` | 1h 内触及后成交量/墙量 |

---

## 13. 执行流程（推荐顺序 · T5 vs BPC）

```text
┌─────────────────────────────────────────────────────────────┐
│ 0. 数据就绪：§14 审计 OI/Funding；缺则 refresh（含 HYPEUSDT） │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Phase 1A — T5β scan（现有列，不改代码）                   │
│    S1–S4 全样本 → S5 TPC 子样本 → S6 SRB 子样本（可选）      │
│    工具：mlbot research scan condition-set / feature-plateau │
└────────────────────────────┬────────────────────────────────┘
                             ▼
                    quick_scan/*.md → DECISION.md
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
         lift 弱 / 无                   lift 有（S5 优先）
              │                             │
              ▼                             ▼
         放弃或转 1B                   Phase 2 接入 TPC
         orderbook 线                  gate 或 entry_filters
              │                      （§6.1，非新 OWB）
              ▼                             │
    Phase 1B：depth 采集 §15                  ▼
              │                      Scan C：持仓路径（可选）
              ▼                             ▼
    墙特征 scan → TPC/SRB        execution 层 ablation（最后）
```

### 13.1 为何先做 T5、不优先 BPC live

| | T5 Phase 1A | BPC |
|--|-------------|-----|
| Live | 嵌 **TPC**（highcap 仅 `tpc`） | 未上线 |
| 独立 edge | 待 scan | 回测 PF **1.37** << TPC **2.36** |
| 工程 | 扫现有列 | 整套 archetype + PCM 竞争 |
| 结论 | **当前主攻** | 仅「TPC 持仓后加仓 leg」值得以后单独验 |

### 13.2 TPC 怎么用 T5（过关后）

| 层 | 文件 | T5 典型用法 |
|----|------|-------------|
| gate | `gate.yaml` | `oi_flow_zscore > 3` → deny 新开 |
| entry | `entry_filters.yaml` | `oi_absorption_score` 高 / `wall_eaten_ratio` 高 → 确认进 |
| execution | `execution.yaml` | 前方卖墙 → 收紧 trailing；止损锚墙外（**Scan C 后再动**） |

**实验顺序**：Scan A/B（入场）→ 改 gate/entry → Scan C（持仓 MAE/MFE）→ 再动 execution。

### 13.3 Phase 1A 最小 scan 清单（复制即用）

1. `oi_flow_zscore >= {1.5, 2, 2.5}`
2. `funding_rate_abs_zscore_50 >= 2`
3. `funding_oi_crowding_score` — `feature-plateau`
4. `oi_exhaustion_score` vs `oi_ignition_score`（cascade / reversal 分开）
5. **S5**：`tpc_pullback_depth` 成立 `AND` 上列任一

---

## 14. OI / Funding 数据审计（2026-06-14 本机）

审计命令：`python scripts/audit_funding_oi_coverage.py`

数据根目录：

```text
data/funding_rate/parquet/<SYMBOL>_YYYY-MM_funding_rate.parquet
data/open_interest/parquet/<SYMBOL>_YYYY-MM_oi_5m.parquet
```

刷新（公开 API，无需 key）：

```bash
# 日常增量（Funding 可调 lookback；OI 实际最多 ~29 天）
python scripts/refresh_funding_oi_data.py \
  --symbols ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,HYPEUSDT --lookback-days 120

# 按月补历史（OI 5m）
PYTHONPATH=src python -m src.data_tools.download_open_interest \
  --symbols ETHUSDT BNBUSDT SOLUSDT XRPUSDT HYPEUSDT \
  --start-year 2026 --start-month 1 --end-year 2026 --end-month 6 \
  --period 5m --parquet-dir data/open_interest/parquet

# 新币 Funding 从上市月起
PYTHONPATH=src python -m src.data_tools.download_funding_rate \
  --symbols HYPEUSDT --start-year 2025 --start-month 1 \
  --parquet-dir data/funding_rate/parquet

python scripts/audit_funding_oi_coverage.py
```

### 14.1 highcap universe 覆盖（`live/highcap/universe.yaml`）

审计：`python scripts/audit_funding_oi_coverage.py`

**最近刷新（2026-06-15）**：`refresh_funding_oi_data.py` + `download_open_interest` / `download_funding_rate`（ETH/BNB/SOL/XRP/HYPE）。

| Symbol | Funding 末 | OI 5m 末 | 备注 |
|--------|-------------|----------|------|
| ETHUSDT | **2026-06-15** | **2026-06-15** | ✅ refresh + 2026 月文件 |
| BNBUSDT | **2026-06-15** | **2026-06-15** | ✅ |
| SOLUSDT | **2026-06-15** | **2026-06-15** | ✅ |
| XRPUSDT | **2026-06-15** | **2026-06-15** | ✅ |
| HYPEUSDT | **2026-06-15** | **2026-06-15** | ✅ 新币；OI 仅 **~2026-06-13 起**（更早月份 API empty） |

（BTC 未列入本次刷新；仓库内仍有至 2026-05-20 的历史。）

### 14.2 结论

- **本次 5 币（ETH/BNB/SOL/XRP/HYPE）**：Funding/OI 已拉到 **2026-06-15**，可跑 T5β Phase 1A。  
- **HYPE**：合约 **新上市**，历史 OI **只有约 2 天+**（自 2026-06-13）；scan 时样本偏少，或 Phase 1 先 **4 币**，HYPE 作 live 增量。  
- **OI 增量限制**：Binance `openInterestHist` 保留约 **29 天**；长期历史靠 `download_open_interest` 按月落盘；日常用 `refresh_funding_oi_data.py`（OI 部分 capped `_OI_RETENTION_DAYS=29`）。  
- Feature 层缺失：`on_missing: nan`，不崩，但该币 T5 列为空。

---

## 15. Orderbook 墙：采集与聚合尺度

**是的，T5α 必须要 orderbook（L2 depth）**；成交流/VPIN 只能当「墙被吃」的弱 proxy，替代不了 resting depth。

### 15.1 如何拿（由易到难）

| 方式 | API | 频率 | 用途 |
|------|-----|------|------|
| **A. REST 快照** | `GET /fapi/v1/depth?symbol=&limit=1000` | 30s～60s cron | 研究 / 2h bar as-of；weight 20 |
| **B. WebSocket** | `<symbol>@depth@100ms` 或 `@depth20@100ms` | 100ms 增量 | Live + 持久度/撤单率 |
| **C. 第三方** | Bookmap / Coinglass 等 | — | 人工对照；自动化需单独评估 |

**仓库建议（对齐 `refresh_funding_oi_data.py` 模式）**：

```text
scripts/refresh_depth_snapshots.py   # 待建
  → data/orderbook/parquet/<SYMBOL>_YYYY-MM-DD_depth_1m.parquet
  列：ts, side, price, qty, mid, spread_bps
  聚合后：wall_* 特征 join 到 2h bar（merge_asof backward）
```

Live VPS：在 `quant-trend-swing` 同机 **WS 订阅 universe 6 币** → 分钟级聚合 parquet 即可（2h 策略不需 100ms 全量存档）。

### 15.2 两个维度：价格分档 vs 墙体量（不要混）

你图里其实有两件事：

| 维度 | 币安 App「1000」 | Bookmap / 你说的「大墙」 |
|------|------------------|-------------------------|
| **价格分档** | **1000 USDT 价格步长**（65k 一档） | 热力图纵轴价位分辨率 |
| **墙体量** | 该档上的 **BTC 张数**（738 BTC） | 「算不算墙」的 **量级阈值** |

**错误**：全市场统一「1000 USDT 价格桶」或「1000 BTC 体量阈值」。

**推荐默认（跨币可扩展）**：

**（1）价格分档 `price_bin`** — 用 **相对尺度**，不用固定 USDT：

```text
bin_width = max(mid * bucket_pct, tick_size * N)

建议 bucket_pct：
  BTC/ETH 主流：0.005（0.5%）→ BTC@65k 约 325U 一档
  高波动 alt：0.01（1%）

备选：bin_width = k * ATR_1h（k=0.25～0.5）→ 波动大时档宽自动变大
```

对照：币安 UI 的 **1000 USDT 步长** ≈ BTC@65k 时 **1.5%** 一档，比 0.5% 更粗；研究时可 **并排** `{bucket_pct: 0.005, 0.01}` 做 plateau，不必死记 1000。

**（2）墙体量「大不大」** — 用 **USD 名义** 或 **相对 OI**，不用固定 BTC 张数：

```text
depth_notional_usd = sum(price * qty)  # 该 price_bin 内

is_wall 当满足任一：
  depth_notional_usd >= wall_usd_min(symbol_tier)
  depth_notional_usd / oi_usd >= wall_oi_frac_min   # 如 3%～5% OI
  depth_zscore_vs_7d >= 2.0                         # 相对自身历史
```

**体量参考（可调，scan 定 τ）**：

| Tier | 示例币 | `wall_usd_min` | 约合 BTC@65k |
|------|--------|----------------|--------------|
| T0 | BTC | **$50M～$80M** | **~770–1230 BTC**（你图 ~738 BTC 在此量级） |
| T1 | ETH | $15M～$25M | ~6k–10k ETH |
| T2 | BNB/SOL | $5M～$10M | — |
| T3 | XRP/ADA/HYPE | $2M～$5M | — |

「1000 BTC」**仅作 BTC 量级直觉**；ETH 不能用 1000 ETH（名义差一个数量级）。

### 15.3 聚合输出（写入 feature bus 的列）

每个 2h bar close（as-of 最近快照）：

```text
wall_bid_notional_usd_max     # 买盘最大墙 USD
wall_ask_notional_usd_max
wall_bid_price, wall_ask_price
wall_nearest_dist_atr         # min(距 bid 墙, 距 ask 墙) / ATR
wall_persist_sec              # 最大墙连续存在时间（需 WS 序列）
wall_cancel_rate_5m           # 撤单/新增比（防 spoof）
wall_eaten_ratio_1h           # 价触及后 1h 成交量 / 墙量
```

### 15.4 防 spoof（与 §3.1 一致）

单帧 `depth@1000` 不够；至少 **B 方案 WS** 攒 5～15 分钟序列再算 `persist` / `cancel_rate`。

---

*维护：Phase 1 产出 `quick_scan/*.md` 后更新 `DECISION.md`；OI/Funding 变更后重跑 `audit_funding_oi_coverage.py` 并更新 §14 表。*
