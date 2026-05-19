# B 系统总结

B 系统是 **120T（2h）趋势 swing 组合**，与 A 层 `spot_accum_simple` 分工：A 负责长期「在车上」与 fattail；B 负责 **高置信度中短期趋势段**，目标 **稳定月度收益、回撤可控**，优化 **胜率 × 盈亏比**，不追 total R、不抢 fattail。

资金大致 **A > 60%**，B 为剩余趋势槽位；宪法 `max_dd: 20%`，PCM 统一管槽位与并发风险。

---

## 设计哲学（与 A 对照）

|            | A（spot_accum） | B（trend swing）           |
| ---------- | --------------- | -------------------------- |
| 核心       | 确保在车上      | 确保每次上车都值得         |
| 最怕       | 错过牛市没仓位  | 低置信度入场稀释 edge      |
| 对「错过」 | 不能接受        | 必须接受                   |
| 持仓       | 年级            | 天～周（~20 bar 因子视野） |

Portfolio 分工（文档共识）：

```text
fattail      → spot_accum（无结构止损、长持）
trend swing  → TPC / BPC / ME / SRB（结构止损、控 maxDD）
震荡收益     → chop_grid（与 TPC 天然对冲）
```

以前用大止损在 trend 里抓 fattail 会顶穿宪法 maxDD；现已把 fattail 还给 A，B 只做 swing。

---

## 统一架构：四层 + PCM

四个策略 **共用管线形态**，不是四个完全独立的「系统」：

```text
Prefilter  →  archetype 是否成立（结构/语义，尽量 locked）
Direction  →  多空（EMA1200 带通 + 各策略信号源）
Gate       →  拒掉大回撤/震荡区（统计 + semantic_chop 等）
EntryFilter→  订单流 + 时机（OR 组合）
Execution  →  止损/加仓/出场（当前四者 largely 同模板）
         ↓
      PCM（宪法槽位、每 symbol 最多 1 个 trend 仓、总风险上限）
```

**PCM 要点**（`constitution.yaml`）：
- `enabled_archetypes`: `tpc` → `bpc` → `me` → `srb`（同 bar 仲裁优先级）
- `max_trend_slots_per_symbol: 1`：同一币同时只一个 trend archetype
- `trend_pool_guard`：未保护前最多 1 个 symbol 开仓，breakeven 后可扩到 2
- 每笔 `risk_per_slot: 1%`，`max_risk_per_trade: 0.01`，可加仓最多 2 次且需 `require_locked_profit`

执行层当前四策略基本一致：`initial_r: 4`、EMA1200 结构出场、trailing、浮盈 R 阶梯加仓（0.5 / 0.25）。

---

## 四个子策略分工

| 策略    | 语义                                                   | Prefilter 锚点                               | 方向源                                              | Entry 侧重                      |
| ------- | ------------------------------------------------------ | -------------------------------------------- | --------------------------------------------------- | ------------------------------- |
| **TPC** | 大趋势内 **回调不破** 再延续；**不要求** Donchian 突破 | `tpc_pullback_depth` + 箱体边缘/突破         | MACD × EMA1200                                      | CVD 吸收 + 恢复力度             |
| **BPC** | **突破 → 回踩 → 延续**（三阶段语义锁定）               | 量能压缩、近端突破、回踩深度、恢复强度 + box | Donchian 方向 × MACD × EMA1200                      | 同 TPC 类订单流确认             |
| **ME**  | **动量扩张**（压缩释放后）                             | ATR 分位带、压缩/OI 门槛 + box               | 动量级联 × EMA1200；**仅做空**                      | box 边缘/突破 + VPIN 点火       |
| **SRB** | **关键 SR 真突破 → 顺势延续**                          | L2 `sr_strength_max`、频谱、L3 突破新鲜度    | MACD × EMA1200（与 TPC 区分在 **结构** 非方向公式） | `srb_sr_success_breakout_score` |

**共享 regime 规则**（四者 prefilter 均含）：
- **箱体中部不开仓**：`box_pos_120` 须在边缘或已有 `box_breakout_*`
- **Gate 统一拒高 chop**：`tpc_semantic_chop > 0.4`（震荡来回扫）
- **Direction 统一**：价须在 EMA1200 单侧带（`inner_abs: 0.03` 离开死区），且与 `ema_1200_slope_10` 同向

与 BPC/TPC/ME 的 **edge 差异**：SRB 吃「L3 关键位突破段」，不靠盒压缩/回踩叙事；与 FBF（假破反向）边界清晰。

---

## 研究结论与收敛方向（文档 + 回测）

**slow_rolling 单策略对比**（`tpc/readme.md`）：

| 策略 | 胜率  | mean_pnl_R | profit_factor |
| ---- | ----- | ---------- | ------------- |
| TPC  | 0.176 | **0.833**  | **2.36**      |
| ME   | 0.425 | 0.400      | 1.70          |
| BPC  | 0.205 | 0.239      | 1.37          |

解读：
- **TPC**：唯一 edge 最清晰，宜作 **主入场基准**
- **BPC**：独立入场 edge 弱，但 continuation 语义适合 **TPC 已持仓后的加仓**
- **ME**：胜率高但盈亏比差，最不稳定；若保留应 **最小仓位**
- **SRB**：研究候选，生产默认不自动同步 live；与前三互补 SR 突破语义

**不建议**三个独立策略平行全开（同向三倍暴露）；若保留多信号，应是 **一框架、三触发、统一仓位**：
- TPC 主仓 → BPC/ME 仅加仓且 capped
- 未触发 TPC 时 BPC/ME 独立入场需缩仓
- 用 PCM 回测验证：`TPC only` vs `TPC+BPC` vs `TPC+BPC+ME`（看 PF 与 maxDD）

现阶段文档倾向：**先把 TPC-only 跑稳**，再逐项加 BPC/ME 做边际验证；不必先建合并版 `trend-swing` 大系统。

---

## 运维心智：什么能动、什么别动

来自 `B系统不变的层.md` / `B系统运维心智梳理.md`：

```text
定期可动（慢变量）：
  regime — EMA1200、chop/box、semantic_chop

异常才查：
  Gate、EntryFilter — 一次统计定稿后 frozen/locked

几乎不动：
  入场形态（pullback/breakout/SR）、Execution 参数
```

维护负担大的根因不是「四个策略」，而是 **每层都在持续 SHAP/重训/改阈值**。收敛办法：

1. **方案二（规则管线）** 优于合并成一个黑盒树模型  
2. 入场条件 **定稿即锁**（BPC 三锚点、`locked: true` 已是方向）  
3. Gate 四策略 **同构 chop 门** + 各自 safety 带，不必四套独立优化循环  
4. **不要**为降 maxDD 收紧止损 → 应降 **每笔风险%**；连续止损先查 regime（震荡禁做）或 entry 质量

**连续止损 / maxDD 诊断**：
- 震荡市连续小亏 → regime 问题，不是 SL 太窄  
- 有趋势仍被洗 → entry 结构问题  
- 可同时多 symbol，但总敞口 cap（如 3–4%）；crypto 高相关，不是真分散  

---

## 与 A 的协作关系

- A 兜底「牛市必须在车上」；B **可以接受错过**，只做高置信度段  
- B **不追 fattail**；超级行情由 spot 阶梯卖承担  
- B 目标是在 **足够多样本** 下几个月维度期望为正（`PF > 1.5`），**不是每月必赚**；震荡月连续小亏属正常  

---

## 一句话

**B 系统 = 四条 120T 趋势 swing 规则链（TPC 为主、BPC/ME/SRB 语义互补），经统一 Gate/chop 过滤与 PCM 槽位风控，在高相关币上「少做但做对」；维护重点是 EMA1200/chop regime，入场与执行定稿后少动，用 PCM 组合回测决定 BPC/ME 是否值得叠加，fattail 留给 A。**
