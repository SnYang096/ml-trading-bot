# B 系统：PCM 如何防止超过宪法预算

**问题**：PCM 应如何控制仓位暴露？

1. 每个 symbol 至少一个策略（6 币 ≈ 6 仓，加指数加仓回撤大）  
2. 每策略最多一仓 + 2 加仓腿（金字塔）  
3. 一次仅一个 symbol 可开仓，盈利 breakeven 后才能开第二个  

配合宪法控制，如何让 B 层以**稳定季度盈利**方式推进？

---

## 判断概要

当前配置存在**结构性风险**，需在并发 symbol、加仓 risk、月度节奏三层同时约束。

---

## 结构性风险

`max_unprotected_symbols: 1`，`max_symbols_after_unlock: 3`，每 symbol 主仓 + 2 加仓腿：

**理论最大风险 ≈ 3 symbol × 3 腿 × 1% = 9%。**

趋势反转时同向止损，月度亏损易顶穿宪法（如 12% max_dd）。这是「maxdd 直接超过宪法」的常见根源。

---

## 稳定季度盈利：三层

### 第一层：同时暴露的 symbol 数

`max_symbols_after_unlock: 3` 在趋势市合理，但 BTC/ETH/SOL 同开相关性近 1，等同三倍同向。

**建议**：symbol 相关性过滤——候选与已有仓位同向且相关性 > 0.8（可调）则不开新仓。

### 第二层：加仓腿 risk

主仓 1% + 两加仓各 1% → 单 symbol 最大 3%，三 symbol 即 9%。

**建议**：加仓递减——主仓 1%，第一加仓 0.5%，第二加仓 0.25%；单 symbol 最大 1.75%，两 symbol 约 3.5%。

### 第三层：月度亏损节奏

宪法常见 daily 6% / weekly 8% / monthly 12%。震荡市连续小亏易触月上限。

**建议**：月亏达 8% 时 **risk_per_slot 降至 0.5%**（软限制），而非立刻停开仓。

---

## 配置建议（示例）

```text
主仓：1% risk，TPC 触发
第一加仓：0.5% risk，BPC + require_locked_profit
第二加仓：0.25% risk，BPC/ME + require_locked_profit

max_unprotected_symbols: 1（不变）
max_symbols_after_unlock: 2（由 3 降至 2）
相关性过滤：同向 symbol 相关性 > 0.8 不新开

月度软限制：月亏 > 8% 时 risk_per_slot → 0.5%
```

最坏同时止损约 **2 × 1.75% = 3.5%**，较易留在宪法缓冲内。

---

## 一句话

**控制暴露的关键是最坏情况下的同时止损不超过宪法缓冲**——加仓递减 risk + 降低并发 symbol 上限，是最直接的两步。
