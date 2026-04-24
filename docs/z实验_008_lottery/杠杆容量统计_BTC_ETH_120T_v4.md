# 杠杆容量统计 v4 — YAML 驱动 + 更严牛门（BTC ∧ 6M 月收益）

## 1. 配置单一数据源

- **`config/strategies/bad-candidates/lottery100/leverage_capacity_v4.yaml`**
- 脚本：**`scripts/analyze_leverage_capacity_v4.py`**（`--config` 可覆盖路径）
- 默认宏观门（`bull_regime` = 各启用子条件 **AND**）：
  1. **周线**：`W-FRI` 收盘 > 周线 EMA(50)，滞后 `lag_weeks: 1`
  2. **月线**：6 个月收益（`pct_change(6)` 在月度末收盘序列上）> `min_return`（默认 `0`），再滞后 `lag_months: 1` 展开到 bar
  3. **可选**：`weekly_ema_uptrend`（周线 EMA 高于 `weeks_back` 前的自己），默认 **关闭**

样本行携带分量：`regime_weekly`、`regime_6m`、`regime_slope`、`bull_regime`。

## 2. 覆盖率（相对 v3）

| 指标 | v3（仅周线） | v4（周线 ∧ 6M>0） |
|------|----------------|---------------------|
| `bull_regime` 为 True 的 bar 占比（锚在 BTC 120T 全段） | ≈ 79.2% | **≈ 67.4%** |

`--bull-only` 后样本量（BTC+ETH × 2 horizon × 2 side）：约 **20976** 行（v3 bull_only 约 24672）。

## 3. 决策树快照（H=120 long，`--bull-only`，train → test 牛市窗）

配置与窗口见 YAML `windows`；与 v2/v3 默认一致：train 2022-08~2023-09，test 2023-10~2024-03。

- **Test**：base ≥100x long ≈ **2.6%**；树预测默认阈值 precision ≈ **4.2%**，lift ≈ **1.65×**
- **Top 1%**（test）：precision ≈ **33.3%**，lift ≈ **12.9×**（n=36，hit=12）

完整规则与 OOS：`reports/leverage_capacity_v4_bull_only/tree_rules_H120_bull_only.md`。

## 4. OOS（2024-04 ~ 2026-02）long，top 1%

在 v4 更严牛门下，OOS **long top 1%** precision ≈ **4.7%**，lift ≈ **1.31×**（仍偏保守，但高于「无牛门」时 v2 的极端弱化）。

## 5. 运行示例

```bash
python scripts/analyze_leverage_capacity_v4.py \
  --config config/strategies/bad-candidates/lottery100/leverage_capacity_v4.yaml \
  --bull-only \
  --output-dir reports/leverage_capacity_v4_bull_only
```

调参：编辑 YAML 中 `monthly_6m_return.min_return`、`weekly_ema_uptrend.enabled` 等，勿改脚本常量。

## 6. 后续

- [ ] 脚本可选读取 `gate_draft.yaml` 中的 **特征门** 做第三段过滤（当前 v4 仅宏观门）
- [ ] `min_return` 网格（0 / 0.05 / 0.10）做敏感性表
