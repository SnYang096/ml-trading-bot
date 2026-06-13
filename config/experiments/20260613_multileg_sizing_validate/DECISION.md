# 多腿联合验证 — 实验结论

**日期：** 2026-06-13
**配置：** `segment_dd_target: 0.072`, equity 10,000, 1min 执行

---

## 1. MTM 止损（max_loss=0.02=180 USDT）— 没有用

| 段 | regime_only 收益 | mtm 收益 | 差值 |
|---|---|---|---|
| bear_2022 | +80.5% | +74.9% | -5.6% |
| bull_2023_2024 | +43.9% | +40.0% | -3.9% |
| recent_range_to_bear | +55.0% | +53.2% | -1.8% |
| recent_6m_oos | +14.4% | +14.3% | -0.1% |

- 所有 symbol/段，MTM 收益全部低于 regime_only
- 胜率几乎不变（<0.2%）
- SOLUSDT 受损最重（-16.3% 累计）
- **决策：保持 `risk_stop_mode: regime_only`**

## 2. Per-symbol 收益（trend_scalp, recent_6m_oos, regime_only）

| Symbol | 交易数 | 胜率 | 收益 | 最差单笔 |
|--------|--------|------|------|----------|
| HYPEUSDT | 374 | 75.7% | +37.71% | -$153 |
| ETHUSDT | 306 | 77.8% | +27.59% | -$145 |
| SOLUSDT | 286 | 75.2% | +21.66% | -$262 |
| XRPUSDT | 296 | 71.6% | +12.01% | -$148 |
| BTCUSDT | 301 | 74.4% | +10.53% | -$103 |
| BNBUSDT | 302 | 67.5% | +0.14% | -$208 |

## 3. Chop Grid（dense 3L, 2 bps fees）

| 段 | 收益 | Max DD | 胜率 |
|---|---|---|---|
| bear_2022 | +8.49% | -0.67% | 88.7% |
| bull_2023_2024 | +7.82% | -0.85% | 86.8% |
| recent_range_to_bear | +4.87% | -0.66% | 86.8% |

## 4. 关键修复

| Bug | 修复 |
|-----|------|
| calibrate_roll `fee_bps: 20` → 0% 胜率 | → 2.0 |
| net cap 滞后 sizing | 0.85→1.20, 0.45→1.00 |
| gross cap 阻塞 trend 多腿 | 1.60→2.70（方案 A） |
| HYPE 数据格式 | 与其他 symbol 相同（flat monthly parquet） |
| HYPE FeatureStore 缺 chop_grid layer | 待 compute |

## 5. 待跑

- [x] 联合回测（`variant_grid.yaml` → 4 segments × chop + trend → joint sim）
- [x] Constitution 矩阵 sweep（Python 注入，避免手写 YAML 缺字段）

## 6. Constitution 矩阵（2026-06-13）

**工具：** `python scripts/sweep_multileg_constitution_matrix.py`（从 live constitution deep-copy，单字段注入）

**根因（旧 A/C 相同）：** `sim_multileg_account` 只读 `kill_switch.max_dd`，忽略 `multi_leg.account.max_drawdown_pct`；`max_symbol_net_notional_pct` 未 enforcement → net_cap 变体无效。

**修复：**
- `resolve_multileg_sim_limits()` — 同步 live 路径（`min(kill_switch.max_dd, multi_leg.account.max_drawdown_pct)` + net caps）
- `simulate_account_with_constitution` — 分级熔断 `tier_derate` / `tier_daily_scaled`
- 矩阵结果：`quick_scan/constitution_matrix.{csv,md}`

| variant | max_dd | net_cap | halted | 备注 |
|---------|--------|---------|--------|------|
| prod | 20% | 1.80 | no | 主要拒单：net_cap（~21k），非 max_dd |
| max_dd_half | 10% | 1.80 | **yes** (2022-01-20) | 更紧 max_dd → 触发 halt；不是「避免停机」 |
| net_cap_100 | 20% | 1.00 | no | 比 prod 少 2 笔（30 vs 32） |
| fuse_daily_scaled | 20% | 1.80 | no | daily 拒单 434 vs 302，收益路径不变 |

**结论：** 手写 YAML 变体 A/C 相同是字段路径 bug；砍半 max_dd 会**更早** halt，不能解决 day-1 停机。下一步应调 `daily_loss_limit` 与 net_cap 的联动（`tier_daily_scaled` 已接入，需配合 segment_dd 回放）。
