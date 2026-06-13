# 多腿联合验证 — 实验结论

**日期：** 2026-06-13
**配置：** `segment_dd_target: 0.072`, equity 10,000, 1min 执行

---

## 1. MTM 止损（max_loss=0.02=180 USDT）— 没有用

| 段                   | regime_only 收益 | mtm 收益 | 差值  |
| -------------------- | ---------------- | -------- | ----- |
| bear_2022            | +80.5%           | +74.9%   | -5.6% |
| bull_2023_2024       | +43.9%           | +40.0%   | -3.9% |
| recent_range_to_bear | +55.0%           | +53.2%   | -1.8% |
| recent_6m_oos        | +14.4%           | +14.3%   | -0.1% |

- 所有 symbol/段，MTM 收益全部低于 regime_only
- 胜率几乎不变（<0.2%）
- SOLUSDT 受损最重（-16.3% 累计）
- **决策：保持 `risk_stop_mode: regime_only`**

## 2. Per-symbol 收益（trend_scalp, recent_6m_oos, regime_only）

| Symbol   | 交易数 | 胜率  | 收益    | 最差单笔 |
| -------- | ------ | ----- | ------- | -------- |
| HYPEUSDT | 374    | 75.7% | +37.71% | -$153    |
| ETHUSDT  | 306    | 77.8% | +27.59% | -$145    |
| SOLUSDT  | 286    | 75.2% | +21.66% | -$262    |
| XRPUSDT  | 296    | 71.6% | +12.01% | -$148    |
| BTCUSDT  | 301    | 74.4% | +10.53% | -$103    |
| BNBUSDT  | 302    | 67.5% | +0.14%  | -$208    |

## 3. Chop Grid（dense 3L, 2 bps fees）

| 段                   | 收益   | Max DD | 胜率  |
| -------------------- | ------ | ------ | ----- |
| bear_2022            | +8.49% | -0.67% | 88.7% |
| bull_2023_2024       | +7.82% | -0.85% | 86.8% |
| recent_range_to_bear | +4.87% | -0.66% | 86.8% |

## 4. 关键修复

| Bug                                    | 修复                                       |
| -------------------------------------- | ------------------------------------------ |
| calibrate_roll `fee_bps: 20` → 0% 胜率 | → 2.0                                      |
| net cap 滞后 sizing                    | 0.85→1.20, 0.45→1.00                       |
| gross cap 阻塞 trend 多腿              | 1.60→2.70（方案 A）                        |
| HYPE 数据格式                          | 与其他 symbol 相同（flat monthly parquet） |
| HYPE FeatureStore 缺 chop_grid layer   | 待 compute                                 |

## 5. Timeline 回测 — DD 矩阵 + 时间过滤实验

**周期：** 2025-12-01 → 2026-05-31 (6个月) | **本金:** $50,000 | **引擎:** chop_grid + trend_scalp 共享账户 | **执行:** 1min bar, MockBinanceAPI

### 参数矩阵（11 变体）

| #   | 变体                | max_dd | daily_loss | seg_dd    | 时间过滤 | 最终权益    | 收益        | MaxDD     | 停机 |
| --- | ------------------- | ------ | ---------- | --------- | -------- | ----------- | ----------- | --------- | ---- |
| 1   | **prod**            | 20%    | 6%         | 0.072     | —        | **878,531** | **+1,657%** | -3.0%     | ✅    |
| 2   | dd_half             | 10%    | 6%         | 0.072     | —        | 878,531     | +1,657%     | -3.0%     | ✅    |
| 3   | dd15                | 15%    | 6%         | 0.072     | —        | 878,531     | +1,657%     | -3.0%     | ✅    |
| 4   | loss8pct            | 20%    | 8%         | 0.072     | —        | 878,531     | +1,657%     | -3.0%     | ✅    |
| 5   | loss10pct           | 20%    | 10%        | 0.072     | —        | 878,531     | +1,657%     | -3.0%     | ✅    |
| 6   | dd_half_loss8       | 10%    | 8%         | 0.072     | —        | 878,531     | +1,657%     | -3.0%     | ✅    |
| 7   | **sz_half**         | 20%    | 6%         | **0.036** | —        | **464,266** | **+829%**   | -2.0%     | ✅    |
| 8   | sz_half_dd_half     | 10%    | 6%         | 0.036     | —        | 464,266     | +829%       | -2.0%     | ✅    |
| 9   | **prod_tfilter**    | 20%    | 6%         | 0.072     | ✅        | **484,150** | **+868%**   | **-4.4%** | ✅    |
| 10  | dd_half_tfilter     | 10%    | 6%         | 0.072     | ✅        | 484,150     | +868%       | -4.4%     | ✅    |
| 11  | **sz_half_tfilter** | 20%    | 6%         | 0.036     | ✅        | **267,075** | **+434%**   | -2.8%     | ✅    |

### 结论

1. ⚠️ **Day-1 halt 是脏状态文件造成的**——之前 prod 在 `2025-12-01 10:00 UTC` 触发 20% DD 停机，是因为 `/tmp/bt_*.json` 残留了前次实验的状态。清零后所有变体均未触发 kill switch，实际 DD 仅 -3%。

2. **max_dd / daily_loss 调参无效**（#1-#6 结果完全相同）——因为实际 DD 只有 -3%，远低于任何 kill switch 阈值。

3. **半仓（seg_dd=0.036）收益减半，DD 略降**——线性缩放，没有非线性改善。DD 从 -3.0% → -2.0%，收益从 +1,657% → +829%。

4. ⚡ **时间过滤（非亚洲盘 UTC 09:00+ + 非周末）适得其反**——收益腰斩 + DD 升高：
   - prod_tfilter: +868% / -4.4% vs prod: +1,657% / -3.0%
   - 原因：过滤亚洲盘 → 策略集中在欧美盘 → 分散度下降 → 集中风险 + 错失亚洲趋势

5. **最差组合：sz_half + 时间过滤**——两个"降风险"措施叠加 → +434%，DD 反升至 -2.8%。

### 建议

- ✅ **prod 配置足够好**：+1,657%，-3% DD，无 halt
- ❌ **时间过滤不建议**：收益腰斩 + DD 升高
- ❌ **半仓不必要**：DD 只从 -3% → -2%，收益减半
- 🔧 **如需进一步降 DD**：微调 `segment_dd_target`（如 0.054），而非砍半

## 6. 已知 Bug 清单（2026-06-13 review）

| Bug                                                                    | 严重度 | 状态                                               |
| ---------------------------------------------------------------------- | ------ | -------------------------------------------------- |
| `except Exception` 裸捕获吞掉引擎 on_bar 异常                          | P0     | ✅ 已修复                                           |
| MockBinanceAPI 缺 `time_in_force` 参数                                 | P0     | ✅ 已修复                                           |
| MockBinanceAPI 缺 `cancel_algo_order`                                  | P1     | ✅ 已修复                                           |
| MockBinanceAPI 缺 `get_symbol_info` / `get_open_orders_for_sl_cleanup` | P1     | ✅ 已修复                                           |
| `/tmp/bt_*.json` 状态文件跨实验残留 → 脏启动                           | P0     | ✅ experiment_dd_matrix.py clean_state()            |
| `engs` 变量未定义（时间过滤代码块）                                    | P0     | ✅ 已修复                                           |
| PnL 匹配用浮点精度 `abs(qty - qty) < 1e-8` 可能失效                    | P1     | ⏳ 待改 `math.isclose()`                            |
| 无 PnL 对账（回测 vs mock 持仓）                                       | P2     | ⏳ 待加 audit log                                   |
| 无逐笔交易日志                                                         | P2     | ⏳ 待加 trade-level log                             |
| `_build_features` 每次加载 6.5 年全量数据                              | P2     | ⏳ 已加 `--load-preload` 缓存，可进一步优化增量加载 |

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

| variant           | max_dd | net_cap | halted               | 备注                                      |
| ----------------- | ------ | ------- | -------------------- | ----------------------------------------- |
| prod              | 20%    | 1.80    | no                   | 主要拒单：net_cap（~21k），非 max_dd      |
| max_dd_half       | 10%    | 1.80    | **yes** (2022-01-20) | 更紧 max_dd → 触发 halt；不是「避免停机」 |
| net_cap_100       | 20%    | 1.00    | no                   | 比 prod 少 2 笔（30 vs 32）               |
| fuse_daily_scaled | 20%    | 1.80    | no                   | daily 拒单 434 vs 302，收益路径不变       |

**结论：** 手写 YAML 变体 A/C 相同是字段路径 bug；砍半 max_dd 会**更早** halt，不能解决 day-1 停机。下一步应调 `daily_loss_limit` 与 net_cap 的联动（`tier_daily_scaled` 已接入，需配合 segment_dd 回放）。
