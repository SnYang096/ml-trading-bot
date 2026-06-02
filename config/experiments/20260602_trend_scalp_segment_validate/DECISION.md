# trend_scalp — market_segment 四段稳定性判决

**日期：** 2026-06-02  
**配置：** prod archetype + `calibrate_roll.default.yaml`（2h 信号 / 1min 执行回放）  
**universe：** BTC / ETH / SOL / BNB / XRP（5 币，与 `meta.yaml` 一致）  
**产物：** `results/trend_scalp/experiments/segment_validate_20260602/`

---

## 1. 四段 pooled 汇总

> **口径（2026-06-02 更新）：** `return_pct` = 等权 multileg 组合（每币一个 capital bucket，取 per-symbol 累加后 **÷ n_symbols**）。`return_pct_pooled` 保留旧「五币 trade 直接相加」作对照。

| Segment | 窗 | segments | trades | segment_win_rate | return_pct (eq-weight) | return_pct_pooled | worst_segment | portfolio_cum_dd |
|---------|-----|---------:|-------:|-----------------:|-----------------------:|------------------:|--------------:|-----------------:|
| **bear_2022** | 2022-01 → 2023-01 | 611 | 3,115 | 72.3% | **59.4%** | 296.9 | -2.42% | -4.69% |
| **bull_2023_2024** | 2023-01 → 2025-01 | 1,199 | 5,858 | 63.9% | **59.0%** | 294.9 | -2.53% | -5.30% |
| **recent_range_to_bear** | 2025-01 → 2026-04 | 783 | 4,060 | 69.5% | **54.7%** | 273.5 | -2.08% | -5.14% |
| **recent_6m_oos** | 2025-10 → 2026-03 | 294 | 1,533 | 70.7% | **20.6%** | 102.7 | -2.08% | -5.14% |

---

## 2. 分币 breakdown（稳定性核心）

**20/20 格全部为正** — 每个 segment × 每个 symbol 的 `return_pct` 均 > 0。

| Symbol | bear_2022 | bull_2023_2024 | recent_range | recent_6m_oos |
|--------|----------:|---------------:|-------------:|--------------:|
| BTCUSDT | +31.0% | +24.6% | +17.6% | +10.7% |
| ETHUSDT | +57.7% | +47.2% | +77.3% | +28.0% |
| SOLUSDT | +115.4% | +120.3% | +83.1% | +31.5% |
| BNBUSDT | +41.9% | +20.4% | +26.0% | +7.6% |
| XRPUSDT | +50.9% | +82.6% | +69.5% | +25.0% |

- **最弱格：** BNB recent_6m_oos +7.6%（仍为正）
- **SOL 在各段领先**，BTC 在各段最稳但幅度较小 — 符合 high-beta alt vs major 预期

---

## 3. 风险指标跨段一致性

| 指标 | 跨段范围 | 解读 |
|------|----------|------|
| worst_segment | -2.08% ~ -2.53% | 单段尾部极窄，四段几乎相同 |
| portfolio_cum_dd | -4.69% ~ -5.30% | 累加曲线 maxDD 稳定 ~5% |
| risk_stop_rate | 0% | `regime_only` 下无 MTM 段内硬止损触发 |
| trade_win_rate | 72.7% ~ 75.9% | 高且稳定 |
| max_gross_units | 2 | 未触达 archetype 上限 4（TREND 单开 + 有限加仓） |

---

## 4. 与 LAYER_PROMOTION_CRITERIA 对照

| 杠 | trend_scalp prod | 结论 |
|----|------------------|------|
| 总 R / return 明显提升 | 四段 eq-weight 均 **+20% ~ +59%** | ✅ 通过 |
| maxDD 不恶化 | worst_segment ~2%、portfolio_cum_dd ~5%，跨段无恶化 | ✅ 通过 |
| 逻辑可解释 | regime 趋势段 + basket TP + flip flat — 与策略设计一致 | ✅ 通过 |

**recent_6m_oos** 作为当前 regime promote 门禁：**+20.6% eq-weight / 6mo，五币全正** — 满足 OOS 门禁。

---

## 5. 判决

**结论：prod trend_scalp 在 market_segment 四段上表现稳定。**

- 无「某段崩盘」或「某币拖垮 pooled」现象
- 尾部风险（worst_segment、portfolio_cum_dd）跨段高度一致
- bull 段 segment_win_rate 略低（64% vs 72%），但 return 仍与 bear 段同级 — 属于段数更多、胜率回归均值，非 edge 消失
- **无需改 archetype**；维持 prod 配置，监控 recent_6m 窗 rolling metrics

**注意：**

1. 回测 fee_bps=20（`calibrate_roll.default.yaml` diagnostic 假设），实盘以 exchange fill 为准
2. `return_pct` 为 trade-level capital-normalized 累加，勿与单账户 CAGR 直接对比；等权均值见 §1
3. trend_scalp 不走 event_backtest 路径；本验证用 `diagnose_dual_add_trend.py` multi-leg simulator（与 prod live engine 同底盘）

---

## 6. 复跑

---

## 7. Backtrader 独立交叉验证（2026-06-02）

**动机：** 四段 return 看起来「过好」，用 **Backtrader** 在 **1min 执行 bar** 上重写 inventory 模拟（不调用 `simulate_dual_add_segment`），复用相同 regime 段发现逻辑，核对 diagnose 引擎是否有 bug。

**脚本：** `scripts/backtest_trend_scalp_backtrader.py`

| 窗 | diagnose return_pct | backtrader return_pct | Δ rel | segments | trades Δ |
|----|--------------------:|----------------------:|------:|---------:|---------:|
| recent_6m_oos | 102.72 | 101.27 | **-1.4%** | 294 = 294 | -12 (-0.8%) |
| bear_2022 | 296.91 | 302.64 | **+1.9%** | 611 = 611 | -4 (-0.1%) |

- 段数完全一致；return 偏差 **< 2%**；单段 pnl 最大绝对差 ~0.02，均值差 ~0.0005
- **结论：原结果与 Backtrader 独立实现一致，不是 simulator 算错**

**为何 return_pct 数字看起来「过大」？**

- 指标 = 所有 trade 的 `pnl_per_capital` 之和 × 100（每笔已除以 `capital_units=4`）
- 6 个月 OOS 有 **1,500+ 笔**小赢单累加 → pooled 102% 不等于账户翻倍
- 更可比口径：**等权 per-symbol 均值** recent_6m = **20.6% / 6mo**（见 §1）

产物：

- `results/trend_scalp/experiments/backtrader_crosscheck_recent_6m/`
- `results/trend_scalp/experiments/backtrader_crosscheck_bear_2022/`

```bash
python scripts/backtest_trend_scalp_backtrader.py \
  --start 2025-10-01 --end 2026-03-31 \
  --execution-timeframe 1min --scale-max-loser-hold-to-signal \
  --take-profit-mode basket --no-reseed-on-flip --risk-stop-mode regime_only \
  --compare-dir results/trend_scalp/experiments/segment_validate_20260602/recent_6m_oos \
  --out-dir results/trend_scalp/experiments/backtrader_crosscheck_recent_6m
```
