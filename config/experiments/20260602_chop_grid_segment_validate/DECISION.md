# chop_grid — market_segment 四段稳定性判决

**日期：** 2026-06-02  
**配置：** prod archetype + `calibrate_roll.default.yaml`（2h 信号 / 1min 执行）  
**universe：** BTC / ETH / SOL / BNB / XRP（5 币）

---

## 1. 四段汇总（等权 portfolio 口径）

**口径（2026-06-03）：** `return_pct` = timeline 组合 equity（canonical）；`return_pct_eq_mean` = 下表 eq-weight 列；`return_pct_pooled` = 五币 trade 直接相加。下表为 **eq-weight 时代** 结果，复跑后 timeline 终值通常接近 eq-mean，但 `max_drawdown_portfolio` 会反映真实路径。

| Segment | 窗 | return_pct (eq-weight†) | return_pct_pooled | segment_win_rate | worst_segment | portfolio_cum_dd |
|---------|-----|----------------------:|------------------:|-----------------:|--------------:|-----------------:|
| bear_2022 | 2022-01 → 2023-01 | **+3.6%** | 18.2% | 41.9% | -1.14% | -4.3% |
| bull_2023_2024 | 2023-01 → 2025-01 | **+5.2%** | 26.1% | 39.4% | -3.46% | -6.0% |
| recent_range_to_bear | 2025-01 → 2026-04 | **+2.2%** | 11.1% | 39.2% | -3.34% | -9.2% |
| **recent_6m_oos** | 2025-10 → 2026-03 | **-0.75%** | -3.7% | 37.3% | -3.34% | -5.2% |

**口径（2026-06-03）：** 见 §1。旧句「return_pct = per-symbol ÷ 5」已 supersede 为 timeline equity。

---

## 2. 与 trend_scalp 对照

| 策略 | recent_6m_oos eq-weight | 四段是否全正 |
|------|------------------------:|-------------|
| **trend_scalp** | **+20.6%** | ✅ 是 |
| **chop_grid** | **-0.75%** | ❌ OOS 略负 |

chop_grid 同样存在 **pooled 指标夸大 ~5×** 的问题；修正后 OOS 接近 flat/略负，**不能**用 pooled +18% 误判为稳定盈利。

---

## 3. 判决

- **return 口径修正后**：chop_grid 四段 eq-weight 幅度小（-0.8% ~ +5.2%），无 trend_scalp 式「过好」假象。
- **OOS 门禁**：recent_6m 略负 → **不满足 promote**；需继续调参或接受 chop 段在当前 prod 阈值下 edge 很薄。
- **风险**：worst_segment ~-3.3%、portfolio_cum_dd 近 -9%（recent_range）— 与 trend_scalp（~2% / ~5%）相比尾部更宽。

产物：`results/chop_grid/experiments/segment_validate_20260602/`

```bash
python scripts/experiment_chop_grid_market_segment.py \
  --out-root results/chop_grid/experiments/segment_validate_20260602 \
  -- \
  --config config/strategies/chop_grid/research/calibrate_roll.default.yaml \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --timeframe 2h --execution-timeframe 1min --no-maps
```
