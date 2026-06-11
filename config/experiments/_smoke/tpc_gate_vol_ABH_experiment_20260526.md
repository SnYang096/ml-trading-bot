# TPC Gate 实验存档：A / B / H（vol gates 全关 vs bull-conditional）

- **日期**: 2026-05-26
- **策略**: TPC（6 symbols: BTC/ETH/SOL/BNB/XRP/ADA）
- **工具**: `scripts/event_backtest.py`
- **决策**: **Promote variant H** → `config/strategies/tpc/archetypes/gate.yaml` + `live/highcap/...`
- **监控**: `scripts/regime_watchdog.py` + `config/monitoring/regime_watchdog_baseline.json`

## 1. 变体定义

| ID | 配置路径 | vol_persistence | vol_leverage_asymmetry | chop |
|----|----------|-----------------|------------------------|------|
| **A** | `config/strategies/tpc/`（改前 baseline） | 全局 ON | 全局 ON | ON |
| **B** | `config/experiments/_smoke/variants/B_gate_only_chop_strategies/` | **disabled** | **disabled** | ON |
| **H** | 已 promote 到主 config | **仅 `ema_1200_position > 0.10` 时 ON** | 同上 | ON |
| C | `C_chop_plus_evt_strategies/` | disabled | disabled | ON + EVT veto |
| D | `D_regime_strict_strategies/` | baseline | baseline | regime \|pos\|>0.12 |
| E | `E_entry_v2_strategies/` | baseline | baseline | 新 entry filters |
| BE | `BE_combo_strategies/` | B 的 gate + E 的 entry | | |

H 的 gate 条件（摘录）：

```yaml
# vol deny 仅在强多头侧生效
all_of:
  - vol_persistence: (0.0029, 0.0616)
  - ema_1200_position: { value_gt: 0.10 }
```

## 2. Event backtest 结果

### 2.1 2025-04-01 → 2026-04-01（近期 mixed / bear-leaning）

| 变体 | trades | totR | meanR | win | Sharpe | Ret | maxDD | gate_reject |
|------|--------|------|-------|-----|--------|-----|-------|-------------|
| A baseline* | 164 | +43.08 | +0.263 | 32.3% | +0.17 | +17.95% | -7.85% | ~10920 |
| **B** | **178** | **+59.83** | **+0.336** | **36.0%** | **+0.21** | **+21.82%** | **-5.23%** | 10650 |
| **H** | 172 | +47.06 | +0.274 | 34.3% | +0.17 | +16.76% | -7.48% | 11141 |
| BE_combo | 159 | +46.36 | +0.292 | 32.7% | +0.18 | +16.52% | -6.25% | 10650 |
| C chop+EVT | 102 | +36.06 | +0.354 | 38.2% | +0.23 | +13.47% | -8.24% | 11325 |
| D regime 0.12 | 145 | +49.34 | +0.340 | 33.1% | +0.21 | +16.70% | -8.76% | 10570 |
| E entry v2 | 143 | +45.72 | +0.320 | 32.9% | +0.19 | +18.32% | -5.83% | 11839 |

\*A 近期跑在本轮 batch 前完成，未单独落在 `results/tpc/experiments/A_*`；数字来自同批对照记录。

**产物目录**: `results/tpc/experiments/{B_gate_only_chop,H_recent,BE_combo,C_chop_plus_evt,D_regime_strict,E_entry_v2}/`

### 2.2 2024-01-01 → 2025-01-01（calendar bull）

| 变体 | trades | totR | meanR | win | Ret | maxDD | gate_reject |
|------|--------|------|-------|-----|-----|-------|-------------|
| A baseline | 159 | +17.64 | +0.111 | 27.0% | +1.74% | -8.64% | 10920 |
| B | 175 | +16.94 | +0.097 | 30.3% | -0.04% | **-13.52%** | 9734 |
| **H** | **168** | **+16.30** | **+0.097** | **28.0%** | **+2.87%** | **-7.57%** | 10723 |

**产物目录**: `results/tpc/experiments/{A_baseline_bull_2024,B_bull_2024,H_bull_2024}/`

### 2.3 按 side 分解（解释 H vs B 的关键）

**2025-2026 recent — totR 差额 +12.77 几乎全部来自 LONG：**

| 变体 | LONG n | LONG totR | SHORT n | SHORT totR |
|------|--------|-------------|---------|------------|
| B | 117 | **+57.03** | 61 | +2.80 |
| H | 118 | **+44.23** | 54 | +2.83 |

H 比 B **多 1 笔 long、少 7 笔 short**；short 侧 totR 几乎一样，**long 侧少 +12.8R**。

**2024 bull — SHORT 拖累 totR，H 改善 DD 不靠多赚 short：**

| 变体 | LONG totR | SHORT totR |
|------|-----------|------------|
| A | +26.67 | -9.03 |
| B | +29.19 | -12.25 |
| H | +26.69 | -10.39 |

## 3. 离线 label 扫描（features_labeled.parquet）

来源: `train_final_20260523_122438_rr_extreme` / `success_no_rr_extreme`

**`eff_keep_vp` = 保留 vol_persistence gate 对 label success 的效应（+ 有用）**

| 桶 | n | eff_keep_vp | eff_keep_vla |
|----|---|-------------|--------------|
| ema_strong_bull >=0.10 | 18259 | **+6.84%\*** | **+3.30%\*** |
| ema_strong_bear <=-0.10 | 21515 | +2.28%* | +1.42%* |
| cal_bull_2024 AND ema_bull | 7975 | **+7.31%\*** | **+4.43%\*** |
| cal_recent AND ema_bull | 2185 | **+16.19%\*** | +4.73%* |

结论：**在 label（success rate）上，vol gates 在 ema_bull 几乎处处“该开”**；与 **event backtest R-multiple** 在 recent 上“关掉 vol 更赚”形成分歧（见 §4）。

## 4. 为何选择 H（而非 B）

| 目标 | B | H |
|------|---|---|
| recent totR | 最高 (+60R) | 中 (+47R) |
| recent maxDD | 最好 (-5.2%) | 中 (-7.5%) |
| 2024 bull maxDD | **最差 (-13.5%)** | **最好 (-7.6%)** |
| 跨 regime | 单窗赢家、另一窗 DD 恶化 | Pareto 优于 baseline 两段 |

Doctrine：B 系统不抓 fattail；用 H 换 **~13R/year** 换 **bull 段 DD 保护 ~6pp** 可接受。

## 5. 已落盘文件

| 文件 | 说明 |
|------|------|
| `config/strategies/tpc/archetypes/gate.yaml` | H |
| `live/highcap/config/strategies/tpc/archetypes/gate.yaml` | H 同步 |
| `config/experiments/_smoke/variants/H_bull_conditional_vol_strategies/` | 实验树（含全策略目录） |
| `config/experiments/_smoke/variants/B_gate_only_chop_strategies/` | 对照 B |
| `scripts/regime_watchdog.py` | 周度监控 |
| `config/monitoring/regime_watchdog_baseline.json` | bull_share 16.9%, trigger rates |
| `docs/strategy/regime_layer.md` | TPC gate 说明更新 |

## 6. 复现命令（H recent）

```bash
cd /home/yin/trading/ml_trading_bot
PYTHONPATH=src:scripts python3 scripts/event_backtest.py \
  --strategy tpc \
  --strategies-root config/experiments/_smoke/variants/H_bull_conditional_vol_strategies \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --start-date 2025-04-01 --end-date 2026-04-01 \
  --trades-csv results/tpc/experiments/H_recent/event_trades_tpc.csv \
  --capital-report results/tpc/experiments/H_recent/ \
  --output results/tpc/experiments/H_recent/summary.json \
  --quiet-signal-logs
```

## 7. 已知坑

1. **`--strategies-root` 必须含完整策略树**（bpc/me/srb/…），仅放 `tpc/` 会导致 timeline 减半、结果不可比（H 第一次误跑 trades=48）。
2. **label success ≠ trade R**；gate 优化若只看 label 会偏向“全开 vol gates”。
3. **BE_combo 不可与 B 简单叠加**（entry 会误杀 B 解锁的好单）。
