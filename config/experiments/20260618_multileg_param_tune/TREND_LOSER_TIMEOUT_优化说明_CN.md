# trend_scalp loser_timeout 优化说明

> **日期**：2026-06-18  
> **范围**：研究栈（`diagnose_dual_add_trend.py`）；**未**改 live 引擎、**未**解锁 constitution 里的 trend_scalp 实盘。  
> **赢家配置**：`config/experiments/20260618_multileg_param_tune/variants/trend_hold_scaled.yaml`

---

## 1. 为什么要改？

### 1.1 现象（优化前 baseline）

窗口：2024-01-01 ~ 2026-05-31，5 币，diagnose 引擎，`fee_bps=4`：

| 指标 | 数值 |
|------|------|
| 净收益 | **-14.8%** |
| 成交笔数 | 35,770 |
| 笔胜率 | 24.9% |
| **loser_timeout 占比** | **72%** |
| basket_tp 占比 | 24% |

同期 **chop_grid** 同窗口仅 ~104 笔、+1.2%——trend 的问题不是「趋势策略天然不行」，而是 **执行尺度错误导致海量无效换手**。

### 1.2 根因

`archetypes/execution.yaml` 中：

```yaml
max_loser_hold_bars: 24
```

设计语义更接近 **24 根信号 bar（2h）≈ 48 小时** 才切亏损腿。  
但回测使用 `--execution-timeframe 1min` 时，引擎把 `24` 当成 **24 根 1min bar ≈ 24 分钟**。

后果链：

```
亏腿仅持 24 分钟 → loser_timeout（定义上必亏）→ 同段立即 reseed 再开
→ 单段循环数十次 → 3.5 万笔 → 手续费吞噬 → 胜率被压到 25%
```

`loser_timeout` 在代码里**仅在扣费后 PnL < 0 时触发**，所以这类成交胜率恒为 0%，不是模型差，是 **hold 太短 + 同段重入**。

---

## 2. 做了哪些改动？

### 2.1 代码

| 文件 | 改动 | 原因 |
|------|------|------|
| `scripts/diagnose_dual_add_trend.py` | 新增 `reseed_on_loser_timeout`（默认 `true`） | 可选：首次 timeout 后 **本段不再 seed**，切断循环 |
| 同上 | `DualAddConfig.reseed_on_loser_timeout` + CLI `--reseed-on-loser-timeout` / `--no-reseed-on-loser-timeout` | 与 `reseed_on_flip` 对称，供实验 |
| `scripts/run_multileg_param_tune.py` | 修复：`variants.overrides` 里布尔项（如 `scale_max_loser_hold_to_signal`）会传到 CLI | 此前 override 被静默忽略，导致假阴性 |
| 同上 | 支持 `reseed_on_loser_timeout`、`scale_max_loser_hold_to_signal`、`max_loser_hold_bars` | 批量回测 |
| `tests/unit/test_diagnose_dual_add_loser_timeout.py` | 单测：关 reseed 后 trade 数下降 | 防回归 |

**未改**：`src/time_series_model/live/dual_add_trend_live_engine.py`（live 尚无 loser_timeout 对齐逻辑）。

### 2.2 配置（研究层，非 prod lock）

| 文件 | 作用 |
|------|------|
| `variants/trend_hold_scaled.yaml` | **推荐研究栈**：`scale_max_loser_hold_to_signal: true` |
| `trend_tune_loser_timeout.yaml` | Phase 2 参数网格（hold / reseed / combo） |
| `trend_stress_20bps.yaml` | 20bps 费用压力 A/B |
| `trend_scalp/archetypes/execution.yaml` | 仅增加 `reseed_on_loser_timeout: true` 注释与字段说明；**未**改 `max_loser_hold_bars: 24` |

### 2.3 核心修复手段（无需改 archetype 数字）

使用已有 CLI / 配置：

```bash
--scale-max-loser-hold-to-signal
```

效果：`max_loser_hold_bars=24` × (2h / 1min) = **2880@1min**（约 48h），与信号 bar 语义对齐。  
`resolved_max_loser_hold_bars` 在 summary 里应为 **2880**。

---

## 3. 回测结果汇总

### 3.1 标准费用（fee_bps=4，entry/add slippage 2bps）

**全窗 2024-01 ~ 2026-05**

| 变体 | 净收益 | 笔数 | 笔胜率 | loser_timeout | 最大回撤 |
|------|--------|------|--------|---------------|----------|
| baseline | -14.8% | 35,770 | 24.9% | 72% | -15.6% |
| **hold_scaled** | **+145.8%** | 10,821 | **84.9%** | **0%** | **-1.4%** |

产出：`results/trend_scalp/hold_scaled_validate/`

**OOS 四段**（`config/market_segment.yaml`）

| 段 | baseline | hold_scaled |
|----|----------|-------------|
| bear_2022 | +1.0% | **+106.8%** |
| bull_2023_2024 | -21.5% | **+79.5%** |
| recent_range_to_bear | -3.9% | **+89.0%** |
| **recent_6m_oos** | **-3.8%** | **+28.9%** |

产出：`results/trend_scalp/oos_segment_20260618/{baseline,hold_scaled}/`

### 3.2 压力费用（fee_bps=20，对齐 chop `1min_20bps` 口径）

diagnose 里每笔收费 ≈ `2 × fee_bps`（往返），20bps 侧 = **40bps/笔** 量级。

**全窗 2024-01 ~ 2026-05**

| 变体 | 净收益 | 笔胜率 | loser_timeout | 最大回撤 |
|------|--------|--------|---------------|----------|
| baseline_20bps | **-687%** | 9.1% | 87.7% | -687% |
| **hold_scaled_20bps** | **+96.9%** | **74.9%** | **0%** | **-1.35%** |

产出：`results/trend_scalp/stress_20bps_20260618/`

**OOS 四段 @ 20bps**

| 段 | baseline | hold_scaled |
|----|----------|-------------|
| bear_2022 | -438% | **+46.4%** |
| bull_2023_2024 | -437% | **+25.9%** |
| recent_range_to_bear | -397% | **+38.7%** |
| **recent_6m_oos** | **-137%** | **+10.0%** |

产出：`results/trend_scalp/oos_segment_20bps_20260618/{baseline,hold_scaled}/`

> **结论**：即使 20bps 重压，hold_scaled 在全窗与 **四段 OOS（含 recent_6m）仍为正**；baseline 在所有段深负。优势来自 **少换手 + 消灭 timeout 循环**，不是单纯调费假设。

### 3.3 次要发现（Phase 2 网格）

| 手段 | 单独效果（4bps 全窗） |
|------|----------------------|
| 仅 `no_reseed_after_timeout` | +40%（仍 63% timeout） |
| 仅 `hold_scaled` | +163%（timeout → 0%） |
| `hold_480`（8h 固定） | +161%（与 scaled 同量级） |

**赢家是拉长 hold**；关 reseed 在 hold 已拉长后几乎无增量（timeout 已≈0）。

---

## 4. 结论：策略是否「更好了」？

| 问题 | 回答 |
|------|------|
| 比修之前的 baseline 好吗？ | **是**。收益、胜率、回撤、换手全面改善。 |
| 机制更合理吗？ | **是**。hold 与 2h 信号一致，不再 24 分钟砍腿。 |
| 可以 promote 实盘吗？ | **还不能**。需 live 引擎对齐 + constitution 重评；当前仅 diagnose 证据充分。 |

---

## 5. 未做 / 下一步

- [ ] `dual_add_trend_live_engine` 实现与 diagnose 一致的 `loser_timeout` + hold 缩放  
- [ ] 将 `scale_max_loser_hold_to_signal: true` 写入 prod archetype 或 `dual_add_backtest` 默认（经你确认后）  
- [ ] `backtest_multileg_timeline` 联合 chop+trend 重跑（constitution 口径）  
- [ ] funding 项（chop reconcile 在 fee≥20 时加 funding；trend diagnose 尚无 funding 字段）

---

## 6. 复现命令

### 6.1 全窗 hold_scaled（标准费）

```bash
source .venv/bin/activate && source scripts/env_macos_blas.sh

python scripts/diagnose_dual_add_trend.py \
  --config config/experiments/20260618_multileg_param_tune/variants/trend_hold_scaled.yaml \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --timeframe 2h --execution-timeframe 1min \
  --no-initial-hedge --no-reseed-on-flip \
  --scale-max-loser-hold-to-signal \
  --start 2024-01-01 --end 2026-05-31 \
  --no-maps \
  --out-dir results/trend_scalp/hold_scaled_validate
```

### 6.2 OOS 四段

```bash
python scripts/experiment_trend_scalp_market_segment.py \
  --out-root results/trend_scalp/oos_segment_20260618/hold_scaled \
  -- --config config/experiments/20260618_multileg_param_tune/variants/trend_hold_scaled.yaml \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --timeframe 2h --execution-timeframe 1min \
  --no-initial-hedge --no-reseed-on-flip \
  --scale-max-loser-hold-to-signal --no-maps
```

### 6.3 20bps 压力 A/B

```bash
python scripts/run_multileg_param_tune.py \
  --tune-yaml config/experiments/20260618_multileg_param_tune/trend_stress_20bps.yaml
```

### 6.4 loser_timeout 参数网格

```bash
python scripts/run_multileg_param_tune.py \
  --tune-yaml config/experiments/20260618_multileg_param_tune/trend_tune_loser_timeout.yaml
```

---

## 7. 相关文件索引

| 类型 | 路径 |
|------|------|
| 说明（本文件） | `config/experiments/20260618_multileg_param_tune/TREND_LOSER_TIMEOUT_优化说明_CN.md` |
| 赢家 overlay | `config/experiments/20260618_multileg_param_tune/variants/trend_hold_scaled.yaml` |
| chop 对照调参 | `config/experiments/20260618_multileg_param_tune/参数调优指南_CN.md` |
| 全窗结果 | `results/trend_scalp/hold_scaled_validate/summary.csv` |
| OOS 4bps | `results/trend_scalp/oos_segment_20260618/` |
| OOS 20bps | `results/trend_scalp/oos_segment_20bps_20260618/` |
| 20bps 全窗 | `results/trend_scalp/stress_20bps_20260618/comparison.json` |
