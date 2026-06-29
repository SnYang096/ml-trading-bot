# Multileg 参数调优指南（chop_grid + trend_scalp）

> **用途**：列出两个 C 层策略的核心旋钮、调参优先级，以及如何用本目录 YAML 批量回测验证。  
> **权威 prod 栈**：`config/strategies/{chop_grid,trend_scalp}/archetypes/*.yaml`  
> **回测引擎**：chop → `scripts/chop_grid_backtest.py`；trend → `scripts/diagnose_dual_add_trend.py`  
> **勿用** `backtest_multileg_timeline` 的复利 +7192% 数字做调参判据。

---

## 1. 怎么改参数、怎么跑

### 1.1 可编辑配置（推荐）

| 文件 | 作用 |
|------|------|
| [`chop_tune.yaml`](chop_tune.yaml) | chop_grid 回测窗口 + 变体列表 |
| [`trend_tune.yaml`](trend_tune.yaml) | trend_scalp 回测窗口 + 变体列表 |

在对应 YAML 的 `variants:` 里 **增删改** `overrides` 即可；`id` 会用作输出子目录名。

### 1.2 一键批量回测

```bash
cd /Users/jerry/project/yin/ml-trading-bot
source .venv/bin/activate && source scripts/env_macos_blas.sh

# chop — 跑 chop_tune.yaml 里全部变体
python scripts/run_multileg_param_tune.py \
  --tune-yaml config/experiments/20260618_multileg_param_tune/chop_tune.yaml

# trend
python scripts/run_multileg_param_tune.py \
  --tune-yaml config/experiments/20260618_multileg_param_tune/trend_tune.yaml

# 只跑部分变体
python scripts/run_multileg_param_tune.py \
  --tune-yaml config/experiments/20260618_multileg_param_tune/chop_tune.yaml \
  --variants baseline,no_replenish,exit_038

# 预览命令、不执行
python scripts/run_multileg_param_tune.py \
  --tune-yaml config/experiments/20260618_multileg_param_tune/chop_tune.yaml \
  --dry-run
```

结果：

- 每变体：`{output_root}/{id}/`（trades、segments、capital_report.html）
- 汇总：`{output_root}/comparison.json` + `comparison.csv`

### 1.3 单变体手动回测（与 YAML 等价）

```bash
# chop baseline（live 费率）
python scripts/chop_grid_backtest.py \
  --config config/strategies/chop_grid/meta.yaml \
  --start 2024-01-01 --end 2026-05-31 \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --maker-fee-bps 2 --taker-fee-bps 5 \
  --no-maps \
  --out-dir results/chop_grid/my_candidate

# trend baseline（对齐 archetype：单腿 TREND + 不 flip reseed）
python scripts/diagnose_dual_add_trend.py \
  --config config/strategies/trend_scalp/meta.yaml \
  --start 2024-01-01 --end 2026-05-31 \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --execution-timeframe 1min \
  --no-initial-hedge --no-reseed-on-flip \
  --fee-bps 4 \
  --no-maps \
  --out-dir results/trend_scalp/my_candidate
```

---

## 2. chop_grid — 参数与优先级

### 2.1 三层结构

| 层 | 文件 | 核心键 |
|----|------|--------|
| Regime | `archetypes/regime.yaml` | `entry_min`, `exit_below`, `exclude_box_prefilter` |
| Prefilter | `archetypes/prefilter.yaml` | `box_pos_60` 带 [0.40, 0.60] |
| Execution | `archetypes/execution.yaml` | spacing / levels / replenish / TP / 段风险 |

### 2.2 调参优先级（盯 forced_rate + return）

| 优先级 | 目标 | YAML `overrides` 键 | prod 默认 | 方向 |
|--------|------|----------------------|-----------|------|
| **P0** | 降 forced 强平 | `exit_chop_min` | 0.33 | ↑ 略早退 chop（0.35–0.38） |
| **P0** | 少假 chop 段 | `chop_min` | 0.52 | ↑ 更严进场（0.55–0.58） |
| **P1** | 省 turnover | `max_replenish_per_level` | 1 | → **0**（A/B 已验证省费） |
| **P1** | 少 cascade | `max_levels` | 3 | → 2 |
| **P1** | 拉大每格利润 | `grid_pct` | 0.0033 | ↑ 0.004–0.005 |
| **P2** | TP 距离 | `tp_spacing_mult` | 2 | 勿轻易 ↓ 到 1（增 turn） |
| **P2** | 段内止损 | `max_loss_per_grid` | 0.03 | ↓ 0.02 试减尾部 |
| **P3** | 形态带 | `box_pos_min`, `box_pos_max` | 0.40 / 0.60 | 分币微调 |
| **P3** | chop 信号 | `chop_signal` | raw | ts_quantile（横截面不稳时） |

### 2.3 判据（不要只看总 return）

| 指标 | 来源 | 说明 |
|------|------|------|
| `return_pct_timeline` | `metrics.json` | 组合净收益 % |
| `forced_rate` | `metrics.json` | **~40% 是主要敌人**，目标 <25% |
| `n_segments` / `n_trades` | 同上 | coverage；段太少说明 regime 过严 |
| `forced_exit_pnl` | cost attribution | forced 对 PnL 的拖累 |
| 分月 / 分币 | `report.html`, `capital_report.html` | ETH/BNB forced 常更差 |

---

## 3. trend_scalp — 参数与优先级

### 3.1 三层结构

| 层 | 文件 | 核心键 |
|----|------|--------|
| Regime | `archetypes/regime.yaml` | `entry_min`, `exit_below`, `cap_entry`, `cap_hold` |
| Prefilter | `archetypes/prefilter.yaml` | 空（全靠 regime） |
| Execution | `archetypes/execution.yaml` | 库存 / add_spacing / basket TP / order_model / risk |

### 3.2 调参优先级

| 优先级 | 目标 | YAML `overrides` 键 | prod 默认 | 方向 |
|--------|------|----------------------|-----------|------|
| **P0** | 少在 chop 里打 | `exit_chop_min`（= cap_entry） | 0.25 | ↓ 0.20 更严 |
| **P0** | 持仓 chop 上限 | `chop_min`（= cap_hold） | 0.40 | ↓ 0.35 |
| **P0** | 趋势质量 | `trend_min` | 0.70 | ↑ 0.75–0.80 少段、提质 |
| **P1** | 加仓间距 | `step_atr_mult` | 0.75 | ↑ 少 add、省费 |
| **P1** | 篮子止盈 | `tp_pct`, `tp_atr_mult` | 0.12% / 0.6 | 略放宽需 fee-aware |
| **P1** | 翻转换手 | `reseed_on_flip` | **false** | 保持 false；true 增 turn |
| **P2** | 暴露上限 | `max_adds_per_side`, `max_net_exposure` | 3 / 2 | ↓ 降 DD |
| **P2** | 段级风险 | `max_loss_per_segment`, `risk_stop_mode` | 0.02 / regime_only | mtm 更紧但易误杀 |
| **P3** | 滑点假设 | `entry_slippage_bps`, `add_slippage_bps` | 2 / 2 | 回测敏感性 |
| **P3** | 初始腿 | `initial_hedge` | false（TREND） | 勿开双开对冲（费+尾） |

### 3.3 判据

| 指标 | 来源 | 说明 |
|------|------|------|
| `portfolio_pnl_per_capital_timeline` | `summary.csv` | 组合 R |
| `forced_rate` | `summary.csv` | 非 basket_tp 出场占比 |
| `risk_stop_rate` | `summary.csv` | 段内 MTM 止损触发（regime_only 下应≈0） |
| `n_segments`, `n_trades` | `summary.csv` | 活跃度 |
| `max_drawdown_r` | `summary.csv` | live 曾 DD>20% halt，优先控 DD |

---

## 4. overrides 键 → CLI 对照（写 YAML 时用）

### chop_grid

| overrides 键 | CLI |
|--------------|-----|
| `chop_min` | `--chop-min` |
| `exit_chop_min` | `--exit-chop-min` |
| `grid_pct` | `--grid-pct` |
| `grid_atr_mult` | `--grid-atr-mult` |
| `max_levels` | `--max-levels` |
| `max_replenish_per_level` | `--max-replenish-per-level`（0 / 1 / null） |
| `tp_spacing_mult` | `--tp-spacing-mult` |
| `max_loss_per_grid` | `--max-loss-per-grid` |
| `min_segment_bars` | `--min-segment-bars` |
| `max_segment_bars` | `--max-segment-bars` |
| `box_pos_min`, `box_pos_max` | `--box-pos-min`, `--box-pos-max` |
| `chop_signal` | `--chop-signal` |

### trend_scalp

| overrides 键 | CLI |
|--------------|-----|
| `trend_min` | `--trend-min` |
| `trend_exit_min` | `--trend-exit-min` |
| `chop_min` | `--chop-min`（持仓 chop 上限） |
| `exit_chop_min` | `--exit-chop-min`（进场 chop 上限） |
| `step_atr_mult` | `--step-atr-mult` |
| `tp_atr_mult` | `--tp-atr-mult` |
| `tp_pct` | `--tp-pct` |
| `max_adds_per_side` | `--max-adds-per-side` |
| `max_net_exposure` | `--max-net-exposure` |
| `max_gross_exposure` | `--max-gross-exposure` |
| `max_loss_per_segment` | `--max-loss-per-segment` |
| `risk_stop_mode` | `--risk-stop-mode` |
| `reseed_on_flip` | `--reseed-on-flip` / `--no-reseed-on-flip` |
| `initial_hedge` | `--initial-hedge` / `--no-initial-hedge` |
| `flip_action` | `--flip-action` |

---

## 5. 示例：新增一个变体

编辑 `chop_tune.yaml`：

```yaml
variants:
  - id: my_combo
    description: 更严进场 + 关补挂
    overrides:
      chop_min: 0.55
      max_replenish_per_level: 0
```

然后：

```bash
python scripts/run_multileg_param_tune.py \
  --tune-yaml config/experiments/20260618_multileg_param_tune/chop_tune.yaml \
  --variants my_combo
```

---

## 6. 相关文件

| 路径 | 说明 |
|------|------|
| `config/strategies/chop_grid/调优与手续费_CN.md` | chop 手续费专项 |
| `config/strategies/trend_scalp/README.md` | trend 策略说明 |
| `scripts/sweep_chop_regime_thresholds.py` | chop regime 网格 sweep |
| `scripts/sweep_chop_grid_levels.py` | chop 层数/spacing sweep |
| `scripts/run_multileg_backtest_with_maps.sh` | 带地图的一键回测 |
