# exp05 — Target-Horizon IC + Regime-Conditional Factor Rotation

修正 exp03 IC horizon 错配问题；按 BTC 趋势 + funding 双维度做 regime-specific factor weights；对比 static vs regime-switching。

## 文件

- `regimes.py` — BTC 30d trend × funding 7d mean 打 combined regime 标签，collapse 到 5 种样本充足的主导 regime
- `run_horizon_ic.py` — 12 个候选因子 × 5 个 horizon (1d/3d/7d/14d/30d) IC
- `run_regime_ic.py` — 在 target horizon (14d) 上按 regime 切分样本计算条件 IC，生成 `regime_weights.yaml`
- `run_regime_switch_backtest.py` — 三策略对比（static mom_only / static IC-weighted / regime_switch）
- **`wf_ic_utils.py` + `run_walk_forward_oos.py`（exp05 v2）** — Walk-forward OOS：每个 rebalance 仅用 `t0` 之前已实现的收益样本拟合 IC 权重，消除全样本 look-ahead

## 关键结论

### Part 1：target horizon 很关键
- exp03 用 1d horizon 误选了 reversal（1d IR +0.11，14d IR -0.02）
- **14d horizon 下 top-3 一致是 low_vol_30d / low_vol_14d / low_vol_7d（IR 0.31-0.38）**
- momentum 属于 mid-long 因子：mom_3d 在 1d IR=-0.10，14d IR=+0.04
- funding 对多数 horizon 都正（14d IR 0.09-0.11）

### Part 2：regime-specific 最好因子
| regime | 样本 bars | 主导因子 (top IR) |
|---|---:|---|
| range_normal | 16145 | low_vol 三个 (IR 0.31-0.38) + funding (0.17-0.20) |
| bull_normal | 6889 | low_vol (0.17-0.22) + rev_24h (0.07 ← 全样本 -0.02) |
| bear | 3864 | 纯 low_vol (IR 0.66-0.72) |
| bull_momentum | 1148 | low_vol + mom 全线 IR > 0.35 |
| range_reversal | 417 | low_vol IR 达 **3.5**，mom_14d 1.58 |

### Part 3：regime switching backtest (full 2023–2026Q1)
| strategy | Net SR | AnnRet | MaxDD |
|---|---:|---:|---:|
| static_mom_only | +0.21 | +7.4% | -45.5% |
| **static_all_weights** (IC-weighted combo) | **+1.01** | **+35.4%** | -44.8% |
| regime_switch | +0.79 | +28.0% | -45.4% |

核心 takeaway:
1. **IC-weighted combo (static all) 最优** — 比 mom_only 的 Sharpe 直接翻 5 倍
2. **regime_switch 未超过 static_all** — 换因子噪声 ≈ regime alpha 收益
3. **不应上线 mom_only** — 全样本 Sharpe 仅 0.21，2025 会亏钱
4. 存在 look-ahead（权重是 full sample 学的），后续需 walk-forward OOS

## 产出目录

- `reports/cross_section/exp05_regime_ic/horizon_ic/`
- `reports/cross_section/exp05_regime_ic/regime_ic/regime_weights.yaml`
- `reports/cross_section/exp05_regime_ic/switch_backtest/`
- **`reports/cross_section/exp05_regime_ic/walk_forward/`** — `wf_weights.jsonl`、`equity_wf_oos.parquet`、`summary.md`

### exp05 v2 用法

```bash
python -m src.cross_section.exp05_regime_ic.run_walk_forward_oos \
  --start 2023-01 --end 2026-03
# 默认：训练窗 180d @1h；每步 refit；按 regime 拟合多档权重
```

常用减噪 / 降维组合：

```bash
# 更长训练窗（显式 180d，与默认相同可省略）
--train-window-bars 4320

# 更低权重重拟合频率：每 2 次 rebalance 才 fit，中间行 jsonl 为 carry_forward
--refit-every-n 2

# 更长持仓（例如 28d）：更少 rebalance、更少 fit 机会（若 refit_every_n=1）
--hold-bars 672

# 弱化 regime：只拟合全截面 ALL 一档权重（回测时各 regime 均回退到 ALL）
--all-weights-only
```

- 权重仅使用 `t0 - train_window` 至 `t0 - horizon` 的 bar 计算 IC（14d fwd 在 `t0` 前已闭合）。
- 训练窗过短或拟合失败时 jsonl 记 `skip`；回测该段回退 **mom_only**。
- 重跑拟合会覆盖 `wf_weights.jsonl`；若只改回测可加 `--skip-fit` 并保留已有 jsonl。

## 与 exp07 paper trading 的接口

`regime_weights.yaml` 的 schema：
```yaml
meta:
  horizon_bars: 336
  ic_threshold: 0.02
  period: "2023-01_to_2026-03"
  symbols: [BTCUSDT, ETHUSDT, ...]
factor_specs:
  mom_14d:   {kind: mom,      lookback: 336, skip: 0}
  low_vol_30d: {kind: low_vol, lookback: 720, skip: 0}
  ...
regime_weights:
  ALL:             {factors: {low_vol_30d: 0.24, low_vol_14d: 0.23, ...}, note: regime_conditional}
  bull_momentum:   {factors: {...}}
  bull_normal:     {factors: {...}}
  range_normal:    {factors: {...}}
  range_reversal:  {factors: {...}}
  bear:            {factors: {...}}
```

exp07 里 `--use-regime-switch` 读这个文件，每次 rebalance 查当前 regime 用对应权重。
