# TPC S50 × BPC PCM + 深回踩杠杆实验

## 背景与判断

**S50（`tpc_pullback_depth > 0.5`）** 在 segment grid 与全窗 trading map 上 R 明显低于 E0/E2，但 **maxDD 显著更小**（约 -5%～-7% vs -10%～-14%）。若目标是「趋势账户里用低回撤腿换更高仓位」，S50 值得单独做 PCM 分工实验，而不是用 segment R 一票否决。

### 实验 A：S50 TPC + BPC PCM 分工

**假设**：BPC 覆盖浅突破/延续腿，S50 覆盖深回踩腿；PCM `max_trend_slots_per_symbol: 1` 保证同币不叠仓。TPC 回撤小 → 宪法层可提高 `tpc.max_risk_per_trade` 或压低 `bpc`，观察组合 R/DD 是否优于 prod `bpc+tpc`。

**风险**：S50 信号稀少（全窗 ~60 笔），组合收益可能仍由 BPC 主导；TPC 提权后若与 BPC 同 bar 竞争，需看 PCM 优先级（`enabled_archetypes` 中 tpc 在前）。

### 实验 B：深回踩 3× 风险暴露（Phase-1）

**你的想法**：大牛市中每次深回撤（≈20%？）加仓到 3× 杠杆，滚动复利 beta。

**Phase-1 可立刻测的近似**：

- S50 已用 `depth > 0.5` 筛深回踩；在 `execution.yaml` 加 `regime_execution.buckets.default.size_multiplier: 3.0`（无需改代码，`srb_regime_bucket` 缺失时走 `default`）。
- 这是「**每笔 S50 入场 3× 仓位**」，还不是「浮盈滚仓、持续维持账户级 3× gross」。

**尚未实现（Phase-2）**：

- `tpc_pullback_depth` 是 **0–1 语义分数**，不等于价格回撤 20%；需用特征分布实证对齐后再做动态分档杠杆。
- 「浮盈滚仓、账户恒定 3× gross」需要加仓 ladder + `max_gross_leverage` 协同，或按 depth 分桶的 `regime_execution`（需扩展 bucket 特征源）。

## 跑法

```bash
# 1) 生成策略树 + constitution 覆盖
python scripts/research/prepare_tpc_s50_pcm_leverage_experiments.py

# 2) PCM 联合回测（bpc,tpc）
bash config/experiments/20260607_tpc_s50_pcm_leverage/run_pcm_grid.sh

# 3) S50 杠杆 solo / PCM 3x
bash config/experiments/20260607_tpc_s50_pcm_leverage/run_leverage_grid.sh
```

## 变体

| ID | 策略 | strategies_root | constitution | 说明 |
|----|------|-----------------|--------------|------|
| `pcm_prod_baseline` | bpc,tpc | `config/strategies` | equal | prod 对照 |
| `pcm_s50_equal` | bpc,tpc | `tpc_s50_bpc_pcm_strategies` | equal | S50 + BPC 等权 |
| `pcm_s50_tpc_heavy` | bpc,tpc | `tpc_s50_bpc_pcm_strategies` | tpc_heavy | TPC 1% / BPC 0.5% |
| `pcm_s50_bpc_heavy` | bpc,tpc | `tpc_s50_bpc_pcm_strategies` | bpc_heavy | 反向对照 |
| `s50_solo_1x` | tpc | `tpc_semantic_depth_gt50_strategies` | — | S50 基线 |
| `s50_solo_3x` | tpc | `tpc_semantic_depth_gt50_3x_strategies` | — | execution 3× |
| `pcm_s50_3x_tpc_heavy` | bpc,tpc | `tpc_s50_bpc_pcm_3x_strategies` | tpc_heavy | A+B 组合 |

窗口：2022-01-01 → 2026-04-01，6 highcap。

## DECISION（待填）

跑完后用 `scripts/research/summarize_entry_semantic_grid.py` 或手动读各目录 `capital_report.json` 填表。
