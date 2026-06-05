# 四棵 legacy 树策略 — 牛熊分段 vector RR（2026-06-04）

| 字段 | 值 |
|------|-----|
| 策略 | `compression_breakout`, `sr_breakout`, `trend_following`, `sr_reversal_rr_reg_long` |
| 配置 | `config/strategies/tree_strategies/<slug>/` |
| 分段 | `config/market_segment.yaml`：`bear_2022` / `bull_2023_2024` / `recent_range_to_bear` |
| 产物 | `results/rd_loop/tree_legacy_bull_bear_20260604/` |

## 跑法

```bash
export PYTHONPATH=src:scripts
bash config/experiments/20260604_tree_legacy_bull_bear/run_segment_matrix.sh
# 日志: /tmp/tree_legacy_bull_bear.log
```

- 已有 artifact：`compression_breakout` / `sr_breakout` → `train_final_20260530_124749_btceth`
- 无 artifact：`trend_following` / `sr_reversal_rr_reg_long` → 先 `train_strategy_pipeline` 再分段扫描
- 入口：固定 **q=0.10**（回归类 top/bottom 10%）；multiclass 用 long-class proba 作分位数元数据
- 汇总：`python scripts/research/summarize_tree_legacy_segment_matrix.py`

## 口径

Vectorbt RR 回测（非 event PCM），BTC+ETH pooled **均值** Sharpe/收益；与 fast_scalp event OOS **不可直接对比**。
