# T5 BTC/ETH Phase 1 — IC + label scan → TPC 子样本

**状态**：进行中（2026-06-15）

## 目标

1. BTC + ETH 上验证 T5β（清算代理）+ T5α（订单墙）特征的 **IC / label lift**
2. TPC 子样本（bull/bear pullback）上跑 **S5 条件增益**

## Phase 0

```bash
mlbot train final --no-docker --prepare-only \
  -c config/strategies/tpc -t 240T \
  --symbol BTCUSDT,ETHUSDT \
  --start-date 2023-01-01 --end-date 2026-06-01 \
  --labels config/strategies/tpc/labels_rr_extreme.yaml \
  --features config/strategies/tpc/features_t5_scan.yaml \
  --output-root results/train_final/tpc/t5_btc_eth_v2
```

产物：`results/train_final/tpc/t5_btc_eth_v2/tpc/features_labeled.parquet`（v2 补 `ema_1200_value_f`）

## Phase 1b（S5 重跑）

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260615_t5_btc_eth_scan/rd_loop_t5_phase1b_s5.yaml
```

输出：`config/experiments/20260615_t5_btc_eth_scan/quick_scan/*.md`

## 数据前提

- OI/Funding：`data/open_interest/parquet`, `data/funding_rate/parquet`
- bookDepth：`data/book_depth/parquet`（Vision `download-book-depth`）
