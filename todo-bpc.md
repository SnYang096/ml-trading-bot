# bpc 完整上线流程

## 1. 已经用 failure_rr_extreme 在 only_long 标签下 找到failure first

训练 failure_rr_extreme
```bash
mlbot train final --no-docker \
  --config config/strategies/bpc \
  --labels config/strategies/bpc/labels_rr_extreme.yaml \
  --symbol BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --data-path data/parquet_data \
  --start-date 2023-01-01 \
  --end-date 2025-11-30 \
  --holdout-start-date 2024-05-01 \
  --holdout-end-date 2025-11-30 \
  --seed 42
```
[bpc_20260131_171621_report.html](http://localhost:8008/bpc_20260131_171621_report.html)

## 2. 导出核心到规则（mlbot train final 会自动导出了）
mlbot train export-rules --no-docker --model-dir models/bpc --strategy bpc --max-splits 30   --generate-risk-gate

mlbot train export-rules \
  --model-dir results/train_final_xxx/bpc \
  --strategy bpc \
  --generate-risk-gate

 /home/yin/trading/ml_trading_bot/models/bpc/bpc_tree_rules.md

## 3. 形成 risk_gate.yaml
后面可以直接copy到 config/nnmultihead/execution_archetypes.yaml 里面

## 4. 训练 failure_no_opportunity, 导出核心规则，形成risk_gate.yaml
```bash
mlbot train final --no-docker \
  --config config/strategies/bpc \
  --labels config/strategies/bpc/labels_no_opportunity.yaml \
  --symbol BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --data-path data/parquet_data \
  --start-date 2023-01-01 \
  --end-date 2025-11-30 \
  --holdout-start-date 2024-05-01 \
  --holdout-end-date 2025-11-30 \
  --seed 42
```

## 5. 分层架构里面编写gate，evidence

config/nnmultihead/execution_archetypes.yaml 里面

## 6. 训练找到 evidence 最佳参数

用平坦高原办法

## 7. oos 测试各种灭绝数据

拿到灭绝数据不会毁灭证据

## 8. oss 测试长时间数据

拿到正常sharp数据

## 9. 上线bpc