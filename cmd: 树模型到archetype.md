# 树模型导出规则到archetype
## 确保数据都在
```bash
mlbot data check-month-coverage --symbol BTCUSDT --start 2022-01 --end 2026-01 --zip-dir data/agg_data --parquet-dir data/parquet_data --no-docker

# 转成1min的ticks加速运算
mlbot data convert --input-dir data/agg_data --output-dir data/parquet_data --pattern 'BTCUSDT-aggTrades-2024-*.zip' 

```
## build 计算缓存，方便后面复用
```bash
mlbot feature-store build --no-docker --config config/strategies/bpc --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT --timeframe 240T --data-path data/parquet_data --start-date 2023-01-01 --end-date 2024-12-31 --root feature_store 
```
## 可视化特征，防止特征计算有问题，看看有没有空的 
```bash
mlbot visualize feature-indicators --no-docker \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2023-01-01 \
  --end-date 2025-12-31 \
  --strategy-config config/strategies/bpc \
  --use-cache
# --force-rebuild 只删除 FeatureStore 缓存，但不删除 monthly 特征缓存（cache/features/monthly/）。
mlbot visualize feature-indicators --no-docker \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2023-01-01 \
  --end-date 2025-12-31 \
  --strategy-config config/strategies/bpc \
  --force-rebuild

# 如果数据不对，可以删除以下几层缓存
echo "清理 2024 年所有相关缓存（vpin, cvd, ofci, bb_width）..." && find /home/yin/trading/ml_trading_bot/cache/features/monthly -name "*2024*" \( -name "*vpin*" -o -name "*cvd*" -o -name "*ofci*" -o -name "*bb_width*" \) -type f -delete && echo "✅ 完成"

echo "🗑️ 清理所有 2024 年缓存..." && find /home/yin/trading/ml_trading_bot/cache -name "*2024*" -type f -delete 2>/dev/null && echo "✅ 完成" && find /home/yin/trading/ml_trading_bot/cache -name "*2024*" -type f 2>/dev/null | wc -l

/home/yin/trading/ml_trading_bot/cache/timeframes/BTCUSDT_240T.parquet && echo "✅ 已删除 BTCUSDT_240T 缓存"

┌─────────────────────────────────────────────────────────────────────────┐
│                           数据流向（从原始数据到特征）                    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 0: 原始数据                                                       │
│  📁 data/parquet_data/                                                  │
│     └── BTCUSDT_2024-*.parquet (tick 聚合数据: timestamp,price,volume,side)│
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 1: Timeframe 缓存 (K线重采样)                            [14M]   │
│  📁 cache/timeframes/                                                   │
│     └── BTCUSDT_240T.parquet (从 tick 重采样生成的 OHLCV+CVD)           │
│  ⚡ 触发: MarketDataLoader 首次加载或 raw 文件更新                       │
│  🗑️ 清理: rm cache/timeframes/BTCUSDT_*.parquet                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 2: Monthly 特征缓存 (单特征计算结果)                     [4.0G]  │
│  📁 cache/features/monthly/                                             │
│     ├── bpc_soft_phase_f_2024-01_*.pkl                                  │
│     ├── vpin_features_2024-01_*.pkl                                     │
│     └── ...                                                              │
│  ⚡ 触发: StrategyFeatureLoader 计算特征时                               │
│  🗑️ 清理: find cache/features/monthly -name "*2024*" -delete            │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 3: FeatureStore 缓存 (策略特征集合)                      [192M]  │
│  📁 feature_store/                                                      │
│     └── bpc_highcap6_240T_v1/                                           │
│         └── BTCUSDT/240T/2024-01.parquet                                │
│  ⚡ 触发: --use-cache 模式                                               │
│  🗑️ 清理: --force-rebuild 或手动删除                                    │
└─────────────────────────────────────────────────────────────────────────┘

# 🔥 全部清理（最彻底）
rm -rf cache/timeframes/BTCUSDT_*.parquet
find cache/features/monthly -name "*BTCUSDT*" -delete
rm -rf feature_store/bpc_*/BTCUSDT

# 🎯 只清理某年（如 2024）
rm cache/timeframes/BTCUSDT_240T.parquet
find cache/features/monthly -name "*2024*" -delete
# FeatureStore 用 --force-rebuild 重建
```
## 训练树模型，效果好就可以导出规则
```bash
  # --end-date 2025-11-30 ← 改为包含 holdout 期间
mlbot train final --no-docker --config config/strategies/bpc --symbol BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT --timeframe 240T --data-path data/parquet_data --start-date 2023-01-01 --end-date 2025-11-30 --holdout-start-date 2024-05-01 --holdout-end-date 2025-11-30 --seed 42

python3 scripts/train_strategy_pipeline.py --config config/strategies/bpc --data-path data/parquet_data --symbol BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT --timeframe 240T --seed 42 --output-root results/bpc_highcap6 --start-date 2023-01-01 --end-date 2024-12-31 --train-all --feature-store-dir feature_store --feature-store-layer bpc_highcap6_240T_v1 --deterministic 2>&1 | tee /tmp/bpc_multisymbol_train.log


# 训练 failure_rr_extreme
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

# 训练 failure_no_opportunity
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

## 导出规则

```bash
mlbot train export-rules --no-docker --model-dir models/bpc --strategy bpc --max-splits 30

```