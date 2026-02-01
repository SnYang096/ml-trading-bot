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

##  训练 Return Tree

┌─────────────────────────────────────────────────────────────┐
│ 1. 标签生成函数 compute_bpc_return_tree_label               │
│    ├── 计算所有样本的 failure_any                           │
│    └── failure_any = 1 的样本 → forward_rr 设为 NaN         │
│                                                             │
│ 2. 训练流水线读取标签                                        │
│    ├── forward_rr 有值 = GOOD 样本                          │
│    └── forward_rr = NaN = failure 样本                      │
│                                                             │
│ 3. filters 配置自动过滤                                      │
│    filters:                                                 │
│      - column: forward_rr                                   │
│        notna: true   ← 只保留非空值                          │
└─────────────────────────────────────────────────────────────┘

使用 labels_return_tree.yaml → 只在 GOOD 样本（~failure_any）上训练
目标是 forward_rr（回归任务）
输出目录：results/train_final_<时间戳>_return_tree/bpc/

练完成后会自动：
保存模型和规则到同一目录
导出 bpc_tree_rules.md（高频分裂特征）
导出 risk_gate_draft.yaml

这些分裂特征就是 1.4 Evidence 轴候选的输入。
```bash
mlbot train final --no-docker \
  --config config/strategies/bpc \
  --labels config/strategies/bpc/labels_return_tree.yaml \
  --symbol BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --start-date 2023-01-01 \
  --end-date 2025-11-30 \
  --holdout-start-date 2024-05-01 \
  --holdout-end-date 2025-11-30 \
  --seed 42

```

## 审核 risk_gate.yaml 和 evidence_candidates.yaml 是否有语义意义
TODO：注意重跑一下训练，特征改了
比如：
usage_guide:
  step_1: 审核 feature 是否有语义意义      ← 你在这里
  step_2: 根据 threshold_examples 定义 bins
  step_3: 填写 usage_hint 和 affects
  step_4: 将确认的 Evidence 轴复制到 execution_archetype.yaml
具体 Review 要点
rank	feature	你需要决定
1	sma_200position (38次) | 语义：趋势位置？bins 怎么定？ |
2	bpc_volume_compression_pct (22次)	语义：压缩程度？
3	hilbert_cvd_price_env_ratio (11次)	语义：CVD/价格周期比？
4	rsi (11次)	经典 RSI，bins 可以用标准分位
5	hilbert_price_env (10次)	价格周期相位？
6	hilbert_cvd_env (9次)	CVD 周期相位？
7	vpin_max20 (8次)	VPIN 极值 - 流动性压力

## Review 完成后 → 进入 阶段二：Evidence 软化（2.1-2.4）