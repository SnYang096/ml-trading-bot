# 树模型导出规则到archetype
## 确保数据都在
```bash
mlbot data check-month-coverage --symbol BTCUSDT --start 2022-01 --end 2026-01 --zip-dir data/agg_data --parquet-dir data/parquet_data --no-docker

mlbot data convert --no-docker --input-dir data/agg_data --output-dir data/parquet_data --pattern '*-aggTrades-202[5]*.zip'

mlbot data convert --no-docker \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --pattern '*-aggTrades-202[5]-*.zip'
# 转成1min的ticks加速运算


```
## build 计算缓存，方便后面复用
```bash
# 只补充没有数据的
mlbot feature-store build --no-docker --config config/strategies/bpc --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT --timeframe 240T --data-path data/parquet_data --start-date 2023-05-01 --end-date 2025-12-31 --root feature_store 

# 如果图形显示结果不对，一般是缓存问题，删除月度特征缓存和feature store缓存
cd /home/yin/trading/ml_trading_bot && rm -rf cache/features/monthly/* && echo "Cleared cache/features/monthly/"

rm -rf feature_store/bpc_highcap6_240T_v1

mlbot feature-store build --no-docker \
  --config config/strategies/bpc \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --start-date 2023-01-01 \
  --end-date 2025-11-30 \
  --warmup-months 3 \
  --force-rebuild

#  --force-rebuild 参数 重新build，或者修改配置文件
# # 会 hash 这些文件的内容：
# - {config_dir}/features.yaml
# - {config_dir}/meta.yaml (可选)
# - {config_dir}/feature_contract.yaml (可选)
# - config/feature_dependencies.yaml (全局)
# 新构建的数据使用了 --warmup-months 3，百分位计算正确，默认也是这个
# 等构建完成后用 mlbot visualize feature-indicators --use-cache 可视化即可看到正常的图表
```
## 可视化特征，防止特征计算有问题，看看有没有空的 
```bash
# 支持单个symbol
mlbot visualize feature-indicators --no-docker \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2023-01-01 \
  --end-date 2025-12-31 \
  --strategy-config config/strategies/bpc \
  --use-cache
# --force-rebuild 只删除 FeatureStore 缓存，但不删除 monthly 特征缓存（cache/features/monthly/）。
mlbot visualize feature-indicators --no-docker \
  --symbol BNBUSDT \
  --timeframe 240T \
  --start-date 2023-01-01 \
  --end-date 2025-12-31 \
  --strategy-config config/strategies/bpc \
  --force-rebuild

  # 所有特征得一个图表BTC：results/feature_indicators/BTCUSDT_240min_feature_indicators_from20230101_to20251231_20260204_121724.html

# 如果数据不对，可以删除以下几层缓存
echo "清理 2024 年所有相关缓存（vpin, cvd, ofci, bb_width）..." && find /home/yin/trading/ml_trading_bot/cache/features/monthly -name "*2024*" \( -name "*vpin*" -o -name "*cvd*" -o -name "*ofci*" -o -name "*bb_width*" \) -type f -delete && echo "✅ 完成"

echo "🗑️ 清理所有 2024 年缓存..." && find /home/yin/trading/ml_trading_bot/cache -name "*2024*" -type f -delete 2>/dev/null && echo "✅ 完成" && find /home/yin/trading/ml_trading_bot/cache -name "*2023*" -type f 2>/dev/null | wc -l

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
## 训练树模型，效果好就可以导出规则，kpi是lift<1.0
注意点：
1. 分两个label训练
① 子类型不均衡：常见的那种 failure 吃掉了模型容量
② 两种 failure 模式在特征空间是“错位”的

2. 不把细节特征放到gate，不然会过滤掉很多交易

 📊 Failure Sub-label Analysis...
      ────────────────────────────────────────
      🌍 Global Failure Rate (baseline):
         failure_rr_extreme:     16.7%  (踩大坑)
         failure_no_opportunity: 4.5%  (入场即反)
      ────────────────────────────────────────
      ✅ Selected Trades (top 30%, n=6168):
         failure_rr_extreme:     0.7%  (lift=0.04x)
         failure_no_opportunity: 0.1%  (lift=0.03x)
      ────────────────────────────────────────
      🎯 Reduction vs unselected: +97.2%
  📜 Tree rules exported to results/train_final_20260204_230907_rr_extreme/bpc/bpc_tree_rules.md
   📜 Risk gate draft exported to results/train_final_20260204_230907_rr_extreme/bpc/risk_gate_draft.yaml
 


```bash
  # --end-date 2025-11-30 ← 改为包含 holdout 期间

# 训练 failure_rr_extreme “未来这 50 根 K 的路径里，会不会出现非常极端的不利 RR（比如一路亏到 -0.8R 以下）。
# 修改features_gate不需要重新build feature store
mlbot train final --no-docker \
  --config config/strategies/bpc \
  --features config/strategies/bpc/features_gate.yaml \
  --labels config/strategies/bpc/labels_rr_extreme.yaml \
  --symbol BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --data-path data/parquet_data \
  --start-date 2023-01-01 \
  --end-date 2025-11-30 \
  --holdout-start-date 2024-05-01 \
  --holdout-end-date 2025-11-30 \
  --seed 42

# 📜 Tree rules exported to results/train_final_20260205_230351_rr_extreme/bpc/bpc_tree_rules.md
# 📜 Risk gate draft exported to results/train_final_20260205_230351_rr_extreme/bpc/risk_gate_draft.yaml

# 分析 Gate 剩余失败归因（推荐使用这个新命令）
# ❗ 提示：只有当训练使用的 features.yaml 含 Evidence 特征（如 features_evidence.yaml 中的列）时，本命令才能在 Evidence 语义空间做归因；否则会自动跳过缺失列。

# 只有结构化的
mlbot analyze gate-residual \
  --model-dir results/train_final_20260205_230351_rr_extreme/bpc \
  --threshold 0.6 \
  --split holdout

# 也可以不训练下面，因为情况非常罕见
# 训练 failure_no_opportunity 入场即反
# ⚠️  重要：训练前必须手动修改 config/strategies/bpc/backtest.yaml！
# 因为 failure_no_opportunity 只占 4.4%，模型预测均值高达 95.6%
# 必须使用 long_entry_threshold: 0.95 才能筛选出优质机会！
#
# 步骤1: 修改 config/strategies/bpc/backtest.yaml
#   long_entry_threshold: 0.8  # 改为
#   long_entry_threshold: 0.95 # 对于 no_opportunity
#
# 步骤2: 运行训练
mlbot train final --no-docker \
  --config config/strategies/bpc \
  --features config/strategies/bpc/features_gate.yaml \
  --labels config/strategies/bpc/labels_no_opportunity.yaml \
  --symbol BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --data-path data/parquet_data \
  --start-date 2023-01-01 \
  --end-date 2025-11-30 \
  --holdout-start-date 2024-05-01 \
  --holdout-end-date 2025-11-30 \
  --seed 42

# 步骤3: 训练后记得改回 long_entry_threshold: 0.8！
```

##  训练 Return Tree，kpi比较复杂，主要有语义是否可用

> 📖 **详细评估框架**: 参见 [Return Tree KPI 量化评估文档](./return_tree_kpi_framework.md)

如果前面gate已经过滤了一些差得机会，我们可以训练 Return Tree
目的：让 Return Tree 在"已被 Gate 过滤后的样本"上学习，看它能否进一步区分好坏。
如果 Return Tree 的 特征重要性 主要来自：
  vol_regime_*（波动regime）
  terminal_risk_score（追末端）
  volume_participation_score（execution时机）
  exhaustion_*（节奏错位）
→ ✅ 证明剩余失败确实来自 Evidence/Execution 层面
如果特征重要性主要来自：
  bpc_dir_*（方向）
  trend_r2_*（趋势结构）
  jump_risk_*（极端风险）
→ ❌ 说明 Gate 还不够，需要加强

判断标准：
Return Tree 的 Top 10 重要特征中：
如果 ≥7个 是 Evidence 特征（vol/terminal/exhaustion/execution）
→ ✅ Gate 已经足够
如果 ≥5个 是 Gate 特征（dir/structure/trend/jump）
→ ❌ Gate 需要加强
训练完后，查看 feature_importance 和 tree_rules.md 就能验证您的假设！
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
  --features config/strategies/bpc/features_evidence.yaml \
  --labels config/strategies/bpc/labels_return_tree.yaml \
  --symbol BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --data-path data/parquet_data \
  --start-date 2023-01-01 \
  --end-date 2025-11-30 \
  --holdout-start-date 2024-05-01 \
  --holdout-end-date 2025-11-30 \
  --seed 42
```



## 审核 gate.yaml 和 evidence.yaml

训练完成后自动产出 `risk_gate_draft.yaml` 和 `evidence_candidates.yaml`，
需要人工审核语义后合并到正式 archetype 配置。

**审核流程**：
1. 检查每条规则/特征的 direction 是否符合语义（数据方向 vs 人类直觉）
2. 对照 tree_rules.md 中的分裂阈值，确认参数合理性
3. 确认 usage_hint 和 affects 填写完整
4. 将审核后的规则合并到 `config/strategies/bpc/archetypes/gate.yaml` 和 `evidence.yaml`

**当前配置状态** (2026-02-08):
- Gate: 3 hard gates + 9 soft filters + 2 guardrails + 1 system safety
- Evidence: 8 条已验证 (按 bad_suppression 排序)

---

# 阶段二：三层配置架构实现 (已完成)

## 2.1 配置结构迁移
配置从 `config/nnmultihead/execution_archetypes.yaml` 迁移到新的三层结构：
```
config/strategies/{strategy}/archetypes/
├── gate.yaml      # Gate 规则 (硬 veto + 软降权 + 护栏)
├── evidence.yaml  # Evidence 规则 (软调整，5档语义映射)
└── execution.yaml # Execution 约束 (RR/持仓)
```

**BPC 策略配置 (2026-02-08)**：
- Gate: 3 hard gates + 9 soft filters + 2 guardrails + 1 system safety
  - Hard: 方向拥挤(0.55) / WPT点火过强(0.209) / WPT未力竭(0.309)
  - Guardrail: 价格位置极端高(0.9) / BPC缩量不足(0.3)
- Evidence: 8 条已验证（按 bad_suppression 排序）
  - Top 3: macd_signal_atr(0.646) / vp_absorption(0.490) / wpt_exhaustion(0.473)
- Execution: SL=1.0R, TP=2.5R, direction_source=structure

## 2.2 核心模块
```bash
# 模块位置
src/time_series_model/archetype/
├── __init__.py   # 导出 StrategyArchetype, GateConfig, EvidenceConfig 等
└── loader.py     # 三层配置加载器
```

**Python 使用**：
```python
from src.time_series_model.archetype import load_strategy_archetype

arch = load_strategy_archetype("bpc")
print(arch.gate.all_rules)      # 所有 Gate 规则
print(arch.evidence.features)   # Evidence 特征列表
print(arch.execution.stop_loss_r)  # Execution 参数

# 应用 Gate 规则
passed, reasons, weight = arch.apply_gate(features_dict)

# 计算 Evidence 评分
score, breakdown = arch.compute_evidence_score(features_dict)
```

## 2.3 CLI 命令更新
```bash
# 应用 Gate 规则 (自动检测 feature store layer，输出到最新训练目录)
# 输入: predictions.parquet (模型预测结果)
# 输出: logs_gated.parquet (带 gate_decision 列)
mlbot gate apply-archetype \
  --logs results/train_final_<timestamp>_rr_extreme/bpc/predictions.parquet \
  --strategy bpc
# 输出: ℹ️ Auto-detected feature store layer for bpc: features_a5ecdb3e27
# 输出: ℹ️ Auto output to: results/train_final_xxx/bpc/logs_gated.parquet

# 手动指定输出路径 (可选)
mlbot gate apply-archetype \
  --logs input.parquet \
  --out my_output.parquet \
  --strategy bpc

# 分析 Gate 剩余失败归因
mlbot analyze gate-residual \
  --model-dir results/train_final_<timestamp>_rr_extreme/bpc \
  --threshold 0.6 \
  --split holdout
```

## 2.4 Gate 平坦高原优化 (lift 目标)

### 原理说明
**为什么用 Lift？**
- Gate 的目标是过滤坏机会，不是单纯提高通过率
- Lift 衡量 "Gate 是否让好样本比例提升"

**Good/Bad 定义** (基于 rr_extreme 标签):
- Good: `bpc_impulse_return_atr >= -0.8` (不踩大坑)
- Bad: `bpc_impulse_return_atr < -0.8` (踩大坑)

```bash
# 统一 Gate 优化（Hard + Soft 一次性优化）
python scripts/optimize_gate_unified.py \
  --strategy bpc \
  --logs results/train_final_<timestamp>_rr_extreme/bpc/logs_gated.parquet \
  --output results/train_final_<timestamp>_rr_extreme/bpc/gate_unified_optimization_v2.json

# 实际使用示例 (2026-02-08):
python scripts/optimize_gate_unified.py \
  --strategy bpc \
  --logs results/train_final_20260208_220616_return_tree/bpc/logs_gated.parquet \
  --output results/train_final_20260208_220616_return_tree/bpc/gate_unified_optimization_v2.json
```

**输出结构**：
- `gate_unified_optimization_v2.json`: 每条规则的 Lift、PassRate、Robustness、阈值
- 优化后手动审核，将结果更新到 `gate.yaml`

**Gate 四层分权架构**：
```
System Safety     → 市场级保护（反身性拥挤），不参与优化
    ↓
Hard Gates (3)    → 必要条件：方向过拥挤、WPT点火过强、WPT未力竭
    ↓
Guardrails (2)    → 策略语义约束：价格位置/缩量程度（固定阈值，不参与优化）
    ↓
Soft Filters (9)  → 增强条件：降权但不否决
```


## 2.5 Evidence 平坦高原优化 (bad_suppression 目标)

### 原理说明
**为什么用 bad_suppression？**
- Evidence 是软调整，不做硬 veto，而是影响置信度/仓位
- **bad_suppression** (主 KPI) = P(score<0.3|bad) - P(score<0.3|good)，衡量 bad 被选择性压制的程度
- **good_amplification** (辅 KPI) = P(score>0.7|good) - P(score>0.7|bad)
- 双重约束：bad_suppression > 0.05 AND good_amplification > 0.05

**bins 参数说明**：
- `bins: [0.25, 0.45, 0.65, 0.85]` 是分位数切分点，把特征值分布切成 5 档
- 每档对应语义标签：suppress(0.0) / downweight(0.25) / neutral(0.5) / favor(0.75) / amplify(1.0)
- bins 既用于优化时计算 KPI，也用于生产时实盘评分（不是中间参数）
- 优化脚本会遍历所有候选 bins 组合，找到 bad_suppression 最大且 plateau 稳定的那组

**Gate-conditioned 分位数阈值模式**：
- 用 good 样本（gate 放行）的分布计算分位数阈值
- 用这些阈值对 **所有样本**（包括 bad）赋分
- bad 样本如果特征值偏低，就会被映射到 suppress 区间

```bash
# 优化 Evidence 分位数划分
python scripts/optimize_evidence_plateau.py \
  --strategy bpc \
  --logs results/train_final_<timestamp>_return_tree/bpc/logs_gated.parquet \
  --output results/train_final_<timestamp>_return_tree/bpc/evidence_optimization.json \
  --min-sharpness 0.05

# 实际使用示例 (2026-02-08):
python scripts/optimize_evidence_plateau.py \
  --strategy bpc \
  --logs results/train_final_20260208_220616_return_tree/bpc/logs_gated.parquet \
  --output results/train_final_20260208_220616_return_tree/bpc/evidence_optimization.json \
  --min-sharpness 0.05
```

**输出**：
- `evidence_optimization.json`: 每条特征的 bins、bad_suppression、sharpness、plateau 判断
- `evidence_optimization.html`: 可视化报告
- 优化后将通过的特征 bins 更新到 `evidence.yaml`

**Plateau 稳定性判断** (三重条件)：
1. CV < 0.5（邻域 bad_suppression 变异系数低）
2. bins 连续（相邻 bins 步长 ≤ 0.15）
3. 语义邻居 ≥ 3 且 semantic_drift < 0.10

**实验结果** (2026-02-08, n=11113, good=10276, bad=837):
```
✅ Optimized (Stable Plateau): 8
   macd_signal_atr          bad_supp=0.646  bins=[0.1,0.3,0.5,0.7]
   vp_absorption_score      bad_supp=0.490  bins=[0.25,0.45,0.65,0.85]
   wpt_exhaustion_score     bad_supp=0.473  bins=[0.2,0.4,0.6,0.8]
   sma_200_position         bad_supp=0.432  bins=[0.25,0.45,0.65,0.85]
   vp_exhaustion_score      bad_supp=0.387  bins=[0.2,0.4,0.6,0.8]
   dist_to_nearest_sr       bad_supp=0.350  bins=[0.1,0.3,0.5,0.7]
   bpc_volume_compression   bad_supp=0.254  bins=[0.15,0.35,0.55,0.75]
   spectrum_price_flatness   bad_supp=0.123  bins=[0.3,0.5,0.7,0.9]
❌ Rejected (No Plateau): 2 — vpin_ma20, evt_tail_shape_right
❌ No valid bins: 5 — sr_strength_max, vol_slope_20, hilbert_price_env,
                      spectrum_cvd_low_freq_ratio, spectrum_cvd_centroid
```

**注意事项**:
- 负 bad_suppression 说明特征方向错误或无区分力
- Evidence 是软的，bins 直接参与实盘评分，不只是优化参数

## 2.6 Execution 层参数网格搜索 (Sharpe 目标)

### 原理说明
**为什么用 Sharpe？**
- Execution 直接影响最终 PnL，Sharpe 是最终评判标准
- 目标: 找到 Sharpe 稳定的参数区间（平坦高原）

**搜索参数范围** (execution.yaml `optimization` 段):
```yaml
optimization:
  enabled: true
  params:
    stop_loss.initial_r:
      range: [1.0, 3.0]      # 初始止损距离 (5值: 1.0/1.5/2.0/2.5/3.0)
      step: 0.5
    stop_loss.trailing.activation_r:
      range: [0.5, 2.0]      # 激活移动止损的盈利阈值 (4值)
      step: 0.5
    stop_loss.trailing.trail_r:
      range: [1.0, 2.5]      # 移动止损距离最高点 (4值: 1.0/1.5/2.0/2.5)
      step: 0.5
```

**网格搜索命令** (共 5×4×4 = 80 组参数):
```bash
python scripts/backtest_execution_layer.py \
  --logs results/train_final_<timestamp>/bpc/logs_gated.parquet \
  --strategy bpc \
  --filter-allowed \
  --grid-search \
  --output results/train_final_<timestamp>/bpc/execution_grid_search.json

# 输出:
#   execution_grid_search.json  — 完整数据
#   execution_grid_search.html  — 美化报告 (含 Heatmap)
```

**单参数回测** (不做网格搜索，只用当前 execution.yaml 参数):
```bash
python scripts/backtest_execution_layer.py \
  --logs results/train_final_<timestamp>/bpc/logs_gated.parquet \
  --strategy bpc \
  --filter-allowed
```

### 输出报告内容
- **核心 KPI**: 最佳 Sharpe / 当前 Sharpe / Delta / 年化 Sharpe
- **平坦高原分析**: Top N 参数组 CV (变异系数) < 0.15 = 稳定
- **Sharpe Heatmap**: 2D 热力图 (参数两两交叉切片)
- **完整排名**: Top 30 参数组合对比表
- **推荐配置**: 自动输出最佳 execution.yaml 参数

### 平坦高原现象解读
- 多种参数组合达到相同 Sharpe = 说明当前参数已经在「平坦高原」内
- 这是好事！意味着参数鲁棒，不会因为小调整而大幅波动
- 可以选择平坦高原的中点作为最终参数

## 2.7 Execution Layer 回测（逐K线 Bar-by-Bar 模拟）

**模拟方式**: 对每个入场信号，从下一根 bar 开始逐 bar 检查：止损 → 止盈 → 移动止损 → 超时。
入场方向由 `bpc_breakout_direction` 结构性特征决定。

**单次回测 + Per-Symbol 报告**:
```bash
python scripts/backtest_execution_layer.py \
  --logs results/train_final_<timestamp>/bpc/predictions.parquet \
  --strategy bpc \
  --output results/train_final_<timestamp>/bpc/execution_backtest
```

**使用 Gate 过滤的回测**:
```bash
python scripts/backtest_execution_layer.py \
  --logs results/train_final_<timestamp>/bpc/logs_gated.parquet \
  --strategy bpc \
  --filter-allowed \
  --output results/train_final_<timestamp>/bpc/execution_gated
```

**Grid Search 参数优化**:
```bash
python scripts/backtest_execution_layer.py \
  --logs results/train_final_<timestamp>/bpc/predictions.parquet \
  --strategy bpc \
  --grid-search \
  --output results/train_final_<timestamp>/bpc/execution_grid
```

**Tiers 分档回测** (按 evidence_score 分 4 档，每档独立执行参数):
```bash
python scripts/backtest_execution_layer.py \
  --logs results/train_final_<timestamp>/bpc/predictions.parquet \
  --strategy bpc \
  --tiers \
  --output results/train_final_<timestamp>/bpc/execution_tiers
```

**Noise Penalty 回测** (加载 FeatureStore 噪声特征，调整 SL/Size):
```bash
python scripts/backtest_execution_layer.py \
  --logs results/train_final_<timestamp>/bpc/predictions.parquet \
  --strategy bpc \
  --noise-penalty
```

**Tiers + Noise Penalty 组合**:
```bash
python scripts/backtest_execution_layer.py \
  --logs results/train_final_<timestamp>/bpc/predictions.parquet \
  --strategy bpc \
  --tiers --noise-penalty \
  --output results/train_final_<timestamp>/bpc/execution_full
```

**入场时机过滤 (Config-driven: `entry_filters.yaml`)**:
```bash
# 所有 filter 定义在 config/strategies/bpc/archetypes/entry_filters.yaml
# 代码动态解析 conditions，无需硬编码

# 推荐: CVD 吸收确认 (Sharpe 0.381, 993 trades, 最佳平衡)
python scripts/backtest_execution_layer.py \
  --logs results/train_final_<timestamp>/bpc/predictions.parquet \
  --strategy bpc \
  --entry-filter deep_pullback_cvd

# 高 Sharpe: 缩量压缩 (0.424, 429 trades)
python scripts/backtest_execution_layer.py \
  --logs results/train_final_<timestamp>/bpc/predictions.parquet \
  --strategy bpc \
  --entry-filter deep_pullback_vol

# 最高 Sharpe: 全确认 (0.492, 121 trades)
python scripts/backtest_execution_layer.py \
  --logs results/train_final_<timestamp>/bpc/predictions.parquet \
  --strategy bpc \
  --entry-filter deep_pullback_full
```

**入场过滤 + Tiers 组合** (推荐，Sharpe=0.395):
```bash
python scripts/backtest_execution_layer.py \
  --logs results/train_final_<timestamp>/bpc/predictions.parquet \
  --strategy bpc \
  --entry-filter deep_pullback_cvd \
  --tiers \
  --output results/train_final_<timestamp>/bpc/execution_filtered_tiers
```

**可用过滤策略 (9 种已启用, 7 种已禁用含原因)**:
| # | Filter | 特征维度 | Sharpe | Trades |
|---|--------|----------|--------|--------|
| - | none | 无过滤 | 0.308 | 20424 |
| 1 | deep_pullback_full | 结构+形态+订单流 | 0.492 | 121 |
| 2 | deep_pullback_vol | 结构+量能 | 0.424 | 429 |
| 3 | deep_pullback_momentum | 结构+动量 | 0.418 | 302 |
| 4 | deep_pullback_wick | 结构+形态 | 0.409 | 241 |
| 5 | deep_pullback_cvd | 结构+订单流 | 0.381 | 993 |
| 6 | deep_pullback_liq_void | 结构+footprint | 0.376 | 342 |
| 7 | deep_pullback_wpt | 结构+小波 | 0.375 | 336 |
| 8 | deep_pullback_bb | 结构+波动率 | 0.370 | 585 |
| 9 | deep_pullback | 结构 | 0.328 | 1762 |

**HTML 报告内容**:
- Overall KPI 卡片 (Sharpe/Mean R/Win Rate)
- Per-Symbol Breakdown 表格
- Per-Symbol Sharpe 柱状图
- 交易散点图 (RR over time，每个 symbol 独立子图 + 月度均线)
- 月度收益 Heatmap (Symbol × Month)
- Per-Tier Breakdown (启用 --tiers 时)
- Grid Search: Sharpe Heatmap + 平坦高原分析

## 2.8 完整 Pipeline 运行
```bash
# 一键执行完整流程
python scripts/run_full_pipeline.py \
  --task-spec config/tasks/bpc.yaml \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --start-date 2023-01-01 \
  --end-date 2025-11-30 \
  --model models/bpc.pt \
  --feature-store-layer bpc_highcap6_240T_v1 \
  --strategy bpc
```

## 2.9 流程总览

完整的从树模型到 archetype 的流水线：

```
① Feature Store Build      mlbot feature-store build ...
       ↓
② 训练 rr_extreme Tree      mlbot train final ... --labels labels_rr_extreme.yaml
       ↓
③ 应用 Gate                 mlbot gate apply-archetype ...
       ↓                       → 产出 logs_gated.parquet
④ Gate 优化                 python scripts/optimize_gate_unified.py ...
       ↓                       → 产出 gate_unified_optimization_v2.json
⑤ 审核 gate.yaml             手动审核 Lift/PassRate/方向
       ↓
⑥ 训练 Return Tree          mlbot train final ... --labels labels_return_tree.yaml
       ↓                       → 产出 evidence_candidates
⑦ Evidence 优化             python scripts/optimize_evidence_plateau.py ...
       ↓                       → 产出 evidence_optimization.json + .html
⑧ 审核 evidence.yaml         更新 bins/sharpness/rank
       ↓
⑨ Execution 网格搜索        python scripts/backtest_execution_layer.py ...
       ↓
⑩ Sharpe Report             最终指标报告
```

**快速验证**：
```bash
# 运行 E2E 测试
python -m pytest tests/test_archetype_e2e.py tests/test_archetype_cli_integration.py -v

# 快速检查配置加载
python -c "
from src.time_series_model.archetype import load_strategy_archetype
arch = load_strategy_archetype('bpc')
print(f'✓ {arch.name}: {len(arch.gate.all_rules)} gates, {len(arch.evidence.features)} evidence')
"
# 期望输出: ✓ bpc: 15 gates, 8 evidence
```

---

# 阶段三：实盘部署 (待执行)

## 3.1 实盘脚本
```bash
# 使用新的策略配置路径运行实盘
python scripts/run_live.py
# 环境变量: MLBOT_STRATEGIES_ROOT=config/strategies
```

## 3.2 清理废弃代码 (可选)
实盘稳定后可删除：
- `config/nnmultihead/execution_archetypes.yaml` (旧配置)
- `nnmultihead` 目录中废弃的代码