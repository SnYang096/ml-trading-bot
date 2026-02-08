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



## 审核 risk_gate.yaml 和 evidence_candidates.yaml 是否有语义意义
TODO：注意重跑一下训练，特征改了
比如：
usage_guide:
  step_1: 审核 feature 是否有语义意义      ← 你在这里
  step_2: 根据 threshold_examples 定义 bins
  step_3: 填写 usage_hint 和 affects
  step_4: 将确认的 Evidence 轴复制到 execution_archetype.yaml -> 这个需要改进


## Review 完成后 → 进入 阶段二：Evidence 软化（2.1-2.4）

---

# 阶段二：三层配置架构实现 (已完成)

## 2.1 配置结构迁移
配置从 `config/nnmultihead/execution_archetypes.yaml` 迁移到新的三层结构：
```
config/strategies/{strategy}/archetypes/
├── gate.yaml      # Gate 规则 (硬 veto)
├── evidence.yaml  # Evidence 规则 (软调整)
└── execution.yaml # Execution 约束 (RR/持仓)
```

**BPC 策略配置示例**：
- Gate: 2 hard gates + 2 soft filters + 1 system safety
- Evidence: 13 个特征轴，5档语义标签 (suppress/downweight/neutral/favor/amplify)
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
mlbot gate apply-archetype \
  --logs results/train_final_20260207_222521_rr_extreme/bpc/predictions.parquet \
  --strategy bpc
# 输出: ℹ️ Auto-detected feature store layer for bpc: features_a5ecdb3e27
# 输出: ℹ️ Auto output to: results/train_final_xxx/bpc/logs_gated.parquet

# 手动指定输出路径 (可选)
mlbot gate apply-archetype \
  --logs input.parquet \
  --out my_output.parquet \
  --strategy bpc
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
# 优化 Gate 规则阈值，寻找 lift 最大且稳定的平坦高原
python scripts/optimize_gate_lift_plateau.py \
  --strategy bpc \
  --logs results/train_final_xxx/bpc/logs_gated.parquet \
  --output gate_params.json \
  --min-lift 0.05 \
  --min-pass-rate 0.20 \
  --max-pass-rate 0.80
```


## 2.5 Evidence 平坦高原优化 (sharpness 目标)

### 原理说明
**为什么用 Sharpness？**
- Evidence 是软调整，不做硬 veto，而是影响置信度/仓位
- Sharpness 衡量特征对好坏样本的区分能力

```bash
# 优化 Evidence 分位数划分，寻找 sharpness 最大的配置
python scripts/optimize_evidence_plateau.py \
  --strategy bpc \
  --logs results/train_final_xxx/bpc/logs_gated.parquet \
  --output evidence_params.json \
  --min-sharpness 0.05
```

**sharpness 定义**：`sharpness = mean(RR | amplify) - mean(RR | suppress)`
- sharpness > 0: 特征高分位时 RR 更高，区分有效
- sharpness ≈ 0: 特征无区分能力
- sharpness < 0: 特征语义反转，需要翻转 mapping

**分析脚本示例** (在 Gate Allow 样本上计算):
```python
import pandas as pd
df = pd.read_parquet('results/train_final_xxx/bpc/logs_gated.parquet')
allowed = df[df['gate_decision'] == 'allow']

for feat in ['macd_signal_atr', 'vpin_ma20', 'sr_strength_max']:
    vals = allowed[feat]
    q20, q80 = vals.quantile([0.2, 0.8])
    suppress_rr = allowed[vals <= q20]['bpc_impulse_return_atr'].mean()
    amplify_rr = allowed[vals >= q80]['bpc_impulse_return_atr'].mean()
    sharpness = amplify_rr - suppress_rr
    print(f'{feat}: sharpness={sharpness:+.4f}')
```

**实验结果示例** (20260207, Gate Allow n=6169):
```
Evidence 特征 Sharpness:
  macd_signal_atr        sharpness=+0.6449  ← 最强区分度
  vol_slope_20           sharpness=+0.2715
  vpin_ma20              sharpness=+0.2500
  dist_to_nearest_sr     sharpness=+0.2383
  sr_strength_max        sharpness=-0.1243  ⚠️ 负值，需翻转
```

**注意事项**:
- 负 sharpness 说明 quantile_mapping 语义错误，需要翻转 labels 或移除特征
- Evidence 是软的，计算分数而非硬过滤

## 2.6 Execution 层参数网格搜索 (Sharpe 目标)

### 原理说明
**为什么用 Sharpe？**
- Execution 直接影响最终 PnL，Sharpe 是最终评判标准
- 目标: 找到 Sharpe 稳定的参数区间（平坦高原）

**搜索参数范围** (execution.yaml):
```yaml
stop_loss:
  type: trailing
  initial_r: [1.0, 1.5, 2.0, 2.5, 3.0]       # 初始止损距离
  trailing:
    activation_r: [0.5, 1.0, 1.5, 2.0]       # 激活移动止损的盈利阈值
    trail_r: [1.0, 1.5, 2.0]                 # 移动止损距离最高点的 R 值
```

**网格搜索脚本**:
```bash
python scripts/backtest_execution_layer.py \
  --logs results/train_final_xxx/bpc/logs_gated.parquet \
  --strategy bpc \
  --filter-allowed
```

**实验结果示例** (20260207, Gate Allow n=6169):
```
Execution 参数网格搜索:
   initial_r | activation_r | trail_r | sharpe | win_rate
   ---------------------------------------------------------
⭐      2.0 |          1.5 |     1.0 | 0.5307 |    69.9%
⭐      1.5 |          2.0 |     1.0 | 0.5307 |    69.9%
⭐      2.0 |          2.0 |     1.0 | 0.5307 |    69.9%
   ... (多组参数达到相同 Sharpe → 平坦高原)
```

**平坦高原现象解读**:
- 多种参数组合达到相同 Sharpe = 说明当前参数已经在「平坦高原」内
- 这是好事！意味着参数鲁棒，不会因为小调整而大幅波动
- 可以选择平坦高原的中点作为最终参数

## 2.7 最终 Sharpe Ratio 报告

**必须输出的指标** (根据用户要求):
```
██████████████████████████████████████████████████████████████████████
       🎯 最终 SHARPE RATIO 报告
██████████████████████████████████████████████████████████████████████

  📌 基准 (无 Gate):    Sharpe = 0.0395
  ✅ Gate 层过滤后:     Sharpe = 0.5307
     → 年化 Sharpe (4H): 20.64

  📊 详细指标:
     Trades:        6,169
     Mean RR:       0.2095
     Std RR:        0.3948
     Win Rate:      69.88%
     Profit Factor: 3.87
```

**计算脚本**:
```python
import pandas as pd
df = pd.read_parquet('results/train_final_xxx/bpc/logs_gated.parquet')
allowed = df[df['gate_decision'] == 'allow']
rr = allowed['bpc_impulse_return_atr']

print(f'Trades: {len(allowed)}')
print(f'Mean RR: {rr.mean():.4f}')
print(f'Std RR: {rr.std():.4f}')
print(f'Sharpe: {rr.mean() / rr.std():.4f}')
print(f'Win Rate: {(rr > 0).mean() * 100:.2f}%')
print(f'Profit Factor: {rr[rr > 0].sum() / abs(rr[rr < 0].sum()):.2f}')
```

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

## 2.9 测试验证
```bash
# 运行 E2E 测试
python -m pytest tests/test_archetype_e2e.py tests/test_archetype_cli_integration.py -v

# 快速验证
python -c "
from src.time_series_model.archetype import load_strategy_archetype
arch = load_strategy_archetype('bpc')
print(f'✓ {arch.name}: {len(arch.gate.all_rules)} gates, {len(arch.evidence.features)} evidence')
"
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