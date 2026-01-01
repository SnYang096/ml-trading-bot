# Feature-Group-Search 临时结果记录

> **状态**: 🔄 搜索进行中
> **更新时间**: 2026-01-01
> **策略**: sr_reversal_rr_reg_long

---

## 📍 结果文件位置

```bash
# 主目录
results/feature_group_search/sr_reversal_expanded/

# 各步骤结果
results/feature_group_search/sr_reversal_expanded/runs/
├── baseline/                    # 基线（无添加特征）
├── step1_add_poolb__xxx/        # Step 1: 添加单个 Pool B 特征
├── step2_add_xxx/               # Step 2: 在 Step 1 最佳基础上继续添加
└── ...

# 每个步骤下有多个 seed
step1_add_poolb__sqs_hal_high_f/
├── seed_1/sr_reversal_rr_reg_long__step1_.../results.json
├── seed_2/...
└── seed_3/...
```

### 查询命令

```bash
# 查看所有步骤的 Sharpe 排名
cd /workspaces/ml_trading_bot
python3 << 'EOF'
import json
from pathlib import Path
from collections import defaultdict

results = defaultdict(list)
for f in Path("results/feature_group_search/sr_reversal_expanded/runs").glob("*/seed_*/*/results.json"):
    step = f.parts[-4]
    with open(f) as fh:
        d = json.load(fh)
    sharpe = d.get('backtest', {}).get('sharpe')
    if sharpe is not None:
        results[step].append(sharpe)

avg = [(sum(v)/len(v), k, len(v)) for k, v in results.items() if v]
avg.sort(reverse=True)
for a, s, n in avg[:20]:
    print(f"Sharpe={a:.3f} ({n} seeds) - {s}")
EOF
```

---

## 📊 当前结果（sr_reversal_expanded）

### Baseline

| Metric | Value |
|--------|-------|
| **Sharpe_mean** | 1.529 |
| Seeds | 3 |

### ✅ Step 1 正面特征组（提升 Sharpe）

| 排名 | 特征组 | Sharpe_mean | vs Baseline | 建议 |
|------|--------|-------------|-------------|------|
| 1 | `sqs_hal_high_f` | **2.088** | +36% | ✅ 加入 |
| 2 | `dl_sequence_features_f` | 1.863 | +22% | ✅ 加入 |
| 3 | `volume_profile_volatility_features_f` | 1.810 | +18% | ✅ 加入 |
| 4 | `sr_strength_max_f` | 1.684 | +10% | ✅ 加入 |
| 5 | `wpt_volatility_features_f` | 1.575 | +3% | 可选 |
| 6 | `evt_features_f` | 1.541 | +1% | 可选 |

### ❌ Step 1 负面特征组（拉低 Sharpe）

| 特征组 | Sharpe_mean | vs Baseline | 建议 |
|--------|-------------|-------------|------|
| `trend_r2_50_f` | **-1.504** | -198% | 🔄 放入 invert_features |
| `dtw_features_reversal_f` | -0.866 | -157% | 🔄 放入 invert_features |
| `order_flow_all_features_f` | -0.765 | -150% | 🔄 放入 invert_features |
| `dtw_features_trend_f` | -0.714 | -147% | 🔄 放入 invert_features |
| `extended_volatility_features_f` | -0.397 | -126% | ❌ 踢掉 |

### 📝 中性特征组（无明显影响）

| 特征组 | Sharpe_mean | 说明 |
|--------|-------------|------|
| `wpt_scene__*` | 1.529 | 与 baseline 相同 |
| `wick_scene__*` | 1.529 | 与 baseline 相同 |
| `vpin_scene__*` | 1.529 | 与 baseline 相同 |

---

## 🔄 建议的 invert_features

将以下负面特征放入 `invert_features` 候选列表：

```yaml
# config/strategies/sr_reversal_rr_reg_long/features.yaml
# 或单独文件 invert_candidates.yaml

invert_features:
  # 严重负面 - 优先尝试 invert
  - trend_r2_50_f
  - dtw_features_reversal_f
  - order_flow_all_features_f
  - dtw_features_trend_f
```

**原理**: 这些特征在"正常"使用时拉低 Sharpe，但 invert 后可能变成正向信号（因为市场可能是反直觉的）。

---

## 📈 Step 2 结果（待更新）

Step 2 基于 Step 1 最佳（`sqs_hal_high_f`，Sharpe=2.088）继续添加特征。

| 添加的特征组 | Sharpe_mean | 说明 |
|-------------|-------------|------|
| `wpt_scene__ignition` | 2.088 | 无提升 |
| `wpt_scene__exhaustion` | 2.088 | 无提升 |
| ... | ... | 待更新 |

---

## 🚀 其他策略结果（待完成）

| 策略 | 状态 | Baseline Sharpe | 最佳 Sharpe |
|------|------|-----------------|-------------|
| sr_reversal_expanded | 🔄 进行中 | 1.529 | 2.088 |
| compression_breakout_expanded | 🔄 进行中 | TBD | TBD |
| sr_breakout_v4 | 🔄 进行中 | TBD | TBD |
| trend_following_v4 | 🔄 进行中 | TBD | TBD |

---

## 📚 复盘指南

### 1. 查看完整结果

```bash
# 查看最终 result.json（搜索完成后生成）
cat results/feature_group_search/sr_reversal_expanded/result.json | python3 -m json.tool

# 查看 HTML 报告
open results/feature_group_search/sr_reversal_expanded/report.html
```

### 2. 查看单个步骤详情

```bash
# 例如查看 step1_add_poolb__sqs_hal_high_f 的详细结果
cat results/feature_group_search/sr_reversal_expanded/runs/step1_add_poolb__sqs_hal_high_f/seed_1/*/results.json | python3 -m json.tool
```

### 3. 查看日志

```bash
# 实时查看
tail -f /tmp/fgs_sr_reversal_expanded.log

# 查看历史
less /tmp/fgs_sr_reversal_expanded.log
```

### 4. 监控脚本

```bash
./scripts/monitor_tasks.sh
```

---

## ⚠️ 注意事项

1. **CV 指标 ≠ Sharpe**: DTW 特征在 CV 上表现好（0.13），但 Sharpe 为负
2. **结果可能变化**: 不同 seed 可能产生不同结果，以 mean 为准
3. **搜索未完成**: 当前结果是中间状态，最终结果需等搜索完成

---

## 🔧 特征分组说明

### 特征函数 vs 特征列

- **组定义**用的是 `_f` 后缀的**特征函数名**（如 `dtw_features_reversal_f`）
- 每个函数会输出**多个特征列**，在 `config/feature_dependencies.yaml` 的 `output_columns` 定义
- 训练时实际使用的是列名，组只是逻辑分组

### ⚠️ 粒度问题与解决方案

**问题**：当前搜索是函数级别，但函数内的列可能效果不同：
- `dtw_features_reversal_f` 输出 ~20 列，可能 hammer 有用但 double_top 有害

**解决方案**：

1. **results.json 现在包含列级别 importance**（已实现）
   ```json
   {
     "feature_importance": {
       "dtw_min_dist_w15": 62.76,
       "dtw_bearish_engulfing_inverse_dist_w15": 44.84,
       ...
     }
   }
   ```

2. **分析命令**（查看列级别 importance）
   ```bash
   cd /workspaces/ml_trading_bot
   python3 << 'EOF'
   import pickle, json
   from pathlib import Path
   
   # 加载模型
   model_dir = Path("results/feature_group_search/sr_reversal_expanded/runs/step1_add_poolb__dtw_features_reversal_f/seed_1")
   model_path = list(model_dir.glob("*/model.pkl"))[0]
   features_path = model_path.parent / "used_features.json"
   
   with open(model_path, "rb") as f:
       models = pickle.load(f)
   with open(features_path, "r") as f:
       used_features = json.load(f)
   
   model = models[0]
   importances = model.feature_importance(importance_type='gain')
   feature_imp = sorted(zip(used_features, importances), key=lambda x: x[1], reverse=True)
   
   # 筛选 DTW 特征
   for name, imp in feature_imp:
       if 'dtw' in name.lower():
           print(f"{imp:10.2f}  {name}")
   EOF
   ```

3. **后续改进**（待实现）
   - 在 `features.yaml` 支持 `exclude_columns` 排除特定列
   - 添加 `mlbot diagnose column-importance` 命令

### DTW 语义化（新增）

原始 DTW 特征按形态模板分组，但没有语义化。已添加 `dtw_scene_semantic_scores_f`：

| 输出列 | 语义含义 | 适用策略 |
|--------|----------|----------|
| `dtw_reversal_bullish_score` | 看涨反转形态（锤子、头肩底、双底、看涨吞没） | SR 反转做多 |
| `dtw_reversal_bearish_score` | 看跌反转形态（射击之星、头肩顶、双顶、看跌吞没） | SR 反转做空 |
| `dtw_continuation_bullish_score` | 看涨延续形态（牛旗） | 趋势跟踪做多 |
| `dtw_continuation_bearish_score` | 看跌延续形态（熊旗） | 趋势跟踪做空 |
| `dtw_compression_score` | 压缩形态（三角形 + 压缩上下文） | 压缩突破 |
| `dtw_exhaustion_score` | 衰竭形态（顶/底 + 趋势衰减） | SR 反转 |

---

*文档自动生成，请勿手动编辑*

