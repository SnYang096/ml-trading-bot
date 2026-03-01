# SHAP 特征筛选方案 — 集成到研究管线

> 创建: 2026-03-02
> 状态: 方案设计

## 1. 问题

当前特征选择是**手动固定**的：

- `features_gate.yaml` / `features_evidence.yaml` 列出 feature nodes
- 来源: 早期 `feature-group-search` 贪心搜索 + 人工判断
- **无自动化裁剪**: 管线每次重训都用同一批特征，不知道哪些已经失效
- **无稳定性验证**: split importance 换一批数据排名就变，不能区分"真重要"和"偶然重要"

已有基础设施：
- `evaluation.py._generate_shap_outputs()` — 降维模块已有 SHAP 计算，但**未接入研究管线**
- `shap==0.43.0` 已在 requirements.txt
- `walk_forward_validation.py` — 已有多 fold 训练框架

## 2. 目标

1. 自动识别"跨时间窗口稳定重要"的特征 → 减少过拟合
2. 自动裁剪低贡献特征 → 减少模型复杂度
3. 集成到 `auto_research_pipeline.py` → 每次重训自动执行
4. 输出可解释的报告 → 方便人工审核

## 3. 设计

### 3.1 在管线中的位置

```
现有管线:
  Step 1: Feature Store
  Step 2: Prepare (features_labeled.parquet)
  Step 3: Prefilter
  Step 4: Direction
  Step 5: Gate Train         ← 用 features_gate.yaml 里的特征
  Step 6: Evidence Train     ← 用 features_evidence.yaml 里的特征
  Step 7: Entry Filter
  Step 8: Execution
  Step 9: Backtest

加入 SHAP 后:
  Step 1: Feature Store
  Step 2: Prepare (features_labeled.parquet)
  Step 2.5: SHAP Feature Selection (NEW)  ← 在 Prepare 之后、Prefilter 之前
  Step 3: Prefilter
  Step 4: Direction
  Step 5: Gate Train         ← 用 SHAP 筛选后的特征
  Step 6: Evidence Train     ← 用 SHAP 筛选后的特征
  ...
```

**为什么放在 Step 2.5**:
- Step 2 产出 `features_labeled.parquet`（全量特征 + 标签），SHAP 需要这个作为输入
- 必须在 Gate/Evidence Train 之前，这样后续训练用筛选后的特征
- 在 Prefilter 之前：SHAP 看全量样本（更多数据 → 更稳定的重要性估计）

### 3.2 核心算法: Walk-Forward SHAP 稳定性筛选

```
输入: features_labeled.parquet (全量特征 + gate_label)
参数: n_folds=4, top_k=20, stability_threshold=0.75

1. 按时间切 N 个 fold (例: 4 个半年窗口)
   ┌─────────┬─────────┬─────────┬─────────┐
   │ Fold 1  │ Fold 2  │ Fold 3  │ Fold 4  │
   │ 23H1    │ 23H2    │ 24H1    │ 24H2    │
   └─────────┴─────────┴─────────┴─────────┘

2. 对每个 fold:
   a. 训练 LightGBM (同 gate 训练配置)
   b. 计算 SHAP values (TreeExplainer, sample=2000)
   c. 排名: mean |SHAP| → rank per feature

3. 聚合:
   ┌──────────────┬───────┬───────┬───────┬───────┬──────────┐
   │ Feature      │ Fold1 │ Fold2 │ Fold3 │ Fold4 │ Stable?  │
   ├──────────────┼───────┼───────┼───────┼───────┼──────────┤
   │ rsi          │ #2    │ #1    │ #3    │ #2    │ ✅ 4/4   │
   │ macd_hist    │ #5    │ #4    │ #7    │ #6    │ ✅ 4/4   │
   │ vpin         │ #3    │ #15   │ #2    │ #18   │ ❌ 2/4   │
   │ wick_ratio   │ #25   │ #22   │ #28   │ #30   │ ❌ 0/4   │
   └──────────────┴───────┴───────┴───────┴───────┴──────────┘

   稳定特征 = 在 >= 75% 的 fold 中排名 top-K

4. 输出:
   - shap_stable_features.json (稳定特征列表 + 统计)
   - shap_feature_report.json  (全特征稳定性矩阵)
   - plots/ 目录 (per-fold beeswarm + 跨 fold 稳定性热力图)
```

### 3.3 输出格式

```json
// shap_stable_features.json
{
  "strategy": "bpc",
  "n_folds": 4,
  "top_k": 20,
  "stability_threshold": 0.75,
  "total_features": 36,
  "stable_features": 18,
  "pruned_features": 18,
  "features": [
    {
      "name": "rsi",
      "node": "rsi_f",
      "mean_rank": 2.0,
      "folds_in_top_k": 4,
      "stability": 1.0,
      "mean_abs_shap": 0.0832
    },
    ...
  ]
}
```

### 3.4 与现有 features YAML 的衔接

当前 `features_gate.yaml` 结构:
```yaml
feature_pipeline:
  requested_features:
    - rsi_f
    - macd_f
    - volume_ratio_f
    ...
```

SHAP 筛选后，`--promote` 会写回一个**裁剪后的** features YAML:
- 只保留 stable features 对应的 feature nodes
- 新增 `_shap_pruned` 注释记录被剪掉的 nodes
- 同时更新 `features_gate.yaml` 和 `features_evidence.yaml`

### 3.5 安全约束

| 约束 | 规则 | 原因 |
|------|------|------|
| 最少特征数 | `stable_count >= 8` | 低于 8 个特征 LightGBM 容易欠拟合 |
| 核心特征保护 | `atr_f` 永远保留 | 执行层 SL/TP 必需 |
| Fallback | 稳定特征不足 → 跳过裁剪 | 宁可不裁也不裁错 |
| A/B 对比 | 裁剪前后都跑 backtest | 确认裁剪不降 Sharpe |

## 4. 实现计划

### 4.1 新脚本: `scripts/shap_feature_selection.py`

```
CLI:
  python scripts/shap_feature_selection.py \
    --logs features_labeled.parquet \
    --strategy bpc \
    --label-col gate_label \
    --n-folds 4 \
    --top-k 20 \
    --stability-threshold 0.75 \
    --output shap_report/ \
    --promote                     # 写回 features_gate.yaml
```

核心函数:
1. `split_time_folds(df, n_folds)` → 按时间均匀切分
2. `train_and_shap(X, y, fold_id)` → LightGBM + TreeExplainer
3. `compute_stability(shap_rankings, top_k, threshold)` → 稳定性矩阵
4. `generate_report(stability, output_dir)` → JSON + plots
5. `promote_features(stable_nodes, strategy, strategies_root)` → 写回 YAML

### 4.2 管线集成: `auto_research_pipeline.py`

在 Step 2 (Prepare) 和 Step 3 (Prefilter) 之间插入:

```python
# ── Step 2.5: SHAP Feature Selection (可选, 默认开启) ──
if not skip_shap:
    run_step("SHAP Feature Selection", [
        "python", "scripts/shap_feature_selection.py",
        "--logs", f"{prepare_dir}/features_labeled.parquet",
        "--strategy", strategy,
        "--strategies-root", strategies_root,
        "--n-folds", "4",
        "--top-k", "20",
        "--output", f"{run_dir}/shap/",
        "--promote",
    ], log, dry_run=dry_run)
```

新增 CLI 参数: `--skip-shap` (跳过 SHAP 筛选, 用于快速迭代)

### 4.3 配置: `config/research_pipeline.yaml`

```yaml
shap_feature_selection:
  enabled: true
  n_folds: 4           # 时间窗口数
  top_k: 20            # 每个 fold 取 top-K
  stability_threshold: 0.75  # 特征在 >= 75% fold 出现才算稳定
  min_stable_features: 8     # 稳定特征不足则跳过裁剪
  protected_nodes:            # 永远保留的节点
    - atr_f
  apply_to:                   # 裁剪哪些 features YAML
    - features_gate.yaml
    - features_evidence.yaml
```

## 5. 预期收益

| 维度 | 现状 | SHAP 筛选后 |
|------|------|-------------|
| 特征数 | ~36 (固定) | ~18-22 (自动) |
| 过拟合风险 | 高 (含噪声特征) | 低 (只保留稳定特征) |
| 模型训练速度 | 基准 | ~1.5x 快 (特征减半) |
| 可解释性 | 无 | 每次训练有 SHAP 报告 |
| 维护成本 | 手动调特征 | 自动化 |

## 6. 风险与缓解

| 风险 | 缓解 |
|------|------|
| SHAP 计算慢 (3年数据) | sample=2000, 4 folds 并行 |
| 裁剪过激导致 Sharpe 下降 | min_stable=8 + A/B backtest |
| 不同策略稳定集差异大 | 每策略独立筛选, 不共享结果 |
| Walk-Forward fold 太少 | 默认 4 folds (每 fold ~9 个月), 足够 |

## 7. 与现有工具的关系

| 工具 | 定位 | SHAP 筛选的关系 |
|------|------|----------------|
| `feature-group-search` | 初始候选池构建 (node 级) | SHAP 在此之后, 做 column 级裁剪 |
| `evaluation.py` SHAP | 降维模块诊断输出 | 复用其 `_generate_shap_outputs`, 扩展为多 fold |
| `walk_forward_validation.py` | IS/OOS Sharpe 对比 | 可共享 fold 切分逻辑 |
| `check_need_retrain.py` | 重训触发 | SHAP 漂移可作为新触发条件 (Phase 2) |

## 8. 实施节奏

| 阶段 | 内容 | 工时 |
|------|------|------|
| Phase 1 | `shap_feature_selection.py` 脚本 (独立可用) | 3-4h |
| Phase 2 | 集成到 `auto_research_pipeline.py` Step 2.5 | 1h |
| Phase 3 | 验证: 三策略各跑一次, 对比裁剪前后 Sharpe | 2h |
| Phase 4 | (可选) 实盘 SHAP 漂移监控 | 后续 |
