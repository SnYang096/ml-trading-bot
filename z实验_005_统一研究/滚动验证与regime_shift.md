# 滚动验证 (Walk-Forward CV) 与 Regime Shift 防御

## 核心问题

单次 holdout 划分 (如 train=2023-2025.07, holdout=2025.08-2026.02) 存在两个缺陷：

1. **Regime 偏差**: holdout 恰好落在某个市场状态 (牛市/熊市/震荡), 评估结果只反映该 regime
2. **数据利用率低**: 只有最后 6 个月做 OOS 验证, 前面 2.5 年的数据永远不做测试

---

## 一、特征层: SHAP 8-Fold 稳定性筛选

> 脚本: `scripts/shap_feature_selection.py`
> 配置: `config/research_pipeline.yaml` → `shap_feature_selection`
> 调用方: `auto_research_pipeline.py` Step 2.5（自动集成, 每次训练都跑）

### 1.1 解决什么问题

LightGBM 训练出的 SHAP 重要性排名可能因**训练时段不同**而变化。某些特征只在牛市有效（regime-specific），用这些特征建 gate/evidence 会过拟合到特定 regime。

**目标**: 找出在**所有时段**都稳定重要的特征, 剔除 regime-specific 特征。

### 1.2 训练数据

- **数据源**: `features_labeled.parquet`（完整训练 + holdout 数据）
- **数据范围**: `start_date` → `end_date`（当前 2023-01-01 → 2026-02-01）
- **数据内容**: 所有 archetype 信号对应的特征行 + label (`success_no_rr_extreme`)
- **典型规模**:
  - BPC (240T/4h bars): ~39,000 行 × 6 symbols → 每fold ~4,875 行
  - ME (60T/1h bars): ~187,000 行 × 6 symbols → 每fold ~23,375 行

### 1.3 如何划分

**等分时间切片, 无 train/test 内部划分**:

```
features_labeled.parquet (按时间排序)
│
├── Fold 1: 前 1/8 数据  (2023-01 → 2023-05)  ~23,000 rows (ME)
├── Fold 2: 第 2/8       (2023-05 → 2023-09)
├── Fold 3: 第 3/8       (2023-09 → 2024-01)
├── Fold 4: 第 4/8       (2024-01 → 2024-05)
├── Fold 5: 第 5/8       (2024-05 → 2024-10)
├── Fold 6: 第 6/8       (2024-10 → 2025-02)
├── Fold 7: 第 7/8       (2025-02 → 2025-07)
└── Fold 8: 最后 1/8     (2025-07 → 2026-02)
```

每个 fold 内部**不再划分 train/test**。整个 fold 数据同时用于:
- 训练一棵独立的 LightGBM
- 在同一数据上计算 TreeExplainer SHAP (sample=2000)
- 输出: mean |SHAP| 排名 → Top-K 特征列表

### 1.4 稳定性判定

```
每个 fold 独立产出 Top-20 特征 (按 mean |SHAP| 排序)

稳定特征 = 在 >= 6/8 fold (75%) 都排入 Top-20
不稳定特征 = 只在 <= 5/8 fold 出现 → 剔除
```

**配置参数** (`research_pipeline.yaml`):

| 参数                  | 全局默认                         | ME 覆盖 | 含义                         |
| --------------------- | -------------------------------- | ------- | ---------------------------- |
| `n_folds`             | 8                                | 8       | 时间窗口数                   |
| `top_k`               | 20                               | 25      | 每 fold 取 Top-K             |
| `stability_threshold` | 0.75                             | 0.50    | 特征在 >= N% fold 出现才保留 |
| `protected_nodes`     | `[atr_f, fer_failure_signals_f]` | 同      | 永远不裁                     |

ME 用更宽松的 0.50 门槛，因为订单流特征 (vpin/shd) 在某些时段强、某些时段弱，0.75 会误裁。

### 1.5 数据够不够

| 策略       | 总行数   | 每 fold 行数 | 判断                                           |
| ---------- | -------- | ------------ | ---------------------------------------------- |
| BPC (240T) | ~39,000  | ~4,875       | 足够训练 LightGBM (只需 SHAP 排名, 不需高精度) |
| ME (60T)   | ~187,000 | ~23,375      | 充足                                           |
| FER (240T) | ~45,000  | ~5,625       | 足够                                           |

SHAP 稳定性不需要模型预测精确，只需排名一致性。每 fold 5000+ 行足以产出稳定的 SHAP 排名。

### 1.6 输出

```
config/strategies/{strategy}/features_gate_shap.yaml    ← 裁剪后的 gate 特征
config/strategies/{strategy}/features_evidence_shap.yaml ← 裁剪后的 evidence 特征
results/.../shap/shap_report.json                        ← 稳定性报告
```

后续 gate/evidence 训练使用 `_shap.yaml` 而非原始 `features_gate.yaml`, 只用稳定特征建模。

---

## 二、参数层: Walk-Forward Decay Ratio

> 脚本: `scripts/walk_forward_validation.py`
> 配置: `config/research_pipeline.yaml` → `dates.holdout_months`
> 调用方: **独立运行**（不在 auto_research_pipeline 中自动调用）

### 2.1 解决什么问题

`auto_research_pipeline` 产出的 Sharpe 包含**优化层过拟合**:
- 模型预测本身是 OOS（模型只在 holdout 前训练）
- 但 gate/evidence/entry_filter/execution 的参数是在 holdout 上优化的
- 最终回测也在同一份 holdout 上
- 报告的 Sharpe 对优化层来说是 **In-Sample**

**目标**: 量化优化层参数在**未来新 regime** 上还能保持多少 Sharpe。

### 2.2 训练数据

- **数据源**: `data/parquet_data/` 原始 K 线 → 管线内部生成 features_labeled.parquet
- **数据范围**: 每个 fold 独立重建 features → 训练 → 优化 → 回测
- **每个 fold 都是完整管线执行** (download → feature → train → optimize → backtest)

### 2.3 如何划分

**Anchored Walk-Forward, 从 end_date 向前倒推**:

```
配置: start_date=2023-01-01, end_date=2026-02-01, holdout_months=6, folds=3

Fold 1 (校准):                                        Fold 2:                                                 Fold 3 (生产):
train: 2023-01 → 2024-08                               train: 2023-01 → 2025-02                                train: 2023-01 → 2025-08
holdout: 2024-08 → 2025-02                             holdout: 2025-02 → 2025-08                              holdout: 2025-08 → 2026-02
(只产 IS, 无 OOS)                                      (IS + OOS)                                               (IS + OOS)
│                                                       │                                                        │
├── 训练 LightGBM                                       ├── 训练 LightGBM                                        ├── 训练 LightGBM
├── SHAP 特征筛选                                       ├── SHAP 特征筛选                                        ├── SHAP 特征筛选
├── 优化 gate/entry_filter/execution                    ├── 优化 gate/entry_filter/execution                     ├── 优化 gate/entry_filter/execution
├── holdout 回测 → IS Sharpe                            ├── holdout 回测 → IS Sharpe                             ├── holdout 回测 → IS Sharpe
└── 冻结配置 ─────────────────────────────────→              └── 冻结配置 ─────────────────────────────────→               │
                                          │                                                              │    │
                                  应用到 Fold 2 预测                                              应用到 Fold 3 预测
                                  → OOS Sharpe                                                    → OOS Sharpe
```

**每个 fold 内部的 train/holdout 划分**:
- train: `start_date` → `holdout_start` (模型训练 + 特征筛选用此数据)
- holdout: `holdout_start` → `end_date` (优化层参数调优 + IS 回测用此数据)
- 训练窗口起点固定 (anchored at 2023-01), 逐步扩大
- holdout 窗口不重叠, 各 6 个月

### 2.4 两阶段验证

**Phase 1 (各 fold 独立运行, 耗时)**:
- 每个 fold 调用完整 `run_strategy_pipeline()`: 数据准备 → LightGBM → SHAP → gate → evidence → entry_filter → execution → 回测
- holdout 上的回测 Sharpe = **IS Sharpe** (因为优化和回测用同一份 holdout)

**Phase 2 (冻结配置 × 新数据, 轻量)**:
- 取 Fold N 的 `predictions.parquet` (模型在 Fold N 训练范围内训练)
- 应用 Fold N-1 的**冻结配置** (`gate.yaml` + `entry_filters.yaml` + `execution.yaml`)
- 在 Fold N 的 holdout 上回测 → **OOS Sharpe**

### 2.5 Decay Ratio 判读

```
Decay Ratio = OOS Sharpe / IS Sharpe

≥ 0.7   → ✅ 参数稳健, 优化层未引入严重过拟合
0.5~0.7 → ⚠️ 中等衰减, 可接受但需持续观察
0.3~0.5 → 🟡 较大衰减, 优化层可能过拟合 (减少优化参数数量)
< 0.3   → 🔴 严重过拟合, 优化层在拟合噪声 (需简化管线)
```

### 2.6 数据够不够

| Fold | 训练期                 | 训练数据量 (ME 60T) | Holdout         | Holdout 数据量 |
| ---- | ---------------------- | ------------------- | --------------- | -------------- |
| 1    | 2023-01→2024-08 (19月) | ~82,000 rows        | 2024-08→2025-02 | ~26,000 rows   |
| 2    | 2023-01→2025-02 (25月) | ~108,000 rows       | 2025-02→2025-08 | ~26,000 rows   |
| 3    | 2023-01→2025-08 (31月) | ~134,000 rows       | 2025-08→2026-02 | ~26,000 rows   |

BPC (240T) 的数据量约为 ME 的 1/4.8, 但 Fold 1 仍有 ~17,000 训练行, 足够 LightGBM + 参数优化。

**交易笔数是更大瓶颈**: 单 fold 6 个月 holdout 可能只有 30-50 笔交易 (BPC), 统计置信度有限。增加 folds=5 可缓解但计算成本翻倍。

---

## 三、两层验证的关系

```
                         auto_research_pipeline (单次)
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
     Step 2.5                  Step 3-5                Walk-Forward
   SHAP 特征筛选            gate/evidence/             (独立脚本)
   (8-fold 稳定性)           execution 优化             (3-fold OOS)
          │                         │                         │
          ▼                         ▼                         ▼
   剔除 regime-specific       产出策略配置              量化优化层
   特征 (内层防御)            (可能过拟合)              过拟合程度
                                                       (外层验证)
```

| 维度         | SHAP 8-Fold                  | Walk-Forward                                      |
| ------------ | ---------------------------- | ------------------------------------------------- |
| **验证对象** | 特征重要性排名               | 优化层参数 (gate/entry_filter/execution)          |
| **验证方式** | 跨 fold 排名一致性           | IS Sharpe vs OOS Sharpe                           |
| **数据划分** | 等分 8 块, 各独立训练        | Anchored expanding, 前 fold 冻结配置→后 fold 验证 |
| **集成方式** | 管线内自动 (Step 2.5)        | 独立脚本, 手动运行                                |
| **耗时**     | ~5 分钟 (训练 8 棵 LightGBM) | ~4.5 小时 (3 fold × 1.5h 完整管线)                |
| **输出**     | `features_*_shap.yaml`       | `wf_summary.json` + Decay Ratio                   |

### 为什么需要两层

- SHAP 8-fold 只验证**特征是否跨 regime 稳定**, 不验证 gate 阈值、entry filter 参数
- Walk-Forward 验证**优化出的参数是否泛化**, 但计算成本高, 不适合日常迭代
- 两层互补: SHAP 快速剔除坏特征 (内层), WF 验证整体管线 (外层)

---

## 四、数据利用率对比

| 方法                  | OOS 数据量                     | 覆盖 regime 数 | 验证层级 |
| --------------------- | ------------------------------ | -------------- | -------- |
| 单次 holdout (6 个月) | 6 个月                         | 1              | 无 (IS)  |
| SHAP 8-fold 稳定性    | 全部数据 (每 fold 约 4.5 个月) | 8              | 特征     |
| Walk-Forward 3 fold   | OOS = 12 个月                  | 2-3            | 参数     |
| Walk-Forward 5 fold   | OOS = 24 个月                  | 3-5            | 参数     |

---

## 五、实际使用

```bash
# 1. 日常迭代 (SHAP 自动包含)
python scripts/auto_research_pipeline.py --all --use-1min
# → 内部自动调用 shap_feature_selection.py, 8-fold 稳定性筛选

# 2. 里程碑验证 (参数锁定前, 独立运行)
python scripts/walk_forward_validation.py --strategy me --folds 3 --resume
python scripts/walk_forward_validation.py --strategy bpc --folds 3 --resume

# 3. 预览 fold 划分 (不执行)
python scripts/walk_forward_validation.py --strategy me --folds 3 --dry-run

# 4. 已有 fold 结果, 只做 OOS 对比
python scripts/walk_forward_validation.py --strategy me --folds 3 --oos-only
```

## 六、推荐工作流

```
日常迭代:
  auto_research_pipeline --all
  → 单次 holdout + SHAP 8-fold
  → 快速验证想法 (~1.5h)

里程碑验证 (参数锁定前):
  walk_forward_validation --folds 3
  → 多 regime OOS 验证 (~4.5h)

判定:
  SHAP 稳定特征 >= 8 个 → 特征池健康
  Decay Ratio ≥ 0.7    → 可部署
  Decay Ratio < 0.5    → 需简化优化层
```

## 七、未来可选改进

1. **Expanding → Sliding window**: 训练窗口不固定起点, 丢弃旧数据 (适应 regime drift)
2. **Purged Group CV**: 在 fold 边界留 gap, 防止 label 跨 fold 泄漏
3. **Combinatorial Purged CV**: 所有 fold 组合都做 OOS, 进一步提高数据利用率
4. **EMA200 周期调优**: 当前固定 200, 可在 WF 中用 100/150/200 做对比
