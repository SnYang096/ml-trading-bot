# 量化模型生命周期管理流程：从研究到生产

## 概述

本文档描述完整的量化模型开发、部署和运维流程，涵盖从离线研究到生产部署，再到持续监控与迭代的全生命周期管理。

**核心原则**：
- ✅ **所有生产训练都应该使用降维后的特征**（Top-K + Autoencoder）
- ✅ **严格的时间序列隔离**，避免数据泄露
- ✅ **完整的 Pipeline 封装**，确保训练和部署一致性
- ✅ **量化的漂移检测指标**，及时发现模型退化
- ✅ **版本控制与回滚能力**，保证系统稳定性

## 完整生命周期流程

### 阶段 1️⃣：离线研究（Research）

**目标**：验证信号是否有效，避免过拟合

#### 1.1 特征工程

构建候选特征池（如 TA-Lib、orderbook 特征、链上数据等）

**关键要求**：
- ✅ 确保无未来函数（所有特征仅用 t 时刻及之前信息）
- ✅ 所有价格类特征使用收益率或差分
- ✅ 移除任何包含"当前 close"的特征
- ✅ 对目标变量做"去连续性"处理（AR(1) 残差）

#### 1.2 特征有效性初筛

**单因子分析**：
- IC（信息系数）、Rank IC、分层回测
- 检查特征分布、稳定性、换手率
- 剔除无效/冗余/高共线性特征

**降维研究** (`make dim-compare`)：

```bash
# 使用一个季度的数据研究降维效果
make dim-compare SYMBOL=BTCUSDT \
  START_DATE=2025-05-01 END_DATE=2025-07-31 \
  ENCODING_DIM=32
```

**输出文件** (在 `results/production_dimensionality_20250501_20250731/`):
- `production_results.json` - 包含所有4个阶段的性能对比
- `dimensionality_report.html` - HTML 可视化报告
- `top_factors.json` - 代表性特征列表（Stage 3: 60-100个特征）✅
- `representative_factors.json` - 代表性特征列表（另一种格式）✅
- `production_autoencoder.pth` - 最佳 Autoencoder 模型（Stage 4）✅

**关键信息**:
- **代表性特征**: `top_factors.json` 包含60-100个特征名称
- **最佳压缩维度**: `production_results.json` 中的 `data_info.stage4_compressed_dim`（如32）
- **Autoencoder 模型**: `production_autoencoder.pth`
- **性能对比**: HTML 报告展示4个阶段的对比

#### 1.3 降维 & 特征选择

**方法**：
- L1 正则（Lasso）
- 基于树模型的 feature importance
- 递归特征消除（RFE）
- Autoencoder 压缩（Stage 4）

**目标**：保留信息量高、鲁棒性强、低相关性的特征子集

**注意事项**：
- ⚠️ **特征选择必须在 CV 内部进行**（嵌套 CV），避免数据泄露
- ⚠️ 优先用业务逻辑+统计检验，而非纯自动化
- ⚠️ 确保特征选择不参与任何测试集决策

#### 1.4 模型训练 & 交叉验证

**使用时间序列 CV**（`PurgedGroupTimeSeriesSplit` 或 `TimeSeriesSplit`）

**评估指标**：
- RMSE、方向准确率、F1、AUC
- Q50 loss ratio（Q50 loss ≤ Q10/Q90 loss）
- IC（Spearman/Pearson 相关性）
- 平衡准确率（处理标签不平衡）

**关键要求**：
- ✅ 测试集必须是纯 out-of-sample（OOS），不能参与任何决策
- ✅ 所有预处理统计量（median, MAD, AR(1) φ）必须在 CV 内部计算
- ✅ 避免 lookahead bias（未来信息泄露）

#### 1.5 样本外验证（Walk-Forward Analysis）

**模拟真实滚动预测**：训练 → 预测下一期 → 滚动推进

**观察重点**：
- 性能是否衰减
- 是否与回测一致
- 不同市场条件下的稳定性

**产出**：
- ✅ 一组有效的特征（`top_factors.json`）
- ✅ 一个表现稳定的模型架构
- ✅ 预处理规则（Winsorize 参数计算方式）
- ✅ Autoencoder 模型（`production_autoencoder.pth`）

---

### 阶段 2️⃣：上线部署（Productionization）

**目标**：将研究转化为可自动运行的系统

#### 2.1 封装完整 Pipeline

**使用 `QuantTradingModel` 类**（`src/ml_trading/models/quant_trading_model.py`）

**包含内容**：
- 特征计算 → 标签清洗（Winsorize）→ AR(1) 残差 → 模型预测 → 信号生成

**保存内容**：
- 模型权重 + 预处理参数（median, sigma, AR(1) φ）+ 特征列表 + 超参数

**保存方式**（使用 joblib）：

```python
from ml_trading.models.quant_trading_model import QuantTradingModel

# 训练和保存
model = QuantTradingModel(
    model_type="quantile",
    quantile_alpha=0.5,
    forward_bars=5,
    feature_cols=feature_list,
    preprocess_params=preprocess_params,
)
model.fit(X_train, y_train, current_returns=current_returns_train)
model.save("models/q50_pipeline_202511.pkl")
```

**输出文件结构**：
```
results/training/{config_dir}/
├── q50_pipeline.pkl      # Q50 模型 Pipeline（包含预处理参数）
├── q10_pipeline.pkl      # Q10 模型 Pipeline（包含预处理参数）
├── q90_pipeline.pkl      # Q90 模型 Pipeline（包含预处理参数）
├── vol_pipeline.pkl      # Volatility 模型 Pipeline
├── training_info.json    # 训练元数据（包含 model_paths）
├── features.txt          # 特征列表
└── scalers.pkl           # 特征缩放器
```

#### 2.2 训练生产模型

**使用降维后的特征训练模型**：

```bash
# 使用降维后的特征训练模型
DIM_RESULTS_DIR=results/production_dimensionality_20250501_20250731

make train SYMBOL=BTCUSDT \
  START_DATE=2025-01-01 END_DATE=2025-07-31 \
  USE_TOP_FACTORS=$(DIM_RESULTS_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_RESULTS_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32
```

**注意**：
- 从 `production_results.json` 读取 `data_info.stage4_compressed_dim` 作为 `ENCODING_DIM`
- **这一步是可选的**，可以跳过。`make auto-rolling-update` **已经包含训练过程**

#### 2.3 实盘模拟 / 小资金试跑

**监控指标**：
- 预测 vs 实际收益
- 信号胜率
- 最大回撤
- 方向准确率

**对比要求**：
- 是否与回测一致？
- 性能是否在合理范围内？

---

### 阶段 3️⃣：持续监控与迭代（MLOps for Quant）

**目标**：应对市场变化，防止模型退化

#### 3.1 定期 Retrain（如每月）

**使用最新 N 个月数据重新训练**

```bash
# 滚动更新（使用降维特征，自动检测所有数据）
DIM_RESULTS_DIR=results/production_dimensionality_20250501_20250731

make auto-rolling-update SYMBOL=BTCUSDT \
  INITIAL_TRAIN_MONTHS=6 \
  USE_TOP_FACTORS=$(DIM_RESULTS_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_RESULTS_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32
```

**增量更新**（每周/每月运行一次）：

```bash
# 只更新新月份（从上次位置继续）
make auto-rolling-update-only SYMBOL=BTCUSDT \
  OUTPUT=results/auto_rolling_btcusdt_XXX \
  USE_TOP_FACTORS=$(DIM_RESULTS_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_RESULTS_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32
```

**关键要求**：
- ✅ **同步更新预处理参数**（median, sigma, AR(1) φ）
- ✅ **同步更新模型**（LightGBM 权重）
- ✅ **同步更新特征**（如果特征工程也更新）

#### 3.2 漂移检测（Drift Detection）

**特征漂移检测**：
- **PSI（Population Stability Index）**：PSI > 0.1 报警
- **KS 检验**：检验特征分布是否变化
- **分布 KL 散度**：量化分布差异

**标签漂移检测**：
- 收益分布变化（波动率、偏度、极值频率）
- 方向比例变化（正负样本比例）

**预测漂移检测**：
- 模型输出分布变化（如 q50 预测均值漂移）
- Q50 loss ratio 变化
- 方向准确率变化

**实现建议**：

```python
def detect_drift(
    feature_train: pd.DataFrame,
    feature_prod: pd.DataFrame,
    label_train: pd.Series,
    label_prod: pd.Series,
    pred_train: np.ndarray,
    pred_prod: np.ndarray,
) -> Dict[str, float]:
    """检测特征、标签、预测漂移"""
    metrics = {}
    
    # 1. 特征漂移（PSI）
    for col in feature_train.columns:
        psi = calculate_psi(feature_train[col], feature_prod[col])
        metrics[f"psi_{col}"] = psi
        if psi > 0.1:
            print(f"⚠️ Feature {col} drift detected: PSI={psi:.4f}")
    
    # 2. 标签漂移
    label_vol_ratio = label_prod.std() / (label_train.std() + 1e-8)
    metrics["label_vol_ratio"] = label_vol_ratio
    if abs(label_vol_ratio - 1.0) > 0.3:
        print(f"⚠️ Label volatility drift: ratio={label_vol_ratio:.4f}")
    
    # 3. 预测漂移
    pred_mean_ratio = pred_prod.mean() / (pred_train.mean() + 1e-8)
    metrics["pred_mean_ratio"] = pred_mean_ratio
    if abs(pred_mean_ratio - 1.0) > 0.2:
        print(f"⚠️ Prediction drift: mean_ratio={pred_mean_ratio:.4f}")
    
    return metrics
```

#### 3.3 决策逻辑

**无显著漂移** → 继续使用新模型（含新预处理参数）

**有漂移但模型仍有效** → 调整预处理（如动态 k）或微调超参

**严重退化/失效** → 回到阶段 1，重新做特征研究

**判断标准**：
- PSI < 0.1：无漂移 ✅
- PSI 0.1-0.25：轻微漂移 ⚠️（监控）
- PSI > 0.25：严重漂移 ❌（需要重新训练或调整）

#### 3.4 版本管理 & 回滚机制

**每次 retrain 保存带时间戳的模型包**：

```
models/
├── q50_pipeline_20251101.pkl  # 2025-11-01 版本
├── q50_pipeline_20251201.pkl  # 2025-12-01 版本
├── q50_pipeline_latest.pkl    # 软链接指向最新版本
└── q50_pipeline_stable.pkl    # 软链接指向稳定版本
```

**版本管理策略**：

```bash
# 保存新版本
version="20251201"
model.save(f"models/q50_pipeline_{version}.pkl")

# 更新软链接
ln -sf models/q50_pipeline_${version}.pkl models/q50_pipeline_latest.pkl

# 如果性能稳定，标记为稳定版本
if performance_ok; then
    ln -sf models/q50_pipeline_${version}.pkl models/q50_pipeline_stable.pkl
fi
```

**回滚机制**：

```python
# 如果新版本出现问题，快速回滚到稳定版本
model = QuantTradingModel.load("models/q50_pipeline_stable.pkl")
```

---

## 完整工作流程（推荐）

### 步骤 1: 研究降维效果 (`make dim-compare`)

**目的**: 找到最优的特征集和压缩维度

```bash
make dim-compare SYMBOL=BTCUSDT \
  START_DATE=2025-05-01 END_DATE=2025-07-31 \
  ENCODING_DIM=32
```

**输出目录**（记录这个路径）：
```
DIM_DIR=results/production_dimensionality_20250501_20250731
```

### 步骤 2: 训练生产模型 (`make train`) - 可选

**目的**: 使用降维后的特征训练单个模型（用于一次性评估或部署）

```bash
make train SYMBOL=BTCUSDT \
  START_DATE=2025-01-01 END_DATE=2025-07-31 \
  USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32
```

**注意**: 
- **这一步是可选的**，可以跳过。`make auto-rolling-update` **已经包含训练过程**
- `make train`: 训练**一个**模型（单个时间段）
- `make auto-rolling-update`: **已经包含训练** - 训练**多个**模型（每月一个）

### 步骤 3: 滚动更新 (`make auto-rolling-update`)

**目的**: 滚动训练多个模型（每月一个），评估模型稳定性，使用降维后的特征

**重要**: `make auto-rolling-update` **已经包含训练过程**，会在循环中为每个月训练一个模型。

```bash
make auto-rolling-update SYMBOL=BTCUSDT \
  INITIAL_TRAIN_MONTHS=6 \
  USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32
```

**输出**:
- `results/auto_rolling_*/monthly_results.csv` - 所有月份的详细结果
- `results/auto_rolling_*/summary.json` - 汇总信息
- `results/auto_rolling_*/monthly_rolling_report.html` - HTML 报告
- `results/auto_rolling_*/model_YYYY-MM/` - 每个月的模型目录（包含 Pipeline 文件）

### 步骤 4: 定期更新（每周/每月运行一次）

```bash
make auto-rolling-update-only SYMBOL=BTCUSDT \
  OUTPUT=results/auto_rolling_btcusdt_XXX \
  USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32
```

---

## 常见陷阱 & 如何避免

| 陷阱 | 后果 | 解法 |
|------|------|------|
| 用全样本做特征选择 | 数据泄露，回测虚高 | 特征选择必须在 CV 内部进行（嵌套 CV） |
| 不更新预处理参数 | 预测分布偏移 | 每次 retrain 必须重新计算 median/sigma/AR(1) φ |
| 只看 RMSE 忽略方向性 | 模型无法交易 | 同时监控 F1/AUC/胜率 |
| 漂移检测只看 accuracy | 滞后严重 | 监控输入特征分布（PSI）更早发现问题 |
| 没有回滚机制 | 出问题无法止损 | 每次上线保留旧版本 |

---

## 部署使用示例

### 加载模型 Pipeline

```python
from ml_trading.models.quant_trading_model import QuantTradingModel

# 加载模型（一个文件包含所有内容）
model_q50 = QuantTradingModel.load("q50_pipeline.pkl")
model_q10 = QuantTradingModel.load("q10_pipeline.pkl")
model_q90 = QuantTradingModel.load("q90_pipeline.pkl")
model_vol = QuantTradingModel.load("vol_pipeline.pkl")
```

### 预测

```python
# 预测（预处理自动应用）
predictions_q50 = model_q50.predict(X_new, current_returns=current_returns_new)
predictions_q10 = model_q10.predict(X_new, current_returns=current_returns_new)
predictions_q90 = model_q90.predict(X_new, current_returns=current_returns_new)
predictions_vol = model_vol.predict(X_new)
```

### 生成交易信号

```python
# 生成交易信号
signals = model_q50.get_trading_signals(
    X_new,
    current_returns=current_returns_new,
    vol_pred=predictions_vol,
    signal_method="risk_adjusted"
)

# 信号包含：
# - signals["direction"]: 方向（1=上涨, -1=下跌）
# - signals["strength"]: 信号强度（风险调整后）
```

---

## 文件结构

### 训练输出结构

```
results/training/{config_dir}/
├── q50_pipeline.pkl      # Q50 模型 Pipeline（包含预处理参数）
├── q10_pipeline.pkl      # Q10 模型 Pipeline（包含预处理参数）
├── q90_pipeline.pkl      # Q90 模型 Pipeline（包含预处理参数）
├── vol_pipeline.pkl      # Volatility 模型 Pipeline
├── training_info.json    # 训练元数据
│   {
│     "model_paths": {
│       "q50": "path/to/q50_pipeline.pkl",
│       "q10": "path/to/q10_pipeline.pkl",
│       "q90": "path/to/q90_pipeline.pkl",
│       "volatility": "path/to/vol_pipeline.pkl"
│     },
│     "preprocess_params": {...},
│     "metrics": {...}
│   }
├── features.txt          # 特征列表
├── scalers.pkl           # 特征缩放器
└── training_report.html  # HTML 报告
```

### `dim-compare` 输出结构

```
results/production_dimensionality_20250501_20250731/
├── production_results.json           # 详细的4阶段对比结果
├── dimensionality_report.html        # HTML 可视化报告
├── top_factors.json                  # 代表性特征列表（兼容格式）
├── representative_factors.json      # 代表性特征列表（原始格式）
└── production_autoencoder.pth       # 最佳 Autoencoder 模型
```

---

## 最佳实践

### 1. 定期重新研究

每季度运行一次 `dim-compare`，评估最新的降维效果

### 2. 使用最新配置

每次生产训练都使用最新的 `top_factors.json` 和 `production_autoencoder.pth`

### 3. 保持一致性

研究、训练、滚动更新都使用相同的降维配置

### 4. 记录配置

保存 `dim-compare` 的输出目录，用于后续训练

### 5. 版本管理

每次 retrain 保存带时间戳的模型，支持快速回滚

### 6. 漂移监控

定期检查 PSI、预测分布、性能指标，及时发现模型退化

### 7. 预处理参数同步

每次 retrain 必须重新计算预处理参数（median, sigma, AR(1) φ），确保训练和部署一致性

---

## 总结

完整的量化模型生命周期管理流程：

**训练 → 特征筛选 → 再训练验证 → 封装上线 → 定期 retrain + 漂移检测 → 动态更新或重启**

这是一套闭环、自适应、生产级的量化模型运维流程。

**核心要点**：
- ✅ 严格的时间序列隔离
- ✅ 完整的 Pipeline 封装（`QuantTradingModel`）
- ✅ 量化的漂移检测指标（PSI、KS 检验等）
- ✅ 版本控制与回滚能力
- ✅ 预处理参数同步更新

只要在执行中注意这些要点，就能建立一个稳健、可维护的量化交易系统 🚀
