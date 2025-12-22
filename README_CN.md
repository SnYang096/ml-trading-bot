# ML Trading Bot

本仓库包含因子研究、降维、模型训练和实盘回测堆栈的生产就绪组件。`src/time_series_model/` 下的代码包含可复用的 Python 包；`scripts/` 目录仅暴露最小的一组命令行入口点，用于包装包 API。

## 快速开始

1. 创建虚拟环境（conda、venv 等）并激活它。
2. 以可编辑模式安装项目：
   ```bash
   pip install -e .[dev]
   ```
3. 安装 Git pre-commit 钩子（可选但推荐）：
   ```bash
   make install-hooks
   ```
   这会在每次提交前自动运行 `mlbot dev format` 和 `mlbot dev lint` 以确保代码质量。
4. 运行帮助命令验证安装：
   ```bash
   mlbot --help
   ```
   或查看所有可用命令：
   ```bash
   mlbot analyze --help
   mlbot train --help
   mlbot diagnose --help
   mlbot optimize --help
   ```

## 推荐使用流程

### 核心工作流（配置驱动架构）

推荐的工作流使用**配置驱动架构**，具有特定策略的配置：

1. **特征分析** (`mlbot analyze factor-eval`): 评估因子并选择最优特征
2. **策略训练** (`mlbot train sr-reversal-long`, `mlbot train sr-reversal-short`, `mlbot train rolling`): 使用策略配置训练模型
3. **消融研究** (`mlbot analyze strategy-feature-compare`): 比较特征配置（可选）
4. **回测** (`mlbot backtest vectorbot`): 验证策略性能

**📖 详细工作流请参见 [完整流程指南](docs/时序模型/完整流程指南.md)。**

### 分步工作流

#### 为什么 SR Reversal 要拆成 Long/Short（以及为什么移除了 `sr_reversal/`）

我们现在刻意使用 **两个方向固定的策略**：
- `sr_reversal_long`：只做多，二元标签 = “在 \(t+1\) 进多单后，是否先到 +2R 再到 -1R”
- `sr_reversal_short`：只做空，二元标签 = “在 \(t+1\) 进空单后，是否先到 +2R 再到 -1R”

这替代了旧的 **双向** 配置（`sr_reversal/`, `combine_mode: any_success`），后者把做多/做空的结果混在同一个标签里。

**标签语义的关键差异**：
- **旧（`any_success`）**：对每根 K 线同时假设做多和做空两种入场；只要任意一个方向“成功”，标签就可能为 1。这会让 `pred` 很难解释成“某个具体方向的成功概率”，并且往往需要依赖额外的“方向来源”（例如 `signal`）来决定到底做多还是做空。
- **新（`long_only` / `short_only`）**：标签只对应一个明确动作。`pred` 就是该方向在 RR 定义下的**成功概率**，阈值更稳定，也更容易保证回测/实盘一致。

**拆分的好处**：
- **目标更干净**：`predict_proba` 直接代表“该方向交易成功的概率”。
- **执行更简单**：方向来自策略本身（不需要 `use_signal_direction`，也不需要 `signal` 列）。
- **控制更细**：多空可以分别设阈值/风控。
- **一致性更强**：离线回测与生产推理语义一致。

我们仍保留一个轻量的**保险丝**（可选），防止 OOD/噪声 regime 下过度交易：
`dist_to_nearest_sr / ATR > K  =>  不交易`。

#### 步骤 0: 验证特征正确性（推荐）

在开始特征评估之前，运行测试以验证特征计算正确且不使用未来数据：

**快速测试**（仅关键特征）：
```bash
make test-key-features-all
```

**综合测试**（所有特征）：
```bash
make test-all-features-comprehensive
```

**这些测试验证的内容**：
- ✅ 无未来数据泄露（时刻 t 的特征仅使用 ≤ t 的数据）
- ✅ 多资产归一化（特征在不同资产间可比较）
- ✅ 流式与批量一致性（生产推理与训练匹配）
- ✅ 无全局归一化（防止前瞻偏差）

**注意**：在进行特征评估之前，这些测试应该通过。如果测试失败，在训练模型之前修复问题。

**📖 更多详细信息请参见 [测试运行说明](docs/测试运行说明.md)。**

#### 步骤 1: 特征分析

**1.1 评估单个因子**：

**基本用法**：
```bash
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-10-31 \
```

**评估特定因子**：
```bash
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --factors atr sqs_hal_high \
  --timeframe 240T
```

**高级选项**（去除相关特征，按最佳滞后过滤）：
```bash
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-10-31 \
  --remove-correlated \
  --correlation-threshold 0.9 \
  --target-lag 20 \
  --lag-tolerance 5 \
  --filter-by-best-lag \
  --open-browser
```

#### 步骤 2: 特征消融研究（必需）

比较不同的特征配置以验证特征选择：

```bash
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-10-31 \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml"
```

**使用滚动窗口**（更稳健）：
```bash
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml" \
  --run-rolling \
  --rolling-train-bars 5000 \
  --rolling-test-bars 1000 \
  --rolling-max-windows 10
```

**此步骤验证的内容**：
- 选定的特征比所有特征表现更好
- 特征选择提高模型泛化能力
- 特征消融显示出有意义的差异

**注意**：在进行滚动训练之前，此步骤是**必需的**。

#### 步骤 3: 模型对比（必需）

验证 ML 模型优于基于规则的策略：

```bash
mlbot diagnose model-comparison \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31
```

**此步骤对比的内容**：
- 基于规则的基线（纯规则策略）
- ML 模型（XGBoost/LightGBM）
- ML + 波动率模型

**此步骤验证的内容**：
- ML 模型显著优于规则
- ML 模型提供稳定的收益
- ML 模型具有合理的交易频率

**注意**：在进行滚动训练之前，此步骤是**必需的**。

#### 步骤 4: 策略训练（可选 - 用于调试）

**⚠️ 可选**：单次训练仅用于调试或快速配置测试。对于生产环境，在完成步骤 2 和 3 后，直接进行步骤 5（滚动训练）。

**4.1 快速验证**（单次训练）：

**SR Reversal 仅做多**：
```bash
mlbot train sr-reversal-long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-10-31 \
```

**SR Reversal 仅做空**：
```bash
mlbot train sr-reversal-short \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-10-31 \
```

#### 步骤 5: 生产训练（滚动窗口 - 推荐）
```bash
# 扩展窗口训练：每个测试月份使用所有之前的月份
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --initial-train-months 6 \
  --min-train-months 3

mlbot train rolling \
  --config config/strategies/sr_reversal_short \
  --symbol BTCUSDT \
  --timeframe 240T \
  --initial-train-months 6 \
  --min-train-months 3
```

**指定日期范围**：
```bash
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start 2024-01-01 \
  --end 2025-10-31 \
  --initial-train-months 6 \
  --min-train-months 3
```

**仅更新（增量）**：
```bash
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --update-only
```

**输出**：
- `results/rolling/{strategy}/{month}/model.pkl` - 每个月的模型
- `results/rolling/{strategy}/monthly_results.json` - 聚合结果

**注意**：只有在完成步骤 2（特征消融研究）和步骤 3（模型对比）后，才进行滚动训练，以确保特征和模型架构已验证。

**使用选定特征**（如果您有特征选择结果）：
```bash
# 如果您有来自 factor-eval 的 features_suggested.yaml，请更新您的策略配置以使用它
# 编辑策略的 features.yaml 以使用选定的特征，然后：
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --initial-train-months 6
```

**输出**：
- `results/auto_rolling_*/monthly_results.csv` - 所有月份的详细结果
- `results/auto_rolling_*/summary.json` - 摘要信息
- `results/auto_rolling_*/monthly_rolling_report.html` - HTML 报告
- `results/auto_rolling_*/model_YYYY-MM.txt` - 每个月的模型

#### 在 Dev Container（Cursor / VS Code）中查看 HTML 报告

如果你在 Dev Container 里运行，`--open-browser` 可能无法自动打开报告。推荐用下面方式：

**方式 A：本地静态服务器 + 端口转发**（最稳定）

```bash
# 示例：把 results 目录作为静态站点暴露出来
python3 -m http.server 8008 --directory results
```

- 在 Cursor/VS Code 打开 **Ports** 面板，转发/打开端口 `8008`
- 然后在本机浏览器打开对应报告，例如：
  - `results/auto_rolling_*/monthly_rolling_report.html`
  - `results/strategy_compare/strategy_feature_compare_report.html`
  - `results/rule_optimization/optimization_report.html`

#### 步骤 6: 定期更新（每周/每月）

从最后训练的月份进行增量更新：

```bash
# 仅更新新月份（从最后位置开始）- 使用 --update-only 标志
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --update-only
```

这将自动检测最后训练的月份并从那里继续。

### 完整工作流管道

从特征评估到滚动训练的完整工作流，按顺序执行命令：

```bash
# 步骤 0: 验证特征正确性（推荐）
make test-key-features-all

# 步骤 1: 特征评估和选择
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start-date 2025-01-01 \
  --end-date 2025-04-30 \
  --remove-correlated \
  --target-lag 5

# 步骤 2: 特征消融研究（验证特征选择）
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start-date 2025-01-01 \
  --end-date 2025-04-30 \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml"

# 步骤 3: 模型对比（验证 ML 优于规则）
mlbot diagnose model-comparison \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start-date 2025-01-01 \
  --end-date 2025-04-30

# 步骤 5: 滚动窗口训练（主要生产工作流）
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start 2025-01-01 \
  --end 2025-04-30 \
  --initial-train-months 3 \
  --min-train-months 3
```

> **提示**  
> - 使用 `--docker` 标志（默认）以在可用时启用 GPU 训练  
> - 训练后查看 `results/rolling_*/summary.json` 中的 `monthly_results`  
> - 如果结果中出现 `drift detected`，考虑重新评估特征或调整参数  


## 数据管道

在训练之前，确保您有数据：

```bash
# 下载 Binance 月度聚合交易数据
mlbot data download \
  --symbols BTCUSDT,ETHUSDT \
  --start-year 2021 \
  --start-month 1

# 将 ZIP 转换为 Parquet（5 分钟 OHLC + 订单流）
mlbot data convert

# 或一次性运行两者（完整管道）
mlbot data pipeline \
  --symbols BTCUSDT,ETHUSDT
```

**更多数据管道示例**：

```bash
# 下载特定日期范围
mlbot data download \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --start-year 2024 \
  --start-month 1 \
  --end-year 2025 \
  --end-month 10

# 转换并清理（转换后删除 ZIP 文件）
mlbot data convert --cleanup
```

## 核心原则

**所有生产训练都应使用降维特征**（Top-K + Autoencoder），而不是原始的 482 个特征。

### 为什么？

1. **更好的性能**：降维特征通常表现更好（如研究所示）
2. **更快的训练**：更少的特征 = 更快的训练
3. **减少过拟合**：降低过拟合风险
4. **一致性**：与研究阶段使用相同的特征集

## 命令对比

| 命令                     | 用途                           | 何时使用                               |
| --------------------------- | --------------------------------- | ----------------------------------------- |
| `mlbot analyze factor-eval` | 因子评估与选择     | **推荐**：主要特征选择方法 |
| `mlbot train sr-reversal-long/short` | 训练单个模型（方向固定） | **可选**：仅用于调试/快速验证 |
| `mlbot train rolling`       | 滚动窗口训练           | **推荐**：主要生产工作流 |

### 关键点

- **工作流顺序**：始终按步骤 0 → 1 → 2 → 3 → 5 的顺序进行
  - 步骤 0: 验证特征正确性（推荐）
  - 步骤 1: 特征评估 (`factor-eval`)
  - 步骤 2: 特征消融研究 (`strategy-feature-compare`) - **必需**
  - 步骤 3: 模型对比 (`model-comparison`) - **必需**
  - 步骤 5: 滚动训练 (`rolling`) - **仅在验证后**

- `mlbot train sr-reversal-long/short`: 为单个时间段训练**一个**模型（方向固定）
  - **不推荐**用于生产评估
  - 仅用于调试或快速配置测试

- `mlbot train rolling`: 以滚动/扩展窗口方式训练**多个**模型（每月一个）
  - **必需**用于生产部署
  - 通过扩展窗口提供更好的评估
  - 只有在验证特征（步骤 2）和模型性能（步骤 3）后使用
  - 步骤 4（单次训练）是可选的，仅用于调试


## 工作流摘要

### 最小工作流（5 步，推荐）

```bash
# 步骤 0: 验证特征正确性（开始前推荐）
make test-key-features-all

# 步骤 1: 特征评估和选择
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --remove-correlated \
  --filter-by-best-lag

# 步骤 2: 特征消融研究（验证特征选择）
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml"

# 步骤 3: 模型对比（验证 ML 优于规则）
mlbot diagnose model-comparison \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# 步骤 4: 滚动训练（仅在验证后）
# 注意：如果您有来自 factor-eval 的 features_suggested.yaml，请更新您的策略配置以使用它
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start 2024-01-01 \
  --end 2025-10-31 \
  --initial-train-months 6 \
  --min-train-months 3
```

### 完整工作流（5 步，包含所有选项）

```bash
# 步骤 0: 验证特征正确性（推荐）
make test-all-features-comprehensive

# 步骤 1: 特征评估（生成包含选定特征的 features_suggested.yaml）
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --remove-correlated \
  --target-lag 20 \
  --filter-by-best-lag

# 步骤 2: 特征消融研究（比较原始特征与选定特征）
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml" \
  --run-rolling \
  --rolling-train-bars 5000 \
  --rolling-test-bars 1000 \
  --rolling-max-windows 10

# 步骤 3: 模型对比（验证 ML 优于规则）
mlbot diagnose model-comparison \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# 步骤 4: 滚动训练（主要生产工作流 - 仅在验证后）
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start 2024-01-01 \
  --end 2025-10-31 \
  --initial-train-months 6 \
  --min-train-months 3
```

### 高级工作流示例

**诊断和优化工作流**：

```bash
# 1. 规则基线（测试纯基于规则的策略）
mlbot diagnose rule-baseline \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# 2. 规则优化（找到最优参数）
mlbot optimize rule \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --search-type random \
  --n-trials 100

# 3. 生成规则平台图表
mlbot optimize rule-plateau-charts \
  --results-csv results/rule_optimization/optimization_results.csv \
  --report-html results/rule_optimization/optimization_report.html

# 4. ML 参数扫描
mlbot optimize ml-param-sweep \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-07-31

# 5. 生成 ML 平台图表
mlbot optimize ml-plateau-charts \
  --timeframe 240T
```

**横截面分析工作流**：

```bash
# 1. 构建横截面面板
mlbot cross-section build-panel \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# 2. 生成 Fama-MacBeth 报告
mlbot cross-section report \
  --panel-path data/cross_sectional_panels/panel.parquet \
  --output-dir results/cross_sectional

# 3. 训练横截面模型
mlbot cross-section train \
  --panel-path data/cross_sectional_panels/panel.parquet

# 4. 自动选择因子
mlbot cross-section select \
  --panel-path data/cross_sectional_panels/panel.parquet \
  --output-path results/cross_sectional/selected_factors.json

# 5. SHAP 分析
mlbot cross-section shap \
  --model-path results/cross_sectional/model.pkl \
  --panel-path data/cross_sectional_panels/panel.parquet

# 6. SHAP 漂移监控
mlbot cross-section shap-drift \
  --model-path results/cross_sectional/model.pkl \
  --panel-path data/cross_sectional_panels/panel.parquet
```

## 文档

- **`docs/workflow_research_to_production.md`** - 完整工作流文档
- **`docs/simplified_workflow.md`** - 简化工作流指南

## 获取帮助

查看所有可用命令：
```bash
mlbot --help
```

查看特定类别的命令：
```bash
mlbot analyze --help      # 分析和评估命令
mlbot train --help        # 训练命令
mlbot diagnose --help     # 诊断命令
mlbot optimize --help     # 优化命令
mlbot backtest --help     # 回测命令
mlbot cross-section --help # 横截面分析命令
mlbot data --help         # 数据管理命令
mlbot features --help     # 特征管理命令
mlbot dev --help          # 开发命令
```

另请参见：
- [迁移指南](docs/MIGRATION_GUIDE.md) - 从 Makefile 迁移到 mlbot 的完整指南
- [Makefile vs mlbot](docs/MAKEFILE_VS_MLBOT.md) - 命令对比表

## 开发环境

**推荐使用 VS Code Dev Container**：
1. 用 VS Code 打开项目
2. 选择 "Reopen in Container"（自动进入 Dev Container）
3. 在容器内直接运行 `mlbot` 命令（无需通过 Makefile）

**命令行使用**：
- 在 Dev Container 中：直接使用 `mlbot` 命令
- 在本地环境：使用 `mlbot` 命令（需要先 `pip install -e .`）

所有 `mlbot` 命令都支持 `--docker/--no-docker` 选项，可以根据环境自动选择合适的执行方式。

