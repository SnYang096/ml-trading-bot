# Makefile 命令 vs mlbot CLI 对比

本文档列出了 Makefile 中的命令与 mlbot CLI 的对应关系。

## ✅ 已支持 mlbot 的命令

### 因子/特征评估

| Makefile 命令     | mlbot 等价命令               | 状态     | 说明              |
| ----------------- | ---------------------------- | -------- | ----------------- |
| `ts-factor-eval`  | `mlbot analyze factor-eval`  | ✅ 已使用 | 单因子 IC/IR 评估 |
| `ts-dim-compare`  | `mlbot analyze dim-compare`  | ⚠️ 可迁移 | 特征降维和筛选    |
| `ts-feature-eval` | `mlbot analyze feature-eval` | ⚠️ 可迁移 | 特征类型评估      |

### 模型训练

| Makefile 命令          | mlbot 等价命令                  | 状态     | 说明                        |
| ---------------------- | ------------------------------- | -------- | --------------------------- |
| `ts-sr-reversal`       | `mlbot train sr-reversal`       | ⚠️ 可迁移 | SR Reversal 模型训练        |
| `ts-sr-reversal-long`  | `mlbot train sr-reversal-long`  | ⚠️ 可迁移 | SR Reversal Long-only 训练  |
| `ts-sr-reversal-short` | `mlbot train sr-reversal-short` | ⚠️ 可迁移 | SR Reversal Short-only 训练 |
| `rolling`              | `mlbot train rolling`           | ⚠️ 可迁移 | 滚动窗口训练                |

### 数据管理

| Makefile 命令   | mlbot 等价命令        | 状态     | 说明              |
| --------------- | --------------------- | -------- | ----------------- |
| `data-download` | `mlbot data download` | ⚠️ 可迁移 | 下载 Binance 数据 |
| `data-convert`  | `mlbot data convert`  | ⚠️ 可迁移 | ZIP 转 Parquet    |
| `data-pipeline` | `mlbot data pipeline` | ⚠️ 可迁移 | 完整数据管道      |

### 特征管理

| Makefile 命令       | mlbot 等价命令              | 状态     | 说明                 |
| ------------------- | --------------------------- | -------- | -------------------- |
| `list-features`     | `mlbot features list`       | ⚠️ 可迁移 | 列出所有特征         |
| `list-features-all` | `mlbot features list --all` | ⚠️ 可迁移 | 列出所有特征（详细） |

### 开发工具

| Makefile 命令 | mlbot 等价命令      | 状态     | 说明           |
| ------------- | ------------------- | -------- | -------------- |
| `format`      | `mlbot dev format`  | ⚠️ 可迁移 | 代码格式化     |
| `lint`        | `mlbot dev lint`    | ⚠️ 可迁移 | 代码检查       |
| `dev-install` | `mlbot dev install` | ⚠️ 可迁移 | 可编辑模式安装 |

---

## ❌ 尚未支持 mlbot 的命令（直接执行 Python 脚本）

以下命令在 Makefile 中直接执行 Python 脚本，尚未迁移到 mlbot CLI：

### 策略评估和对比

- `ts-strategy-feature-compare` - 特征配置对比（Ablation Study）
- `ts-sr-reversal-model-comparison` - SR Reversal 模型对比（规则 vs ML vs ML+Volatility）

### 规则基准和优化

- `ts-sr-reversal-rule-baseline` - SR Reversal 规则基准
- `ts-sr-reversal-1h-baseline` - SR Reversal 规则基准（1h 时间框架）
- `ts-sr-reversal-rule-optimization` - SR Reversal 规则参数优化
- `ts-rule-plateau-charts` - 规则参数 Plateau 图表生成

### ML 模型优化

- `ts-sr-reversal-optuna` - Optuna 阈值优化（快速）
- `ts-sr-reversal-optuna-joint` - Optuna 联合优化（超参数+阈值）
- `ts-sr-reversal-ml-param-sweep` - ML 参数扫描
- `ts-ml-plateau-charts` - ML 参数 Plateau 图表生成

### 回测

- `ts-vectorbot-backtest` - VectorBot 风险管理的回测
- `ts-nautilus-backtest` - Nautilus Trader 回测

### 分析和诊断

- `ts-analyze-ml-volatility` - ML+Volatility 模型性能分析
- `ts-analyze-dtw-volatility` - DTW 特征和波动率模型分析
- `ts-timeframe-comparison` - 时间框架对比报告
- `ts-timeframe-forward-report` - 时间框架 vs 前向 bar 相关性分析
- `feature-indicators` - 特征指标可视化

### 横截面分析

- `cs-build-panel` - 构建多资产因子面板
- `cs-report` - Fama-MacBeth 报告
- `cs-train` - 训练横截面模型
- `cs-catalog` - 因子目录导出
- `cs-select` - 自动因子选择
- `cs-shap` - SHAP 分析
- `cs-logic-check` - 因子逻辑检查
- `cs-shap-drift` - SHAP 漂移监控
- `cs-factor-eval` - 横截面因子评估

---

## 迁移建议

### 优先迁移的命令（高频使用）

1. **`ts-dim-compare`** → `mlbot analyze dim-compare`
2. **`ts-feature-eval`** → `mlbot analyze feature-eval`
3. **`ts-sr-reversal-*`** → `mlbot train sr-reversal-*`
4. **`rolling`** → `mlbot train rolling`

### 使用 mlbot（推荐，适用于所有环境）

mlbot 可以在所有环境中使用，包括本地、DevContainer 和 CI/CD。这是推荐的统一方式：

```bash
# 因子评估
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --target-lag 20 \
  --lag-tolerance 5

# 特征降维
mlbot analyze dim-compare \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T

# 模型训练
mlbot train sr-reversal-long \
  --symbol BTCUSDT \
  --timeframe 240T
```

**mlbot 的优势**:
- ✅ **统一接口**：在所有环境（本地、DevContainer、CI/CD）中使用相同的命令
- ✅ **跨平台兼容**：不依赖 Make，Windows/Linux/Mac 都可用
- ✅ **清晰的命令结构**：`mlbot <category> <command>` 的层次化设计
- ✅ **完整的参数控制**：所有参数都可以精确控制
- ✅ **更好的可维护性**：命令和参数集中管理，易于更新和文档化

### 使用 Makefile（向后兼容，便捷包装）

Makefile 提供便捷的默认参数和向后兼容，底层调用 mlbot 或 Python 脚本：

```bash
# Makefile 提供便捷的默认参数
make ts-factor-eval TS_FACTOR_STRATEGY=config/strategies/sr_reversal_long/features_all.yaml
```

**何时使用 Makefile**:
- ⚡ 需要快速执行常见任务（使用默认参数）
- 🔄 团队已有基于 Makefile 的工作流程
- 🛠️ 需要 Makefile 提供的其他功能（如 Docker 管理、依赖检查等）

**推荐使用 mlbot 的场景**:
- 🎯 **所有场景都推荐使用 mlbot**，特别是：
  - CI/CD 环境（统一接口，易于维护和调试）
  - 需要精确控制参数
  - 跨平台兼容性要求（Windows/Linux/Mac）
  - 需要清晰的命令结构和文档

