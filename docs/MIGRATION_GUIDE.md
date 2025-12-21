# Makefile 到 mlbot CLI 迁移指南

本文档提供从 Makefile 命令迁移到 mlbot CLI 的完整指南。

## 快速参考

### 命令映射表

| Makefile 命令 | mlbot 命令 | 说明 |
|--------------|-----------|------|
| `ts-factor-eval` | `mlbot analyze factor-eval` | 因子评估 |
| `ts-dim-compare` | `mlbot analyze dim-compare` | 特征降维 |
| `ts-feature-eval` | `mlbot analyze feature-eval` | 特征类型评估 |
| `ts-strategy-feature-compare` | `mlbot analyze strategy-feature-compare` | 特征配置对比 |
| `ts-timeframe-comparison` | `mlbot analyze timeframe-comparison` | 时间框架对比 |
| `ts-timeframe-forward-report` | `mlbot analyze timeframe-forward-report` | 时间框架前向相关性分析 |
| `ts-sr-reversal-rule-baseline` | `mlbot diagnose rule-baseline` | 规则基准测试 |
| `ts-test-vpin-thresholds` | `mlbot diagnose test-vpin-thresholds` | VPIN 阈值测试 |
| `ts-analyze-ml-volatility` | `mlbot diagnose ml-volatility` | ML+波动率分析 |
| `ts-analyze-dtw-volatility` | `mlbot diagnose dtw-volatility` | DTW+波动率分析 |
| `ts-sr-reversal-model-comparison` | `mlbot diagnose model-comparison` | 模型对比 |
| `ts-sr-reversal-rule-optimization` | `mlbot optimize rule` | 规则优化 |
| `ts-rule-plateau-charts` | `mlbot optimize rule-plateau-charts` | 规则 Plateau 图表 |
| `ts-sr-reversal-ml-param-sweep` | `mlbot optimize ml-param-sweep` | ML 参数扫描 |
| `ts-ml-plateau-charts` | `mlbot optimize ml-plateau-charts` | ML Plateau 图表 |
| `ts-vectorbot-backtest` | `mlbot backtest vectorbot` | VectorBot 回测 |
| `ts-nautilus-backtest` | `mlbot backtest nautilus` | Nautilus 回测 |
| `cs-build-panel` | `mlbot cross-section build-panel` | 构建面板 |
| `cs-report` | `mlbot cross-section report` | Fama-MacBeth 报告 |
| `cs-train` | `mlbot cross-section train` | 训练模型 |
| `cs-catalog` | `mlbot cross-section catalog` | 因子目录 |
| `cs-select` | `mlbot cross-section select` | 因子选择 |
| `cs-shap` | `mlbot cross-section shap` | SHAP 分析 |
| `cs-logic-check` | `mlbot cross-section logic-check` | 逻辑检查 |
| `cs-shap-drift` | `mlbot cross-section shap-drift` | SHAP 漂移监控 |
| `cs-factor-eval` | `mlbot cross-section factor-eval` | 因子评估 |
| `feature-indicators` | `mlbot visualize feature-indicators` | 特征指标可视化 |
| `ts-sr-reversal` | `mlbot train sr-reversal` | SR Reversal 训练 |
| `ts-sr-reversal-long` | `mlbot train sr-reversal-long` | SR Reversal Long 训练 |
| `ts-sr-reversal-short` | `mlbot train sr-reversal-short` | SR Reversal Short 训练 |
| `rolling` | `mlbot train rolling` | 滚动窗口训练 |
| `data-download` | `mlbot data download` | 数据下载 |
| `data-convert` | `mlbot data convert` | 数据转换 |
| `data-pipeline` | `mlbot data pipeline` | 数据管道 |
| `list-features` | `mlbot features list` | 列出特征 |
| `format` | `mlbot dev format` | 代码格式化 |
| `lint` | `mlbot dev lint` | 代码检查 |
| `dev-install` | `mlbot dev install` | 安装包 |

## 详细迁移示例

### 1. 因子评估

**Makefile**:
```bash
make ts-factor-eval \
  TS_FACTOR_STRATEGY=config/strategies/sr_reversal_long/features_all.yaml \
  TS_FACTOR_SYMBOL=BTCUSDT \
  TS_FACTOR_TIMEFRAME=240T \
  TS_FACTOR_START=2024-01-01 \
  TS_FACTOR_END=2025-10-31 \
  TS_FACTOR_TARGET_LAG=20
```

**mlbot**:
```bash
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --target-lag 20
```

### 2. 特征配置对比

**Makefile**:
```bash
make ts-strategy-feature-compare \
  STRAT_COMPARE_CONFIG=config/strategies/sr_reversal_long \
  STRAT_COMPARE_SYMBOL=BTCUSDT \
  STRAT_COMPARE_TIMEFRAME=240T \
  STRAT_COMPARE_START=2024-01-01 \
  STRAT_COMPARE_END=2025-10-31 \
  STRAT_COMPARE_OVERRIDES="baseline=config/features/baseline.yaml full=config/features/full.yaml"
```

**mlbot**:
```bash
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --feature-overrides "baseline=config/features/baseline.yaml full=config/features/full.yaml"
```

### 3. 规则基准测试

**Makefile**:
```bash
make ts-sr-reversal-rule-baseline \
  SR_BASELINE_CONFIG=config/strategies/sr_reversal \
  SR_BASELINE_SYMBOL=BTCUSDT \
  SR_BASELINE_TIMEFRAME=240T \
  SR_BASELINE_START=2024-01-01 \
  SR_BASELINE_END=2025-10-31
```

**mlbot**:
```bash
mlbot diagnose rule-baseline \
  --strategy-config config/strategies/sr_reversal \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31
```

### 4. 规则优化

**Makefile**:
```bash
make ts-sr-reversal-rule-optimization \
  SR_OPT_CONFIG=config/strategies/sr_reversal \
  SR_OPT_SYMBOL=BTCUSDT \
  SR_OPT_TIMEFRAME=240T \
  SR_OPT_START=2024-01-01 \
  SR_OPT_END=2025-10-31 \
  SR_OPT_SEARCH_TYPE=random \
  SR_OPT_N_TRIALS=100
```

**mlbot**:
```bash
mlbot optimize rule \
  --strategy-config config/strategies/sr_reversal \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --search-type random \
  --n-trials 100
```

### 5. 模型对比

**Makefile**:
```bash
make ts-sr-reversal-model-comparison \
  SR_COMP_CONFIG=config/strategies/sr_reversal \
  SR_COMP_SYMBOL=BTCUSDT \
  SR_COMP_TIMEFRAME=240T \
  SR_COMP_START=2025-01-01 \
  SR_COMP_END=2025-07-31
```

**mlbot**:
```bash
mlbot diagnose model-comparison \
  --strategy-config config/strategies/sr_reversal \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-07-31
```

### 6. 时间框架前向相关性分析

**Makefile**:
```bash
make ts-timeframe-forward-report \
  SYMBOLS=BTCUSDT,ETHUSDT \
  TF_ANALYSIS_TIMEFRAMES=60T,240T \
  TF_ANALYSIS_FORWARD_BARS=1,3,5,10,20 \
  TF_ANALYSIS_RUN_TAG=run1
```

**mlbot**:
```bash
mlbot analyze timeframe-forward-report \
  --symbols BTCUSDT,ETHUSDT \
  --timeframes 60T,240T \
  --forward-bars 1,3,5,10,20 \
  --run-tag run1
```

### 7. 横截面分析

**Makefile**:
```bash
make cs-build-panel \
  SYMBOLS=BTCUSDT,ETHUSDT \
  START_DATE=2024-01-01 \
  END_DATE=2025-10-31
```

**mlbot**:
```bash
mlbot cross-section build-panel \
  --symbols BTCUSDT,ETHUSDT \
  --start-date 2024-01-01 \
  --end-date 2025-10-31
```

## 参数命名变化

大多数参数从 `UPPER_CASE` (Makefile 变量) 改为 `--kebab-case` (CLI 参数):

- `TS_FACTOR_STRATEGY` → `--strategy-config`
- `TS_FACTOR_SYMBOL` → `--symbol` (或 `-s`)
- `TS_FACTOR_TIMEFRAME` → `--timeframe` (或 `-t`)
- `TS_FACTOR_START` → `--start-date`
- `TS_FACTOR_END` → `--end-date`
- `STRAT_COMPARE_CONFIG` → `--strategy-config`
- `SR_BASELINE_CONFIG` → `--strategy-config`
- `SR_OPT_CONFIG` → `--strategy-config`

## 命令组织

mlbot 命令按功能分组：

- `analyze` - 分析和评估命令
- `diagnose` - 诊断命令
- `optimize` - 优化命令
- `backtest` - 回测命令
- `cross-section` - 横截面分析命令
- `visualize` - 可视化命令
- `train` - 训练命令
- `data` - 数据管理命令
- `features` - 特征管理命令
- `dev` - 开发工具命令

使用 `mlbot <group> --help` 查看每个组的详细命令列表。

## 优势

迁移到 mlbot 的优势：

1. **统一接口**: 所有命令使用相同的参数风格
2. **更好的帮助**: 使用 `--help` 查看详细的参数说明
3. **跨平台**: 不依赖 Make，Windows/Linux/Mac 都可用
4. **清晰的层次**: 命令按功能分组，易于查找
5. **完整的参数控制**: 所有参数都可以通过命令行指定

## 获取帮助

查看所有可用命令：
```bash
mlbot --help
```

查看特定命令组的帮助：
```bash
mlbot analyze --help
mlbot diagnose --help
mlbot optimize --help
```

查看特定命令的详细帮助：
```bash
mlbot analyze factor-eval --help
mlbot diagnose rule-baseline --help
mlbot optimize rule --help
```

