# ML Trading Bot

This repository hosts the production-ready components for the factor research, dimensionality reduction, model training, and live-trading backtesting stack. The code under `src/time_series_model/` contains the reusable Python package; the `scripts/` directory now only exposes a minimal set of command-line entry points that wrap the package APIs.

## Quick Start

1. Create a virtual environment (conda, venv, etc.) and activate it.
2. Install the project in editable mode:
   ```bash
   pip install -e .[dev]
   ```
3. Install Git pre-commit hooks (optional but recommended):
   ```bash
   make install-hooks
   ```
   This will automatically run `mlbot dev format` and `mlbot dev lint` before each commit to ensure code quality.
4. Verify the install by running the help command:
   ```bash
   mlbot --help
   ```
   Or see all available commands:
   ```bash
   mlbot analyze --help
   mlbot train --help
   mlbot diagnose --help
   mlbot optimize --help
   ```

## Recommended Usage Flow

### Core Workflow (Config-Driven Architecture)

The recommended workflow uses **config-driven architecture** with strategy-specific configurations:

1. **Feature Analysis** (`mlbot analyze factor-eval`, `mlbot analyze dim-compare`): Evaluate factors and select optimal features
2. **Strategy Training** (`mlbot train sr-reversal`, `mlbot train rolling`): Train models using strategy configs
3. **Ablation Study** (`mlbot analyze strategy-feature-compare`): Compare feature configurations (optional)
4. **Backtesting** (`mlbot backtest vectorbot`): Validate strategy performance

**📖 See [完整流程指南](docs/时序模型/完整流程指南.md) for detailed workflow.**

### Step-by-Step Workflow

#### Step 1: Feature Analysis

**1.1 Evaluate Individual Factors**:

**Basic Usage**:
```bash
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-10-31 \
```

**Evaluate Specific Factors**:
```bash
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal \
  --symbol BTCUSDT \
  --factors atr sqs_hal_high \
  --timeframe 240T
```

**With Advanced Options** (remove correlated, filter by best lag):
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

**1.2 Feature Selection (Dimensionality Reduction)**:
```bash
# Config-driven feature selection (three-stage pipeline)
mlbot analyze dim-compare \
  --config config/strategies/sr_reversal \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

**Output** (in `results/dim_compare/{strategy}_{symbol}_{timestamp}/`):
- `top_factors.json` - Selected top features (for use in rolling training)
- `results.json` - Performance comparison (before vs after reduction)

**Three-Stage Pipeline**:
1. **Stage 1**: Missing/stability filter (removes >20% missing or low variance)
2. **Stage 2**: IC ranking (selects top features by Information Coefficient)
3. **Stage 3**: Correlation-based selection (removes redundant features)

#### Step 2: Strategy Training

**2.1 Quick Validation** (Single Training):

**SR Reversal (Bidirectional)**:
```bash
mlbot train sr-reversal \
  --config config/strategies/sr_reversal \
  --symbol BTCUSDT \
  --timeframe 15T
```

**SR Reversal Long-only**:
```bash
mlbot train sr-reversal-long \
  --symbol BTCUSDT \
  --timeframe 240T
```

**SR Reversal Short-only**:
```bash
mlbot train sr-reversal-short \
  --symbol BTCUSDT \
  --timeframe 240T
```

**2.2 Production Training** (Rolling Window - Recommended):
```bash
# Expanding window training: each test month uses all previous months
mlbot train rolling \
  --config config/strategies/sr_reversal \
  --symbol BTCUSDT \
  --timeframe 240T \
  --initial-train-months 6 \
  --min-train-months 3
```

**With Date Range**:
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

**Update Only (Incremental)**:
```bash
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --update-only
```

**Output**:
- `results/rolling/{strategy}/{month}/model.pkl` - Models for each month
- `results/rolling/{strategy}/monthly_results.json` - Aggregated results

#### Step 3: Ablation Study (Optional)

Compare different feature configurations to evaluate feature group contributions:

**Basic Usage**:
```bash
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal \
  --symbol BTCUSDT \
  --timeframe 240T
```

**With Feature Overrides**:
```bash
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --feature-overrides "baseline=config/features/baseline.yaml full=config/features/full.yaml"
```

**With Rolling Window Evaluation**:
```bash
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --run-rolling \
  --rolling-train-bars 1000 \
  --rolling-test-bars 200 \
  --rolling-step-bars 100 \
  --rolling-max-windows 10
```

**📖 See [消融实验说明](docs/时序模型/消融实验说明.md) for details.**

#### Step 4: Backtesting

Rolling window training with date range:

```bash
# Rolling training (expanding window: each test month uses all previous months)
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start 2024-01-01 \
  --end 2025-10-31 \
  --initial-train-months 6 \
  --min-train-months 3
```

**Note**: `mlbot train rolling` is the recommended approach for all training scenarios, providing better evaluation through expanding window training.

**With Top Factors** (if you have dimensionality reduction results):
```bash
# If you have top_factors.json from dim-compare, you can use it in strategy config
# Edit your strategy's features.yaml to use the selected features, then:
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --initial-train-months 6
```

**Output**:
- `results/auto_rolling_*/monthly_results.csv` - All months' detailed results
- `results/auto_rolling_*/summary.json` - Summary information
- `results/auto_rolling_*/monthly_rolling_report.html` - HTML report
- `results/auto_rolling_*/model_YYYY-MM.txt` - Model for each month

#### Step 4: Periodic Updates (Weekly/Monthly)

Incremental update from last trained month:

```bash
# Only update new months (from last position) - use --update-only flag
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --update-only
```

This will automatically detect the last trained month and continue from there.

### Complete Workflow Pipeline

For a complete workflow from feature evaluation to rolling training, execute commands in sequence:

```bash
# Step 1: Feature evaluation and dimension reduction
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start-date 2025-01-01 \
  --end-date 2025-04-30 \
  --remove-correlated \
  --target-lag 5

mlbot analyze dim-compare \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start-date 2025-01-01 \
  --end-date 2025-04-30

# Step 2: Quick single model training (optional, for validation)
mlbot train sr-reversal-long \
  --symbol BTCUSDT \
  --timeframe 15T

# Step 3: Rolling window training (main production workflow)
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start 2025-01-01 \
  --end 2025-04-30 \
  --initial-train-months 3 \
  --min-train-months 3
```

> **Tips**  
> - Use `--docker` flag (default) to enable GPU training if available  
> - Review `results/rolling_*/summary.json` for `monthly_results` after training  
> - If `drift detected` appears in results, consider re-evaluating features or adjusting parameters  


## Data Pipeline

Before training, ensure you have data:

```bash
# Download Binance monthly aggTrades
mlbot data download \
  --symbols BTCUSDT,ETHUSDT \
  --start-year 2021 \
  --start-month 1

# Convert ZIPs to Parquet (5min OHLC + orderflow)
mlbot data convert

# Or run both in one go (full pipeline)
mlbot data pipeline \
  --symbols BTCUSDT,ETHUSDT
```

**More Data Pipeline Examples**:

```bash
# Download specific date range
mlbot data download \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --start-year 2024 \
  --start-month 1 \
  --end-year 2025 \
  --end-month 10

# Convert with cleanup (remove ZIP files after conversion)
mlbot data convert --cleanup
```

## Core Principle

**All production training should use dimensionality-reduced features** (Top-K + Autoencoder), not the original 482 features.

### Why?

1. **Better Performance**: Reduced features typically perform better (as shown in research)
2. **Faster Training**: Fewer features = faster training
3. **Less Overfitting**: Reduced risk of overfitting
4. **Consistency**: Same feature set as research phase

## Command Comparison

| Command                     | Purpose                           | When to Use                               |
| --------------------------- | --------------------------------- | ----------------------------------------- |
| `mlbot analyze dim-compare` | Research dimensionality reduction | **Recommended**: Before any training      |
| `mlbot train sr-reversal`   | Train single model                | **Optional**: For single evaluation only  |
| `mlbot train rolling`       | Rolling window training           | **Recommended**: Main production workflow |

### Key Points

- `mlbot train sr-reversal`: Trains **one** model for a single time period
- `mlbot train rolling`: Trains **multiple** models (one per month) in a rolling/expanding window fashion
- Both commands train models independently - they do **not** share models
- **Recommended**: Use `mlbot train rolling` for production as it provides better evaluation through expanding windows


## Workflow Summary

### Minimal Workflow (2 Commands, Recommended)

```bash
# 1. Research (find optimal configuration via feature evaluation)
mlbot analyze dim-compare \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-05-01 \
  --end-date 2025-07-31

# 2. Rolling training (trains all models, from history to latest)
# Note: If you have top_factors.json, update your strategy config to use those features
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start 2024-01-01 \
  --end 2025-10-31 \
  --initial-train-months 6 \
  --min-train-months 3
```

### Full Workflow (3 Commands)

```bash
# 1. Research - Feature evaluation and dimension reduction
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --remove-correlated \
  --target-lag 20

mlbot analyze dim-compare \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-05-01 \
  --end-date 2025-07-31

# 2. Train single model (optional, for quick evaluation)
mlbot train sr-reversal-long \
  --symbol BTCUSDT \
  --timeframe 240T

# 3. Rolling training (main production workflow)
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start 2024-01-01 \
  --end 2025-10-31 \
  --initial-train-months 6 \
  --min-train-months 3
```

### Advanced Workflow Examples

**Diagnostic and Optimization Workflow**:

```bash
# 1. Rule baseline (test pure rule-based strategy)
mlbot diagnose rule-baseline \
  --strategy-config config/strategies/sr_reversal \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# 2. Rule optimization (find optimal parameters)
mlbot optimize rule \
  --strategy-config config/strategies/sr_reversal \
  --symbol BTCUSDT \
  --timeframe 240T \
  --search-type random \
  --n-trials 100

# 3. Generate rule plateau charts
mlbot optimize rule-plateau-charts \
  --results-csv results/rule_optimization/optimization_results.csv \
  --report-html results/rule_optimization/optimization_report.html

# 4. ML parameter sweep
mlbot optimize ml-param-sweep \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-07-31

# 5. Generate ML plateau charts
mlbot optimize ml-plateau-charts \
  --timeframe 240T
```

**Cross-Sectional Analysis Workflow**:

```bash
# 1. Build cross-sectional panel
mlbot cross-section build-panel \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# 2. Generate Fama-MacBeth report
mlbot cross-section report \
  --panel-path data/cross_sectional_panels/panel.parquet \
  --output-dir results/cross_sectional

# 3. Train cross-sectional model
mlbot cross-section train \
  --panel-path data/cross_sectional_panels/panel.parquet

# 4. Auto-select factors
mlbot cross-section select \
  --panel-path data/cross_sectional_panels/panel.parquet \
  --output-path results/cross_sectional/selected_factors.json

# 5. SHAP analysis
mlbot cross-section shap \
  --model-path results/cross_sectional/model.pkl \
  --panel-path data/cross_sectional_panels/panel.parquet

# 6. SHAP drift monitoring
mlbot cross-section shap-drift \
  --model-path results/cross_sectional/model.pkl \
  --panel-path data/cross_sectional_panels/panel.parquet
```

## Documentation

- **`docs/workflow_research_to_production.md`** - Complete workflow documentation
- **`docs/simplified_workflow.md`** - Simplified workflow guide
- **`docs/make_train_vs_dim_compare.md`** - Command comparison guide

## Recent Feature Updates

### New Features (2024-12-19)

#### 1. Liquidity Void Price Impact

**Feature**: `liquidity_void_f`  
**New Output Column**: `liquidity_void_price_impact`

Measures price impact (how much price moves per unit volume):
```
price_impact = (high - low) / volume
```

- **Higher values**: Lower liquidity (small volume moves price more)
- **Lower values**: Higher liquidity (more volume needed to move price)

📖 See [Liquidity Void Price Impact Guide](docs/features/liquidity_void_price_impact_guide.md) for details.

#### 2. LVN Detection Improvements

**Feature**: `footprint_basic_f`  
**Improvement**: Local minimum detection for LVN (Low Volume Node)

- Uses `scipy.signal.find_peaks` to detect local minima
- More accurate than global minimum approach
- Identifies "valleys" between high volume regions

📖 See [LVN Improvements](docs/features/lvn_improvements.md) for details.

#### 3. WPT Enhancements

**Feature**: `wpt_volume_energy_f`  
**New Options**:
- `use_log_returns`: Apply WPT to log returns instead of raw price (removes trend bias)
- `adaptive_window`: ATR-based adaptive window sizing (adapts to volatility)

📖 See [WPT Enhancements](docs/features/wpt_enhancements.md) for details.

## Getting Help

View all available commands:
```bash
mlbot --help
```

View commands for specific categories:
```bash
mlbot analyze --help      # Analysis and evaluation commands
mlbot train --help        # Training commands
mlbot diagnose --help     # Diagnostic commands
mlbot optimize --help     # Optimization commands
mlbot backtest --help     # Backtesting commands
mlbot cross-section --help # Cross-sectional analysis commands
mlbot data --help         # Data management commands
mlbot features --help     # Feature management commands
mlbot dev --help          # Development commands
```

See also:
- [Migration Guide](docs/MIGRATION_GUIDE.md) - Complete guide for migrating from Makefile to mlbot
- [Makefile vs mlbot](docs/MAKEFILE_VS_MLBOT.md) - Command comparison table

## 开发环境

**推荐使用 VS Code Dev Container**：
1. 用 VS Code 打开项目
2. 选择 "Reopen in Container"（自动进入 Dev Container）
3. 在容器内直接运行 `mlbot` 命令（无需通过 Makefile）

**命令行使用**：
- 在 Dev Container 中：直接使用 `mlbot` 命令
- 在本地环境：使用 `mlbot` 命令（需要先 `pip install -e .`）

所有 `mlbot` 命令都支持 `--docker/--no-docker` 选项，可以根据环境自动选择合适的执行方式。

