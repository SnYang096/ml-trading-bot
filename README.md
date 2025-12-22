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

1. **Feature Analysis** (`mlbot analyze factor-eval`): Evaluate factors and select optimal features
2. **Strategy Training** (`mlbot train sr-reversal-long`, `mlbot train sr-reversal-short`, `mlbot train rolling`): Train models using strategy configs
3. **Ablation Study** (`mlbot analyze strategy-feature-compare`): Compare feature configurations (optional)
4. **Backtesting** (`mlbot backtest vectorbot`): Validate strategy performance

**📖 See [完整流程指南](docs/时序模型/完整流程指南.md) (Chinese) for detailed workflow.**

### Step-by-Step Workflow

#### Why SR Reversal is split into Long/Short (and why `sr_reversal/` was removed)

We intentionally use **two direction-fixed strategies**:
- `sr_reversal_long`: long-only, binary label = “a long trade opened at \(t+1\) hits +2R before -1R”
- `sr_reversal_short`: short-only, binary label = “a short trade opened at \(t+1\) hits +2R before -1R”

This replaces the old **bidirectional** config (`sr_reversal/`, `combine_mode: any_success`) which mixed long/short outcomes into a single label.

**Key differences in label semantics**:
- **Old (`any_success`)**: for each bar, evaluate both a hypothetical long and a hypothetical short; label can become “success if either direction would have worked”. This makes `pred` harder to interpret as a trade probability for a specific direction, and often pushes execution to rely on a separate “direction source” (e.g., `signal`).
- **New (`long_only` / `short_only`)**: the label matches exactly one action. `pred` becomes a clean **probability of success for that direction**, so thresholding is stable and comparable across backtest/live.

**Why the split is better**:
- **Cleaner target**: `predict_proba` directly represents “probability this long/short trade succeeds under the RR definition”.
- **Simpler execution**: direction comes from the strategy itself (no `use_signal_direction`, no `signal` required).
- **Better control**: tune thresholds/risk separately for long vs short.
- **Consistency**: same semantics in offline backtests and production inference.

We still keep a lightweight **safety fuse** (optional) to prevent over-trading in OOD/noisy regimes:
`dist_to_nearest_sr / ATR > K  =>  no trade`.

#### Step 0: Verify Feature Correctness (Recommended)

Before starting feature evaluation, run tests to verify that features are computed correctly and don't use future data:

**Quick Test** (Key Features Only):
```bash
make test-key-features-all
```

**Comprehensive Test** (All Features):
```bash
make test-all-features-comprehensive
```

**What These Tests Verify**:
- ✅ No future data leakage (features at time t only use data ≤ t)
- ✅ Multi-asset normalization (features are comparable across different assets)
- ✅ Streaming vs batch consistency (production inference matches training)
- ✅ No global normalization (prevents look-ahead bias)

**Note**: These tests should pass before proceeding to feature evaluation. If tests fail, fix the issues before training models.

**📖 See [测试运行说明](docs/测试运行说明.md) for more details.**

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
  --strategy-config config/strategies/sr_reversal_long \
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

#### Step 2: Feature Ablation Study (Required)

Compare different feature configurations to validate feature selection:

```bash
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml"
```

**With Rolling Windows** (More Robust):
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

**What This Validates**:
- Selected features perform better than all features
- Feature selection improves model generalization
- Feature ablation shows meaningful differences

**Note**: This step is **required** before proceeding to rolling training.

#### Step 3: Model Comparison (Required)

Verify that ML models outperform rule-based strategies:

```bash
mlbot diagnose model-comparison \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31
```

**What This Compares**:
- Rule-based baseline (pure rule strategy)
- ML model (XGBoost/LightGBM)
- ML + Volatility model

**What This Validates**:
- ML model significantly outperforms rules
- ML model provides stable returns
- ML model has reasonable trade frequency

**Note**: This step is **required** before proceeding to rolling training.

#### Step 4: Strategy Training (Optional - For Debugging)

**⚠️ Optional**: Single training is only for debugging or quick configuration testing. For production, proceed directly to Step 5 (Rolling Training) after completing Steps 2 and 3.

**4.1 Quick Validation** (Single Training):

**SR Reversal Long-only**:
```bash
mlbot train sr-reversal-long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-10-31 \
```

**SR Reversal Short-only**:
```bash
mlbot train sr-reversal-short \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-10-31 \
```

#### Step 5: Production Training (Rolling Window - Recommended)
```bash
# Expanding window training: each test month uses all previous months
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

**Note**: Only proceed to rolling training after completing Steps 2 (feature ablation) and Step 3 (model comparison) to ensure features and model architecture are validated.

**With Selected Features** (if you have feature selection results):
```bash
# If you have features_suggested.yaml from factor-eval, update your strategy config to use it
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

#### Viewing HTML Reports in a Dev Container (Cursor / VS Code)

If you are running inside a Dev Container, `--open-browser` may not open the report automatically. Use one of these approaches:

**Option A: Local static server + Port forwarding** (works reliably)

```bash
# Example: serve rolling reports directory
python3 -m http.server 8008 --directory results
```

- In Cursor/VS Code, open the **Ports** panel and forward/open port `8008`.
- Open the report in your local browser, e.g.:
  - `results/auto_rolling_*/monthly_rolling_report.html`
  - `results/strategy_compare/strategy_feature_compare_report.html`
  - `results/rule_optimization/optimization_report.html`

#### Step 6: Periodic Updates (Weekly/Monthly)

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
# Step 0: Verify feature correctness (recommended)
make test-key-features-all

# Step 1: Feature evaluation and selection
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start-date 2025-01-01 \
  --end-date 2025-04-30 \
  --remove-correlated \
  --target-lag 5

# Step 2: Feature ablation study (validate feature selection)
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start-date 2025-01-01 \
  --end-date 2025-04-30 \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml"

# Step 3: Model comparison (verify ML outperforms rules)
mlbot diagnose model-comparison \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start-date 2025-01-01 \
  --end-date 2025-04-30

# Step 5: Rolling window training (main production workflow)
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
| `mlbot analyze factor-eval` | Factor evaluation & selection     | **Recommended**: Primary feature selection method |
| `mlbot train sr-reversal-long/short` | Train single model (direction-fixed) | **Optional**: Debugging / quick checks only |
| `mlbot train rolling`       | Rolling window training           | **Recommended**: Main production workflow |

### Key Points

- **Workflow Order**: Always follow Steps 0 → 1 → 2 → 3 → 5 in sequence
  - Step 0: Verify feature correctness (recommended)
  - Step 1: Feature evaluation (`factor-eval`)
  - Step 2: Feature ablation study (`strategy-feature-compare`) - **Required**
  - Step 3: Model comparison (`model-comparison`) - **Required**
  - Step 5: Rolling training (`rolling`) - **Only after validation**

- `mlbot train sr-reversal-long/short`: Trains **one** model for a single time period (direction-fixed)
  - **Not recommended** for production evaluation
  - Use only for debugging or quick configuration testing

- `mlbot train rolling`: Trains **multiple** models (one per month) in a rolling/expanding window fashion
  - **Required** for production deployment
  - Provides better evaluation through expanding windows
  - Only use after validating features (Step 2) and model performance (Step 3)
  - Step 4 (single training) is optional and only for debugging


## Workflow Summary

### Minimal Workflow (5 Steps, Recommended)

```bash
# Step 0: Verify feature correctness (recommended before starting)
make test-key-features-all

# Step 1: Feature evaluation and selection
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --remove-correlated \
  --filter-by-best-lag

# Step 2: Feature ablation study (validate feature selection)
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml"

# Step 3: Model comparison (verify ML outperforms rules)
mlbot diagnose model-comparison \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# Step 4: Rolling training (only after validation)
# Note: If you have features_suggested.yaml from factor-eval, update your strategy config to use it
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start 2024-01-01 \
  --end 2025-10-31 \
  --initial-train-months 6 \
  --min-train-months 3
```

### Full Workflow (5 Steps with All Options)

```bash
# Step 0: Verify feature correctness (recommended)
make test-all-features-comprehensive

# Step 1: Feature evaluation (generates features_suggested.yaml with selected features)
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --remove-correlated \
  --target-lag 20 \
  --filter-by-best-lag

# Step 2: Feature ablation study (compare original vs selected features)
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

# Step 3: Model comparison (verify ML outperforms rules)
mlbot diagnose model-comparison \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# Step 4: Rolling training (main production workflow - only after validation)
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
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# 2. Rule optimization (find optimal parameters)
mlbot optimize rule \
  --strategy-config config/strategies/sr_reversal_long \
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

## Development Environment

**Recommended: VS Code Dev Container**:
1. Open the project in VS Code
2. Select "Reopen in Container" (automatically enters Dev Container)
3. Run `mlbot` commands directly in the container (no Makefile needed)

**Command Line Usage**:
- In Dev Container: Use `mlbot` commands directly
- In local environment: Use `mlbot` commands (requires `pip install -e .` first)

All `mlbot` commands support `--docker/--no-docker` options to automatically select the appropriate execution method based on the environment.

