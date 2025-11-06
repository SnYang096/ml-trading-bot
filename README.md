# ML Trading Bot

This repository hosts the production-ready components for the factor research, dimensionality reduction, model training, and live-trading backtesting stack. The code under `src/ml_trading/` contains the reusable Python package; the `scripts/` directory now only exposes a minimal set of command-line entry points that wrap the package APIs.

## Quick Start

1. Create a virtual environment (conda, venv, etc.) and activate it.
2. Install the project in editable mode:
   ```bash
   pip install -e .[dev]
   ```
3. Verify the install by running the help target:
   ```bash
   make help
   ```

## Recommended Usage Flow

### Core Workflow (3 Commands)

The recommended workflow consists of only 3 commands:

1. **Research** (`make dim-compare`): Find optimal features and compression
2. **Train** (`make train`): Train production model (optional, for single evaluation)
3. **Rolling Update** (`make auto-rolling-update`): Rolling update to latest data (main workflow)

### Step-by-Step Workflow

#### Step 1: Research Dimensionality Reduction

Find optimal features and compression dimension using one quarter of data:

```bash
# Basic: Research dimensionality reduction
make dim-compare SYMBOL=BTCUSDT \
  START_DATE=2025-05-01 END_DATE=2025-07-31 \
  ENCODING_DIM=32

# Enhanced: With VAE and automatic optimization
make dim-compare SYMBOL=BTCUSDT \
  START_DATE=2025-05-01 END_DATE=2025-07-31 \
  AE_TYPE=vae \
  AUTO_ENCODING_GRID=1 \
  AE_AUTO_TUNE=1 \
  AE_TASK_LOSS=1 \
  TASK_WEIGHT=0.1 \
  KL_WEIGHT=1e-3
```

**Enhanced Options**:
- `AE_TYPE=vae`: Use Variational Autoencoder (VAE) instead of standard AE (better latent space)
- `AUTO_ENCODING_GRID=1`: Automatically generate encoding dimensions based on compression ratios
- `AE_AUTO_TUNE=1`: Automatically tune hyperparameters (learning rate, batch size, epochs)
- `AE_TASK_LOSS=1`: Enable task-aware loss (reconstruction + prediction task loss)
- `TASK_WEIGHT=0.1`: Weight for task loss in multi-task training (default: 0.1)
- `KL_WEIGHT=1e-3`: KL divergence weight for VAE (default: 1e-3)

**Output** (in `results/production_dimensionality_20250501_20250731/`):
- `top_factors.json` - Representative features (60-100 features)
- `production_autoencoder.pth` - Best Autoencoder model
- `production_results.json` - Performance comparison
- `dimensionality_report.html` - HTML visualization

**Key Information**:
- Representative features: `top_factors.json`
- Best compression dimension: `production_results.json` → `data_info.stage4_compressed_dim` (e.g., 32)
- Autoencoder model: `production_autoencoder.pth`

#### Step 2: Train Production Model (Optional)

Train a single model using the optimal configuration:

```bash
DIM_DIR=results/production_dimensionality_20250501_20250731

make train SYMBOL=BTCUSDT \
  START_DATE=2025-01-01 END_DATE=2025-07-31 \
  USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32
```

**Note**: This step is **optional** and can be **skipped**. 
- `make train`: Trains **one** model for a single time period (used for one-time evaluation or deployment)
- `make auto-rolling-update`: **Already includes training** - trains **multiple** models (one per month), evaluates model stability over time
- Both commands train models independently - they do **not** share models

**Output**:
- `models/trained_model_*.pkl` - Production model
- `models/trained_model_*_scalers.pkl` - Feature scalers
- `models/trained_model_*_info.json` - Model metadata
- `models/trained_model_*_info_report.html` - HTML report

#### Step 3: Rolling Update (Main Workflow)

Rolling update to latest available data using optimal configuration:

```bash
DIM_DIR=results/production_dimensionality_20250501_20250731

# Rolling update (automatically detects all available data)
make auto-rolling-update SYMBOL=BTCUSDT \
  INITIAL_TRAIN_MONTHS=6 \
  USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32
```

**Note**: `make train` and `make auto-rolling-update` train **independent** models. The model from `make train` is **not** used by `auto-rolling-update`.

**Output**:
- `results/auto_rolling_*/monthly_results.csv` - All months' detailed results
- `results/auto_rolling_*/summary.json` - Summary information
- `results/auto_rolling_*/monthly_rolling_report.html` - HTML report
- `results/auto_rolling_*/model_YYYY-MM.txt` - Model for each month

#### Step 4: Periodic Updates (Weekly/Monthly)

Incremental update from last trained month:

```bash
# Only update new months (from last position)
make auto-rolling-update-only SYMBOL=BTCUSDT \
  OUTPUT=results/auto_rolling_btcusdt_XXX \
  USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32
```

## Data Pipeline

Before training, ensure you have data:

```bash
# Download Binance monthly aggTrades
make data-download DOWNLOAD_SYMBOLS="BTCUSDT ETHUSDT" \
  DOWNLOAD_START_YEAR=2021 DOWNLOAD_START_MONTH=1

# Convert ZIPs to Parquet (5min OHLC + orderflow)
make data-convert

# Or run both in one go
make data-pipeline DOWNLOAD_SYMBOLS="BTCUSDT ETHUSDT"
```

## Core Principle

**All production training should use dimensionality-reduced features** (Top-K + Autoencoder), not the original 482 features.

### Why?

1. **Better Performance**: Reduced features typically perform better (as shown in research)
2. **Faster Training**: Fewer features = faster training
3. **Less Overfitting**: Reduced risk of overfitting
4. **Consistency**: Same feature set as research phase

## Command Comparison

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `make dim-compare` | Research dimensionality reduction | **Required**: Before any training |
| `make train` | Train single model | **Optional**: For single evaluation only |
| `make auto-rolling-update` | Rolling update | **Required**: Main production workflow |

### Key Points

- `make train`: Trains **one** model for a single time period
- `make auto-rolling-update`: **Already includes training** - trains **multiple** models (one per month) in a rolling fashion
- Both commands train models independently - they do **not** share models
- **You can skip `make train`** if you only need rolling update functionality


## Workflow Summary

### Minimal Workflow (2 Commands, Recommended)

```bash
# 1. Research (find optimal configuration)
make dim-compare SYMBOL=BTCUSDT \
  START_DATE=2025-05-01 END_DATE=2025-07-31 \
  ENCODING_DIM=32

DIM_DIR=results/production_dimensionality_20250501_20250731

# 2. Rolling update (trains all models, from history to latest)
make auto-rolling-update SYMBOL=BTCUSDT \
  INITIAL_TRAIN_MONTHS=6 \
  USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32
```

### Full Workflow (3 Commands)

```bash
# 1. Research
make dim-compare SYMBOL=BTCUSDT \
  START_DATE=2025-05-01 END_DATE=2025-07-31 \
  ENCODING_DIM=32

DIM_DIR=results/production_dimensionality_20250501_20250731

# 2. Train single model (optional, for evaluation)
make train SYMBOL=BTCUSDT \
  START_DATE=2025-01-01 END_DATE=2025-07-31 \
  USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32

# 3. Rolling update (main workflow)
make auto-rolling-update SYMBOL=BTCUSDT \
  INITIAL_TRAIN_MONTHS=6 \
  USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32
```

## Documentation

- **`docs/workflow_research_to_production.md`** - Complete workflow documentation
- **`docs/simplified_workflow.md`** - Simplified workflow guide
- **`docs/make_train_vs_dim_compare.md`** - Command comparison guide

## See Also

Run `make help` to see all available commands and their usage.

## 开发环境

开发者 A：用 VS Code 打开项目 → 自动进入 Dev Container → 运行 make train → 直接在容器内高效训练。
开发者 B：用 Vim/命令行 → 先确保镜像存在 → 运行 make train → Makefile 自动拉起容器完成任务。

# TODOs
make train
如需进一步提升，可考虑：
集成 SHAP 解释性
支持动态滚动训练窗口
添加模型版本管理和预测缓存
整体而言，代码质量高，结构清晰，工程实践成熟。