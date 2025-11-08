
# ---------------------------------------------------------------------------
# ML Trading Project
# Streamlined commands for production workflows
# ---------------------------------------------------------------------------

PYTHON := python3
PIP := pip3

# Detect Dev Container environment (env var or marker file)
INSIDE_FROM_ENV := $(if $(DEV_CONTAINER),yes,no)
INSIDE_FROM_FILE := $(shell if [ -f /.devcontainer-env ]; then echo yes; else echo no; fi)
INSIDE_CONTAINER ?= $(if $(filter yes,$(INSIDE_FROM_ENV) $(INSIDE_FROM_FILE)),yes,no)

# Docker configuration
DOCKER_COMPOSE := docker-compose
DOCKER_SERVICE := ml-gpu
DOCKER_IMAGE ?= hansenlovefiona017/lightgbm-runtime:v0.0.5
BUILDER_IMAGE ?= lightgbm-builder

# Common paths (override when invoking make, e.g. `make train DATA_DIR=data/parquet_data`)
DATA_DIR ?= data/parquet_data
MODEL_DIR ?= models
RESULTS_DIR ?= results

SYMBOL ?= BTCUSDT
SYMBOLS ?= BTCUSDT,ETHUSDT,SOLUSDT,BNB,XRP,ADA,DOGE,DOT,MATIC,SHIB
# SYMBOLS ?= BTCUSDT
START_DATE ?= 2024-11-01
END_DATE ?= 2025-04-30
YEAR ?= 2024
START_YEAR ?= 2021
END_YEAR ?= 2025

SYMBOL_LOWER := $(shell echo $(SYMBOL) | tr '[:upper:]' '[:lower:]')
START_TAG := $(subst -,,$(START_DATE))
END_TAG := $(subst -,,$(END_DATE))
MODEL_NAME ?= trained_model

MODEL_PATH ?= $(MODEL_DIR)/$(MODEL_NAME)_$(SYMBOL_LOWER)_$(START_TAG)_$(END_TAG).pkl
SCALER_PATH ?= $(MODEL_DIR)/$(MODEL_NAME)_$(SYMBOL_LOWER)_$(START_TAG)_$(END_TAG)_scalers.pkl
OOS_DATA ?= $(DATA_DIR)/$(SYMBOL)-aggTrades-2025-06.parquet
OVERWRITE ?= 0
OVERWRITE_FLAG := $(if $(filter 1 true yes,$(OVERWRITE)),--overwrite,)

# ---------------------------------------------------------------------------
# Training configuration (simple names) + backward-compatible BASELINE_* aliases
# ---------------------------------------------------------------------------
FREQ ?= 15T
FREQS ?= 15T,60T,240T
CV_FOLDS ?= 5
OOS_MONTHS ?= 4
OOS_START ?=
OOS_END ?=
INITIAL_TRAIN_MONTHS ?= 6
MIN_TRAIN_MONTHS ?= 3
FBS ?= 1,3,5

# Optional rolling window month bounds
ROLLING_START ?=
ROLLING_END ?=

# Docker command template (mounts volumes and sets PYTHONPATH)
ifeq ($(INSIDE_CONTAINER),yes)
DOCKER_RUN :=
DOCKER_RUN_NO_TTY :=
else
DOCKER_RUN := docker run --rm -it \
	--runtime=nvidia \
	-e NVIDIA_VISIBLE_DEVICES=all \
	-e CUDA_VISIBLE_DEVICES=0 \
	-e PYTHONPATH=/workspace/src \
	-e PYTHONUNBUFFERED=1 \
	-v $(PWD):/workspace \
	-v $(PWD)/data/parquet_data:/workspace/data/parquet_data \
	-w /workspace \
	--shm-size=8gb \
	$(DOCKER_IMAGE)

DOCKER_RUN_NO_TTY := docker run --rm \
	--runtime=nvidia \
	-e NVIDIA_VISIBLE_DEVICES=all \
	-e CUDA_VISIBLE_DEVICES=0 \
	-e PYTHONPATH=/workspace/src \
	-e PYTHONUNBUFFERED=1 \
	-v $(PWD):/workspace \
	-v $(PWD)/data/parquet_data:/workspace/data/parquet_data \
	-w /workspace \
	--shm-size=8gb \
	$(DOCKER_IMAGE)
endif


.PHONY: help clean format lint dev-install docker-build docker-install builder-shell \
	data-download data-convert data-pipeline \
	train train-quantile tune-q50-params rolling rolling-multi rolling-update-only vectorbot-backtest \
		dim-compare nautilus-backtest feature-report factor-analysis

help:
	@echo "ML Trading Project"
	@echo "===================="
	@echo "Local development commands (run on host):"
	@echo "  make dev-install          # Install project in editable mode"
	@echo "  make format               # Format code with black"
	@echo "  make lint                 # Lint code with flake8"
	@echo ""
	@echo "Docker setup commands:"
	@echo "  make docker-build         # Build Docker image (lightgbm-runtime:latest)"
	@echo "  make docker-install       # Install project inside Docker container"
	@echo "  make builder-shell        # Open bash in $(BUILDER_IMAGE)"
	@echo ""
	@echo "Data commands:"
	@echo "  make data-download       # Download Binance aggTrades ZIPs (non-interactive)"
	@echo "  make data-convert        # Convert ZIPs to Parquet (5min OHLC + orderflow)"
	@echo "  make data-pipeline       # Download then convert"
	@echo ""
	@echo "Training/ML commands (run in Docker):"
	@echo "  Core Workflow (Recommended):"
	@echo "    make train              # Train classification model (default: binary + return regression + volatility)"
	@echo "    make train-quantile     # Train quantile regression model (q10, q50, q90 + volatility)"
	@echo "    make train-quantile PARAMS_FILE=results/params/q50_params_*.json  # Use pre-trained parameters"
	@echo "    make train-quantile AUTO_TUNE=1 TUNE_TRIALS=20  # Auto-tune hyperparameters"
	@echo "    make tune-q50-params    # Pre-train Q50 parameter search (for quantile models)"
	@echo "    make rolling            # Rolling training to latest data (main workflow)"
	@echo ""
	@echo "  Data commands:"
	@echo "    make data-download     # Download Binance data"
	@echo "    make data-convert      # Convert ZIPs to Parquet"
	@echo "    make data-pipeline     # Download then convert"
	@echo ""
	@echo "  Other commands:"
	@echo "    make feature-report    # Generate feature IC/IR HTML report"
	@echo "    make factor-analysis   # Factor effectiveness analysis using Alphalens (IC, quantile backtest, decay)"
	@echo "    make vectorbot-backtest # Run VectorBot risk-managed backtest"
	@echo "    make nautilus-backtest  # Run Nautilus Trader backtest"
	@echo ""
	@echo ""
	@echo "Override defaults, e.g. \"make train SYMBOLS=\"BTCUSDT ETHUSDT\" START_DATE=2024-10-01 END_DATE=2024-12-31\""
	@echo ""
	@echo "Note: Training commands run in Docker. Make sure Docker image is built: make docker-build"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

format:
	PYTHONPATH=src $(PYTHON) -m black src/ml_trading/ tests/ scripts/

lint:
	PYTHONPATH=src $(PYTHON) -m flake8 src/ tests/ scripts/

dev-install:
	$(PIP) install -e .

docker-build:
	@echo "🔨 Building Docker image $(DOCKER_IMAGE)..."
	docker build -f docker/Dockerfile.gpu -t $(DOCKER_IMAGE) .
	

docker-install:
	@echo "📦 Installing project inside Docker container..."
	$(DOCKER_RUN) pip3 install -e /workspace

# ---------------------------------------------------------------------------
# Data: download Binance monthly aggTrades ZIPs and convert to Parquet
# ---------------------------------------------------------------------------

# Download configuration
AGG_DATA_DIR ?= data/agg_data
DOWNLOAD_SYMBOLS ?= $(SYMBOLS)
comma := ,
DOWNLOAD_SYMBOLS_LIST := $(strip $(subst $(comma), ,$(DOWNLOAD_SYMBOLS)))
DOWNLOAD_START_YEAR ?= 2020
DOWNLOAD_START_MONTH ?= 1
DOWNLOAD_END_YEAR ?= $(shell date +%Y)
DOWNLOAD_END_MONTH ?= $(shell date +%m)

data-download:
	@echo "📥 Downloading Binance monthly aggTrades ZIPs to $(AGG_DATA_DIR) ..."
	@echo "Symbols=$(DOWNLOAD_SYMBOLS) Range=$(DOWNLOAD_START_YEAR)-$(DOWNLOAD_START_MONTH) → $(DOWNLOAD_END_YEAR)-$(DOWNLOAD_END_MONTH)"
	@mkdir -p $(AGG_DATA_DIR)
	# Non-interactive confirm: auto-continue (downloads directly into agg_data)
	@yes | $(PYTHON) scripts/utils/download_training_data.py \
		--data-dir $(AGG_DATA_DIR) \
		--parquet-dir $(DATA_DIR) \
		$(if $(DOWNLOAD_SYMBOLS_LIST),--symbols $(DOWNLOAD_SYMBOLS_LIST)) \
		--start-year $(DOWNLOAD_START_YEAR) \
		--start-month $(DOWNLOAD_START_MONTH) \
		--end-year $(DOWNLOAD_END_YEAR) \
		--end-month $(DOWNLOAD_END_MONTH)

data-convert:
	@echo "🔄 Converting ZIPs under data/agg_data → Parquet under data/parquet_data ..."
	$(PYTHON) scripts/data_conversion/convert_zip_to_parquet.py --cleanup yes

data-pipeline:
	@$(MAKE) data-download \
		DOWNLOAD_SYMBOLS="$(DOWNLOAD_SYMBOLS)" \
		DOWNLOAD_START_YEAR=$(DOWNLOAD_START_YEAR) \
		DOWNLOAD_START_MONTH=$(DOWNLOAD_START_MONTH) \
		DOWNLOAD_END_YEAR=$(DOWNLOAD_END_YEAR) \
		DOWNLOAD_END_MONTH=$(DOWNLOAD_END_MONTH)
	@$(MAKE) data-convert

builder-shell:
	@echo "🔧 Opening interactive shell in $(BUILDER_IMAGE) ..."
	DOCKER_IMAGE=$(BUILDER_IMAGE) $(DOCKER_RUN) bash

# ---------------------------------------------------------------------------
# Feature diagnostics
# ---------------------------------------------------------------------------

FEATURE_REPORT_INPUT ?= data/parquet_data/BTC-USD_2024-10.parquet
FEATURE_REPORT_OUTPUT ?= reports/feature_report.html
FEATURE_REPORT_START ?=
FEATURE_REPORT_END ?=
FEATURE_REPORT_HORIZON ?= 1
FEATURE_REPORT_ARGS ?=
# Example:
#   make feature-report FEATURE_REPORT_INPUT=data/parquet_data/ETH-USD_2024-10.parquet \
#                       FEATURE_REPORT_OUTPUT=reports/eth_report.html \
#                       FEATURE_REPORT_START=2024-10-01 FEATURE_REPORT_END=2024-12-31 \
#                       FEATURE_REPORT_HORIZON=3 \
#                       FEATURE_REPORT_ARGS="--no-enhanced --no-dl"

feature-report:
	@echo "📊 Generating feature IC report (Docker) ..."
	@mkdir -p $(dir $(FEATURE_REPORT_OUTPUT))
	$(DOCKER_RUN_NO_TTY) python3 scripts/analysis/feature_quality_report.py \
		--input $(FEATURE_REPORT_INPUT) \
		--output $(FEATURE_REPORT_OUTPUT) \
		--future-horizon $(FEATURE_REPORT_HORIZON) \
		$(if $(FEATURE_REPORT_START),--start-date $(FEATURE_REPORT_START)) \
		$(if $(FEATURE_REPORT_END),--end-date $(FEATURE_REPORT_END)) \
		$(FEATURE_REPORT_ARGS)

# ---------------------------------------------------------------------------
# Factor analysis using Alphalens
# ---------------------------------------------------------------------------

FACTOR_ANALYSIS_OUTPUT_DIR ?= results/factor_analysis
FACTOR_ANALYSIS_PERIODS ?= 1,4,24
FACTOR_ANALYSIS_QUANTILES ?= 10
FACTOR_ANALYSIS_FACTOR_NAME ?=
FACTOR_ANALYSIS_FEATURE_TYPE ?= baseline

factor-analysis:
	@echo "📊 Factor effectiveness analysis using Alphalens for $(SYMBOLS) ($(START_DATE) → $(END_DATE))..."
	@echo "Example: make factor-analysis SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT START_DATE=2024-10-01 END_DATE=2024-12-31"
	@echo "       Symbols: $(SYMBOLS) (comma-separated for multi-asset analysis)"
	@echo "       Feature Type: $(FACTOR_ANALYSIS_FEATURE_TYPE)"
	@echo "       Periods: $(FACTOR_ANALYSIS_PERIODS) bars (e.g., 1,4,24 for 15min, 1h, 6h prediction)"
	@echo "       Quantiles: $(FACTOR_ANALYSIS_QUANTILES) (for Top vs Bottom analysis)"
	@if [ -n "$(FACTOR_ANALYSIS_FACTOR_NAME)" ]; then \
		echo "       Factor Name: $(FACTOR_ANALYSIS_FACTOR_NAME)"; \
	fi
	$(DOCKER_RUN_NO_TTY) python3 scripts/analysis/factor_analysis_alphalens.py \
		$(if $(shell echo $(START_DATE) | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}$$'),--start $(shell echo $(START_DATE) | cut -c1-7),) \
		$(if $(shell echo $(END_DATE) | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}$$'),--end $(shell echo $(END_DATE) | cut -c1-7),) \
		--data-dir /workspace/$(DATA_DIR) \
		--symbol $(SYMBOLS) \
		--freq $(FREQ) \
		--feature-type $(FACTOR_ANALYSIS_FEATURE_TYPE) \
		--output-dir /workspace/$(FACTOR_ANALYSIS_OUTPUT_DIR) \
		--periods $(FACTOR_ANALYSIS_PERIODS) \
		--quantiles $(FACTOR_ANALYSIS_QUANTILES) \
		$(if $(FACTOR_ANALYSIS_FACTOR_NAME),--factor-name $(FACTOR_ANALYSIS_FACTOR_NAME),)

FORWARD_BARS_TRAIN ?= 5,15,45,288

TRAIN_USE_TOP_FACTORS ?=
TRAIN_FEATURE_TYPE ?= baseline
TRAIN_TOPK ?=
TRAIN_TOPK_SOURCE ?=
# Model type: classification (default) or quantile
MODEL_TYPE ?= classification
DIRECTION_THRESHOLD ?= f1_optimize
SAFE_MULTI_ASSET ?= 1

# Auto-tune hyperparameters
AUTO_TUNE ?= 0
TUNE_TRIALS ?= 20
PARAMS_FILE ?=

tune-q50-params:
	@echo "🔍 Pre-training Q50 parameter search for $(SYMBOLS) ($(START_DATE) → $(END_DATE))..."
	@echo "Example: make tune-q50-params SYMBOLS=BTCUSDT,ETHUSDT START_DATE=2024-11-01 END_DATE=2025-04-30"
	@echo "       Symbols: $(SYMBOLS) (comma-separated)"
	@echo "       Timeframe: $(FREQ)"
	@echo "       Forward Bars: $(FORWARD_BARS_TRAIN)"
	@echo "       Trials: $(TUNE_TRIALS)"
	@mkdir -p results/params
	$(DOCKER_RUN_NO_TTY) python3 scripts/optimization/tune_q50_params.py \
		$(if $(shell echo $(START_DATE) | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}$$'),--start $(shell echo $(START_DATE) | cut -c1-7),) \
		$(if $(shell echo $(END_DATE) | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}$$'),--end $(shell echo $(END_DATE) | cut -c1-7),) \
		--data-dir /workspace/$(DATA_DIR) \
		--symbol $(SYMBOLS) \
		--freq $(FREQ) \
		--forward-bars $(shell echo $(FORWARD_BARS_TRAIN) | cut -d',' -f1) \
		--n-trials $(TUNE_TRIALS) \
		--n-splits 3 \
		--max-files 10

train:
	@echo "🚀 Training classification model (default) for $(SYMBOLS) ($(START_DATE) → $(END_DATE))..."
	@echo "Example: make train SYMBOLS=BTCUSDT,ETHUSDT START_DATE=2024-11-01 END_DATE=2025-04-01"
	@echo "       Model Type: $(MODEL_TYPE) (classification: binary + return regression + volatility)"
	@echo "       Forward Bars: $(FORWARD_BARS_TRAIN) bars ahead for prediction"
	@echo "       Symbols: $(SYMBOLS) (comma-separated for multi-asset training)"
	@if [ "$(SAFE_MULTI_ASSET)" = "1" ]; then \
		echo "       🔒 Safe Multi-Asset: Enabled (each symbol processed independently)"; \
	fi
	@if [ -n "$(PARAMS_FILE)" ]; then \
		echo "       📂 Using pre-trained parameters from: $(PARAMS_FILE)"; \
	fi
	$(DOCKER_RUN_NO_TTY) python3 -m ml_trading.pipeline.training.train \
		$(if $(shell echo $(START_DATE) | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}$$'),--start $(shell echo $(START_DATE) | cut -c1-7),) \
		$(if $(shell echo $(END_DATE) | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}$$'),--end $(shell echo $(END_DATE) | cut -c1-7),) \
        --data-dir /workspace/$(DATA_DIR) \
		--symbol $(SYMBOLS) \
		--freq $(FREQS) \
        --forward-bars $(FORWARD_BARS_TRAIN) \
		--cv-folds $(CV_FOLDS) \
		--feature-type $(TRAIN_FEATURE_TYPE) \
		$(if $(TRAIN_USE_TOP_FACTORS),--use-top-factors $(TRAIN_USE_TOP_FACTORS),) \
		$(if $(TRAIN_TOPK),--topk $(TRAIN_TOPK),) \
		$(if $(TRAIN_TOPK_SOURCE),--topk-source $(TRAIN_TOPK_SOURCE),) \
		$(if $(filter 1,$(SAFE_MULTI_ASSET)),--safe-multi-asset,) \
		$(if $(filter 1,$(AUTO_TUNE)),--auto-tune-params,) \
		$(if $(TUNE_TRIALS),--tune-trials $(TUNE_TRIALS),) \
		$(if $(PARAMS_FILE),--params-file /workspace/$(PARAMS_FILE),) \
		--model-type $(MODEL_TYPE) \
		--oos-months $(OOS_MONTHS) \
		$(if $(OOS_START),--oos-start $(OOS_START),) \
		$(if $(OOS_END),--oos-end $(OOS_END),) \
		--disable-target-winsorize \
  	--disable-feature-winsorize \
        --gpu


FORWARD_BARS ?= 3

ROLLING_FREQ ?= $(FREQ)
ROLLING_FBS ?= $(FORWARD_BARS)
ROLLING_OUTPUT ?=
ROLLING_FEATURE_TYPE ?= $(TRAIN_FEATURE_TYPE)
ROLLING_USE_TOP_FACTORS ?=
ROLLING_TOPK ?=
ROLLING_TOPK_SOURCE ?=
ROLLING_USE_AUTOENCODER ?=
ROLLING_ENCODING_DIM ?=

rolling:
	@echo "🔄 Rolling training (regression) for $(SYMBOLS) tf=$(ROLLING_FREQ) fb=$(ROLLING_FBS)"
	@echo "       Symbols: $(SYMBOLS) (comma-separated for multi-asset training)"
	@if [ -n "$(ROLLING_USE_AUTOENCODER)" ]; then \
		echo "       Autoencoder: $(ROLLING_USE_AUTOENCODER) (dim=$(ROLLING_ENCODING_DIM))"; \
	fi
	$(DOCKER_RUN_NO_TTY) python3 -m ml_trading.pipeline.training.rolling \
		--data-dir /workspace/$(DATA_DIR) \
		--symbol $(SYMBOLS) \
		$(if $(ROLLING_START),--start $(ROLLING_START),) \
		$(if $(ROLLING_END),--end $(ROLLING_END),) \
		--initial-train-months $(INITIAL_TRAIN_MONTHS) \
		--min-train-months $(MIN_TRAIN_MONTHS) \
		--freq $(ROLLING_FREQ) \
		--forward-bars $(ROLLING_FBS) \
		--cv-folds $(CV_FOLDS) \
		$(if $(ROLLING_OUTPUT),--output $(ROLLING_OUTPUT),) \
		--feature-type $(ROLLING_FEATURE_TYPE) \
		$(if $(ROLLING_USE_TOP_FACTORS),--use-top-factors $(ROLLING_USE_TOP_FACTORS),) \
		$(if $(ROLLING_TOPK),--topk $(ROLLING_TOPK),) \
		$(if $(ROLLING_TOPK_SOURCE),--topk-source $(ROLLING_TOPK_SOURCE),) \
		$(if $(ROLLING_USE_AUTOENCODER),--use-autoencoder $(ROLLING_USE_AUTOENCODER),) \
		$(if $(ROLLING_ENCODING_DIM),--encoding-dim $(ROLLING_ENCODING_DIM),) \
		--gpu

rolling-multi:
	@echo "🔄 Rolling training (multi-config) for $(SYMBOLS) tfs=$(FREQS) fbs=$(FBS)"
	@echo "       Symbols: $(SYMBOLS) (comma-separated for multi-asset training)"
	@if [ -n "$(ROLLING_USE_AUTOENCODER)" ]; then \
		echo "       Autoencoder: $(ROLLING_USE_AUTOENCODER) (dim=$(ROLLING_ENCODING_DIM))"; \
	fi
	@if [ "$(INSIDE_CONTAINER)" = "yes" ]; then \
		FB_LIST=$(FBS) python3 -m ml_trading.pipeline.training.rolling \
		--data-dir /workspace/$(DATA_DIR) \
		--symbol $(SYMBOLS) \
		$(if $(ROLLING_START),--start $(ROLLING_START),) \
		$(if $(ROLLING_END),--end $(ROLLING_END),) \
		--initial-train-months $(INITIAL_TRAIN_MONTHS) \
		--min-train-months $(MIN_TRAIN_MONTHS) \
		$(foreach tf,$(subst ,, $(FREQS)),--freq $(tf)) \
		--cv-folds $(CV_FOLDS) \
		--feature-type $(ROLLING_FEATURE_TYPE) \
		$(if $(ROLLING_USE_TOP_FACTORS),--use-top-factors $(ROLLING_USE_TOP_FACTORS),) \
		$(if $(ROLLING_TOPK),--topk $(ROLLING_TOPK),) \
		$(if $(ROLLING_TOPK_SOURCE),--topk-source $(ROLLING_TOPK_SOURCE),) \
		$(if $(ROLLING_USE_AUTOENCODER),--use-autoencoder $(ROLLING_USE_AUTOENCODER),) \
		$(if $(ROLLING_ENCODING_DIM),--encoding-dim $(ROLLING_ENCODING_DIM),) \
		--gpu; \
	else \
		docker run --rm \
			--runtime=nvidia \
			-e NVIDIA_VISIBLE_DEVICES=all \
			-e CUDA_VISIBLE_DEVICES=0 \
			-e PYTHONPATH=/workspace/src \
			-e PYTHONUNBUFFERED=1 \
			-e FB_LIST=$(FBS) \
			-v $(PWD):/workspace \
			-v $(PWD)/data/parquet_data:/workspace/data/parquet_data \
			-w /workspace \
			--shm-size=8gb \
			$(DOCKER_IMAGE) python3 -m ml_trading.pipeline.training.rolling \
		--data-dir /workspace/$(DATA_DIR) \
		--symbol $(SYMBOLS) \
		$(if $(ROLLING_START),--start $(ROLLING_START),) \
		$(if $(ROLLING_END),--end $(ROLLING_END),) \
		--initial-train-months $(INITIAL_TRAIN_MONTHS) \
		--min-train-months $(MIN_TRAIN_MONTHS) \
		$(foreach tf,$(subst ,, $(FREQS)),--freq $(tf)) \
		--cv-folds $(CV_FOLDS) \
		--feature-type $(ROLLING_FEATURE_TYPE) \
		$(if $(ROLLING_USE_TOP_FACTORS),--use-top-factors $(ROLLING_USE_TOP_FACTORS),) \
		$(if $(ROLLING_TOPK),--topk $(ROLLING_TOPK),) \
		$(if $(ROLLING_TOPK_SOURCE),--topk-source $(ROLLING_TOPK_SOURCE),) \
		$(if $(ROLLING_USE_AUTOENCODER),--use-autoencoder $(ROLLING_USE_AUTOENCODER),) \
		$(if $(ROLLING_ENCODING_DIM),--encoding-dim $(ROLLING_ENCODING_DIM),) \
		--gpu; \
	fi

BACKTEST_START ?=$(START_DATE)
BACKTEST_END ?=$(END_DATE)
BACKTEST_SYMBOL ?=$(SYMBOL)
BACKTEST_MODEL ?=$(MODEL_PATH)

vectorbot-backtest:
	@echo "🤖 Running VectorBot backtest with model=$(BACKTEST_MODEL) symbol=$(BACKTEST_SYMBOL) range=$(BACKTEST_START)→$(BACKTEST_END) ..."
	$(DOCKER_RUN_NO_TTY) bash -c "python3 scripts/backtesting/vectorbot_backtest.py \
		$(if $(BACKTEST_MODEL),--model '$(BACKTEST_MODEL)') \
		$(if $(BACKTEST_SYMBOL),--symbol '$(BACKTEST_SYMBOL)') \
		$(if $(BACKTEST_START),--start '$(BACKTEST_START)') \
		$(if $(BACKTEST_END),--end '$(BACKTEST_END)')"

nautilus-backtest:
	@echo "⛵ Running Nautilus AE+LGB backtest (host env, requires nautilus-trader installed)..."
	PYTHONPATH=src $(PYTHON) scripts/backtesting/nautilus_dim_backtest.py \
		--data-dir $(DATA_DIR) \
		--results-dir $(RESULTS_DIR)/$(NAUTILUS_RESULTS_DIR) \
		--symbols $(SYMBOLS) \
		--timeframe 5T \
		--start $(START_DATE) --end $(END_DATE)




# ---------------------------------------------------------------------------
# Dimensionality: production-style comparison (original vs compressed)
# ---------------------------------------------------------------------------

DIM_COMPARE_ARGS ?=
HORIZONS ?= 1,5,10,15
BINARY_SIGNALS ?= 1
LABEL_THRESHOLD ?= 0.0
DIM_COMPARE_FEATURE_TYPE ?= comprehensive

# make dim-compare SYMBOL=BTCUSDT \
#   START_DATE=2025-05-01 END_DATE=2025-07-31 \
#   AE_TYPE=vae \
#   AUTO_ENCODING_GRID=1 \
#   AE_AUTO_TUNE=1 \
#   AE_TASK_LOSS=1 \
#   TASK_WEIGHT=0.1 \
#   KL_WEIGHT=1e-3 \
#   TUNE_TRIALS=15

dim-compare:
	@echo "🔬 Comparing feature sets (no autoencoder) for $(SYMBOLS) ..."
	@echo "Usage: make dim-compare SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT HORIZONS=1,5,10,15 DIM_COMPARE_FEATURE_TYPE=baseline"
	@echo "       Multi-horizon training enabled: $(HORIZONS)"
	@echo "       Feature type: $(DIM_COMPARE_FEATURE_TYPE)"
	@echo "       Symbols: $(SYMBOLS) (comma-separated for multi-asset training)"
	$(DOCKER_RUN_NO_TTY) python3 -m ml_trading.pipeline.dimensionality.dimensionality_comparison \
		--data-path /workspace/data/parquet_data \
		--symbol $(SYMBOLS) \
		--feature-type $(DIM_COMPARE_FEATURE_TYPE) \
		$(if $(START_DATE),--train-start $(START_DATE)) \
		$(if $(END_DATE),--train-end $(END_DATE)) \
		$(if $(HORIZONS),--horizons $(HORIZONS)) \
		$(DIM_COMPARE_ARGS)
	@echo "📝 HTML report saved with: {SYMBOL}_{FEATURE_TYPE}_{START_DATE}_{END_DATE}_dimensionality_report.html"


