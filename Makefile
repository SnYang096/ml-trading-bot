
# ---------------------------------------------------------------------------
# ML Trading Project
# Streamlined commands for production workflows
# ---------------------------------------------------------------------------

PYTHON := python3
PIP := pip3

# Docker configuration
DOCKER_COMPOSE := docker-compose
DOCKER_SERVICE := ml-gpu
DOCKER_IMAGE ?= hansenlovefiona017/lightgbm-runtime:v0.0.3
BUILDER_IMAGE ?= lightgbm-builder

# Common paths (override when invoking make, e.g. `make train DATA_DIR=data/parquet_data`)
DATA_DIR ?= data/parquet_data
MODEL_DIR ?= models
RESULTS_DIR ?= results

SYMBOL ?= BTCUSDT
SYMBOLS ?= $(SYMBOL)
START_DATE ?= 2025-05-01
END_DATE ?= 2025-07-31
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

# Docker command template (mounts volumes and sets PYTHONPATH)
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


.PHONY: help clean format lint dev-install docker-build docker-install builder-shell \
	data-download data-convert data-pipeline \
	train auto-rolling-update auto-rolling-update-only vectorbot-backtest oos-june \
		dim-compare nautilus-backtest feature-report \
		baseline-train baseline-rolling baseline-rolling-multi

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
	@echo "    make dim-compare        # Step 1: Research dimensionality reduction (find optimal features)"
	@echo "    make train              # Step 2: Train production model (optional, for single evaluation)"
	@echo "    make auto-rolling-update # Step 3: Rolling update to latest data (main workflow)"
	@echo ""
	@echo "  Data commands:"
	@echo "    make data-download     # Download Binance data"
	@echo "    make data-convert      # Convert ZIPs to Parquet"
	@echo "    make data-pipeline     # Download then convert"
	@echo ""
	@echo "  Other commands:"
	@echo "    make feature-report    # Generate feature IC/IR HTML report"
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
DOWNLOAD_START_YEAR ?= 2021
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
		$(if $(DOWNLOAD_SYMBOLS),--symbols $(DOWNLOAD_SYMBOLS)) \
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

FORWARD_BARS_TRAIN ?= 1

train:
	@echo "🚀 Training production model for $(SYMBOLS) ($(START_DATE) → $(END_DATE))..."
	@echo "Example: make train SYMBOLS=\"BTCUSDT ETHUSDT\" START_DATE=2024-10-01 END_DATE=2024-12-31 FORWARD_BARS_TRAIN=5"
	@echo "       Forward Bars (Horizon): $(FORWARD_BARS_TRAIN) bars ahead for prediction"
	$(DOCKER_RUN) python3 -m ml_trading.models.train_model \
		--symbols $(SYMBOLS) \
		--start-date $(START_DATE) \
		--end-date $(END_DATE) \
		--data-dir $(DATA_DIR) \
		--output-dir $(MODEL_DIR) \
		--model-name $(MODEL_NAME) $(OVERWRITE_FLAG) \
		--overwrite \
		--forward-bars $(FORWARD_BARS_TRAIN)


FORWARD_BARS ?= 3

auto-rolling-update:
	@echo "🚀 Auto Rolling Update: Train and update $(SYMBOL) to latest available data..."
	@echo "Usage: make auto-rolling-update SYMBOL=BTCUSDT INITIAL_TRAIN_MONTHS=6 FORWARD_BARS=5"
	@echo "       make auto-rolling-update SYMBOL=BTCUSDT USE_TOP_FACTORS=path/to/top_factors.json USE_AUTOENCODER=path/to/ae.pth ENCODING_DIM=32 FORWARD_BARS=15"
	@echo "       Forward Bars (Horizon): $(FORWARD_BARS) bars ahead for prediction"
	$(DOCKER_RUN_NO_TTY) python3 -m ml_trading.pipeline.rolling.auto_rolling_update \
		--data-dir $(DATA_DIR) \
		--symbol $(SYMBOL) \
		--initial-train-months $(if $(INITIAL_TRAIN_MONTHS),$(INITIAL_TRAIN_MONTHS),6) \
		$(if $(MIN_TRAIN_MONTHS),--min-train-months $(MIN_TRAIN_MONTHS),) \
		$(if $(OUTPUT),--output $(OUTPUT),) \
		$(if $(filter 1 true yes,$(ADD_ORDER_FLOW)),--add-order-flow,) \
		$(if $(filter 1 true yes,$(UPDATE_ONLY)),--update-only,) \
		$(if $(USE_TOP_FACTORS),--use-top-factors $(USE_TOP_FACTORS),) \
		$(if $(USE_AUTOENCODER),--use-autoencoder $(USE_AUTOENCODER) --encoding-dim $(ENCODING_DIM),) \
		--forward-bars $(FORWARD_BARS)

auto-rolling-update-only:
	@echo "🔄 Auto Rolling Update: Only update $(SYMBOL) from last trained month..."
	@echo "Usage: make auto-rolling-update-only SYMBOL=BTCUSDT OUTPUT=results/auto_rolling_btcusdt_XXX FORWARD_BARS=5"
	@echo "       make auto-rolling-update-only SYMBOL=BTCUSDT OUTPUT=results/XXX USE_TOP_FACTORS=path/to/top_factors.json USE_AUTOENCODER=path/to/ae.pth ENCODING_DIM=32 FORWARD_BARS=15"
	@echo "       Forward Bars (Horizon): $(FORWARD_BARS) bars ahead for prediction"
	$(DOCKER_RUN_NO_TTY) python3 -m ml_trading.pipeline.rolling.auto_rolling_update \
		--data-dir $(DATA_DIR) \
		--symbol $(SYMBOL) \
		--update-only \
		$(if $(OUTPUT),--output $(OUTPUT),) \
		$(if $(filter 1 true yes,$(ADD_ORDER_FLOW)),--add-order-flow,) \
		$(if $(USE_TOP_FACTORS),--use-top-factors $(USE_TOP_FACTORS),) \
		$(if $(USE_AUTOENCODER),--use-autoencoder $(USE_AUTOENCODER) --encoding-dim $(ENCODING_DIM),) \
		--forward-bars $(FORWARD_BARS)

vectorbot-backtest:
	@echo "🤖 Running VectorBot backtest with $(MODEL_PATH)..."
	$(DOCKER_RUN) bash -c "MODEL_PATH=$(MODEL_PATH) python3 scripts/backtesting/vectorbot_backtest.py"

nautilus-backtest:
	@echo "⛵ Running Nautilus AE+LGB backtest (host env, requires nautilus-trader installed)..."
	PYTHONPATH=src $(PYTHON) scripts/backtesting/nautilus_dim_backtest.py \
		--data-dir $(DATA_DIR) \
		--results-dir $(RESULTS_DIR)/$(NAUTILUS_RESULTS_DIR) \
		--symbols $(SYMBOLS) \
		--timeframe 5T \
		--start $(START_DATE) --end $(END_DATE)

oos-june:
	@echo "🧪 Evaluating June OOS performance..."
	$(DOCKER_RUN) bash -c "MODEL_PATH=$(MODEL_PATH) SCALER_PATH=$(SCALER_PATH) OOS_DATA=$(OOS_DATA) python3 scripts/backtesting/oos_june.py"


# ---------------------------------------------------------------------------
# Dimensionality: production-style comparison (original vs compressed)
# ---------------------------------------------------------------------------

ENCODING_DIM ?= 16
AE_TYPE ?= production
DIM_COMPARE_ARGS ?=
HORIZONS ?= 1,5,10,15
BINARY_SIGNALS ?= 1
LABEL_THRESHOLD ?= 0.0

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
	@echo "🔬 Comparing original features vs compressed/Top-K for $(SYMBOL) ..."
	@echo "Usage: make dim-compare SYMBOL=BTCUSDT ENCODING_DIM=16 HORIZONS=1,5,10,15"
	@echo "       Enhanced options: AE_TYPE=vae AUTO_ENCODING_GRID=1 AE_AUTO_TUNE=1 AE_TASK_LOSS=1 BINARY_SIGNALS=$(BINARY_SIGNALS) LABEL_THRESHOLD=$(LABEL_THRESHOLD)"
	@echo "       Multi-horizon training enabled: $(HORIZONS)"
	$(DOCKER_RUN_NO_TTY) python3 -m ml_trading.pipeline.dimensionality.dimensionality_comparison \
		--data-path /workspace/data/parquet_data \
		--symbol $(SYMBOL) \
		--encoding-dim $(ENCODING_DIM) \
		$(if $(ENCODING_GRID),--encoding-grid $(ENCODING_GRID)) \
		$(if $(AE_TYPE),--ae-type $(AE_TYPE)) \
		$(if $(KL_WEIGHT),--kl-weight $(KL_WEIGHT)) \
		$(if $(filter 1 true yes,$(AUTO_ENCODING_GRID)),--auto-encoding-grid) \
		$(if $(filter 1 true yes,$(AE_AUTO_TUNE)),--ae-auto-tune) \
		$(if $(TUNE_TRIALS),--tune-trials $(TUNE_TRIALS)) \
		$(if $(filter 1 true yes,$(AE_TASK_LOSS)),--ae-task-loss) \
		--binary-signals \
		$(if $(LABEL_THRESHOLD),--label-threshold $(LABEL_THRESHOLD)) \
		$(if $(TASK_WEIGHT),--task-weight $(TASK_WEIGHT)) \
		$(if $(START_DATE),--train-start $(START_DATE)) \
		$(if $(END_DATE),--train-end $(END_DATE)) \
		$(if $(HORIZONS),--horizons $(HORIZONS)) \
		$(DIM_COMPARE_ARGS)
	@echo "📝 HTML report is saved next to production_results.json (dimensionality_report.html)"

# ---------------------------------------------------------------------------
# Baseline: SR + Compression features only (single + rolling)
# Defaults aligned with dim-compare (HORIZONS/START_DATE/END_DATE)
# ---------------------------------------------------------------------------

# Multi-config defaults
BASELINE_FREQS ?= 5T
BASELINE_FBS ?= $(HORIZONS)
BASELINE_START ?= $(shell echo $(START_DATE) | cut -c1-7)
BASELINE_END ?= $(shell echo $(END_DATE) | cut -c1-7)

# Single-config defaults derive from multi-config
BASELINE_FREQ ?= $(word 1,$(subst ,, ,$(BASELINE_FREQS)))
BASELINE_FB ?= $(word 1,$(subst ,, ,$(BASELINE_FBS)))

# CV defaults
BASELINE_CV_FOLDS ?= 0
BASELINE_CV_ON_ROLLING ?= 0
INITIAL_TRAIN_MONTHS ?= 6
MIN_TRAIN_MONTHS ?= 3

.PHONY: baseline-train baseline-rolling baseline-rolling-multi

baseline-train:
	@echo "🧱 Baseline single training (SR+Compression) with GPU: $(SYMBOL) tf=$(BASELINE_FREQ) fb=$(BASELINE_FB)"
	PYTHONPATH=src $(PYTHON) -m ml_trading.pipeline.baseline.train_baseline \
		$(if $(BASELINE_START),--start $(BASELINE_START),) \
		$(if $(BASELINE_END),--end $(BASELINE_END),) \
		--data-dir $(DATA_DIR) \
		--symbol $(SYMBOL) \
		--freq $(BASELINE_FREQ) \
		--forward-bars $(BASELINE_FB) \
		--cv-folds $(BASELINE_CV_FOLDS) \
		--gpu

baseline-rolling:
	@echo "🔄 Baseline rolling (SR+Compression) with GPU: $(SYMBOL) tf=$(BASELINE_FREQ) fb=$(BASELINE_FB)"
	PYTHONPATH=src $(PYTHON) -m ml_trading.pipeline.baseline.rolling_baseline \
		--data-dir $(DATA_DIR) \
		--symbol $(SYMBOL) \
		$(if $(BASELINE_START),--start $(BASELINE_START),) \
		$(if $(BASELINE_END),--end $(BASELINE_END),) \
		--initial-train-months $(INITIAL_TRAIN_MONTHS) \
		--min-train-months $(MIN_TRAIN_MONTHS) \
		--freq $(BASELINE_FREQ) \
		--forward-bars $(BASELINE_FB) \
		--cv-folds $(BASELINE_CV_FOLDS) \
		$(if $(filter 1 true yes,$(BASELINE_CV_ON_ROLLING)),--cv-on-rolling,) \
		--gpu

# Multi-config: support multiple timeframes and horizons
# - Timeframes via BASELINE_FREQS (comma-separated or repeated in CLI)
# - Horizons via BASELINE_FBS (comma-separated), passed by FB_LIST env var
baseline-rolling-multi:
	@echo "🔄 Baseline rolling (multi-config) with GPU: $(SYMBOL) tfs=$(BASELINE_FREQS) fbs=$(BASELINE_FBS)"
	FB_LIST=$(BASELINE_FBS) PYTHONPATH=src $(PYTHON) -m ml_trading.pipeline.baseline.rolling_baseline \
		--data-dir $(DATA_DIR) \
		--symbol $(SYMBOL) \
		$(if $(BASELINE_START),--start $(BASELINE_START),) \
		$(if $(BASELINE_END),--end $(BASELINE_END),) \
		--initial-train-months $(INITIAL_TRAIN_MONTHS) \
		--min-train-months $(MIN_TRAIN_MONTHS) \
		$(foreach tf,$(subst ,, $(BASELINE_FREQS)),--freq $(tf)) \
		--cv-folds $(BASELINE_CV_FOLDS) \
		$(if $(filter 1 true yes,$(BASELINE_CV_ON_ROLLING)),--cv-on-rolling,) \
		--gpu

