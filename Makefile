
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
	train rolling-monthly rolling-quarterly vectorbot-backtest oos-june dimensionality-demo dimensionality-real \
	dim-compare

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
	@echo "  make feature-report       # Generate feature IC/IR HTML report"
	@echo "  make train               # Train production LightGBM pipeline"
	@echo "  make rolling-monthly      # Monthly rolling retraining"
	@echo "  make rolling-quarterly    # Quarterly rolling retraining"
	@echo "  make vectorbot-backtest   # Run VectorBot risk-managed backtest"
	@echo "  make oos-june             # Evaluate June OOS performance"
	@echo "  make dimensionality-demo  # Run dimensionality pipeline on sample data"
	@echo "  make dimensionality-real  # Run dimensionality pipeline on real data"
	@echo "  make dim-compare         # Train full vs compressed/Top-K and compare metrics"
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

train:
	@echo "🚀 Training production model for $(SYMBOLS) ($(START_DATE) → $(END_DATE))..."
	@echo "Example: make train SYMBOLS=\"BTCUSDT ETHUSDT\" START_DATE=2024-10-01 END_DATE=2024-12-31"
	$(DOCKER_RUN) python3 -m ml_trading.models.train_model \
		--symbols $(SYMBOLS) \
		--start-date $(START_DATE) \
		--end-date $(END_DATE) \
		--data-dir $(DATA_DIR) \
		--output-dir $(MODEL_DIR) \
		--model-name $(MODEL_NAME) $(OVERWRITE_FLAG)

train-topk:
	@echo "🚀 Training with Top-K factors for $(SYMBOLS) ($(START_DATE) → $(END_DATE))..."
	@echo "Usage: make train-topk SYMBOLS=BTCUSDT START_DATE=YYYY-MM-DD END_DATE=YYYY-MM-DD TOP_FACTORS=path/to/top_factors.json"
	$(DOCKER_RUN) python3 -m ml_trading.models.train_model \
		--symbols $(SYMBOLS) \
		--start-date $(START_DATE) \
		--end-date $(END_DATE) \
		--data-dir $(DATA_DIR) \
		--output-dir $(MODEL_DIR) \
		--model-name $(MODEL_NAME) \
		--use-top-factors $(TOP_FACTORS) \
		$(OVERWRITE_FLAG)

train-ae:
	@echo "🚀 Training with Autoencoder-compressed features for $(SYMBOLS) ($(START_DATE) → $(END_DATE))..."
	@echo "Usage: make train-ae SYMBOLS=BTCUSDT START_DATE=YYYY-MM-DD END_DATE=YYYY-MM-DD AE_PATH=results/.../production_autoencoder.pth ENCODING_DIM=16"
	$(DOCKER_RUN) python3 -m ml_trading.models.train_model \
		--symbols $(SYMBOLS) \
		--start-date $(START_DATE) \
		--end-date $(END_DATE) \
		--data-dir $(DATA_DIR) \
		--output-dir $(MODEL_DIR) \
		--model-name $(MODEL_NAME) \
		--use-autoencoder $(AE_PATH) \
		--encoding-dim $(ENCODING_DIM) \
		$(OVERWRITE_FLAG)

rolling-monthly:
	@echo "📆 Running monthly rolling retraining for $(SYMBOL) $(YEAR)..."
	$(DOCKER_RUN) python3 scripts/rolling/monthly_rolling_retrain.py \
		--data-dir $(DATA_DIR) \
		--symbol $(SYMBOL) \
		--year $(YEAR) \
		--output $(RESULTS_DIR)/monthly_rolling_$(SYMBOL)_$(YEAR)

rolling-quarterly:
	@echo "📈 Running quarterly rolling retraining for $(SYMBOL) $(START_YEAR)-$(END_YEAR)..."
	$(DOCKER_RUN) python3 scripts/rolling/quarterly_rolling_retrain.py \
		--data-dir $(DATA_DIR) \
		--symbols $(SYMBOL) \
		--start-year $(START_YEAR) \
		--end-year $(END_YEAR) \
		--output $(RESULTS_DIR)/quarterly_rolling_$(SYMBOL)

vectorbot-backtest:
	@echo "🤖 Running VectorBot backtest with $(MODEL_PATH)..."
	$(DOCKER_RUN) bash -c "MODEL_PATH=$(MODEL_PATH) python3 scripts/backtesting/vectorbot_backtest.py"

oos-june:
	@echo "🧪 Evaluating June OOS performance..."
	$(DOCKER_RUN) bash -c "MODEL_PATH=$(MODEL_PATH) SCALER_PATH=$(SCALER_PATH) OOS_DATA=$(OOS_DATA) python3 scripts/backtesting/oos_june.py"

dimensionality-demo:
	@echo "🌀 Running dimensionality pipeline (sample data)..."
	@echo "Example : make dimensionality-demo START_DATE=2024-01-01 END_DATE=2024-06-30"
	$(DOCKER_RUN) python3 -m ml_trading.pipeline.dimensionality.pipeline \
		--n-samples 5000 \
		--n-factors 120 \
		--encoding-dim 16 \
		$(if $(START_DATE),--train-start $(START_DATE)) \
		$(if $(END_DATE),--train-end $(END_DATE)) \
		--visualize \
		--generate-report

dimensionality-real:
	@echo "🏭 Running dimensionality pipeline on real data..."
	@echo "make dimensionality-real DATA_DIR=data/parquet_data SYMBOL=BTCUSDT START_DATE=2024-01-01 END_DATE=2024-12-31"
	$(DOCKER_RUN) python3 -m ml_trading.pipeline.dimensionality.pipeline \
		--use-real-data \
		--data-path /workspace/data/parquet_data \
		--symbol $(SYMBOL) \
		$(if $(START_DATE),--train-start $(START_DATE)) \
		$(if $(END_DATE),--train-end $(END_DATE)) \
		--encoding-dim 16 \
		--top-k 40 \
		--save-model \
		--save-topk-model \
		--visualize \
		--generate-report

# ---------------------------------------------------------------------------
# Dimensionality: production-style comparison (original vs compressed)
# ---------------------------------------------------------------------------

ENCODING_DIM ?= 16
DIM_COMPARE_ARGS ?=

dim-compare:
	@echo "🔬 Comparing original features vs compressed/Top-K for $(SYMBOL) ..."
	@echo "Usage: make dim-compare SYMBOL=BTCUSDT ENCODING_DIM=16 DIM_COMPARE_ARGS=\"--top-k 50\""
	$(DOCKER_RUN) python3 -m ml_trading.pipeline.dimensionality.production_training \
		--data-path /workspace/data/parquet_data \
		--symbol $(SYMBOL) \
		--encoding-dim $(ENCODING_DIM) \
		$(if $(START_DATE),--train-start $(START_DATE)) \
		$(if $(END_DATE),--train-end $(END_DATE)) \
		$(DIM_COMPARE_ARGS)
	@echo "📝 HTML report is saved next to production_results.json (dimensionality_report.html)"

