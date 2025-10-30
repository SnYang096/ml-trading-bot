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
# ML Trading Project
# Streamlined commands for production workflows
# ---------------------------------------------------------------------------

PYTHON := python3
PIP := pip3

# Docker configuration
DOCKER_COMPOSE := docker-compose
DOCKER_SERVICE := ml-gpu
DOCKER_IMAGE ?= lightgbm-runtime:latest

# Common paths (override when invoking make, e.g. `make train DATA_DIR=/mnt/parquet_data`)
DATA_DIR ?= data/parquet_data
MODEL_DIR ?= models
RESULTS_DIR ?= results

SYMBOL ?= BTCUSDT
SYMBOLS ?= $(SYMBOL)
START_DATE ?= 2025-05-01
END_DATE ?= 2025-05-31
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
	-w /workspace \
	--shm-size=8gb \
	$(DOCKER_IMAGE)

.PHONY: help clean format lint dev-install docker-build docker-install train rolling-monthly rolling-quarterly vectorbot-backtest oos-june dimensionality-demo dimensionality-real

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
	$(DOCKER_RUN) python3 -m ml_trading.pipeline.dimensionality.pipeline \
		--synthetic-length 5000 \
		--synthetic-factors 120 \
		--encoding-dim 16 \
		--visualize \
		--generate-report





dimensionality-real:
	@echo "🏭 Running dimensionality pipeline on real data..."
	$(DOCKER_RUN) python3 -m ml_trading.pipeline.dimensionality.pipeline \
		--use-real-data \
		--data-path $(DATA_DIR) \
		--symbol $(SYMBOL) \
		--encoding-dim 16 \
		--top-k 40 \
		--save-model \
		--save-topk-model \
		--visualize \
		--generate-report

