# ---------------------------------------------------------------------------
# ML Trading Project
# Streamlined commands for production workflows
# ---------------------------------------------------------------------------

PYTHON := python3
PIP := pip3

# Common paths (override when invoking make, e.g. `make train-enhanced DATA_DIR=/mnt/parquet_data`)
DATA_DIR ?= data/parquet_data
MODEL_DIR ?= models
RESULTS_DIR ?= results

TRAIN_DATA ?= $(DATA_DIR)/BTCUSDT-aggTrades-2025-05.parquet
OOS_DATA ?= $(DATA_DIR)/BTCUSDT-aggTrades-2025-06.parquet
MODEL_PATH ?= $(MODEL_DIR)/trained_model_enhanced_may_2025.pkl
SCALER_PATH ?= $(MODEL_DIR)/feature_scalers_enhanced_may_2025.pkl
SYMBOL ?= BTCUSDT
YEAR ?= 2024
START_YEAR ?= 2021
END_YEAR ?= 2025

.PHONY: help clean format lint dev-install train-enhanced rolling-monthly rolling-quarterly vectorbot-backtest oos-june dimensionality-demo dimensionality-real

help:
	@echo "ML Trading Project"
	@echo "===================="
	@echo "Core commands:"
	@echo "  make dev-install          # Install project in editable mode"
	@echo "  make train-enhanced       # Train enhanced LightGBM pipeline"
	@echo "  make rolling-monthly      # Monthly rolling retraining"
	@echo "  make rolling-quarterly    # Quarterly rolling retraining"
	@echo "  make vectorbot-backtest   # Run VectorBot risk-managed backtest"
	@echo "  make oos-june             # Evaluate June OOS performance"
	@echo "  make dimensionality-demo  # Run dimensionality pipeline on sample data"
    @echo "  make dimensionality-real  # Run dimensionality pipeline on real data"
    @echo ""
    @echo "Override defaults, e.g. \"make train-enhanced TRAIN_DATA=/data/BTC.parquet SYMBOL=ETHUSDT\""

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

train-enhanced:
    @echo "🚀 Training enhanced production model..."
    PYTHONPATH=src TRAIN_DATA=$(TRAIN_DATA) $(PYTHON) scripts/training/train_model_enhanced.py

rolling-monthly:
	@echo "📆 Running monthly rolling retraining for $(SYMBOL) $(YEAR)..."
	PYTHONPATH=src $(PYTHON) scripts/rolling/monthly_rolling_retrain.py \
		--data-dir $(DATA_DIR) \
		--symbol $(SYMBOL) \
		--year $(YEAR) \
		--output $(RESULTS_DIR)/monthly_rolling_$(SYMBOL)_$(YEAR)

rolling-quarterly:
	@echo "📈 Running quarterly rolling retraining for $(SYMBOL) $(START_YEAR)-$(END_YEAR)..."
	PYTHONPATH=src $(PYTHON) scripts/rolling/quarterly_rolling_retrain.py \
		--data-dir $(DATA_DIR) \
		--symbols $(SYMBOL) \
		--start-year $(START_YEAR) \
		--end-year $(END_YEAR) \
		--output $(RESULTS_DIR)/quarterly_rolling_$(SYMBOL)

vectorbot-backtest:
	@echo "🤖 Running VectorBot backtest with $(MODEL_PATH)..."
	PYTHONPATH=src MODEL_PATH=$(MODEL_PATH) $(PYTHON) scripts/backtesting/vectorbot_backtest.py

oos-june:
	@echo "🧪 Evaluating June OOS performance..."
	PYTHONPATH=src MODEL_PATH=$(MODEL_PATH) SCALER_PATH=$(SCALER_PATH) OOS_DATA=$(OOS_DATA) $(PYTHON) scripts/backtesting/oos_june.py

dimensionality-demo:
	@echo "🌀 Running dimensionality pipeline (sample data)..."
	PYTHONPATH=src $(PYTHON) -m ml_trading.pipeline.dimensionality.pipeline \
		--synthetic-length 5000 \
		--synthetic-factors 120 \
		--encoding-dim 16 \
		--visualize \
		--generate-report





dimensionality-real:
	@echo "🏭 Running dimensionality pipeline on real data..."
	PYTHONPATH=src $(PYTHON) -m ml_trading.pipeline.dimensionality.pipeline \
		--use-real-data \
		--data-path $(DATA_DIR) \
		--symbol $(SYMBOL) \
		--encoding-dim 16 \
		--top-k 40 \
		--save-model \
		--save-topk-model \
		--visualize \
		--generate-report

