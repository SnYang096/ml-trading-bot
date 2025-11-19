
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
# SYMBOLS ?= BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,DOTUSDT
SYMBOLS ?= BTCUSDT,ETHUSDT
START_DATE ?= 2025-1-01
END_DATE ?= 2025-10-30
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
FREQ ?= 240T
FREQS ?= 15T,60T,240T
CV_FOLDS ?= 5
OOS_MONTHS ?= 4
OOS_START ?=
OOS_END ?=
INITIAL_TRAIN_MONTHS ?= 3
MIN_TRAIN_MONTHS ?= 3
FBS ?= 1,3,5

# Optional rolling window month bounds
ROLLING_START ?= 2024-11-01
ROLLING_END ?= 2025-04-30

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


.PHONY: help clean format lint dev-install install-hooks docker-build docker-install builder-shell \
	data-download data-convert data-pipeline \
	train train-quantile tune-q50-params rolling rolling-multi rolling-update-only vectorbot-backtest \
		dim-compare nautilus-backtest factor-analysis \
		timeframe-forward-report feature-indicators \
	cross-sectional-catalog \
	cross-sectional-build-panel cross-sectional-report cross-sectional-train cross-sectional-workflow

help:
	@echo "ML Trading Project"
	@echo "===================="
	@echo "Local development commands (run on host):"
	@echo "  make dev-install          # Install project in editable mode"
	@echo "  make install-hooks        # Install Git pre-commit hooks (run make format & lint before commit)"
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
	@echo "    make rolling            # Rolling training (recommended for production)"
	@echo "    make tune-q50-params    # Pre-train Q50 parameter search (for quantile models)"
	@echo ""
	@echo "  Data commands:"
	@echo "    make data-download     # Download Binance data"
	@echo "    make data-convert      # Convert ZIPs to Parquet"
	@echo "    make data-pipeline     # Download then convert"
	@echo ""
	@echo "  Other commands:"
	@echo "    make ts-r-rank-ic-train # Rank IC regression training (TSCV + OOS testing)"
	@echo "    make factor-analysis   # Factor effectiveness analysis using Alphalens (IC, quantile backtest, decay)"
	@echo "    make cross-sectional-build-panel  # Generate multi-asset factor panels for CS modelling"
	@echo "    make cross-sectional-report  # Fama-MacBeth + Newey-West + IC/IR markdown report"
	@echo "    make cross-sectional-train   # Train cross-sectional models (boosting/Fama-MacBeth)"
	@echo "    make cross-sectional-workflow# Build panel + report + train in one go"
	@echo "    make cross-sectional-catalog # Categorise factors from an existing panel"
	@echo "    make vectorbot-backtest # Run VectorBot risk-managed backtest"
	@echo "    make nautilus-backtest  # Run Nautilus Trader backtest"
	@echo ""
	@echo ""
	@echo "Override defaults, e.g. \"make rolling SYMBOLS=\"BTCUSDT ETHUSDT\" ROLLING_START=2024-10 ROLLING_END=2024-12\""
	@echo ""
	@echo "Note: Training commands run in Docker. Make sure Docker image is built: make docker-build"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

format:
	PYTHONPATH=src $(PYTHON) -m black src/time_series_model/ src/cross_sectional/ src/data_tools/ tests/ scripts/

lint:
	PYTHONPATH=src $(PYTHON) -m flake8 src/time_series_model/ src/cross_sectional/ src/data_tools/ tests/ scripts/

dev-install:
	$(PIP) install -e .

install-hooks:
	@echo "📦 Installing Git hooks..."
	@bash scripts/install-git-hooks.sh

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
# Factor Management: Test and compute specific factors
# ---------------------------------------------------------------------------

FACTOR_TEST_FACTORS ?=
FACTOR_TEST_SYMBOL ?= BTCUSDT
FACTOR_TEST_START_DATE ?= 2024-01-01
FACTOR_TEST_END_DATE ?= 2025-9-30
FACTOR_TEST_FEATURE_TYPE ?= comprehensive
FACTOR_TEST_TIMEFRAME ?= 240T
FACTOR_TEST_OUTPUT_DIR ?=

factor-test:
	@if [ -z "$(FACTOR_TEST_FACTORS)" ]; then \
		echo "❌ 错误: 必须指定 FACTOR_TEST_FACTORS"; \
		echo "用法: make factor-test FACTOR_TEST_FACTORS='rsi_7 zigzag_normalized' FACTOR_TEST_SYMBOL=BTCUSDT"; \
		exit 1; \
	fi
	@echo "🧪 测试因子: $(FACTOR_TEST_FACTORS)"
	@echo "   交易对: $(FACTOR_TEST_SYMBOL)"
	@echo "   时间范围: $(FACTOR_TEST_START_DATE) 到 $(FACTOR_TEST_END_DATE)"
	$(DOCKER_RUN_NO_TTY) python3 scripts/factor_management/test_single_factor.py \
		--factors $(FACTOR_TEST_FACTORS) \
		--data-path /workspace/$(DATA_DIR) \
		--symbol $(FACTOR_TEST_SYMBOL) \
		--start-date $(FACTOR_TEST_START_DATE) \
		--end-date $(FACTOR_TEST_END_DATE) \
		--feature-type $(FACTOR_TEST_FEATURE_TYPE) \
		--timeframe $(FACTOR_TEST_TIMEFRAME) \
		$(if $(FACTOR_TEST_OUTPUT_DIR),--output-dir /workspace/$(FACTOR_TEST_OUTPUT_DIR),)

FACTOR_COMPUTE_FACTORS ?=
FACTOR_COMPUTE_INPUT ?=
FACTOR_COMPUTE_DATA_PATH ?= $(DATA_DIR)
FACTOR_COMPUTE_SYMBOL ?=
FACTOR_COMPUTE_START_DATE ?=
FACTOR_COMPUTE_END_DATE ?=
FACTOR_COMPUTE_OUTPUT ?= results/factors/computed_factors.csv
FACTOR_COMPUTE_FEATURE_TYPE ?= comprehensive
FACTOR_COMPUTE_FORMAT ?= csv

factor-compute:
	@if [ -z "$(FACTOR_COMPUTE_FACTORS)" ]; then \
		echo "❌ 错误: 必须指定 FACTOR_COMPUTE_FACTORS"; \
		echo "用法: make factor-compute FACTOR_COMPUTE_FACTORS='rsi_7 macd' FACTOR_COMPUTE_INPUT=data/btcusdt.parquet FACTOR_COMPUTE_OUTPUT=factors/rsi_macd.csv"; \
		exit 1; \
	fi
	@echo "🔧 计算因子: $(FACTOR_COMPUTE_FACTORS)"
	@echo "   输出: $(FACTOR_COMPUTE_OUTPUT)"
	$(DOCKER_RUN_NO_TTY) python3 scripts/factor_management/compute_specific_factors.py \
		--factors $(FACTOR_COMPUTE_FACTORS) \
		$(if $(FACTOR_COMPUTE_INPUT),--input /workspace/$(FACTOR_COMPUTE_INPUT),) \
		$(if $(FACTOR_COMPUTE_SYMBOL),--data-path /workspace/$(FACTOR_COMPUTE_DATA_PATH),) \
		$(if $(FACTOR_COMPUTE_SYMBOL),--symbol $(FACTOR_COMPUTE_SYMBOL),) \
		$(if $(FACTOR_COMPUTE_START_DATE),--start-date $(FACTOR_COMPUTE_START_DATE),) \
		$(if $(FACTOR_COMPUTE_END_DATE),--end-date $(FACTOR_COMPUTE_END_DATE),) \
		--output /workspace/$(FACTOR_COMPUTE_OUTPUT) \
		--feature-type $(FACTOR_COMPUTE_FEATURE_TYPE) \
		--format $(FACTOR_COMPUTE_FORMAT)

# ---------------------------------------------------------------------------
# Alphalens test (verify installation and basic functionality)
# ---------------------------------------------------------------------------

test-alphalens:
	@echo "🧪 Testing Alphalens installation and basic functionality in Docker..."
	$(DOCKER_RUN_NO_TTY) python3 scripts/test_alphalens.py

alphalens-example:
	@echo "📊 Running complete Alphalens example with comprehensive analysis..."
	$(DOCKER_RUN_NO_TTY) python3 scripts/alphalens_example.py

alphalens-evaluate:
	@echo "📊 Evaluating trading signal quality using Alphalens..."
	$(DOCKER_RUN_NO_TTY) python3 scripts/alphalens_evaluate_predictions.py

# ---------------------------------------------------------------------------
# Factor analysis using Alphalens （跑不起来）
# ---------------------------------------------------------------------------


TF_CONFIG_PEARSON ?= 0.03
TF_CONFIG_PVALUE ?= 1e-5
TF_CONFIG_MIN_SAMPLES ?= 500
TF_CONFIG_TOP_PER_SYMBOL ?= 5
TF_CONFIG_TOP_PER_GROUP ?= 10

TRAIN_FEATURE_TYPE ?= baseline
DIRECTION_THRESHOLD ?= f1_optimize


FACTOR_ANALYSIS_OUTPUT_DIR ?= results/factor_analysis
FACTOR_ANALYSIS_PERIODS ?= 24
FACTOR_ANALYSIS_QUANTILES ?= 10
FACTOR_ANALYSIS_FACTOR_NAME ?=
FACTOR_ANALYSIS_FEATURE_TYPE ?= baseline

factor-analysis:
	@echo "📊 Factor effectiveness analysis using Alphalens for $(SYMBOLS) ($(START_DATE) → $(END_DATE))..."
	@echo "Example: make factor-analysis SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT START_DATE=2024-10-01 END_DATE=2024-12-31 FREQ=15T"
	@echo "       Symbols: $(SYMBOLS) (comma-separated for multi-asset analysis)"
	@echo "       Timeframe: $(FREQ) (override with FREQ=5T,15T,60T,240T, etc.)"
	@echo "       Feature Type: $(FACTOR_ANALYSIS_FEATURE_TYPE)"
	@echo "       Periods: $(FACTOR_ANALYSIS_PERIODS) bars (e.g., 1,4,24 for 15min, 1h, 6h prediction)"
	@echo "       Quantiles: $(FACTOR_ANALYSIS_QUANTILES) (for Top vs Bottom analysis)"
	@if [ -n "$(FACTOR_ANALYSIS_FACTOR_NAME)" ]; then \
		echo "       Factor Name: $(FACTOR_ANALYSIS_FACTOR_NAME)"; \
	fi
	@echo "       Note: Alphalens frequency warnings are expected for intraday data (workaround applied)"
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

timeframe-forward-report:
	@echo "🧮 Analysing timeframe and forward bar correlations for $(SYMBOLS)..."
	@mkdir -p $(TF_ANALYSIS_OUTPUT_DIR)
	SYMBOLS_SPACE="$(shell echo $(SYMBOLS) | tr ',' ' ')" ; \
	TIMEFRAMES_SPACE="$(shell echo $(TF_ANALYSIS_TIMEFRAMES) | tr ',' ' ')" ; \
	FORWARD_BARS_SPACE="$(shell echo $(TF_ANALYSIS_FORWARD_BARS) | tr ',' ' ')" ; \
	$(DOCKER_RUN_NO_TTY) python3 -m time_series_model.analysis.timeframe_forward_correlation \
		--data-dir /workspace/$(DATA_DIR) \
		--output-dir /workspace/$(TF_ANALYSIS_OUTPUT_DIR) \
		--symbols $$SYMBOLS_SPACE \
		--timeframes $$TIMEFRAMES_SPACE \
		--forward-bars $$FORWARD_BARS_SPACE \
		$(if $(TF_ANALYSIS_START),--start $(TF_ANALYSIS_START),) \
		$(if $(TF_ANALYSIS_END),--end $(TF_ANALYSIS_END),) \
		--max-lag $(TF_ANALYSIS_MAX_LAG) \
		--min-samples $(TF_ANALYSIS_MIN_SAMPLES) \
		--top-k $(TF_ANALYSIS_TOP_K) \
		--feature-type $(TF_ANALYSIS_FEATURE_TYPE) \
		$(if $(TF_ANALYSIS_EXTRA_FEATURES),--extra-features $(TF_ANALYSIS_EXTRA_FEATURES),) \
		$(if $(TF_ANALYSIS_RUN_TAG),--run-tag $(TF_ANALYSIS_RUN_TAG),)
	@echo "✅ Timeframe correlation report saved under $(TF_ANALYSIS_OUTPUT_DIR)"
ifneq ($(TF_ANALYSIS_RUN_TAG),)
	@TF_RUN_DIR="$(TF_ANALYSIS_OUTPUT_DIR)/$(TF_ANALYSIS_RUN_TAG)" ; \
	if [ ! -f "$$TF_RUN_DIR/timeframe_forward_details.csv" ]; then \
		echo "❌ Cannot find $$TF_RUN_DIR/timeframe_forward_details.csv -- did timeframe-forward-report succeed?"; \
		exit 1; \
	fi
	@echo "🧾 Building strategy configuration from $(TF_ANALYSIS_RUN_TAG)..."
	if ! $(DOCKER_RUN_NO_TTY) python3 -m time_series_model.analysis.timeframe_feature_selector \
		--details-csv "/workspace/$(TF_ANALYSIS_OUTPUT_DIR)/$(TF_ANALYSIS_RUN_TAG)/timeframe_forward_details.csv" \
		--output-dir "/workspace/$(TF_ANALYSIS_OUTPUT_DIR)/$(TF_ANALYSIS_RUN_TAG)/config" \
		--pearson-threshold $(TF_CONFIG_PEARSON) \
		--pvalue-threshold $(TF_CONFIG_PVALUE) \
		--min-samples $(TF_CONFIG_MIN_SAMPLES) \
		--top-features-per-symbol $(TF_CONFIG_TOP_PER_SYMBOL) \
		--top-features-per-group $(TF_CONFIG_TOP_PER_GROUP); then \
		echo "⚠️ No strategy groups generated (likely thresholds too strict)."; \
	fi
	@if [ -d "$(TF_ANALYSIS_OUTPUT_DIR)/$(TF_ANALYSIS_RUN_TAG)/config" ]; then \
		echo "✅ Strategy configs written to $(TF_ANALYSIS_OUTPUT_DIR)/$(TF_ANALYSIS_RUN_TAG)/config"; \
	else \
		echo "ℹ️ Strategy configuration directory not created."; \
	fi
endif

# ---------------------------------------------------------------------------
# Dimensionality: Three-stage feature selection (before vs after reduction)
# ---------------------------------------------------------------------------

DIM_COMPARE_ARGS ?=
HORIZONS ?= 24
DIM_COMPARE_FEATURE_TYPE ?= baseline
DIM_COMPARE_TIMEFRAME ?= 60T
DIM_COMPARE_VALIDATE_PIPELINE ?= true
DIM_COMPARE_REPORT_HTML ?=
DIM_COMPARE_EXPORT_MODEL ?=

dim-compare:
	@echo "🔬 Dimensionality Reduction Comparison for $(SYMBOLS) ..."
	@echo "Usage: make dim-compare SYMBOLS=BTCUSDT,ETHUSDT HORIZONS=1,5,10,15 DIM_COMPARE_FEATURE_TYPE=comprehensive DIM_COMPARE_TIMEFRAME=5T"
	@echo "       This runs three-stage feature selection:"
	@echo "         Stage 1: Missing/stability filter (removes >20%% missing or low variance)"
	@echo "         Stage 2: IC ranking (selects top features by Information Coefficient)"
	@echo "         Stage 3: Correlation-based representative selection (removes redundant features)"
	@echo ""
	@echo "       Configuration:"
	@echo "         Multi-horizon training: $(HORIZONS)"
	@echo "         Feature type: $(DIM_COMPARE_FEATURE_TYPE)"
	@echo "         Timeframe: $(DIM_COMPARE_TIMEFRAME)"
	@echo "         Symbols: $(SYMBOLS) (comma-separated for multi-asset training)"
	@if [ "$(START_DATE)" != "" ]; then \
		echo "         Training period: $(START_DATE) → $(END_DATE)"; \
	fi
	@if [ "$(DIM_COMPARE_VALIDATE_PIPELINE)" = "1" ] || [ "$(DIM_COMPARE_VALIDATE_PIPELINE)" = "true" ]; then \
		echo "         Pipeline validation: enabled (synthetic signal injection)"; \
	else \
		echo "         Pipeline validation: disabled"; \
	fi
	@echo "         Stability validation: $(if $(ENABLE_STABILITY_VALIDATION),enabled,disabled)"
	@if [ "$(ENABLE_STABILITY_VALIDATION)" = "1" ] || [ "$(ENABLE_STABILITY_VALIDATION)" = "true" ]; then \
		echo "         Validation start: $(VALIDATION_START)"; \
		echo "         Validation years: $(VALIDATION_YEARS)"; \
	fi
	@if [ "$(DIM_COMPARE_REPORT_HTML)" != "" ]; then \
		echo "         HTML report: $(DIM_COMPARE_REPORT_HTML)"; \
	fi
	@if [ "$(DIM_COMPARE_EXPORT_MODEL)" != "" ]; then \
		echo "         Export model: $(DIM_COMPARE_EXPORT_MODEL)"; \
	fi
	$(DOCKER_RUN_NO_TTY) python3 -m time_series_model.pipeline.dimensionality.dimensionality_comparison \
		--data-path /workspace/data/parquet_data \
		--symbol $(SYMBOLS) \
		--feature-type $(DIM_COMPARE_FEATURE_TYPE) \
		--timeframe $(DIM_COMPARE_TIMEFRAME) \
		$(if $(START_DATE),--train-start $(START_DATE)) \
		$(if $(END_DATE),--train-end $(END_DATE)) \
		--horizons $(HORIZONS) \
		$(if $(filter true 1,$(DIM_COMPARE_VALIDATE_PIPELINE)),--validate-pipeline) \
		$(if $(ENABLE_STABILITY_VALIDATION),--enable-stability-validation) \
		$(if $(VALIDATION_START),--validation-start $(VALIDATION_START)) \
		$(if $(VALIDATION_YEARS),--validation-years $(VALIDATION_YEARS)) \
		$(if $(DIM_COMPARE_REPORT_HTML),--report-html /workspace/$(DIM_COMPARE_REPORT_HTML)) \
		$(if $(DIM_COMPARE_EXPORT_MODEL),--export-model /workspace/$(DIM_COMPARE_EXPORT_MODEL)) \
		$(DIM_COMPARE_ARGS)


# ---------------------------------------------------------------------------
# Feature Indicators Visualization
# ---------------------------------------------------------------------------

FEATURE_INDICATORS_OUTPUT ?= results/feature_indicators/$(SYMBOL)_$(TIMEFRAME).html
FEATURE_INDICATORS_FEATURE_TYPES ?= hurst,hilbert,wavelet,spectral

feature-indicators:
	@echo "📈 Generating feature indicators visualization for $(SYMBOL)..."
	@echo "   Timeframe: $(TIMEFRAME)"
	@echo "   Feature types: $(FEATURE_INDICATORS_FEATURE_TYPES)"
	@echo "   Output: $(FEATURE_INDICATORS_OUTPUT)"
	@mkdir -p $(dir $(FEATURE_INDICATORS_OUTPUT))
	$(DOCKER_RUN_NO_TTY) python3 scripts/visualization/feature_indicator_visualizer.py \
		--data-path /workspace/data/parquet_data \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--feature-types $(FEATURE_INDICATORS_FEATURE_TYPES) \
		--feature-type comprehensive \
		$(if $(START_DATE),--start-date $(START_DATE)) \
		$(if $(END_DATE),--end-date $(END_DATE)) \
		--output $(FEATURE_INDICATORS_OUTPUT)
	@echo "✅ Feature indicators visualization saved to $(FEATURE_INDICATORS_OUTPUT)"



# ---------------------------------------------------------------------------
# Timeframe/forward selection analysis
# ---------------------------------------------------------------------------

TF_ANALYSIS_TIMEFRAMES ?= 15T,30T,60T,120T,240T
TF_ANALYSIS_FORWARD_BARS ?= 3,6,12,24
TF_ANALYSIS_START ?= $(START_DATE)
TF_ANALYSIS_END ?= $(END_DATE)
TF_ANALYSIS_MAX_LAG ?= 5
TF_ANALYSIS_MIN_SAMPLES ?= 500
TF_ANALYSIS_TOP_K ?= 5
TF_ANALYSIS_FEATURE_TYPE ?= baseline
TF_ANALYSIS_EXTRA_FEATURES ?=
TF_ANALYSIS_RUN_TAG ?=
TF_ANALYSIS_OUTPUT_DIR ?= results/timeframe_forward

# Auto-tune hyperparameters
AUTO_TUNE ?= 0
TUNE_TRIALS ?= 20
PARAMS_FILE ?=


FORWARD_BARS ?= 3

ROLLING_FREQ ?= $(FREQ)
ROLLING_FBS ?= $(FORWARD_BARS)
DIRECTION_THRESHOLD ?= f1_optimize
ROLLING_OUTPUT ?=
ROLLING_FEATURE_TYPE ?= $(TRAIN_FEATURE_TYPE)
ROLLING_USE_TOP_FACTORS ?=
ROLLING_TOPK ?=
ROLLING_TOPK_SOURCE ?=


rolling:
	@echo "🔄 Rolling training (regression) for $(SYMBOLS) tf=$(ROLLING_FREQ) fb=$(ROLLING_FBS)"
	@echo "       Symbols: $(SYMBOLS) (comma-separated for multi-asset training)"
	$(DOCKER_RUN_NO_TTY) python3 -m time_series_model.pipeline.training.rolling \
		--data-dir /workspace/$(DATA_DIR) \
		--symbol $(SYMBOLS) \
		$(if $(ROLLING_START),--start $(ROLLING_START),) \
		$(if $(ROLLING_END),--end $(ROLLING_END),) \
		--initial-train-months $(INITIAL_TRAIN_MONTHS) \
		--min-train-months $(MIN_TRAIN_MONTHS) \
		--freq $(ROLLING_FREQ) \
		--forward-bars $(ROLLING_FBS) \
		--cv-folds $(CV_FOLDS) \
		$(if $(filter-out 0,$(CV_FOLDS)),--cv-on-rolling,) \
		$(if $(ROLLING_OUTPUT),--output $(ROLLING_OUTPUT),) \
		--feature-type $(ROLLING_FEATURE_TYPE) \
		$(if $(ROLLING_USE_TOP_FACTORS),--use-top-factors $(ROLLING_USE_TOP_FACTORS),) \
		$(if $(ROLLING_TOPK),--topk $(ROLLING_TOPK),) \
		$(if $(ROLLING_TOPK_SOURCE),--topk-source $(ROLLING_TOPK_SOURCE),) \
		--direction-threshold $(DIRECTION_THRESHOLD) \
		--gpu


BACKTEST_START ?=$(START_DATE)
BACKTEST_END ?=$(END_DATE)
BACKTEST_SYMBOL ?=$(SYMBOL)
BACKTEST_MODEL ?=$(MODEL_PATH)
vectorbot-backtest:
	@echo "🤖 Running VectorBot backtest with model=$(BACKTEST_MODEL) symbol=$(BACKTEST_SYMBOL) range=$(BACKTEST_START)→$(BACKTEST_END) ..."
	$(DOCKER_RUN_NO_TTY) bash -c "python3 -m time_series_model.backtesting.vectorbot \
		$(if $(BACKTEST_MODEL),--model '$(BACKTEST_MODEL)') \
		$(if $(BACKTEST_SYMBOL),--symbol '$(BACKTEST_SYMBOL)') \
		$(if $(BACKTEST_START),--start '$(BACKTEST_START)') \
		$(if $(BACKTEST_END),--end '$(BACKTEST_END)')"

nautilus-backtest:
	@echo "⛵ Running Nautilus AE+LGB backtest (host env, requires nautilus-trader installed)..."
	PYTHONPATH=src $(PYTHON) -m time_series_model.backtesting.nautilus_dim \
		--data-dir $(DATA_DIR) \
		--results-dir $(RESULTS_DIR)/$(NAUTILUS_RESULTS_DIR) \
		--symbols $(SYMBOLS) \
		--timeframe 5T \
		--start $(START_DATE) --end $(END_DATE) \
		--output-dir $(RESULTS_DIR)/nautilus_backtests

# ---------------------------------------------------------------------------
# Rank IC Regression Training (Standalone)
# ---------------------------------------------------------------------------

RANK_IC_SYMBOL ?= $(SYMBOL)
RANK_IC_HORIZON ?= 24
RANK_IC_TIMEFRAME ?= 240T
RANK_IC_FEATURE_TYPE ?= baseline,order_flow,alpha101
RANK_IC_N_SPLITS ?= 5
RANK_IC_TEST_SIZE ?= 0.15
RANK_IC_OUTPUT_DIR ?= results/rank_ic_training
RANK_IC_FILTER_HIGH_CONF ?= 0
RANK_IC_MIN_TREND_STRENGTH ?= 1.0
RANK_IC_SMOOTH_TARGET ?= 0
RANK_IC_CHECK_LEAKAGE ?= 0

ts-r-rank-ic-train:
	@echo "🎯 Rank IC Regression Training (TSCV + OOS Testing)..."
	@echo "   Symbol: $(RANK_IC_SYMBOL)"
	@echo "   Horizon: $(RANK_IC_HORIZON)"
	@echo "   Timeframe: $(RANK_IC_TIMEFRAME)"
	@echo "   Feature Type: $(RANK_IC_FEATURE_TYPE)"
	@echo "   TSCV Folds: $(RANK_IC_N_SPLITS)"
	@echo "   OOS Test Size: $(RANK_IC_TEST_SIZE)"
	@echo "   Output: $(RANK_IC_OUTPUT_DIR)"
	$(DOCKER_RUN_NO_TTY) python3 -m time_series_model.pipeline.training.train_rank_ic_standalone \
		--data-path /workspace/$(DATA_DIR) \
		--symbol $(RANK_IC_SYMBOL) \
		$(if $(START_DATE),--train-start $(START_DATE),) \
		$(if $(END_DATE),--train-end $(END_DATE),) \
		--horizon $(RANK_IC_HORIZON) \
		--timeframe $(RANK_IC_TIMEFRAME) \
		--feature-type $(RANK_IC_FEATURE_TYPE) \
		--n-splits $(RANK_IC_N_SPLITS) \
		--test-size $(RANK_IC_TEST_SIZE) \
		--output-dir /workspace/$(RANK_IC_OUTPUT_DIR) \
		$(if $(filter 1 true yes,$(RANK_IC_FILTER_HIGH_CONF)),--filter-high-confidence,) \
		--min-trend-strength $(RANK_IC_MIN_TREND_STRENGTH) \
		$(if $(filter 1 true yes,$(RANK_IC_SMOOTH_TARGET)),--smooth-target,) \
		$(if $(filter 1 true yes,$(RANK_IC_CHECK_LEAKAGE)),--check-leakage,)
	@echo "✅ Training complete. Check results in $(RANK_IC_OUTPUT_DIR)"


# ---------------------------------------------------------------------------
# Feature Type Evaluation
# ---------------------------------------------------------------------------

FEATURE_EVAL_SYMBOL ?= $(SYMBOL)
FEATURE_EVAL_TIMEFRAME ?= 240T
FEATURE_EVAL_HORIZON ?= 24
FEATURE_EVAL_TYPES ?= baseline,default,enhanced,hurst,wavelet,hilbert,spectral,order_flow,alpha101
FEATURE_EVAL_LEAKAGE_THRESHOLD ?= 0.04
FEATURE_EVAL_OUTPUT_DIR ?= results/feature_evaluation

feature-eval:
	@echo "🔍 Feature Type Evaluation (IC + Leakage Detection)..."
	@echo "   Symbol: $(FEATURE_EVAL_SYMBOL)"
	@echo "   Timeframe: $(FEATURE_EVAL_TIMEFRAME)"
	@echo "   Horizon: $(FEATURE_EVAL_HORIZON)"
	@echo "   Feature Types: $(FEATURE_EVAL_TYPES)"
	@echo "   Output: $(FEATURE_EVAL_OUTPUT_DIR)"
	$(DOCKER_RUN_NO_TTY) python3 -m time_series_model.pipeline.training.feature_type_evaluator \
		--data-path /workspace/$(DATA_DIR) \
		--symbol $(FEATURE_EVAL_SYMBOL) \
		$(if $(START_DATE),--train-start $(START_DATE),) \
		$(if $(END_DATE),--train-end $(END_DATE),) \
		--timeframe $(FEATURE_EVAL_TIMEFRAME) \
		--horizon $(FEATURE_EVAL_HORIZON) \
		--feature-types $(FEATURE_EVAL_TYPES) \
		--output-dir /workspace/$(FEATURE_EVAL_OUTPUT_DIR) \
		--test-leakage \
		--leakage-threshold $(FEATURE_EVAL_LEAKAGE_THRESHOLD)
	@echo "✅ Evaluation complete. Check results in $(FEATURE_EVAL_OUTPUT_DIR)"


# ---------------------------------------------------------------------------
# Cross-sectional feature generation & analysis
# ---------------------------------------------------------------------------

CS_BUILD_SYMBOLS ?= $(SYMBOLS)
CS_BUILD_TIMEFRAME ?= $(FREQ)
CS_BUILD_HORIZON ?= 12
CS_BUILD_START ?= 2024-11-01
CS_BUILD_END ?= 2025-04-30
CS_BUILD_FEATURE_TYPE ?= baseline
CS_BUILD_OUTPUT ?= $(RESULTS_DIR)/feature_exports/cs_panel_$(shell echo $(CS_BUILD_SYMBOLS) | tr ' ,' '__' | cut -c1-40)_$(CS_BUILD_TIMEFRAME)_$(CS_BUILD_HORIZON)b_$(CS_BUILD_FEATURE_TYPE)_$(shell echo $(CS_BUILD_START))_$(shell echo $(CS_BUILD_END)).parquet
CS_BUILD_DROPNA ?= 1

cross-sectional-build-panel:
	@echo "🛠  Building cross-sectional panel for $(CS_BUILD_SYMBOLS)..."
	@mkdir -p $(dir $(CS_BUILD_OUTPUT))
	CS_BUILD_SYMBOLS_SPACE="$(shell echo $(CS_BUILD_SYMBOLS) | tr ',' ' ')" ; \
	$(DOCKER_RUN_NO_TTY) python3 scripts/cross_sectional/generate_panel.py \
		--symbols $$CS_BUILD_SYMBOLS_SPACE \
		--timeframe $(CS_BUILD_TIMEFRAME) \
		--horizon $(CS_BUILD_HORIZON) \
		$(if $(CS_BUILD_START),--start-date $(CS_BUILD_START),) \
		$(if $(CS_BUILD_END),--end-date $(CS_BUILD_END),) \
		--feature-type $(CS_BUILD_FEATURE_TYPE) \
		--output $(CS_BUILD_OUTPUT) \
		$(if $(filter 0,$(CS_BUILD_DROPNA)),--no-dropna,) \
		$(if $(DATA_DIR),--data-path $(DATA_DIR),)
	@echo "✅ Panel saved to $(CS_BUILD_OUTPUT)"

# ---------------------------------------------------------------------------
# Cross-sectional Fama-MacBeth + Newey-West reporting
# ---------------------------------------------------------------------------

CS_INPUT ?= $(RESULTS_DIR)/feature_exports/*.parquet
CS_OUTPUT ?= $(RESULTS_DIR)/cross_sectional/fama_macbeth_report.md
CS_HORIZON ?= 12
CS_MAX_LAG ?= 5
CS_PERIODS_PER_YEAR ?= auto
CS_WINSOR ?= 3.0
CS_REPORT_EXTRA ?=

cross-sectional-report:
	@echo "📊 Cross-sectional Fama-MacBeth analysis for $(SYMBOLS)..."
	@mkdir -p $(dir $(CS_OUTPUT))
	$(DOCKER_RUN_NO_TTY) python3 scripts/cross_sectional/run_famacbeth_report.py \
		--input $(CS_INPUT) \
		--output $(CS_OUTPUT) \
		--symbols "$(SYMBOLS)" \
		--horizon $(CS_HORIZON) \
		--max-lag $(CS_MAX_LAG) \
		--periods-per-year $(CS_PERIODS_PER_YEAR) \
		--winsor $(CS_WINSOR) \
		$(CS_REPORT_EXTRA)
	@echo "✅ Report generated: $(CS_OUTPUT)"

# ---------------------------------------------------------------------------
# Cross-sectional training (boosting / Fama-MacBeth)
# ---------------------------------------------------------------------------

CS_TRAIN_INPUT ?= $(CS_INPUT)
CS_TRAIN_OUTPUT_DIR ?= $(RESULTS_DIR)/cross_sectional/models
CS_TRAIN_MODEL ?= boosting
CS_TRAIN_MODEL_NAME ?= cs_boosting.joblib
CS_TRAIN_FEATURE_COLS ?=
CS_TRAIN_FEATURE_FILE ?=
CS_TRAIN_EXTRA ?=
CS_TRAIN_PRED_NAME ?= predictions.parquet
CS_TRAIN_METRICS_NAME ?= metrics.json
CS_TRAIN_AUTO_SELECT ?= 0
CS_TRAIN_SELECT_TOPK ?=
CS_TRAIN_IC_THRESHOLD ?=
CS_TRAIN_IR_THRESHOLD ?=
CS_TRAIN_SELECTION_STAT ?= ic

cross-sectional-train:
	@echo "🚀 Cross-sectional training ($(CS_TRAIN_MODEL)) for $(SYMBOLS)..."
	@mkdir -p $(CS_TRAIN_OUTPUT_DIR)
	$(DOCKER_RUN_NO_TTY) python3 scripts/cross_sectional/train_cross_sectional_model.py \
		--input $(CS_TRAIN_INPUT) \
		--output-dir $(CS_TRAIN_OUTPUT_DIR) \
		--symbols "$(SYMBOLS)" \
		--horizon $(CS_HORIZON) \
		--model $(CS_TRAIN_MODEL) \
		--winsor $(CS_WINSOR) \
		--periods-per-year $(CS_PERIODS_PER_YEAR) \
		--model-name $(CS_TRAIN_MODEL_NAME) \
		--predictions-name $(CS_TRAIN_PRED_NAME) \
		--metrics-name $(CS_TRAIN_METRICS_NAME) \
		$(if $(CS_TRAIN_FEATURE_FILE),--feature-file $(CS_TRAIN_FEATURE_FILE),) \
		$(if $(filter 1,$(CS_TRAIN_AUTO_SELECT)),--auto-select,) \
		$(if $(CS_TRAIN_SELECT_TOPK),--select-topk $(CS_TRAIN_SELECT_TOPK),) \
		$(if $(CS_TRAIN_IC_THRESHOLD),--ic-threshold $(CS_TRAIN_IC_THRESHOLD),) \
		$(if $(CS_TRAIN_IR_THRESHOLD),--ir-threshold $(CS_TRAIN_IR_THRESHOLD),) \
		$(if $(CS_TRAIN_SELECTION_STAT),--selection-stat $(CS_TRAIN_SELECTION_STAT),) \
		$(if $(CS_TRAIN_FEATURE_COLS),--feature-cols "$(CS_TRAIN_FEATURE_COLS)",) \
		$(CS_TRAIN_EXTRA)
	@echo "✅ Cross-sectional artefacts saved under $(CS_TRAIN_OUTPUT_DIR)"

# ---------------------------------------------------------------------------
# Full cross-sectional workflow (panel -> report -> training)
# ---------------------------------------------------------------------------

cross-sectional-workflow:
	@echo "🔄 Running end-to-end cross-sectional pipeline..."
	$(MAKE) cross-sectional-build-panel
	$(MAKE) cross-sectional-report CS_INPUT="$(CS_BUILD_OUTPUT)" SYMBOLS="$(CS_BUILD_SYMBOLS)" CS_HORIZON=$(CS_BUILD_HORIZON)
	$(MAKE) cross-sectional-train CS_TRAIN_INPUT="$(CS_BUILD_OUTPUT)" SYMBOLS="$(CS_BUILD_SYMBOLS)" CS_HORIZON=$(CS_BUILD_HORIZON)

CS_CATALOG_INPUT ?= $(CS_BUILD_OUTPUT)
CS_CATALOG_OUTPUT ?= results/cross_sectional/factor_sets

cross-sectional-catalog:
	@echo "🗂  Exporting factor catalogue from $(CS_CATALOG_INPUT)..."
	$(DOCKER_RUN_NO_TTY) python3 scripts/cross_sectional/export_factor_catalog.py \
		--input $(CS_CATALOG_INPUT) \
		--output-dir $(CS_CATALOG_OUTPUT)
	@echo "✅ Factor sets saved to $(CS_CATALOG_OUTPUT)"

CS_SELECT_INPUT ?= $(CS_BUILD_OUTPUT)
CS_SELECT_OUTPUT ?= results/cross_sectional/selected_factors.txt
CS_SELECT_OUTPUT_JSON ?= results/cross_sectional/selection_summary.json
CS_SELECT_TARGET ?=
CS_SELECT_MIN_ASSETS ?= 4
CS_SELECT_PER_CATEGORY_TOP ?= 2
CS_SELECT_GLOBAL_TOP ?= 12
CS_SELECT_IC_THRESHOLD ?=
CS_SELECT_IR_THRESHOLD ?=
CS_SELECT_RANKING ?= ic
CS_SELECT_INCLUDE ?=
CS_SELECT_EXTRA ?=

cross-sectional-select:
	@echo "🧠 Auto-selecting factors from $(CS_SELECT_INPUT)..."
	$(DOCKER_RUN_NO_TTY) python3 scripts/cross_sectional/auto_select_factors.py \
		--input $(CS_SELECT_INPUT) \
		$(if $(CS_SELECT_TARGET),--target $(CS_SELECT_TARGET),) \
		--min-assets $(CS_SELECT_MIN_ASSETS) \
		--per-category-top $(CS_SELECT_PER_CATEGORY_TOP) \
		--global-top $(CS_SELECT_GLOBAL_TOP) \
		$(if $(CS_SELECT_IC_THRESHOLD),--ic-threshold $(CS_SELECT_IC_THRESHOLD),) \
		$(if $(CS_SELECT_IR_THRESHOLD),--ir-threshold $(CS_SELECT_IR_THRESHOLD),) \
		--ranking-stat $(CS_SELECT_RANKING) \
		$(if $(CS_SELECT_INCLUDE),--include-categories $(CS_SELECT_INCLUDE),) \
		--output $(CS_SELECT_OUTPUT) \
		--output-json $(CS_SELECT_OUTPUT_JSON) \
		$(CS_SELECT_EXTRA)
	@echo "✅ Selected factors saved to $(CS_SELECT_OUTPUT)"

CS_SHAP_MODEL ?= $(CS_TRAIN_OUTPUT_DIR)/$(CS_TRAIN_MODEL_NAME)
CS_SHAP_PANEL ?= $(CS_BUILD_OUTPUT)
CS_SHAP_FEATURE_FILE ?= $(CS_AUTO_FEATURE_FILE)
CS_SHAP_TARGET ?=
CS_SHAP_TOPK ?= 10
CS_SHAP_OUTPUT ?= results/cross_sectional/shap_reports
CS_SHAP_MAX_SAMPLES ?= 2000
CS_SHAP_ADDITIONAL ?=

cross-sectional-shap:
	@echo "📈 Running SHAP analysis..."
	$(DOCKER_RUN_NO_TTY) python3 scripts/cross_sectional/run_shap_analysis.py \
		--model $(CS_SHAP_MODEL) \
		--panel $(CS_SHAP_PANEL) \
		$(if $(CS_SHAP_FEATURE_FILE),--feature-file $(CS_SHAP_FEATURE_FILE),) \
		$(if $(CS_SHAP_TARGET),--target $(CS_SHAP_TARGET),) \
		--topk $(CS_SHAP_TOPK) \
		--output-dir $(CS_SHAP_OUTPUT) \
		--max-samples $(CS_SHAP_MAX_SAMPLES) \
		$(CS_SHAP_ADDITIONAL)

CS_LOGIC_EXPECTATIONS ?=
CS_LOGIC_OUTPUT ?= results/cross_sectional/shap_logic_report.md
CS_LOGIC_TOLERANCE ?= 0.0
CS_LOGIC_EXTRA ?=

cross-sectional-logic-check:
	@echo "🧐 Validating factor economic logic..."
	$(DOCKER_RUN_NO_TTY) python3 scripts/cross_sectional/run_factor_logic_check.py \
		--shap-manifest $(CS_SHAP_OUTPUT)/manifest.json \
		--expectations $(CS_LOGIC_EXPECTATIONS) \
		--tolerance $(CS_LOGIC_TOLERANCE) \
		--output $(CS_LOGIC_OUTPUT) \
		$(CS_LOGIC_EXTRA)

CS_DRIFT_BASELINE ?= results/cross_sectional/shap_baseline.json
CS_DRIFT_THRESHOLD ?= 0.5
CS_DRIFT_OUTPUT ?= results/cross_sectional/shap_drift_report.md
CS_DRIFT_UPDATE ?= 0
CS_DRIFT_EXTRA ?=

cross-sectional-shap-drift:
	@echo "📉 Checking SHAP drift..."
	$(DOCKER_RUN_NO_TTY) python3 scripts/cross_sectional/run_shap_drift_monitor.py \
		--current $(CS_SHAP_OUTPUT)/manifest.json \
		--baseline $(CS_DRIFT_BASELINE) \
		--threshold $(CS_DRIFT_THRESHOLD) \
		--output $(CS_DRIFT_OUTPUT) \
		$(if $(filter 1,$(CS_DRIFT_UPDATE)),--update-baseline,) \
		$(CS_DRIFT_EXTRA)

CS_AUTO_PER_CATEGORY_TOP ?= 2
CS_AUTO_GLOBAL_TOP ?= 12
CS_AUTO_IC_THRESHOLD ?= 0.01
CS_AUTO_IR_THRESHOLD ?= 0.5
CS_AUTO_MIN_ASSETS ?= 4
CS_AUTO_FEATURE_FILE ?= results/cross_sectional/selected_factors.txt

cross-sectional-auto:
	@echo "🤖 Running fully automated cross-sectional pipeline..."
	$(MAKE) cross-sectional-build-panel
	$(MAKE) cross-sectional-select \
		CS_SELECT_INPUT="$(CS_BUILD_OUTPUT)" \
		CS_SELECT_OUTPUT="$(CS_AUTO_FEATURE_FILE)" \
		CS_SELECT_OUTPUT_JSON="results/cross_sectional/selection_summary.json" \
		CS_SELECT_MIN_ASSETS=$(CS_AUTO_MIN_ASSETS) \
		CS_SELECT_PER_CATEGORY_TOP=$(CS_AUTO_PER_CATEGORY_TOP) \
		CS_SELECT_GLOBAL_TOP=$(CS_AUTO_GLOBAL_TOP) \
		CS_SELECT_IC_THRESHOLD=$(CS_AUTO_IC_THRESHOLD) \
		CS_SELECT_IR_THRESHOLD=$(CS_AUTO_IR_THRESHOLD)
	$(MAKE) cross-sectional-report \
		CS_INPUT="$(CS_BUILD_OUTPUT)" \
		SYMBOLS="$(CS_BUILD_SYMBOLS)" \
		CS_HORIZON=$(CS_BUILD_HORIZON) \
		CS_REPORT_EXTRA="--feature-file $(CS_AUTO_FEATURE_FILE)"
	$(MAKE) cross-sectional-train \
		CS_TRAIN_INPUT="$(CS_BUILD_OUTPUT)" \
		SYMBOLS="$(CS_BUILD_SYMBOLS)" \
		CS_HORIZON=$(CS_BUILD_HORIZON) \
		CS_PERIODS_PER_YEAR=$(CS_PERIODS_PER_YEAR) \
		CS_TRAIN_FEATURE_FILE="$(CS_AUTO_FEATURE_FILE)"
	$(MAKE) cross-sectional-shap \
		CS_SHAP_MODEL="$(CS_TRAIN_OUTPUT_DIR)/$(CS_TRAIN_MODEL_NAME)" \
		CS_SHAP_PANEL="$(CS_BUILD_OUTPUT)" \
		CS_SHAP_FEATURE_FILE="$(CS_AUTO_FEATURE_FILE)"
	$(MAKE) cross-sectional-logic-check \
		CS_LOGIC_EXPECTATIONS="$(CS_LOGIC_EXPECTATIONS)"
	$(MAKE) cross-sectional-shap-drift \
		CS_DRIFT_BASELINE="$(CS_DRIFT_BASELINE)"
