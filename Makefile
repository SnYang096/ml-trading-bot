
# ---------------------------------------------------------------------------
# ML Trading Project
# Streamlined commands for production workflows
# ---------------------------------------------------------------------------

PYTHON := python3
PIP := pip3
SUDO ?=
HOST_UID ?= $(shell id -u)
HOST_GID ?= $(shell id -g)

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
	--user $(HOST_UID):$(HOST_GID) \
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
	--user $(HOST_UID):$(HOST_GID) \
	-e PYTHONPATH=/workspace/src \
	-e PYTHONUNBUFFERED=1 \
	-v $(PWD):/workspace \
	-v $(PWD)/data/parquet_data:/workspace/data/parquet_data \
	-w /workspace \
	--shm-size=8gb \
	$(DOCKER_IMAGE)
endif


.PHONY: help clean format lint fix-permissions fix-ownership dev-install install-hooks docker-build docker-install builder-shell \
	data-download data-convert data-pipeline \
	train train-quantile tune-q50-params rolling rolling-multi rolling-update-only \
	ts-vectorbot-backtest ts-nautilus-backtest \
	ts-dim-compare ts-feature-eval ts-factor-eval ts-timeframe-forward-report \
	ts-strategy-feature-compare feature-indicators \
	vectorbot-backtest nautilus-backtest feature-eval timeframe-forward-report \
	cs-catalog cs-select cs-shap cs-shap-drift cs-auto cs-logic-check \
	cs-build-panel cs-report cs-train cs-workflow \
	test-wpt-volume-profile test-wpt-volume-profile-simple

help:
	@echo "ML Trading Project"
	@echo "===================="
	@echo "Local development commands (run on host):"
	@echo "  make dev-install          # Install project in editable mode"
	@echo "  make install-hooks        # Install Git pre-commit hooks (run make format & lint before commit)"
	@echo "  make format               # Format code with black"
	@echo "  make lint                 # Lint code with flake8"
	@echo ""
	@echo "Testing commands (run in Docker):"
	@echo "  make test-wpt-volume-profile        # Test WPT volume profile improvements (pytest format)"
	@echo "  make test-wpt-volume-profile-simple # Test WPT volume profile improvements (simple script)"
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
	@echo "    make rolling            # Config-driven rolling training (expanding window, recommended for production)"
	@echo "    make tune-q50-params    # Pre-train Q50 parameter search (for quantile models)"
	@echo ""
	@echo "  Data commands:"
	@echo "    make data-download     # Download Binance data"
	@echo "    make data-convert      # Convert ZIPs to Parquet"
	@echo "    make data-pipeline     # Download then convert"
	@echo ""
	@echo "  Other commands:"
	@echo "    make ts-r-rank-ic-train # Rank IC regression training (TSCV + OOS testing)"
	@echo "    make ts-sr-reversal # SR Reversal model training (XGBoost Binary)"
	@echo "    make ts-sr-reversal-optuna # Optuna search for SR signal params"
	@echo "    make ts-sr-breakout # SR Breakout model training (XGBoost Regression)"
	@echo "    make ts-compression-breakout # Compression Breakout model training (CatBoost Multiclass)"
	@echo "    make ts-trend-following # Trend Following model training (LightGBM Regression)"
	@echo "    make ts-feature-eval    # Time-series feature IC / leakage evaluation"
	@echo "    make ts-factor-eval     # Time-series factor IC / win-rate evaluation (single asset)"
	@echo "    make ts-dim-compare     # Dimensionality comparison & top factor selection"
	@echo "    make ts-timeframe-forward-report # Timeframe vs forward-bar correlation analysis"
	@echo "    make ts-strategy-feature-compare # Ablation Study: Compare multiple feature configs for a strategy"
	@echo "    make ts-vectorbot-backtest # Run VectorBot risk-managed backtest"
	@echo "    make ts-nautilus-backtest  # Run Nautilus Trader backtest"
	@echo "    make cs-factor-eval    # Cross-sectional factor evaluation (IC, decay, quantile spread)"
	@echo "    make cs-build-panel    # Generate multi-asset factor panels for CS modelling"
	@echo "    make cs-report         # Fama-MacBeth + Newey-West + IC/IR markdown report"
	@echo "    make cs-train          # Train cross-sectional models (boosting/Fama-MacBeth)"
	@echo "    make cs-workflow       # Build panel + report + train in one go"
	@echo "    make cs-catalog        # Categorise factors from an existing panel"
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

PERM_DIR ?= scripts/diagnostics
PERM_MODE ?= 664
fix-permissions:
	@echo "🔐 Updating file permissions under $(PERM_DIR) to mode $(PERM_MODE)..."
	@$(SUDO) find $(PERM_DIR) -type f -exec chmod $(PERM_MODE) {} +
	@echo "✅ Permissions updated."

dev-install:
	$(PIP) install -e .

OWNER_DIR ?= scripts
OWNER_USER ?= yin
fix-ownership:
	@echo "👤 Changing file ownership under $(OWNER_DIR) to $(OWNER_USER)..."
	@$(SUDO) chown -R $(OWNER_USER) $(OWNER_DIR)
	@echo "✅ Ownership updated."

fix-all-ownership:
	@echo "👤 Changing file ownership for src/, tests/, and scripts/ to $(OWNER_USER)..."
	@$(SUDO) chown -R $(OWNER_USER) src/ tests/ scripts/
	@echo "✅ Ownership updated."

install-hooks:
	@echo "📦 Installing Git hooks..."
	@bash scripts/install-git-hooks.sh

docker-build:
	@echo "🔨 Building Docker image $(DOCKER_IMAGE)..."
	docker build -f docker/Dockerfile.gpu -t $(DOCKER_IMAGE) .
	

docker-install:
	@echo "📦 Installing project inside Docker container..."
	$(DOCKER_RUN) pip3 install -e /workspace/ml_trading_bot

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
# TS factor evaluation
# ---------------------------------------------------------------------------

TS_FACTOR_STRATEGY ?= config/strategies/factor_ts_simple
TS_FACTOR_FACTORS ?= rsi atr macd macd_signal macd_histogram
TS_FACTOR_SYMBOL ?= BTCUSDT
TS_FACTOR_TIMEFRAME ?= 240T
TS_FACTOR_START ?=
TS_FACTOR_END ?=
TS_FACTOR_QUANTILE ?= 0.2
TS_FACTOR_OUTPUT_DIR ?= results/factor_ts_eval
TS_FACTOR_MODE ?= strategy
TS_FACTOR_IC_DECAY_LAGS ?= 1,3,5,10,20

ts-factor-eval:
	@if [ -z "$(TS_FACTOR_FACTORS)" ]; then \
		echo "❌ 错误: 必须指定 TS_FACTOR_FACTORS"; \
		echo "用法: make ts-factor-eval TS_FACTOR_FACTORS='atr sqs_hal_high' TS_FACTOR_STRATEGY=config/strategies/sr_reversal"; \
		echo "   或者: make ts-factor-eval TS_FACTOR_FACTORS='rsi,macd' (逗号分隔会自动转换)"; \
		exit 1; \
	fi
	@echo "📈 TS 因子评价: $(TS_FACTOR_FACTORS)"
	@echo "   策略配置: $(TS_FACTOR_STRATEGY)"
	@echo "   IC 衰减分析: $(TS_FACTOR_IC_DECAY_LAGS) bars"
	@# Convert comma-separated factors to space-separated (support both formats)
	@FACTORS_SPACE=$$(echo "$(TS_FACTOR_FACTORS)" | tr ',' ' '); \
	$(DOCKER_RUN_NO_TTY) python3 scripts/factor_management/factor_ts_eval.py \
		--strategy-config /workspace/$(TS_FACTOR_STRATEGY) \
		--symbol $(TS_FACTOR_SYMBOL) \
		--factors $$FACTORS_SPACE \
		--data-path /workspace/$(DATA_DIR) \
		--timeframe $(TS_FACTOR_TIMEFRAME) \
		$(if $(TS_FACTOR_START),--start-date $(TS_FACTOR_START),) \
		$(if $(TS_FACTOR_END),--end-date $(TS_FACTOR_END),) \
		--quantile $(TS_FACTOR_QUANTILE) \
		--feature-mode $(TS_FACTOR_MODE) \
		--ic-decay-lags $(TS_FACTOR_IC_DECAY_LAGS) \
		--output-dir /workspace/$(TS_FACTOR_OUTPUT_DIR)



# ---------------------------------------------------------------------------
# Strategy feature comparison
# ---------------------------------------------------------------------------

STRAT_COMPARE_CONFIG ?= config/strategies/sr_reversal
STRAT_COMPARE_DATA_PATH ?= $(DATA_DIR)
STRAT_COMPARE_SYMBOL ?= BTCUSDT
STRAT_COMPARE_TIMEFRAME ?= 240T
STRAT_COMPARE_START ?= 2024-01-01
STRAT_COMPARE_END ?= 2025-10-31
STRAT_COMPARE_TEST_SIZE ?= 0.15
STRAT_COMPARE_OUTPUT_DIR ?= results/strategy_compare
STRAT_COMPARE_OVERRIDES ?=
STRAT_COMPARE_RUN_ROLLING ?= false
STRAT_COMPARE_ROLL_TRAIN ?= 5000
STRAT_COMPARE_ROLL_TEST ?= 1000
STRAT_COMPARE_ROLL_STEP ?= 1000
STRAT_COMPARE_ROLL_MAX ?= 5

# Ablation Study (消融实验): Compare strategy performance across different feature configurations
# This command trains the same strategy with different feature sets to evaluate
# the contribution of each feature group. Use --feature-overrides to specify variants.
# Example: make ts-strategy-feature-compare STRAT_COMPARE_CONFIG=config/strategies/sr_reversal \
#          STRAT_COMPARE_OVERRIDES="baseline=config/features/baseline.yaml full=config/features/full.yaml"
# make ts-strategy-feature-compare STRAT_COMPARE_CONFIG=config/strategies/sr_reversal \
#           STRAT_COMPARE_OVERRIDES="full=config/strategies/sr_reversal/features_full.yaml"

ts-strategy-feature-compare:
	@echo "🆚 Ablation Study: Comparing feature variants for $(STRAT_COMPARE_CONFIG)"
	@$(DOCKER_RUN_NO_TTY) python3 scripts/strategy_management/strategy_feature_compare.py \
		--strategy-config /workspace/$(STRAT_COMPARE_CONFIG) \
		--symbol $(STRAT_COMPARE_SYMBOL) \
		--data-path /workspace/$(STRAT_COMPARE_DATA_PATH) \
		--timeframe $(STRAT_COMPARE_TIMEFRAME) \
		$(if $(STRAT_COMPARE_START),--start-date $(STRAT_COMPARE_START),) \
		$(if $(STRAT_COMPARE_END),--end-date $(STRAT_COMPARE_END),) \
		--test-size $(STRAT_COMPARE_TEST_SIZE) \
		--output-dir /workspace/$(STRAT_COMPARE_OUTPUT_DIR) \
		$(if $(STRAT_COMPARE_OVERRIDES),--feature-overrides $(STRAT_COMPARE_OVERRIDES),) \
		$(if $(filter true,$(STRAT_COMPARE_RUN_ROLLING)),--run-rolling,) \
		--rolling-train-bars $(STRAT_COMPARE_ROLL_TRAIN) \
		--rolling-test-bars $(STRAT_COMPARE_ROLL_TEST) \
		--rolling-step-bars $(STRAT_COMPARE_ROLL_STEP) \
		--rolling-max-windows $(STRAT_COMPARE_ROLL_MAX)

# SR Reversal Rule Baseline: Test pure rule-based SR+RR strategy without ML (sr_reversal strategy only)
# This helps diagnose whether low trade count is due to:
# - Too few SR signals (feature/rule issue)
# - Poor baseline edge (SR definition issue)
# - Model/threshold being too conservative (ML issue)
SR_BASELINE_CONFIG ?= config/strategies/sr_reversal
SR_BASELINE_SYMBOL ?= BTCUSDT
SR_BASELINE_DATA_PATH ?= $(DATA_DIR)
SR_BASELINE_TIMEFRAME ?= 240T
SR_BASELINE_START ?= 2024-01-01
SR_BASELINE_END ?= 2025-10-31
SR_BASELINE_TICK_MODE ?= auto  # VPIN tick 数据模式：auto=<=120分钟自动启用，on=强制启用，off=禁用
SR_BASELINE_TICKS_DIR ?= data/parquet_data  # tick 级 parquet 数据目录（统一数据源，不再区分 OHLCV 和 tick）
SR_BASELINE_TICKS_LOOKBACK ?= 60  # VPIN 计算时向前/向后额外加载的分钟数（用于边界处理）

ts-sr-reversal-rule-baseline:
	@echo "📊 SR Reversal Rule Baseline: Testing pure rule-based SR+RR strategy (no ML)"
	@$(DOCKER_RUN_NO_TTY) python3 scripts/diagnostics/sr_reversal_rule_baseline.py \
		--strategy-config /workspace/$(SR_BASELINE_CONFIG) \
		--symbol $(SR_BASELINE_SYMBOL) \
		--data-path /workspace/$(SR_BASELINE_DATA_PATH) \
		--timeframe $(SR_BASELINE_TIMEFRAME) \
		$(if $(SR_BASELINE_START),--start-date $(SR_BASELINE_START),) \
		$(if $(SR_BASELINE_END),--end-date $(SR_BASELINE_END),) \
		--tick-data-mode $(SR_BASELINE_TICK_MODE) \
		--ticks-dir /workspace/$(SR_BASELINE_TICKS_DIR) \
		--ticks-lookback-minutes $(SR_BASELINE_TICKS_LOOKBACK)

# SR Reversal 1h Baseline: Test rule-based strategy on 1h timeframe (more trades, finer granularity)
# Adjusts max_holding_bars to maintain same holding period as 4h (200 bars ≈ 8.3 days)
ts-sr-reversal-1h-baseline:
	@echo "📊 SR Reversal Rule Baseline (1h): Testing pure rule-based SR+RR strategy on 1h timeframe"
	@$(DOCKER_RUN_NO_TTY) python3 scripts/diagnostics/sr_reversal_rule_baseline.py \
		--strategy-config /workspace/$(SR_BASELINE_CONFIG) \
		--symbol $(SR_BASELINE_SYMBOL) \
		--data-path /workspace/$(SR_BASELINE_DATA_PATH) \
		--timeframe 60T \
		$(if $(SR_BASELINE_START),--start-date $(SR_BASELINE_START),) \
		$(if $(SR_BASELINE_END),--end-date $(SR_BASELINE_END),) \
		--tick-data-mode $(SR_BASELINE_TICK_MODE) \
		--ticks-dir /workspace/$(SR_BASELINE_TICKS_DIR) \
		--ticks-lookback-minutes $(SR_BASELINE_TICKS_LOOKBACK) \
		--max-holding-bars 200

ts-test-vpin-thresholds:
	@echo "🧪 Testing different VPIN thresholds for SR Reversal"
	@$(DOCKER_RUN_NO_TTY) python3 scripts/diagnostics/test_vpin_thresholds.py \
		--strategy-config /workspace/$(SR_BASELINE_CONFIG) \
		--symbol $(SR_BASELINE_SYMBOL) \
		--data-path /workspace/$(SR_BASELINE_DATA_PATH) \
		--timeframe $(SR_BASELINE_TIMEFRAME) \
		$(if $(SR_BASELINE_START),--start-date $(SR_BASELINE_START),) \
		$(if $(SR_BASELINE_END),--end-date $(SR_BASELINE_END),)

# SR Reversal Model Diagnosis: Analyze why model doesn't open trades
# Compares model predictions vs rule baseline on signal points
SR_DIAG_CONFIG ?= config/strategies/sr_reversal
SR_DIAG_SYMBOL ?= BTCUSDT
SR_DIAG_DATA_PATH ?= $(DATA_DIR)
SR_DIAG_TIMEFRAME ?= 240T
SR_DIAG_START ?= 2024-01-01
SR_DIAG_END ?= 2025-10-31
SR_DIAG_TEST_SIZE ?= 0.15
SR_DIAG_LONG_THRESHOLD ?= 0.6
SR_DIAG_SHORT_THRESHOLD ?= 0.4

ts-sr-reversal-diagnosis:
	@echo "🔍 SR Reversal Model Diagnosis: Why doesn't model open trades?"
	@$(DOCKER_RUN_NO_TTY) python3 scripts/diagnostics/sr_reversal_model_diagnosis.py \
		--strategy-config /workspace/$(SR_DIAG_CONFIG) \
		--symbol $(SR_DIAG_SYMBOL) \
		--data-path /workspace/$(SR_DIAG_DATA_PATH) \
		--timeframe $(SR_DIAG_TIMEFRAME) \
		$(if $(SR_DIAG_START),--start-date $(SR_DIAG_START),) \
		$(if $(SR_DIAG_END),--end-date $(SR_DIAG_END),) \
		--test-size $(SR_DIAG_TEST_SIZE) \
		--long-entry-threshold $(SR_DIAG_LONG_THRESHOLD) \
		--short-entry-threshold $(SR_DIAG_SHORT_THRESHOLD)

# SR Reversal Rule Optimization: Find parameter plateaus and compare with ML model
SR_OPT_CONFIG ?= config/strategies/sr_reversal
SR_OPT_SYMBOL ?= BTCUSDT
SR_OPT_DATA_PATH ?= $(DATA_DIR)
SR_OPT_TIMEFRAME ?= 240T
SR_OPT_START ?= 2024-01-01
SR_OPT_END ?= 2025-10-31
SR_OPT_SEARCH_TYPE ?= random
SR_OPT_N_TRIALS ?= 100
SR_OPT_OUTPUT_DIR ?= results/rule_optimization

# SR Reversal Rule Optimization: Find parameter plateaus using grid/random/Optuna search (sr_reversal strategy only)
# Automatically generates plateau charts after optimization completes.
# Outputs: results/rule_optimization/optimization_results.csv and optimization_report.html (with charts)
ts-sr-reversal-rule-optimization:
	@echo "🔍 SR Reversal Rule Parameter Optimization: Finding parameter plateaus"
	@echo "   Symbol: $(SR_OPT_SYMBOL)"
	@echo "   Timeframe: $(SR_OPT_TIMEFRAME)"
	@echo "   Search Type: $(SR_OPT_SEARCH_TYPE)"
	@echo "   N Trials: $(SR_OPT_N_TRIALS)"
	@echo "   Output: $(SR_OPT_OUTPUT_DIR)"
	@$(DOCKER_RUN_NO_TTY) python3 scripts/diagnostics/sr_reversal_rule_optimization.py \
		--strategy-config /workspace/$(SR_OPT_CONFIG) \
		--symbol $(SR_OPT_SYMBOL) \
		--data-path /workspace/$(SR_OPT_DATA_PATH) \
		--timeframe $(SR_OPT_TIMEFRAME) \
		$(if $(SR_OPT_START),--start-date $(SR_OPT_START),) \
		$(if $(SR_OPT_END),--end-date $(SR_OPT_END),) \
		--output-dir /workspace/$(SR_OPT_OUTPUT_DIR) \
		--search-type $(SR_OPT_SEARCH_TYPE) \
		--n-trials $(SR_OPT_N_TRIALS)
	@echo ""
	@echo "🖼️  Generating rule plateau heatmaps and scatter charts..."
	@$(DOCKER_RUN_NO_TTY) python3 scripts/diagnostics/generate_rule_plateau_charts.py \
		--results-csv /workspace/results/rule_optimization/optimization_results.csv \
		--report-html /workspace/results/rule_optimization/optimization_report.html

# Rule Plateau Charts: Generate heatmaps/scatter plots from rule optimization results (standalone, can be run separately)
# Reads: results/rule_optimization/optimization_results.csv
# Updates: results/rule_optimization/optimization_report.html (injects charts)
ts-rule-plateau-charts:
	@echo "🖼️  Generating rule plateau heatmaps and scatter charts"
	@$(DOCKER_RUN_NO_TTY) python3 scripts/diagnostics/generate_rule_plateau_charts.py \
		--results-csv /workspace/results/rule_optimization/optimization_results.csv \
		--report-html /workspace/results/rule_optimization/optimization_report.html

# SR Reversal ML Parameter Sweep: Generate parameter grid data for plateau analysis (sr_reversal strategy only)
ts-sr-reversal-ml-param-sweep:
	@echo "🔁 Running ML parameter sweep for SR Reversal plateau analysis"
	@$(DOCKER_RUN_NO_TTY) python3 scripts/diagnostics/sr_reversal_ml_parameter_sweep.py \
		--strategy-config /workspace/$(SR_COMP_CONFIG) \
		--symbol $(SR_COMP_SYMBOL) \
		--data-path /workspace/$(SR_COMP_DATA_PATH) \
		--timeframe $(SR_COMP_TIMEFRAME) \
		$(if $(SR_COMP_START),--start-date $(SR_COMP_START),) \
		$(if $(SR_COMP_END),--end-date $(SR_COMP_END),) \
		--test-size $(SR_COMP_TEST_SIZE) \
		--output-dir /workspace/$(SR_COMP_OUTPUT_DIR)

# ML Plateau Charts: Generate heatmaps/scatter plots from ML parameter sweep CSV (generic, works for any strategy)
ts-ml-plateau-charts:
	@echo "🖼️  Generating ML plateau heatmaps and scatter charts"
	@$(DOCKER_RUN_NO_TTY) python3 scripts/diagnostics/generate_ml_plateau_charts.py \
		--results-csv /workspace/results/model_comparison/ml_param_sweep.csv \
		--report-html /workspace/results/model_comparison/comparison_report.html

# SR Reversal Model Comparison: Rule-based vs ML vs ML+Volatility
SR_COMP_CONFIG ?= config/strategies/sr_reversal
SR_COMP_SYMBOL ?= BTCUSDT
SR_COMP_DATA_PATH ?= $(DATA_DIR)
SR_COMP_TIMEFRAME ?= 240T
SR_COMP_START ?= 2024-01-01
SR_COMP_END ?= 2025-10-31
SR_COMP_TEST_SIZE ?= 0.15
SR_COMP_OUTPUT_DIR ?= results/model_comparison_$(subst T,,$(SR_COMP_TIMEFRAME))h
SR_COMP_RULE_PARAMS ?= results/rule_optimization/optimization_results.csv
SR_COMP_TICK_MODE ?= auto
SR_COMP_TICKS_DIR ?= data/parquet_data
SR_COMP_TICKS_LOOKBACK ?= 60

ts-sr-reversal-model-comparison:
	@echo "📊 SR Reversal Model Comparison: Rule-based vs ML vs ML+Volatility"
	@echo "   Symbol: $(SR_COMP_SYMBOL)"
	@echo "   Timeframe: $(SR_COMP_TIMEFRAME)"
	@echo "   Test Size: $(SR_COMP_TEST_SIZE)"
	@echo "   Output: $(SR_COMP_OUTPUT_DIR)"
	@$(DOCKER_RUN_NO_TTY) python3 scripts/diagnostics/sr_reversal_model_comparison.py \
		--strategy-config /workspace/$(SR_COMP_CONFIG) \
		--symbol $(SR_COMP_SYMBOL) \
		--data-path /workspace/$(SR_COMP_DATA_PATH) \
		--timeframe $(SR_COMP_TIMEFRAME) \
		$(if $(SR_COMP_START),--start-date $(SR_COMP_START),) \
		$(if $(SR_COMP_END),--end-date $(SR_COMP_END),) \
		--test-size $(SR_COMP_TEST_SIZE) \
		--output-dir /workspace/$(SR_COMP_OUTPUT_DIR) \
		--tick-data-mode $(SR_COMP_TICK_MODE) \
		--ticks-dir /workspace/$(SR_COMP_TICKS_DIR) \
		--ticks-lookback-minutes $(SR_COMP_TICKS_LOOKBACK) \
		$(if $(SR_COMP_RULE_PARAMS),--rule-params /workspace/$(SR_COMP_RULE_PARAMS),)

ts-analyze-ml-volatility:
	@echo "🔍 Analyzing ML+Volatility Model Performance Issues"
	@$(DOCKER_RUN_NO_TTY) python3 scripts/diagnostics/analyze_ml_volatility_model.py

ts-analyze-dtw-volatility:
	@echo "🔍 Analyzing DTW Features and Volatility Model"
	@$(DOCKER_RUN_NO_TTY) python3 scripts/diagnostics/analyze_dtw_and_volatility.py

TF_CONFIG_PEARSON ?= 0.03
TF_CONFIG_PVALUE ?= 1e-5
TF_CONFIG_MIN_SAMPLES ?= 500
TF_CONFIG_TOP_PER_SYMBOL ?= 5
TF_CONFIG_TOP_PER_GROUP ?= 10

TRAIN_FEATURE_TYPE ?= baseline
DIRECTION_THRESHOLD ?= f1_optimize

ts-timeframe-forward-report:
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
		echo "❌ Cannot find $$TF_RUN_DIR/timeframe_forward_details.csv -- did ts-timeframe-forward-report succeed?"; \
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

timeframe-forward-report:
	@echo "⚠️ 'timeframe-forward-report' has been renamed to 'ts-timeframe-forward-report'. Please update your workflows."
	@$(MAKE) ts-timeframe-forward-report

# ---------------------------------------------------------------------------
# Dimensionality: Three-stage feature selection (before vs after reduction)
# ---------------------------------------------------------------------------

DIM_COMPARE_ARGS ?=
HORIZONS ?= 24
DIM_COMPARE_CONFIG ?= config/strategies/sr_reversal
DIM_COMPARE_TIMEFRAME ?= 15T

# Config-driven dimensionality comparison: Three-stage feature selection
# Stage 1: Missing/stability filter → Stage 2: IC ranking → Stage 3: Correlation-based selection
# Outputs top_factors.json for use in rolling training
ts-dim-compare:
	@if [ -z "$(DIM_COMPARE_CONFIG)" ]; then \
		echo "❌ 错误: 必须指定 DIM_COMPARE_CONFIG"; \
		echo "用法: make ts-dim-compare DIM_COMPARE_CONFIG=config/strategies/sr_reversal SYMBOL=BTCUSDT"; \
		exit 1; \
	fi
	@echo "🔬 Config-Driven Dimensionality Comparison"
	@echo "   策略配置: $(DIM_COMPARE_CONFIG)"
	@echo "   交易对: $(SYMBOL)"
	@echo "   时间周期: $(DIM_COMPARE_TIMEFRAME)"
	@$(DOCKER_RUN_NO_TTY) python3 scripts/dimensionality/dim_compare.py \
		--config /workspace/$(DIM_COMPARE_CONFIG) \
		--symbol $(SYMBOL) \
		--data-path /workspace/$(DATA_DIR) \
		--timeframe $(DIM_COMPARE_TIMEFRAME) \
		$(if $(START_DATE),--train-start $(START_DATE)) \
		$(if $(END_DATE),--train-end $(END_DATE))

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

ROLLING_CONFIG ?= config/strategies/sr_reversal
ROLLING_TIMEFRAME ?= 15T
ROLLING_UPDATE_ONLY ?= false

# Config-driven rolling training: Expanding window training for time-series strategies
# Each test month uses all previous months for training (simulating real-world deployment)
rolling:
	@if [ -z "$(ROLLING_CONFIG)" ]; then \
		echo "❌ 错误: 必须指定 ROLLING_CONFIG"; \
		echo "用法: make rolling ROLLING_CONFIG=config/strategies/sr_reversal SYMBOL=BTCUSDT"; \
		exit 1; \
	fi
	@echo "🔄 Config-Driven Rolling Training"
	@echo "   策略配置: $(ROLLING_CONFIG)"
	@echo "   交易对: $(SYMBOL)"
	@echo "   时间周期: $(ROLLING_TIMEFRAME)"
	@$(DOCKER_RUN_NO_TTY) python3 scripts/rolling/rolling_train.py \
		--config /workspace/$(ROLLING_CONFIG) \
		--symbol $(SYMBOL) \
		--data-dir /workspace/$(DATA_DIR) \
		--timeframe $(ROLLING_TIMEFRAME) \
		--initial-train-months $(INITIAL_TRAIN_MONTHS) \
		--min-train-months $(MIN_TRAIN_MONTHS) \
		--output-root /workspace/results/rolling \
		$(if $(ROLLING_START),--start $(ROLLING_START),) \
		$(if $(ROLLING_END),--end $(ROLLING_END),) \
		$(if $(filter true,$(ROLLING_UPDATE_ONLY)),--update-only,)


BACKTEST_START ?=$(START_DATE)
BACKTEST_END ?=$(END_DATE)
BACKTEST_SYMBOL ?=$(SYMBOL)
BACKTEST_MODEL ?=$(MODEL_PATH)
ts-vectorbot-backtest:
	@echo "🤖 Running VectorBot backtest with model=$(BACKTEST_MODEL) symbol=$(BACKTEST_SYMBOL) range=$(BACKTEST_START)→$(BACKTEST_END) ..."
	$(DOCKER_RUN_NO_TTY) bash -c "python3 -m time_series_model.backtesting.vectorbot \
		$(if $(BACKTEST_MODEL),--model '$(BACKTEST_MODEL)') \
		$(if $(BACKTEST_SYMBOL),--symbol '$(BACKTEST_SYMBOL)') \
		$(if $(BACKTEST_START),--start '$(BACKTEST_START)') \
		$(if $(BACKTEST_END),--end '$(BACKTEST_END)')"

vectorbot-backtest:
	@echo "⚠️ 'vectorbot-backtest' has been renamed to 'ts-vectorbot-backtest'. Please update your workflows."
	@$(MAKE) ts-vectorbot-backtest

ts-nautilus-backtest:
	@echo "⛵ Running Nautilus AE+LGB backtest (host env, requires nautilus-trader installed)..."
	PYTHONPATH=src $(PYTHON) -m time_series_model.backtesting.nautilus_dim \
		--data-dir $(DATA_DIR) \
		--results-dir $(RESULTS_DIR)/$(NAUTILUS_RESULTS_DIR) \
		--symbols $(SYMBOLS) \
		--timeframe 5T \
		--start $(START_DATE) --end $(END_DATE) \
		--output-dir $(RESULTS_DIR)/nautilus_backtests

nautilus-backtest:
	@echo "⚠️ 'nautilus-backtest' has been renamed to 'ts-nautilus-backtest'. Please update your workflows."
	@$(MAKE) ts-nautilus-backtest

# ---------------------------------------------------------------------------
# Rank IC Regression Training (Standalone)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Feature Type Evaluation
# ---------------------------------------------------------------------------

FEATURE_EVAL_SYMBOL ?= $(SYMBOL)
FEATURE_EVAL_TIMEFRAME ?= 240T
FEATURE_EVAL_HORIZON ?= 24
FEATURE_EVAL_TYPES ?= baseline
FEATURE_EVAL_LEAKAGE_THRESHOLD ?= 0.04
FEATURE_EVAL_OUTPUT_DIR ?= results/feature_evaluation
FEATURE_EVAL_START_DATE ?=2023-01-01
FEATURE_EVAL_END_DATE ?=2025-10-31
FEATURE_EVAL_TOP_FACTORS_COUNT ?=50
FEATURE_EVAL_TOP_FACTORS_IC_THRESHOLD ?= 0.02
FEATURE_EVAL_TRAIN_ONLY ?= 1
FEATURE_EVAL_TEST_SIZE ?= 0.15

ts-feature-eval:
	@echo "🔍 Feature Type Evaluation (IC Ranking + Top Factors Selection)..."
	@echo "   Symbol: $(FEATURE_EVAL_SYMBOL)"
	@echo "   Timeframe: $(FEATURE_EVAL_TIMEFRAME)"
	@echo "   Horizon: $(FEATURE_EVAL_HORIZON)"
	@echo "   Feature Types: $(FEATURE_EVAL_TYPES)"
	@echo "   Start Date: $(if $(FEATURE_EVAL_START_DATE),$(FEATURE_EVAL_START_DATE),Not specified - will load all available data)"
	@echo "   End Date: $(if $(FEATURE_EVAL_END_DATE),$(FEATURE_EVAL_END_DATE),Not specified)"
	@echo "   Train Only: $(if $(filter 1 true yes,$(FEATURE_EVAL_TRAIN_ONLY)),Yes (test_size=$(FEATURE_EVAL_TEST_SIZE)),No - using all data)"
	@echo "   Top Factors: $(if $(FEATURE_EVAL_TOP_FACTORS_COUNT),Top $(FEATURE_EVAL_TOP_FACTORS_COUNT) features,IC threshold >= $(FEATURE_EVAL_TOP_FACTORS_IC_THRESHOLD))"
	@echo "   Output: $(FEATURE_EVAL_OUTPUT_DIR)"
	$(DOCKER_RUN_NO_TTY) python3 -m time_series_model.pipeline.training.feature_type_evaluator \
		--data-path /workspace/$(DATA_DIR) \
		--symbol $(FEATURE_EVAL_SYMBOL) \
		$(if $(FEATURE_EVAL_START_DATE),--train-start $(FEATURE_EVAL_START_DATE),) \
		$(if $(FEATURE_EVAL_END_DATE),--train-end $(FEATURE_EVAL_END_DATE),) \
		--timeframe $(FEATURE_EVAL_TIMEFRAME) \
		--horizon $(FEATURE_EVAL_HORIZON) \
		--feature-types $(FEATURE_EVAL_TYPES) \
		--output-dir /workspace/$(FEATURE_EVAL_OUTPUT_DIR) \
		$(if $(FEATURE_EVAL_TOP_FACTORS_COUNT),--top-factors-count $(FEATURE_EVAL_TOP_FACTORS_COUNT),--top-factors-ic-threshold $(FEATURE_EVAL_TOP_FACTORS_IC_THRESHOLD)) \
		$(if $(filter 1 true yes,$(FEATURE_EVAL_TRAIN_ONLY)),--train-only --test-size $(FEATURE_EVAL_TEST_SIZE),)
	@echo "✅ Evaluation complete. Check results in $(FEATURE_EVAL_OUTPUT_DIR)"
	@echo "📄 top_factors.json generated for ts-r-rank-ic-train"

feature-eval:
	@echo "⚠️ 'feature-eval' has been renamed to 'ts-feature-eval'. Please update your workflows."
	@$(MAKE) ts-feature-eval


RANK_IC_SYMBOL ?= $(SYMBOL)
RANK_IC_HORIZON ?= 24
RANK_IC_TIMEFRAME ?= 240T
RANK_IC_FEATURE_TYPE ?= baseline
RANK_IC_N_SPLITS ?= 5
RANK_IC_TEST_SIZE ?= 0.15
RANK_IC_OUTPUT_DIR ?= results/rank_ic_training
# Filter to high-confidence samples only (strong trends)
# When enabled (1/true/yes), filters out choppy/consolidation periods where signals are weak,
# focusing training on periods with clear trends (trend_strength >= RANK_IC_MIN_TREND_STRENGTH).
# This can improve model performance by training only on high-quality samples.
# Default: 0 (disabled - use all samples)
RANK_IC_FILTER_HIGH_CONF ?= 0
# Minimum trend strength threshold for high-confidence filtering
# Only samples with trend_strength >= this value will be kept when RANK_IC_FILTER_HIGH_CONF is enabled
# Default: 1.0
RANK_IC_MIN_TREND_STRENGTH ?= 1.0
RANK_IC_SMOOTH_TARGET ?= 0
# Top factors file path (if not set, will use FEATURE_EVAL_OUTPUT_DIR/top_factors.json)
# Note: Data leakage detection has been moved to verify-feature-correlation target
# Set this to override the default path from feature evaluation
RANK_IC_TOP_FACTORS ?=
RANK_IC_TSCV_GAP ?= 24
# Signal generation method for trading signals
# Options:
#   - "quantile": Use quantile-based signals (default, most stable)
#   - "sign": Use prediction sign directly (simpler, may be more volatile)
#   - "hybrid": Combine sign and quantile methods (balanced approach)
#   - "optimized": Optimize threshold based on historical performance (requires future_return)
# Default: quantile
RANK_IC_SIGNAL_METHOD ?= hybrid
# Whether to calibrate predictions to match true return distribution
# When enabled (1/true/yes), uses sigmoid scaling to calibrate predictions,
# making them better match the actual return distribution (requires future_return in data).
# This can improve signal quality by reducing prediction bias.
# Default: 0 (disabled)
RANK_IC_CALIBRATE_PREDICTIONS ?= false

# Default to using top_factors.json from feature evaluation output
# If RANK_IC_TOP_FACTORS is explicitly set, use that instead
RANK_IC_TOP_FACTORS_PATH ?= $(if $(RANK_IC_TOP_FACTORS),$(RANK_IC_TOP_FACTORS),$(FEATURE_EVAL_OUTPUT_DIR)/top_factors.json)

ts-r-rank-ic-train:
	@echo "🎯 Rank IC Regression Training (TSCV + OOS Testing)..."
	@echo "   Symbol: $(RANK_IC_SYMBOL)"
	@echo "   Horizon: $(RANK_IC_HORIZON)"
	@echo "   Timeframe: $(RANK_IC_TIMEFRAME)"
	@echo "   Feature Type: $(RANK_IC_FEATURE_TYPE)"
	@echo "   Top Factors: $(if $(RANK_IC_TOP_FACTORS_PATH),$(RANK_IC_TOP_FACTORS_PATH),Not specified - will generate all features)"
	@echo "   TSCV Folds: $(RANK_IC_N_SPLITS)"
	@echo "   OOS Test Size: $(RANK_IC_TEST_SIZE)"
	@echo "   TSCV Gap: $(RANK_IC_TSCV_GAP)"
	@echo "   Signal Method: $(RANK_IC_SIGNAL_METHOD)"
	@echo "   Calibrate Predictions: $(RANK_IC_CALIBRATE_PREDICTIONS)"
	@echo "   Output: $(RANK_IC_OUTPUT_DIR)"

# SR Reversal Model Training
SR_REVERSAL_SYMBOL ?= $(SYMBOL)
SR_REVERSAL_HORIZON ?= 24
SR_REVERSAL_TIMEFRAME ?= 15T
SR_REVERSAL_FEATURE_TYPE ?= comprehensive
SR_REVERSAL_TEST_SIZE ?= 0.15
SR_REVERSAL_OUTPUT_DIR ?= results/sr_reversal_model
SR_SR_OPTUNA_STRATEGY ?= config/strategies/sr_reversal
SR_SR_OPTUNA_SYMBOL ?= $(SR_REVERSAL_SYMBOL)
SR_SR_OPTUNA_TIMEFRAME ?= $(SR_REVERSAL_TIMEFRAME)
SR_SR_OPTUNA_START ?=
SR_SR_OPTUNA_END ?=
SR_SR_OPTUNA_TEST_SIZE ?= 0.15
SR_SR_OPTUNA_WARMUP ?= 200
SR_SR_OPTUNA_TRIALS ?= 30
SR_SR_OPTUNA_OUTPUT ?= results/sr_reversal_optuna

ts-sr-reversal:
	@echo "🔄 Training SR Reversal Model..."
	@echo "   Symbol: $(SR_REVERSAL_SYMBOL)"
	@echo "   Horizon: $(SR_REVERSAL_HORIZON)"
	@echo "   Timeframe: $(SR_REVERSAL_TIMEFRAME)"
	@echo "   Feature Type: $(SR_REVERSAL_FEATURE_TYPE)"
	@echo "   Test Size: $(SR_REVERSAL_TEST_SIZE)"
	@echo "   Output: $(SR_REVERSAL_OUTPUT_DIR)"
	$(DOCKER_RUN_NO_TTY) python3 -m src.time_series_model.strategies.sr_reversal.train \
		--data-path /workspace/$(DATA_DIR) \
		--symbol $(SR_REVERSAL_SYMBOL) \
		--horizon $(SR_REVERSAL_HORIZON) \
		--timeframe $(SR_REVERSAL_TIMEFRAME) \
		--feature-type $(SR_REVERSAL_FEATURE_TYPE) \
		--test-size $(SR_REVERSAL_TEST_SIZE) \
		--output-dir /workspace/$(SR_REVERSAL_OUTPUT_DIR)

ts-sr-reversal-optuna:
	@echo "🔍 Optuna search for SR Reversal signal parameters..."
	@$(DOCKER_RUN_NO_TTY) python3 scripts/optimization/ts_sr_reversal_optuna.py \
		--strategy-config /workspace/$(SR_SR_OPTUNA_STRATEGY) \
		--symbol $(SR_SR_OPTUNA_SYMBOL) \
		--data-path /workspace/$(DATA_DIR) \
		--timeframe $(SR_SR_OPTUNA_TIMEFRAME) \
		$(if $(SR_SR_OPTUNA_START),--start-date $(SR_SR_OPTUNA_START),) \
		$(if $(SR_SR_OPTUNA_END),--end-date $(SR_SR_OPTUNA_END),) \
		--test-size $(SR_SR_OPTUNA_TEST_SIZE) \
		--test-warmup-bars $(SR_SR_OPTUNA_WARMUP) \
		--n-trials $(SR_SR_OPTUNA_TRIALS) \
		--output-dir /workspace/$(SR_SR_OPTUNA_OUTPUT)

# SR Breakout Model Training
SR_BREAKOUT_SYMBOL ?= $(SYMBOL)
SR_BREAKOUT_HORIZON ?= 24
SR_BREAKOUT_TIMEFRAME ?= 15T
SR_BREAKOUT_FEATURE_TYPE ?= comprehensive
SR_BREAKOUT_TEST_SIZE ?= 0.15
SR_BREAKOUT_OUTPUT_DIR ?= results/sr_breakout_model

ts-sr-breakout:
	@echo "📈 Training SR Breakout Model..."
	@echo "   Symbol: $(SR_BREAKOUT_SYMBOL)"
	@echo "   Horizon: $(SR_BREAKOUT_HORIZON)"
	@echo "   Timeframe: $(SR_BREAKOUT_TIMEFRAME)"
	@echo "   Feature Type: $(SR_BREAKOUT_FEATURE_TYPE)"
	@echo "   Test Size: $(SR_BREAKOUT_TEST_SIZE)"
	@echo "   Output: $(SR_BREAKOUT_OUTPUT_DIR)"
	$(DOCKER_RUN_NO_TTY) python3 -m src.time_series_model.strategies.sr_breakout.train \
		--data-path /workspace/$(DATA_DIR) \
		--symbol $(SR_BREAKOUT_SYMBOL) \
		--horizon $(SR_BREAKOUT_HORIZON) \
		--timeframe $(SR_BREAKOUT_TIMEFRAME) \
		--feature-type $(SR_BREAKOUT_FEATURE_TYPE) \
		--test-size $(SR_BREAKOUT_TEST_SIZE) \
		--output-dir /workspace/$(SR_BREAKOUT_OUTPUT_DIR)

# Compression Breakout Model Training
COMPRESSION_BREAKOUT_SYMBOL ?= $(SYMBOL)
COMPRESSION_BREAKOUT_HORIZON ?= 24
COMPRESSION_BREAKOUT_TIMEFRAME ?= 15T
COMPRESSION_BREAKOUT_FEATURE_TYPE ?= comprehensive
COMPRESSION_BREAKOUT_TEST_SIZE ?= 0.15
COMPRESSION_BREAKOUT_OUTPUT_DIR ?= results/compression_breakout_model

ts-compression-breakout:
	@echo "💥 Training Compression Breakout Model..."
	@echo "   Symbol: $(COMPRESSION_BREAKOUT_SYMBOL)"
	@echo "   Horizon: $(COMPRESSION_BREAKOUT_HORIZON)"
	@echo "   Timeframe: $(COMPRESSION_BREAKOUT_TIMEFRAME)"
	@echo "   Feature Type: $(COMPRESSION_BREAKOUT_FEATURE_TYPE)"
	@echo "   Test Size: $(COMPRESSION_BREAKOUT_TEST_SIZE)"
	@echo "   Output: $(COMPRESSION_BREAKOUT_OUTPUT_DIR)"
	$(DOCKER_RUN_NO_TTY) python3 -m src.time_series_model.strategies.compression_breakout.train \
		--data-path /workspace/$(DATA_DIR) \
		--symbol $(COMPRESSION_BREAKOUT_SYMBOL) \
		--horizon $(COMPRESSION_BREAKOUT_HORIZON) \
		--timeframe $(COMPRESSION_BREAKOUT_TIMEFRAME) \
		--feature-type $(COMPRESSION_BREAKOUT_FEATURE_TYPE) \
		--test-size $(COMPRESSION_BREAKOUT_TEST_SIZE) \
		--output-dir /workspace/$(COMPRESSION_BREAKOUT_OUTPUT_DIR)

# Trend Following Strategy Training
TREND_FOLLOWING_SYMBOL ?= $(SYMBOL)
TREND_FOLLOWING_HORIZON ?= 50
TREND_FOLLOWING_TIMEFRAME ?= 15T
TREND_FOLLOWING_FEATURE_TYPE ?= baseline,enhanced
TREND_FOLLOWING_TEST_SIZE ?= 0.15
TREND_FOLLOWING_OUTPUT_DIR ?= results/strategies/trend_following

ts-trend-following:
	@echo "📊 Training Trend Following Model..."
	@echo "   Symbol: $(TREND_FOLLOWING_SYMBOL)"
	@echo "   Horizon: $(TREND_FOLLOWING_HORIZON)"
	@echo "   Timeframe: $(TREND_FOLLOWING_TIMEFRAME)"
	@echo "   Feature Type: $(TREND_FOLLOWING_FEATURE_TYPE)"
	@echo "   Test Size: $(TREND_FOLLOWING_TEST_SIZE)"
	@echo "   Output: $(TREND_FOLLOWING_OUTPUT_DIR)"
	$(DOCKER_RUN_NO_TTY) python3 -m src.time_series_model.strategies.trend_following.train \
		--data-path /workspace/$(DATA_DIR) \
		--symbol $(TREND_FOLLOWING_SYMBOL) \
		--horizon $(TREND_FOLLOWING_HORIZON) \
		--timeframe $(TREND_FOLLOWING_TIMEFRAME) \
		--feature-type $(TREND_FOLLOWING_FEATURE_TYPE) \
		--test-size $(TREND_FOLLOWING_TEST_SIZE) \
		--output-dir /workspace/$(TREND_FOLLOWING_OUTPUT_DIR)
	$(DOCKER_RUN_NO_TTY) python3 -m time_series_model.pipeline.training.train_rank_ic_standalone \
		--data-path /workspace/$(DATA_DIR) \
		--symbol $(RANK_IC_SYMBOL) \
		$(if $(FEATURE_EVAL_START_DATE),--train-start $(FEATURE_EVAL_START_DATE),) \
		$(if $(FEATURE_EVAL_END_DATE),--train-end $(FEATURE_EVAL_END_DATE),) \
		--horizon $(RANK_IC_HORIZON) \
		--timeframe $(RANK_IC_TIMEFRAME) \
		--feature-type $(RANK_IC_FEATURE_TYPE) \
		--n-splits $(RANK_IC_N_SPLITS) \
		--test-size $(RANK_IC_TEST_SIZE) \
		--tscv-gap $(RANK_IC_TSCV_GAP) \
		--output-dir /workspace/$(RANK_IC_OUTPUT_DIR) \
		$(if $(filter 1 true yes,$(RANK_IC_FILTER_HIGH_CONF)),--filter-high-confidence,) \
		--min-trend-strength $(RANK_IC_MIN_TREND_STRENGTH) \
		$(if $(filter 1 true yes,$(RANK_IC_SMOOTH_TARGET)),--smooth-target,) \
		$(if $(RANK_IC_TOP_FACTORS_PATH),--top-factors /workspace/$(RANK_IC_TOP_FACTORS_PATH),) \
		--signal-method $(RANK_IC_SIGNAL_METHOD) \
		$(if $(filter 1 true yes,$(RANK_IC_CALIBRATE_PREDICTIONS)),--calibrate-predictions,)
	@echo "✅ Training complete. Check results in $(RANK_IC_OUTPUT_DIR)"


# Verify feature correlation (distinguish real Alpha vs data leakage)
verify-feature-correlation:
	@echo "🔬 Verifying Feature Correlation (Real Alpha vs Data Leakage)..."
	@echo "   Symbol: $(RANK_IC_SYMBOL)"
	@echo "   Horizon: $(RANK_IC_HORIZON)"
	@echo "   Timeframe: $(RANK_IC_TIMEFRAME)"
	@echo "   Top Factors: $(if $(RANK_IC_TOP_FACTORS_PATH),$(RANK_IC_TOP_FACTORS_PATH),Not specified - will use all features)"
	$(DOCKER_RUN_NO_TTY) python3 scripts/verify_feature_correlation.py \
		--data-path /workspace/$(DATA_DIR) \
		--symbol $(RANK_IC_SYMBOL) \
		$(if $(FEATURE_EVAL_START_DATE),--start-date $(FEATURE_EVAL_START_DATE),) \
		$(if $(FEATURE_EVAL_END_DATE),--end-date $(FEATURE_EVAL_END_DATE),) \
		--timeframe $(RANK_IC_TIMEFRAME) \
		--horizon $(RANK_IC_HORIZON) \
		$(if $(RANK_IC_TOP_FACTORS_PATH),--top-factors /workspace/$(RANK_IC_TOP_FACTORS_PATH),)
	@echo "✅ Verification complete."



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

cs-build-panel:
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

cs-report:
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

cs-train:
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

cs-workflow:
	@echo "🔄 Running end-to-end cross-sectional pipeline..."
	$(MAKE) cs-build-panel
	$(MAKE) cs-report CS_INPUT="$(CS_BUILD_OUTPUT)" SYMBOLS="$(CS_BUILD_SYMBOLS)" CS_HORIZON=$(CS_BUILD_HORIZON)
	$(MAKE) cs-train CS_TRAIN_INPUT="$(CS_BUILD_OUTPUT)" SYMBOLS="$(CS_BUILD_SYMBOLS)" CS_HORIZON=$(CS_BUILD_HORIZON)

CS_CATALOG_INPUT ?= $(CS_BUILD_OUTPUT)
CS_CATALOG_OUTPUT ?= results/cross_sectional/factor_sets

cs-catalog:
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

cs-select:
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

cs-shap:
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

cs-logic-check:
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

cs-shap-drift:
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

cs-auto:
	@echo "🤖 Running fully automated cross-sectional pipeline..."
	$(MAKE) cs-build-panel
	$(MAKE) cs-select \
		CS_SELECT_INPUT="$(CS_BUILD_OUTPUT)" \
		CS_SELECT_OUTPUT="$(CS_AUTO_FEATURE_FILE)" \
		CS_SELECT_OUTPUT_JSON="results/cross_sectional/selection_summary.json" \
		CS_SELECT_MIN_ASSETS=$(CS_AUTO_MIN_ASSETS) \
		CS_SELECT_PER_CATEGORY_TOP=$(CS_AUTO_PER_CATEGORY_TOP) \
		CS_SELECT_GLOBAL_TOP=$(CS_AUTO_GLOBAL_TOP) \
		CS_SELECT_IC_THRESHOLD=$(CS_AUTO_IC_THRESHOLD) \
		CS_SELECT_IR_THRESHOLD=$(CS_AUTO_IR_THRESHOLD)
	$(MAKE) cs-report \
		CS_INPUT="$(CS_BUILD_OUTPUT)" \
		SYMBOLS="$(CS_BUILD_SYMBOLS)" \
		CS_HORIZON=$(CS_BUILD_HORIZON) \
		CS_REPORT_EXTRA="--feature-file $(CS_AUTO_FEATURE_FILE)"
	$(MAKE) cs-train \
		CS_TRAIN_INPUT="$(CS_BUILD_OUTPUT)" \
		SYMBOLS="$(CS_BUILD_SYMBOLS)" \
		CS_HORIZON=$(CS_BUILD_HORIZON) \
		CS_PERIODS_PER_YEAR=$(CS_PERIODS_PER_YEAR) \
		CS_TRAIN_FEATURE_FILE="$(CS_AUTO_FEATURE_FILE)"
	$(MAKE) cs-shap \
		CS_SHAP_MODEL="$(CS_TRAIN_OUTPUT_DIR)/$(CS_TRAIN_MODEL_NAME)" \
		CS_SHAP_PANEL="$(CS_BUILD_OUTPUT)" \
		CS_SHAP_FEATURE_FILE="$(CS_AUTO_FEATURE_FILE)"
	$(MAKE) cs-logic-check \
		CS_LOGIC_EXPECTATIONS="$(CS_LOGIC_EXPECTATIONS)"
	$(MAKE) cs-shap-drift \
		CS_DRIFT_BASELINE="$(CS_DRIFT_BASELINE)"



# ---------------------------------------------------------------------------
# Alphalens test (verify installation and basic functionality)
# ---------------------------------------------------------------------------

test-alphalens:
	@echo "🧪 Testing Alphalens installation and basic functionality in Docker..."
	$(DOCKER_RUN_NO_TTY) python3 scripts/test_alphalens.py

# ---------------------------------------------------------------------------
# WPT Volume Profile Tests
# ---------------------------------------------------------------------------

test-wpt-volume-profile:
	@echo "🧪 Testing WPT Volume Profile improvements (pytest format)..."
	@$(DOCKER_RUN_NO_TTY) python3 -m pytest tests/test_wpt_volume_profile_improvements.py -v

test-wpt-volume-profile-simple:
	@echo "🧪 Testing WPT Volume Profile improvements (simple script)..."
	@$(DOCKER_RUN_NO_TTY) python3 tests/test_wpt_improvements_simple.py

alphalens-example:
	@echo "📊 Running complete Alphalens example with comprehensive analysis..."
	$(DOCKER_RUN_NO_TTY) python3 scripts/alphalens_example.py

alphalens-evaluate:
	@echo "📊 Evaluating trading signal quality using Alphalens..."
	$(DOCKER_RUN_NO_TTY) python3 scripts/alphalens_evaluate_predictions.py


CS_FACTOR_FEATURES_CONFIG ?= config/tests/factor_test/features.yaml
CS_FACTOR_SYMBOLS ?= BTCUSDT,ETHUSDT
CS_FACTOR_TIMEFRAME ?= 240T
CS_FACTOR_HORIZON ?= 24
CS_FACTOR_QUANTILES ?= 5
CS_FACTOR_IC_LAGS ?= 1,3,5
CS_FACTOR_MIN_XS ?= 3
CS_FACTOR_OUTPUT_DIR ?= results/cross_sectional_eval

cs-factor-eval:
	@echo "📊 Cross-sectional factor evaluation for $(CS_FACTOR_SYMBOLS) ($(START_DATE) → $(END_DATE))"
	@$(DOCKER_RUN_NO_TTY) python3 scripts/factor_management/cross_sectional_eval.py \
		--features-config $(CS_FACTOR_FEATURES_CONFIG) \
		--symbols $(CS_FACTOR_SYMBOLS) \
		--data-path /workspace/$(DATA_DIR) \
		--timeframe $(CS_FACTOR_TIMEFRAME) \
		$(if $(START_DATE),--start-date $(START_DATE),) \
		$(if $(END_DATE),--end-date $(END_DATE),) \
		--horizon $(CS_FACTOR_HORIZON) \
		--quantiles $(CS_FACTOR_QUANTILES) \
		--ic-decay-lags $(CS_FACTOR_IC_LAGS) \
		--min-cross-sectional $(CS_FACTOR_MIN_XS) \
		--output-dir /workspace/$(CS_FACTOR_OUTPUT_DIR)
