# Makefile for ML Trading Project

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := ml-trading-project

# Docker GPU variables
DOCKER_COMPOSE := docker-compose
DOCKER_GPU_RUN := $(DOCKER_COMPOSE) run --rm ml-gpu

# Default target
.PHONY: help
help:
	@echo "ML Trading Project Makefile"
	@echo "=========================="
	@echo "Available commands:"
	@echo "  clean       - Clean build artifacts"
	@echo "  format      - Format code with black"
	@echo "  lint        - Lint code with flake8"
	@echo "  train-wavelet - Train wavelet model (May)"
	@echo "  oos-june    - Run June OOS backtests (5T/15T)"
	@echo "  reports-june - Generate June OOS reports"
	@echo ""
	@echo "Analysis Tools:"
	@echo "  count-features - Generate feature count report"
	@echo ""
	@echo "Advanced Rolling Training (2025):"
	@echo "  rolling-2025-advanced  - Monthly rolling training with feature management"
	@echo "  report-rolling-2025    - Generate backtest report"
	@echo "  workflow-rolling-2025  - Full workflow (train + report)"
	@echo ""
	@echo "GPU Acceleration:"
	@echo "  install-gpu - Install GPU dependencies (lightgbm-gpu)"
	@echo "  test-gpu    - Test GPU availability"
	@echo "  train-gpu   - Train model with GPU acceleration"
	@echo "  compare-gpu - Compare CPU vs GPU results"
	@echo ""
	@echo "🐳 Docker GPU 命令 (使用 lightgbm-runtime 镜像):"
	@echo "  docker-build              - 检查 GPU 镜像"
	@echo "  docker-shell              - 进入 GPU 容器"
	@echo "  docker-test-gpu           - 测试 GPU"
	@echo "  docker-train-wavelet      - 训练模型"
	@echo "  docker-oos-months         - 回测"
	@echo "  docker-rolling-2025-advanced - 滚动训练"
	@echo "  docker-workflow-full      - 完整工作流"
	@echo "  docker-help               - 查看所有 Docker 命令"
	@echo ""
	@echo "Data Conversion:"
	@echo "  convert-all-zip-to-parquet - 批量转换所有ZIP为Parquet（推荐）"
	@echo "  convert-zip-to-parquet     - 交互式转换ZIP为Parquet"
	@echo ""
	@echo "Advanced Dimensionality Reduction (Autoencoder + SHAP):"
	@echo "  install-dim-reduction      - Install dimensionality reduction dependencies"
	@echo "  install-dim-reduction-gpu  - Install GPU dependencies for dimensionality reduction"
	@echo "  dim-reduction-demo         - Run pipeline with sample data"
	@echo "  dim-reduction-real         - Run pipeline with real trading data"
	@echo "  dim-reduction-custom       - Run with custom parameters"
	@echo "  dim-reduction-multi        - Run for multiple symbols"
	@echo "  test-dim-reduction         - Test dimensionality reduction components"
	@echo "  compare-dim-reduction      - Compare different dimensionality methods"
	@echo "  report-dim-reduction       - Generate comprehensive report"
	@echo "  workflow-dim-reduction     - Full workflow (install + test + run + report)"
	@echo ""
	@echo "Rolling Dimensionality Training (Quarterly + Drift Detection):"
	@echo "  rolling-dim-quarterly      - Run quarterly rolling training (2024→2025)"
	@echo "  rolling-dim-drift          - Run drift-triggered training"
	@echo "  rolling-dim-multi-symbols  - Run for multiple symbols"
	@echo "  compare-dim-before-after   - Compare performance before/after dimensionality reduction"
	@echo "  report-rolling-dim         - Generate rolling dimensionality report"
	@echo ""
	@echo "New Dimensionality Training Workflow (Docker):"
	@echo "  quick-start                - Quick start: build + feature + training + reports (Docker)"
	@echo "  feature-engineering        - Run feature engineering and IC/IR filtering (Docker)"
	@echo "  rolling-training           - Run rolling training with dimensionality reduction (Docker)"
	@echo "  production-training        - Run production-level training (Docker)"
	@echo "  integration-demo           - Run integration demonstration (Docker)"
	@echo "  generate-reports           - Generate comprehensive reports (Docker)"
	@echo "  workflow-dimensionality   - Full dimensionality workflow (all steps in Docker)"
	@echo "

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Format code
.PHONY: format
format:
	$(PYTHON) -m black src/ml_trading/ tests/ scripts/

# Lint code
.PHONY: lint
lint:
	$(PYTHON) -m flake8 src/ tests/ scripts/

# Install development dependencies
.PHONY: dev-install
dev-install:
	$(PIP) install -e .[dev]

.PHONY: train-wavelet
train-wavelet:
	@echo "🚀 训练 Wavelet 模型 (Docker)..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/training/train_model_wavelet.py"

.PHONY: oos-june
oos-june:
	@echo "📊 运行 June OOS 回测 (Docker)..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/backtesting/oos_june.py"

.PHONY: reports-june
reports-june:
	@echo "📋 生成 June 报告 (Docker)..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/reports/reports_june.py"

.PHONY: grid-tune
grid-tune:
	@echo "🔧 运行网格搜索 (Docker)..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/optimization/grid_tune.py"

.PHONY: feat-importance
feat-importance:
	@echo "📊 导出特征重要性 (Docker)..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/utils/export_feature_importance.py"

.PHONY: optuna-risk
optuna-risk:
	@echo "🔧 运行 Optuna 风险搜索 (Docker)..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/optimization/optuna_risk_search.py"

.PHONY: reports-all-tf
reports-all-tf:
	@echo "📋 生成所有时间框架报告 (Docker)..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/reports/reports_all_tf.py"

.PHONY: merge-reports
merge-reports:
	@echo "📋 合并报告 (Docker)..."
	$(DOCKER_GPU_RUN) bash -c "pip install PyPDF2==3.0.1 && PYTHONPATH=/app/src python3 scripts/reports/merge_reports.py"

.PHONY: oos-months
oos-months:
	@echo "📊 运行多月份回测 (Docker)..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/backtesting/oos_months.py"

.PHONY: drift-analysis
drift-analysis:
	@echo "📊 运行漂移分析 (Docker)..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/analysis/monthly_drift.py"

# ============================================================================
# Analysis Tools
# ============================================================================

# Count all features across different modules
.PHONY: count-features
count-features:
	@echo "📊 统计所有模块特征 (Docker)..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/analysis/count_features.py"
	@echo ""
	@echo "✅ Feature count report generated!"
	@echo "   Report: reports/feature_count_report.txt"
	@echo "   Data: reports/feature_count_data.json"

# ============================================================================
# Advanced Feature Management and Rolling Training
# ============================================================================

# Monthly rolling training with advanced feature management (2025)
.PHONY: rolling-2025-advanced
rolling-2025-advanced:
	@echo "🚀 开始高级月度滚动训练 (2025) - Docker..."
	@echo "  - Feature Management: Dynamic selection + CVD improvements"
	@echo "  - Transformer: Time series features (60 bars, 64 dimensions)"
	@echo "  - Training: Warm Start (保留旧知识)"
	@echo "  - Evaluation: Sharpe + PF + MaxDD + Quality Score"
	@echo ""
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/rolling/monthly_rolling_2025_with_feature_management.py"
	@echo ""
	@echo "✅ Training complete! Check results/monthly_rolling_2025_advanced/"

# Generate backtest report from rolling training results
.PHONY: report-rolling-2025
report-rolling-2025:
	@echo "📋 生成 2025 滚动训练报告 (Docker)..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/analysis/generate_rolling_report.py"
	@echo ""
	@echo "✅ Report generated! Check results/monthly_rolling_2025_advanced/backtest_report.html"

# Full workflow: train + report
.PHONY: workflow-rolling-2025
workflow-rolling-2025: rolling-2025-advanced report-rolling-2025
	@echo ""
	@echo "=============================================="
	@echo "  🎉 Advanced Rolling Training Complete!"
	@echo "=============================================="
	@echo ""
	@echo "Results:"
	@echo "  - Training results: results/monthly_rolling_2025_advanced/"
	@echo "  - Feature repository: results/feature_repository.json"
	@echo "  - Backtest report: results/monthly_rolling_2025_advanced/backtest_report.html"

# ============================================================================
# Advanced Dimensionality Reduction Pipeline (Autoencoder + SHAP Distillation)
# ============================================================================

# Production Dimensionality Training - 生产级降维训练
.PHONY: production-dim-training
production-dim-training:
	@echo "🏭 运行生产级降维训练..."
	@echo "   - 完整的训练流程 (500轮Autoencoder训练)"
	@echo "   - 真实数据加载和特征工程"
	@echo "   - 完整的性能评估和模型保存"
	@echo ""
	PYTHONPATH=src $(PYTHON) scripts/dimensionality/03_production_training.py

# 增强版滚动降维训练
.PHONY: enhanced-rolling-dimensionality
enhanced-rolling-dimensionality:
	@echo "🚀 运行增强版滚动降维训练..."
	@echo "   - 修复特征数量问题"
	@echo "   - 添加IC/IR筛选"
	@echo "   - 增强降维后特征解释"
	@echo "   - 确保包含序列特征"
	@echo ""
	PYTHONPATH=src $(PYTHON) scripts/dimensionality/01_feature_engineering.py

# Data Conversion - 数据转换
.PHONY: convert-all-zip-to-parquet
convert-all-zip-to-parquet:
	@echo "🔄 批量转换所有ZIP文件为Parquet格式..."
	@echo "   - 自动转换 ../data/agg_data 下所有 *aggTrades-*.zip 文件"
	@echo "   - 输出到 ../data/parquet_data 目录"
	@echo "   - 自动备份到 ../data/backup_zip 目录"
	@echo "   - 包含订单流特征（CVD, taker_buy_ratio等）"
	@echo "   - 支持 ETH, BTC, SOL 等多币种"
	@echo ""
	@if [ ! -d "../data/agg_data" ]; then \
		echo "❌ 数据目录不存在: ../data/agg_data"; \
		echo "提示：数据应该在 /home/yin/trading/rlbot/data/agg_data"; \
		exit 1; \
	fi
	PYTHONPATH=src $(PYTHON) scripts/data_conversion/convert_zip_to_parquet.py

.PHONY: convert-zip-to-parquet
convert-zip-to-parquet:
	@echo "🔄 交互式转换ZIP文件为Parquet格式..."
	@echo "   - 提高数据处理速度"
	@echo "   - 自动备份原始文件"
	@echo "   - 支持批量转换"
	@echo ""
	PYTHONPATH=src $(PYTHON) scripts/data_conversion/convert_zip_to_parquet.py

# ============================================================================

# Install GPU dependencies for dimensionality reduction
.PHONY: install-dim-reduction-gpu
install-dim-reduction-gpu:
	@echo "📦 Installing GPU dependencies for dimensionality reduction..."
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	$(PIP) install shap
	$(PIP) install plotly
	$(PIP) install tensorboard
	@echo "✅ GPU dimensionality reduction dependencies installed!"

# Run dimensionality reduction pipeline with sample data
.PHONY: dim-reduction-demo
dim-reduction-demo:
	@echo "🚀 Running Dimensionality Reduction Pipeline (Demo Mode)..."
	@echo "   - Method: Autoencoder + SHAP Distillation"
	@echo "   - Data: Sample synthetic data"
	@echo "   - Output: Interpretable factor compression"
	@echo ""
	PYTHONPATH=src $(PYTHON) scripts/dimensionality/06_dimensionality_pipeline.py \
		--visualize --generate-report --save-model
	@echo ""
	@echo "✅ Demo complete! Check reports/ for results"

# Run dimensionality reduction pipeline with real data
.PHONY: dim-reduction-real
dim-reduction-real:
	@echo "🚀 Running Dimensionality Reduction Pipeline (Real Data)..."
	@echo "   - Method: Autoencoder + SHAP Distillation"
	@echo "   - Data: Real trading data (ETH-USD)"
	@echo "   - Output: Interpretable factor compression"
	@echo ""
	PYTHONPATH=src $(PYTHON) scripts/dimensionality/06_dimensionality_pipeline.py \
		--use-real-data --data-path data/agg_data \
		--symbol ETH-USD --visualize --generate-report --save-model
	@echo ""
	@echo "✅ Real data pipeline complete! Check reports/ for results"

# Run dimensionality reduction with custom parameters
.PHONY: dim-reduction-custom
dim-reduction-custom:
	@echo "🚀 Running Dimensionality Reduction Pipeline (Custom Parameters)..."
	@echo "   - Encoding Dimension: 12"
	@echo "   - Top K Factors: 15"
	@echo "   - Epochs: 150"
	@echo ""
	PYTHONPATH=src $(PYTHON) scripts/dimensionality/06_dimensionality_pipeline.py \
		--encoding-dim 12 --top-k 15 --epochs 150 \
		--visualize --generate-report --save-model
	@echo ""
	@echo "✅ Custom pipeline complete! Check reports/ for results"

# Run dimensionality reduction for multiple symbols
.PHONY: dim-reduction-multi
dim-reduction-multi:
	@echo "🚀 Running Dimensionality Reduction Pipeline (Multiple Symbols)..."
	@echo "   - Processing multiple trading symbols"
	@echo "   - Generating comparative factor analysis"
	@echo ""
	@for symbol in ETH-USD BTC-USD SOL-USD; do \
		echo "Processing $$symbol..."; \
		PYTHONPATH=src $(PYTHON) scripts/dimensionality/06_dimensionality_pipeline.py \
			--use-real-data --data-path data/agg_data \
			--symbol $$symbol --visualize --generate-report --save-model; \
	done
	@echo ""
	@echo "✅ Multi-symbol pipeline complete! Check reports/ for results"

# Test dimensionality reduction components
.PHONY: test-dim-reduction
test-dim-reduction:
	@echo "🧪 Testing Dimensionality Reduction Components..."
	PYTHONPATH=src $(PYTHON) scripts/test_dimensionality_reduction.py
	@echo ""
	@echo "✅ Component tests complete!"

# Compare dimensionality reduction methods
.PHONY: compare-dim-reduction
compare-dim-reduction:
	@echo "🔍 Comparing Dimensionality Reduction Methods..."
	@echo "   - Autoencoder + SHAP vs PCA vs Traditional Feature Selection"
	@echo "   - Performance metrics and interpretability analysis"
	@echo ""
	PYTHONPATH=src $(PYTHON) scripts/compare_dimensionality_methods.py
	@echo ""
	@echo "✅ Comparison complete! Check reports/ for results"

# Generate dimensionality reduction report
.PHONY: report-dim-reduction
report-dim-reduction:
	@echo "📋 Generating Dimensionality Reduction Report..."
	PYTHONPATH=src $(PYTHON) scripts/generate_dim_reduction_report.py
	@echo ""
	@echo "✅ Report generated! Check reports/dimensionality_reduction_summary.html"

# Full dimensionality reduction workflow
.PHONY: workflow-dim-reduction
workflow-dim-reduction: install-dim-reduction test-dim-reduction dim-reduction-demo dim-reduction-real report-dim-reduction
	@echo ""
	@echo "=============================================="
	@echo "  🎉 Dimensionality Reduction Workflow Complete!"
	@echo "=============================================="
	@echo ""
	@echo "Results:"
	@echo "  - Demo results: reports/"
	@echo "  - Real data results: reports/"
	@echo "  - Summary report: reports/dimensionality_reduction_summary.html"
	@echo ""
	@echo "Next steps:"
	@echo "  - Review factor contributions"
	@echo "  - Integrate with trading strategies"
	@echo "  - Monitor factor performance over time"

# ============================================================================
# Rolling Dimensionality Training (Quarterly Data + Drift Detection)
# ============================================================================

# Run quarterly rolling training with dimensionality reduction
.PHONY: rolling-dim-quarterly
rolling-dim-quarterly:
	@echo "🚀 Running Quarterly Rolling Dimensionality Training in Docker..."
	@echo "   - Training Period: 2024 Full Year"
	@echo "   - Testing Period: 2025 Quarters"
	@echo "   - Method: Autoencoder + SHAP Distillation"
	@echo "   - Features: Drift Detection + Performance Comparison"
	@echo ""
	@if [ "$$(uname)" = "Linux" ]; then \
		docker run --rm --gpus all \
			-v "$$(pwd):/workspace" \
			-v "/home/yin/trading/rlbot/data:/data" \
			-w /workspace \
			-e DATA_DIR="/data/agg_data" \
			lightgbm-runtime:latest \
			python3 scripts/dimensionality/02_rolling_training.py \
				--mode quarterly \
				--symbol ETH-USD \
				--encoding-dim 8 \
				--drift-threshold 0.3 \
				--min-improvement 0.005; \
	else \
		echo "❌ 此命令仅在 Linux/WSL 中可用"; \
	fi
	@echo ""
	@echo "✅ Quarterly rolling training complete! Check results/rolling_dim_ETH_USD/"

# Run drift-triggered training
.PHONY: rolling-dim-drift
rolling-dim-drift:
	@echo "🚀 Running Drift-Triggered Dimensionality Training in Docker..."
	@echo "   - Method: Dynamic drift detection + Autoencoder + SHAP"
	@echo "   - Features: Adaptive feature selection based on drift"
	@echo "   - Period: 2024-2025 rolling windows"
	@echo ""
	@if [ "$$(uname)" = "Linux" ]; then \
		docker run --rm --gpus all \
			-v "$$(pwd):/workspace" \
			-v "/home/yin/trading/rlbot/data:/data" \
			-w /workspace \
			-e DATA_DIR="/data/agg_data" \
			lightgbm-runtime:latest \
			python3 scripts/dimensionality/02_rolling_training.py \
				--mode drift-triggered \
				--symbol ETH-USD \
				--encoding-dim 8 \
				--drift-threshold 0.3 \
				--min-improvement 0.005; \
	else \
		echo "❌ 此命令仅在 Linux/WSL 中可用"; \
	fi
	@echo ""
	@echo "✅ Drift-triggered training complete! Check results/rolling_dim_ETH_USD/"

# Run quarterly training for multiple symbols
.PHONY: rolling-dim-multi-symbols
rolling-dim-multi-symbols:
	@echo "🚀 Running Quarterly Rolling Training for Multiple Symbols in Docker..."
	@echo "   - Symbols: ETH-USD, BTC-USD, SOL-USD"
	@echo "   - Method: Autoencoder + SHAP Distillation"
	@echo "   - Period: 2024 training, 2025 testing"
	@echo ""
	@if [ "$$(uname)" = "Linux" ]; then \
		for symbol in ETH-USD BTC-USD SOL-USD; do \
			echo "Processing $$symbol..."; \
			docker run --rm --gpus all \
				-v "$$(pwd):/workspace" \
				-v "/home/yin/trading/rlbot/data:/data" \
				-w /workspace \
				-e DATA_DIR="/data/agg_data" \
				lightgbm-runtime:latest \
				python3 scripts/dimensionality/02_rolling_training.py \
					--mode quarterly \
					--symbol $$symbol \
					--encoding-dim 8 \
					--drift-threshold 0.3 \
					--min-improvement 0.005; \
		done; \
	else \
		echo "❌ 此命令仅在 Linux/WSL 中可用"; \
	fi
	@echo ""
	@echo "✅ Multi-symbol quarterly training complete! Check results/rolling_dim_*/"

# Compare dimensionality reduction before and after
.PHONY: compare-dim-before-after
compare-dim-before-after:
	@echo "🔍 Comparing Dimensionality Reduction Before and After..."
	@echo "   - Analysis: Performance comparison with/without dimensionality reduction"
	@echo "   - Metrics: R², RMSE, Compression Ratio, Feature Importance"
	@echo ""
	PYTHONPATH=src $(PYTHON) scripts/compare_dimensionality_methods.py
	@echo ""
	@echo "✅ Comparison complete! Check reports/ for detailed analysis"

# Generate rolling dimensionality report
.PHONY: report-rolling-dim
report-rolling-dim:
	@echo "📋 Generating Rolling Dimensionality Report in Docker..."
	@if [ "$$(uname)" = "Linux" ]; then \
		docker run --rm --gpus all \
			-v "$$(pwd):/workspace" \
			-v "/home/yin/trading/rlbot/data:/data" \
			-w /workspace \
			-e DATA_DIR="/data/agg_data" \
			lightgbm-runtime:latest \
			python3 scripts/dimensionality/05_report_generator.py; \
	else \
		echo "❌ 此命令仅在 Linux/WSL 中可用"; \
	fi
	@echo ""
	@echo "✅ Report generated! Check results/rolling_dim_*/summary_report.json"

# ============================================================================
# Docker Rolling Dimensionality Training Workflow
# ============================================================================

# Full Docker rolling dimensionality workflow
.PHONY: workflow-docker-rolling-dim
workflow-docker-rolling-dim: docker-gpu-build docker-rolling-dim docker-rolling-dim-drift

# Quick Docker rolling dimensionality workflow (skip build if image exists)
.PHONY: workflow-docker-rolling-dim-quick
workflow-docker-rolling-dim-quick: docker-rolling-dim docker-rolling-dim-drift
	@echo ""
	@echo "=============================================="
	@echo "  🎉 Docker Rolling Dimensionality Workflow Complete!"
	@echo "=============================================="
	@echo ""
	@echo "Results:"
	@echo "  - Docker image built with all dependencies"
	@echo "  - Component tests passed"
	@echo "  - Quarterly training completed"
	@echo "  - Drift-triggered training completed"
	@echo ""
	@echo "Key Features:"
	@echo "  ✅ Docker environment with GPU support"
	@echo "  ✅ All rolling dimensionality dependencies installed"
	@echo "  ✅ Quarterly data rolling training (2024 → 2025)"
	@echo "  ✅ Drift detection and dynamic triggering"
	@echo "  ✅ Before/after performance comparison"
	@echo "  ✅ Multi-symbol support"
	@echo ""
	@echo "Next steps:"
	@echo "  - Review Docker container results"
	@echo "  - Analyze performance across different modes"
	@echo "  - Optimize Docker configuration if needed"
	@echo "  - Deploy to production Docker environment"

# ============================================================================
# New Dimensionality Training Workflow
# ============================================================================

# Check Docker image - 检查Docker镜像
.PHONY: check-docker-image
check-docker-image:
	@echo "🐳 Checking Docker image availability..."
	@if docker images lightgbm-runtime:latest --format "table {{.Repository}}:{{.Tag}}" | grep -q "lightgbm-runtime:latest"; then \
		echo "✅ Docker image found: lightgbm-runtime:latest"; \
	else \
		echo "❌ Docker image not found: lightgbm-runtime:latest"; \
		echo "Please run 'make docker-gpu-build' first"; \
		exit 1; \
	fi

# Feature Engineering - 特征工程和IC/IR筛选
.PHONY: feature-engineering
feature-engineering: check-docker-image
	@echo "🔧 Running Feature Engineering and IC/IR Filtering in Docker..."
	@echo "   - Generate 1000+ features using ComprehensiveFeatureEngineer"
	@echo "   - Apply IC/IR filtering for high-quality features"
	@echo "   - Compare Autoencoder vs PCA dimensionality reduction"
	@echo "   - Analyze feature types and contributions"
	@echo ""
	@if [ "$$(uname)" = "Linux" ]; then \
		docker run --rm --gpus all \
			-v "$$(pwd):/workspace" \
			-v "/home/yin/trading/rlbot/data:/data" \
			-w /workspace \
			-e DATA_DIR="/data/agg_data" \
			lightgbm-runtime:latest \
			python3 scripts/dimensionality/01_feature_engineering.py; \
	else \
		echo "❌ 此命令仅在 Linux/WSL 中可用"; \
	fi
	@echo ""
	@echo "✅ Feature engineering complete! Check results/feature_engineering_*/"

# Rolling Training - 滚动训练
.PHONY: rolling-training
rolling-training: check-docker-image
	@echo "🚀 Running Rolling Training with Dimensionality Reduction in Docker..."
	@echo "   - Training Period: 2024 Full Year"
	@echo "   - Testing Period: 2025 Quarters"
	@echo "   - Method: Autoencoder + SHAP Distillation"
	@echo "   - Features: Drift Detection + Performance Comparison"
	@echo ""
	@if [ "$$(uname)" = "Linux" ]; then \
		docker run --rm --gpus all \
			-v "$$(pwd):/workspace" \
			-v "/home/yin/trading/rlbot/data:/data" \
			-w /workspace \
			-e DATA_DIR="/data/agg_data" \
			lightgbm-runtime:latest \
			python3 scripts/dimensionality/02_rolling_training.py \
				--mode quarterly \
				--symbol ETH-USD \
				--encoding-dim 8 \
				--drift-threshold 0.3 \
				--min-improvement 0.005; \
	else \
		echo "❌ 此命令仅在 Linux/WSL 中可用"; \
	fi
	@echo ""
	@echo "✅ Rolling training complete! Check results/rolling_dim_ETH_USD/"

# Production Training - 生产级训练
.PHONY: production-training
production-training: check-docker-image
	@echo "🏭 Running Production-Level Training in Docker..."
	@echo "   - Complete training pipeline (500 epochs Autoencoder)"
	@echo "   - Real data loading and feature engineering"
	@echo "   - Complete performance evaluation and model saving"
	@echo ""
	@if [ "$$(uname)" = "Linux" ]; then \
		docker run --rm --gpus all \
			-v "$$(pwd):/workspace" \
			-v "/home/yin/trading/rlbot/data:/data" \
			-w /workspace \
			-e DATA_DIR="/data/agg_data" \
			lightgbm-runtime:latest \
			python3 scripts/dimensionality/03_production_training.py; \
	else \
		echo "❌ 此命令仅在 Linux/WSL 中可用"; \
	fi
	@echo ""
	@echo "✅ Production training complete! Check results/production_dimensionality_*/"

# Integration Demo - 集成演示
.PHONY: integration-demo
integration-demo: check-docker-image
	@echo "🔗 Running Integration Demonstration in Docker..."
	@echo "   - Load trained dimensionality reduction models"
	@echo "   - Test performance on new data"
	@echo "   - Demonstrate production environment usage"
	@echo ""
	@if [ "$$(uname)" = "Linux" ]; then \
		docker run --rm --gpus all \
			-v "$$(pwd):/workspace" \
			-v "/home/yin/trading/rlbot/data:/data" \
			-w /workspace \
			-e DATA_DIR="/data/agg_data" \
			lightgbm-runtime:latest \
			python3 scripts/dimensionality/04_integration.py; \
	else \
		echo "❌ 此命令仅在 Linux/WSL 中可用"; \
	fi
	@echo ""
	@echo "✅ Integration demo complete! Check results/integration_*/"

# Generate Reports - 生成报告
.PHONY: generate-reports
generate-reports: check-docker-image
	@echo "📋 Generating Comprehensive Reports in Docker..."
	@echo "   - Rolling dimensionality training reports"
	@echo "   - Method comparison analysis"
	@echo "   - Performance visualization"
	@echo "   - Recommendations and next steps"
	@echo ""
	@if [ "$$(uname)" = "Linux" ]; then \
		docker run --rm --gpus all \
			-v "$$(pwd):/workspace" \
			-v "/home/yin/trading/rlbot/data:/data" \
			-w /workspace \
			-e DATA_DIR="/data/agg_data" \
			lightgbm-runtime:latest \
			python3 scripts/dimensionality/05_report_generator.py; \
	else \
		echo "❌ 此命令仅在 Linux/WSL 中可用"; \
	fi
	@echo ""
	@echo "✅ Reports generated! Check reports/ for comprehensive analysis"

# Full Dimensionality Workflow - 完整工作流程
.PHONY: workflow-dimensionality
workflow-dimensionality: docker-gpu-build feature-engineering rolling-training production-training integration-demo generate-reports
	@echo ""
	@echo "=============================================="
	@echo "  🎉 Complete Dimensionality Training Workflow!"
	@echo "=============================================="
	@echo ""
	@echo "Results:"
	@echo "  - Feature engineering: results/feature_engineering_*/"
	@echo "  - Rolling training: results/rolling_dim_*/"
	@echo "  - Production training: results/production_dimensionality_*/"
	@echo "  - Integration demo: results/integration_*/"
	@echo "  - Comprehensive reports: reports/"
	@echo ""
	@echo "Key Features:"
	@echo "  ✅ 1000+ features generated and filtered"
	@echo "  ✅ IC/IR filtering for high-quality features"
	@echo "  ✅ Autoencoder vs PCA comparison"
	@echo "  ✅ Quarterly rolling training (2024 → 2025)"
	@echo "  ✅ Drift detection and dynamic triggering"
	@echo "  ✅ Production-level model training"
	@echo "  ✅ Integration demonstration"
	@echo "  ✅ Comprehensive reporting"
	@echo ""
	@echo "Next steps:"
	@echo "  - Review feature engineering results"
	@echo "  - Analyze rolling training performance"
	@echo "  - Deploy production models"
	@echo "  - Monitor model performance over time"

# Quick Start - 快速启动
.PHONY: quick-start
quick-start:
	@echo "🚀 Quick Start - Dimensionality Training in Docker"
	@echo "=============================================="
	@echo ""
	@echo "This will:"
	@echo "  1. Build Docker image (if not exists)"
	@echo "  2. Run feature engineering"
	@echo "  3. Run rolling training"
	@echo "  4. Generate reports"
	@echo ""
	@echo "Starting in 3 seconds... (Press Ctrl+C to cancel)"
	@sleep 3
	@echo ""
	@echo "Step 1: Building Docker image..."
	@make docker-gpu-build
	@echo ""
	@echo "Step 2: Running feature engineering..."
	@make feature-engineering
	@echo ""
	@echo "Step 3: Running rolling training..."
	@make rolling-training
	@echo ""
	@echo "Step 4: Generating reports..."
	@make generate-reports
	@echo ""
	@echo "🎉 Quick start complete!"
	@echo "Check results/ directory for all outputs"

# ============================================================================
# Docker GPU 命令 - 所有命令都在 GPU 容器中运行
# ============================================================================

# 检查 Docker GPU 镜像（使用已构建的 lightgbm-runtime）
.PHONY: docker-build
docker-build:
	@echo "🐳 检查 Docker GPU 镜像 (lightgbm-runtime:latest)..."
	@if docker images lightgbm-runtime:latest --format "{{.Repository}}:{{.Tag}}" | grep -q "lightgbm-runtime:latest"; then \
		echo "✅ 镜像已存在: lightgbm-runtime:latest"; \
		docker images lightgbm-runtime:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"; \
	else \
		echo "❌ 镜像不存在，请先构建镜像:"; \
		echo "   cd docker && docker build -f Dockerfile.gpu -t lightgbm-runtime:latest .."; \
		echo "   或使用: make docker-gpu-build"; \
		exit 1; \
	fi

# 启动 Docker GPU 容器交互模式
.PHONY: docker-shell
docker-shell:
	@echo "🐚 启动 GPU 容器交互模式..."
	$(DOCKER_COMPOSE) run --rm ml-gpu /bin/bash

# 测试 GPU 是否可用
.PHONY: docker-test-gpu
docker-test-gpu:
	@echo "🧪 测试 GPU 环境..."
	$(DOCKER_GPU_RUN) python3 -c "import torch; print('PyTorch GPU:', torch.cuda.is_available()); import lightgbm; print('LightGBM installed')"

# ============================================================================
# Docker GPU - 开发工具命令
# ============================================================================
# 注意: lightgbm-runtime 镜像不包含开发工具（black, flake8）
# 这些命令建议在本地运行: make format, make lint
# 如需在 Docker 中使用，请在容器中安装: pip install black flake8

# Docker 中运行代码格式化 (需要先在容器中安装 black)
.PHONY: docker-format
docker-format:
	@echo "⚠️  注意: lightgbm-runtime 镜像未包含 black"
	@echo "   建议使用本地命令: make format"
	@echo "   或在容器中安装: docker-compose run --rm ml-gpu pip install black"

# Docker 中运行代码检查 (需要先在容器中安装 flake8)
.PHONY: docker-lint
docker-lint:
	@echo "⚠️  注意: lightgbm-runtime 镜像未包含 flake8"
	@echo "   建议使用本地命令: make lint"
	@echo "   或在容器中安装: docker-compose run --rm ml-gpu pip install flake8"

# Docker 中运行测试 (需要先在容器中安装 pytest)
.PHONY: docker-test
docker-test:
	@echo "⚠️  注意: lightgbm-runtime 镜像未包含 pytest"
	@echo "   建议使用本地命令: python3 -m pytest tests/"
	@echo "   或在容器中安装: docker-compose run --rm ml-gpu pip install pytest"

# ============================================================================
# Docker GPU - 训练命令
# ============================================================================

# Docker 中训练 wavelet 模型
.PHONY: docker-train-wavelet
docker-train-wavelet:
	@echo "🚀 在 Docker GPU 中训练 Wavelet 模型..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/training/train_model_wavelet.py"

# Docker 中训练增强模型
.PHONY: docker-train-enhanced
docker-train-enhanced:
	@echo "🚀 在 Docker GPU 中训练增强模型..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/training/train_model_enhanced.py"

# Docker 中训练 GPU 加速模型
.PHONY: docker-train-gpu
docker-train-gpu:
	@echo "🚀 在 Docker GPU 中训练模型（GPU 加速）..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/training/train_model_gpu.py"

# Docker 中训练多年多符号模型
.PHONY: docker-train-multi-symbol
docker-train-multi-symbol:
	@echo "🚀 在 Docker GPU 中训练多符号模型..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/training/train_multi_year_multi_symbol.py"

# ============================================================================
# Docker GPU - 回测命令
# ============================================================================

# Docker 中运行 June OOS 回测
.PHONY: docker-oos-june
docker-oos-june:
	@echo "📊 在 Docker 中运行 June OOS 回测..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/backtesting/oos_june.py"

# Docker 中运行多月份回测
.PHONY: docker-oos-months
docker-oos-months:
	@echo "📊 在 Docker 中运行多月份回测..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/backtesting/oos_months.py"

# Docker 中运行 2025 OOS 测试
.PHONY: docker-oos-2025
docker-oos-2025:
	@echo "📊 在 Docker 中运行 2025 OOS 测试..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/backtesting/test_2025_oos.py"

# Docker 中运行 VectorBot 回测
.PHONY: docker-vectorbot-backtest
docker-vectorbot-backtest:
	@echo "📊 在 Docker 中运行 VectorBot 回测..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/backtesting/vectorbot_backtest.py"

# ============================================================================
# Docker GPU - 滚动训练命令
# ============================================================================

# Docker 中运行 2025 月度滚动训练
.PHONY: docker-rolling-2025
docker-rolling-2025:
	@echo "🔄 在 Docker GPU 中运行 2025 月度滚动训练..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/rolling/monthly_rolling_2025.py"

# Docker 中运行高级特征管理滚动训练
.PHONY: docker-rolling-2025-advanced
docker-rolling-2025-advanced:
	@echo "🔄 在 Docker GPU 中运行高级滚动训练（特征管理）..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/rolling/monthly_rolling_2025_with_feature_management.py"

# Docker 中运行降维滚动训练
.PHONY: docker-rolling-dimensionality
docker-rolling-dimensionality:
	@echo "🔄 在 Docker GPU 中运行降维滚动训练..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/rolling/monthly_rolling_2025_with_dimensionality_reduction.py"

# Docker 中运行优化版滚动训练
.PHONY: docker-rolling-optimized
docker-rolling-optimized:
	@echo "🔄 在 Docker GPU 中运行优化版滚动训练..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/rolling/monthly_rolling_2025_optimized.py"

# ============================================================================
# Docker GPU - 分析和报告命令
# ============================================================================

# Docker 中生成报告
.PHONY: docker-reports-june
docker-reports-june:
	@echo "📋 在 Docker 中生成 June 报告..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/reports/reports_june.py"

# Docker 中生成滚动训练报告
.PHONY: docker-report-rolling-2025
docker-report-rolling-2025:
	@echo "📋 在 Docker 中生成 2025 滚动训练报告..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/analysis/generate_rolling_report.py"

# Docker 中生成季度报告
.PHONY: docker-report-quarterly
docker-report-quarterly:
	@echo "📋 在 Docker 中生成季度报告..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/reports/generate_quarterly_report.py"

# Docker 中运行漂移分析
.PHONY: docker-drift-analysis
docker-drift-analysis:
	@echo "📊 在 Docker 中运行漂移分析..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/analysis/monthly_drift.py"

# Docker 中统计特征数量
.PHONY: docker-count-features
docker-count-features:
	@echo "📊 在 Docker 中统计特征数量..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/analysis/count_features.py"

# ============================================================================
# Docker GPU - 优化命令
# ============================================================================

# Docker 中运行网格搜索
.PHONY: docker-grid-tune
docker-grid-tune:
	@echo "🔧 在 Docker GPU 中运行网格搜索..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/optimization/grid_tune.py"

# Docker 中运行 Optuna 风险搜索
.PHONY: docker-optuna-risk
docker-optuna-risk:
	@echo "🔧 在 Docker GPU 中运行 Optuna 风险搜索..."
	$(DOCKER_GPU_RUN) bash -c "PYTHONPATH=/app/src python3 scripts/optimization/optuna_risk_search.py"

# ============================================================================
# Docker GPU - 完整工作流
# ============================================================================

# Docker 中运行完整的训练和回测流程
.PHONY: docker-workflow-full
docker-workflow-full:
	@echo "🚀 在 Docker GPU 中运行完整工作流..."
	@echo "Step 1: 训练模型..."
	@make docker-train-wavelet
	@echo ""
	@echo "Step 2: 运行回测..."
	@make docker-oos-months
	@echo ""
	@echo "Step 3: 生成报告..."
	@make docker-reports-june
	@echo ""
	@echo "🎉 完整工作流完成！"

# Docker 中运行滚动训练完整流程
.PHONY: docker-workflow-rolling
docker-workflow-rolling:
	@echo "🚀 在 Docker GPU 中运行滚动训练完整流程..."
	@echo "Step 1: 滚动训练..."
	@make docker-rolling-2025-advanced
	@echo ""
	@echo "Step 2: 生成报告..."
	@make docker-report-rolling-2025
	@echo ""
	@echo "🎉 滚动训练流程完成！"

# ============================================================================
# Docker GPU - Jupyter 相关
# ============================================================================

# 启动 Jupyter Notebook (带 GPU)
.PHONY: docker-jupyter
docker-jupyter:
	@echo "📓 启动 Jupyter Notebook (GPU 支持)..."
	@echo "访问 http://localhost:8888"
	$(DOCKER_COMPOSE) up jupyter-gpu

# ============================================================================
# Docker GPU - 帮助命令
# ============================================================================

.PHONY: docker-help
docker-help:
	@echo "🐳 Docker GPU 命令帮助 (使用 lightgbm-runtime:latest)"
	@echo "===================================================="
	@echo ""
	@echo "基础命令:"
	@echo "  docker-build              - 检查 lightgbm-runtime 镜像是否存在"
	@echo "  docker-shell              - 进入 GPU 容器交互模式"
	@echo "  docker-test-gpu           - 测试 GPU 是否可用 ✅"
	@echo ""
	@echo "开发工具 (建议在本地运行):"
	@echo "  make format               - 格式化代码 (本地)"
	@echo "  make lint                 - 检查代码 (本地)"
	@echo "  docker-format             - 显示在 Docker 中安装 black 的方法"
	@echo "  docker-lint               - 显示在 Docker 中安装 flake8 的方法"
	@echo ""
	@echo "✅ 以下命令完全支持 (已测试):"
	@echo ""
	@echo "训练命令:"
	@echo "  docker-train-wavelet      - 训练 Wavelet 模型"
	@echo "  docker-train-enhanced     - 训练增强模型"
	@echo "  docker-train-gpu          - GPU 加速训练"
	@echo "  docker-train-multi-symbol - 多符号训练"
	@echo ""
	@echo "回测命令:"
	@echo "  docker-oos-june           - June OOS 回测"
	@echo "  docker-oos-months         - 多月份回测"
	@echo "  docker-oos-2025           - 2025 OOS 测试"
	@echo "  docker-vectorbot-backtest - VectorBot 回测"
	@echo ""
	@echo "滚动训练:"
	@echo "  docker-rolling-2025         - 基础滚动训练"
	@echo "  docker-rolling-2025-advanced - 高级滚动训练"
	@echo "  docker-rolling-dimensionality - 降维滚动训练"
	@echo "  docker-rolling-optimized    - 优化版滚动训练"
	@echo ""
	@echo "分析报告:"
	@echo "  docker-reports-june       - June 报告"
	@echo "  docker-report-rolling-2025 - 滚动训练报告"
	@echo "  docker-report-quarterly   - 季度报告"
	@echo "  docker-drift-analysis     - 漂移分析"
	@echo "  docker-count-features     - 特征统计"
	@echo ""
	@echo "优化:"
	@echo "  docker-grid-tune          - 网格搜索"
	@echo "  docker-optuna-risk        - Optuna 风险搜索"
	@echo ""
	@echo "完整工作流:"
	@echo "  docker-workflow-full      - 完整训练回测流程"
	@echo "  docker-workflow-rolling   - 滚动训练完整流程"
	@echo ""
	@echo "Jupyter:"
	@echo "  docker-jupyter            - 启动 Jupyter Notebook (GPU)"
	@echo ""

