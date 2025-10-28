# Project Structure

This document explains the organization of the ML Trading Project.

## Directory Structure

```
ml_project/
├── src/
│   └── ml_trading/
│       ├── __init__.py
│       ├── main.py
│       ├── config/
│       │   └── settings.py
│       ├── data/
│       │   ├── data_loader.py
│       │   └── feature_engineering.py
│       ├── models/
│       │   └── lightgbm_model.py
│       ├── pipeline/
│       │   ├── multi_tf_pipeline.py
│       │   └── risk_management.py
│       ├── strategies/
│       │   └── ml_strategy.py
│       └── utils/
├── tests/
│   └── test_pipeline.py
├── docs/
│   └── project_structure.md
├── requirements.txt
├── setup.py
├── pyproject.toml
├── Makefile
└── README.md
```

## Component Overview

### 1. Configuration (`config/`)
- `settings.py`: Global configuration parameters for the trading system

### 2. Data Handling (`data/`)
- `data_loader.py`: Loads and preprocesses market data, handles multi-timeframe resampling
- `feature_engineering.py`: Computes technical indicators and other features

### 3. Models (`models/`)
- `lightgbm_model.py`: Implementation of LightGBM models for classification and regression tasks

### 4. Pipeline (`pipeline/`)
- `multi_tf_pipeline.py`: Implements the three-stage multi-timeframe pipeline
- `risk_management.py`: Handles risk management including dynamic stop loss/take profit

### 5. Strategies (`strategies/`)
- `ml_strategy.py`: Main strategy class that integrates all components

### 6. Main Execution (`main.py`)
- Entry point for running the complete strategy

## Multi-Timeframe Three-Stage Pipeline

### Stage 1: Signal Classification
- Binary classification model to determine if there's a good opportunity to go long or short
- Trained on multiple timeframes (5m, 15m, 45m)

### Stage 2: Return Regression
- Regression model to predict expected returns
- Used to determine position sizing

### Stage 3: Ensemble & Risk Management
- Combines signals from multiple timeframes
- Applies dynamic risk management rules

## Key Features

1. **Multi-Timeframe Analysis**: Models trained on 5-minute, 15-minute, and 45-minute data
2. **Three-Stage Pipeline**: 
   - Stage 1: Signal classification (long/short/no position)
   - Stage 2: Return prediction (for position sizing)
   - Stage 3: Ensemble and risk management
3. **Dynamic Risk Management**: 
   - Structural failure detection (consecutive losses)
   - Dynamic stop loss/take profit based on historical volatility
4. **Optuna Integration**: Hyperparameter optimization support
5. **Nautilus Trader Compatibility**: Designed to integrate with Nautilus Trader for backtesting and live trading