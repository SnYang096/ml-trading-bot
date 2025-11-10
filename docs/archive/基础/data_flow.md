# 🌟 High Win Rate Opportunity Strategy - Complete Data Flow

This document describes the complete data flow for the high win rate opportunity strategy implemented in this ML trading project.

## Data Flow Diagram

```
          ┌─────────────────────┐
          │   原始市场数据      │
          │  (价格、成交量等)  │
          └─────────┬──────────┘
                    │
                    ▼
          ┌─────────────────────┐
          │ 特征工程 & 技术指标 │
          │ (ATR, ZigZag, etc.)│
          └─────────┬──────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │ 多时间尺度 pipeline      │
       │                         │
       │  5m / 15m / 45m 各训练  │
       │  阶段1: 好机会二分类    │
       │  阶段2: 预期收益回归    │
       └─────────┬─────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │ 多时间尺度 Ensemble     │
       │  综合各周期置信度与收益 │
       │  生成初步仓位 ensemble_pos│
       └─────────┬─────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │ 结构失败止损 (n_fail)   │
       │  连续低score → 平仓     │
       └─────────┬─────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │ 动态止盈/止损机制       │
       │  基于历史均值±k*std调整 │
       │  止盈减半、止损全平     │
       └─────────┬─────────────┘
                    │
                    ▼
          ┌─────────────────────┐
          │ 最终仓位输出         │
          │ ensemble_pos 动态值 │
          └─────────────────────┘
```

## 🔹 Process Description

### 1. Feature Engineering
Technical indicators computed:
- **ATR (Average True Range)**: Measures market volatility
- **ZigZag**: Identifies significant price movements
- **RSI (Relative Strength Index)**: Momentum oscillator
- **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator
- **Bollinger Bands**: Volatility bands above and below a moving average
- **Price Change & Volatility**: Rate of change and standard deviation of returns
- **Volume Features**: Volume moving averages and ratios

### 2. Multi-Timeframe Pipeline
Three-stage approach for each timeframe (5m, 15m, 45m):

#### Stage 1: Opportunity Classification
- **Model**: LightGBM binary classifier
- **Target**: Predict if the next period has a good long/short opportunity
- **Method**: If next period return > threshold, long; if < -threshold, short

#### Stage 2: Expected Return Regression
- **Model**: LightGBM regression model
- **Target**: Predict the expected return for the next period
- **Purpose**: Determine position sizing based on confidence

#### Stage 3: Ensemble
- **Method**: Combine signals from all timeframes
- **Output**: Weighted average of signals and returns
- **Decision**: Generate discrete signals (-1, 0, 1) based on ensemble confidence

### 3. Risk Management

#### Structural Failure Stop (n_fail)
- **Mechanism**: Monitor consecutive losing trades
- **Action**: If连续 losses ≥ threshold, close positions
- **Purpose**: Prevent continued losses during unfavorable market conditions

#### Dynamic Take Profit/Stop Loss
- **Calculation**: Based on historical mean ± k * standard deviation
- **Adjustment**: Floating levels that adapt to market volatility
- **Execution**: 
  - Take profit: Reduce position by half
  - Stop loss: Fully close position

### 4. Final Position Output
- **Output**: Dynamic ensemble_pos value ready for order execution
- **Integration**: Compatible with Nautilus Trader for backtesting and live trading

## Implementation Components

### Data Module (`src/time_series_model/data/`)
- `data_loader.py`: Handles market data loading and multi-timeframe resampling
- `feature_engineering.py`: Computes all technical indicators

### Models Module (`src/time_series_model/models/`)
- `lightgbm_model.py`: LightGBM implementation with Optuna optimization support

### Pipeline Module (`src/time_series_model/pipeline/`)
- `multi_tf_pipeline.py`: Implements the three-stage multi-timeframe pipeline
- `risk_management.py`: Dynamic risk management algorithms

### Strategies Module (`src/time_series_model/strategies/`)
- `ml_strategy.py`: Main strategy integrating all components

## Key Features

1. **Multi-Timeframe Analysis**: Models trained on 5-minute, 15-minute, and 45-minute data
2. **Three-Stage Pipeline**: 
   - Stage 1: Signal classification (long/short/hold)
   - Stage 2: Return prediction (for position sizing)
   - Stage 3: Ensemble and risk management
3. **Dynamic Risk Management**: 
   - Structural failure detection (consecutive losses)
   - Dynamic stop loss/take profit based on historical volatility
4. **Optuna Integration**: Hyperparameter optimization support
5. **Nautilus Trader Compatibility**: Designed for backtesting and live trading integration

## Usage Examples

See the `examples/` directory for:
- `nautilus_integration.py`: Conceptual Nautilus Trader integration
- `optuna_optimization.py`: Hyperparameter optimization example
- `real_data_example.py`: Using with real market data
- `visualization.py`: Signal and performance visualization
- `backtest_example.py`: Simple backtesting implementation