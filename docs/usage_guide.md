# Usage Guide

This guide explains how to use the ML Trading Project.

## Installation

1. **Using uv (recommended):**
   ```bash
   uv init
   uv add lightgbm nautilus-trader pandas numpy scikit-learn optuna matplotlib seaborn python-dotenv
   ```

2. **Using pip:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Development installation:**
   ```bash
   pip install -e .[dev]
   ```

## Project Structure

See [project_structure.md](project_structure.md) for detailed information about the project organization.

## Running the Strategy

### Basic Usage

To run the main strategy:

```bash
python src/ml_trading/main.py
```

Or using the Makefile:

```bash
make run
```

### Training the Strategy

The strategy will automatically train when you run it. The training process includes:

1. Loading market data (generates synthetic data by default)
2. Resampling to multiple timeframes (5m, 15m, 45m)
3. Engineering technical indicators
4. Training stage 1 models (signal classification)
5. Training stage 2 models (return regression)

### Generating Signals

After training, the strategy will generate trading signals using the three-stage pipeline:

1. Stage 1: Predict trading signals (long/short/hold)
2. Stage 2: Predict expected returns
3. Stage 3: Ensemble multi-timeframe signals and apply risk management

## Customizing the Strategy

### Configuration

Modify `src/ml_trading/config/settings.py` to change:

- Timeframes for analysis
- Technical indicators
- Model parameters
- Risk management parameters

### Using Your Own Data

To use your own market data:

1. Modify the `MarketDataLoader` in `src/ml_trading/data/data_loader.py`
2. Update the `load_data()` method to read from your data source
3. Ensure your data has the required columns: timestamp, open, high, low, close, volume

### Hyperparameter Optimization

To optimize model hyperparameters using Optuna:

```bash
python examples/optuna_optimization.py
```

This will run hyperparameter optimization for both stage 1 and stage 2 models.

## Integration with Nautilus Trader

The project is designed to integrate with Nautilus Trader for backtesting and live trading. See `examples/nautilus_integration.py` for a conceptual example.

## Testing

Run tests with:

```bash
make test
```

Or directly:

```bash
python -m pytest tests/ -v
```

## Code Quality

### Formatting

Format code with Black:

```bash
make format
```

### Linting

Lint code with Flake8:

```bash
make lint
```

## Directory Overview

- `src/ml_trading/`: Main source code
- `tests/`: Unit tests
- `examples/`: Example scripts and integrations
- `docs/`: Documentation
- `logs/`: Log files (created automatically)

## Key Components

### Data Module
- `data_loader.py`: Loads and resamples market data
- `feature_engineering.py`: Computes technical indicators

### Models Module
- `lightgbm_model.py`: LightGBM implementation with Optuna optimization

### Pipeline Module
- `multi_tf_pipeline.py`: Three-stage multi-timeframe pipeline
- `risk_management.py`: Dynamic risk management

### Strategies Module
- `ml_strategy.py`: Main strategy integrating all components

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you've installed all dependencies
2. **Data issues**: Check that your data has the required format
3. **Memory issues**: Reduce the size of your training data

### Getting Help

For issues with the project, please:
1. Check the documentation
2. Run tests to identify problems
3. Review error messages carefully
4. Create an issue on the project repository