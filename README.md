# ML Trading Bot

This repository hosts the production-ready components for the factor research, dimensionality reduction, model training, and live-trading backtesting stack. The code under `src/ml_trading/` contains the reusable Python package; the `scripts/` directory now only exposes a minimal set of command-line entry points that wrap the package APIs.

## Quick Start

1. Create a virtual environment (conda, venv, etc.) and activate it.
2. Install the project in editable mode:
   ```bash
   pip install -e .[dev]
   ```
3. Verify the install by running the help target:
   ```bash
   make help
   ```

## Core Workflows

| Workflow | Command | Notes |
| --- | --- | --- |
| Production training | `make train` | Uses `SYMBOL`, `START_DATE`, `END_DATE` (defaults in Makefile). Reads matching files under `DATA_DIR`, trains with enhanced features, and saves artefacts to `models/`.
| Monthly rolling retrain | `make rolling-monthly` | Sliding monthly windows for a single symbol. Override `DATA_DIR`, `SYMBOL`, `YEAR` as needed.
| Quarterly rolling retrain | `make rolling-quarterly` | Expanding quarterly windows with drift-aware evaluation. Overrides identical to monthly.
| VectorBot backtest | `make vectorbot-backtest` | Loads the latest trained model (`MODEL_PATH`). Produces trade logs and equity curves.
| June 2025 OOS evaluation | `make oos-june` | Replays held-out data (override `OOS_DATA`). Reuses scalers and strategy from training output.
| Dimensionality pipeline (synthetic) | `make dimensionality-demo` | Runs the full Autoencoder + SHAP flow on generated demo data.
| Dimensionality pipeline (real) | `make dimensionality-real` | Executes the pipeline on raw agg-trade data. Requires `DATA_DIR` with Parquet or ZIP agg-trade archives.

All targets accept overrides, for example:

```bash
make train SYMBOL=ETHUSDT START_DATE=2024-01-01 END_DATE=2024-06-30 OVERWRITE=1
make train SYMBOLS="BTCUSDT ETHUSDT" START_DATE=2024-01-01 END_DATE=2024-12-31 OVERWRITE=1
make rolling-quarterly DATA_DIR=/mnt/data/parquet_data SYMBOL=ETHUSDT START_YEAR=2022 END_YEAR=2025
```

## Directory Layout

- `src/ml_trading/` – installable Python package that provides feature pipelines, model trainers, autoencoder stack, and rolling evaluation utilities.
- `ml_trading/models/train_model.py` – production LightGBM training entry point.
- `scripts/rolling/` – rolling retraining orchestration (`monthly_rolling_retrain.py`, `quarterly_rolling_retrain.py`).
- `scripts/backtesting/` – minimal backtesting interfaces (`vectorbot_backtest.py`, `oos_june.py`).
- `scripts/analysis/` – diagnostic tooling that reuses the shared package (e.g. exporting SHAP/feature importance reports).
- `scripts/utils/` – helper scripts such as exporting feature importance visualisations.

## Data Expectations

- Aggregated trade parquet files are expected under `data/parquet_data/` (or the path provided through `DATA_DIR`).
- Legacy ZIP inputs are still accepted when `TRAIN_ZIP`/`OOS_DATA` point to `.zip` archives; the scripts will extract and process them transparently.
- Results are written to `results/` and models to `models/` by default.

## Tips

- Keep `pip install -e .` up to date after refactors so the `ml_trading` package remains importable from the scripts.
- Pass `PYTHONPATH=src` when running scripts manually, or execute them via the Makefile targets which set the environment automatically.
- For GPU LightGBM or PyTorch usage, ensure the corresponding CUDA-enabled wheels are installed prior to running the training commands.
