# ML Trading Bot

**中文版**: [README_CN.md](README_CN.md)  
**文档索引**: [docs/README.md](docs/README.md)

This repository hosts the production-ready components for the factor research, dimensionality reduction, model training, and live-trading backtesting stack. The code under `src/time_series_model/` contains the reusable Python package; the `scripts/` directory now only exposes a minimal set of command-line entry points that wrap the package APIs.

## Quick Start

1. Create a virtual environment (conda, venv, etc.) and activate it.
2. Install the project in editable mode:
   ```bash
   pip install -e .[dev]
   ```
3. Install Git pre-commit hooks (optional but recommended):
   ```bash
   make install-hooks
   ```
   This will automatically run `mlbot dev format` and `mlbot dev lint` before each commit to ensure code quality.
4. Verify the install by running the help command:
   ```bash
   mlbot --help
   ```
   Or see all available commands:
   ```bash
   mlbot analyze --help
   mlbot train --help
   mlbot diagnose --help
   mlbot optimize --help
   ```

## Recommended Usage Flow

### Core Workflow (Config-Driven Architecture)

The recommended workflow uses **config-driven architecture** with strategy-specific configurations:

1. **Feature Analysis** (`mlbot analyze factor-eval`): Evaluate factors and select optimal features
2. **Strategy Training** (`mlbot train sr-reversal-long`, `mlbot train sr-reversal-short`, `mlbot train rolling`): Train models using strategy configs
3. **Ablation Study** (`mlbot analyze strategy-feature-compare`): Compare feature configurations (optional)
4. **Backtesting** (`mlbot backtest vectorbot`): Validate strategy performance

**📖 See [完整流程指南](docs/时序模型/完整流程指南.md) (Chinese) for detailed workflow.**

### Architecture entry points (recommended reading)

- **Industrial Experiment Loop (Layer A/B/C, TaskSpec, Filter→Wrapper, stability rules)**: `docs/architecture/EXPERIMENT_LOOP_ARCHITECTURE.md`
- **NN Multi-head Path Primitives + Router→Execution (NO/MEAN/TREND) architecture**: `docs/时序模型/架构：NN多头路径原语（Path Primitives）+Router解耦升级.md`
- **Feature Search Playbook (Pool B + semantic groups, greedy baselines, singleton ablation, and the roadmap for Halving/Beam/SFFS)**: `docs/strategies/FEATURE_SEARCH_PLAYBOOK.md`
- **Plateau Optimization Methodology (为什么慢 + 怎么改)**: `docs/guides/PLATEAU_OPTIMIZATION_METHODOLOGY.md` - 关键：Plateau 搜索不允许同时调节超过 3 个连续阈值参数

Quick mental model:
- **PolicyTask (direct entry)**: train a model that directly produces trade signals; fastest research loop.
- **PrimitivesTask (router primitives → execution)**: train a shared “path primitives” router (dir/mfe/mae/t) and let Execution map it to actions under strict safety constraints; more reusable and often more stable long-term.

### Step-by-Step Workflow

#### Why SR Reversal is split into Long/Short (and why `sr_reversal/` was removed)

We intentionally use **two direction-fixed strategies**:
- `sr_reversal_long`: long-only, binary label = “a long trade opened at \(t+1\) hits +2R before -1R”
- `sr_reversal_short`: short-only, binary label = “a short trade opened at \(t+1\) hits +2R before -1R”

This replaces the old **bidirectional** config (`sr_reversal/`, `combine_mode: any_success`) which mixed long/short outcomes into a single label.

**Key differences in label semantics**:
- **Old (`any_success`)**: for each bar, evaluate both a hypothetical long and a hypothetical short; label can become “success if either direction would have worked”. This makes `pred` harder to interpret as a trade probability for a specific direction, and often pushes execution to rely on a separate “direction source” (e.g., `signal`).
- **New (`long_only` / `short_only`)**: the label matches exactly one action. `pred` becomes a clean **probability of success for that direction**, so thresholding is stable and comparable across backtest/live.

**Why the split is better**:
- **Cleaner target**: `predict_proba` directly represents “probability this long/short trade succeeds under the RR definition”.
- **Simpler execution**: direction comes from the strategy itself (no `use_signal_direction`, no `signal` required).
- **Better control**: tune thresholds/risk separately for long vs short.
- **Consistency**: same semantics in offline backtests and production inference.

We still keep a lightweight **safety fuse** (optional) to prevent over-trading in OOD/noisy regimes:
`dist_to_nearest_sr / ATR > K  =>  no trade`.

#### Research → Production Playbook (Timeframe, Data Length, Execution “Personality”)

This section is a practical guide for running **stable, comparable research** across strategies, and then safely pushing a configuration to **rolling training** and live deployment.

##### 1) Recommended timeframe per strategy

- **SR Reversal (mean reversion / reversal at SR)**: prefer **4H** (cleaner structure, low churn). 1H is possible but noisier and usually needs stronger filters.
- **SR Breakout**: **1H–4H**. 4H is steadier; 1H gives more samples but higher false-breakout rate.
- **Compression Breakout**: **1H–4H**. Often works well at 1H for more setups, but needs strict confirmation.
- **Trend Following**: **4H–1D**. Lower frequency tends to be more robust under costs.

##### 2) How much data do you need? (rule of thumb)

Two constraints matter:
- **Market regimes**: you need multiple regimes (trend, chop, high vol, low vol).
- **Trade count**: you need enough completed trades to make any metric meaningful.

For crypto on **4H**, the following is a good starting point:

| Strategy | Recommended history (minimum) | Better | Why |
|---|---:|---:|---|
| `sr_reversal_long/short` | 12–18 months | 24–36 months | Reversals are regime-sensitive; need enough “failed vs clean” SR touches |
| `sr_breakout` | 18–24 months | 36+ months | Breakouts change character across volatility cycles |
| `compression_breakout` | 18–24 months | 36+ months | Needs many compression→expansion events across regimes |
| `trend_following` | 24–36 months | 4–6 years | Trend strategies need long history to cover extended trends + mean-reverting years |

**When you change timeframe, convert bar-based parameters to time-based equivalents.**
Example: from 4H → 1H, a parameter in “bars” should be multiplied by ~4 to represent the same wall-clock duration:
- `rr.max_holding_bars`, label holding horizon, rolling window sizes, etc.

##### 3) Multi-symbol training (recommended, but keep it controlled)

You can increase both time and symbols, but do it in a way that preserves interpretability:
- **Start with a small, liquid universe** (e.g., 3–8 large caps).
- **Train per strategy config**, then evaluate **per-symbol** (don’t only look at pooled averages).
- **Keep costs realistic per symbol** (fees/slippage), because higher-frequency variants can look good before costs.
- Prefer **rolling training** to validate that the strategy generalizes month-by-month.

##### 4) Strategy “personality”: pyramiding / multiple positions

Different strategies should not share the same position management rules.

- **SR Reversal (`sr_reversal_*`)**
  - **Pyramiding**: generally **NO** (reversal edges are fragile; adding often increases drawdown).
  - **Max concurrent positions**: typically **1 per symbol** (direction-fixed long-only / short-only).
  - **Goal**: fewer, higher-quality trades; avoid overtrading.

- **Trend Following**
  - **Pyramiding**: often **YES**, but only under strict rules (e.g., add after price moves +1R and signal remains strong).
  - **Profit taking**: often **NO fixed TP**; trend strategies typically rely on trailing exits / stop logic.

- **SR Breakout**
  - **Pyramiding**: **sometimes** (e.g., allow 1 add-on after breakout confirms, not immediately at the first spike).
  - Good practice: add only when breakout holds (retest/confirmation), not on the initial impulse.

- **Compression Breakout**
  - **Pyramiding**: usually **NO** at first. Start without adding to keep the research clean.
  - If you later add pyramiding, do it with a single add-on and strict confirmation, then re-validate stability.

##### 5) Stops & exits: keep a stable template during research

To make ablation / parameter comparisons meaningful, stabilize execution first:
- **Stop loss**: prefer **ATR-based** (consistent across volatility regimes).
- **Take profit**:
  - For **SR Reversal / Breakouts**: RR-style partial/fixed TP is reasonable (e.g., +2R).
  - For **Trend Following**: often avoid a hard TP; use trailing stops / trend invalidation exits.
- **Holding horizon**: `max_holding_bars` should be consistent with your **label definition**.
  - If you change label horizon (e.g., from 50 bars to 20 bars), expect feature selection / best-lag behavior to change.

##### 6) What to keep fixed (to make research reproducible)

When comparing features / models, **do not change everything at once**. Keep these fixed:
- **Timeframe**
- **Label definition** (RR settings, holding horizon)
- **Backtest execution semantics** (RR exits vs probability exits, costs, slippage)
- **Trade frequency target** (e.g., “~20 trades/year on 4H SR reversal”), then tune thresholds to hit it
- **Universe** (symbols) and evaluation windows
- **Rolling protocol** (train months, test months, and step size)

Once you have a stable baseline, you can safely run:
1) `factor-eval` (select features)
2) `strategy-feature-compare` (ablation)
3) `model-comparison`
4) `train rolling` (production training)

##### 7) Intrabar entries (within the candle) vs close-of-bar entries

Default recommendation: **enter on bar close** for research and production consistency.
It is simpler, more reproducible, and avoids subtle look-ahead / execution assumption bugs.

When intrabar entries can make sense:
- **Breakout / momentum** styles (SR breakout, some compression breakout) where earlier entry materially improves R-multiple.
- **Shorter timeframes** (e.g., 1H and below), and only if you can model execution honestly.

How to do intrabar safely (avoid overly-optimistic backtests):
- Use a **lower timeframe execution feed** (e.g., 1m) or a **bar-within-bar** model; don’t assume you can enter at the best price inside the candle.
- Use conservative assumptions: **next-tick/next-bar entry**, realistic slippage, and explicit order type rules.
- Keep entry/exit rules consistent with labels; if labels assume \(t+1\) entry, don’t silently switch to “within-bar” entry without re-validating.

Strategy guidance:
- **SR Reversal**: usually **close-of-bar** (needs confirmation; intrabar tends to increase whipsaw).
- **SR Breakout**: can be **intrabar** if you have confirmation logic and realistic execution; otherwise close-of-bar is safer.
- **Compression Breakout**: start with **close-of-bar**; move to intrabar only after the baseline is stable.
- **Trend Following**: typically **close-of-bar** (or next-bar) with pyramiding rules; intrabar is rarely necessary.

#### Step 0: Verify Feature Correctness (Recommended)

Before starting feature evaluation, run tests to verify that features are computed correctly and don't use future data:

**Quick Test** (Key Features Only):
```bash
make test-key-features-all
```

**Comprehensive Test** (All Features):
```bash
make test-all-features-comprehensive
```

**What These Tests Verify**:
- ✅ No future data leakage (features at time t only use data ≤ t)
- ✅ Multi-asset normalization (features are comparable across different assets)
- ✅ Streaming vs batch consistency (production inference matches training)
- ✅ No global normalization (prevents look-ahead bias)

**Note**: These tests should pass before proceeding to feature evaluation. If tests fail, fix the issues before training models.

**📖 See [测试运行说明](docs/测试运行说明.md) for more details.**

#### Step 1: Feature Analysis

**1.1 Evaluate Individual Factors**:

**Basic Usage**:
```bash
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-10-31 \
```

**Evaluate Specific Factors**:
```bash
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --factors atr sqs_hal_high \
  --timeframe 240T
```

**With Advanced Options** (remove correlated, filter by best lag):
```bash
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-10-31 \
  --remove-correlated \
  --correlation-threshold 0.9 \
  --target-lag 20 \
  --lag-tolerance 5 \
  --filter-by-best-lag \
  --open-browser
```

#### Step 2: Feature Ablation Study (Required)

Compare different feature configurations to validate feature selection:

```bash
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml"
```

**With Rolling Windows** (More Robust):
```bash
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml" \
  --run-rolling \
  --rolling-train-bars 5000 \
  --rolling-test-bars 1000 \
  --rolling-max-windows 10
```

**What This Validates**:
- Selected features perform better than all features
- Feature selection improves model generalization
- Feature ablation shows meaningful differences

**Note**: This step is **required** before proceeding to rolling training.

#### Step 3: Model Comparison (Required)

Verify that ML models outperform rule-based strategies and compare different strategy configurations:

**Basic Usage (Single Strategy Config)**:
```bash
mlbot diagnose model-comparison \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31
```

**Multi-Strategy Comparison (Recommended)**:
Compare different strategy configurations (labels, backtest, stop-loss/take-profit, features):

```bash
mlbot diagnose model-comparison \
  --strategy-config sr_reversal_long,sr_reversal_long_vol,sr_reversal_rr_reg_long \
  --rule-based-entry src.time_series_model.diagnostics.sr_reversal_model_comparison.evaluate_rule_based \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31
```

**Parameters**:
- `--strategy-config`: Comma-separated list of strategy configs (supports relative paths like `sr_reversal_long` or absolute paths)
- `--rule-based-entry`: (Optional) Python module path for rule-based strategy entry point, used to generate rule baseline comparison

**What This Compares**:
- Rule-based baseline (pure rule strategy, if `--rule-based-entry` is provided)
- ML model (XGBoost/LightGBM)
- ML + Volatility Model (if volatility model is enabled in strategy config)

**Comparison Report Includes**:
- Performance metrics: trades, win rate, breakeven rate, Total R, Sharpe ratio
- Configuration differences: label generator, task type, stop-loss/take-profit parameters, feature count
- Multi-strategy side-by-side comparison table

**Difference from `strategy-feature-compare`**:
- `model-comparison`: Requires copying configs (compares different strategy configs, e.g., different labels, backtest, stop-loss/take-profit settings)
- `strategy-feature-compare`: No config copying needed (same directory with different feature configs, for feature ablation studies)

**What This Validates**:
- ML model significantly outperforms rules
- ML model provides stable returns
- ML model has reasonable trade frequency
- Performance differences between different strategy configurations

**Note**: This step is **required** before proceeding to rolling training.

#### Step 4: Strategy Training (Optional - For Debugging)

**⚠️ Optional**: Single training is only for debugging or quick configuration testing. For production, proceed directly to Step 5 (Rolling Training) after completing Steps 2 and 3.

**4.1 Quick Validation** (Single Training):

**SR Reversal Long-only**:
```bash
mlbot train sr-reversal-long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-10-31 \
```

**SR Reversal Short-only**:
```bash
mlbot train sr-reversal-short \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-10-31 \
```

#### Step 5: Production Training (Rolling Window - Recommended)
```bash
# Expanding window training: each test month uses all previous months
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --initial-train-months 6 \
  --min-train-months 3

mlbot train rolling \
  --config config/strategies/sr_reversal_short \
  --symbol BTCUSDT \
  --timeframe 240T \
  --initial-train-months 6 \
  --min-train-months 3
```

**With Date Range**:
```bash
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start 2024-01-01 \
  --end 2025-10-31 \
  --initial-train-months 6 \
  --min-train-months 3
```

**Update Only (Incremental)**:
```bash
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --update-only
```

**Output**:
- `results/rolling/{strategy}/{month}/model.pkl` - Models for each month
- `results/rolling/{strategy}/monthly_results.json` - Aggregated results

**Note**: Only proceed to rolling training after completing Steps 2 (feature ablation) and Step 3 (model comparison) to ensure features and model architecture are validated.

**With Selected Features** (if you have feature selection results):
```bash
# If you have features_suggested.yaml from factor-eval, update your strategy config to use it
# Edit your strategy's features.yaml to use the selected features, then:
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --initial-train-months 6
```

**Output**:
- `results/auto_rolling_*/monthly_results.csv` - All months' detailed results
- `results/auto_rolling_*/summary.json` - Summary information
- `results/auto_rolling_*/monthly_rolling_report.html` - HTML report
- `results/auto_rolling_*/model_YYYY-MM.txt` - Model for each month

#### Viewing HTML Reports in a Dev Container (Cursor / VS Code)

If you are running inside a Dev Container, `--open-browser` may not open the report automatically. Use one of these approaches:

**Option A: Local static server + Port forwarding** (works reliably)

```bash
# Example: serve rolling reports directory
mlbot server
# or (manual)
# python3 -m http.server 8008 --directory results
```

- In Cursor/VS Code, open the **Ports** panel and forward/open port `8008`.
- Open the report in your local browser, e.g.:
  - `results/auto_rolling_*/monthly_rolling_report.html`
  - `results/strategy_compare/strategy_feature_compare_report.html`
  - `results/rule_optimization/optimization_report.html`

If port `8008` is already in use, you can force-kill the owning process inside the container:

```bash
mlbot server --force
```

#### Step 6: Periodic Updates (Weekly/Monthly)

Incremental update from last trained month:

```bash
# Only update new months (from last position) - use --update-only flag
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --update-only
```

This will automatically detect the last trained month and continue from there.

### Complete Workflow Pipeline

For a complete workflow from feature evaluation to rolling training, execute commands in sequence:

```bash
# Step 0: Verify feature correctness (recommended)
make test-key-features-all

# Step 1: Feature evaluation and selection
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start-date 2025-01-01 \
  --end-date 2025-04-30 \
  --remove-correlated \
  --target-lag 5

# Step 2: Feature ablation study (validate feature selection)
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start-date 2025-01-01 \
  --end-date 2025-04-30 \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml"

# Step 3: Model comparison (verify ML outperforms rules)
# Single strategy config
mlbot diagnose model-comparison \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start-date 2025-01-01 \
  --end-date 2025-04-30

# Multi-strategy comparison (recommended)
mlbot diagnose model-comparison \
  --strategy-config sr_reversal_long,sr_reversal_long_vol,sr_reversal_rr_reg_long \
  --rule-based-entry src.time_series_model.diagnostics.sr_reversal_model_comparison.evaluate_rule_based \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start-date 2025-01-01 \
  --end-date 2025-04-30

# Step 5: Rolling window training (main production workflow)
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start 2025-01-01 \
  --end 2025-04-30 \
  --initial-train-months 3 \
  --min-train-months 3
```

> **Tips**  
> - Use `--docker` flag (default) to enable GPU training if available  
> - Review `results/rolling_*/summary.json` for `monthly_results` after training  
> - If `drift detected` appears in results, consider re-evaluating features or adjusting parameters  


## Data Pipeline

Before training, ensure you have data:

```bash
# Download Binance monthly aggTrades
mlbot data download \
  --symbols BTCUSDT,ETHUSDT \
  --start-year 2021 \
  --start-month 1

# Convert ZIPs to Parquet (5min OHLC + orderflow)
mlbot data convert

# Or run both in one go (full pipeline)
mlbot data pipeline \
  --symbols BTCUSDT,ETHUSDT
```

**More Data Pipeline Examples**:

```bash
# Download specific date range
mlbot data download \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --start-year 2024 \
  --start-month 1 \
  --end-year 2025 \
  --end-month 10

# Convert with cleanup (remove ZIP files after conversion)
mlbot data convert --cleanup
```

### Extended data: Market Cap (CoinGecko) and Funding Rate (Binance)

These datasets are used for **feature engineering** (e.g., market-cap normalization for orderflow signals, funding-rate features). They do not replace the core OHLCV / aggTrades pipeline above.

#### 1) Market cap snapshots (CoinGecko)

- **Prerequisite**: set API key via env var (do NOT commit it):

```bash
export COINGECKO_API_KEY='...'
```

- **Update the full universe** (by default, symbols are loaded from the universe YAML referenced by `config/data/market_cap.yaml`):

```bash
mlbot data update-market-cap \
  --config config/data/market_cap.yaml \
  --no-docker
```

- **Update a small set of symbols** (debugging):

```bash
mlbot data update-market-cap \
  --config config/data/market_cap.yaml \
  --symbols BTCUSDT,ETHUSDT \
  --no-docker
```

- **Skip already-fresh snapshots** (recommended):

```bash
mlbot data update-market-cap \
  --config config/data/market_cap.yaml \
  --max-age-days 7 \
  --no-docker
```

- **Force re-download**:

```bash
mlbot data update-market-cap \
  --config config/data/market_cap.yaml \
  --force \
  --no-docker
```

- **Default storage**:
  - `data/market_cap/<SYMBOL>.parquet`
  - `data/market_cap/market_cap_manifest.json`

#### 2) Funding rate (Binance monthly ZIP → Parquet)

- **Download using universe config**:

```bash
mlbot data download-funding-rate \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-set starter_a \
  --start-year 2024 \
  --start-month 1 \
  --no-docker
```

- **Progress + skip cached months** (default behavior: skip existing parquet first, otherwise skip existing ZIP):

```bash
mlbot data download-funding-rate \
  --symbols BTCUSDT,ETHUSDT \
  --start-year 2024 \
  --start-month 1 \
  --progress-every 10 \
  --no-docker
```

- **Force re-download**:

```bash
mlbot data download-funding-rate \
  --symbols BTCUSDT,ETHUSDT \
  --start-year 2024 \
  --start-month 1 \
  --force \
  --no-docker
```

- **Default storage**:
  - ZIP: `data/funding_rate/zip/`
  - Parquet: `data/funding_rate/parquet/`

## Core Principle

**All production training should use dimensionality-reduced features** (Top-K + Autoencoder), not the original 482 features.

### Why?

1. **Better Performance**: Reduced features typically perform better (as shown in research)
2. **Faster Training**: Fewer features = faster training
3. **Less Overfitting**: Reduced risk of overfitting
4. **Consistency**: Same feature set as research phase

## Command Comparison

| Command                     | Purpose                           | When to Use                               |
| --------------------------- | --------------------------------- | ----------------------------------------- |
| `mlbot analyze factor-eval` | Factor evaluation & selection     | **Recommended**: Primary feature selection method |
| `mlbot train sr-reversal-long/short` | Train single model (direction-fixed) | **Optional**: Debugging / quick checks only |
| `mlbot train rolling`       | Rolling window training           | **Recommended**: Main production workflow |

### Key Points

- **Workflow Order**: Always follow Steps 0 → 1 → 2 → 3 → 5 in sequence
  - Step 0: Verify feature correctness (recommended)
  - Step 1: Feature evaluation (`factor-eval`)
  - Step 2: Feature ablation study (`strategy-feature-compare`) - **Required**
  - Step 3: Model comparison (`model-comparison`) - **Required**
  - Step 5: Rolling training (`rolling`) - **Only after validation**

- `mlbot train sr-reversal-long/short`: Trains **one** model for a single time period (direction-fixed)
  - **Not recommended** for production evaluation
  - Use only for debugging or quick configuration testing

- `mlbot train rolling`: Trains **multiple** models (one per month) in a rolling/expanding window fashion
  - **Required** for production deployment
  - Provides better evaluation through expanding windows
  - Only use after validating features (Step 2) and model performance (Step 3)
  - Step 4 (single training) is optional and only for debugging


## Workflow Summary

### Minimal Workflow (5 Steps, Recommended)

```bash
# Step 0: Verify feature correctness (recommended before starting)
make test-key-features-all

# Step 1: Feature evaluation and selection
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --remove-correlated \
  --filter-by-best-lag

# Step 2: Feature ablation study (validate feature selection)
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml"

# Step 3: Model comparison (verify ML outperforms rules)
# Single strategy config
mlbot diagnose model-comparison \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# Multi-strategy comparison (recommended)
mlbot diagnose model-comparison \
  --strategy-config sr_reversal_long,sr_reversal_long_vol,sr_reversal_rr_reg_long \
  --rule-based-entry src.time_series_model.diagnostics.sr_reversal_model_comparison.evaluate_rule_based \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# Step 4: Rolling training (only after validation)
# Note: If you have features_suggested.yaml from factor-eval, update your strategy config to use it
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start 2024-01-01 \
  --end 2025-10-31 \
  --initial-train-months 6 \
  --min-train-months 3
```

### Full Workflow (5 Steps with All Options)

```bash
# Step 0: Verify feature correctness (recommended)
make test-all-features-comprehensive

# Step 1: Feature evaluation (generates features_suggested.yaml with selected features)
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --remove-correlated \
  --target-lag 20 \
  --filter-by-best-lag

# Step 2: Feature ablation study (compare original vs selected features)
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml" \
  --run-rolling \
  --rolling-train-bars 5000 \
  --rolling-test-bars 1000 \
  --rolling-max-windows 10

# Step 3: Model comparison (verify ML outperforms rules)
# Single strategy config
mlbot diagnose model-comparison \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# Multi-strategy comparison (recommended)
mlbot diagnose model-comparison \
  --strategy-config sr_reversal_long,sr_reversal_long_vol,sr_reversal_rr_reg_long \
  --rule-based-entry src.time_series_model.diagnostics.sr_reversal_model_comparison.evaluate_rule_based \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# Step 4: Rolling training (main production workflow - only after validation)
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start 2024-01-01 \
  --end 2025-10-31 \
  --initial-train-months 6 \
  --min-train-months 3
```

### Advanced Workflow Examples

**Diagnostic and Optimization Workflow**:

```bash
# 1. Rule baseline (test pure rule-based strategy)
mlbot diagnose rule-baseline \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# 2. Rule optimization (find optimal parameters)
mlbot optimize rule \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --search-type random \
  --n-trials 100

# 3. Generate rule plateau charts
mlbot optimize rule-plateau-charts \
  --results-csv results/rule_optimization/optimization_results.csv \
  --report-html results/rule_optimization/optimization_report.html

# 4. ML parameter sweep
mlbot optimize ml-param-sweep \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-07-31

# 5. Generate ML plateau charts
mlbot optimize ml-plateau-charts \
  --timeframe 240T
```

**Cross-Sectional Analysis Workflow**:

```bash
# build-panel has been removed. CS now recommends FeatureStore + YAML workflow:

# 1) Build FeatureStore partitions (incremental cache)
mlbot cross-section build-store --no-docker \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT \
  --timeframe 240T \
  --start-date 2025-07-01 --end-date 2025-12-31 \
  --factor-set-yaml config/cross_sectional/cs_factor_sets_crypto.yaml \
  --factor-set crypto_alpha101_cs_rank \
  --features-store-layer cs_alpha101_cs_rank_4h_v1 \
  --warmup-bars 600

# 2) Run end-to-end workflow (eval/select/report/train/backtest/index.html)
mlbot cross-section workflow --no-docker \
  --config config/cross_sectional/pipeline_alpha101_cs_rank_4h_feature_store.yaml

# Note: workflow will snapshot the assembled panel to:
#   output_root/panel_from_feature_store.parquet
```

## Documentation

- **[文档索引](docs/README.md)** - 统一文档导航入口（推荐从这里开始）
- **[系统架构](docs/ARCHITECTURE.md)** - 系统架构（统一版）
- **[工作流文档](docs/workflow/PIPELINE_WORKFLOW.md)** - 完整工作流命令序列
- **[上线MVP闭环](docs/guides/DEPLOYMENT_MVP_WORKFLOW_CN.md)** - MVP工作流指南（中文）

## Recent Feature Updates

### New Features (2024-12-19)

#### 1. Liquidity Void Price Impact

**Feature**: `liquidity_void_f`  
**New Output Column**: `liquidity_void_price_impact`

Measures price impact (how much price moves per unit volume):
```
price_impact = (high - low) / volume
```

- **Higher values**: Lower liquidity (small volume moves price more)
- **Lower values**: Higher liquidity (more volume needed to move price)

📖 See [Liquidity Void Price Impact Guide](docs/features/liquidity_void_price_impact_guide.md) for details.

#### 2. LVN Detection Improvements

**Feature**: `footprint_basic_f`  
**Improvement**: Local minimum detection for LVN (Low Volume Node)

- Uses `scipy.signal.find_peaks` to detect local minima
- More accurate than global minimum approach
- Identifies "valleys" between high volume regions

📖 See [LVN Improvements](docs/features/lvn_improvements.md) for details.

#### 3. WPT Enhancements

**Feature**: `wpt_volume_energy_f`  
**New Options**:
- `use_log_returns`: Apply WPT to log returns instead of raw price (removes trend bias)
- `adaptive_window`: ATR-based adaptive window sizing (adapts to volatility)

📖 See [WPT Enhancements](docs/features/wpt_enhancements.md) for details.

## Getting Help

View all available commands:
```bash
mlbot --help
```

View commands for specific categories:
```bash
mlbot analyze --help      # Analysis and evaluation commands
mlbot train --help        # Training commands
mlbot diagnose --help     # Diagnostic commands
mlbot optimize --help     # Optimization commands
mlbot backtest --help     # Backtesting commands
mlbot cross-section --help # Cross-sectional analysis commands
mlbot data --help         # Data management commands
mlbot features --help     # Feature management commands
mlbot dev --help          # Development commands
```

See also:
- [Migration Guide](docs/MIGRATION_GUIDE.md) - Complete guide for migrating from Makefile to mlbot
- [Makefile vs mlbot](docs/MAKEFILE_VS_MLBOT.md) - Command comparison table

## Development Environment

**Recommended: VS Code Dev Container**:
1. Open the project in VS Code
2. Select "Reopen in Container" (automatically enters Dev Container)
3. Run `mlbot` commands directly in the container (no Makefile needed)

**Command Line Usage**:
- In Dev Container: Use `mlbot` commands directly
- In local environment: Use `mlbot` commands (requires `pip install -e .` first)

All `mlbot` commands support `--docker/--no-docker` options to automatically select the appropriate execution method based on the environment.

