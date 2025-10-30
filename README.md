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


### make train 和 dimensionality-real 是如何配合的，是不是先用dimensionality-real去找到降维的features，然后make train会去使用降维的features去训练，如何比较降维前和降维后的质量呢？

结论（最重要）: 目前 make dimensionality-real 和 make train 是两条独立流水线。dimensionality-real 会在研究/评估阶段做 AE+SHAP→Top-K→LightGBM，并把 Top-K 因子清单、报告、以及可选的 Top-K 模型产出到 results/；make train 则走“全量综合特征（ComprehensiveFeatureEngineer）→LightGBM”的生产训练，不会自动消费 Top-K 因子结果。

dimensionality-real 产出
Top-K 因子与权重: results/.../top_factors_<symbol>.json
可选 Top-K LightGBM: results/.../lightgbm_topk_*.txt
报告/可视化与滚动评估汇总
在代码中由 run_dimensionality_reduction_pipeline 保存，核心片段如下（会写入 top_factors 文件）:

    top_factor_path = save_top_factors(
        output_dir,
        args.symbol,
        top_factors,
        top_weights,
        getattr(engine, "factor_contributions", None),
    )
    ```
- `make train` 执行内容
  - 使用 `MultiTimeframeComprehensiveEngineer` 的全量特征，训练生产 LightGBM，保存模型与缩放器到 `models/`，不读取 Top-K 因子文件。
  - 入口在 `ml_trading.models.train_model:main`，流水线集中在 `train_symbol(...)`。

- 如何比较降维前后质量
  - 快速方式（推荐）: 跑 `dimensionality-real`，其内部已在训练“原始全量特征 vs Top-K/压缩特征”并输出对比指标与滚动评估，报告里直接对比。
  - 生产对比（若你需要复现到生产训练口径）:
    1) 运行 `dimensionality-real` 得到 `top_factors_*.json`。
    2) 使用 `src/ml_trading/pipeline/dimensionality/production_training.py` 的生产训练函数，里面会同时训练“原始特征模型”和“压缩特征模型”，并在 `production_results.json` 给出对比（R²、RMSE、MAE等）。
    3) 如需将 Top-K 真正引入生产训练，可扩展 `ml_trading.models.train_model`：增加参数读取 `top_factors_*.json` 以过滤特征，再训练并与默认全量特征的生产模型对比保存两套指标。

- 常用命令示例
  - 运行降维研究（真实数据）并出报告/Top-K:
    ```
    make dimensionality-real SYMBOL=BTC-USD DATA_DIR=data/parquet_data
    ```
  - 生产训练（全量特征）:
    ```
    make train SYMBOLS="BTCUSDT" START_DATE=2024-10-01 END_DATE=2024-12-31
    ```
  - 使用“生产降维对比脚本”生成原始 vs 压缩对比（可按需新增 Make 目标指向 `production_training.py`）:
    ```
    python3 -m ml_trading.pipeline.dimensionality.production_training \
      --data-path data/parquet_data --symbol BTC-USD --encoding-dim 16 --autoencoder-epochs 500
    ```

小结
- 现在是用 `dimensionality-real` 先做Top-K研究与报告；`make train` 仍然训练全量综合特征，不会自动用Top-K。
- 对比可直接看 `dimensionality-real` 的报告或用 `production_training.py` 的输出。
- 如果你要把Top-K接到生产训练里，我可以给 `train_model.py` 加一个参数（例如 `--top-factors path/to/top_factors.json`）以及特征过滤逻辑，并新增一个Make目标来一键运行对比。

### src/ml_trading/pipeline/dimensionality/production_training.py 和 ml_trading.models.@train_model.py 代码一样的吗？有什么不同点，训练生产用的模型用哪个，还是和合并重构一下？

- 不一样。两者定位和产物不同：
  - production_training.py
    - 目标: 研究/生产化原型的“降维管线”（AE 压缩 + 用压缩特征训练 LightGBM），并对比“原始全量特征 vs 压缩特征”表现。
    - 产物: AE 模型权重、压缩特征 LightGBM、对比指标 JSON 等（偏研究评估/集成演示）。
  - models/train_model.py
    - 目标: 生产训练主流程（多时间框综合特征 ComprehensiveFeatureEngineer → LightGBM → 保存模型/Scaler/信息）。
    - 产物: 生产用的策略对象、缩放器、训练信息等，走既有生产回测/上线路径。

- 训练生产用模型用哪个？
  - 现在应当用 `ml_trading.models.train_model`（Makefile 的 `make train` 已绑定它），这是现行生产训练主线。
  - `production_training.py` 更像是把“降维法”打包成一个可对比的生产化原型与评估脚本，不直接替代现有生产主线。

- 是否合并重构？
  - 建议保留单一生产入口 `train_model.py`，并扩展参数以可选启用降维/Top-K：
    - 新增参数示例：
      - `--use-top-factors path/to/top_factors.json`（读取 `dimensionality-real` 产出的 Top-K 列表，过滤特征训练）
      - `--use-autoencoder path/to/autoencoder.pth --encoding-dim 16`（先用 AE 压缩再训练）
    - 把 production_training.py 里与 AE 压缩、评估对比的通用逻辑抽到 `ml_trading/pipeline/dimensionality/` 下的可复用函数，`train_model.py` 调用。
    - 继续保留 `production_training.py` 作为“研究对比/回归测试”的 CLI，但底层复用同一实现，避免重复代码。

- 实际建议
  - 近期要上产：继续用 `make train`（不变）。
  - 要把降维纳入生产：按上面方案给 `train_model.py` 增配参与调用路径，我可以直接提交对应编辑，并在 Makefile 增加一个 `make train-topk`/`make train-ae` 目标，便于一键对比与落地。

  I'll add a small todo list to track adding new CLI options, refactoring reusable utilities, updating train_model.py, and adding Makefile targets. Then I'll implement code edits, starting with adding CLI flags and Top-K filtering into train_model.py. After that, I'll wire optional autoencoder-based compression using an existing UnifiedAutoencoder loader if provided. Finally, I'll add make targets.

[1 tool called]

I'll start by adding the new CLI flags to `train_model.py` and wiring the Top-K filter logic. Then I'll add the autoencoder compression option and finally update the Makefile with two new targets.

[6 tools called]

- 我已在 `ml_trading.models.train_model` 增加参数：`--use-top-factors`, `--use-autoencoder`, `--encoding-dim`。
- 已实现两种路径：
  - Top-K 特征过滤：读取 `top_factors_*.json`，对各时间框工程特征按 Top-K 名单取交集（保底留常用列如 `close/volume/taker_buy_ratio/cvd` 若存在）。
  - AE 压缩：加载 `UnifiedAutoencoder(.pth)`，将工程特征压缩为 `compressed_feature_*` 再训练。
- 在 `Makefile` 新增目标：
  - `make train-topk`：使用 Top-K 训练。需传 `TOP_FACTORS=path/to/top_factors.json`
  - `make train-ae`：使用 AE 压缩训练。需传 `AE_PATH=.../autoencoder.pth ENCODING_DIM=16`

用法示例
- Top-K 训练：
  ```
  make train-topk SYMBOLS="BTCUSDT" START_DATE=2024-10-01 END_DATE=2024-12-31 TOP_FACTORS=results/.../top_factors_BTC-USD.json
  ```
- AE 压缩训练：
  ```
  make train-ae SYMBOLS="BTCUSDT" START_DATE=2024-10-01 END_DATE=2024-12-31 AE_PATH=results/.../production_autoencoder.pth ENCODING_DIM=16
  ```

注意
- `--use-autoencoder` 需与 `--encoding-dim` 同时提供，模型以当前工程特征维度为 `input_dim` 初始化。
- 当前未动 `production_training.py`，后续可把其 AE/评估通用逻辑抽成工具函数供复用（待办保持中）。

- 我已完成新增 CLI、Top-K 过滤、AE 压缩，以及 Makefile 新目标。您可以直接运行上述命令进行对比训练。