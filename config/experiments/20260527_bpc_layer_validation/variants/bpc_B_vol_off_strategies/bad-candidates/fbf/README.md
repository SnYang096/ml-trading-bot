# FBF (Failed Breakout Failure)

## Status

FBF is archived under `bad-candidates` for now.

## Why It Was Moved Back

FBF's old slow runs looked stable as a short-cycle failed-breakout strategy:

- `20260416_223806`: `+30.48R / 263 trades`
- `20260413_162634`: `+36.76R / 240 trades`
- `20260417_191527`: `+16.95R / 213 trades`

But the same short-cycle shape did not reproduce in fast validation:

- turbo threshold chain `20260425_230544`: `-0.69R / 134 trades`
- locked-fast, with threshold rewriting disabled, `20260426_091353`: `-5.15R / 224 trades`

The locked-fast run kept the old-style signal density but still lost money. That suggests the gap is not just the turbo threshold optimizer rewriting FBF. The strategy appears sensitive to the fast event replay / execution path, so slow realistic validation is required before promoting it again.

## Current Interpretation

FBF is not a trend-following strategy. It trades failed boundary breaks: detect the false breakout direction, then follow the real unwind direction for a short execution window.

Mechanically it is still a useful research idea, but not production-ready in this pipeline until the fast-vs-slow discrepancy is explained.

## Reproduction Configs

- `config/strategies/bad-candidates/pipelines/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_fbf_only.yaml`
- `config/strategies/bad-candidates/pipelines/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_fbf_old_locked.yaml`
