# P3 roadmap — research hygiene & cross-root tooling (ADR §11.4 / T30–T33)

This document tracks **optional** follow-ups that should stay out of P0/P1 layout PRs.

---

## Time contract — `non_rolling`, unified cutoff, full_cycle viz (ADR §14)

**Primary spec**：`docs/architecture/ADR_strategy_research_archetypes_layout.md` **§14**（时间轴、`non_rolling` 与 PCM 各层跑频）。

**Intent**（与 P3 工具链正交，建议 **P1 loader / deploy 稳定后** 独立 epic，勿与 T33 混 PR）：

1. **`research/validate_static.full_study.yaml`** + **`rolling.mode: non_rolling`**（最终命名以 §14 为准）：静态 Train \| Val \| Test；文档化 stage 范围 vs turbo/slow。
2. **`time_split_policy`（或等价）**：walk-forward 下 **Prefilter / Gate / EntryFilter** 共用 **单一 cutoff 解析**（与 **`calibration_months` + 目标月 M + cadence** 对齐）；替代「仅整条实验固定 `test_start`」与 fast 滚动脱节。
3. **Golden tests**：对固定 `month_token`、`calibration_months`、`validation_months` 断言 `calib_start/end`、cutoff、OOS 段与 §14 表格一致。
4. **full_cycle / grid_backtest**：日期 **单真相源**；报告/地图 **竖线** 标 `holdout_start` / `test_start`（含 warmup 的约定写进 runbook）。

**Dependencies**：`scripts/auto_research_pipeline.py` 中所有 `--cutoff-date` / `validation_end` 拼装点需清单化后收敛（§14.3）。

---

## T33 — `pipeline list-runs` across history roots

- **Goal**: `mlbot pipeline list-runs --all-history-roots` (or a small manifest JSON under `results/`) so operators can see experiments regardless of which `output.history_dir` a rolling run used.
- **Non-goals**: Changing default `history_dir`; rewriting existing reports.
- **Suggested approach**: Scan known roots (`results/research_history`, `results/chop_grid`, …) from a configurable list; unify row schema (strategy, timestamp, status, sharpe, path).

## T30–T32 — report schema, rollback, repo hygiene

- Align `report.json` optional fields with deploy/adopt metadata where useful.
- Keep rollback stories in `LIVE_PRODUCTION_RUNBOOK_CN.md` + git-centric deploy revert.

## Dependencies

- P1 `src/config/strategy_layout.py` and deploy profile work should be merged first so list-runs can reuse path conventions.
