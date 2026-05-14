# Trend / Fat-tail slot guard — validation matrix (event backtest)

This note captures the experiment plan wired into `scripts/validate_add_regime_study.py` for comparing **max drawdown (R)** vs **total R** under trend pool slot rules, optional breakeven unlock, optional BTC anchor, and add ladders aligned with `current_chop_055` semantics.

## Goal

- Reduce chop / regime-switch tail risk by limiting how many **unprotected** trend symbols can run before winners reach breakeven lock.
- After unlock, allow controlled expansion (more symbols) and **float-R ladder adds**, without reverting to unconstrained multi-symbol trend stacking.

## Orchestration

Run the full matrix (long-running):

```bash
python scripts/validate_add_regime_study.py \
  --start-date 2022-01-01 \
  --end-date 2026-05-01 \
  --out-dir results/trend_slot_guard_full_quiet_20260513 \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --data-path data/parquet_data \
  --skip-panel --skip-rule --skip-ml
```

Resume finished scenarios only (reuse existing `event_backtest_<name>.json`):

```bash
python scripts/validate_add_regime_study.py ... --resume-existing
```

Subset:

```bash
python scripts/validate_add_regime_study.py ... --scenarios trend_pool_unlock3_be1_add3_current_locked,trend_pool_btc_anchor_unlock3_be1_add3_current_locked
```

Outputs:

- Per scenario: `results/<out-dir>/event_backtest_<scenario>.json`, trades CSV, capital report artifacts.
- Summary: `add_regime_study_summary.json`
- Ranked report: `report.md` (sort by max DD first, then by total R).

Event backtest extras:

- `--quiet-signal-logs` is passed by the study script to cut INFO spam from signal / PCM modules.
- `--constitution-yaml` points at each scenario’s patched constitution copy under `<out-dir>/strategy_variants/<scenario>/constitution/constitution.yaml`.

## Mechanics (high level)

### Constitution: `resource_allocation.slot_policy.trend_pool_guard`

When `enabled: true`:

- **`max_unprotected_symbols`**: pool-wide cap on distinct symbols whose trend slots are **not** yet protected per `unlock_on` (default `breakeven_locked`).
- **`max_symbols_after_unlock`**: cap on distinct symbols with open trend slots once expansion is allowed (non‑zero enables post‑unlock cap).
- **`unlock_on`**: `breakeven_locked` (or `stop_risk_nonnegative`) — must match simulator state surfaced via open-position snapshot.
- **BTC anchor (optional)**:
  - `anchor_symbol: BTCUSDT`
  - `require_anchor_first: true` — until BTC has at least one **protected** trend slot, block **new** trend entries on **non‑anchor** symbols (add intents are not gated here).

PCM rejects attributable in funnel / JSON:

- `reject_pcm_trend_pool_unprotected_cap`
- `reject_pcm_trend_pool_post_unlock_cap`
- `reject_pcm_trend_pool_anchor_first`

### Execution patch (per scenario)

- **`add_position`**: `max_add_times`, `add_size_multipliers`, optional `min_current_r_by_add` for ladder thresholds.
- **`add_regime_gate`**: semantic chop gate (`bpc_semantic_chop_ts_q lte 0.55`) aligned with `current_chop_055`.
- **`stop_loss.breakeven`** (when scenario enables it): unlock trigger uses **`breakeven_trigger_r` in ATR units** (study defaults to **1 ATR** so breakeven lock is reachable early enough for slot unlock experiments).

### Constitution executor

- **`require_locked_profit`** on adds maps to simulator breakeven lock before ladder adds when enabled.

## Scenario list (names)

| Scenario | Pool guard | Anchor | Breakeven unlock | Adds |
|----------|-----------|--------|------------------|------|
| `current_chop_055` | off | — | off | baseline `1x/2x/3x`, max 3 |
| `chop_055_add_1x_max1` | off | — | off | 1× `1x`, locked-profit adds |
| `chop_055_add_light_max2` | off | — | off | 2 legs `0.5x/1x`, locked-profit adds |
| `trend_pool_one_symbol_be_noadd` | on, max 1 symbol total | — | 1 ATR | adds disabled (`allow_add_position: false`) |
| `trend_pool_unlock3_be_noadd` | on, unlock to ≤3 symbols | — | 1 ATR | adds disabled |
| `trend_pool_unlock3_be1_add3_current_locked` | on, unlock to ≤3 symbols | — | 1 ATR | **same ladder as baseline** `1/2/3` × `[1,2,3]` min R |
| `trend_pool_unlock3_be1_add3_light_locked` | on, unlock to ≤3 symbols | — | 1 ATR | lighter sizes `0.5/1/1.5` |
| `trend_pool_btc_anchor_unlock3_be1_add3_current_locked` | on, unlock to ≤3 symbols | **BTC first** | 1 ATR | same as `current` ladder |
| `trend_pool_unlock3_be_add1_locked` | on | — | 1 ATR | 1 add `0.5x` |
| `trend_pool_unlock3_be_add2_locked` | on | — | 1 ATR | 2 adds `0.5x/1x` |
| `trend_pool_unlock3_be_add2_loose` | on | — | 1 ATR | 2 adds, **no** locked-profit requirement |
| `trend_pool_unlock6_be_add1_locked` | on, unlock to ≤6 symbols | — | 1 ATR | 1 add `0.5x` |

Exact YAML mutations live in `scenario_specs` inside `scripts/validate_add_regime_study.py`; treat that dict as source of truth if this table drifts.

## Interpreting results

Primary KPIs written into `report.md`:

1. Sort **max_drawdown_r ascending**, break ties with higher **total_r**.
2. Sort **total_r descending**, break ties with lower **max_drawdown_r**.

Cross-check **`add_count`**, **`reject_add_locked_profit_required`**, and PCM trend-pool counters to see whether DD changes come from slot gating vs add gating vs anchor sequencing.
