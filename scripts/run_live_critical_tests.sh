#!/usr/bin/env bash
# Live-critical regression suite (mock / in-memory only — no exchange keys).
#
# Runs tier-by-tier (fail-fast) before deploy. CI job: safety-regression-tests.
# Local: make test-live-critical
#
# Design: docs/architecture/account_safety_gate_CN.md §12
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${PYTHONPATH:-src:scripts}"

# Exclude only slow replay (not a deploy gate).
PYTEST_ARGS=(-q --tb=short -k "not hype_replay")

_run_tier() {
  local label="$1"
  shift
  local -a files=("$@")
  if ((${#files[@]} == 0)); then
    return 0
  fi
  echo ""
  echo "━━ ${label} (${#files[@]} files) ━━"
  pytest "${files[@]}" "${PYTEST_ARGS[@]}"
}

# Tier 0 — Jun 16 postmortem (DB wipe, orphan TG, orchestrator sync gate)
TIER0=(
  tests/order_management/test_live_safety_regressions.py
)

# Tier 0b — trend_scalp engine FULL FILE (48 tests incl. 6b3d59d9 partial-SL fix)
# Must NOT use -k subset — all protection_fill / ensure_protection / late_fill gated.
TIER0B=(
  tests/unit/test_dual_add_trend_live_engine.py
  tests/unit/test_segment_lifecycle.py
)

# Tier 1 — C layer
TIER1=(
  tests/order_management/test_multi_leg_kill_switch.py
  tests/order_management/test_multi_leg_risk_governor.py
  tests/order_management/test_multi_leg_storage.py
  tests/order_management/test_multi_leg_orchestrator.py
  tests/order_management/test_multi_leg_reconciliation.py
  tests/order_management/test_grid_execution_adapter.py
  tests/order_management/test_multi_leg_daemon.py
  tests/order_management/test_run_multi_leg_live_make_api.py
  tests/order_management/test_run_multi_leg_live_symbol_resolution.py
  tests/order_management/test_multi_leg_order_backfill.py
  tests/order_management/test_multileg_timeline_daemon_parity.py
  tests/order_management/test_trend_position_truth_sync.py
  tests/order_management/test_spot_order_manager.py
  tests/order_management/test_multileg_user_stream_routing.py
  tests/order_management/test_binance_user_stream.py
  tests/unit/test_mock_binance_pending_match_case.py
)

# Tier 2 — B layer
TIER2=(
  tests/unit/test_live_enforcement.py
  tests/unit/test_safety_runtime.py
  tests/unit/test_safety_extreme_scenarios.py
  tests/unit/test_live_order_submission_must_be_guarded.py
  tests/unit/test_slot_release_order_failed_fix.py
  tests/unit/test_risk_position_sizing.py
  tests/unit/test_multileg_symbol_owner.py
  tests/unit/test_chop_grid_concurrency.py
  tests/unit/test_multileg_timeline_account.py
  tests/unit/test_multileg_portfolio_metrics.py
)

# Tier 3 — chop_grid + trade executor + user-stream sync (dual_add in Tier 0b)
TIER3=(
  tests/unit/test_chop_grid_ensure_protection.py
  tests/unit/test_chop_grid_protection_on_bar.py
  tests/unit/test_chop_grid_dust_and_late_fill.py
  tests/unit/test_chop_grid_live_engine_hooks.py
  tests/unit/test_chop_execution_bridge.py
  tests/unit/test_trade_executor.py
  tests/unit/test_execution_truth_sync.py
  tests/unit/test_order_flow_listener_execution_sync.py
)

# Tier 4 — A layer spot
TIER4=(
  tests/unit/test_spot_live_recovery.py
  tests/unit/test_spot_new_buy_balance_gate.py
  tests/unit/test_spot_accum_simple_policy.py
  tests/unit/test_spot_pending_reconcile_metrics.py
)

# Tier 5 — CMS / account truth
TIER5=(
  tests/business_console/test_multileg_position_truth.py
  tests/business_console/test_open_positions_list.py
  tests/business_console/test_multileg_leg_pnl.py
  tests/business_console/test_multileg_exit_pairing.py
  tests/business_console/test_account_reconciliation_multileg.py
  tests/business_console/test_account_pnl_reconciliation.py
  tests/business_console/test_account_reconciliation.py
  tests/business_console/test_multi_leg_reconcile.py
  tests/business_console/test_spot_pnl.py
  tests/business_console/test_spot_ledger_book.py
  tests/business_console/test_exchange_balances.py
  tests/business_console/test_exchange_balances_spot.py
  tests/business_console/test_account_summary.py
)

TOTAL_FILES=$((
  ${#TIER0[@]} + ${#TIER0B[@]} + ${#TIER1[@]} + ${#TIER2[@]} +
  ${#TIER3[@]} + ${#TIER4[@]} + ${#TIER5[@]}
))

echo "🛡️  Live-critical regression suite (${TOTAL_FILES} files, mock only, fail-fast by tier)"

_run_tier "Tier 0 · Jun-16 accident regressions" "${TIER0[@]}"
_run_tier "Tier 0b · trend_scalp engine (FULL test_dual_add_trend_live_engine.py)" "${TIER0B[@]}"
_run_tier "Tier 1 · C layer (kill-switch, orchestrator, user-stream)" "${TIER1[@]}"
_run_tier "Tier 2 · B layer (safety, sizing, guards)" "${TIER2[@]}"
_run_tier "Tier 3 · chop_grid + trade executor" "${TIER3[@]}"
_run_tier "Tier 4 · spot A layer" "${TIER4[@]}"
_run_tier "Tier 5 · CMS / account truth" "${TIER5[@]}"

echo ""
echo "✅ Live-critical tests passed (${TOTAL_FILES} files)"
