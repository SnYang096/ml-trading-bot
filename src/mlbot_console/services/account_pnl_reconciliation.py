"""Reconcile local PnL books vs exchange equity / unrealized (A/B/C scopes)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from mlbot_console.services.account_summary import (
    _is_all_symbols,
    build_account_summary,
)

logger = logging.getLogger(__name__)

_FUTURES_SCOPES = frozenset({"trend", "multi_leg"})


def _tol_usdt(*, reference: float, floor_env: str, pct_env: str) -> float:
    floor = float(os.getenv(floor_env, "2.0"))
    pct = float(os.getenv(pct_env, "0.05"))
    return max(floor, abs(reference) * pct)


def _orphan_unrealized_tol() -> float:
    return float(os.getenv("MLBOT_RECON_ORPHAN_UNREALIZED_TOL_USDT", "0.5"))


def _issue(
    *,
    kind: str,
    scope: str,
    message: str,
    **extra: Any,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "kind": kind,
        "scope": scope,
        "message": message,
    }
    row.update(extra)
    return row


def reconcile_scope_pnl(
    scope: str,
    *,
    scope_block: Mapping[str, Any],
    exchange_row: Mapping[str, Any],
    strategy_rows: List[Mapping[str, Any]],
    symbol: str = "*",
) -> List[Dict[str, Any]]:
    """Compare one scope's local PnL vs exchange snapshot."""
    issues: List[Dict[str, Any]] = []
    if not exchange_row.get("ok"):
        issues.append(
            _issue(
                kind="exchange_unavailable",
                scope=scope,
                message=str(exchange_row.get("error") or "exchange fetch failed"),
                error_code=exchange_row.get("error_code"),
            )
        )
        return issues

    symbol_scoped = not _is_all_symbols(symbol)
    local_u = float(scope_block.get("unrealized_pnl") or 0.0)
    open_pos = int(scope_block.get("open_positions") or 0)
    closed = int(scope_block.get("closed_trades") or 0)

    ex_u = exchange_row.get("unrealized_pnl_usdt")
    ex_equity = exchange_row.get("equity_usdt")
    ex_wallet = exchange_row.get("wallet_balance_usdt")

    if ex_u is not None and scope != "spot":
        ex_u_f = float(ex_u)
        delta_u = ex_u_f - local_u
        tol = _tol_usdt(
            reference=ex_u_f,
            floor_env="MLBOT_RECON_PNL_TOL_USDT",
            pct_env="MLBOT_RECON_PNL_TOL_PCT",
        )
        if abs(delta_u) > tol:
            issues.append(
                _issue(
                    kind="unrealized_pnl_mismatch",
                    scope=scope,
                    message=(
                        f"交易所未实现盈亏 {ex_u_f:.2f} vs 本地估算 {local_u:.2f} "
                        f"(Δ={delta_u:+.2f}, tol={tol:.2f})"
                    ),
                    exchange_unrealized=ex_u_f,
                    local_unrealized=local_u,
                    delta=delta_u,
                    tolerance_usdt=tol,
                )
            )

    orphan_tol = _orphan_unrealized_tol()
    if (
        scope == "trend"
        and open_pos == 0
        and ex_u is not None
        and abs(float(ex_u)) > orphan_tol
        and abs(local_u) <= orphan_tol
    ):
        issues.append(
            _issue(
                kind="exchange_position_not_in_local_db",
                scope=scope,
                message=(
                    f"交易所有浮盈 {float(ex_u):.2f} USDT，但本地 trend DB 未平=0；"
                    "币安有持仓而 positions 表未同步（检查 live runner / 是否手工平仓）"
                ),
                exchange_unrealized=float(ex_u),
                local_unrealized=local_u,
                open_positions=open_pos,
            )
        )

    if open_pos == 0 and abs(local_u) > orphan_tol:
        issues.append(
            _issue(
                kind="orphan_local_unrealized",
                scope=scope,
                message=(
                    f"本地未平仓数为 0 但浮盈 {local_u:.2f} USDT "
                    f"(>{orphan_tol:.2f})，多为订单配对/成交价缺失"
                ),
                local_unrealized=local_u,
                open_positions=open_pos,
                closed_trades=closed,
            )
        )

    for strat in strategy_rows:
        if str(strat.get("scope") or "") != scope:
            continue
        s_open = int(strat.get("open_positions") or 0)
        s_u = float(strat.get("unrealized_pnl") or 0.0)
        if s_open == 0 and abs(s_u) > orphan_tol:
            issues.append(
                _issue(
                    kind="strategy_orphan_unrealized",
                    scope=scope,
                    strategy=str(strat.get("strategy") or ""),
                    message=(
                        f"策略 {strat.get('strategy')} 未平=0 但浮盈 {s_u:.2f} USDT"
                    ),
                    local_unrealized=s_u,
                    realized_pnl=float(strat.get("realized_pnl") or 0.0),
                    closed_trades=int(strat.get("closed_trades") or 0),
                )
            )

    if (
        not symbol_scoped
        and scope in _FUTURES_SCOPES
        and ex_equity is not None
        and ex_wallet is not None
        and ex_u is not None
    ):
        implied = float(ex_wallet) + float(ex_u)
        eq = float(ex_equity)
        id_tol = max(1.0, abs(eq) * 0.002)
        if abs(eq - implied) > id_tol:
            issues.append(
                _issue(
                    kind="exchange_equity_identity",
                    scope=scope,
                    message=(
                        f"交易所权益 {eq:.2f} ≠ 钱包 {float(ex_wallet):.2f} + "
                        f"浮盈 {float(ex_u):.2f} (Δ={eq - implied:+.2f})"
                    ),
                    equity_usdt=eq,
                    wallet_balance_usdt=float(ex_wallet),
                    unrealized_pnl_usdt=float(ex_u),
                    delta=eq - implied,
                )
            )

    if scope == "spot":
        ex_holdings_val = float(exchange_row.get("holdings_value_usdt") or 0.0)
        ex_row = scope_block.get("exchange") or {}
        local_holdings_val = float(ex_row.get("ledger_holdings_value_usdt") or 0.0)
        if local_holdings_val > 0 or ex_holdings_val > 0:
            delta_v = ex_holdings_val - local_holdings_val
            tol = _tol_usdt(
                reference=ex_holdings_val,
                floor_env="MLBOT_RECON_SPOT_HOLDINGS_TOL_USDT",
                pct_env="MLBOT_RECON_SPOT_HOLDINGS_TOL_PCT",
            )
            if abs(delta_v) > tol:
                issues.append(
                    _issue(
                        kind="spot_holdings_value_mismatch",
                        scope=scope,
                        message=(
                            f"现货持仓市值 交易所 {ex_holdings_val:.2f} vs "
                            f"本地账本 {local_holdings_val:.2f} (Δ={delta_v:+.2f})"
                        ),
                        exchange_holdings_value_usdt=ex_holdings_val,
                        local_holdings_value_usdt=local_holdings_val,
                        delta=delta_v,
                        tolerance_usdt=tol,
                    )
                )

    return issues


def reconcile_pnl_vs_exchange(
    *,
    trend_db: Path,
    spot_db: Path,
    spot_ledger_db: Path,
    multi_leg_db: Path,
    feature_bus_root: Path,
    symbol: str = "*",
    lookback_days: int = 0,
) -> Dict[str, Any]:
    """Build account summary and compare local PnL vs exchange equity per scope."""
    summary = build_account_summary(
        trend_db=trend_db,
        spot_db=spot_db,
        spot_ledger_db=spot_ledger_db,
        multi_leg_db=multi_leg_db,
        feature_bus_root=feature_bus_root,
        symbol=symbol,
        lookback_days=lookback_days,
    )
    ledger = summary.get("exchange_ledger") or {}
    accounts = {str(a.get("scope") or ""): a for a in ledger.get("accounts") or []}
    strategies = list(summary.get("strategies") or [])

    per_scope: Dict[str, Any] = {}
    all_issues: List[Dict[str, Any]] = []

    for scope_block in summary.get("scopes") or []:
        scope = str(scope_block.get("scope") or "")
        if not scope:
            continue
        ex = accounts.get(scope) or {"scope": scope, "ok": False, "error": "missing"}
        issues = reconcile_scope_pnl(
            scope,
            scope_block=scope_block,
            exchange_row=ex,
            strategy_rows=strategies,
            symbol=symbol,
        )
        per_scope[scope] = {
            "scope": scope,
            "ok": len(issues) == 0,
            "issues": issues,
            "local": {
                "realized_pnl": float(scope_block.get("realized_pnl") or 0.0),
                "unrealized_pnl": float(scope_block.get("unrealized_pnl") or 0.0),
                "open_positions": int(scope_block.get("open_positions") or 0),
                "closed_trades": int(scope_block.get("closed_trades") or 0),
            },
            "exchange": {
                "equity_usdt": ex.get("equity_usdt"),
                "wallet_balance_usdt": ex.get("wallet_balance_usdt"),
                "unrealized_pnl_usdt": ex.get("unrealized_pnl_usdt"),
                "available_usdt": ex.get("available_usdt"),
            },
        }
        all_issues.extend(issues)

    totals = summary.get("totals") or {}
    ledger_totals = ledger.get("totals") or {}
    local_u_sum = sum(float(s.get("local", {}).get("unrealized_pnl") or 0) for s in per_scope.values())
    ex_u_sum = float(ledger_totals.get("exchange_unrealized_pnl_usdt") or 0.0)
    global_delta = ex_u_sum - local_u_sum
    global_tol = _tol_usdt(
        reference=ex_u_sum,
        floor_env="MLBOT_RECON_GLOBAL_PNL_TOL_USDT",
        pct_env="MLBOT_RECON_GLOBAL_PNL_TOL_PCT",
    )
    symbol_scoped = not _is_all_symbols(symbol)
    if (
        not symbol_scoped
        and ledger_totals.get("accounts_ok", 0) > 0
        and abs(global_delta) > global_tol
    ):
        all_issues.append(
            _issue(
                kind="global_unrealized_pnl_mismatch",
                scope="all",
                message=(
                    f"全账户交易所浮盈合计 {ex_u_sum:.2f} vs 本地合计 {local_u_sum:.2f} "
                    f"(Δ={global_delta:+.2f}, tol={global_tol:.2f})"
                ),
                exchange_unrealized_sum=ex_u_sum,
                local_unrealized_sum=local_u_sum,
                delta=global_delta,
                tolerance_usdt=global_tol,
            )
        )

    ok = len(all_issues) == 0
    if not ok:
        for issue in all_issues:
            logger.warning(
                "account_pnl_reconcile scope=%s kind=%s: %s",
                issue.get("scope"),
                issue.get("kind"),
                issue.get("message"),
            )

    return {
        "ok": ok,
        "symbol": summary.get("symbol"),
        "lookback_days": lookback_days,
        "issues": all_issues,
        "scopes": per_scope,
        "totals": {
            "local_realized_pnl": float(totals.get("realized_pnl") or 0.0),
            "local_unrealized_pnl": float(totals.get("unrealized_pnl") or 0.0),
            "exchange_equity_usdt": ledger_totals.get("equity_usdt"),
            "exchange_wallet_usdt": ledger_totals.get("wallet_balance_usdt"),
            "exchange_unrealized_usdt": ex_u_sum,
            "local_unrealized_sum": local_u_sum,
            "global_unrealized_delta": global_delta,
        },
        "fetched_at": ledger.get("fetched_at"),
    }
