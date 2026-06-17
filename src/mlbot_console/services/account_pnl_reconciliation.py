"""Reconcile local PnL books vs exchange equity / unrealized / realized (A/B/C scopes).

Extended to compare local DB realized PnL against Binance income history
(REALIZED_PNL + COMMISSION + FUNDING_FEE) for each futures scope.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from mlbot_console.services.account_summary import build_account_summary
from mlbot_console.services.exchange_income import fetch_scope_income
from mlbot_console.services.symbols import is_all_symbols
from mlbot_console.services.tg_notify import check_reconciliation_alerts

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

    symbol_scoped = not is_all_symbols(symbol)
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
    local_u_sum = sum(
        float(s.get("local", {}).get("unrealized_pnl") or 0) for s in per_scope.values()
    )
    ex_u_sum = float(ledger_totals.get("exchange_unrealized_pnl_usdt") or 0.0)
    global_delta = ex_u_sum - local_u_sum
    global_tol = _tol_usdt(
        reference=ex_u_sum,
        floor_env="MLBOT_RECON_GLOBAL_PNL_TOL_USDT",
        pct_env="MLBOT_RECON_GLOBAL_PNL_TOL_PCT",
    )
    symbol_scoped = not is_all_symbols(symbol)
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


def reconcile_realized_pnl(
    *,
    scope: str,
    local_realized_pnl: float,
    local_commission: float = 0.0,
    symbol: str = "*",
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
    lookback_days: int = 90,
) -> Dict[str, Any]:
    """Compare local DB realized PnL against Binance income history.

    The Binance ``/fapi/v1/income`` endpoint returns every income event:
    REALIZED_PNL (trade P&L), COMMISSION (fees), and FUNDING_FEE (funding).
    The "net income" from the exchange is::

        exchange_net = REALIZED_PNL + COMMISSION + FUNDING_FEE

    The local DB computes PnL from entry/exit ``average_price`` pairs, which
    does **not** subtract commissions or include funding.  So the expected
    relationship is::

        local_realized_pnl ≈ exchange_realized_pnl

    Any significant gap may indicate:
    - Phantom positions inflating local PnL
    - Missed fill reports
    - Price rounding differences
    - Commission not being tracked locally

    Returns a dict with ``ok``, ``issues``, ``exchange``, and ``local`` keys.
    """
    issues: List[Dict[str, Any]] = []

    try:
        income = fetch_scope_income(
            scope,
            symbol=symbol,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
        )
    except Exception as exc:
        logger.warning(
            "reconcile_realized_pnl: fetch failed for scope=%s: %s", scope, exc
        )
        return {
            "ok": False,
            "scope": scope,
            "issues": [
                _issue(
                    kind="income_fetch_failed",
                    scope=scope,
                    message=f"交易所收入流水获取失败: {exc}",
                )
            ],
            "exchange": {},
            "local": {
                "realized_pnl": local_realized_pnl,
                "commission": local_commission,
            },
        }

    if not income.get("available"):
        return {
            "ok": False,
            "scope": scope,
            "issues": [
                _issue(
                    kind="income_unavailable",
                    scope=scope,
                    message=f"交易所收入流水不可用: {income.get('error', 'unknown')}",
                )
            ],
            "exchange": income.get("total", {}),
            "local": {
                "realized_pnl": local_realized_pnl,
                "commission": local_commission,
            },
        }

    ex_total = income.get("total", {})
    ex_realized = float(ex_total.get("realized_pnl", 0.0))
    ex_commission = float(ex_total.get("commission", 0.0))
    ex_funding = float(ex_total.get("funding_fee", 0.0))
    ex_net = float(ex_total.get("net_income", 0.0))

    # ---- Raw PnL delta (local realized vs exchange REALIZED_PNL) ----
    # This is the "true" PnL gap before any adjustments.
    # A large gap here suggests the local entry/exit PnL algorithm disagrees
    # with the exchange — possible causes:
    #   - Phantom positions inflating local PnL
    #   - Missed fill reports
    #   - Price rounding / fee-inclusive price differences
    raw_pnl_delta = local_realized_pnl - ex_realized
    tol_raw = max(50.0, abs(ex_realized) * 0.10)  # 10% tolerance, floor 50 USDT

    if abs(raw_pnl_delta) > tol_raw:
        issues.append(
            _issue(
                kind="realized_pnl_gap",
                scope=scope,
                message=(
                    f"已实现PnL原始差异: 本地 {local_realized_pnl:+.2f} vs 交易所 {ex_realized:+.2f} "
                    f"(Δ={raw_pnl_delta:+.2f}, tol={tol_raw:.2f})"
                ),
                local_realized_pnl=local_realized_pnl,
                exchange_realized_pnl=ex_realized,
                raw_pnl_delta=raw_pnl_delta,
                tolerance_usdt=tol_raw,
            )
        )

    # ---- Adjusted net PnL (Plan A) ----
    # Local PnL is gross (no commission deducted), because the engine doesn't
    # record commission from Binance user-stream.  Use the exchange COMMISSION
    # and FUNDING_FEE as the authoritative source to compute a local "net" figure
    # and compare it against the exchange net income.
    adjusted_local_net = local_realized_pnl + ex_commission + ex_funding
    delta_net = adjusted_local_net - ex_net
    tol_net = max(10.0, abs(ex_net) * 0.05)  # 5% tolerance, floor 10 USDT

    if abs(delta_net) > tol_net:
        issues.append(
            _issue(
                kind="net_pnl_mismatch",
                scope=scope,
                message=(
                    f"净收入差异: 本地调整后 {adjusted_local_net:+.2f} vs 交易所 {ex_net:+.2f} "
                    f"(Δ={delta_net:+.2f}, tol={tol_net:.2f})"
                ),
                adjusted_local_net=adjusted_local_net,
                exchange_net=ex_net,
                delta_net=delta_net,
                tolerance_usdt=tol_net,
            )
        )

    # ---- Abnormal commission (absolute threshold) ----
    _commission_alert_threshold = float(
        __import__("os").environ.get("MLBOT_COMMISSION_ALERT_THRESHOLD_USDT", "500")
    )
    if abs(ex_commission) > _commission_alert_threshold:
        issues.append(
            _issue(
                kind="commission_abnormal",
                scope=scope,
                message=(
                    f"⚠ 异常手续费: 交易所累计 {ex_commission:+.2f} USDT "
                    f"超过阈值 {_commission_alert_threshold:.0f} USDT，"
                    "请检查是否存在滑点、手续费率异常或频繁交易"
                ),
                exchange_commission=ex_commission,
                threshold_usdt=_commission_alert_threshold,
            )
        )

    # Informational: local commission is not recorded (engine user-stream bug)
    if abs(ex_commission) > 1.0 and local_commission == 0.0:
        issues.append(
            _issue(
                kind="commission_not_recorded",
                scope=scope,
                message=(
                    f"手续费未记录: 本地 DB 为 0，交易所 {ex_commission:+.2f} USDT "
                    "（引擎 user-stream 未写入，已用交易所数据替代）"
                ),
                exchange_commission=ex_commission,
            )
        )

    # Informational: significant funding fees
    if abs(ex_funding) > 10.0:
        issues.append(
            _issue(
                kind="funding_fee_significant",
                scope=scope,
                message=(
                    f"资金费率费用累计 {ex_funding:+.2f} USDT（交易所），"
                    "已纳入本地净收入计算"
                ),
                exchange_funding_fee=ex_funding,
            )
        )

    result = {
        "ok": len(issues) == 0,
        "scope": scope,
        "issues": issues,
        "exchange": {
            "realized_pnl": ex_realized,
            "commission": ex_commission,
            "funding_fee": ex_funding,
            "net_income": ex_net,
            "record_count": income.get("record_count", 0),
            "by_symbol": income.get("by_symbol", {}),
            "fetched_at": income.get("fetched_at"),
        },
        "local": {
            "realized_pnl": float(local_realized_pnl),
            "commission": float(local_commission),
            "adjusted_net": float(adjusted_local_net),
            "delta_net": float(delta_net),
            "raw_pnl_delta": float(raw_pnl_delta),
        },
    }

    # ---- Fire TG alerts for abnormal conditions ----
    # Best-effort: alert failures must not break the reconciliation response.
    try:
        check_reconciliation_alerts(
            reconciliation_result=result,
            lookback_days=lookback_days,
        )
    except Exception:
        logger.warning("reconcile_realized_pnl: tg alert check failed", exc_info=True)

    return result
