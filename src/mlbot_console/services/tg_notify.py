"""Telegram alerting for CMS reconciliation and account monitoring.

Thin wrapper around ``monitoring.telegram.send_telegram_message`` so
the CMS layer can fire alerts for:

- Abnormal commission (absolute threshold)
- Large PnL reconciliation gaps
- Commission-not-recorded warnings

Environment variables
---------------------
``MLBOT_COMMISSION_ALERT_THRESHOLD_USDT``
    Absolute commission threshold (default 500 USDT).
``MLBOT_PNL_GAP_ALERT_THRESHOLD_USDT``
    Raw PnL gap threshold (default 1000 USDT).
``MLBOT_TG_RECON_COOLDOWN_SEC``
    Cooldown between reconciliation alerts (default 3600 = 1 hour).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ---- thresholds ----
COMMISSION_ALERT_THRESHOLD_USDT = _env_float(
    "MLBOT_COMMISSION_ALERT_THRESHOLD_USDT", 500.0
)
PNL_GAP_ALERT_THRESHOLD_USDT = _env_float("MLBOT_PNL_GAP_ALERT_THRESHOLD_USDT", 1000.0)
TG_RECON_COOLDOWN_SEC = int(_env_float("MLBOT_TG_RECON_COOLDOWN_SEC", 3600.0))


def _send_tg(text: str, *, stamp_key: str, cooldown_sec: int | None = None) -> bool:
    """Send a Telegram message via the shared monitoring module."""
    try:
        from monitoring.telegram import send_telegram_message

        return send_telegram_message(
            text,
            stamp_key=stamp_key,
            cooldown_sec=(
                cooldown_sec if cooldown_sec is not None else TG_RECON_COOLDOWN_SEC
            ),
        )
    except Exception:
        logger.warning("tg_notify: send failed for key=%s", stamp_key, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Message formatters
# ---------------------------------------------------------------------------


def _fmt_usdt(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:+,.2f}"


def format_commission_alert(
    *,
    scope: str,
    lookback_days: int,
    ex_commission: float,
    ex_net: float,
    ex_realized: float,
    local_realized: float,
    threshold: float | None = None,
) -> str:
    """Format a Telegram message for abnormal commission."""
    th = threshold or COMMISSION_ALERT_THRESHOLD_USDT
    lines = [
        f"⚠️ 异常手续费告警 — {scope}",
        f"回看: {lookback_days} 天",
        "",
        f"交易所手续费: {_fmt_usdt(ex_commission)} USDT",
        f"告警阈值:     {abs(th):,.0f} USDT",
        "",
        f"交易所 REALIZED_PNL: {_fmt_usdt(ex_realized)}",
        f"本地 已实现PnL:      {_fmt_usdt(local_realized)}",
        f"交易所 净收入:       {_fmt_usdt(ex_net)}",
        "",
        f"⚠ 手续费超过阈值，请检查交易策略是否存在滑点、手续费率异常或频繁交易问题。",
    ]
    return "\n".join(lines)


def format_pnl_gap_alert(
    *,
    scope: str,
    lookback_days: int,
    local_realized: float,
    ex_realized: float,
    ex_commission: float,
    ex_funding: float,
    ex_net: float,
    adjusted_local_net: float,
    delta_net: float,
    tol_net: float,
    threshold: float | None = None,
) -> str:
    """Format a Telegram message for large PnL reconciliation gap."""
    th = threshold or PNL_GAP_ALERT_THRESHOLD_USDT
    raw_delta = local_realized - ex_realized
    lines = [
        f"🚨 PnL 对账差异告警 — {scope}",
        f"回看: {lookback_days} 天",
        "",
        f"原始差额 (本地-交易所 REALIZED): {_fmt_usdt(raw_delta)} USDT",
        f"告警阈值: {abs(th):,.0f} USDT",
        "",
        f"本地 已实现PnL:        {_fmt_usdt(local_realized)}",
        f"交易所 REALIZED_PNL:   {_fmt_usdt(ex_realized)}",
        f"交易所 COMMISSION:     {_fmt_usdt(ex_commission)}",
        f"交易所 FUNDING_FEE:    {_fmt_usdt(ex_funding)}",
        f"交易所 净收入:         {_fmt_usdt(ex_net)}",
        "",
        f"调整后本地净收入:      {_fmt_usdt(adjusted_local_net)}",
        f"净收入差额:            {_fmt_usdt(delta_net)} (tol={tol_net:.2f})",
        "",
        f"⚠ 本地 PnL 计算与交易所记录存在显著差异，请排查是否存在漏单、幽灵仓位或价格取整问题。",
    ]
    return "\n".join(lines)


def format_commission_missing_alert(
    *,
    scope: str,
    lookback_days: int,
    ex_commission: float,
) -> str:
    """Format a Telegram message for commission not being recorded locally."""
    lines = [
        f"📋 手续费未记录提醒 — {scope}",
        f"回看: {lookback_days} 天",
        "",
        f"交易所手续费: {_fmt_usdt(ex_commission)} USDT",
        f"本地 DB 记录: 0.00 USDT",
        "",
        f"引擎 user-stream 未写入手续费字段（Binance n=0 已知问题）。",
        f"对账模块已使用交易所数据替代。",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Alert check entry point
# ---------------------------------------------------------------------------


def check_reconciliation_alerts(
    *,
    reconciliation_result: Dict[str, Any],
    lookback_days: int = 90,
) -> List[str]:
    """Check reconciliation result for alert conditions and send TG messages.

    Returns list of alert keys that were sent.
    """
    sent: List[str] = []

    scope = str(reconciliation_result.get("scope") or "unknown")
    ex = reconciliation_result.get("exchange") or {}
    lo = reconciliation_result.get("local") or {}
    issues = reconciliation_result.get("issues") or []

    ex_realized = float(ex.get("realized_pnl", 0.0))
    ex_commission = float(ex.get("commission", 0.0))
    ex_funding = float(ex.get("funding_fee", 0.0))
    ex_net = float(ex.get("net_income", 0.0))
    local_realized = float(lo.get("realized_pnl", 0.0))
    adjusted_net = float(lo.get("adjusted_net", 0.0))
    delta_net = float(lo.get("delta_net", 0.0))

    # Compute tolerance for net mismatch
    tol_net_alert = max(10.0, abs(ex_net) * 0.05)

    # ---- Alert 1: Abnormal commission ----
    if abs(ex_commission) > COMMISSION_ALERT_THRESHOLD_USDT:
        msg = format_commission_alert(
            scope=scope,
            lookback_days=lookback_days,
            ex_commission=ex_commission,
            ex_net=ex_net,
            ex_realized=ex_realized,
            local_realized=local_realized,
        )
        if _send_tg(msg, stamp_key=f"recon:commission:{scope}"):
            sent.append("commission_high")

    # ---- Alert 2: Large PnL gap ----
    raw_delta = local_realized - ex_realized
    if abs(raw_delta) > PNL_GAP_ALERT_THRESHOLD_USDT:
        msg = format_pnl_gap_alert(
            scope=scope,
            lookback_days=lookback_days,
            local_realized=local_realized,
            ex_realized=ex_realized,
            ex_commission=ex_commission,
            ex_funding=ex_funding,
            ex_net=ex_net,
            adjusted_local_net=adjusted_net,
            delta_net=delta_net,
            tol_net=tol_net_alert,
        )
        if _send_tg(msg, stamp_key=f"recon:pnl_gap:{scope}"):
            sent.append("pnl_gap_large")

    # ---- Alert 3: Commission not recorded (informational, low frequency) ----
    has_commission_issue = any(
        i.get("kind") == "commission_not_recorded" for i in issues
    )
    if has_commission_issue and abs(ex_commission) > 50.0:
        msg = format_commission_missing_alert(
            scope=scope,
            lookback_days=lookback_days,
            ex_commission=ex_commission,
        )
        # Long cooldown — once per day
        if _send_tg(
            msg, stamp_key=f"recon:commission_missing:{scope}", cooldown_sec=86400
        ):
            sent.append("commission_missing")

    if sent:
        logger.info(
            "tg_notify: reconciliation alerts sent: scope=%s keys=%s",
            scope,
            sent,
        )

    return sent
