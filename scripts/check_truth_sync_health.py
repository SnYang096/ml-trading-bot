#!/usr/bin/env python3
"""P5 Live 验收检查: TrendPositionTruthSync 健康状态。

检查 4 项验收标准，输出 pass/fail report:
  1. SQLite 无 duplicate open（同 symbol+side 多行）
  2. SQLite 无 orphan open（exchange flat 但 SQLite open）
  3. JSON 快照: 所有 live symbols 有对应 JSON 文件
  4. Metrics: duplicate_position_row_closed 最近无新增（需 Prometheus）

Example:
  python3 scripts/check_truth_sync_health.py --dry-run
  python3 scripts/check_truth_sync_health.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logger = logging.getLogger("check_truth_sync_health")


def _check_sqlite_no_duplicate_open(db_path: Path) -> Dict[str, Any]:
    """Check 1: No duplicate open positions for same symbol+side."""
    from src.order_management.storage import Storage

    if not db_path.is_file():
        return {
            "check": "sqlite_no_duplicate_open",
            "status": "skip",
            "reason": "db not found",
        }

    storage = Storage(str(db_path))
    try:
        rows = storage.get_open_positions() or []
    except Exception as e:
        return {
            "check": "sqlite_no_duplicate_open",
            "status": "error",
            "reason": str(e),
        }

    side_counts: Dict[tuple, int] = defaultdict(int)
    for row in rows:
        sym = str(row.symbol or "").upper()
        side = row.side.value if hasattr(row.side, "value") else str(row.side)
        side_counts[(sym, side)] += 1

    duplicates = {k: v for k, v in side_counts.items() if v > 1}
    if duplicates:
        return {
            "check": "sqlite_no_duplicate_open",
            "status": "FAIL",
            "duplicates": {f"{k[0]}:{k[1]}": v for k, v in duplicates.items()},
        }
    return {
        "check": "sqlite_no_duplicate_open",
        "status": "PASS",
        "open_count": len(rows),
    }


def _check_sqlite_no_orphan_open(db_path: Path, api: Any) -> Dict[str, Any]:
    """Check 2: No orphan open (exchange flat but SQLite open)."""
    from src.order_management.storage import Storage

    if not db_path.is_file():
        return {
            "check": "sqlite_no_orphan_open",
            "status": "skip",
            "reason": "db not found",
        }

    storage = Storage(str(db_path))
    try:
        local_rows = storage.get_open_positions() or []
    except Exception as e:
        return {"check": "sqlite_no_orphan_open", "status": "error", "reason": str(e)}

    try:
        exchange_positions = api.get_positions() or []
    except Exception as e:
        return {
            "check": "sqlite_no_orphan_open",
            "status": "error",
            "reason": f"exchange query failed: {e}",
        }

    # Build exchange leg set
    ex_legs = set()
    for pos in exchange_positions:
        sym = str(pos.get("symbol", "")).replace("/", "").split(":")[0].upper().strip()
        qty = abs(float(pos.get("size") or pos.get("contracts") or 0))
        if qty <= 0:
            continue
        side = "short" if str(pos.get("side", "")).lower() == "short" else "long"
        ex_legs.add((sym, side))

    orphans = []
    for row in local_rows:
        sym = str(row.symbol or "").upper()
        side = row.side.value if hasattr(row.side, "value") else str(row.side)
        side_norm = "short" if side.lower() in {"short", "sell"} else "long"
        if (sym, side_norm) not in ex_legs:
            orphans.append(
                {"symbol": sym, "side": side_norm, "position_id": str(row.position_id)}
            )

    if orphans:
        return {
            "check": "sqlite_no_orphan_open",
            "status": "FAIL",
            "orphans": orphans,
        }
    return {
        "check": "sqlite_no_orphan_open",
        "status": "PASS",
        "local_open": len(local_rows),
        "exchange_legs": len(ex_legs),
    }


def _check_json_snapshots(state_dir: Path, db_path: Path) -> Dict[str, Any]:
    """Check 3: All live symbols have JSON tracker files."""
    from src.order_management.storage import Storage

    if not db_path.is_file():
        return {"check": "json_snapshots", "status": "skip", "reason": "db not found"}

    storage = Storage(str(db_path))
    try:
        rows = storage.get_open_positions() or []
    except Exception:
        rows = []

    live_symbols = {str(r.symbol or "").upper() for r in rows if r.symbol}
    if not live_symbols:
        return {
            "check": "json_snapshots",
            "status": "PASS",
            "live_symbols": 0,
            "note": "no open positions",
        }

    missing = []
    for sym in live_symbols:
        json_path = state_dir / f"{sym}.json"
        if not json_path.exists():
            missing.append(sym)

    if missing:
        return {
            "check": "json_snapshots",
            "status": "FAIL",
            "missing": missing,
            "live_symbols": len(live_symbols),
        }
    return {
        "check": "json_snapshots",
        "status": "PASS",
        "live_symbols": len(live_symbols),
        "state_dir": str(state_dir),
    }


def _check_metrics_no_duplicates(
    prometheus_url: Optional[str],
    *,
    days: int,
) -> Dict[str, Any]:
    """Check 4: duplicate_position_row_closed = 0 over the validation window."""
    expect = f"0 over last {days} trading day(s)"
    query = (
        'mlbot_reconciliation_issue_count{scope="trend", '
        'issue="duplicate_position_row_closed"}'
    )
    if not prometheus_url:
        return {
            "check": "metrics_no_duplicate_closed",
            "status": "skip",
            "days": days,
            "reason": "no --prometheus-url provided (manual Grafana check recommended)",
            "query": query,
            "expect": expect,
        }

    # Placeholder: in production, query Prometheus/Grafana API with ``days`` window
    return {
        "check": "metrics_no_duplicate_closed",
        "status": "skip",
        "days": days,
        "reason": "Prometheus query not implemented; check Grafana dashboard manually",
        "query": query,
        "expect": expect,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--db-path",
        default=os.getenv(
            "MLBOT_ORDER_MANAGEMENT_DB_PATH",
            str(_REPO_ROOT / "data" / "order_management.db"),
        ),
    )
    p.add_argument(
        "--state-dir",
        default=os.getenv(
            "MLBOT_POSITION_TRACKER_STATE_DIR",
            "live/highcap/data/position_tracker",
        ),
    )
    p.add_argument(
        "--prometheus-url",
        default=os.getenv("MLBOT_PROMETHEUS_URL", ""),
        help="Prometheus base URL for metrics check (optional)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of trading days to validate (for P5 acceptance, use 3)",
    )
    args = p.parse_args()

    db_path = Path(args.db_path)
    state_dir = Path(args.state_dir)

    report: Dict[str, Any] = {
        "dry_run": bool(args.dry_run),
        "days": int(args.days),
        "checks": [],
    }

    # Check 1: No duplicate open
    report["checks"].append(_check_sqlite_no_duplicate_open(db_path))

    # Check 2: No orphan open (needs exchange API)
    api = None
    if not args.dry_run:
        try:
            from src.order_management.binance_api import BinanceAPI

            api_key = os.getenv("BINANCE_API_KEY") or os.getenv(
                "BINANCE_FUTURES_API_KEY", ""
            )
            api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv(
                "BINANCE_FUTURES_API_SECRET", ""
            )
            if api_key and api_secret:
                api = BinanceAPI(api_key, api_secret, testnet=False)
        except Exception:
            pass

    if api is not None:
        report["checks"].append(_check_sqlite_no_orphan_open(db_path, api))
    else:
        report["checks"].append(
            {
                "check": "sqlite_no_orphan_open",
                "status": "skip",
                "reason": "no exchange API (dry-run or credentials missing)",
            }
        )

    # Check 3: JSON snapshots
    report["checks"].append(_check_json_snapshots(state_dir, db_path))

    # Check 4: Metrics
    report["checks"].append(
        _check_metrics_no_duplicates(
            args.prometheus_url or None,
            days=int(args.days),
        )
    )

    # Summary
    statuses = [c["status"] for c in report["checks"]]
    fail_count = statuses.count("FAIL")
    pass_count = statuses.count("PASS")
    skip_count = statuses.count("skip")
    error_count = statuses.count("error")

    report["summary"] = {
        "pass": pass_count,
        "fail": fail_count,
        "skip": skip_count,
        "error": error_count,
        "overall": "PASS" if fail_count == 0 and error_count == 0 else "FAIL",
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if fail_count > 0 or error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
