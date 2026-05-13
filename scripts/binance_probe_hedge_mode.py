#!/usr/bin/env python3
"""Probe Binance USD-M Futures hedge mode (GET /fapi/v1/positionSide/dual).

Does not print secrets. Typical usage on the server::

    PYTHONPATH=src python scripts/binance_probe_hedge_mode.py \\
      --env-file /opt/quant-engine/live/binance_mainnet.env --profile multi-leg

Classic stack keys (quant-engine)::

    PYTHONPATH=src python scripts/binance_probe_hedge_mode.py \\
      --env-file /opt/quant-engine/live/binance_mainnet.env --profile classic

Optional: after a successful probe (no HTTP/auth error), try switching to hedge::

    PYTHONPATH=src python scripts/binance_probe_hedge_mode.py ... \\
      --profile multi-leg --try-enable-hedge

Closing positions / cancelling orders may be required before Binance accepts mode switch.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.order_management.binance_api import BinanceAPI  # noqa: E402


def _apply_env_file(path: Path, *, override: bool) -> None:
    text = path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = val


def _mask(s: str) -> str:
    s = str(s).strip()
    if len(s) <= 8:
        return "(too_short)"
    return f"{s[:4]}…{s[-4:]}"


def _resolve_keys(profile: str, allow_shared: bool) -> tuple[str, str]:
    if profile == "classic":
        k = os.getenv("BINANCE_API_KEY", "").strip()
        s = os.getenv("BINANCE_API_SECRET", "").strip()
        if not k or not s:
            raise SystemExit(
                "classic profile needs BINANCE_API_KEY and BINANCE_API_SECRET "
                "(after --env-file if used)."
            )
        return k, s

    k = os.getenv("MULTI_LEG_BINANCE_FUTURES_API_KEY", "").strip()
    s = os.getenv("MULTI_LEG_BINANCE_FUTURES_API_SECRET", "").strip()
    if (not k or not s) and allow_shared:
        k = os.getenv("BINANCE_API_KEY", "").strip()
        s = os.getenv("BINANCE_API_SECRET", "").strip()
    if not k or not s:
        raise SystemExit(
            "multi-leg profile needs MULTI_LEG_BINANCE_FUTURES_API_KEY/SECRET "
            "or pass --allow-shared-account with BINANCE_* set."
        )
    return k, s


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional dotenv-style file (lines KEY=value); overrides env when set.",
    )
    p.add_argument(
        "--profile",
        choices=("classic", "multi-leg"),
        default="classic",
        help="classic = BINANCE_* (trend/engine); multi-leg = MULTI_LEG_* (+optional shared).",
    )
    p.add_argument(
        "--allow-shared-account",
        action="store_true",
        help="multi-leg profile only: fallback to BINANCE_* if MULTI_LEG_* missing.",
    )
    p.add_argument(
        "--testnet",
        action="store_true",
        help="Use Binance Futures testnet endpoints.",
    )
    p.add_argument(
        "--try-enable-hedge",
        action="store_true",
        help="If probe succeeds and account is one-way, POST dualSide=true then refresh.",
    )
    args = p.parse_args()

    if args.env_file:
        if not args.env_file.is_file():
            raise SystemExit(f"env file not found: {args.env_file}")
        _apply_env_file(args.env_file, override=True)

    api_key, api_secret = _resolve_keys(args.profile, args.allow_shared_account)
    label = "MULTI_LEG_*" if args.profile == "multi-leg" else "BINANCE_*"
    print(f"Using profile={args.profile} key_source≈{label} api_key={_mask(api_key)}")

    api = BinanceAPI(
        api_key=api_key,
        api_secret=api_secret,
        testnet=bool(args.testnet),
        use_proxy=None,
    )

    print(f"hedge_mode={api.hedge_mode!r}")
    print(f"hedge_mode_probe_error={api.hedge_mode_probe_error!r}")

    if api.hedge_mode_probe_error:
        print(
            "\nProbe failed: fix Futures API permissions / IP whitelist / key pair "
            "before interpreting hedge_mode.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if args.try_enable_hedge:
        if api.hedge_mode:
            print("Already hedge mode; nothing to POST.")
            raise SystemExit(0)
        print("POST /fapi/v1/positionSide/dual dualSide=true …")
        try:
            out = api.set_dual_side_position(True)
            print(
                f"POST raw response keys={list(out.keys()) if isinstance(out, dict) else type(out)}"
            )
        except Exception as exc:
            print(f"POST failed: {exc}", file=sys.stderr)
            raise SystemExit(3) from exc
        api.refresh_hedge_mode()
        print(f"after refresh hedge_mode={api.hedge_mode!r}")
        print(f"after refresh hedge_mode_probe_error={api.hedge_mode_probe_error!r}")
        raise SystemExit(0 if api.hedge_mode else 4)


if __name__ == "__main__":
    main()
