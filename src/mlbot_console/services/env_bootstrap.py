"""Load API keys from .env files for the business console."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_console_env_files(repo_root: Path) -> None:
    """Load API keys from .env files without overwriting existing environment variables."""
    # Order of precedence (lowest to highest priority, but load_dotenv(override=False) means first loaded wins)
    # Actually, we want to load them in order, and since override=False is default, the first file to define a variable wins.
    # But if we want the environment variables already present to win, load_dotenv handles that by default.
    
    # 1. /opt/quant-engine/.env (or MLBOT_CONSOLE_ENV_FILE)
    explicit_env = os.getenv("MLBOT_CONSOLE_ENV_FILE")
    if explicit_env:
        paths = [Path(explicit_env)]
    else:
        paths = [
            repo_root / ".env",
            repo_root / "live" / "binance_mainnet.env",
            repo_root / "live" / "binance_spot_mainnet.env",
        ]
        # Also try /opt/quant-engine paths if we are not in that directory
        opt_root = Path("/opt/quant-engine")
        if repo_root != opt_root:
            paths.extend([
                opt_root / ".env",
                opt_root / "live" / "binance_mainnet.env",
                opt_root / "live" / "binance_spot_mainnet.env",
            ])

    loaded_any = False
    for p in paths:
        if p.is_file():
            load_dotenv(dotenv_path=str(p), override=False)
            logger.info("Loaded env file: %s", p)
            loaded_any = True
            
    if not loaded_any:
        logger.debug("No .env files found for console API keys.")


def credentials_status() -> Dict[str, Dict[str, Any]]:
    """Return the configuration status of exchange API keys."""
    from mlbot_console.services.exchange_balances import _SCOPE_META
    
    out: Dict[str, Dict[str, Any]] = {}
    for scope, meta in _SCOPE_META.items():
        key_envs = meta["key_envs"]
        secret_envs = meta["secret_envs"]
        
        has_key = any(bool(os.getenv(k, "").strip()) for k in key_envs)
        has_secret = any(bool(os.getenv(k, "").strip()) for k in secret_envs)
        
        out[scope] = {
            "configured": has_key and has_secret,
            "key_envs": key_envs,
        }
    return out
