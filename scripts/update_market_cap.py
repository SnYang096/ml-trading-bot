from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pandas as pd
import requests
import yaml


def _coingecko_headers(
    api_key: str | None, *, force_pro: bool = False
) -> Dict[str, str] | None:
    if not api_key:
        return None
    # Heuristic:
    # - demo keys commonly look like "CG-...." -> use demo header only
    # - otherwise use pro header only
    if (not force_pro) and str(api_key).startswith("CG-"):
        return {"x-cg-demo-api-key": api_key}
    return {"x-cg-pro-api-key": api_key}


def _coingecko_effective_url(url: str, api_key: str | None) -> str:
    """
    CoinGecko requires Pro keys to hit pro-api.coingecko.com (otherwise 400 with error_code=10010).
    Demo keys (CG-*) use api.coingecko.com.
    """
    if not api_key:
        return url
    is_demo = str(api_key).startswith("CG-")
    if is_demo:
        return url
    # Pro key: if user is hitting the public host, rewrite to pro host.
    # Support both with/without trailing slash after /api/v3
    url = url.replace(
        "https://api.coingecko.com/api/v3/",
        "https://pro-api.coingecko.com/api/v3/",
    )
    url = url.replace(
        "https://api.coingecko.com/api/v3",
        "https://pro-api.coingecko.com/api/v3",
    )
    return url


def _coingecko_get(
    url: str, *, params: Dict[str, Any], api_key: str | None = None
) -> Dict[str, Any]:
    # Light retry for transient 429/5xx
    last_err: Exception | None = None
    for attempt in range(6):
        try:
            eff_url = _coingecko_effective_url(url, api_key)
            headers = _coingecko_headers(api_key)
            resp = requests.get(eff_url, params=params, headers=headers, timeout=60)
            if resp.status_code in (429, 500, 502, 503, 504):
                # Record last error so caller sees something meaningful if we exhaust retries.
                last_err = RuntimeError(f"HTTP {resp.status_code} for {resp.url}")
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_s = float(retry_after)
                    except Exception:
                        sleep_s = min((2**attempt) * 5.0, 120.0)
                else:
                    sleep_s = min((2**attempt) * 5.0, 120.0)
                time.sleep(sleep_s)
                continue
            try:
                resp.raise_for_status()
            except Exception as e:
                # Preserve body for debugging (CoinGecko often returns helpful JSON error payload)
                body = (resp.text or "").strip()
                body_short = body[:8000]  # avoid huge dumps
                # Special case: CoinGecko Pro keys require the pro host (error_code=10010).
                # Sometimes keys look like demo keys but behave like pro keys; detect and retry.
                if (
                    resp.status_code == 400
                    and ("error_code" in body_short and "10010" in body_short)
                    and "pro-api.coingecko.com" in body_short
                    and "pro-api.coingecko.com" not in eff_url
                ):
                    eff_url2 = eff_url.replace(
                        "https://api.coingecko.com/api/v3/",
                        "https://pro-api.coingecko.com/api/v3/",
                    ).replace(
                        "https://api.coingecko.com/api/v3",
                        "https://pro-api.coingecko.com/api/v3",
                    )
                    headers2 = _coingecko_headers(api_key, force_pro=True)
                    resp2 = requests.get(
                        eff_url2, params=params, headers=headers2, timeout=60
                    )
                    if resp2.status_code in (429, 500, 502, 503, 504):
                        time.sleep(min(2**attempt, 30))
                        continue
                    try:
                        resp2.raise_for_status()
                        return resp2.json()
                    except Exception:
                        body2 = (resp2.text or "").strip()[:8000]
                        raise RuntimeError(
                            f"HTTP {resp2.status_code} for {resp2.url}. Body: {body2}"
                        ) from e
                raise RuntimeError(
                    f"HTTP {resp.status_code} for {resp.url}. Body: {body_short}"
                ) from e
            return resp.json()
        except Exception as e:
            last_err = e
            time.sleep(min((2**attempt) * 2.0, 30.0))
    raise RuntimeError(f"CoinGecko request failed after retries: {last_err}")


def fetch_market_caps_daily(
    *,
    coingecko_id: str,
    vs_currency: str = "usd",
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch full market cap history (daily-ish) from CoinGecko and return a daily series.

    CoinGecko returns:
      market_caps: [[ts_ms, mcap], ...]
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart"
    payload = _coingecko_get(
        url, params={"vs_currency": vs_currency, "days": "max"}, api_key=api_key
    )
    mcap = payload.get("market_caps", [])
    if not mcap:
        raise ValueError(f"Empty market_caps for coin id '{coingecko_id}'")
    df = pd.DataFrame(mcap, columns=["ts_ms", "market_cap_usd"])
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.drop(columns=["ts_ms"])
    df["date"] = df["timestamp"].dt.floor("D")
    # keep last value of each day
    out = (
        df.sort_values("timestamp")
        .groupby("date", as_index=False)["market_cap_usd"]
        .last()
        .set_index("date")
        .sort_index()
    )
    out.index.name = "date"
    return out


def fetch_market_cap_snapshot(
    *,
    coingecko_id: str,
    vs_currency: str = "usd",
    api_key: str | None = None,
) -> float:
    """
    Fetch current market cap (snapshot) for a coin id.

    Uses /coins/markets which returns market_cap directly.
    """
    # Primary: /coins/markets (fast)
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        payload = _coingecko_get(
            url,
            params={
                "vs_currency": vs_currency,
                "ids": coingecko_id,
                "order": "market_cap_desc",
                "per_page": 1,
                "page": 1,
                "sparkline": "false",
            },
            api_key=api_key,
        )
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"Empty /coins/markets payload for '{coingecko_id}'")
        mcap = payload[0].get("market_cap", None)
        if mcap is None:
            raise ValueError(
                f"Missing market_cap in /coins/markets payload for '{coingecko_id}'"
            )
        return float(mcap)
    except Exception:
        # Fallback: /coins/{id} -> market_data.market_cap[vs_currency]
        url2 = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}"
        payload2 = _coingecko_get(
            url2,
            params={
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false",
                "sparkline": "false",
            },
            api_key=api_key,
        )
        md = payload2.get("market_data", {}) or {}
        mc = (md.get("market_cap", {}) or {}).get(str(vs_currency).lower(), None)
        if mc is None:
            raise ValueError(
                f"Fallback /coins/{{id}} missing market_data.market_cap[{vs_currency}] for '{coingecko_id}'"
            )
        return float(mc)


def _load_universe_symbols(
    universe_yaml: str,
    *,
    universe_set: str | None,
) -> Tuple[str, list[str]]:
    """
    Load base tokens from config/download/crypto_4h_token_universe_groups.yaml and return:
      (quote, ["BTCUSDT", "ETHUSDT", ...])
    """
    ypath = Path(universe_yaml)
    cfg = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
    quote = str(cfg.get("quote", "USDT")).strip().upper() or "USDT"

    sets = cfg.get("universe_sets", {}) or {}
    if not sets:
        raise ValueError(f"No universe_sets found in {universe_yaml}")

    chosen_set = universe_set or next(iter(sets.keys()))
    if chosen_set not in sets:
        raise KeyError(
            f"Universe set '{chosen_set}' not found in {universe_yaml}. "
            f"Available: {list(sets.keys())[:20]}"
        )

    groups = (sets[chosen_set] or {}).get("groups", {}) or {}
    bases: list[str] = []
    for _, base_list in groups.items():
        if not base_list:
            continue
        for b in base_list:
            b = str(b).strip().upper()
            if b:
                bases.append(b)

    bases = sorted(set(bases))
    symbols = [f"{b}{quote}" for b in bases]
    return quote, symbols


def _infer_base_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    for q in ("USDT", "USD", "USDC", "BUSD", "FDUSD"):
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)]
    # If already a base token (BTC), return as-is
    return s


def _resolve_coingecko_id_for_symbol(
    sym: str,
    *,
    overrides: Dict[str, Dict[str, Any]],
    vs_currency: str,
    api_key: str | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Resolve CoinGecko coin id for a trading symbol like BTCUSDT.
    Priority:
      1) overrides[sym].coingecko_id
      2) search endpoint by base symbol (best-effort)
    """
    if sym in overrides:
        cid = str(overrides[sym].get("coingecko_id", "")).strip()
        if cid:
            return cid, {"source": "override"}

    base = _infer_base_symbol(sym)
    # CoinGecko search works well for canonical tickers (BTC, ETH, SOL, ...)
    url = "https://api.coingecko.com/api/v3/search"
    payload = _coingecko_get(url, params={"query": base.lower()}, api_key=api_key)
    coins = payload.get("coins", []) or []
    if not coins:
        raise ValueError(
            f"Unable to resolve CoinGecko id for {sym} (base={base}): empty search results"
        )

    base_l = base.lower()
    # Prefer exact symbol match; if multiple, pick best market_cap_rank (smaller is better).
    exact = [
        c
        for c in coins
        if str(c.get("symbol", "")).lower() == base_l and str(c.get("id", "")).strip()
    ]
    candidates = exact if exact else [c for c in coins if str(c.get("id", "")).strip()]
    if not candidates:
        raise ValueError(f"Unable to resolve CoinGecko id for {sym} (base={base})")

    def _rank_key(c: Dict[str, Any]) -> tuple[int, int]:
        # market_cap_rank: 1 is largest; missing rank should be worse
        r = c.get("market_cap_rank", None)
        try:
            r_int = int(r)
            if r_int <= 0:
                r_int = 10**9
        except Exception:
            r_int = 10**9
        # prefer exact symbol matches (already filtered), keep stable tie-break with original order
        return (r_int, 0)

    best = sorted(candidates, key=_rank_key)[0]
    meta = {
        "source": "search",
        "query": base.lower(),
        "picked_from": "exact_symbol" if exact else "top",
        "symbol": str(best.get("symbol", "")),
        "name": str(best.get("name", "")),
        "market_cap_rank": best.get("market_cap_rank", None),
    }
    return str(best["id"]), meta


def _backup_if_exists(path: Path) -> Path | None:
    if not path.exists():
        return None
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak.{ts}")
    bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return bak


def _write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    # keep YAML readable and stable
    path.write_text(
        yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _should_skip_existing(
    *,
    out_path: Path,
    mode: str,
    max_age_days: int,
    force: bool,
) -> bool:
    if force or not out_path.exists() or out_path.stat().st_size <= 0:
        return False
    m = str(mode or "static").lower().strip()
    if m == "daily":
        # daily mode is "max history" right now; if file exists, skip
        return True
    # static mode: treat as fresh if asof_date within max_age_days
    try:
        df = pd.read_parquet(out_path)
        if (
            isinstance(df.index, pd.DatetimeIndex)
            and "market_cap_usd" in df.columns
            and len(df) > 0
        ):
            asof = pd.to_datetime(df.index.max(), utc=True)
            age_days = (pd.Timestamp.now(tz="UTC").floor("D") - asof.floor("D")).days
            return age_days <= int(max_age_days)
    except Exception:
        return False
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="config/data/market_cap.yaml",
        help="YAML config defining provider + symbol->coingecko_id mapping",
    )
    # NOTE: symbols are auto-discovered from config (and optional universe_yaml).
    # Keep the flag for overrides/debug.
    ap.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated symbols to update (default: auto from config/universe)",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Override data_dir in config (default: use config.data_dir)",
    )
    ap.add_argument(
        "--writeback-config",
        action="store_true",
        help="Write back discovered symbol->coingecko_id mappings into the config YAML",
    )
    ap.add_argument(
        "--no-writeback-config",
        action="store_true",
        help="Disable writeback of discovered mappings (override default behavior)",
    )
    ap.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write update manifest JSON alongside data files",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if a fresh file exists.",
    )
    ap.add_argument(
        "--max-age-days",
        type=int,
        default=1,
        help="For static mode: skip symbols if existing snapshot is within this many days.",
    )
    ap.add_argument(
        "--sleep-sec",
        type=float,
        default=0.1,
        help="Sleep between symbols (rate-limit friendly).",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    provider = str(cfg.get("provider", "coingecko")).lower()
    if provider != "coingecko":
        raise ValueError(f"Unsupported provider: {provider}")

    vs_currency = str(cfg.get("vs_currency", "usd"))
    sym_cfg: Dict[str, Dict[str, Any]] = cfg.get("symbols", {}) or {}
    mode = str(cfg.get("mode", "static")).lower().strip()  # static|daily
    api_key_env = (
        str(cfg.get("api_key_env", "COINGECKO_API_KEY")).strip() or "COINGECKO_API_KEY"
    )
    api_key = os.getenv(api_key_env)
    # Safety: if someone accidentally put the API key itself into api_key_env,
    # accept it but do NOT print it (and strongly recommend using env vars).
    if not api_key and api_key_env.lower().startswith("cg-") and len(api_key_env) >= 20:
        api_key = api_key_env
        print(
            "⚠️  Detected a CoinGecko key-like string in config. "
            "For safety, please move it to an env var (COINGECKO_API_KEY) instead of committing secrets."
        )
    if not api_key:
        print(
            "⚠️  CoinGecko API key not found. Requests may fail with 401. "
            "Set it like: export COINGECKO_API_KEY=<your_key>"
        )

    # Determine symbols to update:
    # - If --symbols provided: use it
    # - Else: union(config.symbols keys, universe_yaml symbols if configured)
    chosen = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    symbols: list[str]
    if chosen:
        symbols = chosen
    else:
        symbols = sorted(set(sym_cfg.keys()))
        universe_yaml = str(cfg.get("universe_yaml", "")).strip()
        if universe_yaml:
            universe_set = cfg.get("universe_set", None)
            _, uni_symbols = _load_universe_symbols(
                universe_yaml, universe_set=universe_set
            )
            symbols = sorted(set(symbols) | set(uni_symbols))

    if not symbols:
        raise ValueError(
            "No symbols selected for update (empty config + empty universe)"
        )

    out_dir = Path(args.output_dir or cfg.get("data_dir", "data/market_cap"))
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "provider": provider,
        "vs_currency": vs_currency,
        "updated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "symbols": {},
    }

    # Default: always write back discovered mappings, unless explicitly disabled.
    do_writeback = True
    if args.no_writeback_config:
        do_writeback = False
    if args.writeback_config:
        do_writeback = True

    discovered: Dict[str, Dict[str, Any]] = {}
    cfg.setdefault("symbols", {})

    for sym in symbols:
        out_path = out_dir / f"{sym}.parquet"
        if _should_skip_existing(
            out_path=out_path,
            mode=mode,
            max_age_days=int(args.max_age_days),
            force=bool(args.force),
        ):
            print(f"⏩ Skip {sym}: existing snapshot is fresh ({out_path})")
            # still record manifest entry so users know it was skipped
            manifest["symbols"][sym] = {
                "mode": mode,
                "path": str(out_path),
                "skipped": True,
            }
            continue

        coin_id, meta = _resolve_coingecko_id_for_symbol(
            sym,
            overrides=sym_cfg,
            vs_currency=vs_currency,
            api_key=api_key,
        )
        if meta.get("source") == "search":
            discovered[sym] = {"coingecko_id": coin_id, **meta}
            # persist mapping in-memory (writeback later)
            if (
                sym not in cfg["symbols"]
                or not str(cfg["symbols"][sym].get("coingecko_id", "")).strip()
            ):
                cfg["symbols"][sym] = {"coingecko_id": coin_id}

        print(f"📥 Fetching market cap: {sym} ({coin_id})")
        if mode == "daily":
            df_daily = fetch_market_caps_daily(
                coingecko_id=coin_id, vs_currency=vs_currency, api_key=api_key
            )
            df_daily.to_parquet(out_path)
            manifest["symbols"][sym] = {
                "coingecko_id": coin_id,
                "mode": "daily",
                "rows": int(df_daily.shape[0]),
                "min_date": str(df_daily.index.min()),
                "max_date": str(df_daily.index.max()),
                "path": str(out_path),
                **({"resolved_by": meta} if meta else {}),
            }
        elif mode == "static":
            mcap = fetch_market_cap_snapshot(
                coingecko_id=coin_id, vs_currency=vs_currency, api_key=api_key
            )
            # pandas may return tz-aware UTC already (newer versions); keep this robust.
            asof = pd.Timestamp.now(tz="UTC").floor("D")
            df_static = pd.DataFrame(
                {"market_cap_usd": [mcap]}, index=pd.DatetimeIndex([asof], name="date")
            )
            df_static.to_parquet(out_path)
            manifest["symbols"][sym] = {
                "coingecko_id": coin_id,
                "mode": "static",
                "rows": 1,
                "asof_date": str(asof),
                "market_cap_usd": float(mcap),
                "path": str(out_path),
                **({"resolved_by": meta} if meta else {}),
            }
        else:
            raise ValueError(f"Unknown mode in config: {mode} (expected static|daily)")

        time.sleep(float(args.sleep_sec))

    if do_writeback and discovered:
        bak = _backup_if_exists(cfg_path)
        _write_yaml(cfg_path, cfg)
        print(
            f"🧷 Wrote back {len(discovered)} symbol->coingecko_id mappings into {cfg_path}"
            + (f" (backup: {bak})" if bak else "")
        )

    if args.write_manifest:
        mpath = out_dir / "market_cap_manifest.json"
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"🧾 Wrote manifest: {mpath}")

    print("✅ Done.")


if __name__ == "__main__":
    main()
