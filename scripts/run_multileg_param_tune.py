#!/usr/bin/env python3
"""Batch multileg parameter tune from YAML variant grids.

Example::

    python scripts/run_multileg_param_tune.py \\
      --tune-yaml config/experiments/20260618_multileg_param_tune/chop_tune.yaml

    python scripts/run_multileg_param_tune.py \\
      --tune-yaml config/experiments/20260618_multileg_param_tune/trend_tune.yaml \\
      --variants baseline,no_replenish
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHOP_OVERRIDE_CLI: Dict[str, str] = {
    "chop_min": "--chop-min",
    "exit_chop_min": "--exit-chop-min",
    "grid_pct": "--grid-pct",
    "grid_atr_mult": "--grid-atr-mult",
    "max_levels": "--max-levels",
    "max_replenish_per_level": "--max-replenish-per-level",
    "tp_spacing_mult": "--tp-spacing-mult",
    "max_loss_per_grid": "--max-loss-per-grid",
    "max_open_levels_total": "--max-open-levels-total",
    "min_segment_bars": "--min-segment-bars",
    "max_segment_bars": "--max-segment-bars",
    "box_pos_min": "--box-pos-min",
    "box_pos_max": "--box-pos-max",
    "chop_signal": "--chop-signal",
    "maker_fee_bps": "--maker-fee-bps",
    "taker_fee_bps": "--taker-fee-bps",
    "forced_exit_slippage_bps": "--forced-exit-slippage-bps",
}

TREND_OVERRIDE_CLI: Dict[str, str] = {
    "trend_min": "--trend-min",
    "trend_exit_min": "--trend-exit-min",
    "chop_min": "--chop-min",
    "exit_chop_min": "--exit-chop-min",
    "step_atr_mult": "--step-atr-mult",
    "tp_atr_mult": "--tp-atr-mult",
    "tp_pct": "--tp-pct",
    "max_adds_per_side": "--max-adds-per-side",
    "max_net_exposure": "--max-net-exposure",
    "max_gross_exposure": "--max-gross-exposure",
    "max_loss_per_segment": "--max-loss-per-segment",
    "risk_stop_mode": "--risk-stop-mode",
    "max_loser_hold_bars": "--max-loser-hold-bars",
    "flip_action": "--flip-action",
    "fee_bps": "--fee-bps",
    "entry_slippage_bps": "--entry-slippage-bps",
    "add_slippage_bps": "--add-slippage-bps",
    "market_exit_slippage_bps": "--market-exit-slippage-bps",
    "execution_timeframe": "--execution-timeframe",
}

BOOL_FLAGS = frozenset(
    {
        "no_maps",
        "initial_hedge",
        "reseed_on_flip",
        "reseed_on_loser_timeout",
        "scale_max_loser_hold_to_signal",
        "block_stable_box",
        "exclude_box",
    }
)


def _load_tune_yaml(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected dict in tune yaml: {path}")
    return raw


def _merge_backtest(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(overrides)
    return merged


def _append_cli_value(cmd: List[str], flag: str, value: Any) -> None:
    if value is None:
        cmd.extend([flag, "null"])
        return
    if isinstance(value, bool):
        cmd.append(flag)
        return
    cmd.extend([flag, str(value)])


def _append_bool_flag(cmd: List[str], key: str, value: Any) -> None:
    if value is None:
        return
    enabled = bool(value)
    if key == "no_maps":
        if enabled:
            cmd.append("--no-maps")
        return
    if key == "initial_hedge":
        cmd.append("--initial-hedge" if enabled else "--no-initial-hedge")
        return
    if key == "reseed_on_flip":
        cmd.append("--reseed-on-flip" if enabled else "--no-reseed-on-flip")
        return
    if key == "reseed_on_loser_timeout":
        cmd.append("--reseed-on-loser-timeout" if enabled else "--no-reseed-on-loser-timeout")
        return
    if key == "scale_max_loser_hold_to_signal":
        cmd.append("--scale-max-loser-hold-to-signal" if enabled else "--no-scale-max-loser-hold-to-signal")
        return
    if key == "block_stable_box":
        cmd.append("--block-stable-box" if enabled else "--no-block-stable-box")
        return
    if key == "exclude_box":
        cmd.append("--exclude-box" if enabled else "--no-exclude-box")
        return


def _build_chop_cmd(
    *,
    tune: Dict[str, Any],
    variant: Dict[str, Any],
    out_dir: Path,
) -> List[str]:
    bt = dict(tune.get("backtest") or {})
    overrides = variant.get("overrides") or {}
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/chop_grid_backtest.py"),
        "--config",
        str(PROJECT_ROOT / str(tune.get("base_config", "config/strategies/chop_grid/meta.yaml"))),
        "--data-dir",
        str(bt.get("data_dir", "data/parquet_data")),
        "--symbols",
        str(bt.get("symbols", "BTCUSDT")),
        "--start",
        str(bt.get("start", "2024-01-01")),
        "--end",
        str(bt.get("end", "2026-05-31")),
        "--warmup-days",
        str(int(bt.get("warmup_days", 120))),
        "--initial-capital",
        str(float(bt.get("initial_capital", 10_000))),
        "--out-dir",
        str(out_dir),
    ]
    for key in ("maker_fee_bps", "taker_fee_bps", "forced_exit_slippage_bps"):
        if key in bt and key not in overrides:
            _append_cli_value(cmd, CHOP_OVERRIDE_CLI[key], bt[key])
    for key, flag in CHOP_OVERRIDE_CLI.items():
        if key in overrides:
            _append_cli_value(cmd, flag, overrides[key])
    for key in BOOL_FLAGS:
        if key in overrides:
            _append_bool_flag(cmd, key, overrides[key])
        elif key in bt:
            _append_bool_flag(cmd, key, bt[key])
    return cmd


def _build_trend_cmd(
    *,
    tune: Dict[str, Any],
    variant: Dict[str, Any],
    out_dir: Path,
) -> List[str]:
    bt = dict(tune.get("backtest") or {})
    overrides = variant.get("overrides") or {}
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/diagnose_dual_add_trend.py"),
        "--config",
        str(PROJECT_ROOT / str(tune.get("base_config", "config/strategies/trend_scalp/meta.yaml"))),
        "--data-dir",
        str(bt.get("data_dir", "data/parquet_data")),
        "--symbols",
        str(bt.get("symbols", "BTCUSDT")),
        "--start",
        str(bt.get("start", "2024-01-01")),
        "--end",
        str(bt.get("end", "2026-05-31")),
        "--warmup-days",
        str(int(bt.get("warmup_days", 120))),
        "--initial-capital",
        str(float(bt.get("initial_capital", 10_000))),
        "--out-dir",
        str(out_dir),
    ]
    for key in (
        "max_loser_hold_bars",
        "execution_timeframe",
        "fee_bps",
        "entry_slippage_bps",
        "add_slippage_bps",
        "market_exit_slippage_bps",
    ):
        if key in bt and key not in overrides:
            flag = TREND_OVERRIDE_CLI.get(key, f"--{key.replace('_', '-')}")
            _append_cli_value(cmd, flag, bt[key])
    for key, flag in TREND_OVERRIDE_CLI.items():
        if key in overrides:
            _append_cli_value(cmd, flag, overrides[key])
    for key in BOOL_FLAGS:
        if key in overrides:
            _append_bool_flag(cmd, key, overrides[key])
        elif key in bt:
            _append_bool_flag(cmd, key, bt[key])
    return cmd


def _extract_chop_metrics(out_dir: Path) -> Dict[str, Any]:
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        return {"error": f"missing {metrics_path}"}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    m = payload.get("metrics") or {}
    trade = m.get("trade_summary") or {}
    seg = m.get("segment_summary") or {}
    return {
        "return_pct_timeline": trade.get("return_pct_timeline"),
        "n_trades": trade.get("n_trades"),
        "n_segments": seg.get("n_segments"),
        "win_rate_trades": trade.get("win_rate"),
        "forced_rate": trade.get("forced_rate"),
        "segment_win_rate": seg.get("win_rate"),
        "max_drawdown": m.get("max_drawdown"),
        "sharpe_r": m.get("sharpe_r"),
    }


def _extract_trend_metrics(out_dir: Path) -> Dict[str, Any]:
    summary_path = out_dir / "summary.csv"
    if not summary_path.exists():
        return {"error": f"missing {summary_path}"}
    with summary_path.open(encoding="utf-8") as fh:
        row = next(csv.DictReader(fh), None)
    if not row:
        return {"error": "empty summary.csv"}
    out: Dict[str, Any] = {}
    for key in (
        "portfolio_pnl_per_capital_timeline",
        "return_pct_timeline",
        "n_trades",
        "n_segments",
        "trade_win_rate",
        "win_rate_trades",
        "forced_rate",
        "risk_stop_rate",
        "tp_rate",
        "loser_timeout_rate",
        "max_drawdown_r",
        "max_drawdown_portfolio",
        "sharpe_r",
    ):
        if key in row and row[key] not in ("", None):
            try:
                out[key] = float(row[key])
            except ValueError:
                out[key] = row[key]
    return out


def _run_variant(cmd: List[str], *, dry_run: bool) -> Tuple[int, str]:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return 0, ""
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)
    return proc.returncode, proc.stdout + proc.stderr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tune-yaml",
        required=True,
        help="Path to chop_tune.yaml or trend_tune.yaml",
    )
    ap.add_argument(
        "--variants",
        default="",
        help="Comma-separated variant ids (default: all in yaml)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--output-root",
        default="",
        help="Override output_root from yaml",
    )
    args = ap.parse_args()

    tune_path = Path(args.tune_yaml)
    if not tune_path.is_absolute():
        tune_path = PROJECT_ROOT / tune_path
    tune = _load_tune_yaml(tune_path)

    strategy = str(tune.get("strategy", "")).strip().lower()
    engine = str(tune.get("engine", "")).strip().lower()
    if not engine:
        engine = "chop_grid_backtest" if strategy == "chop_grid" else "diagnose_dual_add_trend"

    output_root = Path(args.output_root or tune.get("output_root") or "results/param_tune")
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    all_variants: List[Dict[str, Any]] = list(tune.get("variants") or [])
    if not all_variants:
        raise SystemExit("no variants in tune yaml")

    filter_ids = {x.strip() for x in args.variants.split(",") if x.strip()}
    if filter_ids:
        variants = [v for v in all_variants if str(v.get("id", "")) in filter_ids]
        missing = filter_ids - {str(v.get("id", "")) for v in variants}
        if missing:
            raise SystemExit(f"unknown variant ids: {sorted(missing)}")
    else:
        variants = all_variants

    rows: List[Dict[str, Any]] = []
    for variant in variants:
        vid = str(variant.get("id", "unnamed"))
        out_dir = output_root / vid
        if engine == "chop_grid_backtest":
            cmd = _build_chop_cmd(tune=tune, variant=variant, out_dir=out_dir)
            extract = _extract_chop_metrics
        elif engine in ("diagnose_dual_add_trend", "trend_scalp"):
            cmd = _build_trend_cmd(tune=tune, variant=variant, out_dir=out_dir)
            extract = _extract_trend_metrics
        else:
            raise SystemExit(f"unsupported engine: {engine!r}")

        rc, _ = _run_variant(cmd, dry_run=args.dry_run)
        row: Dict[str, Any] = {
            "id": vid,
            "description": variant.get("description", ""),
            "overrides": variant.get("overrides") or {},
            "out_dir": str(out_dir),
            "exit_code": rc,
        }
        if not args.dry_run and rc == 0:
            row.update(extract(out_dir))
        rows.append(row)

    comparison = {
        "tune_yaml": str(tune_path),
        "strategy": strategy,
        "engine": engine,
        "output_root": str(output_root),
        "variants": rows,
    }
    if not args.dry_run:
        (output_root / "comparison.json").write_text(
            json.dumps(comparison, indent=2, default=str), encoding="utf-8"
        )
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with (output_root / "comparison.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote comparison -> {output_root / 'comparison.json'}")
        print(f"Wrote comparison -> {output_root / 'comparison.csv'}")


if __name__ == "__main__":
    main()
