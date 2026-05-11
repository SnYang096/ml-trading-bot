#!/usr/bin/env python3
"""
Tune locked prefilter thresholds with rolling windows.

This script keeps locked semantic features fixed, but searches threshold values
on validation/test windows by repeatedly running auto_research_pipeline.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# BPC：网格键 = infer 产出的 param 名；新增 locked 标量后请在 locked_threshold_tuning 增加 ``{param}_values``，
# 否则 tune 使用启发式默认（见 _bpc_grid_csv_for_param）。
_BPC_GRID_DEFAULTS = {
    "bpc_recent_breakout_strength_min": "0.40,0.50",
    "bpc_pullback_depth_max": "0.55,0.65",
    "bpc_recovery_strength_min": "0.50,0.60",
}


def _binding_param_grid_csv(param_name: str, tcfg: Dict[str, Any]) -> str:
    """Resolve ``{param}_values`` for inferred binding params (BPC / bindings templates)."""
    key = f"{param_name}_values"
    if key in tcfg and tcfg[key] is not None:
        return str(tcfg[key])
    if param_name in _BPC_GRID_DEFAULTS:
        return _BPC_GRID_DEFAULTS[param_name]
    if param_name.endswith("_min"):
        return "0.35,0.45,0.55"
    if param_name.endswith("_max"):
        return "0.55,0.65,0.75"
    return "0.45,0.55"


from scripts.locked_prefilter_utils import (
    build_override_prefilter as build_override_prefilter_base,
    infer_writeback_bindings_from_prefilter,
)
from scripts.pipeline.config import load_pipeline_config


@dataclass
class CaseParams:
    """ME uses explicit fields; BPC / bindings use ``custom_params`` (infer 导出的参数名)."""

    mode: str = "bindings"
    atr_lower: float = 0.0
    atr_upper: float = 0.0
    me_accel_abs_min: float = 0.0
    me_cvd_min: float = 0.0
    compression_min: float = 0.0
    decay_upper: float = 0.0
    oi_min: float = 0.0
    custom_params: Dict[str, float] | None = None

    def key(self) -> str:
        if self.custom_params:
            parts = [
                f"{k}={float(v):.4g}" for k, v in sorted(self.custom_params.items())
            ]
            return "cfg{" + ", ".join(parts) + "}"
        if self.mode == "me":
            return (
                f"atr[{self.atr_lower:.4g},{self.atr_upper:.4g}]_"
                f"accel_abs>={self.me_accel_abs_min:.4g}_"
                f"cvd>={self.me_cvd_min:.4g}"
            )
        raise RuntimeError("bindings/bpc cases must set custom_params")


def parse_float_list(text: str) -> List[float]:
    vals = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(float(p))
    return vals


def parse_text_list(text: str) -> List[str]:
    out = []
    for part in text.split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


def parse_values_list(v: Any) -> List[float]:
    if isinstance(v, str):
        return parse_float_list(v)
    if isinstance(v, (list, tuple)):
        out: List[float] = []
        for x in v:
            out.append(float(x))
        return out
    if isinstance(v, (int, float)):
        return [float(v)]
    raise ValueError(f"unsupported values payload: {type(v).__name__}")


def list_strategy_runs(history_dir: Path, strategy: str) -> List[str]:
    strat_dir = history_dir / strategy
    if not strat_dir.exists():
        return []
    return sorted([p.name for p in strat_dir.iterdir() if p.is_dir()])


def build_override_prefilter(
    prod_prefilter_path: Path,
    params: CaseParams,
    output_path: Path,
    *,
    writeback_bindings: List[Dict[str, Any]] | None = None,
    strict_bindings: bool = True,
) -> Path:
    if params.custom_params is not None:
        payload = dict(params.custom_params)
    elif params.mode == "me":
        payload = {
            "atr_lower": params.atr_lower,
            "atr_upper": params.atr_upper,
            "me_accel_abs_min": params.me_accel_abs_min,
            "me_cvd_min": params.me_cvd_min,
        }
    else:
        raise RuntimeError(
            "BPC/bindings locked tuning requires CaseParams.custom_params "
            "(infer 参数名，与 archetypes/prefilter.yaml 一致)"
        )
    tpl = params.mode
    if tpl == "fer":
        tpl = "bindings"
    return build_override_prefilter_base(
        prod_prefilter_path,
        output_path,
        payload,
        bindings=writeback_bindings,
        strict_bindings=strict_bindings,
        template=tpl,
    )


def run_one_window(
    strategy: str,
    config_path: Path,
    end_date: str,
    override_prefilter: Path,
    history_dir: Path,
    skip_shap: bool = False,
) -> Dict[str, Any]:
    before = set(list_strategy_runs(history_dir, strategy))
    cmd = [
        sys.executable,
        "scripts/auto_research_pipeline.py",
        "--strategy",
        strategy,
        "--config",
        str(config_path),
        "--no-adopt",
        "--locked-prefilter-override",
        str(override_prefilter),
    ]
    if end_date:
        cmd.extend(["--end-date", end_date])
    if skip_shap:
        cmd.append("--skip-shap")
    cmd.append("--disable-auto-locked-tuning")

    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    after = set(list_strategy_runs(history_dir, strategy))
    new_runs = sorted(after - before)
    run_id = new_runs[-1] if new_runs else None
    report = None
    metrics: Dict[str, Any] = {}
    decision = "ERROR"

    if run_id:
        rp = history_dir / strategy / run_id / "report.json"
        if rp.exists():
            report = rp
            data = json.loads(rp.read_text(encoding="utf-8"))
            metrics = data.get("backtest_metrics", {}) or {}
            decision = (data.get("comparison", {}) or {}).get("decision", "UNKNOWN")

    return {
        "return_code": proc.returncode,
        "run_id": run_id,
        "report_path": str(report) if report else "",
        "decision": decision,
        "metrics": metrics,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def aggregate_case(
    case_results: List[Dict[str, Any]],
    target_trades_min: int,
    target_trades_max: int,
    trade_penalty_low: float,
    trade_penalty_high: float,
    stability_penalty: float,
) -> Dict[str, Any]:
    sharpes: List[float] = []
    trades: List[int] = []
    ok_windows = 0

    for r in case_results:
        m = r.get("metrics") or {}
        if not m:
            continue
        s = m.get("sharpe_per_trade")
        t = m.get("total_trades")
        if isinstance(s, (float, int)) and isinstance(t, (float, int)):
            sharpes.append(float(s))
            trades.append(int(t))
            ok_windows += 1

    if not sharpes:
        return {
            "ok_windows": 0,
            "median_sharpe": float("-inf"),
            "positive_ratio": 0.0,
            "median_trades": 0.0,
            "low_gap": 0.0,
            "high_gap": 0.0,
            "sharpe_std": 0.0,
            "score": float("-inf"),
        }

    median_sharpe = statistics.median(sharpes)
    positive_ratio = sum(1 for s in sharpes if s > 0) / len(sharpes)
    median_trades = float(statistics.median(trades))
    low_gap = max(0.0, float(target_trades_min) - median_trades)
    high_gap = max(0.0, median_trades - float(target_trades_max))
    sharpe_std = statistics.pstdev(sharpes) if len(sharpes) > 1 else 0.0
    score = (
        median_sharpe
        - trade_penalty_low * low_gap
        - trade_penalty_high * high_gap
        - stability_penalty * sharpe_std
    )

    return {
        "ok_windows": ok_windows,
        "median_sharpe": median_sharpe,
        "positive_ratio": positive_ratio,
        "median_trades": median_trades,
        "low_gap": low_gap,
        "high_gap": high_gap,
        "sharpe_std": sharpe_std,
        "score": score,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Tune locked prefilter thresholds")
    p.add_argument(
        "--strategy",
        default="fer-short",
        help="与管线 YAML strategies.* 键一致；BPC/ME 分别为 bpc、me（勿再用 bpc-long 等作键名）",
    )
    p.add_argument(
        "--template",
        choices=["auto", "fer", "bindings", "me", "bpc"],
        default="auto",
        help="locked 调优模板: auto / fer(同 bindings) / bindings / me / bpc；bindings+bpc 使用 infer 参数名",
    )
    p.add_argument("--config", default="config/pipelines/pcm_orchestrate_2h.yaml")
    p.add_argument(
        "--end-dates",
        default="",
        help="comma-separated end dates; empty means single latest-date run",
    )
    p.add_argument("--fer-lower-values", default="0.0,0.05")
    p.add_argument("--fer-upper-values", default="0.25,0.35,0.45")
    p.add_argument("--sr-min-values", default="0.45,0.55,0.65")
    p.add_argument("--dist-max-values", default="0.8,1.2,1.6")
    p.add_argument("--sqs-min-values", default="0.45,0.55,0.65")
    p.add_argument("--atr-lower-values", default="0.15,0.25,0.35")
    p.add_argument("--atr-upper-values", default="0.70,0.82,0.90")
    p.add_argument("--me-accel-abs-min-values", default="0.08,0.12,0.16")
    p.add_argument("--me-cvd-min-values", default="0.50,0.60,0.70")
    p.add_argument("--compression-min-values", default="0.02,0.05,0.08")
    p.add_argument("--decay-upper-values", default="0.15,0.25,0.35")
    p.add_argument("--oi-min-values", default="0.25,0.35,0.45")
    p.add_argument("--max-cases", type=int, default=0, help="0 means all")
    p.add_argument("--min-trades-target", type=int, default=60)
    p.add_argument(
        "--max-trades-target",
        type=int,
        default=0,
        help="0 means auto derive from min-trades-target",
    )
    p.add_argument(
        "--trade-penalty",
        type=float,
        default=0.002,
        help="legacy alias for trade-penalty-low",
    )
    p.add_argument("--trade-penalty-low", type=float, default=None)
    p.add_argument("--trade-penalty-high", type=float, default=0.001)
    p.add_argument("--stability-penalty", type=float, default=0.0)
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--output-dir",
        default="results/locked_tuning",
        help="where tuning summary files are saved",
    )
    args = p.parse_args()

    cfg_path = Path(args.config)
    # Research configs commonly use ``extends``; use the same loader as the
    # main pipeline so low-freedom overlays still expose inherited strategy
    # fields such as ``strategies.<name>.config``.
    cfg = load_pipeline_config(cfg_path)
    scfg = (cfg.get("strategies", {}) or {}).get(args.strategy, {})
    if not scfg:
        raise SystemExit(f"unknown strategy: {args.strategy}")
    tcfg = ((scfg.get("kpi_gates", {}) or {}).get("prefilter", {}) or {}).get(
        "locked_threshold_tuning", {}
    ) or {}

    prod_cfg_dir = PROJECT_ROOT / scfg["config"]
    prod_prefilter = prod_cfg_dir / "archetypes" / "prefilter.yaml"
    if not prod_prefilter.exists():
        raise SystemExit(f"prefilter not found: {prod_prefilter}")

    raw_pf: Dict[str, Any] = (
        yaml.safe_load(prod_prefilter.read_text(encoding="utf-8")) or {}
    )
    explicit_wb = [
        b for b in (tcfg.get("writeback_bindings") or []) if isinstance(b, dict)
    ]
    resolved_writeback = explicit_wb or infer_writeback_bindings_from_prefilter(raw_pf)

    out_root = PROJECT_ROOT / args.output_dir / args.strategy
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    history_dir = PROJECT_ROOT / (cfg.get("output", {}) or {}).get(
        "history_dir", "results/research_history"
    )

    end_dates = parse_text_list(args.end_dates)
    if not end_dates:
        # Empty means let pipeline auto-detect latest end date once.
        end_dates = [""]

    template = args.template
    if template == "fer":
        template = "bindings"
    if template == "auto":
        if args.strategy == "me":
            template = "me"
        elif args.strategy == "bpc":
            template = "bpc"
        else:
            template = "bindings"

    strict_bindings = bool(tcfg.get("writeback_strict", True))
    generic_search_space = tcfg.get("search_space", {}) or {}

    cases: List[CaseParams] = []
    if (
        isinstance(generic_search_space, dict)
        and generic_search_space
        and resolved_writeback
        and template in ("bpc", "bindings")
    ):
        keys = [str(k) for k in generic_search_space.keys()]
        values_lists = [parse_values_list(generic_search_space[k]) for k in keys]
        for combo in itertools.product(*values_lists):
            params_map = {k: float(v) for k, v in zip(keys, combo)}
            cases.append(CaseParams(mode=template, custom_params=params_map))
    elif template == "me":
        atr_lows = parse_float_list(
            str(tcfg.get("atr_lower_values", args.atr_lower_values))
        )
        atr_ups = parse_float_list(
            str(tcfg.get("atr_upper_values", args.atr_upper_values))
        )
        accel_abs_mins = parse_float_list(
            str(tcfg.get("me_accel_abs_min_values", args.me_accel_abs_min_values))
        )
        cvd_mins = parse_float_list(
            str(tcfg.get("me_cvd_min_values", args.me_cvd_min_values))
        )
        for lo, hi, accel_abs, cvd_min in itertools.product(
            atr_lows, atr_ups, accel_abs_mins, cvd_mins
        ):
            if lo > hi:
                continue
            cases.append(
                CaseParams(
                    mode="me",
                    atr_lower=lo,
                    atr_upper=hi,
                    me_accel_abs_min=accel_abs,
                    me_cvd_min=cvd_min,
                )
            )
    elif template in ("bpc", "bindings"):
        keys = [
            str(b.get("param", "")).strip()
            for b in resolved_writeback
            if isinstance(b, dict) and str(b.get("param", "")).strip()
        ]
        if not keys:
            raise SystemExit(
                "locked tuning: 无 writeback 绑定；请在 locked_threshold_tuning 写 "
                "writeback_bindings，或检查 archetypes/prefilter.yaml 的 locked 规则"
            )
        value_lists: List[List[float]] = []
        for k in keys:
            value_lists.append(parse_float_list(_binding_param_grid_csv(k, tcfg)))
        for combo in itertools.product(*value_lists):
            cases.append(
                CaseParams(mode=template, custom_params=dict(zip(keys, combo)))
            )
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    elif int(tcfg.get("max_cases", 0) or 0) > 0:
        cases = cases[: int(tcfg.get("max_cases", 0))]

    target_trades_min = int(
        tcfg.get(
            "target_trades_min",
            tcfg.get("min_trades_target", args.min_trades_target),
        )
    )
    _default_max = (
        int(args.max_trades_target)
        if int(args.max_trades_target or 0) > 0
        else int(target_trades_min * 4)
    )
    target_trades_max = int(tcfg.get("target_trades_max", _default_max))
    _legacy_low = (
        args.trade_penalty_low
        if args.trade_penalty_low is not None
        else args.trade_penalty
    )
    trade_penalty_low = float(
        tcfg.get("trade_penalty_low", tcfg.get("trade_penalty", _legacy_low))
    )
    trade_penalty_high = float(tcfg.get("trade_penalty_high", args.trade_penalty_high))
    stability_penalty = float(tcfg.get("stability_penalty", args.stability_penalty))

    print("=" * 90)
    print(
        f"🔬 Locked Prefilter Threshold Tuning: {args.strategy} (template={template})"
    )
    print(f"Cases={len(cases)}, Windows={len(end_dates)}, dry_run={args.dry_run}")
    print("=" * 90)

    summary_rows: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases, 1):
        print(f"\n[{idx}/{len(cases)}] {case.key()}")
        case_dir = out_dir / f"case_{idx:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        override_path = build_override_prefilter(
            prod_prefilter,
            case,
            case_dir / "prefilter_locked_override.yaml",
            writeback_bindings=resolved_writeback if resolved_writeback else None,
            strict_bindings=strict_bindings,
        )

        per_window: List[Dict[str, Any]] = []
        for win_idx, end_date in enumerate(end_dates, 1):
            if args.dry_run:
                r = {
                    "window": win_idx,
                    "end_date": end_date or "<auto>",
                    "return_code": 0,
                    "run_id": "",
                    "decision": "DRY_RUN",
                    "metrics": {},
                }
            else:
                print(
                    f"  - window {win_idx}/{len(end_dates)} end_date={end_date or '<auto>'}"
                )
                r = run_one_window(
                    strategy=args.strategy,
                    config_path=cfg_path,
                    end_date=end_date if end_date else "",
                    override_prefilter=override_path,
                    history_dir=history_dir,
                    skip_shap=args.skip_shap,
                )
            per_window.append(r)

        agg = aggregate_case(
            per_window,
            target_trades_min=target_trades_min,
            target_trades_max=target_trades_max,
            trade_penalty_low=trade_penalty_low,
            trade_penalty_high=trade_penalty_high,
            stability_penalty=stability_penalty,
        )
        row = {
            "case_id": idx,
            "mode": case.mode,
            "fer_lower": 0.0,
            "fer_upper": 0.0,
            "sr_min": 0.0,
            "dist_max": 0.0,
            "fer_sqs_min": 0.0,
            "atr_lower": case.atr_lower,
            "atr_upper": case.atr_upper,
            "me_accel_abs_min": case.me_accel_abs_min,
            "me_cvd_min": case.me_cvd_min,
            "compression_min": case.compression_min,
            "decay_upper": case.decay_upper,
            "oi_min": case.oi_min,
            "custom_params_json": json.dumps(
                case.custom_params or {}, ensure_ascii=False
            ),
            **agg,
            "windows": per_window,
        }
        summary_rows.append(row)
        print(
            f"  => score={row['score']:+.4f}, median_sharpe={row['median_sharpe']:+.4f}, "
            f"positive_ratio={row['positive_ratio']:.1%}, median_trades={row['median_trades']:.1f}, "
            f"low_gap={row['low_gap']:.1f}, high_gap={row['high_gap']:.1f}, sharpe_std={row['sharpe_std']:.4f}"
        )

    summary_rows.sort(key=lambda x: x["score"], reverse=True)

    summary_json = out_dir / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "strategy": args.strategy,
                "config": str(cfg_path),
                "end_dates": end_dates,
                "target_trades_min": target_trades_min,
                "target_trades_max": target_trades_max,
                "trade_penalty_low": trade_penalty_low,
                "trade_penalty_high": trade_penalty_high,
                "stability_penalty": stability_penalty,
                "rows": summary_rows,
                "template": template,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case_id",
                "mode",
                "fer_lower",
                "fer_upper",
                "sr_min",
                "dist_max",
                "fer_sqs_min",
                "atr_lower",
                "atr_upper",
                "me_accel_abs_min",
                "me_cvd_min",
                "compression_min",
                "decay_upper",
                "oi_min",
                "custom_params_json",
                "score",
                "median_sharpe",
                "positive_ratio",
                "median_trades",
                "low_gap",
                "high_gap",
                "sharpe_std",
                "ok_windows",
            ]
        )
        for r in summary_rows:
            w.writerow(
                [
                    r["case_id"],
                    r["mode"],
                    r["fer_lower"],
                    r["fer_upper"],
                    r["sr_min"],
                    r["dist_max"],
                    r["fer_sqs_min"],
                    r["atr_lower"],
                    r["atr_upper"],
                    r["me_accel_abs_min"],
                    r["me_cvd_min"],
                    r["compression_min"],
                    r["decay_upper"],
                    r["oi_min"],
                    r["custom_params_json"],
                    r["score"],
                    r["median_sharpe"],
                    r["positive_ratio"],
                    r["median_trades"],
                    r["low_gap"],
                    r["high_gap"],
                    r["sharpe_std"],
                    r["ok_windows"],
                ]
            )

    print("\nTop 5 cases:")
    for i, r in enumerate(summary_rows[:5], 1):
        if r["mode"] == "me":
            param_desc = (
                f"atr=[{r['atr_lower']:.3g},{r['atr_upper']:.3g}] "
                f"accel_abs>={r['me_accel_abs_min']:.3g} "
                f"cvd>={r['me_cvd_min']:.3g}"
            )
        else:
            param_desc = f"params={r.get('custom_params_json')}"
        print(
            f"  {i}. case={r['case_id']:03d} score={r['score']:+.4f} "
            f"sharpe={r['median_sharpe']:+.4f} trades={r['median_trades']:.1f} "
            f"{param_desc}"
        )

    print(f"\n✅ Saved: {summary_json}")
    print(f"✅ Saved: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
