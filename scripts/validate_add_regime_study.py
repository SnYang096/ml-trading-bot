#!/usr/bin/env python3
"""Bar-level add-on regime study and event-backtest validation."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger("validate_add_regime_study")


def _copy_bpc_tree(dst_parent: Path) -> Path:
    src = _REPO / "config" / "strategies" / "bpc"
    dst = dst_parent / "bpc"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst_parent


def _copy_constitution(dst_parent: Path) -> Path:
    src = _REPO / "config" / "constitution" / "constitution.yaml"
    dst_dir = dst_parent / "constitution"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "constitution.yaml"
    shutil.copy2(src, dst)
    return dst


def _patch_execution(dst_parent: Path, mutator: Callable[[dict], None]) -> None:
    p = dst_parent / "bpc" / "archetypes" / "execution.yaml"
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    mutator(doc)
    p.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _patch_constitution(p: Path, mutator: Callable[[dict], None]) -> None:
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    mutator(doc)
    p.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _run(cmd: List[str], *, cwd: Path) -> int:
    logger.info("run: %s", " ".join(cmd))
    return int(subprocess.run(cmd, cwd=str(cwd)).returncode)


def _run_event_backtest(
    *,
    repo: Path,
    strategies_root: Path,
    start_date: str,
    end_date: str,
    symbols_csv: str,
    data_path: Path,
    out_json: Path,
    out_csv: Path,
    cap_dir: Path,
    no_kill_switch: bool,
    inject_ml: Optional[Path],
    fee_rate: float,
    constitution_yaml: Optional[Path] = None,
) -> int:
    cmd: List[str] = [
        sys.executable,
        str(repo / "scripts" / "event_backtest.py"),
        "--strategy",
        "bpc",
        "--strategies-root",
        str(strategies_root),
        "--data-path",
        str(data_path),
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--symbols",
        symbols_csv,
        "--output",
        str(out_json),
        "--export",
        str(out_csv),
        "--capital-report",
        str(cap_dir),
        "--fee-rate",
        str(fee_rate),
        "--quiet-signal-logs",
    ]
    if no_kill_switch:
        cmd.append("--no-kill-switch")
    if inject_ml is not None:
        cmd.extend(["--inject-add-ml-scores", str(inject_ml)])
    if constitution_yaml is not None:
        cmd.extend(["--constitution-yaml", str(constitution_yaml)])
    return _run(cmd, cwd=repo)


def _read_metrics(out_json: Path) -> Dict[str, Any]:
    if not out_json.is_file():
        return {"error": "missing_json"}
    obj = json.loads(out_json.read_text(encoding="utf-8"))
    f = obj.get("funnel") or {}
    ap = obj.get("add_position_stats") or {}
    return {
        "n_trades": int(obj.get("n_trades", 0)),
        "total_r": float(obj.get("total_r", 0.0)),
        "max_drawdown_r": float(obj.get("max_drawdown_r", 0.0)),
        "sharpe_r": float(obj.get("sharpe_r", 0.0)),
        "reject_kill_switch": int(f.get("reject_kill_switch", 0) or 0),
        "signals_generated": int(f.get("signals_generated", 0) or 0),
        "add_count": int(ap.get("add_count", 0) or 0),
        "add_rejected_count": int(ap.get("rejected_count", 0) or 0),
        "add_mean_r": float(ap.get("add_mean_r", 0.0) or 0.0),
        "add_win_rate": float(ap.get("add_win_rate", 0.0) or 0.0),
        "max_observed_leverage": float(ap.get("max_observed_leverage", 0.0) or 0.0),
        "max_observed_notional_frac": float(
            ap.get("max_observed_notional_frac", 0.0) or 0.0
        ),
        "reject_add_locked_profit_required": int(
            f.get("reject_add_locked_profit_required", 0) or 0
        ),
        "reject_pcm_trend_pool_unprotected_cap": int(
            f.get("reject_pcm_trend_pool_unprotected_cap", 0) or 0
        ),
        "reject_pcm_trend_pool_anchor_first": int(
            f.get("reject_pcm_trend_pool_anchor_first", 0) or 0
        ),
        "reject_pcm_trend_pool_post_unlock_cap": int(
            f.get("reject_pcm_trend_pool_post_unlock_cap", 0) or 0
        ),
        "reject_pcm_trend_symbol_conflict": int(
            f.get("reject_pcm_trend_symbol_conflict", 0) or 0
        ),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", default="2022-01-01")
    ap.add_argument("--end-date", default="2026-05-01")
    ap.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT",
    )
    ap.add_argument("--data-path", type=Path, default=_REPO / "data" / "parquet_data")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--fee-rate", type=float, default=0.0004)
    ap.add_argument("--no-kill-switch", action="store_true")
    ap.add_argument("--quick", action="store_true", help="2024-01–2025-12 缩窗")
    ap.add_argument("--skip-panel", action="store_true")
    ap.add_argument("--skip-rule", action="store_true")
    ap.add_argument("--skip-ml", action="store_true")
    ap.add_argument("--skip-event", action="store_true")
    ap.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse existing event_backtest_*.json files in --out-dir.",
    )
    ap.add_argument(
        "--scenarios",
        default="",
        help="Comma-separated scenario names to run/read; empty runs the full matrix.",
    )
    ap.add_argument("--timeframe", default="120T")
    ap.add_argument("--horizon-bars", type=int, default=12)
    ap.add_argument("--good-mfe-atr", type=float, default=1.0)
    ap.add_argument("--max-mae-atr", type=float, default=1.0)
    ap.add_argument("--quality-penalty", type=float, default=1.0)
    ap.add_argument("--current-chop-lte", type=float, default=0.55)
    args = ap.parse_args()

    start_date = args.start_date
    end_date = args.end_date
    if args.quick:
        start_date = "2024-01-01"
        end_date = "2025-12-31"

    out_dir = args.out_dir or (_REPO / "results" / "add_regime_study_latest")
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "strategy_variants"
    work.mkdir(exist_ok=True)
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    panel_path = analysis_dir / "add_bar_panel.parquet"
    panel_summary_path = analysis_dir / "add_bar_panel.summary.json"
    if not args.skip_panel:
        rc = _run(
            [
                sys.executable,
                str(_REPO / "scripts" / "build_add_bar_panel.py"),
                "--strategy",
                "bpc",
                "--strategies-root",
                "config/strategies",
                "--timeframe",
                str(args.timeframe),
                "--symbols",
                ",".join(symbols),
                "--data-path",
                str(args.data_path),
                "--start-date",
                start_date,
                "--end-date",
                end_date,
                "--horizon-bars",
                str(int(args.horizon_bars)),
                "--good-mfe-atr",
                str(float(args.good_mfe_atr)),
                "--max-mae-atr",
                str(float(args.max_mae_atr)),
                "--quality-penalty",
                str(float(args.quality_penalty)),
                "--output",
                str(panel_path),
                "--summary-json",
                str(panel_summary_path),
            ],
            cwd=_REPO,
        )
        if rc != 0:
            return rc

    rule_dir = analysis_dir / "rule_search"
    rule_dir.mkdir(exist_ok=True)
    if not args.skip_rule:
        rc = _run(
            [
                sys.executable,
                str(_REPO / "scripts" / "analyze_add_bar_rules.py"),
                "--panel",
                str(panel_path),
                "--out-dir",
                str(rule_dir),
            ],
            cwd=_REPO,
        )
        if rc != 0:
            return rc

    ml_dir = analysis_dir / "ml_train"
    ml_dir.mkdir(exist_ok=True)
    if not args.skip_ml:
        rc = _run(
            [
                sys.executable,
                str(_REPO / "scripts" / "train_add_bar_ml.py"),
                "--panel",
                str(panel_path),
                "--out-dir",
                str(ml_dir),
            ],
            cwd=_REPO,
        )
        if rc != 0:
            return rc

    best_rule = {}
    best_rule_path = rule_dir / "best_rule.json"
    if best_rule_path.exists():
        best_rule = json.loads(best_rule_path.read_text(encoding="utf-8"))
    best_ml = {}
    best_ml_path = ml_dir / "best_ml_gate.json"
    if best_ml_path.exists():
        best_ml = json.loads(best_ml_path.read_text(encoding="utf-8"))
    ml_score_path = ml_dir / "add_ml_scores.parquet"

    if args.skip_event:
        print(
            json.dumps(
                {
                    "panel": str(panel_path),
                    "best_rule": best_rule,
                    "best_ml": best_ml,
                    "ml_scores": str(ml_score_path),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    scenarios: List[Tuple[str, Path, Optional[Path], Optional[Path]]] = []

    def _mut_chop_base(doc: dict) -> None:
        ap0 = doc.setdefault("add_position", {})
        ap0["add_regime_gate"] = {
            "enabled": True,
            "allow_if_all": [
                {
                    "feature": "bpc_semantic_chop_ts_q",
                    "lte": float(args.current_chop_lte),
                },
            ],
        }

    def _mut_add_profile(
        doc: dict,
        *,
        max_add_times: int,
        multipliers: List[float],
        min_current_r_by_add: Optional[List[float]] = None,
    ) -> None:
        ap0 = doc.setdefault("add_position", {})
        ap0["max_add_times"] = int(max_add_times)
        ap0["add_size_multipliers"] = [float(x) for x in multipliers]
        if min_current_r_by_add is not None:
            ap0["min_current_r_by_add"] = [float(x) for x in min_current_r_by_add]

    def _mut_breakeven(
        doc: dict,
        *,
        enabled: bool,
        trigger_r: float = 1.0,
        lock_level_r: float = 0.0,
        measure: str = "atr",
    ) -> None:
        stop_loss = doc.setdefault("stop_loss", {})
        breakeven = stop_loss.setdefault("breakeven", {})
        breakeven["enabled"] = bool(enabled)
        breakeven["trigger_r"] = float(trigger_r)
        breakeven["lock_level_r"] = float(lock_level_r)
        breakeven["measure"] = str(measure)

    def _mut_constitution_common(
        doc: dict,
        *,
        require_locked_profit: bool,
        trend_pool_enabled: bool,
        max_unprotected_symbols: int,
        max_symbols_after_unlock: int,
        set_one_symbol_per_strategy: bool,
        allow_add_position: bool,
        anchor_symbol: str = "",
        require_anchor_first: bool = False,
    ) -> None:
        ra = doc.setdefault("resource_allocation", {})
        slot_policy = ra.setdefault("slot_policy", {})
        slot_policy["trend_pool_guard"] = {
            "enabled": bool(trend_pool_enabled),
            "max_unprotected_symbols": int(max_unprotected_symbols),
            "unlock_on": "breakeven_locked",
            "max_symbols_after_unlock": int(max_symbols_after_unlock),
        }
        if anchor_symbol:
            slot_policy["trend_pool_guard"]["anchor_symbol"] = str(anchor_symbol)
            slot_policy["trend_pool_guard"]["require_anchor_first"] = bool(
                require_anchor_first
            )
        psl = ra.setdefault("per_strategy_limits", {})
        for fam in ("bpc", "tpc", "me"):
            cfg = dict(psl.get(fam) or {})
            cfg["allow_add_position"] = bool(allow_add_position)
            cfg["require_locked_profit"] = bool(require_locked_profit)
            if set_one_symbol_per_strategy:
                cfg["max_slots"] = 1
            psl[fam] = cfg

    scenario_specs: List[Dict[str, Any]] = [
        {
            "name": "current_chop_055",
            "add_max": 3,
            "add_mult": [1.0, 2.0, 3.0],
            "require_locked_profit": False,
            "trend_pool_enabled": False,
            "max_unprotected_symbols": 0,
            "max_symbols_after_unlock": 0,
            "set_one_symbol_per_strategy": False,
            "allow_add_position": True,
            "breakeven_enabled": False,
        },
        {
            "name": "chop_055_add_1x_max1",
            "add_max": 1,
            "add_mult": [1.0],
            "require_locked_profit": True,
            "trend_pool_enabled": False,
            "max_unprotected_symbols": 0,
            "max_symbols_after_unlock": 0,
            "set_one_symbol_per_strategy": False,
            "allow_add_position": True,
            "breakeven_enabled": False,
        },
        {
            "name": "chop_055_add_light_max2",
            "add_max": 2,
            "add_mult": [0.5, 1.0],
            "require_locked_profit": True,
            "trend_pool_enabled": False,
            "max_unprotected_symbols": 0,
            "max_symbols_after_unlock": 0,
            "set_one_symbol_per_strategy": False,
            "allow_add_position": True,
            "breakeven_enabled": False,
        },
        {
            "name": "trend_pool_one_symbol_be_noadd",
            "add_max": 1,
            "add_mult": [0.5],
            "require_locked_profit": True,
            "trend_pool_enabled": True,
            "max_unprotected_symbols": 1,
            "max_symbols_after_unlock": 1,
            "set_one_symbol_per_strategy": False,
            "allow_add_position": False,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 1.0,
        },
        {
            "name": "trend_pool_unlock3_be_noadd",
            "add_max": 1,
            "add_mult": [0.5],
            "require_locked_profit": True,
            "trend_pool_enabled": True,
            "max_unprotected_symbols": 1,
            "max_symbols_after_unlock": 3,
            "set_one_symbol_per_strategy": False,
            "allow_add_position": False,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 1.0,
        },
        {
            "name": "trend_pool_unlock3_be1_add3_current_locked",
            "add_max": 3,
            "add_mult": [1.0, 2.0, 3.0],
            "add_min_r": [1.0, 2.0, 3.0],
            "require_locked_profit": True,
            "trend_pool_enabled": True,
            "max_unprotected_symbols": 1,
            "max_symbols_after_unlock": 3,
            "set_one_symbol_per_strategy": False,
            "allow_add_position": True,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 1.0,
        },
        {
            "name": "trend_pool_unlock3_be1_add3_light_locked",
            "add_max": 3,
            "add_mult": [0.5, 1.0, 1.5],
            "add_min_r": [1.0, 2.0, 3.0],
            "require_locked_profit": True,
            "trend_pool_enabled": True,
            "max_unprotected_symbols": 1,
            "max_symbols_after_unlock": 3,
            "set_one_symbol_per_strategy": False,
            "allow_add_position": True,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 1.0,
        },
        {
            "name": "trend_pool_btc_anchor_unlock3_be1_add3_current_locked",
            "add_max": 3,
            "add_mult": [1.0, 2.0, 3.0],
            "add_min_r": [1.0, 2.0, 3.0],
            "require_locked_profit": True,
            "trend_pool_enabled": True,
            "max_unprotected_symbols": 1,
            "max_symbols_after_unlock": 3,
            "set_one_symbol_per_strategy": False,
            "allow_add_position": True,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 1.0,
            "anchor_symbol": "BTCUSDT",
            "require_anchor_first": True,
        },
        {
            "name": "trend_pool_unlock3_be_add1_locked",
            "add_max": 1,
            "add_mult": [0.5],
            "add_min_r": [1.0],
            "require_locked_profit": True,
            "trend_pool_enabled": True,
            "max_unprotected_symbols": 1,
            "max_symbols_after_unlock": 3,
            "set_one_symbol_per_strategy": False,
            "allow_add_position": True,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 1.0,
        },
        {
            "name": "trend_pool_unlock3_be_add2_locked",
            "add_max": 2,
            "add_mult": [0.5, 1.0],
            "add_min_r": [1.0, 2.0],
            "require_locked_profit": True,
            "trend_pool_enabled": True,
            "max_unprotected_symbols": 1,
            "max_symbols_after_unlock": 3,
            "set_one_symbol_per_strategy": False,
            "allow_add_position": True,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 1.0,
        },
        {
            "name": "trend_pool_unlock3_be_add2_loose",
            "add_max": 2,
            "add_mult": [0.5, 1.0],
            "add_min_r": [1.0, 2.0],
            "require_locked_profit": False,
            "trend_pool_enabled": True,
            "max_unprotected_symbols": 1,
            "max_symbols_after_unlock": 3,
            "set_one_symbol_per_strategy": False,
            "allow_add_position": True,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 1.0,
        },
        {
            "name": "trend_pool_unlock6_be_add1_locked",
            "add_max": 1,
            "add_mult": [0.5],
            "add_min_r": [1.0],
            "require_locked_profit": True,
            "trend_pool_enabled": True,
            "max_unprotected_symbols": 1,
            "max_symbols_after_unlock": 6,
            "set_one_symbol_per_strategy": False,
            "allow_add_position": True,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 1.0,
        },
    ]
    selected = {x.strip() for x in str(args.scenarios or "").split(",") if x.strip()}
    if selected:
        known = {str(s["name"]) for s in scenario_specs}
        unknown = sorted(selected - known)
        if unknown:
            raise SystemExit(f"Unknown scenario(s): {', '.join(unknown)}")
        scenario_specs = [s for s in scenario_specs if str(s["name"]) in selected]

    for spec in scenario_specs:
        root = work / str(spec["name"])
        _copy_bpc_tree(root)
        cpath = _copy_constitution(root)

        def _mut_exec(doc: dict, spec: Dict[str, Any] = spec) -> None:
            _mut_chop_base(doc)
            _mut_add_profile(
                doc,
                max_add_times=int(spec["add_max"]),
                multipliers=list(spec["add_mult"]),
                min_current_r_by_add=(
                    list(spec["add_min_r"])
                    if spec.get("add_min_r") is not None
                    else None
                ),
            )
            _mut_breakeven(
                doc,
                enabled=bool(spec.get("breakeven_enabled", False)),
                trigger_r=float(spec.get("breakeven_trigger_r", 1.0)),
                lock_level_r=float(spec.get("breakeven_lock_level_r", 0.0)),
                measure=str(spec.get("breakeven_measure", "atr")),
            )

        def _mut_const(doc: dict, spec: Dict[str, Any] = spec) -> None:
            _mut_constitution_common(
                doc,
                require_locked_profit=bool(spec["require_locked_profit"]),
                trend_pool_enabled=bool(spec["trend_pool_enabled"]),
                max_unprotected_symbols=int(spec["max_unprotected_symbols"]),
                max_symbols_after_unlock=int(spec["max_symbols_after_unlock"]),
                set_one_symbol_per_strategy=bool(spec["set_one_symbol_per_strategy"]),
                allow_add_position=bool(spec.get("allow_add_position", True)),
                anchor_symbol=str(spec.get("anchor_symbol", "") or ""),
                require_anchor_first=bool(spec.get("require_anchor_first", False)),
            )

        _patch_execution(root, _mut_exec)
        _patch_constitution(cpath, _mut_const)
        scenarios.append((str(spec["name"]), root, None, cpath))

    rows_out: List[Dict[str, Any]] = []
    for name, root, inj, constitution_yaml in scenarios:
        oj = out_dir / f"event_backtest_{name}.json"
        oc = out_dir / f"event_trades_{name}.csv"
        if bool(args.resume_existing) and oj.is_file():
            logger.info("reuse existing: %s", oj)
            rc = 0
        else:
            rc = _run_event_backtest(
                repo=_REPO,
                strategies_root=root,
                start_date=start_date,
                end_date=end_date,
                symbols_csv=",".join(symbols),
                data_path=args.data_path,
                out_json=oj,
                out_csv=oc,
                cap_dir=oj.parent,
                no_kill_switch=bool(args.no_kill_switch),
                inject_ml=inj,
                fee_rate=float(args.fee_rate),
                constitution_yaml=constitution_yaml,
            )
        m = _read_metrics(oj)
        spec = next((s for s in scenario_specs if str(s["name"]) == name), {})
        m.update({"scenario": name, "exit_code": rc, "config": dict(spec)})
        rows_out.append(m)
        print(json.dumps(m, indent=2, ensure_ascii=False))

    summary_path = out_dir / "add_regime_study_summary.json"
    summary_path.write_text(
        json.dumps(rows_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Add Regime Study Report",
        "",
        f"- Window: `{start_date}` → `{end_date}`",
        f"- Symbols: `{','.join(symbols)}`",
        "",
        "## Scenarios",
    ]
    for r in rows_out:
        lines.append(
            f"- `{r.get('scenario')}`: total_r={r.get('total_r'):.4f}, "
            f"max_drawdown_r={r.get('max_drawdown_r'):.4f}, n_trades={r.get('n_trades')}, "
            f"reject_kill_switch={r.get('reject_kill_switch')}, add_count={r.get('add_count')}, "
            f"reject_add_locked_profit_required={r.get('reject_add_locked_profit_required')}, "
            f"trend_pool_rejects="
            f"{int(r.get('reject_pcm_trend_pool_anchor_first', 0) or 0) + int(r.get('reject_pcm_trend_pool_unprotected_cap', 0) or 0) + int(r.get('reject_pcm_trend_pool_post_unlock_cap', 0) or 0)}"
        )
    lines.extend(["", "## Ranking: MaxDD First"])
    for i, r in enumerate(
        sorted(
            rows_out,
            key=lambda x: (
                float(x.get("max_drawdown_r", 0.0) or 0.0),
                -float(x.get("total_r", 0.0) or 0.0),
            ),
        ),
        start=1,
    ):
        lines.append(
            f"{i}. `{r.get('scenario')}`: max_drawdown_r={r.get('max_drawdown_r'):.4f}, "
            f"total_r={r.get('total_r'):.4f}, add_count={r.get('add_count')}, "
            f"trend_pool_rejects="
            f"{int(r.get('reject_pcm_trend_pool_anchor_first', 0) or 0) + int(r.get('reject_pcm_trend_pool_unprotected_cap', 0) or 0) + int(r.get('reject_pcm_trend_pool_post_unlock_cap', 0) or 0)}"
        )
    lines.extend(["", "## Ranking: TotalR First"])
    for i, r in enumerate(
        sorted(
            rows_out,
            key=lambda x: (
                -float(x.get("total_r", 0.0) or 0.0),
                float(x.get("max_drawdown_r", 0.0) or 0.0),
            ),
        ),
        start=1,
    ):
        lines.append(
            f"{i}. `{r.get('scenario')}`: total_r={r.get('total_r'):.4f}, "
            f"max_drawdown_r={r.get('max_drawdown_r'):.4f}, add_count={r.get('add_count')}, "
            f"max_notional_frac={r.get('max_observed_notional_frac'):.4f}"
        )
    rep = out_dir / "report.md"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote report → %s", rep)
    print(f"\nWrote summary → {summary_path}")
    return 0 if all(r.get("exit_code", 1) == 0 for r in rows_out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
