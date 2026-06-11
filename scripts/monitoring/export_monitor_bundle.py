"""Export / promote monitor_bundle v1 (R&D draft → git baseline).

Called by rd_loop ``monitor_bundle`` step (draft) and
``mlbot research promote-baseline`` (Phase 5).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plateau_stability import PlateauRange
from src.monitoring.regime_health import (
    entry_pass_rate_from_window,
    has_labeled_regime_schema,
    has_multileg_regime_schema,
    multileg_config,
    regime_shares_from_window,
)
from src.research.stat_kernels.drift import series_percentile
from src.research.stat_kernels.ic import rank_ic
from src.research.writeback.plateau_baseline import load_yaml
from src.time_series_model.regime.threshold_calibrator import load_regime_yaml

DEFAULT_PSI_FEATURES = [
    "ema_1200_position",
    "vol_persistence",
    "vol_leverage_asymmetry",
]
IC_TARGET = "forward_rr"
BUNDLE_VERSION = 1


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_rules_hash(regime_yaml: Dict[str, Any]) -> Optional[str]:
    ar = regime_yaml.get("allowed_regimes")
    if not isinstance(ar, dict) or not ar:
        return None
    canonical = json.dumps(ar, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def infer_schema(regime_yaml: Dict[str, Any]) -> str:
    if has_multileg_regime_schema(regime_yaml):
        return "multileg"
    if has_labeled_regime_schema(regime_yaml):
        return "labeled"
    plateaus = (regime_yaml.get("last_calibration") or {}).get("plateaus") or []
    if plateaus:
        return "plateau"
    return "labeled"


def _resolve_regime_yaml(
    strategy: str,
    *,
    strategies_root: Path,
    regime_yaml_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    path = regime_yaml_path or (
        strategies_root / strategy / "archetypes" / "regime.yaml"
    )
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    raw = load_regime_yaml(path)
    if not raw:
        raw = load_yaml(path)
    return path, raw


def _symbols_from_df(df: pd.DataFrame) -> List[str]:
    for col in ("symbol", "_symbol"):
        if col in df.columns:
            return sorted({str(x) for x in df[col].dropna().unique()})
    return []


def _plateau_from_series(
    series: pd.Series,
    *,
    feature: str,
    operator: str,
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> Optional[Dict[str, Any]]:
    start = series_percentile(series, q_low, min_n=20)
    end = series_percentile(series, q_high, min_n=20)
    mid = series_percentile(series, 0.50, min_n=20)
    if start is None or end is None or mid is None:
        return None
    if start > end:
        start, end = end, start
    return {
        "feature": feature,
        "operator": operator,
        "plateau": {"start": start, "end": end, "mid": mid},
    }


def _build_plateaus_from_window(
    df: pd.DataFrame,
    regime_yaml: Dict[str, Any],
    *,
    feature_specs: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    specs = feature_specs or []
    lc = regime_yaml.get("last_calibration") or {}
    for entry in lc.get("plateaus") or []:
        if not isinstance(entry, dict):
            continue
        feat = str(entry.get("feature") or "")
        op = str(entry.get("operator") or ">=")
        if feat and not any(s.get("feature") == feat for s in specs):
            specs.append({"feature": feat, "operator": op})

    out: List[Dict[str, Any]] = []
    for spec in specs:
        feat = str(spec.get("feature") or "")
        op = str(spec.get("operator") or ">=")
        if not feat or feat not in df.columns:
            continue
        row = _plateau_from_series(df[feat], feature=feat, operator=op)
        if row:
            out.append(row)
    return out


def _build_multileg_baseline(
    df: pd.DataFrame,
    strategy: str,
    regime_yaml: Dict[str, Any],
) -> Dict[str, Any]:
    ml = multileg_config(regime_yaml)
    entry_feature = str(ml.get("entry_feature") or "")
    entry_min = float(ml.get("entry_min"))
    rate, median, skip = entry_pass_rate_from_window(
        df,
        entry_feature=entry_feature,
        entry_min=entry_min,
    )
    if skip or rate is None:
        raise ValueError(f"multileg baseline failed for {strategy}: {skip}")
    return {
        strategy: {
            "entry_pass_rate": rate,
            "median_entry_feature": median,
        }
    }


def _ic_rows_for_features(
    df: pd.DataFrame,
    features: List[str],
    *,
    target: str = IC_TARGET,
    min_n: int = 100,
) -> List[Dict[str, Any]]:
    if target not in df.columns:
        return []
    y = pd.to_numeric(df[target], errors="coerce")
    rows: List[Dict[str, Any]] = []
    for feat in features:
        if feat not in df.columns:
            continue
        x = pd.to_numeric(df[feat], errors="coerce")
        rho, p, n = rank_ic(x, y, min_n=min_n)
        if n < min_n or np.isnan(rho):
            continue
        rows.append(
            {
                "feature": feat,
                "bucket": "all",
                "n": n,
                "rank_ic": rho,
                "p_value": p,
            }
        )
    return rows


def _default_psi_features(
    strategy: str, extra: Optional[List[str]] = None
) -> List[str]:
    feats = list(DEFAULT_PSI_FEATURES)
    if strategy == "tpc" and "adx_50" not in feats:
        feats.append("adx_50")
    if extra:
        for f in extra:
            if f and f not in feats:
                feats.append(f)
    return feats


def export_monitor_bundle(
    *,
    strategy: str,
    layer: str,
    parquet: Path,
    out_dir: Path,
    regime_yaml_path: Optional[Path] = None,
    strategies_root: Path | str = "config/strategies",
    psi_features: Optional[List[str]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    run_smoke: bool = True,
) -> Dict[str, Any]:
    """Write draft bundle.json + PSI ref parquet under out_dir."""
    pq = Path(parquet)
    if not pq.is_absolute():
        pq = (PROJECT_ROOT / pq).resolve()
    if not pq.is_file():
        raise FileNotFoundError(f"parquet not found: {pq}")

    root = Path(strategies_root)
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()

    regime_path, regime_yaml = _resolve_regime_yaml(
        strategy, strategies_root=root, regime_yaml_path=regime_yaml_path
    )
    schema = infer_schema(regime_yaml)
    df = pd.read_parquet(pq)

    out = Path(out_dir)
    if not out.is_absolute():
        out = (PROJECT_ROOT / out).resolve()
    ref_dir = out / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    psi_feats = _default_psi_features(strategy, psi_features)
    ic_cols = list(dict.fromkeys(psi_feats + [IC_TARGET]))
    ic_cols = [c for c in ic_cols if c in df.columns]
    ref_name = f"{strategy}_psi_ref.parquet"
    ref_path = ref_dir / ref_name
    df[ic_cols].to_parquet(ref_path, index=False)

    regime_block: Dict[str, Any] = {}
    if schema == "labeled":
        shares = regime_shares_from_window(df, regime_yaml)
        regime_block["regime_shares"] = shares
        rh = compute_rules_hash(regime_yaml)
        if rh:
            regime_block["rules_hash"] = rh
    elif schema == "multileg":
        regime_block["multileg_baseline"] = _build_multileg_baseline(
            df, strategy, regime_yaml
        )
    elif schema == "plateau":
        regime_block["plateaus"] = _build_plateaus_from_window(df, regime_yaml)

    ic_rows = _ic_rows_for_features(df, psi_feats)
    ic_block: Optional[Dict[str, Any]] = None
    if ic_rows:
        ic_block = {"target": IC_TARGET, "rows": ic_rows}

    cal = dict(calibration or {})
    cal.setdefault("parquet", _repo_rel(pq))
    cal.setdefault("n_rows", len(df))
    syms = _symbols_from_df(df)
    if syms:
        cal.setdefault("symbols", syms)

    bundle: Dict[str, Any] = {
        "version": BUNDLE_VERSION,
        "strategy": strategy,
        "layer": layer,
        "schema": schema,
        "calibration": cal,
        "regime": regime_block,
        "psi": {
            "features": psi_feats,
            "reference_parquet": f"monitor_bundle/reference/{ref_name}",
            "sha256": _sha256_file(ref_path),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if ic_block:
        bundle["ic"] = ic_block

    smoke_report: Dict[str, Any] = {"skipped": True}
    if run_smoke:
        smoke_report = run_bundle_smoke(
            strategy=strategy,
            parquet=pq,
            bundle=bundle,
            regime_yaml=regime_yaml,
            psi_ref_path=ref_path,
            psi_features=psi_feats,
        )
    bundle["smoke"] = smoke_report

    out.mkdir(parents=True, exist_ok=True)
    bundle_path = out / "bundle.json"
    bundle_path.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    smoke_path = out / "smoke_report.json"
    smoke_path.write_text(
        json.dumps(smoke_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "bundle_path": bundle_path,
        "bundle": bundle,
        "smoke_report": smoke_report,
        "regime_yaml_path": regime_path,
    }


def run_bundle_smoke(
    *,
    strategy: str,
    parquet: Path,
    bundle: Dict[str, Any],
    regime_yaml: Dict[str, Any],
    psi_ref_path: Path,
    psi_features: List[str],
) -> Dict[str, Any]:
    """Smoke on calibration window: drift + PSI should be OK vs freshly exported baseline."""
    import argparse

    from scripts.regime_drift_monitor import run_drift_monitor
    from scripts.regime_watchdog import run_watchdog

    tmp_baseline = PROJECT_ROOT / "results/monitoring/_bundle_smoke_baseline.json"
    tmp_baseline.parent.mkdir(parents=True, exist_ok=True)

    regime = bundle.get("regime") or {}
    entry: Dict[str, Any] = {
        "source": _repo_rel(parquet),
        "n": bundle.get("calibration", {}).get("n_rows"),
    }
    if regime.get("regime_shares"):
        entry["regime_shares"] = regime["regime_shares"]
    if regime.get("rules_hash"):
        entry["rules_hash"] = regime["rules_hash"]
    if regime.get("multileg_baseline"):
        entry["multileg_baseline"] = regime["multileg_baseline"]

    ic_path = tmp_baseline.parent / f"factor_ic_smoke_{strategy}.json"
    ic_payload: Dict[str, Any] = {
        "target": IC_TARGET,
        "source_parquet": _repo_rel(psi_ref_path),
        "ts": datetime.now(timezone.utc).strftime("%Y%m%d_bundle_smoke"),
        "rows": (bundle.get("ic") or {}).get("rows") or [],
    }
    ic_path.write_text(json.dumps(ic_payload, indent=2) + "\n", encoding="utf-8")

    baseline_doc = {
        "factor_ic_baseline_ref": _repo_rel(ic_path),
        strategy: entry,
    }
    tmp_baseline.write_text(json.dumps(baseline_doc, indent=2) + "\n", encoding="utf-8")

    drift_args = argparse.Namespace(
        strategies=strategy,
        window_parquet=str(parquet),
        strategies_root="config/strategies",
        out_dir="results/monitoring/_bundle_smoke_drift",
        drift_quantile=0.5,
        emit_rd_loop_suggestions=False,
        baseline_json=str(tmp_baseline),
        regime_share_tol=0.10,
    )
    drift_exit = run_drift_monitor(drift_args)

    wd_args = argparse.Namespace(
        strategies=strategy,
        window_parquet=str(parquet),
        strategies_root="config/strategies",
        baseline_json=str(tmp_baseline),
        bull_share_tol=0.10,
        trigger_drift_tol_rel=0.50,
        out_dir="results/monitoring/_bundle_smoke_watchdog",
        ic_baseline_json=str(ic_path),
        psi_features=",".join(psi_features),
        psi_tol=0.25,
        ic_flip_min_abs=0.02,
    )
    watchdog_exit = run_watchdog(wd_args)

    current_hash = compute_rules_hash(regime_yaml)
    baseline_hash = regime.get("rules_hash")
    rules_stale = bool(baseline_hash and current_hash and baseline_hash != current_hash)

    return {
        "drift_exit": drift_exit,
        "watchdog_exit": watchdog_exit,
        "rules_stale": rules_stale,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_regime_calibration_preserve_comments(path: Path, lc: Dict[str, Any]) -> None:
    """Append or replace last_calibration block without rewriting whole regime.yaml."""
    block = yaml.dump(
        {"last_calibration": lc},
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(block, encoding="utf-8")
        return
    text = path.read_text(encoding="utf-8")
    if "last_calibration:" in text:
        lines = text.splitlines()
        out: List[str] = []
        skip = False
        for line in lines:
            if line.startswith("last_calibration:"):
                skip = True
                continue
            if skip:
                if line and not line[0].isspace() and ":" in line:
                    skip = False
                    out.append(line)
                continue
            out.append(line)
        text = "\n".join(out).rstrip()
    path.write_text(text + "\n\n" + block, encoding="utf-8")


def _merge_regime_last_calibration(
    regime_yaml: Dict[str, Any],
    regime_block: Dict[str, Any],
    *,
    parquet_rel: str,
) -> Dict[str, Any]:
    out = copy.deepcopy(regime_yaml)
    lc = out.get("last_calibration")
    if not isinstance(lc, dict):
        lc = {}
        out["last_calibration"] = lc
    lc["timestamp"] = datetime.now(timezone.utc).isoformat()
    lc["data_source"] = parquet_rel
    if regime_block.get("regime_shares"):
        lc["regime_shares"] = regime_block["regime_shares"]
    if regime_block.get("plateaus"):
        lc["plateaus"] = regime_block["plateaus"]
    if regime_block.get("multileg_baseline"):
        mb = lc.get("multileg_baseline")
        if not isinstance(mb, dict):
            mb = {}
            lc["multileg_baseline"] = mb
        mb.update(regime_block["multileg_baseline"])
    if regime_block.get("rules_hash"):
        lc["rules_hash"] = regime_block["rules_hash"]
    return out


def _update_watchdog_baseline_json(
    path: Path,
    *,
    strategy: str,
    entry: Dict[str, Any],
    ic_baseline_ref: str,
) -> None:
    doc: Dict[str, Any] = {}
    if path.is_file():
        doc = json.loads(path.read_text(encoding="utf-8"))
    doc["factor_ic_baseline_ref"] = ic_baseline_ref
    doc[strategy] = entry
    path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _update_strategy_support(*, strategy: str) -> None:
    path = PROJECT_ROOT / "config/monitoring/strategy_support.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ready = list(data.get("pcm_drift_ready") or [])
    if strategy not in ready:
        ready.append(strategy)
    data["pcm_drift_ready"] = sorted(set(ready))
    path.write_text(
        yaml.dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def promote_monitor_bundle(
    bundle: Dict[str, Any],
    *,
    dry_run: bool = False,
    enable_drift_ready: bool = False,
    strategies_root: Path | str = "config/strategies",
    regime_yaml_path: Optional[Path] = None,
    bundle_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Promote bundle.json contents into git-tracked monitoring paths."""
    strategy = str(bundle["strategy"])
    regime_block = bundle.get("regime") or {}
    cal = bundle.get("calibration") or {}
    parquet_rel = str(cal.get("parquet") or "")

    root = Path(strategies_root)
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()

    regime_path, regime_yaml = _resolve_regime_yaml(
        strategy, strategies_root=root, regime_yaml_path=regime_yaml_path
    )

    psi = bundle.get("psi") or {}
    psi_feats = psi.get("features") or DEFAULT_PSI_FEATURES
    ref_rel_in_bundle = str(psi.get("reference_parquet") or "")
    ref_src: Optional[Path] = None
    if bundle_dir and ref_rel_in_bundle:
        ref_src = (
            Path(bundle_dir) / ref_rel_in_bundle.replace("monitor_bundle/", "")
        ).resolve()
        if not ref_src.is_file():
            ref_src = (
                Path(bundle_dir) / "reference" / f"{strategy}_psi_ref.parquet"
            ).resolve()

    git_ref = (
        PROJECT_ROOT / "config/monitoring/reference" / f"{strategy}_psi_ref.parquet"
    )
    ic_json = (
        PROJECT_ROOT
        / "config/monitoring"
        / f"factor_ic_baseline_{strategy}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    )
    watchdog_json = PROJECT_ROOT / "config/monitoring/regime_watchdog_baseline.json"
    live_regime = (
        PROJECT_ROOT
        / "live/highcap/config/strategies"
        / strategy
        / "archetypes"
        / "regime.yaml"
    )

    actions: List[str] = []

    entry: Dict[str, Any] = {
        "source": parquet_rel,
        "n": cal.get("n_rows"),
    }
    if regime_block.get("regime_shares"):
        entry["regime_shares"] = regime_block["regime_shares"]
        bull = regime_block["regime_shares"].get("bull")
        if bull is not None:
            entry["bull_share"] = bull
    if regime_block.get("rules_hash"):
        entry["rules_hash"] = regime_block["rules_hash"]
    if regime_block.get("multileg_baseline"):
        entry["multileg_baseline"] = regime_block["multileg_baseline"]

    ic_payload = {
        "target": IC_TARGET,
        "source_parquet": _repo_rel(git_ref),
        "ts": datetime.now(timezone.utc).strftime("%Y%m%d_promote"),
        "rows": (bundle.get("ic") or {}).get("rows") or [],
    }

    updated_regime = _merge_regime_last_calibration(
        regime_yaml, regime_block, parquet_rel=parquet_rel
    )

    actions.append(f"write {watchdog_json} [{strategy}]")
    actions.append(f"write {git_ref}")
    actions.append(f"write {ic_json}")
    actions.append(f"write {regime_path} last_calibration")
    if live_regime.parent.exists():
        actions.append(f"write {live_regime} last_calibration")
    if enable_drift_ready:
        actions.append(f"append {strategy} to strategy_support.yaml pcm_drift_ready")

    if dry_run:
        return {"dry_run": True, "actions": actions, "bundle": bundle}

    git_ref.parent.mkdir(parents=True, exist_ok=True)
    if ref_src and ref_src.is_file():
        git_ref.write_bytes(ref_src.read_bytes())
    elif not git_ref.is_file():
        raise FileNotFoundError(f"PSI ref missing for promote: {ref_src or git_ref}")

    ic_json.write_text(
        json.dumps(ic_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _update_watchdog_baseline_json(
        watchdog_json,
        strategy=strategy,
        entry=entry,
        ic_baseline_ref=_repo_rel(ic_json),
    )
    lc_block = updated_regime.get("last_calibration") or {}
    _write_regime_calibration_preserve_comments(regime_path, lc_block)
    if live_regime.parent.is_dir():
        _write_regime_calibration_preserve_comments(live_regime, lc_block)
    if enable_drift_ready:
        _update_strategy_support(strategy=strategy)

    bundle["psi"]["promoted_reference_parquet"] = _repo_rel(git_ref)
    bundle["psi"]["sha256"] = _sha256_file(git_ref)

    return {
        "dry_run": False,
        "actions": actions,
        "watchdog_baseline": watchdog_json,
        "psi_reference": git_ref,
        "ic_baseline": ic_json,
        "regime_yaml": regime_path,
    }


def load_bundle(path: Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def export_and_promote_direct(
    *,
    strategy: str,
    layer: str,
    parquet: Path,
    dry_run: bool = False,
    enable_drift_ready: bool = False,
    **export_kw: Any,
) -> Dict[str, Any]:
    """One-shot export to temp dir then promote (TPC migration path)."""
    tmp = PROJECT_ROOT / "results/monitoring/_bundle_promote_tmp" / strategy
    exp = export_monitor_bundle(
        strategy=strategy,
        layer=layer,
        parquet=parquet,
        out_dir=tmp,
        run_smoke=True,
        **export_kw,
    )
    return promote_monitor_bundle(
        exp["bundle"],
        dry_run=dry_run,
        enable_drift_ready=enable_drift_ready,
        bundle_dir=tmp,
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="export_monitor_bundle (dev/debug)")
    p.add_argument("--strategy", required=True)
    p.add_argument("--layer", default="regime")
    p.add_argument("--parquet", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--promote", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    exp = export_monitor_bundle(
        strategy=args.strategy,
        layer=args.layer,
        parquet=Path(args.parquet),
        out_dir=Path(args.out_dir),
    )
    if args.promote:
        promote_monitor_bundle(
            exp["bundle"],
            dry_run=args.dry_run,
            bundle_dir=Path(args.out_dir),
        )
    print(json.dumps({"bundle": str(exp["bundle_path"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
