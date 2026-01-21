#!/usr/bin/env python3
"""
Apply tree-gate veto rules on top of rule-router mode_3action outputs.

Inputs:
  - mode_3action parquet/csv (symbol, timestamp, mode, mode_action, ...)
  - FeatureStore (root + layer) for the same timeframe/window
  - execution_archetypes.yaml gate_rules (per archetype)
  - live meta_router config (to select a single archetype per regime)
  - optional evidence_quantiles.json (for quantile-based rules)

Output:
  - gated mode file with added columns:
      gate_ok, gate_decision, gate_reasons, gate_archetype
    and mode/mode_action overridden to NO_TRADE on veto.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store import FeatureStore, FeatureStoreSpec  # noqa: E402
from src.time_series_model.core.constitution.execution_evidence import (  # noqa: E402
    load_evidence_quantiles,
)
from src.time_series_model.live.meta_router_config import (  # noqa: E402
    load_meta_router_live_config,
)
from src.time_series_model.live.tree_gate import apply_gate_rules  # noqa: E402
from src.time_series_model.nnmultihead.strategy_profile import (  # noqa: E402
    load_execution_archetypes_registry,
)


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _ensure_timestamp_col(df: pd.DataFrame, *, col: str = "timestamp") -> pd.DataFrame:
    if col in df.columns:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out[col] = out.index
        return out
    return df


def _read_feature_store_range(
    *,
    features_store_root: str,
    layer: str,
    symbols: List[str],
    timeframe: str,
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    store = FeatureStore(str(features_store_root))
    parts = []
    for sym in symbols:
        spec = FeatureStoreSpec(
            layer=str(layer), symbol=str(sym), timeframe=str(timeframe)
        )
        start_ts = pd.Timestamp(start) if start else pd.Timestamp("1970-01-01")
        end_ts = pd.Timestamp(end) if end else pd.Timestamp("2100-01-01")
        df_sym = store.read_range(spec, start=start_ts, end=end_ts)
        if df_sym.empty:
            raise ValueError(f"Empty FeatureStore read for symbol={sym}, layer={layer}")
        if "symbol" not in df_sym.columns:
            df_sym = df_sym.copy()
            df_sym["symbol"] = sym
        parts.append(df_sym)
    df = pd.concat(parts, axis=0, ignore_index=False)
    if "timestamp" not in df.columns:
        if getattr(df.index, "name", None) == "timestamp":
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df.index, utc=False, errors="coerce")
        else:
            raise KeyError(
                "Expected FeatureStore data to have a 'timestamp' column or index named 'timestamp'"
            )
    return df


def _enabled_archetypes(
    *, live_cfg_path: str, regime: str, archetypes: Dict[str, object]
) -> List[str]:
    cfg = load_meta_router_live_config(live_cfg_path)
    rr = str(regime).upper()
    xs = cfg.enabled_archetypes.get(rr) or []
    return [x for x in xs if x in archetypes]


def main() -> int:
    p = argparse.ArgumentParser(description="Apply tree gate on mode_3action.")
    p.add_argument("--mode", required=True, help="mode_3action file")
    p.add_argument("--out", required=True, help="output gated mode file")
    p.add_argument("--features-store-root", default="feature_store")
    p.add_argument("--features-store-layer", required=True)
    p.add_argument("--symbols", default=None, help="Comma-separated symbols")
    p.add_argument("--timeframe", default="240T")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
    )
    p.add_argument(
        "--live-config",
        default="config/nnmultihead/live/meta_router_live_config.yaml",
        help="Use enabled_archetypes to select per-regime archetype",
    )
    p.add_argument("--evidence-quantiles", default=None)
    p.add_argument(
        "--physics-regime",
        default=None,
        help="Optional physics_regime parquet to merge (adds tc/te semantic scores).",
    )
    p.add_argument(
        "--semantic-score-floors",
        default=None,
        help="Optional JSON with tc/te semantic score floors (p05/p10).",
    )
    args = p.parse_args()

    mode_df = _ensure_timestamp_col(_read_any(Path(args.mode)))
    if "symbol" not in mode_df.columns or "timestamp" not in mode_df.columns:
        raise KeyError("mode_3action must include symbol and timestamp columns")

    symbols = (
        [s.strip() for s in str(args.symbols).split(",") if s.strip()]
        if args.symbols
        else sorted(mode_df["symbol"].astype(str).unique().tolist())
    )
    feats = _read_feature_store_range(
        features_store_root=str(args.features_store_root),
        layer=str(args.features_store_layer),
        symbols=symbols,
        timeframe=str(args.timeframe),
        start=args.start_date,
        end=args.end_date,
    )
    feats = feats.copy()
    # If timestamp is both index and column, drop index to avoid merge ambiguity.
    if getattr(feats.index, "name", None) == "timestamp":
        feats = feats.reset_index(drop=True)
    feats["symbol"] = feats["symbol"].astype(str)
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], errors="coerce")
    mode_df = mode_df.copy()
    mode_df["symbol"] = mode_df["symbol"].astype(str)
    mode_df["timestamp"] = pd.to_datetime(mode_df["timestamp"], errors="coerce")

    merged = mode_df.merge(
        feats, on=["symbol", "timestamp"], how="left", suffixes=("", "_feat")
    )

    # Optional: merge physics_regime scores
    if args.physics_regime:
        pw_df = _ensure_timestamp_col(_read_any(Path(args.physics_regime)))
        if "symbol" not in pw_df.columns or "timestamp" not in pw_df.columns:
            raise KeyError("physics_regime must include symbol and timestamp columns")
        pw_df = pw_df.copy()
        pw_df["symbol"] = pw_df["symbol"].astype(str)
        pw_df["timestamp"] = pd.to_datetime(pw_df["timestamp"], errors="coerce")
        keep_cols = [
            "symbol",
            "timestamp",
            "tc_semantic_score",
            "te_semantic_score",
        ]
        keep_cols = [c for c in keep_cols if c in pw_df.columns]
        merged = merged.merge(pw_df[keep_cols], on=["symbol", "timestamp"], how="left")

    # Optional: semantic score floors
    semantic_floors = None
    if args.semantic_score_floors:
        with open(args.semantic_score_floors, "r") as f:
            semantic_floors = json.load(f)
        # Expected keys: tc_semantic_score_p05, te_semantic_score_p10

    arches = load_execution_archetypes_registry(str(args.execution_archetypes))
    quantiles_raw = load_evidence_quantiles(args.evidence_quantiles)

    gate_ok: List[bool] = []
    gate_decision: List[str] = []
    gate_reasons: List[str] = []
    gate_arch: List[str] = []

    for _, row in merged.iterrows():
        mode = str(row.get("mode") or "NO_TRADE").upper()
        regime = str(
            row.get("regime") or mode
        ).upper()  # Use regime if available, fallback to mode
        if mode == "NO_TRADE":
            gate_ok.append(True)
            gate_decision.append("no_trade")
            gate_reasons.append("")
            gate_arch.append("")
            continue
        # Semantic score floor veto (Gate-only)
        if semantic_floors and regime in ("TC", "TE"):
            if regime == "TC":
                floor = semantic_floors.get("tc_semantic_score_p05")
                score = row.get("tc_semantic_score")
                if (
                    floor is not None
                    and pd.notna(score)
                    and float(score) < float(floor)
                ):
                    gate_ok.append(False)
                    gate_decision.append("veto")
                    gate_reasons.append("tc_semantic_floor")
                    gate_arch.append("semantic_floor")
                    continue
            if regime == "TE":
                floor = semantic_floors.get("te_semantic_score_p10")
                score = row.get("te_semantic_score")
                if (
                    floor is not None
                    and pd.notna(score)
                    and float(score) < float(floor)
                ):
                    gate_ok.append(False)
                    gate_decision.append("veto")
                    gate_reasons.append("te_semantic_floor")
                    gate_arch.append("semantic_floor")
                    continue

        # Map TE/TC to TREND for enabled_archetypes lookup
        regime_for_lookup = "TREND" if regime in ("TE", "TC") else regime
        candidates = _enabled_archetypes(
            live_cfg_path=str(args.live_config),
            regime=regime_for_lookup,
            archetypes=arches,
        )
        if not candidates:
            gate_ok.append(True)
            gate_decision.append("no_archetype")
            gate_reasons.append("")
            gate_arch.append("")
            continue

        # If regime is TE/TC, prioritize matching archetype
        # TE -> TrendExpansionTE, TC -> TrendContinuationTC
        if regime == "TE":
            prioritized = ["TrendExpansionTE"] + [
                c for c in candidates if c != "TrendExpansionTE"
            ]
            candidates = prioritized
        elif regime == "TC":
            prioritized = ["TrendContinuationTC"] + [
                c for c in candidates if c != "TrendContinuationTC"
            ]
            candidates = prioritized

        quantiles = None
        if isinstance(quantiles_raw, dict):
            sym_q = quantiles_raw.get(str(row.get("symbol")))
            quantiles = sym_q if isinstance(sym_q, dict) else quantiles_raw

        chosen = None
        last_reasons: List[str] = []
        for arch_name in candidates:
            arch = arches.get(arch_name)
            if not arch:
                continue
            if not arch.gate_rules:
                chosen = arch_name
                last_reasons = []
                break
            ok, reasons = apply_gate_rules(
                gate_rules=arch.gate_rules,
                features=row.to_dict(),
                quantiles=quantiles,
            )
            if ok:
                chosen = arch_name
                last_reasons = []
                break
            last_reasons = list(reasons or [])

        if chosen:
            gate_ok.append(True)
            gate_decision.append("allow")
            gate_reasons.append("")
            gate_arch.append(chosen)
        else:
            gate_ok.append(False)
            gate_decision.append("veto")
            gate_reasons.append(";".join(last_reasons or ["gate_all_candidates_veto"]))
            gate_arch.append(candidates[0] if candidates else "")

    out = mode_df.copy()
    out["gate_ok"] = gate_ok
    out["gate_decision"] = gate_decision
    out["gate_reasons"] = gate_reasons
    out["gate_archetype"] = gate_arch
    veto = ~out["gate_ok"].astype(bool)
    out.loc[veto, "mode"] = "NO_TRADE"
    if "mode_action" in out.columns:
        out.loc[veto, "mode_action"] = 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        out.to_parquet(out_path, index=False)
    else:
        out.to_csv(out_path, index=False)

    print("✅ Saved gated mode:", out_path)
    print(
        "   gate_decisions:",
        json.dumps(out["gate_decision"].value_counts().to_dict(), ensure_ascii=False),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
