#!/usr/bin/env python3
"""
Apply tree-gate veto rules based on archetype.

Inputs:
  - logs parquet/csv (symbol, timestamp, ...)
  - FeatureStore (root + layer) for the same timeframe/window
  - execution_archetypes.yaml gate_rules (per archetype)
  - optional evidence_quantiles.json (for quantile-based rules)

Output:
  - gated file with added columns:
      gate_ok, gate_decision, gate_reasons, gate_archetype

Note: Physical features (path_efficiency_pct, jump_risk_pct, etc.) are loaded
directly from FeatureStore. Regime classification has been migrated to gate rules.
"""
from __future__ import annotations

import argparse
import json
import os
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


def _is_mean_regime_suitable_for_fr_et(
    row: pd.Series, features: Optional[pd.DataFrame] = None
) -> bool:
    """
    Check if the current regime is suitable for FR/ET mean reversion.

    Additional checks beyond basic MEAN_REGIME classification.
    Uses physical features only (no pred_dir_prob):
    - path_efficiency较低（低效路径，适合mean reversion）
    - 价格偏离均值的程度（z-score）
    - 波动率压缩指标
    - 趋势强度应该较低

    Args:
        row: Row data with regime and physical features
        features: Optional features DataFrame for percentile calculations

    Returns:
        bool: True if suitable for FR/ET, False otherwise
    """
    # Check if regime is MEAN_REGIME or suitable for mean reversion
    regime = str(row.get("regime", "")).upper()
    regime_normalized = regime.replace("_REGIME", "") if "_REGIME" in regime else regime

    # If not MEAN regime, check if it has mean reversion characteristics
    if regime_normalized != "MEAN":
        # For non-MEAN regimes, apply stricter filters
        # Only allow FR/ET if path_efficiency is very low (choppy/mean-reverting)
        path_eff_pct = row.get("path_efficiency_pct")
        if pd.notna(path_eff_pct):
            # Only allow if path_efficiency is in bottom 30% (very inefficient paths)
            if path_eff_pct > 0.3:
                return False

    # Check path_efficiency (lower is better for mean reversion)
    path_eff_pct = row.get("path_efficiency_pct")
    if pd.notna(path_eff_pct):
        # Prefer low path efficiency (bottom 40%)
        if path_eff_pct > 0.4:
            return False

    # Check price deviation (higher is better for mean reversion)
    deviation_z_abs_pct = row.get("deviation_z_abs_pct")
    if pd.notna(deviation_z_abs_pct):
        # Need significant deviation (top 60%)
        if deviation_z_abs_pct < 0.4:
            return False

    # Check price direction consistency (lower is better for mean reversion)
    price_dir_consistency_pct = row.get("price_dir_consistency_pct")
    if pd.notna(price_dir_consistency_pct):
        # Prefer unstable direction (bottom 50%)
        if price_dir_consistency_pct > 0.5:
            return False

    # Check ATR percentile (higher volatility is better for mean reversion)
    atr_percentile = row.get("atr_percentile")
    if pd.notna(atr_percentile):
        # Need some volatility (at least 50th percentile)
        if atr_percentile < 0.5:
            return False

    # Check jump_risk (lower is better for mean reversion)
    jump_risk_pct = row.get("jump_risk_pct")
    if pd.notna(jump_risk_pct):
        # Avoid extreme jump risk (top 20%)
        if jump_risk_pct > 0.8:
            return False

    return True


def _compute_archetype_score(row: pd.Series, arch_name: str) -> float:
    """
    Compute archetype selection score based on mfe, mae, ttm, and archetype-specific semantic score.

    Formula: score = (eff * mfe * time_penalty) * semantic_bonus
    where:
    - eff = mfe / (mae + eps) - efficiency ratio (higher is better)
    - mfe - maximum favorable excursion (higher is better)
    - time_penalty = 1.0 / (1.0 + ttm / 10.0) - prefer faster MFE (lower ttm is better)
    - semantic_bonus - archetype-specific semantic score (higher is better)

    Args:
        row: Row data with mfe/mae/ttm and semantic scores
        arch_name: Archetype name (e.g., "TrendContinuationTC", "TrendExpansionTE")

    Returns:
        float: Score for archetype selection (higher is better)
    """
    mfe = float(row.get("head_mfe_atr", 0.0) or 0.0)
    mae = float(row.get("head_mae_atr", 0.0) or 0.0)
    ttm = float(row.get("head_t_to_mfe", 0.0) or 0.0)

    # Avoid division by zero
    eps = 1e-6
    if mae < eps:
        return 0.0

    # Efficiency ratio: mfe / mae (higher is better)
    eff = mfe / (mae + eps)

    # Time penalty: prefer faster MFE (lower ttm is better)
    # Normalize ttm by dividing by 10 to make it comparable to eff
    time_penalty = 1.0 / (1.0 + ttm / 10.0)

    # Base score: efficiency * mfe magnitude * time bonus
    base_score = eff * mfe * time_penalty

    # Archetype-specific semantic score bonus
    # Map archetype name to semantic score column
    arch_upper = str(arch_name).upper()
    semantic_col = None
    if "TC" in arch_upper or "TRENDCONTINUATION" in arch_upper:
        semantic_col = "tc_semantic_score"
    elif "TE" in arch_upper or "TRENDEXPANSION" in arch_upper:
        semantic_col = "te_semantic_score"
    elif "FR" in arch_upper or "FAILUREREVERSION" in arch_upper:
        semantic_col = "fr_semantic_score"
    elif "ET" in arch_upper or "EXHAUSTIONTURN" in arch_upper:
        semantic_col = "et_semantic_score"
    elif "VOLMEAN" in arch_upper or "COMPRESSION" in arch_upper:
        # VolMeanCompressionExpansionReversion is a MEAN regime archetype
        # Use fr_semantic_score as fallback (both are MEAN regime)
        semantic_col = "fr_semantic_score"

    semantic_bonus = 1.0
    if semantic_col and semantic_col in row.index:
        semantic_val = float(row.get(semantic_col, 0.0) or 0.0)
        # Normalize semantic score to [0.5, 1.5] range to provide meaningful bonus
        # semantic scores are typically in [0, 1], so we scale them
        semantic_bonus = 0.5 + semantic_val  # Maps [0, 1] to [0.5, 1.5]

    # Combined score with semantic bonus
    score = base_score * semantic_bonus

    return score


def _enabled_archetypes(*, db_path: str, archetypes: Dict[str, object]) -> List[str]:
    cfg = load_meta_router_live_config(db_path=db_path)
    xs = cfg.enabled_archetypes or []
    return [x for x in xs if x in archetypes]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Apply tree gate based on regime and archetype."
    )
    p.add_argument(
        "--logs",
        required=True,
        help="logs file (must contain symbol, timestamp)",
    )
    p.add_argument("--out", required=True, help="output gated file")
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
        "--db-path",
        default=os.getenv("MLBOT_ORDER_MANAGEMENT_DB_PATH", "data/order_management.db"),
        help="Order management DB path (live_config stored here)",
    )
    p.add_argument("--evidence-quantiles", default=None)
    p.add_argument(
        "--semantic-score-floors",
        default=None,
        help="[DEPRECATED] Optional JSON with tc/te semantic score thresholds (tc_p95 ceiling, te_p10 floor).",
    )
    p.add_argument(
        "--disable-regime-filter",
        action="store_true",
        help="Disable regime filtering: allow all regimes (including NO_TRADE) and all archetypes as candidates.",
    )
    p.add_argument(
        "--disable-gate-veto",
        action="store_true",
        help="Disable gate_rules veto (deny_if/allow_if), but keep semantic score veto and archetype selection.",
    )
    p.add_argument(
        "--disable-semantic-veto",
        action="store_true",
        help="Disable semantic score veto (tc/te semantic score thresholds), but keep gate_rules veto and archetype selection.",
    )
    args = p.parse_args()

    logs_df = _ensure_timestamp_col(_read_any(Path(args.logs)))
    if "symbol" not in logs_df.columns or "timestamp" not in logs_df.columns:
        raise KeyError("logs file must include symbol and timestamp columns")
    # Regime column is optional (not used by gate rules, but kept for diagnostics)
    if "regime" not in logs_df.columns:
        logs_df["regime"] = "NO_TRADE"  # Default for diagnostics only

    symbols = (
        [s.strip() for s in str(args.symbols).split(",") if s.strip()]
        if args.symbols
        else sorted(logs_df["symbol"].astype(str).unique().tolist())
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
    logs_df = logs_df.copy()
    logs_df["symbol"] = logs_df["symbol"].astype(str)
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"], errors="coerce")

    merged = logs_df.merge(
        feats, on=["symbol", "timestamp"], how="left", suffixes=("", "_feat")
    )

    # Physical features (path_efficiency_pct, jump_risk_pct, etc.) are now loaded directly from FeatureStore
    # No need to merge from physics_regime file - regime classification has been migrated to gate rules

    # Optional: semantic score floors
    semantic_floors = None
    if args.semantic_score_floors:
        with open(args.semantic_score_floors, "r") as f:
            semantic_floors = json.load(f)
        # Expected keys: tc_semantic_score_p95 (ceiling), te_semantic_score_p10 (floor)

    arches = load_execution_archetypes_registry(str(args.execution_archetypes))
    quantiles_raw = load_evidence_quantiles(args.evidence_quantiles)

    gate_ok: List[bool] = []
    gate_decision: List[str] = []
    gate_reasons: List[str] = []
    gate_arch: List[str] = []
    # For parallel archetype trading: store row data for each archetype
    parallel_rows: List[Dict] = (
        []
    )  # List of {row_data, gate_ok, gate_decision, gate_reasons, gate_arch}

    for _, row in merged.iterrows():
        regime = str(row.get("regime") or "NO_TRADE").upper()

        # Normalize regime name (TC_REGIME -> TC, TE_REGIME -> TE, etc.)
        regime_normalized = regime
        if "_REGIME" in regime:
            regime_normalized = regime.replace("_REGIME", "")

        # Regime filtering logic (can be disabled for experiment)
        # When disabled, completely ignore regime column and allow all regimes
        use_parallel = args.disable_regime_filter and args.disable_gate_veto

        if not args.disable_regime_filter:
            if regime_normalized == "NO_TRADE" or regime_normalized == "NONE":
                if not use_parallel:
                    gate_ok.append(True)
                    gate_decision.append("no_trade")
                    gate_reasons.append("")
                    gate_arch.append("")
                continue

        # Semantic score veto (Gate-only, can be disabled for experiment)
        # ⚠️ IMPORTANT: Based on E2E analysis:
        # - TC_REGIME: Low scores (0-0.127) have highest Sharpe (5.811), high scores (>0.321) are negative
        #   → Veto HIGH scores (p95), keep LOW sweet spot
        # - TE_REGIME: High scores (0.308-0.761) have highest Sharpe (6.215)
        #   → Veto LOW scores (p10), keep HIGH signals
        if (
            not args.disable_semantic_veto
            and semantic_floors
            and regime_normalized in ("TC", "TE")
        ):
            if regime_normalized == "TC":
                # Veto high-score toxic zone, keep low-score sweet spot
                ceiling = semantic_floors.get("tc_semantic_score_p95")
                score = row.get("tc_semantic_score")
                if (
                    ceiling is not None
                    and pd.notna(score)
                    and float(score) > float(ceiling)
                ):
                    if not use_parallel:
                        gate_ok.append(False)
                        gate_decision.append("veto")
                        gate_reasons.append("tc_semantic_ceiling")
                        gate_arch.append("semantic_ceiling")
                    continue
            if regime_normalized == "TE":
                # Veto low-score noise, keep high-score signals
                floor = semantic_floors.get("te_semantic_score_p10")
                score = row.get("te_semantic_score")
                if (
                    floor is not None
                    and pd.notna(score)
                    and float(score) < float(floor)
                ):
                    if not use_parallel:
                        gate_ok.append(False)
                        gate_decision.append("veto")
                        gate_reasons.append("te_semantic_floor")
                        gate_arch.append("semantic_floor")
                    continue

        # Archetype candidate selection (can be disabled for experiment)
        # When regime filter is disabled, ignore regime column completely and allow all archetypes
        if args.disable_regime_filter:
            # Allow all archetypes as candidates when regime filter is disabled
            candidates = list(arches.keys())
        else:
            # Map TE/TC to TREND for enabled_archetypes lookup
            # ET_REGIME maps to MEAN (for archetype selection, but ET has its own regime)
            # Note: When regime filter is enabled, we still use regime for archetype selection
            candidates = _enabled_archetypes(
                db_path=str(args.db_path),
                archetypes=arches,
            )
            if not candidates:
                if not use_parallel:
                    gate_ok.append(True)
                    gate_decision.append("no_archetype")
                    gate_reasons.append("")
                    gate_arch.append("")
                continue

            # If regime is TE/TC/ET, prioritize matching archetype
            # TE -> TrendExpansionTE, TC -> TrendContinuationTC, ET -> ExhaustionTurnET
            if regime_normalized == "ET":
                prioritized = ["ExhaustionTurnET"] + [
                    c for c in candidates if c != "ExhaustionTurnET"
                ]
                candidates = prioritized
            elif regime_normalized == "TE":
                prioritized = ["TrendExpansionTE"] + [
                    c for c in candidates if c != "TrendExpansionTE"
                ]
                candidates = prioritized
            elif regime_normalized == "TC":
                prioritized = ["TrendContinuationTC"] + [
                    c for c in candidates if c != "TrendContinuationTC"
                ]
                candidates = prioritized

        quantiles = None
        if isinstance(quantiles_raw, dict):
            sym_q = quantiles_raw.get(str(row.get("symbol")))
            quantiles = sym_q if isinstance(sym_q, dict) else quantiles_raw

        # Collect all passing candidates with their scores
        passing_candidates: List[tuple[str, float, List[str]]] = (
            []
        )  # (arch_name, score, reasons)
        last_reasons: List[str] = []

        for arch_name in candidates:
            arch = arches.get(arch_name)
            if not arch:
                continue
            if not arch.gate_rules:
                # If no gate_rules, always pass - compute score
                score = _compute_archetype_score(row, arch_name)
                passing_candidates.append((arch_name, score, []))
                continue

            # Gate veto logic (can be disabled for experiment)
            if args.disable_gate_veto:
                # Skip gate_rules check, directly allow
                ok = True
                reasons = []
            else:
                ok, reasons = apply_gate_rules(
                    gate_rules=arch.gate_rules,
                    features=row.to_dict(),
                    quantiles=quantiles,
                )

            if ok:
                # Compute score for this passing archetype
                score = _compute_archetype_score(row, arch_name)
                passing_candidates.append((arch_name, score, []))
            else:
                last_reasons = list(reasons or [])

        # Parallel archetype trading: when both regime filter and gate veto are disabled,
        # allow all passing archetypes to trade in parallel
        use_parallel = args.disable_regime_filter and args.disable_gate_veto

        if use_parallel and passing_candidates:
            # Create a row for each passing archetype
            for arch_name, score, _ in passing_candidates:
                parallel_rows.append(
                    {
                        "row_data": row.to_dict(),
                        "gate_ok": True,
                        "gate_decision": "allow",
                        "gate_reasons": "",
                        "gate_arch": arch_name,
                    }
                )
            # Skip adding to regular lists when using parallel mode
            continue
        else:
            # Original logic: select the best archetype based on score
            chosen = None
            if passing_candidates:
                # Sort by score (descending) and select the best
                passing_candidates.sort(key=lambda x: x[1], reverse=True)
                chosen, best_score, _ = passing_candidates[0]
                # If multiple candidates have the same score, prefer the first one in original order
                # (This maintains backward compatibility when scores are equal)

            if chosen:
                gate_ok.append(True)
                gate_decision.append("allow")
                gate_reasons.append("")
                gate_arch.append(chosen)
            else:
                gate_ok.append(False)
                gate_decision.append("veto")
                gate_reasons.append(
                    ";".join(last_reasons or ["gate_all_candidates_veto"])
                )
                gate_arch.append(candidates[0] if candidates else "")

    # Handle parallel archetype trading: create multiple rows for each timestamp
    if parallel_rows:
        # Create DataFrame from parallel rows
        parallel_data = []
        for item in parallel_rows:
            row_dict = item["row_data"].copy()
            row_dict["gate_ok"] = item["gate_ok"]
            row_dict["gate_decision"] = item["gate_decision"]
            row_dict["gate_reasons"] = item["gate_reasons"]
            row_dict["gate_archetype"] = item["gate_arch"]
            parallel_data.append(row_dict)

        # Convert to DataFrame
        parallel_df = pd.DataFrame(parallel_data)
        # Merge with original logs_df to ensure all columns are preserved
        # Get the original rows that were NOT expanded (those that went through normal flow)
        if len(gate_ok) > 0:
            normal_rows = logs_df.copy()
            normal_rows["gate_ok"] = gate_ok
            normal_rows["gate_decision"] = gate_decision
            normal_rows["gate_reasons"] = gate_reasons
            normal_rows["gate_archetype"] = gate_arch

            # Identify which rows were expanded by checking timestamp+symbol
            expanded_keys = set(zip(parallel_df["symbol"], parallel_df["timestamp"]))
            normal_mask = ~normal_rows.apply(
                lambda r: (r["symbol"], r["timestamp"]) in expanded_keys, axis=1
            )
            normal_rows_filtered = normal_rows[normal_mask]
            out = pd.concat([normal_rows_filtered, parallel_df], ignore_index=True)
        else:
            # All rows were expanded to parallel
            out = parallel_df.copy()
    else:
        # Original logic: single archetype per row
        out = logs_df.copy()
        out["gate_ok"] = gate_ok
        out["gate_decision"] = gate_decision
        out["gate_reasons"] = gate_reasons
        out["gate_archetype"] = gate_arch

    # Update mode column based on gate_ok and gate_archetype
    # This is critical for diagnose_e2e_kpi.py which uses mode column to count trades
    # If gate_ok=True and gate_archetype is set, infer mode from archetype
    # If gate_ok=False or gate_archetype is empty, set mode to NO_TRADE
    def _infer_mode_from_gate(row):
        gate_ok_val = row.get("gate_ok", False)
        gate_arch_val = str(row.get("gate_archetype") or "").strip()

        if not gate_ok_val or not gate_arch_val:
            return "NO_TRADE"

        # Infer mode from archetype
        arch_upper = gate_arch_val.upper()
        if "TC" in arch_upper or "TE" in arch_upper:
            return "TREND"
        elif "FR" in arch_upper or "ET" in arch_upper:
            return "MEAN"
        else:
            return "NO_TRADE"

    # Update mode column for rows where gate decision was made
    out["mode"] = out.apply(_infer_mode_from_gate, axis=1)

    # Set regime to NO_TRADE for vetoed rows
    veto = ~out["gate_ok"].astype(bool)
    out.loc[veto, "regime"] = "NO_TRADE"

    # Recompute ret_mean for ET archetype using ET-specific config
    # This is needed because ret_mean was computed with MEAN config, but ET needs its own config
    et_mask = out["gate_ok"].astype(bool) & out["gate_archetype"].astype(
        str
    ).str.contains("ET", case=False, na=False)
    if et_mask.any() and "ret_mean" in out.columns:
        try:
            from src.time_series_model.rl.execution_returns_rr import (
                RRExecutionReturnsConfig,
                compute_rr_execution_mode_returns,
            )

            print(
                f"🔄 Recomputing ret_mean for {et_mask.sum()} ET samples with ET-specific config..."
            )
            et_samples = out[et_mask].copy()

            # Required columns for RR execution
            required_cols = [
                "symbol",
                "timestamp",
                "high",
                "low",
                "close",
                "head_dir_score",
                "head_mfe_atr",
                "head_mae_atr",
            ]
            missing_cols = [c for c in required_cols if c not in et_samples.columns]
            if missing_cols:
                print(
                    f"⚠️  Missing columns for ET ret_mean recomputation: {missing_cols}. Skipping."
                )
            else:
                # Use ET-specific config
                et_cfg = RRExecutionReturnsConfig(
                    et_use_time_exit=True,
                    et_use_trailing_stop=True,
                    et_trailing_atr_mult=2.0,
                    et_take_profit_r=1.5,
                    et_stop_loss_r=1.5,
                    et_use_breakeven_stop=False,
                )

                # Recompute ret_mean for ET samples
                ret_mean_et, _ = compute_rr_execution_mode_returns(
                    et_samples,
                    cfg=et_cfg,
                    archetype_col="gate_archetype",
                )

                # Update ret_mean for ET samples
                out.loc[et_mask, "ret_mean"] = ret_mean_et.values
                print(f"✅ Updated ret_mean for {et_mask.sum()} ET samples")
        except Exception as e:
            print(
                f"⚠️  Failed to recompute ret_mean for ET: {e}. Using original ret_mean."
            )

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
