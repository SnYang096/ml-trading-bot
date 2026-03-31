# Archetype-Based Architecture (2026-01)

## Overview

The trading system has been migrated from a regime-based architecture to a pure archetype-based architecture. This document describes the changes and the new architecture.

## Key Changes

### 1. Regime Filtering Removed

**Previous Architecture:**
- Regime classification (TC_REGIME, TE_REGIME, MEAN_REGIME, ET_REGIME, NO_TRADE) was used to filter which archetypes could be candidates
- `meta_router_live_config.yaml` defined `enabled_archetypes` per regime
- `apply_archetype_gate.py` used regime to select archetype candidates

**New Architecture:**
- All archetypes are now candidates for every sample
- Archetype selection is based solely on gate rules and evidence rules
- Regime column is kept for backward compatibility but is not used for filtering

### 2. Conflict Rules Removed

**Previous Behavior:**
- `_get_archetype_conflict_rules()` defined which archetypes should be closed when a new one appeared
- ET appearing would automatically close TC/TE positions
- FR/TC were mutually exclusive

**New Behavior:**
- No automatic position closing based on archetype conflicts
- Slot management only checks basic compatibility (TC+TE compatible, FR+ET compatible)
- Rotation logic simplified: only replaces weakest slot based on ppath, no archetype compatibility checks

### 3. Multiple Archetypes Through Gate

**Previous Behavior:**
- If multiple archetypes passed gate, compatibility was checked
- Compatible archetypes (e.g., TC+TE) could trade in parallel
- Incompatible archetypes would select the highest-scoring one

**New Behavior:**
- If multiple archetypes pass gate, the trade is **directly rejected** (no position opened)
- This ensures unambiguous archetype selection
- Only single archetype passing gate results in a trade

### 4. PCM Simplification

**Changes in `src/time_series_model/portfolio/pcm.py`:**
- Removed `_get_archetype_conflict_rules()` function
- Removed conflict-based position closing logic
- Removed archetype compatibility checks in rotation logic
- Simplified slot2 opening: only checks basic compatibility, no mutual exclusion rules

## Implementation Details

### Gate Logic (`scripts/apply_archetype_gate.py`)

```python
# All archetypes are candidates (regime filter removed)
candidates = list(arches.keys())

# If multiple archetypes pass gate, reject
if len(passing_candidates) > 1:
    gate_ok.append(False)
    gate_decision.append("veto")
    gate_reasons.append(f"multiple_archetypes_passed:{arch_names}")
    continue
```

### PCM Logic (`src/time_series_model/portfolio/pcm.py`)

```python
# Simplified: only basic compatibility check for slot management
# No conflict rules, no automatic position closing
if not _are_archetypes_compatible(candidate_arch, slot_arch):
    # Deny entry if incompatible and no free slot
    if len(slots) >= int(policy.max_slots):
        return PCMDecision(allow_entry=False, ...)
```

## Backward Compatibility

### Deprecated Flags

- `--disable-regime-filter`: Marked as deprecated, has no effect (all archetypes are always candidates)
- `--live-config`: Marked as deprecated, `enabled_archetypes` is no longer used

### Configuration Files

- `config/nnmultihead/live/meta_router_live_config.yaml`: `enabled_archetypes` section marked as `[LEGACY]` but kept for reference

## Benefits

1. **Simpler Logic**: Removed complex conflict resolution and regime-based filtering
2. **Clearer Selection**: Single archetype per trade, no ambiguity
3. **More Flexible**: All archetypes evaluated on every sample, gate rules determine selection
4. **Easier to Debug**: No hidden regime-based filtering, all logic in gate/evidence rules

## Migration Notes

- Regime classification still runs (for backward compatibility and diagnostics)
- Regime column is still present in logs but not used for filtering
- Gate rules and evidence rules now have full control over archetype selection
- PCM slot management simplified but still respects basic compatibility (TC+TE, FR+ET)
