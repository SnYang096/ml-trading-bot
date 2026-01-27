# Gate When-Then Execution Order

This document defines the **single, unified rule schema** for gate evaluation and the **exact execution order** used by runtime code. It ensures configuration and logic remain consistent.

## Rule Schema (Unified)
Each gate rule is a **when-then** rule with optional metadata:

- `id`: Unique rule identifier
- `phase`: One of `safety`, `exclusions`, `preconditions`, `evidence`, `decision`
- `priority`: Integer; lower runs earlier within the same phase
- `reason`: Human-readable explanation (e.g., `强语义`, `regime`, `看起来像趋势`)
- `when`: Condition block (supports `all_of`, `any_of`, `min_matches`, `not`)
- `then.action`: One of `deny`, `require`, `allow`

Example:

```yaml
- id: fr_looks_trendy
  phase: preconditions
  priority: 2
  reason: "看起来像趋势，后面更容易获得高RR反转"
  when:
    all_of:
      - path_efficiency_pct:
          quantile_gte: 0.4
      - price_dir_consistency_pct:
          quantile_gte: 0.5
  then:
    action: require
```

## Condition Syntax
Leaf conditions use the same operators as evidence rules:

- `value_gt`, `value_gte`, `value_lt`, `value_lte`
- `quantile_gt`, `quantile_gte`, `quantile_lt`, `quantile_lte`
- `any_key_contains`

Compound conditions:

- `all_of: [ ... ]`
- `any_of: [ ... ]`
- `min_matches: N` (only used with `any_of`)
- `not: { ... }`

## Execution Order (Fixed)
Rules execute in this **fixed phase order**:

```
safety → exclusions → preconditions → evidence → decision
```

### Phase Semantics
- **safety**: hard veto; any matched `deny` stops evaluation.
- **exclusions**: hard veto for obvious non-viable regimes.
- **preconditions**: required structural conditions; any missing `require` fails.
- **evidence**: confirmation signals; any missing `require` fails.
- **decision**: final allow/deny. If no explicit allow, `default_action` applies.

## Pseudocode (Runtime Contract)

```python
def evaluate_when_then(archetype, features, quantiles):
    rules = archetype.when_then_rules

    for rule in phase("safety"):
        if match(rule.when):
            if rule.then.action == "deny":
                return False

    for rule in phase("exclusions"):
        if match(rule.when):
            if rule.then.action == "deny":
                return False

    for rule in phase("preconditions"):
        if rule.then.action == "require" and not match(rule.when):
            return False

    for rule in phase("evidence"):
        if rule.then.action == "require" and not match(rule.when):
            return False

    return default_action != "deny"
```

## Notes
- `priority` only affects ordering **within** the same phase.
- `reason` is for readability and debugging, not logic.
- `default_action` must be explicit per archetype (recommended: `deny` for FR/ET, `allow` for TC/TE).
