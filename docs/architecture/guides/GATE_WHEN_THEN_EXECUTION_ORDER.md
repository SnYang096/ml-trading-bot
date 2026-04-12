# Gate When-Then Execution Order

## 与代码对齐（必读）

仓库里存在 **两条** Gate 求值路径，本文件前半部分描述的是 **`tree_gate`** 的 when-then 引擎；**实盘 `GenericLiveStrategy` 默认不走这条五段式逻辑**。

| 路径 | 入口 | 行为摘要 |
|------|------|----------|
| **A. 五段式 when-then** | `src/time_series_model/live/tree_gate.py` → `apply_gate_rules` → `apply_when_then_rules` | 使用固定 phase 顺序、`require` / `allow` / `default_action` 语义；被 `apply_archetype_gate.py`、若干诊断/对比脚本等调用。 |
| **B. 实盘策略（loader）** | `src/time_series_model/archetype/loader.py` → `StrategyArchetype.apply_gate` | 按 `GateConfig.all_rules` 排序遍历（**`system_safety` → `hard_gate` → `guardrail`**，同 phase 内再按 `priority`）；**仅当 `when` 命中且 `then.action == "deny"` 时立即拒绝**；不实现下文的 `preconditions` / `evidence` / `decision` / `require` 失败语义。 |

若你要让**文档中的五段式**与某条链路严格一致，请在设计评审里明确：实盘是否应改为调用 `apply_gate_rules(archetype.gate_rules, …)`，或继续以 B 为准并单独维护 B 的说明。

---

## Rule Schema (Unified) — `tree_gate` / `when_then_rules`

Each gate rule is a **when-then** rule with optional metadata:

- `id`: Unique rule identifier
- `phase`: Runtime bucket must be one of **`safety`**, **`exclusions`**, **`preconditions`**, **`evidence`**, **`decision`** for `apply_when_then_rules` to place the rule in the intended bucket.  
  **YAML 侧常用别名**：`system_safety` / `hard_gate` / `guardrail`（见 `GateRule`）。在 `apply_when_then_rules` 中，**凡不在上述五段式集合内的 `phase` 会被归并到 `exclusions` 桶**（再与同桶规则一起按 `priority`、`id` 排序），**不会**单独按 `gate.yaml` 里 `schema.evaluation_order` 再拆段执行。
- `priority`: Integer; lower runs earlier **within the same phase bucket**
- `reason`: Human-readable explanation (e.g., `强语义`, `regime`, `看起来像趋势`)
- `when`: Condition block (supports `all_of`, `any_of`, `min_matches`, `not`, list-of-clauses as AND, or legacy `{key: {op: value}}` leaves — implemented in `tree_gate._eval_when_clause`)
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

Leaf conditions are evaluated via `compute_execution_evidence` (same operator vocabulary as evidence-style rules), including:

- `value_gt`, `value_gte`, `value_lt`, `value_lte` (aliases `value_ge` → `value_gte`, `value_le` → `value_lte` in `tree_gate`)
- `quantile_gt`, `quantile_gte`, `quantile_lt`, `quantile_lte`
- `any_key_contains`

Compound conditions:

- `all_of: [ ... ]`
- `any_of: [ ... ]` with optional `min_matches: N` **inside the same dict as `any_of`**
- `not: { ... }`
- `when` as a **YAML list** of clauses: treated as **logical AND** of items

## Execution Order — `apply_when_then_rules` only

Rules execute in this **fixed phase order** (`tree_gate._PHASE_ORDER`):

```
safety → exclusions → preconditions → evidence → decision
```

### Phase Semantics (`tree_gate`)

- **safety / exclusions**: if `then.action == "deny"` and `when` matches → **immediate veto** `(False, reasons)`.
- **preconditions / evidence**: if `then.action == "require"` and `when` does **not** match → **fail** `(False, reasons)`.
- **decision**: `then.action == "allow"` with a matching `when` sets an internal **allow_hit**; after all phases, if `allow_hit` then pass; else fall back to **`default_action`** (`gate_rules.default_action`, defaulting to `"deny"` in `apply_gate_rules` when missing).
- **`then.action == "deny"` in `preconditions` / `evidence` / `decision`** is **not** handled like safety/exclusions (no automatic short-circuit in those phases); prefer placing hard vetoes in `safety` / `exclusions` or use the **loader live path** (B) where any matched `deny` always stops.

## Pseudocode (`apply_when_then_rules`, simplified)

```python
def apply_when_then_rules(when_then_rules, features, quantiles, default_action):
    bucket rules by phase in order: safety, exclusions, preconditions, evidence, decision
    allow_hit = False
    for phase in PHASE_ORDER:
        for rule in sorted(rules_in(phase), by=(priority, id)):
            matched = eval_when(rule.when, features, quantiles)
            action = rule.then.get("action", "").lower()

            if phase in ("safety", "exclusions") and action == "deny" and matched:
                return False, reasons

            if phase in ("preconditions", "evidence") and action == "require" and not matched:
                return False, reasons

            if action == "allow" and matched:
                allow_hit = True

    if allow_hit:
        return True, reasons
    return default_action != "deny", reasons
```

## Notes

- `priority` only affects ordering **within** the same phase **bucket** after normalization.
- `reason` is for readability and debugging; veto messages use `reason` or `id`.
- For **`tree_gate`**, set `default_action` explicitly on `gate_rules` when you rely on `allow_hit` vs default (see `apply_gate_rules` default `"deny"` when key absent).
- **`StrategyArchetype.gate_rules`** exposes a `when_then_rules` list for tooling, but **`apply_gate` (live)** does not call `tree_gate`; do not assume five-phase semantics on live without wiring `apply_gate_rules`.
