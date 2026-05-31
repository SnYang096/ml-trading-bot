# 策略层条件语法统一计划（prefilter / regime / gate / entry）

## 现状：不是四套解析器，但能力不一致

| 层 | 配置路径 | 评估入口 | 语法能力 |
|----|----------|----------|----------|
| **regime** | `archetypes/regime.yaml` → `rules[]` | `RegimeConfig.evaluate()` | `feature`+`operator`+`value`（条间 **AND**）；`any_of` 子规则组（**OR**） |
| **prefilter** | `archetypes/prefilter.yaml` → `rules[]` | `PrefilterConfig.evaluate()` | **与 regime 同实现**（`RegimeConfig` 子类复用） |
| **gate** | `archetypes/gate.yaml` → `hard_gates[].when` | `_evaluate_when_clause()` | `when` DSL：`{feat: {value_gt: …}}`、`all_of`、`any_of`、`min_matches` |
| **entry** | `archetypes/entry_filters.yaml` → `filters[].conditions` | `entry_filter._check_single` / `_build_mask_from_conditions` | `feature`+`operator`+`value`（filter 内 **AND**）；filter 间 `combination_mode` OR/AND |

**结论**：regime 与 prefilter **共用** `PrefilterConfig` 解析器；**gate 单独**用 `_evaluate_when_clause`；**entry 单独**用 entry_filter 模块。文档 `docs/strategy/regime_layer.md` 已写 regime/prefilter 同 schema，但 **未** 包含 gate 的 `all_of`/`when` 嵌套。

因此 E1「`ema_1200_position>0.10` 且 `pullback_depth>=0.55`」在 gate 里一行 `all_of` 即可，在 prefilter 里 **无法表达**（只能扁平 AND 或 `any_of` 单特征，不能 `(ema∧depth)∨(ema∧depth_bear)`）。

## 目标语法（各层统一子集）

建议 **canonical DSL** = 现有 gate `when` 子集（已在 `loader._evaluate_when_clause`）：

```yaml
# 单条规则 / 单个 gate.when / 可嵌入 prefilter.rules[]
when:
  all_of:
    - ema_1200_position: { value_gt: 0.10 }
    - tpc_pullback_depth: { value_gte: 0.55 }
  # 或 any_of / min_matches
```

**entry** 保持 `conditions[]` + `combination_mode`（语义是「扳机组合」，与 veto 型 gate 略不同），但 `conditions` 内可逐步支持 `all_of` 嵌套（可选 Phase 3）。

## 分阶段实施

### Phase A — 文档与校验（1 PR）

- [ ] 在 `docs/strategy/regime_layer.md` 标明：prefilter/regime **当前** schema vs **目标** schema。
- [ ] `strategy_validation.py`：对 prefilter/regime 的 `rules` 增加可选 `when:` 键检测（与扁平 `feature` 二选一）。
- [ ] 实验目录 `_variant_snippets/` 注释指向本计划。

### Phase B — prefilter/regime 复用 `_evaluate_when_clause`（1 PR，推荐优先）

- [ ] `PrefilterConfig.evaluate()`：若 rule 含 `when:`，调用 `_evaluate_when_clause(rule["when"], features)`；否则走现有 `_check_single` / `any_of`（**向后兼容**）。
- [ ] 向量路径：`backtest_execution_layer._apply_prefilter_vectorized` 对齐（若存在）。
- [ ] 单测：`tests/` — `all_of` ema+depth；缺失特征 `on_missing` 与 gate 一致。

### Phase C — regime-conditional depth（配置 + smoke）

- [ ] `prefilter.yaml` 示例（TPC）：

```yaml
rules:
  - feature: tpc_pullback_depth
    operator: <=
    value: 0.85
  - when:
      any_of:
        - all_of:
            - ema_1200_position: { value_gte: 0.10 }
            - tpc_pullback_depth: { value_gte: 0.55 }
        - all_of:
            - ema_1200_position: { value_lte: -0.10 }
            - tpc_pullback_depth: { value_gte: 0.35 }
```

- [ ] event_backtest smoke vs 当前 prod prefilter + E2_or。

### Phase D — entry 嵌套（可选）

- [ ] `entry_filter` 的 `conditions` 支持 `all_of` 块，或文档明确 entry 只用扁平 conditions。

### Phase E — 统一命名

- [ ] gate 的 `value_gt` 与 prefilter 的 `operator: '>='` 在文档中列对照表；长期可考虑 YAML 锚点/生成器，**不**强制一层改 operator 字符串。

## 非目标

- 不合并 gate 与 prefilter 的 **语义阶段**（仍 regime → prefilter → gate → entry）。
- 不在本计划内改 rd_loop / quick_scan 写回逻辑（Phase C 后再接）。

## 与 TPC deep_pullback 实验的关系

- **已 promote**：prod prefilter（depth 上限）+ **E2_or entry**（见 `config/strategies/tpc/archetypes/entry_filters.yaml`）。
- **未 promote**：E1 静态 depth 下界、E3 PE gate — 待 Phase B/C 后再用统一语法做 regime-conditional depth 实验。
