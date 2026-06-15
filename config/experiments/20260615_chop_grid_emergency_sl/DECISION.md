# DECISION — chop_grid emergency SL

**日期**：2026-06-15  
**结论**：**不开启**任何 per-leg SL / entry-% emergency SL；维持 prod `execution.yaml` 现状。

## 假设

| ID | 假设 | 证据 | 决策 |
|----|------|------|------|
| H1 | spacing × mult `per_leg_stop_loss` 可在不显著伤收益的前提下提供兜底 | Phase A：sl_4x 四段 +7.1% vs baseline +58.8%，`grid_sl` 21%；sl_8x 仍少 ~20pp | **reject** |
| H2 | entry-% `emergency_stop_loss`（-12%/-15%/-20%）在 canonical 四段有保护价值且假止损 <5% | Phase B：四段 `emergency_sl` 触发 **0%**；em_15/em_20 与 baseline 完全一致 | **reject** |
| H3 | 极端窗口（bear_2022 / LUNA / FTX）下单腿会跌破 -12%，emergency SL 有实际作用 | Phase C：三窗口 `emergency_sl` 仍 **0%**；单腿最差 ~-7.8%；`risk_exit`+`regime_exit` 先触发 | **reject** |

## Prod 配置（锁定）

```yaml
# live/highcap/config/strategies/chop_grid/archetypes/execution.yaml
risk:
  per_leg_stop_loss: false
  max_loss_per_grid: 0.03
  force_exit_on_regime_loss: true
  # 不添加 emergency_stop_loss
```

## 结果路径

| Phase | 汇总 |
|-------|------|
| A spacing×mult | `results/chop_grid/experiments/emergency_sl_20260615/QUICK_SUMMARY.md` |
| B entry-% | `results/chop_grid/experiments/emergency_sl_entry_pct_20260615/QUICK_SUMMARY.md` |
| C stress | `results/chop_grid/experiments/emergency_sl_stress_20260615/QUICK_SUMMARY.md` |

## Promote

- [x] 实验完成，决策：**不 promote** emergency SL 到 prod
- [ ] `monitor_bundle/smoke_report.json` — N/A（无配置变更）
- [ ] `mlbot research promote-baseline` — N/A
