# TPC Scale Sweep — 2026-06-17

## 假设

| 变体             | 假设                                              | 改动                                      |
| ---------------- | ------------------------------------------------- | ----------------------------------------- |
| **E0_prod**      | Baseline                                          | 无                                        |
| **A1_chop_030**  | chop 0.4 太严，放宽到 0.3 允许更多 mild chop 入场 | gate.yaml: chop >0.3 deny                 |
| **A2_chop_050**  | chop 0.4 太松，收紧到 0.5 减少噪声                | gate.yaml: chop >0.5 deny                 |
| **B1_box240**    | 20天级箱体应过滤（240-bar window）                | gate.yaml: +box_stability_240≥0.85 deny   |
| **B2_box480**    | 40天级箱体应过滤（480-bar window，更保守）        | gate.yaml: +box_stability_480≥0.85 deny   |
| **C1_bull_long** | Bull 段空单 -4.90R 拖后腿，应禁空                 | regime.yaml: side_mask short_when ema<0.1 |
| **D1_add_stop**  | 加仓独立止损优于继承母仓                          | execution.yaml: inherit_parent_stop=false |

## Promote 准则

1. Σ R 明显提升（≥+10% vs E0）
2. maxDD 不恶化
3. 逻辑可解释 + regime-aware

## 运行

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260617_tpc_scale_sweep/tpc_scale_sweep_grid.yaml \
  --quiet-signal-logs
```

## 状态

- [ ] Phase 3 回测
- [ ] 结果分析
- [ ] DECISION
