# TPC Gate 最终干净判决（G0 vs G1，仅 canonical 三阶段）

**实验 ID**: 20260601_1300_tpc_gate_canonical_g0_g1  
**目的**: 用**最简配置**在 `config/market_segment.yaml` 定义的三个官方真实市场阶段上，完成 TPC gate 层的最终 lock 决策。

这是 TPC gate 系列（0530 ablation → 0531 validate → 0601 extend → 0602 monotonic → 1130 带 G10 尝试）之后的**收官实验**。

## 为什么需要这个“极简干净”版本

- 之前 `20260601_tpc_gate_validate` 运行时 market_segment 里还残留中文 segment id，导致结果目录混杂 `高位震荡/回落` 等历史名称。
- `20260601_1130_tpc_gate_final_lock` 包含了 G10（vol_leverage <0.03 单边）变体，YAML 描述字段未加引号导致崩溃，只跑完 bear_2022 的 G0/G1。
- 本实验**彻底剥离**所有历史 baggage：
  - 只保留 G0（prod 基线）和 G1（两条 bull vol 中间带 disabled）
  - 强制使用当前干净的三个 English segment
  - 输出路径完全干净：`results/tpc/experiments/gate_canonical_final/{G0,G1}/...`

## 核心原则（见同级 LAYER_PROMOTION_CRITERIA.md）

只有在三个阶段上**同时**满足：
- 总 R-multiple 明显提升
- maxDD 不恶化
- 逻辑可解释 + regime-aware

才允许写入生产 `gate.yaml` 并 `locked: true`。

本实验跑完后将**直接按此三条杠**给出最终 promote / delete 建议。

## 物料

- Grid: `tpc_gate_g0_vs_g1_canonical.yaml`
- 策略树（已冻结快照）:
  - G0: `config/experiments/20260601_1300_tpc_gate_canonical_g0_g1/variants/tpc_gate_ablate_G0_prod_strategies`
  - G1: `config/experiments/20260601_1300_tpc_gate_canonical_g0_g1/variants/tpc_gate_ablate_G1_no_bull_vol_strategies`
- 市场阶段定义: `config/market_segment.yaml`（bear_2022 / bull_2023_2024 / recent_range_to_bear）
- 决策准则: `../LAYER_PROMOTION_CRITERIA.md`

## 运行命令

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260601_1300_tpc_gate_canonical_g0_g1/tpc_gate_g0_vs_g1_canonical.yaml \
  --quiet-signal-logs
```

后台推荐（避免阻塞）：
```bash
nohup ... 2>&1 > /tmp/tpc_gate_final_1300.log &
```

## 预期结果位置

- `results/tpc/experiments/gate_canonical_final/G0/bear_2022/` ...
- `results/tpc/experiments/gate_canonical_final/G1/bull_2023_2024/` ...
- （共 3 segments × 2 variants = 6 组完整报告 + trades csv + capital json/html）

## 跑完后的标准动作

1. 汇总三阶段表格（Total R、maxDD、CAGR、胜率、tail contrib、信号漏斗关键数字）。
2. 按 `LAYER_PROMOTION_CRITERIA.md` 给出明确结论：
   - G1 满足三条杠 → 推荐 lock G1 形态 + **物理删除** gate.yaml 里所有剩余 disabled 的 vol_persistence / vol_leverage / EVT 规则（及对应注释）。
   - 否则保留 G0 + 同样清理。
3. 生成 patch 直接应用到：
   - `config/strategies/tpc/archetypes/gate.yaml`
   - `live/highcap/config/strategies/tpc/archetypes/gate.yaml`
4. 更新本实验的 `DECISION.md`（从 stub 变成最终判决书）。
5. 在 `config/experiments/README.md` 索引里标记本实验为 “TPC gate 最终 lock”。

## 历史关联

- 所有先前的 gate ablation 证据一致指向 G1 是最优。
- 本实验只是用**最干净的 canonical 窗口 + 最新 market_segment 定义**把结论最后确认一次，避免任何命名/配置污染。

跑完后即可彻底、干净地把 TPC gate 配置锁死，不再留任何 ambiguous disabled 包袱。

---

## 状态（2026-06-01）

**已完成**。见 [`DECISION.md`](DECISION.md)：G1 promote，三阶段 +5.93R vs G0；prod/live `gate.yaml` 仅保留 chop。
