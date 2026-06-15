# T5 wall entry — DECISION

## Scan 结论（Phase 1c，BTC+ETH）

- **Bull pullback**：`wall_dist≤2` label lift +25pp（|z|=3.1）
- **Bear pullback**：`wall_dist≤2.5` label lift +6.8pp（|z|=2.8）
- **Neutral 死区**：近墙 **负** lift → 不做全局 filter

## 变体

| ID | 设计 |
|----|------|
| W1 | AND：vol_confirm + 多单 wall≤2 |
| W2 | AND：vol + 多≤2 / 空≤2.5 |
| W4 | AND：vol + 对称≤2 对照 |

## Phase 3 结果（2026-06-15，BTC+ETH event_backtest）

| variant | bear_2022 R | bull_2023_2024 R | recent R | Σ R | maxDD 段内最差 | Σ 笔数 |
|---------|-------------|------------------|----------|-----|----------------|--------|
| **E0_prod** | -4.17 | **+4.95** | **+7.13** | **+7.91** | -9.0% | 206 |
| W1_bull_wall2 | -1.50 | -1.44 | +2.25 | -0.68 | -7.2% | 103 |
| W2_asym_wall | 0.00† | -1.01 | +4.54 | +3.54 | -0.2% | 13 |
| W4_sym_wall2 | 0.00† | -1.01 | +1.71 | +0.71 | -1.0% | 6 |

† bear_2022：W2/W4 **0 笔成交**（filter 过严）

### vs E0_prod（Δ Σ R）

- W1：-8.59R（牛市段 -6.39R，笔数 73→22）
- W2：-4.37R（近期段相对最好但仍 -2.59R，笔数极少）
- W4：-7.21R

## 判决

**不 promote。** Phase 1 label lift 未转化为 event_backtest 总 R 增益；三条杠均未过：

1. Σ total_r 全面低于 prod（W1/W4 跨段为负或接近零）
2. 虽部分段 maxDD 收窄，但以 **牺牲几乎全部成交** 为代价（W2/W4 全样本仅 6–13 笔）
3. AND 组合（vol + wall）与 scan 子集条件不等价，牛市段反向最明显

## Promote?

- [x] canonical 三阶段 Total R / maxDD 过关 → **未过**
- [ ] trading map 语义对齐 — 未做（无 promote 必要）

**后续**：若重试，考虑 (a) 保持 prod `combination_mode: or` 仅追加 wall 为可选 tier；(b) regime-conditional（仅 bull pullback 子集）；(c) 先修 `ema_1200_position` pipeline 再复扫。
