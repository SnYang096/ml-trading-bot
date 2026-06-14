# DECISION — TPC trend_pool_guard sweep (2026-06-12)

## Live 部署（2026-06-14）

**决策**：`live/highcap` TPC `trend_pool_guard` 由 **G0 1/2 → G3 3/6**（`max_unprotected_symbols: 3`, `max_symbols_after_unlock: 6`）。

**依据（interim）**：canonical **bear_2022** G3 **18.11R / -19.5%** vs G0 **3.73R / -16.8%**（+14.4R）；当前 regime 偏 bear，先放宽并发抓信号。

**风险**：canonical 9/9 未跑完；smoke 显示 G3≈G2、G1 DD 更浅。canonical 完成后若 recent/bull 反转结论，可回滚或改 G1。

**配置**：`live/highcap/config/constitution/constitution.yaml` + `config/constitution/constitution.yaml`（研究镜像）

---

## 预结论（2026-06-14 smoke + 分析，canonical 待跑完）

**假设**：生产 `1/2` 过严，改 `3/6`（G3）可多抓并发信号、提升 total R。

**已跑**：6 币 × 2024 全年 smoke（`tpc_pool_guard_smoke_grid.yaml`），策略树 = 当前 prod `config/strategies`。

| Variant | unprot / total | total_r | maxDD | trades | vs G0 ΔR |
|---------|----------------|--------:|------:|-------:|---------:|
| **G0 prod** | 1 / 2 | **10.29** | **-6.9%** | 93 | — |
| **G1** | 1 / 3 | **21.37** | **-6.9%** | 96 | **+11.1** |
| **G2** | 3 / 3 | 22.69 | -10.4% | 143 | +12.4 |
| **G3** | 3 / 6 | 22.69 | -10.4% | 143 | +12.4 |
| G4 off | — | 16.20 | -23.5% | 212 | +5.9 |

### 读数

1. **相对 prod 1/2，放宽并发能加 R** — smoke 上 G1/G2/G3 均显著高于 G0。
2. **G3 = G2（完全相同）** — 2024 从未需要第 4 个 symbol，`max_symbols_after_unlock: 6` **未产生边际**；你关心的「3 裸仓 → BE 后扩到 6」在 smoke 窗里等价于 **G2 的 3/3**。
3. **G1（1/3）可能是更优折中** — 相对 G0 **+11.1R** 且 **maxDD 不变**（仍 -6.9%）；仍保留「先 1 裸仓、BE 后解锁」的 prod 纪律，只把总 cap 从 2 提到 3。
4. **G2/G3 的代价是 DD** — 相对 G0，maxDD 从 -6.9% 加深到 -10.4%（约 +50% 相对回撤）；trade 数 +54%。
5. **G4 证伪「无 guard」** — R 不如 G2/G3，DD 崩到 -23.5%。

### 对「3/6」倾向的判定（smoke 级别，非 promote）

| 问题 | smoke 答案 |
|------|------------|
| 3/6 比 prod 1/2 更好吗？ | **R 上是的**（+12.4R），**DD 更差**（-10.4% vs -6.9%） |
| 3/6 比 1/3（G1）更好吗？ | **否** — G3 仅 +1.3R，DD 多 -3.5pp |
| 「6」cap 有贡献吗？ | **2024 smoke 无** — 与 G2 相同 |
| 能否 promote？ | **不能** — 缺 canonical 三阶段 + bear 段 |

### 待跑（canonical）

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260612_tpc_trend_pool_guard_sweep/tpc_pool_guard_canonical_grid.yaml \
  --quiet-signal-logs

python3 scripts/research/summarize_pool_guard_grid.py results/tpc/experiments/pool_guard_20260612/canonical
```

Grid：`tpc_pool_guard_canonical_grid.yaml`（G0 vs G3 × bear/bull/recent，6 币）。

**阻塞**：全窗 2022→2026 grid 在 XRPUSDT 加载时报 `tz-naive vs tz-aware`（见 `results/tpc/experiments/pool_guard_20260612/full_run.log`）；canonical 分段窗可规避。

---

## 正式表（canonical 跑完后填）

| Variant | bear R | bull R | recent R | **sum R** | **worst maxDD** | unprot_reject | post_unlock_reject |
|---------|--------|--------|----------|-----------|-----------------|---------------|-------------------|
| G0_prod_1_2 | | | | | | | |
| G3_be3_6 | | | | | | | |

## 结论（draft）

- [ ] 维持 prod **1/2**
- [ ] 改为 **1/3**（G1）— smoke 提示 R↑ DD≈
- [ ] 改为 **3/3**（G2）
- [ ] 改为 **3/6**（G3）
- [ ] 需要新代码：显式 `base_symbols + protected_count` 动态 cap

## 理由

（canonical 完成后：DD vs R 权衡、reject 漏斗、与 correlation_guard 交互）
