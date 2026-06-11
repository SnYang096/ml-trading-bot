# BPC lookback + box-retest 验证（2026-06-11）

## R&D 阶段进度

流程定义见 [`../LAYER_PROMOTION_CRITERIA.md`](../LAYER_PROMOTION_CRITERIA.md) §标准 R&D 阶段。

| Phase | 状态 | 说明 |
|-------|------|------|
| 0 | ✅ | `bpc_soft_phase_f` / `box_breakout_*` 可算 |
| 1 | ✅ | `rd_loop_bpc_box_pullback_phase1.yaml` + [`PHASE1_REPORT.md`](PHASE1_REPORT.md) |
| 2 | ✅ | retest 改 `box_pos_120<=0.85` + depth≥0.12 |
| 3 | ✅ | grid 21/21（2026-06-09，见 [`DECISION.md`](DECISION.md)） |
| 4 | 🔄 | trading map 后台跑（watcher → `run_trading_maps.sh`） |
| 5 | ⏳ | — |

**已知限制（2026-06-11 修复后）**：

- `B_L240`：仅 `bpc_soft_phase_f` @240；`box_structure_f` 仍 @120（README 已注明）。
- retest 规则为静态 `box_pos_120` cap，非时序「曾高 → 已回落」（见 `BPC_SEMANTICS.md` §7）。
- 新增 **`B0_retest`**：prod L20 + Phase1 定参 retest，避免 L120 树套用 L20 扫描阈值。

## Phase 1 跑法（mlbot 命令族）

```bash
# 0) 刷新 prod parquet（含 box_structure_f）
RUN_ID=train_final_$(date +%Y%m%d_%H%M%S)
mlbot train final --no-docker --prepare-only \
  -c config/strategies/bpc -t 240T \
  --symbol BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --start-date 2022-01-01 --end-date 2026-04-01 \
  --output-root results/train_final/bpc/${RUN_ID}

# 0b) lookback L=120/240：各实验树再 prepare（改 binding 无单列可扫）
mlbot train final --no-docker --prepare-only \
  -c config/experiments/20260611_bpc_lookback_retest_validate/variants/bpc_lb120_strategies/bpc -t 240T \
  --symbol BTCUSDT,ETHUSDT,SOLUSDT --start-date 2022-01-01 --end-date 2026-04-01 \
  --output-root results/train_final/bpc/bpc_lb120_${RUN_ID}
# 同理 bpc_lb240_strategies → 更新 rd_loop yaml 里 parquet 路径

# 1) 批量 scan（depth 带 / box N / retest 组合 / IC）
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260611_bpc_lookback_retest_validate/rd_loop_bpc_box_pullback_phase1.yaml
```

产物：`results/rd_loop/bpc_box_pullback_20260611/quick_scan/*.md` → 人读 plateau 后写 `DECISION.md` Phase 2 τ。

| 字段 | 值 |
|------|-----|
| 策略 | bpc |
| Grid | [`bpc_lookback_retest_grid.yaml`](bpc_lookback_retest_grid.yaml) |
| 变体树 | `config_experiments/bpc_lb*_strategies/`（静态 YAML，无 prepare 脚本） |

## 背景

Trading map 复盘：prod BPC 常在 **突破延续段/尖顶追高**，而非「压缩区突破 → 回测 → 再延续」。根因：`lookback_breakout=20`（≈1.7d）+ `depth<=0.55`（浅位/贴顶易过）+ 无 box 硬规则。

背景文档：[B系统入场语义与执行层周期错配_CN.md](../../docs/strategy/B系统入场语义与执行层周期错配_CN.md) §2.2

**语义设计全文**（突破/回撤/延续 vs 反追高，含回测验收清单）：[`BPC_SEMANTICS.md`](BPC_SEMANTICS.md)

## 变体

| ID | lookback | prefilter 额外 |
|----|----------|----------------|
| **B0_prod** | 20 | prod 三锚点 + vol_compression |
| **B0_retest** | 20 | prod lookback + retest（`box_pos_120<=0.85` + `depth>=0.12`）；Phase1 τ 对齐 |
| **B_L120** | 120（~10d） | 同上，仅拉长 soft_phase / bars_since / vol_compression |
| **B_L240** | 240（~20d） | soft_phase@240 only；**box 仍 @120**（见 `bpc_lb240_strategies/README.md`） |
| **B_L120_retest** | 120 | + `box_pos_120<=0.85` + `depth>=0.12`（Phase1 反追高；**非** box_breakout） |

实验树覆盖：`bpc_soft_phase_f.lookback_breakout` / `vol_ma_window` / `node_cache_version` + `bars_since_extreme_f.lookback`。

## Phase 3 跑法（读完 Phase 1 再跑）

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260611_bpc_lookback_retest_validate/bpc_lookback_retest_grid.yaml \
  --quiet-signal-logs
```

全窗 trading map（胜出者，BTC/SOL）：

```bash
bash config/experiments/20260611_bpc_lookback_retest_validate/run_trading_maps.sh
```

## 结果

- 分段：`results/bpc/experiments/lookback_retest_20260611/<variant>/<segment>/`
- 地图：`results/bpc/maps/lookback_retest_20260611/`

## 决策

见 [`DECISION.md`](DECISION.md)。重点看 **bull_2023_2024** trading map 入场是否从「尖顶追高」移到压缩突破后的回测区。
