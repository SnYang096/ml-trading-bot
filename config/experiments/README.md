# config/experiments — 按实验计划归档

每个**完整实验**一个目录：`config/experiments/<YYYYMMDD>_<strategy>_<topic>/`

| 内容 | 说明 |
|------|------|
| `README.md` | **实验卡片**：物料清单、跑法、`results/` 路径、链到决策文档（不写长表） |
| `DECISION.md` / `*_experiment_*.md` | **决策全文**：假设表、变体/回测结果、promote 结论（原 `docs/decisions/` 已迁入） |
| `rd_loop_*.yaml` | offline 扫描编排（mlbot research + 可选 variant_grid + **`tree_steps`** 树通道） |
| `*_grid.yaml` | event_backtest 变体网格 |

### `tree_steps`（树通道 fast_scalp 等）

| mode | 说明 |
|------|------|
| `prepare-only` | `train_strategy_pipeline.py --prepare-only` → features_labeled.parquet |
| `ic-prune` | `mlbot research ic-prune`（内核 [`src/research/stat_kernels/ic_prune.py`](../../src/research/stat_kernels/ic_prune.py)） |
| `train` | 浅树训练（同 mlbot train final） |
| `tau-scan` | `scripts/research/tree_holdout_tau_rr_scan.py` |
| `filter-predictions` | 按 symbol/split 切 predictions.parquet |

示例：`20260529_fast_scalp/`（Phase 1）、`20260530_fast_scalp_alts_majors/`（历史 alpha rebuild）、**`20260602_fast_scalp_tree_validate/`**（树模型两轨验证，[`TRAINING.md`](20260602_fast_scalp_tree_validate/TRAINING.md)）。

**变体策略树：** 放在本目录下 `variants/<topic>_strategies/`（仅含本实验相关策略，已剔除无关兄弟目录）。历史根目录 [`config_experiments/`](../config_experiments/) 已弃用，见 [`CONFIG_EXPERIMENTS_MIGRATION.md`](CONFIG_EXPERIMENTS_MIGRATION.md)。

## 标准 R&D 流程（新特征 / 新周期 — 必读）

完整 promote 准则见 [`LAYER_PROMOTION_CRITERIA.md`](LAYER_PROMOTION_CRITERIA.md)。**核心顺序**：

```text
Phase 0  特征可算（parquet）
    ↓
Phase 1  IC + label scan / 历史窗宽扫描  ← 证明「有用」+ 候选 τ / lookback
    ↓
Phase 2  人读扫描 → DECISION 定参 → 本实验 `variants/` 静态树
    ↓
Phase 3  segment_matrix variant-grid（因果 R）
    ↓
Phase 4  trading map 人审（语义对齐）
    ↓
Phase 5  三条杠 promote → prod YAML locked
```

| Phase | 命令 / 文件 | 说明 |
|-------|-------------|------|
| 0 | `train_strategy_pipeline.py --prepare-only` | `features_labeled.parquet` 含新列 |
| 1a | **`mlbot research scan`** + **`rd_loop.py --hypothesis-yaml …/rd_loop_*.yaml`** | `feature-plateau` / `condition-set` / `pair-scan` / `gate-plateau` / `ic-decay`（**默认 Phase 1 入口**） |
| 1b | `mlbot research ic` / `plateau` / `segment` | IC@H、lift/snotio 精标、分桶 stratify |
| 1c | `mlbot train final --prepare-only` × N | **lookback / binding 参数**改特征计算窗宽时，各 L 各跑一份 parquet，再对每份跑 1a |
| 1d | `scripts/research/scan_*.py` | **例外**：仅当 1a+1c 成本过高且扫的是 **OHLC 重算 binding**（如 TPC `scan_tpc_pullback_lookback.py`）；**禁止**为 B/C 规则栈新写 `scan_<strategy>_*.py` |
| 2 | `DECISION.md` 假设表 + τ | **禁止手拍窗宽进 grid** |
| 3 | `python -m scripts.event_backtest --variant-grid …/*_grid.yaml` | canonical 三阶段；yaml 加 `trading_map: true` 则同跑分段地图 |
| 4 | 同上 grid 或 `*_bull_maps_grid.yaml` | **优先分段地图**（与 R 同窗）；全窗 map 仅当必须看跨段连续 K 线 |
| 5 | [`LAYER_PROMOTION_CRITERIA.md`](LAYER_PROMOTION_CRITERIA.md) | Total R ↑、maxDD 不恶化、可解释 |

**反模式（不得当 promote 依据）**：

- 跳过 Phase 1 直接 `*_grid.yaml`（窗宽/阈值来自直觉）
- **AI/人写新 `scripts/research/scan_*.py`** 代替 `mlbot research scan` + `rd_loop`（已有命令族）
- IC/label 显著但未经 Phase 3 三阶段回测
- 多尺度特征未对齐（例：box 突破 @120 + soft_phase @240）且无扫描文档

**Phase 1 命令族（复制即用）**：

```bash
# 单条 scan（等价于 rd_loop 里一条 quick_layer_scans）
mlbot research scan feature-plateau --strategy bpc --layer prefilter \
  --parquet results/train_final/bpc/<run>/bpc/features_labeled.parquet \
  --label success_no_rr_extreme \
  --feature bpc_pullback_depth --operator '<=' \
  --grid '0.35,0.45,0.50,0.55,0.65' \
  --filter 'tpc_semantic_chop<=0.40' \
  --output results/rd_loop/bpc_box_pullback/quick_scan/depth_plateau.md

# 批量编排（推荐）
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260611_bpc_lookback_retest_validate/rd_loop_bpc_box_pullback_phase1.yaml
```

完整子命令表：[`docs/strategy/R&D工具矩阵_CN.md`](../docs/strategy/R&D工具矩阵_CN.md) §1 · [`docs/完整命令速查表.md`](../docs/完整命令速查表.md) §research。

> **跨 Layer 决策准则（2026-06 新增）**：  
> 所有 gate / entry_filters / prefilter / regime / direction 等规则的最终 promote，必须遵守 [`LAYER_PROMOTION_CRITERIA.md`](LAYER_PROMOTION_CRITERIA.md) 里的“三条杠”：  
> **在 canonical 三个市场阶段上，总 R 明显提升 + maxDD 不恶化 + 逻辑可解释** 才允许写入生产 YAML 并 `locked: true`。  
> IC/label scan 仅用于生成假设。TPC gate 系列是本准则的第一次完整落地。  
> **树模型职责（2026-06）**：树嵌进 B/C 做 **排序/否决**，不替代规则语义；见 [`docs/strategy/短期树独立策略_设计与落地_CN.md`](../docs/strategy/短期树独立策略_设计与落地_CN.md) §1.4。  
> **Promote 后平台基线（远程 drift）**：同文件 **§4** + [`docs/strategy/漂移监控_mlbot_monitor_CN.md`](../docs/strategy/漂移监控_mlbot_monitor_CN.md) §10（git 提交 monitoring JSON/plateaus；**不上传** train_final parquet）。

**跨实验校准：** [`_cross/regime_thresholds/`](_cross/regime_thresholds/)（regime τ 季度标定日志）。

**架构迁移（计划）：** 周/月监控、`pre_deploy` 门禁将迁出 `config/strategies/*/research/`，改用 `config/monitoring/` 与实验目录内 manifest；见 [`docs/strategy/配置与监控_manifest迁移计划_CN.md`](../docs/strategy/配置与监控_manifest迁移计划_CN.md)。

**历史链接：** [`docs/decisions/README.md`](../docs/decisions/README.md) 为迁移索引；`docs/strategy/` 内旧 URL 不批量改。

## 跑法

```bash
# 完整 R&D loop（扫描 → 可选 grid → decision doc）
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/<dir>/rd_loop_*.yaml

# 仅因果 backtest
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/<dir>/*_grid.yaml --quiet-signal-logs
```

**Web 浏览**：`mlbot rolling-dashboard`（默认 `:8008`）→ 打开 `/rd`。

## 实验索引

| 目录 | 策略 | 主题 |
|------|------|------|
| [`20260526_chop_grid_semantic_proxy/`](20260526_chop_grid_semantic_proxy/) | chop_grid | 语义代理 baseline |
| [`20260527_bpc_layer_validation/`](20260527_bpc_layer_validation/) | bpc | 分层验证 + ABH gate |
| [`20260527_bpc_entry_v2/`](20260527_bpc_entry_v2/) | bpc | entry v2 |
| [`20260527_bpc_regime_ema/`](20260527_bpc_regime_ema/) | bpc | regime EMA grid |
| [`20260527_srb_entry_plateau/`](20260527_srb_entry_plateau/) | srb | entry plateau scan |
| [`20260527_tpc_regime_slope_signed/`](20260527_tpc_regime_slope_signed/) | tpc | regime slope 分符号 |
| [`20260528_me_compression_breakout/`](20260528_me_compression_breakout/) | me | 压缩突破分层 + no_box |
| [`20260528_me_direction/`](20260528_me_direction/) | me | direction 优化 |
| [`20260528_me_entry_filter/`](20260528_me_entry_filter/) | me | entry_filter / orderflow |
| [`20260528_me_gate_anti/`](20260528_me_gate_anti/) | me | gate anti |
| [`20260528_me_prefilter_v4/`](20260528_me_prefilter_v4/) | me | prefilter v4 漏斗 |
| [`20260528_me_prod_holdout/`](20260528_me_prod_holdout/) | me | prod holdout |
| [`20260528_tpc_me_trading_map/`](20260528_tpc_me_trading_map/) | tpc, me | 交易地图 bull/bear |
| [`20260529_fast_scalp/`](20260529_fast_scalp/) | fast_scalp | IC 剪枝 + pooled 训练（Phase 1） |
| [`20260530_fast_scalp_alts_majors/`](20260530_fast_scalp_alts_majors/) | fast_scalp | 历史 alpha rebuild Phase 0–4（归档） |
| [**`20260602_fast_scalp_tree_validate/`**](20260602_fast_scalp_tree_validate/) | **fast_scalp** | **双 head + exec-aligned/gate 两轨验证**（[`TRAINING.md`](20260602_fast_scalp_tree_validate/TRAINING.md)） |
| [`20260602_trend_scalp_segment_validate/`](20260602_trend_scalp_segment_validate/) | trend_scalp | market_segment 四段 multi-leg 稳定性（prod archetype） |
| [`20260602_chop_grid_segment_validate/`](20260602_chop_grid_segment_validate/) | chop_grid | market_segment 四段 + 等权 portfolio return 口径 |
| [`20260603_chop_grid_oos_tune/`](20260603_chop_grid_oos_tune/) | chop_grid | OOS spacing/regime/box_pos promote |
| [`20260603_chop_grid_exec_align/`](20260603_chop_grid_exec_align/) | chop_grid | 1min exec 与 live 对齐 |
| [`20260603_chop_grid_replenish_ablation/`](20260603_chop_grid_replenish_ablation/) | chop_grid | post-TP replenish 0/1/unlimited |
| [**`20260604_chop_grid_stack_ablation/`**](20260604_chop_grid_stack_ablation/) | **chop_grid** | **入场栈分层 ablation + dense 3L @2bps** |
| [`20260529_short_term_swing_ic_plateau/`](20260529_short_term_swing_ic_plateau/) | short_term_swing | IC plateau |
| [`20260529_tpc_direction_ema_align/`](20260529_tpc_direction_ema_align/) | tpc | direction EMA1200 对齐 + trail |
| [`20260529_tpc_gate_plateau/`](20260529_tpc_gate_plateau/) | tpc | gate plateau |
| [`20260530_tpc_deep_pullback/`](20260530_tpc_deep_pullback/) | tpc | 深回撤 + 吸收（H1–H4） |
| [`_smoke/`](_smoke/) | tpc | CI / 工具 smoke（非正式实验） |

| [`20260531_tpc_gate_validate/`](20260531_tpc_gate_validate/) | tpc | gate ablation Phase 1 |
| [`20260601_1124_tpc_regime_gate_extend/`](20260601_1124_tpc_regime_gate_extend/) | tpc | regime gate extend |
| [`20260601_1125_tpc_gate_validate/`](20260601_1125_tpc_gate_validate/) | tpc | gate G0/G1 by segment (mixed names) |
| [`20260601_1126_tpc_gate_monotonic_validate/`](20260601_1126_tpc_gate_monotonic_validate/) | tpc | monotonic single-sided gate label scan |
| [`20260601_1130_tpc_gate_final_lock/`](20260601_1130_tpc_gate_final_lock/) | tpc | gate final lock attempt (含 G10，YAML 问题中断) |
| [`20260601_1210_short_term_swing_wide_top100/`](20260601_1210_short_term_swing_wide_top100/) | short_term_swing | wide top100 IC + tree |
| [`20260601_1300_tpc_gate_canonical_g0_g1/`](20260601_1300_tpc_gate_canonical_g0_g1/) | tpc | **最终干净 G0 vs G1 判决**（仅 canonical 三阶段，按 LAYER_PROMOTION_CRITERIA.md lock） |
| [`20260604_tpc_entry_semantic_validate/`](20260604_tpc_entry_semantic_validate/) | tpc | **入场语义 S50(depth>0.5) + S51(EMA略下) + E1/E2/E3 + turbo** × canonical + 全窗；笔记 [`TPC语义约束与树标签对齐_CN.md`](../docs/strategy/TPC语义约束与树标签对齐_CN.md) |
| [**`20260610_tpc_macro_pullback_replace/`**](20260610_tpc_macro_pullback_replace/) | **tpc** | macro 替代 depth；**Phase 1 扫描已完成** → Phase 3 grid |
| [**`20260611_bpc_lookback_retest_validate/`**](20260611_bpc_lookback_retest_validate/) | **bpc** | ⚠️ Phase 3 超前；Phase 1 → `rd_loop_bpc_box_pullback_phase1.yaml`（mlbot scan） |
| [`20260601_1322_tree_forward_rr_ic_small_pool/`](20260601_1322_tree_forward_rr_ic_small_pool/) | fast_scalp, short_term_swing | label vs forward_rr IC + small pool comparison |

## 新建实验 checklist

1. `mkdir config/experiments/<YYYYMMDD>_<strategy>_<topic>/`
2. **先** `rd_loop_*.yaml` 编排 **`mlbot research scan`**（Phase 1）；**不要**新写 `scripts/research/scan_*.py`（TPC lookback OHLC 例外见 `LAYER_PROMOTION_CRITERIA.md`）
3. `DECISION.md` 记录 Phase 2 定参（τ、lookback、多尺度对齐说明）
4. **再** `*_grid.yaml`（Phase 3）；`variant_grid` 用**项目根相对路径**
5. 变体策略树：`config/experiments/<dir>/variants/<topic>_strategies/`（只保留本策略子目录 + `_shared` / `feature_dependencies.yaml`）
6. `run_trading_maps.sh`（Phase 4，可选）
7. 在本表追加一行索引；README 标明当前 **Phase 0–5 进度**
