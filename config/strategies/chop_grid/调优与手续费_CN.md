# Chop Grid 调优与手续费指南

> **目的**：在你本地 2024-01 ~ 2026-05、5 币回测结论基础上，说明还有没有更好的调参方向、以及能否省手续费。  
> **权威配置**：`config/strategies/chop_grid/archetypes/*.yaml` > 本文件 > `README.md` 历史段落。  
> **对齐日期**：2026-06-18（对照 `results/chop_grid/diagnose_2024-01-01_2026-05-31/`）

---

## 1. 你这次回测在说什么？

### 1.1 研究引擎（`chop_grid_backtest.py`）— 应用来调参

| 指标 | 数值 |
|------|------|
| 区间 | 2024-01-01 ~ 2026-05-31 |
| 币种 | BTC / ETH / SOL / BNB / XRP |
| 组合初始 | $50,000（5 币 × $10k 桶） |
| **净收益** | **+1.13%**（约 +$564） |
| chop 段数 | 30（29 个月里约一半月份无网格） |
| 成交笔数 | 104 |
| 笔胜率 | 78.8% |
| 段胜率 | 86.7% |
| 最大回撤 | -0.07% |

**结论**：整体微盈，但 **coverage 极低**；不是「策略爆赚」，而是「少数 chop 段里略赚」。

### 1.2 Live 引擎（`backtest_multileg_timeline --no-trend`）— 不要用来调参

同窗口 **+7192%**（复利仓位、1.6 万笔）与上面 **不可比**。  
调优、省费、是否 promote，一律以 **diagnose / chop_grid_backtest** 为准。

---

## 2. 手续费：钱去哪了？

### 2.1 回测里的费用模型

| 路径 | 入场 | 止盈出场 | 强制出场（regime 失效） |
|------|------|----------|-------------------------|
| 默认 diagnose | maker 4 bps | maker 4 bps | **taker 4 bps** + 可选 slippage |
| Live prod 参考 | maker **2 bps** | maker 2 bps | taker ~5 bps（见 multileg 配置） |

当前 prod 执行层已是 **limit 网格 + `order_type: limit`**，正常 TP 走 maker；**贵的是 forced exit**。

### 2.2 你这次 run 的费用拆解（`metrics.json`）

| 项 | 数值 | 含义 |
|----|------|------|
| **forced 出场占比** | **39.4%** | 约 4 成平仓走 regime 强平（taker） |
| grid TP 手续费累计 | 504 bps（全样本 bps 加总） | maker 往返 |
| forced 手续费累计 | 328 bps | taker 侧更贵 |
| forced 净 PnL 贡献 | **-0.45%**（组合口径） | 强平是主要「吃利润」来源 |
| 毛 alpha（gross） | +42% pooled 量级 | 扣费后只剩 ~1.1% timeline 净收益 |

**核心矛盾**：网格 **毛 edge 有**，但 **turnover + 近 40% taker 强平** 把利润磨到很薄。  
省费 ≠ 单纯把 `fee_bps` 改小（那是改假设）；要 **少交易、少强平、拉大每格利润**。

### 2.3 能否省手续费？— 可以，按优先级

#### P0 — 减少 forced exit（最有效）

forced 是 **maker 入场 + taker 出场**，还常伴随不利价格。

| 手段 | 配置 / 脚本 | 思路 |
|------|-------------|------|
| 更早退出 chop | `regime.yaml` → `exit_below`（现 0.33） | 略提高 → 趋势露头时更快撤网格、强平库存 |
| 更严进入 chop | `entry_min`（现 0.52） | 略提高 → 少在「假 chop」里开网格 |
| 段内止损 | `execution.yaml` → `max_loss_per_grid: 0.03` | 略收紧 → 单边被套时少扛到 regime 尾 |
| 缩短最长段 | `max_segment_bars: 120` | 减少「chop 已死但还挂着」的段 |
|  hysteresis 扫参 | `scripts/sweep_chop_regime_thresholds.py` | 系统扫 `chop_min` × `exit_chop_min` |

看结果时盯 **`forced_rate`**、**`forced_exit_pnl`**，不要只看总 return。

#### P1 — 减少成交次数（降 turnover）

| 手段 | 配置 | 思路 |
|------|------|------|
| 关补挂 | `max_replenish_per_level_per_segment: 0` | 每档最多 1 次 TP，约减半重复挂单 |
| 加宽间距 | `spacing.min_pct` / `atr_mult` | 现 **dense 3L**：`atr_mult=0.01`, `min_pct=0.33%`；可对照 README 推荐的 **0.50 ATR / 0.4%** |
| 少层数 | `max_levels_per_side: 2` | 少挂外档，少 cascade 成交 |
| 扫层数 | `scripts/sweep_chop_grid_levels.py` | fixed_spacing vs fixed_span 对比 |

历史：`sweep_chop_grid_replenish.py` 推荐 **`max_replenish=1`**（现 prod 已是 1）；再降到 **0** 可试省费，但 coverage/收益可能再降。

#### P2 — 拉大每格净利润（在 spacing 固定时）

| 手段 | 配置 | 思路 |
|------|------|------|
| TP 倍数 | `tp_spacing_mult: 2`（现值） | 实验文档：mult=2 在 dense 3L 下 Sharpe / 成本鲁棒性较好；**降到 1** 会增 turn、增费 |
| 略加宽 spacing | 同上 P1 | 每格目标更大，单笔 fee 占比更低 |

#### P3 — 实盘费率（真降成本，非回测假设）

- VIP / BNB 抵扣 → maker 2 bps 甚至更低（prod 已按 2 bps 设计）。  
- 回测对比时用：`--maker-fee-bps 2 --taker-fee-bps 5` 贴近 live，不要用 4/4 当乐观估计。

#### 不建议

- 把回测 `fee_bps` 改成 0「看好看数字」— 不解决实盘。  
- 用 timeline **compound +7192%** 论证调参成功。  
-  unlimited 补挂（`max_replenish: null`）— 历史 OOS 更差。

---

## 3. 有没有更好的调优？

### 3.1 当前 prod 栈（dense 3L）在优化什么

`archetypes/execution.yaml` 注释：E7 四段验证栈 — **间距由 `min_pct=0.33%` 主导**（`atr_mult=0.01` 几乎不绑），3 层/侧，TP = 2× spacing。

这与 README 里早期研究默认 **0.50 ATR** 不同：**更密、更 turn、更吃 fee 鲁棒性**。  
你本次仅 30 段、104 笔，说明 **regime 0.52/0.33 + prefilter** 已经极严；再「加收益」往往要先 **加 coverage** 或 **换窗口**，不是无脑放宽。

### 3.2 调参杠杆（按 R&D 顺序）

| 优先级 | 层 | 旋钮 | 现 locked 值 | 调优方向 |
|--------|-----|------|--------------|----------|
| 1 | Regime | `entry_min` / `exit_below` | 0.52 / 0.33 | sweep；trade-off：coverage vs 段质量 |
| 2 | Prefilter | `box_pos_60` 带 | [0.40, 0.60] | 实验目录 `20260617_chop_grid_prefilter_fix` 有 A1 变体 |
| 3 | Execution | spacing / levels / tp_mult | dense 3L | `sweep_chop_grid_levels.py` |
| 4 | Execution | replenish | 1 | 0 vs 1 省费对比 |
| 5 | Regime | `exclude_box_prefilter` | **false**（现 prod） | README 推荐 `chop_not_box`；可 A/B |
| 6 | 信号 | `chop_signal: raw vs ts_quantile` | raw | 横截面不稳时用分位数 chop |

**仓库内实验结论（勿跳过 Phase 3）**：

- `config/experiments/20260603_multileg_segment_validate/chop_grid/DECISION.md`：长窗 timeline 仍为正，**近 6m OOS 略负**，**未 promote**。  
- OOS 弱因：**1min 执行 + 20 bps 全成本**（含 funding 假设）。  
- 调参目标应是 **OOS timeline return > 0 且 segment_win_rate > 40%**，不是 in-sample +1%。

### 3.3 分币种（你这次）

| Symbol | 段数 | return（桶内） | forced 拖累 |
|--------|------|----------------|-------------|
| SOL | 3 | 最高 | 小 |
| XRP | 6 | 高 | 小 |
| BNB | 5 | 中 | forced 负贡献最大 |
| BTC | 9 | 中 | 中 |
| ETH | 7 | 中 | forced 负贡献较大 |

可先 **ETH/BNB 加严 prefilter 或略提高 entry_min** 做分币实验，不必 5 币同一套阈值。

### 3.4 推荐工作流（可复制）

```bash
cd /Users/jerry/project/yin/ml-trading-bot
source .venv/bin/activate && source scripts/env_macos_blas.sh

# Step 1 — Regime 阈值 sweep（便宜，先跑）
python scripts/sweep_chop_regime_thresholds.py \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --start 2024-01-01 --end 2026-05-31 \
  --sweep-chop-min 0.45,0.50,0.52,0.55,0.58 \
  --sweep-exit-chop-min 0.28,0.33,0.38 \
  --out-csv results/chop_grid/sweep_regime_2024_2026.csv

# Step 2 — 层数 / spacing（OOS 窗）
python scripts/sweep_chop_grid_levels.py   # 见脚本内 segment 定义

# Step 3 — 补挂 0 vs 1
python scripts/sweep_chop_grid_replenish.py

# Step 4 — 候选组合 full backtest + 报告
python scripts/chop_grid_backtest.py \
  --config config/strategies/chop_grid/meta.yaml \
  --start 2024-01-01 --end 2026-05-31 \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --maker-fee-bps 2 --taker-fee-bps 5 \
  --out-dir results/chop_grid/candidate_<name>
```

读结果时打开：

- `metrics.json` → `forced_exit_loss_attribution`、`cost_attribution`  
- `report.html` → Alpha/Cost Diagnostics、By Month  
- `capital_report.html` → 净权益曲线  

---

## 4. 一句话策略建议

| 问题 | 建议 |
|------|------|
| 有没有更好调优？ | **有**，但应在 **regime 阈值 → spacing/层数 → prefilter** 上 sweep，并以 **OOS timeline + forced_rate** 判据；现 +1.13% 说明 **edge 很薄**，大幅提收益靠 **加 coverage 或降成本**，不是加杠杆。 |
| 能否省手续费？ | **能**：首要 **降 forced exit 比例**（约 40% → 目标 <25%），其次 **降 turnover**（replenish=0、略宽 spacing），实盘用 **maker 2 bps**；每格利润用 **tp_spacing_mult≥2** 维持。 |
| 值不值得上实盘？ | 实验 **未 promote**（OOS 略负）；继续 R&D 可以，prod 应等 **OOS 三条杠** 或与 trend_scalp 联合 timeline 再验。 |

---

## 5. 相关文件

| 文件 | 用途 |
|------|------|
| `archetypes/regime.yaml` | chop 进出场阈值 |
| `archetypes/prefilter.yaml` | box_pos 带 |
| `archetypes/execution.yaml` | spacing / replenish / TP / 风险 |
| `scripts/sweep_chop_regime_thresholds.py` | 阈值网格 |
| `scripts/sweep_chop_grid_levels.py` | 层数 / spacing |
| `scripts/sweep_chop_grid_replenish.py` | 补挂 |
| `config/experiments/20260603_multileg_segment_validate/chop_grid/DECISION.md` | promote 判决 |
| `scripts/run_multileg_backtest_with_maps.sh` | 一键回测 + 地图 |

---

## 6. 快速对照命令（省费 A/B）

同一窗口，对比 **现 prod** vs **省 turn 候选**（示例）：

```bash
# A — 现 prod（baseline）
python scripts/chop_grid_backtest.py \
  --config config/strategies/chop_grid/meta.yaml \
  --start 2024-01-01 --end 2026-05-31 \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --maker-fee-bps 2 --taker-fee-bps 5 \
  --out-dir results/chop_grid/ab_baseline

# B — 省费候选：关补挂 + live 费率（spacing 仍用 yaml）
python scripts/chop_grid_backtest.py \
  --config config/strategies/chop_grid/meta.yaml \
  --start 2024-01-01 --end 2026-05-31 \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --max-replenish-per-level 0 \
  --maker-fee-bps 2 --taker-fee-bps 5 \
  --out-dir results/chop_grid/ab_no_replenish

# 对比 forced_rate / return_pct
python -c "
import json
for d in ['ab_baseline','ab_no_replenish']:
    m=json.load(open(f'results/chop_grid/{d}/metrics.json'))['metrics']
    t=m['trade_summary']
    print(d, 'return%', t['return_pct_timeline'], 'forced%', t['forced_rate']*100)
"
```

若 B 的 `forced_rate` 仍高，优先动 **regime exit_below**，不要只改 fee 假设。
