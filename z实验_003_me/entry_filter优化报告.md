# ME Entry Filter 优化报告

## 结论

**ME 60T entry filter 最优配置: 单一 filter `bb_width_normalized_pct >= 0.873`**

- Baseline snotio=18.20 → Entry Filter snotio=24.37 (**+33.9%**)
- Trades: 1220 (占全量 5120 的 23.8%)
- Plateau 稳定区间: [0.842, 0.968], confidence=LOW (区间较窄但原始P90在区间内)

生产配置: `config/strategies/me/archetypes/entry_filters.yaml` v6.0

---

## 优化流程

### 1. 全特征 Scan (--scan)
- 224 个数值特征 × 6 阈值 (P10/P20/P30/P70/P80/P90)
- 排除 `feature_dependencies.yaml` 中的 `raw_scale_columns` (19个未归一化列)
- Top 结果: bb_width_normalized_pct >= P90 snotio=33.81 (+85.8%)

### 2. Execution 验证 (--all)
- 18 个候选 filter (10 手动 + 8 scan) 全部执行模拟
- Scan 发现的 filter 占据 Top 8

### 3. Plateau 高原验证
- KPI: **snotio** (非 Sharpe — 之前有 bug 已修)
- bb_width plateau: [0.842, 0.968], 推荐 0.873
- 多个 scan filter 阈值下调后 snotio 骤降至 baseline → 不稳定

## 4. Greedy Dedup 去冗余
- Jaccard 重叠矩阵 → 贪心前向选择
- **仅 bb_width_high 入选**，任何第二 filter OR 均稀释 snotio

### 四步逐步收紧逻辑

每一步是对前一步的验证 + 过滤，bb_width 是唯一通过全部四关的 filter：

| 步骤 | 工具 | 作用 | bb_width 结果 |
|---|---|---|---|
| 1. scan | `--scan` 224×6 | 粗筛 Top 候选 | snotio=33.81 #1 |
| 2. execution | `--all` 18 filter | 确认 snotio 排名 | 排名 #1 |
| 3. plateau | 阈值滑窗 CV | 淘汰"碰巧好"的 | [0.842, 0.968] 稳定 |
| 4. dedup | Jaccard + greedy | 去冗余，验证 N=1 最优 | 唯一入选 |

> `--all` 的 Best-per-N 图已证明 N=1 (bb_width) 是最优点；
> plateau + greedy dedup 从稳定性和冗余角度独立验证了同一结论。

---

## 关键发现

### 冗余分析 (Jaccard 重叠)
| 簇 | 成员 | Jaccard |
|---|---|---|
| VPIN 簇 | vpin_high, scan_funding_rate_abs_low, scan_vpin_ma20_high, scan_vpin_ma10_p80 | 0.98~1.00 |
| BB 簇 | scan_bb_width_high, scan_bb_width_p80 | 0.90 |
| OI 簇 | oi_absorption, oi_absorption_loose | 1.00 |
| VPIN-rank 簇 | vpin_rank_high, vpin_spike | 0.71 |

18 个 filter 实际只有 ~5 个独立信号。

### 单独 snotio (plateau 推荐阈值)
| Filter | snotio | 状态 |
|---|---|---|
| scan_bb_width_high | 24.37 | ✅ 选中 |
| scan_cvd_divergence_low | 24.15 | Jac=0.10 低重叠,但 OR 稀释 |
| scan_bb_width_p80 | 23.38 | 与 bb_width_high 重叠 0.90 |
| scan_vpin_max20_high | 22.62 | |
| wick_compression_high | 19.14 | |
| vpin_rank_high | 19.07 | |
| 其余 | ≤ baseline | 不可用 |

### 被淘汰的旧 filter
| Filter | 原因 |
|---|---|
| accel_burst_vol_regime | snotio=15.71 < baseline(18.20), plateau 后更差 |
| volume_participation | plateau [0.000, 0.200] 从 0 开始 = 无过滤力; 与 Evidence volume_surge 语义重叠 |

### 严格阈值 vs Plateau 推荐
- bb_width P90=0.957 在 plateau [0.842, 0.968] **内** → 严格阈值稳定可用
- vpin_ma20 P90=0.391 在 plateau [0.113, 0.339] **外** → 严格阈值不稳定
- **判定标准: 严格阈值是否在 plateau 区间内**

---

## 工具改进记录

| 改动 | 文件 |
|---|---|
| plateau KPI 从 Sharpe → snotio | `scripts/optimize_entry_filter_plateau.py` |
| 新增 `--research` 读取研究文件 | `scripts/optimize_entry_filter_plateau.py` |
| 新增 greedy dedup (Jaccard + 贪心) | `scripts/optimize_entry_filter_plateau.py` |
| 修复 `_generate_scan_range` 不支持负值 | `scripts/optimize_entry_filter_plateau.py` |
| 修复 `plateau_width` 负数 bug | `scripts/optimize_entry_filter_plateau.py` |
| 新增 `--scan` 全特征扫描 | `scripts/optimize_entry_filter_snotio.py` |
| 未归一化特征从 YAML 加载 | `config/feature_dependencies.yaml` `raw_scale_columns` |

---

## 评估命令

```bash
EVIDENCE_RESULT_DIR="results/train_final_20260221_215306_return_tree/me"

# execution + scan
python scripts/optimize_entry_filter_snotio.py --all \
  --logs ${EVIDENCE_RESULT_DIR}/predictions.parquet --strategy me
python scripts/optimize_entry_filter_snotio.py --scan \
  --logs ${EVIDENCE_RESULT_DIR}/predictions.parquet --strategy me

# plateau + dedup
python scripts/optimize_entry_filter_plateau.py --research \
  --logs ${EVIDENCE_RESULT_DIR}/predictions.parquet --strategy me
```
