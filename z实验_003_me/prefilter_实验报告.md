# ME Prefilter 特征区分力实验报告

> 日期：2026-02-21
> 数据源：`results/train_final_20260221_125709_rr_extreme/me/features_labeled.parquet`
> 脚本：`analyze_archetype_feature_stratification.py --select-recent 6`
> 配置：`config/strategies/me/prefilter.yaml`

---

## 1. 实验设置

| 参数 | 值 |
|------|-----|
| 数据生成方式 | `--prepare-only` (全周期 features + labels, 无模型训练) |
| 总数据量 | 157,086 行 × 246 列 |
| 币对 | BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, ADAUSDT |
| 时间范围 | 2023-01 → 2025-12 (约 35 个月) |
| 特征选择窗口 (Mode A) | 最近 6 个月 (2025-06-29 → 2025-12-29, 26,358 行) |
| Temporal Rolling 数据 | 全周期 157,086 行 |
| Rolling 窗口候选 | [2, 3, 4, 6] 个月 |
| 全局 bad rate | 44.4% |
| 标签 | `success_no_rr_extreme` |

**分析模式**: Mode A — 用近 6 个月筛选有显著区分力的特征，再在全周期上做 rolling 验证稳定性。

---

## 2. 候选特征清单

来自 `prefilter.yaml`，共 5 个特征组，展开为 **21 个候选特征**：

| 特征组 | 展开列 | 数量 |
|--------|--------|------|
| me_soft_phase_f | me_atr_pct, me_vol_regime, me_accel_2k, me_accel_5k, me_accel_persistence, me_multi_tf_alignment, me_cvd_alignment, me_cvd_strength, me_volume_surge, me_volume_accel, me_delta_net_flow | 11 |
| me_failure_f | me_false_expansion, me_vol_divergence, me_flow_exhaustion, me_failure_score | 4 |
| me_context_f | me_jump_risk_suitable, me_reflex_risk, me_regime_suitable | 3 |
| vol_regime_features_f | vol_zscore, vol_percentile_approx | 2 |
| atr_percentile_f | atr_percentile | 1 |

---

## 3. 核心发现

### 3.1 atr_percentile — 唯一稳定的王牌特征

| 阈值 | 近 6 月 bad_rate diff | 全周期 CV | 近期趋势 | 判定 |
|------|----------------------|-----------|----------|------|
| **P95** (≥0.965) | **-15.9%** | 0.75 | 2025 下半年持续 -12% ~ -18% | **可用，强信号** |
| **P90** (≥0.922) | **-10.9%** | 0.70 | 2025 下半年持续 -10% ~ -16% | **可用，最稳** |
| P80 (≥0.815) | -4.0% | 0.76 | 从 -12% 衰退至 -4% | **不可用，正在失效** |

**关键观察**：
- P90/P95 在 2023 下半年到 2024 初有一段走弱期（P95 有窗口翻正），说明 **信号有周期性，不是永远有效**
- P80 阈值太松，2025 年以来持续衰退：-12% → -8% → -5.5% → -4%，不应进入 prefilter
- **建议 prefilter 使用 P90 作为门槛**，P95 作为强信号增强

### 3.2 me_accel_2k P90 — 已失效

| 阈值 | 近 6 月 bad_rate diff | 全周期 CV | 近期趋势 | 判定 |
|------|----------------------|-----------|----------|------|
| P90 (≥1.222) | **+2.6%** | 1.04 | 最后 3 窗口翻正: +0.7% → +1.4% → +2.6% | **失效，信号反转** |

- 2023-2024 中段曾有 -3% ~ -5% 的正信号
- 2025 下半年完全翻正 → 高加速度不再降低 bad rate，反而增加
- **结论：me_accel_2k 不应进入 prefilter**（仍可作为 Evidence 模型输入）

### 3.3 ME 专有特征 — 普遍不稳定

| 特征 | 近 6 月最大 |bad_rate diff| | CV | 判定 |
|------|--------------------------|------|------|
| me_volume_surge P95 | ~-4% | >0.5 | 弱正信号，不够稳定 |
| me_vol_regime | <-3% | >0.8 | 不稳定 |
| me_cvd_alignment | <-2% | >1.0 | 极不稳定 |
| me_cvd_strength | <-2% | >1.0 | 极不稳定 |
| me_failure_score | <-2% | >0.8 | 不稳定 |
| me_reflex_risk | <-1% | >1.0 | 无信号 |

**结论**：当前 ME 专有特征在 prefilter 层面没有独立区分力，区分力主要来自通用波动率特征（atr_percentile）。

---

## 4. atr_percentile 全生命周期 (6m Rolling)

```
时间          P95 diff    P90 diff    P80 diff    解读
─────────────────────────────────────────────────────────
2023 上半年    -2~-4%      -6~-8%      -6~-8%     P80/P90 强，P95 弱
2023 下半年    -0~-4%      -2~-5%      -1~-4%     整体走弱期
2024 上半年    -4~-15%     -3~-5%      +0~-1%     P95 先恢复，P80 无效
2024 下半年    -5~-9%      -7~-9%      -3~-8%     全面走强
2025 上半年    -10~-18%    -10~-16%    -8~-12%    历史最强
2025 下半年    -12~-16%    -10~-11%    -4~-4%     P95/P90 稳定，P80 衰退
```

**周期性规律**：信号强度有 ~6-12 个月的周期波动，但 P90/P95 在弱期也基本不翻正（仅 P95 在 2023-09 有一次 +1.3%），说明高阈值的底线保护力强。

---

## 5. 与上一轮对比 (predictions.parquet vs features_labeled.parquet)

| 维度 | 上一轮 (predictions.parquet) | 本轮 (features_labeled.parquet) |
|------|------|------|
| 数据覆盖 | Holdout only (2024-05 → 2026-01) | **全周期 (2023-01 → 2025-12)** |
| 总行数 | ~50k | **157k** |
| Rolling 可观察窗口数 | ~10 个 | **~30 个** |
| 能看到的信号周期 | 仅 1.5 年 | **3 年完整生命周期** |
| 新发现 | 无 | atr_percentile 有 ~6-12 月周期性走弱 |

**结论**：`--prepare-only` 全周期数据使分析更可信，新流程合理。

---

## 6. Prefilter 配置建议

基于本次分析，ME prefilter 推荐规则：

```yaml
# Gate hard_gate 候选
pre_filter:
  - feature: atr_percentile
    operator: ">="
    percentile: 90          # P90, 约 0.922
    rationale: "全周期 CV=0.70, 近 6 月 bad_rate 降低 10.9%, 3 年内仅 1 次窗口翻正"
```

**不建议纳入**：
- `atr_percentile P80`：近期衰退至 -4%，不可靠
- `me_accel_2k`：信号已反转
- 其他 ME 专有特征：CV > 0.5，独立区分力不足

**可保留为 soft filter / Evidence 输入**（非 prefilter 硬门槛）：
- `me_volume_surge P95`：有弱正信号，适合做 Evidence 加分项
- `me_accel_5k`：中观加速度，适合 Evidence 强度评估

---

## 7. 下一步行动

| 序号 | 行动 | 责任 |
|------|------|------|
| 1 | 更新 `archetypes/gate.yaml` hard_gate: `atr_percentile >= P90` | 用户 |
| 2 | 运行 Direction 验证 (Step 4) | 用户 |
| 3 | 正式 Gate 训练 (Step 5) | 用户 |
| 4 | 定期复跑 `--select-recent 6` 监控 atr_percentile 衰退 | 周期性 |

---

## 附录：实验命令

```bash
# Step 2: 数据准备
mlbot train final --no-docker --prepare-only \
  --config config/strategies/me \
  --features config/strategies/me/features_gate.yaml \
  --labels config/strategies/me/labels_rr_extreme.yaml \
  --symbol BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 60T --data-path data/parquet_data \
  --start-date 2023-01-01 --end-date 2026-01-01 \
  --holdout-start-date 2024-05-01 --holdout-end-date 2026-01-01 --seed 42

# Step 3: Prefilter 分析 (Mode A)
python scripts/analyze_archetype_feature_stratification.py \
  --logs results/train_final_20260221_125709_rr_extreme/me/features_labeled.parquet \
  --strategy me --config config/strategies/me/prefilter.yaml \
  --select-recent 6
```
