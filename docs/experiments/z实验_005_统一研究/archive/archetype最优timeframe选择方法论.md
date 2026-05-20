# Archetype 最优 Timeframe 选择方法论

> 创建时间: 2026-02-21
> 背景: ME 从 240T 切到 60T 后，gate 规则全部失效，引发根本问题——如何确定每个 archetype 适合哪个 timeframe？

---

## 问题定义

系统有多个 archetype（ME/BPC/FER/LV），每个有候选 timeframe（15T/60T/240T）。
需要一个**可量化、可复现**的方法确定最优 timeframe，而不是纯靠经验直觉。

---

## 方法论：三级验证漏斗

### Level 1: 原生特征区分力扫描（最快，几分钟）

**核心思想**：如果 archetype 专属特征在某个 timeframe 上连好坏样本都分不开，那模型也学不到。

**操作**：用 `analyze_archetype_feature_stratification.py` 对每个候选 timeframe 跑分位数分层：

```bash
# 在目标 timeframe 上构建 FS → 训练 → 拿到 predictions.parquet
# 然后分析原生特征区分力
python scripts/analyze_archetype_feature_stratification.py \
  --logs results/train_final_xxx/{strategy}/predictions.parquet \
  --strategy {strategy} \
  --config config/strategies/{strategy}/prefilter.yaml
```

**判定标准**：
| 指标 | 通过条件 | 含义 |
|------|----------|------|
| 正信号 bad rate 差异 | > 5% | 原生特征能区分 archetype 存在域 |
| 反信号 bad rate 差异 | > 5% | 原生特征能识别失败模式 |
| 跨 symbol 一致性 | > 4/6 symbol 同向 | 不是单币种过拟合 |
| 滚动窗口稳定性 | CV < 1.0 | 区分力不是某个时期的巧合 |

**决策**：
- 全部不通过 → 此 timeframe 不适合此 archetype，直接排除
- 部分通过 → 进入 Level 2

### Level 2: Prefilter + Gate 优化（中等，约 30 分钟）

**核心思想**：在 Level 1 通过的 timeframe 上，跑完整 prefilter → 训练 → gate 优化流程。

**操作**：
```bash
# 1. 训练
mlbot train final --timeframe {TF} --archetype-prefilter ... --labels ...

# 2. Gate 应用
mlbot gate apply-archetype --logs predictions.parquet --strategy {strategy}

# 3. Gate 优化
python scripts/optimize_gate_unified.py --strategy {strategy} --logs logs_gated.parquet
```

**判定标准**：
| 指标 | 通过条件 | 含义 |
|------|----------|------|
| Gate Lift | > 1.05 | Gate 有区分力（至少降 5% bad rate） |
| Gate pass rate | 10%-70% | 不过宽也不过严 |
| 模型 CV | > 0.01 | 模型至少比随机好 |
| 专属特征 importance > 0 | ≥ 2 个 | archetype 语义被模型捕获 |

**决策**：
- 多个 timeframe 都通过 → 进入 Level 3
- 只有一个通过 → 就用这个
- 全部不通过 → archetype 本身的 alpha 假设需要重新审视

### Level 3: 整体回测对比（最慢，约 1-2 小时）

**核心思想**：多个 timeframe 都有 alpha 时，需要用端到端回测比 Sharpe/最大回撤。

**操作**：用 `backtest_execution_layer.py` 在每个候选 timeframe 上跑完整回测。

**对比维度**：
| 维度 | 含义 | 决策权重 |
|------|------|----------|
| Sharpe Ratio | 风险调整收益 | ⭐⭐⭐ 核心 |
| 交易频率 | 低 TF 交易多、高 TF 交易少 | ⭐⭐ 需足够样本 |
| 最大回撤 | 尾部风险 | ⭐⭐ 实盘关键 |
| Failure rate（top 30%） | 模型优选交易的质量 | ⭐⭐ 模型价值 |
| 每笔平均 R | 单笔收益 | ⭐ 参考 |

---

## 为什么不能只看 Sharpe（跳过 Level 1/2）

1. **成本高**：每个 timeframe × 每个 archetype 要跑完整 pipeline，组合爆炸
2. **掩盖根因**：Sharpe 是终端指标，如果差，你不知道是特征没用、gate 没用还是 label 不匹配
3. **Level 1 几分钟就能排除不可能的 timeframe**，避免无谓算力浪费

## 为什么不能只看原生特征区分力（只做 Level 1）

1. **区分力 ≠ 可学习性**：特征有区分力，模型未必能学到（非线性、交互效应）
2. **区分力 ≠ 盈利**：特征能分好坏，但扣除交易成本后可能不赚钱
3. **区分力 ≠ 稳定性**：某个时期有效，不代表全周期都有效
4. **交易频率效应**：60T 的区分力可能比 240T 弱，但交易量 4x → 总利润可能更高

---

## 当前 Archetype × Timeframe 状态

| Archetype | 当前 TF | Level 1 | Level 2 | Level 3 | 备注 |
|-----------|---------|---------|---------|---------|------|
| BPC | 240T | ✅ 专属+通用都有用 | ✅ CV=0.079 | ✅ 已实盘 | 240T 已验证 |
| ME | 60T | ⚠️ 仅 cvd_alignment 有效 | ⚠️ CV=0.003 | 🔲 待做 | 60T 需进一步验证 |
| ME | 240T | ✅ atr_pct 有效 (Lift=18%) | ✅ Gate 有 plateau | 🔲 待做 | 240T 已有 gate v3 |
| FER | 240T | ✅ trapped_score 有效 | ✅ optimizer 验证 | 🔲 待做 | |
| LV | 15T | ⚠️ 无专属特征 | ⚠️ 通用特征统治 | 🔲 待做 | 需要开发专属特征 |

### ME 60T vs 240T 对比

| 维度 | 60T | 240T |
|------|-----|------|
| 有效 gate 规则 | 仅 cvd_alignment (Lift=1.21) | atr_pct (Lift=1.18) + 2 条候选 |
| 模型 CV | 0.003 (极低) | 待测 |
| Prefilter 后样本量 | ~8700 | 待测 |
| 专属特征 importance | 仅 cvd_alignment(100.7)，其余全=0 | atr_pct(92)，少数非零 |
| 优势 | cvd_alignment 信号更实时 | 多个特征有区分力 |
| 劣势 | 模型几乎没学到东西 | 交易频率低 |

**初步判断**：ME 在 240T 上有更丰富的特征信号，60T 的 cvd_alignment 虽然区分力强但是 binary 特征，模型整体学习能力弱。**建议在 240T 上也跑一轮 prefilter 训练做对比**。

---

## 快速启动命令：多 Timeframe 对比实验

```bash
# Step 1: 在每个候选 timeframe 上训练（假设 FS 已构建）
for TF in 60T 240T; do
  mlbot train final --no-docker \
    --config config/strategies/me \
    --features config/strategies/me/features_gate.yaml \
    --labels config/strategies/me/labels_rr_extreme.yaml \
    --archetype-prefilter config/strategies/me/archetypes/prefilter.yaml \
    --symbol BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
    --timeframe ${TF} --data-path data/parquet_data \
    --start-date 2023-01-01 --end-date 2026-01-01 \
    --holdout-start-date 2024-05-01 --holdout-end-date 2026-01-01 --seed 42
done

# Step 2: 对比原生特征区分力
for TF in 60T 240T; do
  RESULT_DIR=$(ls -td results/train_final_*/me | head -1)
  python scripts/analyze_archetype_feature_stratification.py \
    --logs ${RESULT_DIR}/predictions.parquet \
    --strategy me \
    --config config/strategies/me/prefilter.yaml
done

# Step 3: 如果两个都通过 Level 1/2，跑 Sharpe 对比
```

---

## 附录：各 Archetype 的 Timeframe 先验直觉

供参考，非结论：

| Archetype | 语义 | 直觉偏好 TF | 理由 |
|-----------|------|-------------|------|
| ME | 动能扩张 | 60T-240T | 趋势需要时间展开，太短则噪音多 |
| BPC | 压缩突破 | 240T | 压缩是中长期结构，需要足够 bar 累积 |
| FER | 反转失败 | 60T-240T | 反转信号在多周期都存在 |
| LV | 清算脆弱性 | 15T-60T | 清算是快事件，短周期捕获更及时 |

**但直觉必须用数据验证（Level 1→2→3）**。ME 从 240T 切到 60T 后 gate 全部失效就是活生生的反例。
