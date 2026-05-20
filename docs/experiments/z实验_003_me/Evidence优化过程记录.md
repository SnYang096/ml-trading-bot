# ME Evidence 优化过程记录

## 时间线
**执行日期**: 2026-02-16  
**执行环节**: Evidence 特征精简（基于 RR 分层）

---

## 问题发现

### 初始错误理解
❌ **错误**: Evidence 优化基于 failure 判断 `forward_rr >= -0.8`  
✅ **正确**: Evidence 优化基于 **RR 分层**（Q20/Q80）

### 参考文档
- `docs/z实验_001_bpc/return_tree_kpi_framework.md` 第 76-105 行
- `docs/z实验_001_bpc/return_tree训练报告解读.md`

---

## 关键修复

### 1. 修正 Evidence 优化脚本

**文件**: `scripts/optimize_evidence_plateau.py`  
**位置**: 第 1028-1042 行

**修复内容**:
```python
# 修复前（错误）：基于 failure 判断
if args.label_col not in df.columns:
    if rr_col is not None:
        df[args.label_col] = (df[rr_col] >= -0.8).astype(int)
        print(f"ℹ️ Auto-generated '{args.label_col}' column from '{rr_col}' (threshold: -0.8)")

# 修复后（正确）：基于 RR 分层
if args.label_col not in df.columns:
    if rr_col is not None:
        # ❗ Evidence优化基于RR分层: Good = Q4-Q5 (高RR), Bad = Q1-Q2 (低RR)
        q20 = df[rr_col].quantile(0.2)
        q80 = df[rr_col].quantile(0.8)
        # Good: RR > Q80, Bad: RR < Q20, Middle: 排除
        df_good = df[df[rr_col] > q80].copy()
        df_bad = df[df[rr_col] < q20].copy()
        df_good[args.label_col] = 1
        df_bad[args.label_col] = 0
        df = pd.concat([df_good, df_bad], ignore_index=True)
        print(f"ℹ️ Auto-generated '{args.label_col}' based on RR stratification")
        print(f"   Q20={q20:.2f}, Q80={q80:.2f}")
        print(f"   Good (RR > Q80): {len(df_good)}, Bad (RR < Q20): {len(df_bad)}")
```

---

### 2. ME Evidence 配置格式转换

**问题**: ME 的 `evidence.yaml` 使用列表格式（150 个特征），不兼容优化脚本（期望 BPC 的对象格式）

**解决方案**: 基于 ME 策略语义筛选 16 个核心候选特征

**筛选依据**（ME 语义：压缩 → 扩张 → 突破）:
- ATR 扩张、成交量确认
- VPIN 系列（订单流活跃度）
- 路径效率、SR 距离
- WPT 突破信心、假突破风险
- 技术指标：MACD、RSI

**转换格式**:
```yaml
evidence:
  - id: evidence_vpin
    feature: vpin
    direction: positive
    usage_hint: VPIN活跃度,影响执行速度和信心
    affects:
      - trailing_speed
      - confidence_boost
```

---

## 执行命令

### 1. 检查数据分布
```bash
cd /home/yin/trading/ml_trading_bot

python3 -c "
import pandas as pd
import numpy as np

df = pd.read_parquet('results/strategies/me/predictions.parquet')
print(f'总样本数: {len(df)}')
print(f'\nforward_rr统计:')
print(f'  Min: {df[\"forward_rr\"].min():.2f}')
print(f'  Max: {df[\"forward_rr\"].max():.2f}')
print(f'  Mean: {df[\"forward_rr\"].mean():.2f}')
print(f'  Std: {df[\"forward_rr\"].std():.2f}')

q20 = df['forward_rr'].quantile(0.2)
q80 = df['forward_rr'].quantile(0.8)
print(f'\n分位数:')
print(f'  Q20 (Bad阈值): {q20:.2f}')
print(f'  Q80 (Good阈值): {q80:.2f}')

n_bad = (df['forward_rr'] < q20).sum()
n_good = (df['forward_rr'] > q80).sum()
print(f'\n样本分层:')
print(f'  Bad samples (RR < Q20): {n_bad}')
print(f'  Good samples (RR > Q80): {n_good}')
print(f'  Middle samples: {len(df) - n_bad - n_good}')
"
```

**输出结果**:
```
总样本数: 368
forward_rr统计:
  Min: -0.78
  Max: 54.91
  Mean: 8.35
  Std: 8.54
分位数:
  Q20 (Bad阈值): 1.88
  Q80 (Good阈值): 15.61
样本分层:
  Bad samples (RR < Q20): 74
  Good samples (RR > Q80): 74
  Middle samples: 220
```

---

### 2. 运行 Evidence 优化
```bash
cd /home/yin/trading/ml_trading_bot

python scripts/optimize_evidence_plateau.py \
  --strategy me \
  --logs results/strategies/me/predictions.parquet \
  --output results/strategies/me/evidence_optimization.json
```

**输出结果**:
```
✅ Loaded strategy: me
   Evidence features: 15
✅ Loaded 368 rows from results/strategies/me/predictions.parquet
ℹ️ Auto-generated 'is_good' based on RR stratification
   Q20=1.88, Q80=15.61
   Good (RR > Q80): 74, Bad (RR < Q20): 74
   Good samples: 74, Bad samples: 74
   Good rate: 0.500

📋 Optimizing Evidence Features:
  Processing: vpin (direction: higher_is_better)
    ✅ Optimized (Stable Plateau):
       Bins: [0.1, 0.3, 0.5, 0.7]
       Bad Suppression: 0.243 ⭐
       Good Amplification: 0.108 ⭐
       Neighbors: 4 | Semantic drift: 0.095

  Processing: rsi (direction: higher_is_better)
    ✅ Optimized (Stable Plateau):
       Bins: [0.25, 0.45, 0.65, 0.85]
       Bad Suppression: 0.230 ⭐
       Good Amplification: 0.162 ⭐
       Neighbors: 4 | Semantic drift: 0.095

  Processing: macd_atr (direction: higher_is_better)
    ✅ Optimized (Stable Plateau):
       Bins: [0.25, 0.45, 0.55, 0.75]
       Bad Suppression: 0.203 ⭐
       Good Amplification: 0.189 ⭐
       Neighbors: 3 | Semantic drift: 0.068

📊 Summary:
   ✅ Optimized (Stable Plateau): 3
   ❌ Rejected (No Plateau): 4
   ❌ No valid bins: 8
   ⏭ Skipped: 0
```

---

### 3. 同步配置到实盘
```bash
cd /home/yin/trading/ml_trading_bot

# 同步 Evidence 配置
cp config/strategies/me/archetypes/evidence.yaml \
   live/highcap/config/strategies/me/archetypes/evidence.yaml

echo "✅ Evidence配置已同步到实盘目录"
```

---

## 优化结果

### 有效特征（3个）

| Rank | 特征 | Bad Suppression | Good Amplification | Quantile Bins | 用途 |
|------|------|-----------------|-------------------|---------------|------|
| 1 | **VPIN** | 0.243 ⭐⭐⭐ | 0.108 | [0.1, 0.3, 0.5, 0.7] | 订单流活跃度，影响执行速度和信心 |
| 2 | **RSI** | 0.230 ⭐⭐ | 0.162 | [0.25, 0.45, 0.65, 0.85] | 技术指标位置，影响仓位和信心 |
| 3 | **MACD ATR** | 0.203 ⭐⭐ | 0.189 | [0.25, 0.45, 0.55, 0.75] | 动量确认，影响持有时间和 trailing 速度 |

### 拒绝原因统计

- **8 个特征**: 不满足双重约束（bad_suppression + good_amplification）
  - `atr_percentile`, `volume_ratio_pct`, `vpin_momentum`, 
  - `price_dir_consistency_pct`, `path_efficiency_pct`, 
  - `dist_to_nearest_sr`, `liquidity_void_detected`, `cvd_change_5_normalized`

- **4 个特征**: 无稳定 plateau（邻域语义不稳定）
  - `vpin_trend`, `wpt_breakout_confidence`, 
  - `wpt_false_breakout_risk`, `vp_entropy`

---

## 关键技术点

### Evidence 优化逻辑（参考 BPC）

1. **样本分层**:
   - Good: `forward_rr > Q80` (高 RR 交易)
   - Bad: `forward_rr < Q20` (低 RR 交易)
   - Middle: 排除（不参与优化）

2. **主 KPI**: `bad_suppression`
   ```
   bad_suppression = P(score < 0.3 | bad) - P(score < 0.3 | good)
   ```
   - 衡量特征区分高 RR 和低 RR 的能力
   - 正值越大越好（Bad 被选择性压制）

3. **双重约束**:
   - `bad_suppression > 0.05` (压制 Bad)
   - `good_amplification > 0.05` (放大 Good)
   - 同时满足才认为有效

4. **Plateau 判断**:
   - CV < 0.5 (数值稳定)
   - Bins 距离 < 0.15 (空间连续)
   - 邻域 >= 3 (邻居充足)
   - Semantic drift < 0.10 (语义稳定)

---

## 产出文件

| 文件 | 位置 | 说明 |
|------|------|------|
| 优化结果 JSON | `results/strategies/me/evidence_optimization.json` | 完整优化数据 |
| HTML 报告 | `results/strategies/me/evidence_optimization.html` | 可视化报告 (12KB) |
| Evidence 配置 | `config/strategies/me/archetypes/evidence.yaml` | 研究配置 |
| Evidence 配置 (实盘) | `live/highcap/config/strategies/me/archetypes/evidence.yaml` | 实盘配置 |

---

## 技术验证

### 代码复用性
✅ **完全复用 BPC 框架**
- 脚本：`scripts/optimize_evidence_plateau.py` (同一个文件)
- 只修复了 RR 分层逻辑 bug
- ME 和 BPC 调用相同代码

### 参考文档
- `docs/z实验_001_bpc/return_tree_kpi_framework.md`
- `docs/z实验_001_bpc/return_tree训练报告解读.md`
- `docs/z实验_001_bpc/entry_filter_design.md`

---

## 下一步

1. ✅ Evidence 优化完成（3 个有效特征）
2. ⏳ Entry Filter 已添加（2 个规则：订单流确认 + 空间确认）
3. ⏳ Execution 已优化（Sharpe=0.26, activation_r=0.5, trail_r=0.5）
4. ⏳ Gate 已添加（2 个 guardrail：ATR 扩张 + 放量确认）
5. ⏳ 待验证：PCM 联合回测（ME + BPC）
