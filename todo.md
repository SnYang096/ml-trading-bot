1. 'BreakoutPullbackContinuation','HTFBiasLTFEntry',  'MomentumExpansion','FailedBreakoutFade','LiquiditySweepRejection'，'AuctionExhaustionReversal'的方向方案重构 [策略中dir的使用方式](策略中dir的使用方式.md), 
2. nn模型保持仓位下注大小，止损止盈辅助功能。这个对三种mean模式是不是不友好
3. safety功能，用来降速甚至停止，但nn模型也能控制size，是不是不要多个输入：最大回撤，连亏次数，kl散度，mi互信息
4. 跑通order management和前面特征计算gate，archtype链接
5. 拿到上面6个archtype的稳定参数和稳健性报告。对比树模型测语义，确保语义实现的没问题
6. 上测试网测试
7. 规则类的消融方法，在archetypes那边做的
8. 一个新特征 vp_boundary_stability_score  dist_to_nearest_sr sr_strength_max
9. 建立6个树模型 
10. 世界怎么对我 label，树模型 =》regime
11. 我怎么对世界，特征 bpc 还不够
12. gate灭绝交易，做软化分层

python src/time_series_model/visualization/feature_indicator_visualizer.py \
  --data-path data/parquet_data \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-12-31 \
  --strategy-config config/strategies/compression_breakout

---

## 🟦 套路 A：策略训练 vs 🟥 套路 B：Archetype 审计

> ❗ **两套代码必须分开，不能混用**

| | 🟦 套路 A：策略训练 | 🟥 套路 B：Archetype 审计 |
|---|---|---|
| **目标** | Sharpe / 可交易性 | 发现“我在哪些情况下是错的” |
| **样本** | 结构预筛选（非回踩 = NaN） | **全样本** |
| **检测条件** | 作为筛选条件 | 作为 **feature** |
| **RR 类型** | execution-aware（有止损止盈） | path_extreme（无执行绑定） |
| **信仰** | 可以有 | **必须无** |
| **代码文件** | `xxx_label.py` | `xxx_audit_label.py` |

### 文件结构

```
src/time_series_model/strategies/labels/
├── me_label.py              # 🟦 策略训练
├── aer_label.py             # 🟦 策略训练 (exhaustion_reversal_label.py)
├── bpc_label.py             # 🟦 策略训练
├── htf_label.py             # 🟦 策略训练
├── fbf_label.py             # 🟦 策略训练
├── lsr_label.py             # 🟦 策略训练
├── bpc_audit_label.py       # 🟥 Archetype 审计
└── archetype_audit_labels.py # 🟥 通用审计框架（全部 6 个）
```

---

## 6 个 Archetype 对齐策略创建 TODO

### 背景与目标

**不做**：把 execution_archetypes 里现有规则搬到树那边（那些规则你也不确定对不对）。

**要做**：按 6 个 archetype 的语义在树这边设计 6 个新策略；树还是「自己学规则、自己训」，只是策略的交易逻辑/语义和 6 个 archetype 一一对应。

**现状**：原来 4 个策略（sr_reversal, compression_breakout, sr_breakout, trend_following）既对不上 6 个 archetype，部分 Sharpe 还负，所以用这 6 个「语义对齐 archetype」的新策略来替代/补充。

### 目录结构（完全复用现有）

```
config/strategies/<strategy_name>/
├── labels.yaml
├── model.yaml
├── features.yaml
├── meta.yaml
├── backtest.yaml
└── evaluation.yaml
```

6 个新策略 = 6 个新目录：`bpc`、`htf`、`me`、`fbf`、`lsr`、`aer`

### 模型训练方式（完全复用现有）

继续用同一套：`train_strategy_model`、LightGBM、task_type（regression / multiclass）、target_col、TS-CV 等。

### Label 设计方案

| Archetype | 语义（交易想法） | 和现有 label 的关系 |
|-----------|-----------------|--------------------|
| **MomentumExpansion** | 压缩后波动/区间扩张，放量突破 | ✅ 新写 `me_label`，检测压缩+扩张突破 |
| **AuctionExhaustionReversal** | 趋势末端衰竭（量/波动极值）后反转 | ✅ 可在 sr_reversal 基础上加「衰竭」过滤，或新写 `exhaustion_reversal_label` |
| **BreakoutPullbackContinuation** | 趋势中先回踩再延续原方向 | ❌ 需新写：trend_sign 定方向 + 回踩检测 + forward RR |
| **HTFBiasLTFEntry** | 大周期定方向，小周期定入场 | ❌ 需新写：HTF 趋势方向 + LTF 入场信号 + forward return |
| **FailedBreakoutFade** | 假突破（突破后失败）→ 反手 fade | ❌ 需新写：failed_breakout 检测（破高/低后收回）+ fade 方向 RR |
| **LiquiditySweepRejection** | 流动性扫损（sweep）后价格拒绝并反向 | ❌ 需新写：sweep 检测（wick 扫前高/低后收回）+ 反向 forward return |

**结论**：
- **全部新写**：ME、AER、BPC、HTF、FBF、LSR（6 个新 label 函数）
- 每个 archetype 对齐策略都有独立的 label 文件，便于维护和调参

### 策略总览

| 策略 | 全名 | Label 方案 | 优先级 |
|------|------|------------|--------|
| ME | MomentumExpansion | **新写** me_label | ⭐⭐⭐ |
| AER | AuctionExhaustionReversal | 扩展 sr_reversal | ⭐⭐⭐ |
| BPC | BreakoutPullbackContinuation | **新写** | ⭐⭐⭐⭐⭐ |
| HTF | HTFBiasLTFEntry | **新写** | ⭐⭐⭐⭐ |
| FBF | FailedBreakoutFade | **新写** | ⭐⭐⭐⭐ |
| LSR | LiquiditySweepRejection | **新写** | ⭐⭐⭐ |

### Step 1：可复用策略（ME, AER）

- [x] **S1.1** 创建 ME 策略目录结构 ✅
  - 位置：`config/strategies/me/`
  - labels.yaml 指向 `compression_breakout_label.py`
  - 必要时调整 params

- [x] **S1.2** 创建 AER 策略目录结构 ✅
  - 位置：`config/strategies/aer/`
  - 新写 `compute_exhaustion_reversal_label()` 或在 sr_reversal 基础上加衰竭过滤
  - 位置：`src/time_series_model/strategies/labels/exhaustion_reversal_label.py`

### Step 2：新 Label 策略（BPC, HTF, FBF, LSR）

- [x] **S2.1** BPC - BreakoutPullbackContinuation ✅
  - 新写 `compute_bpc_label()`
  - 逻辑：trend_sign 定方向 + 回踩检测 + forward RR
  - 位置：`src/time_series_model/strategies/labels/bpc_label.py`
  - 配置：`config/strategies/bpc/`

- [x] **S2.2** HTF - HTFBiasLTFEntry ✅
  - 新写 `compute_htf_label()`
  - 逻辑：HTF 趋势方向 + LTF 入场信号 + forward return
  - 位置：`src/time_series_model/strategies/labels/htf_label.py`
  - 配置：`config/strategies/htf/`

- [x] **S2.3** FBF - FailedBreakoutFade ✅
  - 新写 `compute_fbf_label()`
  - 逻辑：failed_breakout 检测（破高/低后收回）+ fade 方向 RR
  - 位置：`src/time_series_model/strategies/labels/fbf_label.py`
  - 配置：`config/strategies/fbf/`

- [x] **S2.4** LSR - LiquiditySweepRejection ✅
  - 新写 `compute_lsr_label()`
  - 逻辑：sweep 检测（wick 扫前高/低后收回）+ 反向 forward return
  - 位置：`src/time_series_model/strategies/labels/lsr_label.py`
  - 配置：`config/strategies/lsr/`

### Step 3：训练与验证

- [x] **S3.1** 为每个策略创建完整配置 ✅
  - labels.yaml, model.yaml, features.yaml, meta.yaml, backtest.yaml, evaluation.yaml

- [ ] **S3.2** 训练 6 个策略模型
  - 使用现有 `mlbot train` 流程
  - 检查 Sharpe / 命中率

- [ ] **S3.3** 对比 6 个策略与原 4 个策略
  - 确认语义对齐
  - 确认性能改进

---

## Outcome-Based Tree Labeling 方案实施 TODO（用于审计上述策略）

> 文档：`docs/strategies/OUTCOME_BASED_TREE_LABELING.md`

### Phase 1：基础设施

- [x] **1.1** 实现 `compute_forward_rr_label()` 函数 ✅
  - 位置：`src/time_series_model/strategies/labels/outcome_based_label.py`
  - 支持 direction = 'long' / 'short'
  - 支持 label_meta 元信息输出
  - 额外实现：`compute_delta_rr()`, `extract_negative_leaves()`, `validate_rule_stability()`

- [x] **1.2** 创建 outcome_audit 策略配置 ✅
  - 位置：`config/strategies/outcome_audit/`
  - 包含：labels.yaml, model.yaml, features.yaml, meta.yaml, evaluation.yaml
  - 树配置：max_depth=3, min_data_in_leaf=500

### Phase 2：负规则导出

- [x] **2.1** 实现 `extract_negative_rules.py` 脚本 ✅
  - 位置：`scripts/extract_negative_rules.py`
  - 条件：mean_rr < -0.3, delta_rr < -0.2, coverage > 2%
  - 支持从 LightGBM 模型提取树规则

- [x] **2.2** 实现 `validate_rule_stability()` 函数 ✅
  - 时间切片稳定性检验
  - 阈值扰动稳定性检验
  - 输出 veto_level: hard / soft / discard

### Phase 3：BPC Archetype 审计

- [x] **3.1** 创建 BPC 审计配置 ✅
  - 位置：`config/archetypes/bpc/`
  - 定义 7 条可证伪假设 (hypotheses.yaml)
  - 假设 → 特征分组映射
  - 审计流程配置 (audit.yaml)
  - Gate 配置 (gate.yaml)

- [ ] **3.2** BPC Long Dataset 审计
  - 训练 forward_rr_long 浅树
  - 导出负规则
  - 映射到假设 1-7

- [ ] **3.3** BPC Short Dataset 审计
  - 训练 forward_rr_short 浅树
  - 导出负规则
  - 映射到假设 1-7

- [ ] **3.4** 生成 BPC 审计报告
  - 哪些假设被否定
  - veto_hard / veto_soft 规则列表
  - archetype_health 评估

### Phase 4：Gate 写回

- [ ] **4.1** 创建 BPC gate 配置
  - 位置：`config/archetypes/bpc/gate.yaml`
  - 写入通过稳定性检验的负规则
  - 标注 hypothesis 来源

- [ ] **4.2** 实现 Gate 执行逻辑
  - veto_hard → 绝对禁止
  - veto_soft → 仓位减半

### Phase 5：其他 Archetype 审计（可选）

- [ ] **5.1** ME (MomentumExpansion) 审计
- [ ] **5.2** AER (AuctionExhaustionReversal) 审计
- [ ] **5.3** HTF (HTFBiasLTFEntry) 审计
- [ ] **5.4** FBF (FailedBreakoutFade) 审计
- [ ] **5.5** LSR (LiquiditySweepRejection) 审计

---

### 验收标准

1. ✅ forward_rr 标签携带完整 label_meta
2. ✅ 负规则筛选满足 delta_rr < -0.2（对照组差异）
3. ✅ veto 分级与 gate 语义强绑定
4. ✅ 每条负规则可追溯到被否定的假设
5. ✅ 审计报告输出格式标准化

已完成的任务
1. 6 个 Archetype 对齐策略创建 ✅
策略	Label 文件	配置目录	状态
ME	复用 compression_breakout	config/strategies/me/	✅
AER	exhaustion_reversal_label.py	config/strategies/aer/	✅
BPC	bpc_label.py	config/strategies/bpc/	✅
HTF	htf_label.py	config/strategies/htf/	✅
FBF	fbf_label.py	config/strategies/fbf/	✅
LSR	lsr_label.py	config/strategies/lsr/	✅
2. Outcome-Based Tree Labeling 方案 ✅
Phase 1: 基础设施
outcome_based_label.py - compute_forward_rr_label(), compute_delta_rr(), validate_rule_stability()
config/strategies/outcome_audit/ - 5 个配置文件
Phase 2: 负规则导出
extract_negative_rules.py - 完整脚本
Phase 3: BPC Archetype 审计配置
hypotheses.yaml - 7 条可证伪假设
audit.yaml - 审计流程
gate.yaml - Gate 配置
3. 待执行任务（需要数据）
TODO 中的以下任务需要实际数据才能执行：
S3.2 训练 6 个策略模型
3.2-3.4 BPC 审计执行
Phase 4 Gate 写回
Phase 5 其他 Archetype 审计

# 查看 BTCUSDT 2023 年 parquet 文件数量
ls /home/yin/trading/ml_trading_bot/data/parquet_data/BTCUSDT*2023*.parquet | wc -l

# 应该看到 12 个文件（01-12月）才是完整的

