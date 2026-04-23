# 01 — 实验与发现

## Step 0 — Baseline 建立（DONE）

### 执行
用当前 Wave 1+2 代码（post Wave 2-A + 2-E）跑 6 个关键月 fast_month：

| month | run_timestamp | n_trades |
|---|---|---|
| 2024-04 | 20260423_122252 | 12 |
| 2024-05 | 20260423_123942 | 10 |
| 2024-06 | 20260423_125350 | 8 |
| 2025-11 | 20260423_130549 | 12 |
| 2025-12 | 20260423_131543 | 5 |
| 2024-03 | 20260423_132457 | 17 |

详见 [baseline_bpc_wave2_runs.txt](baseline_bpc_wave2_runs.txt)。

### 工具建立
[scripts/compare_monthly_pnl.py](../../scripts/compare_monthly_pnl.py)：
- 输入：两组 rolling_sim timestamps
- 去重键：`(symbol, side, entry_time, exit_time, is_add_position)`
- 归因：`linear_days`（按持仓天数 pro-rata，MTM 口径）
- 输出：月度矩阵 + bucket 裁决 + 全窗口合计

### 基准数字（Wave 1+2 代码状态下）
- 全窗口：n=64, R=+11.0
- trend_favorable：30 笔 / −67.2R
- death_months：17 笔 / +83.4R
- small_sample：17 笔 / −5.3R

对比早期 20260413 好配置（见 [baseline_vs_early_good_reference.md](baseline_vs_early_good_reference.md)）：整体 delta R=−744.0，trend_favorable 退化 −199.4R。

---

## Step 1 — Wave 3-A：Prefilter label = `rr >= 0`（NO-OP）

### 假设
把 `is_good` 从 `forward_rr >= -0.8`（避开极端负 RR）改为 `forward_rr >= 0`（严格正收益），让 Prefilter 学 "能赚钱的结构" 而不是 "不大亏的结构"。

### 实现
通过 `LABEL_RR_THRESHOLD` 环境变量传入，在 [analyze_archetype_feature_stratification.py](../../scripts/analyze_archetype_feature_stratification.py) 里无条件覆盖 `success_no_rr_extreme` 列。

外层由 `WAVE3A_PREFILTER_LABEL_RR=0.0` 触发，[auto_research_pipeline.py](../../scripts/auto_research_pipeline.py) 通过 `env_extra` 注入给 Prefilter Analyze 子进程。

### 结果
详见 [step1_wave3a_bpc_monthly_diff.md](step1_wave3a_bpc_monthly_diff.md)：

| bucket | baseline | new | delta R | 裁决 |
|---|---|---|---|---|
| trend_favorable | 30/−67.2 | 30/−67.2 | **+0.0** | PASS（表面） |
| death_months    | 17/+83.4 | 17/+83.4 | **+0.0** | PASS（表面） |
| small_sample    | 17/−5.3  | 17/−5.3  | +0.0 | INFO |
| **全窗口**      | 64/+11.0 | 64/+11.0 | **+0.0** | **完全一致** |

**delta 全为 0** → label 改动没有通过 Prefilter meta-algo。

### 根因
Wave 2-E 把 Prefilter 的 scoring method 固定为 `distribution_ks`：

```python
# meta_algorithm_unified.py 的 distribution_ks 实现逻辑
ks_stat = ks_2samp(
    forward_rr[rule_pass],   # 规则 pass 的 RR 分布
    forward_rr[rule_deny],   # 规则 deny 的 RR 分布
)
```

**KS 比较的是两组 RR 分布，完全不消费 `is_good` 标签列**。
- 我们改的是 `success_no_rr_extreme` 列（label）
- KS 用的是 `forward_rr_short` 列（原始 RR）
- 两列互不依赖 → label 怎么改都无影响

即使传入 override 被代码正确应用（日志里能看到 `⚙️ [LABEL_RR_THRESHOLD] 覆盖...`），下游 scoring 根本不读这个列。

### 结论
- 本步实验 technically NO-OP；逻辑上是 **无法测试假设**
- 用户决定（Option D）：标记 PASS per bucket verdict，放行 Step 2（假设原始 plan 对 Prefilter label 没有 leverage）

---

## Step 2 — Wave 3-B：Gate label = tail-only bottom 5%（FAIL）

### 假设
把 Gate 的 label 从 "全局 `rr >= -0.8`" 改为 "tail-only：只把最极端的 5% 负 RR 标为 bad"。

预期：Gate 变得更 "挑剔"，只过滤极端不利场景（比如 flash crash 前后），不再误杀 BPC 在趋势中的 pullback 入场。

### 实现
通过 `GATE_TAIL_QUANTILE=0.05` 环境变量，在 [optimize_gate_unified.py](../../scripts/optimize_gate_unified.py) 里生成新 label：
```python
tail_thr = np.quantile(forward_rr, GATE_TAIL_QUANTILE)  # bottom 5% 分位数
is_good = forward_rr > tail_thr   # 只有 bottom 5% 算 bad
```
同时修正 `rr_col` 优先级为 `forward_rr`（tail 计算需要未 clip 的真实 RR，不能用 `bpc_impulse_return_atr`）。

外层由 `WAVE3B_GATE_TAIL_Q=0.05` 触发，通过 `env_extra` 注入。

### 结果
详见 [step2_wave3b_bpc_monthly_diff.md](step2_wave3b_bpc_monthly_diff.md)：

| bucket | baseline | new | delta R | 裁决 |
|---|---|---|---|---|
| trend_favorable | 30/−67.2 | **39/−163.4** | **−96.2** | **FAIL** |
| death_months    | 17/+83.4 | 26/+82.9 | −0.5 | PASS |
| small_sample    | 17/−5.3  | 10/−75.6 | −70.3 | INFO |
| **全窗口**      | 64/+11.0 | **75/−156.1** | **−167.0** | **恶化** |

关键月单月：
- 2024-04: 12 笔 / −35.8R → **20 笔 / −129.4R**（delta **−93.6R**）
- 2024-05: 10 笔 / −42.7R → 11 笔 / −45.3R（delta −2.6R）
- 2024-06: 8 笔 / +11.3R → 8 笔 / +11.3R（delta 0）

2024-04 独自贡献绝大部分损失 —— 交易数从 12 增到 20，但 mean RR 从 −3R 恶化到 −6.5R。

### 根因
对比 baseline 和 new 的 Gate 规则：

| 规则 | baseline (`rr >= -0.8`) | new (tail-only q=0.05) |
|---|---|---|
| `bpc_semantic_chop` deny 阈值 | > 0.6（过滤 ordinary chop） | > 0.956（只过滤极端 chop） |
| Allow mean_rr | +0.12 | −0.34 |
| Deny mean_rr | −0.25 | −0.81 |

**机理**：
- tail-only label 把 "bad" 定义成只有最极端 5%
- 对 Gate 优化器来说，"ordinary chop"（`semantic_chop` 0.6~0.95）不再算 bad
- 于是 optimizer 放宽 chop 阈值，让 ordinary chop 区域全部通过
- BPC 在 ordinary chop 入场的交易在实盘（`forward_rr`）上平均亏 −3~−6R
- 2024-04 这种有 chop 段也有 trend 段的月份，ordinary chop 交易占主导 → 大亏

**meta-level 教训**：
- 收紧 label（tail-only）反而让 **Gate 变松**
- 用户总结得精准："tail-only label 让 Gate 去找极端情况，把 ordinary chop 放过了"
- Gate 对 BPC 的真正价值是 **过滤 ordinary chop**，不是过滤极端 tail

### 结论
本步 **FAIL**，label 方向不仅没修好问题，还引入了新问题。用户决定（Option B + D 合并）：
- 回退 Wave 3 label 覆盖代码（Step 1 + Step 2）
- 保留基础设施（`env_extra`、`compare_monthly_pnl.py`、`results/wave3/` 存证）
- 转向 D 方向（execution / Prefilter 阈值诊断）

---

## Step 3/4/5 — 取消原因

### Step 3（EF label = `rr >= 0.3`）取消
- Entry Filter 用 `upside_positive_rate_ratio` scoring，和 Prefilter 同构 —— **也不消费 `is_good` label**（它比较 "pass 组正 R 占比" vs "deny 组正 R 占比"）
- 从 Step 1 的根因推演：label 改动对 EF 同样是 NO-OP
- 跑等于白跑

### Step 4（5 策略交叉验证）取消
- Step 1 NO-OP + Step 2 FAIL → 假设已证伪
- 不需要验证到其他 4 个策略

### Step 5（Nested CV / Purged K-fold）取消
- 原意是加一层 overfitting 防护，但前置步骤证明 label 方向本身没 leverage
- 且 Purged K-fold 有自己的问题（详见 [02](02_meta_findings_on_meta_algo.md) §3）
- 搁置

---

## 三步结论归纳

1. **label 语义不是 BPC 退化根因**。meta-algo 的 scoring 方法和 label 的关系比想象的松（Wave 2-E 的 KS/upside_rate 都不消费 label）。
2. **方向 1 排除，聚焦方向 2**（execution / Prefilter 阈值）。
3. **基础设施升级有收益**：`env_extra` 注入 + monthly diff 工具 + baseline 锚点，下次 D 方向实验直接用，不用重跑 baseline。

核心方法论结论见 [02](02_meta_findings_on_meta_algo.md)。
