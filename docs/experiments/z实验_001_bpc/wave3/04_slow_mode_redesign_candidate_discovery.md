# 04 — 慢管线改造方案：候选发现器

> **状态**：设计文档。本次 Wave 3 **不实现**，仅记录方案。实际实施时单开 PR。

## 目标

把 slow 模式从 "自动落地决策者" 降级为 "候选发现器"：

- **不再**自动写入 `config/strategies/<stg>/gate_draft.yaml` / `prefilter.yaml`
- **转而**产出结构化候选报告到 `results/candidates/<stg>_<timestamp>/`
- **引入**多方法并列运行 + 共识矩阵，增加候选信号置信度
- **保留** Wave 2-E 的单方法 decision mode 作为 escape hatch（默认关闭）

## 产出格式

### 目录结构（每次 slow 跑一次产生一个）

```
results/candidates/bpc_20260501_120000/
├── candidate_report.md             ← 人审入口
├── gate_candidates.yaml            ← 各 method 生成的 gate 规则集
├── prefilter_candidates.yaml       ← 各 method 生成的 prefilter 规则集
├── entry_filter_candidates.yaml    ← 各 method 生成的 EF 规则集
└── raw_scores/
    ├── distribution_ks.json
    ├── mean_effect.json
    ├── upside_positive_rate_ratio.json
    └── tail_bad_rate_ratio.json
```

### candidate_report.md 模板

```markdown
# BPC Candidate Report — 2026-05-01

## 评估窗口
- Train: 2025-02 ~ 2026-01 (12 mo)
- Val:   2026-02 ~ 2026-04 (3 mo)
- Test:  2026-05 (1 mo)
- 样本量: train 8640 bar / val 2160 bar / test 720 bar

## Prefilter 候选

### 共识矩阵 (候选特征 × scoring methods)

| 特征                     | ks  | mean | upside | tail | 共识度 | 当前状态          | 建议     |
|--------------------------|:---:|:----:|:------:|:----:|:------:|-------------------|----------|
| bpc_semantic_chop        |  ✓  |   ✓  |   ✓    |  ✓   |  4/4   | locked            | 保持     |
| ema_1200_position        |  ✓  |   ✓  |   -    |  -   |  2/4   | locked            | 保持     |
| bb_width_normalized_pct  |  ✓  |   ✓  |   ✓    |  -   |  3/4   | 未 locked         | **新候选, 上 A/B** |
| vpin_quantile_rank_20    |  -  |   -  |   ✓    |  -   |  1/4   | 未 locked         | 跳过 (疑 overfit) |
| bpc_impulse_return_atr_long |  -  |   -  |   -    |  -   |  0/4   | locked            | **考虑退役** |

### 退役候选
- `bpc_impulse_return_atr_long`: 0/4 方法选中, Allow mean_rr 同比下降 0.3 → 建议下次 rebuild 时人工验证是否移除

### 各方法详细
（见 `raw_scores/*.json`）

#### distribution_ks Top 5
1. `bpc_semantic_chop`: KS=0.42, p=0.001, effect=0.8
2. ...

## Gate 候选
（相同格式）

## Entry Filter 候选
（相同格式）

## 人审建议
- **高优先级** (共识 ≥ 3 + 语义清晰): bb_width_normalized_pct
- **中优先级** (共识 2 + 和已有特征不重复): (无)
- **低优先级 / 跳过** (共识 1 + 分数异常高): vpin_quantile_rank_20

## 运行信息
- 耗时: 28 min
- 代码状态: git commit abc1234
```

## YAML 配置演化

### 当前（Wave 2-E 后）
```yaml
meta_algo:
  prefilter:
    scoring_method_fallbacks:       # list 形式，但实际代码 "挑 winner"
      - distribution_ks
    archetype_scoring_method: distribution_ks  # 单一方法
  gate:
    archetype_scoring_method: tail_bad_rate_ratio
  entry_filter:
    archetype_scoring_method: upside_positive_rate_ratio
```

### 改造后
```yaml
meta_algo:
  mode: candidate                    # candidate | decision
  
  prefilter:
    scoring_methods:                 # candidate 模式: list 全跑并列
      - distribution_ks
      - mean_effect
      - upside_positive_rate_ratio
      - tail_bad_rate_ratio
    decision_method: distribution_ks # decision 模式: 单方法 (Wave 2-E 保留)
  
  gate:
    scoring_methods:
      - distribution_ks
      - mean_effect
      - tail_bad_rate_ratio
    decision_method: tail_bad_rate_ratio
  
  entry_filter:
    scoring_methods:
      - upside_positive_rate_ratio
      - mean_effect
    decision_method: upside_positive_rate_ratio
  
  consensus_threshold: 2             # 共识度 ≥ 2 进入候选报告
```

### 命令行 flag
```bash
# 默认: candidate mode, 不落地
python scripts/auto_research_pipeline.py --strategy bpc --stage slow_mode

# 显式切到 decision mode (Wave 2-E 行为, escape hatch)
python scripts/auto_research_pipeline.py --strategy bpc --stage slow_mode \
    --meta-algo-mode decision --apply-candidates
```

双开关（`--meta-algo-mode` + `--apply-candidates`）防止误触发自动落地。

## 代码改造量估计

### 涉及文件
- [`scripts/auto_research_pipeline.py`](../../scripts/auto_research_pipeline.py) — 落盘路径 + flag 解析
- [`scripts/meta_algorithm_unified.py`](../../scripts/meta_algorithm_unified.py) — multi-method 并列 + 共识矩阵
- [`scripts/optimize_gate_unified.py`](../../scripts/optimize_gate_unified.py) — Gate 也接 multi-method
- [`scripts/analyze_archetype_feature_stratification.py`](../../scripts/analyze_archetype_feature_stratification.py) — Prefilter 也接 multi-method
- 新增: `scripts/generate_candidate_report.py` — 共识矩阵 markdown 生成

### 估算

| 模块 | 行数估计 | 说明 |
|---|---|---|
| 落盘路径改写 | ~50 | 加一个 `output_base_dir` 参数, candidate mode 走 `results/candidates/...`, decision mode 走 `config/strategies/...` |
| Multi-method 并列执行 | ~30 | 把现有 `if archetype_scoring_method == "ks": ...` 改成 `for method in scoring_methods: results[method] = run(method)` |
| 共识矩阵生成器 | ~80 | 新 helper: 聚合 N 个 method 的 top-K 特征列表, 计算共识度, 生成 markdown 表 |
| `--apply-candidates` flag + mode 判断 | ~10 | argparse + 条件分支 |
| 单元测试 | ~100 | 覆盖 mode 切换、consensus 计算、落盘路径 |

**总计 ~170 行生产代码 + 100 行测试，1~2 个工作日**。

## 实施步骤（建议）

1. **Phase 1: 不改行为，只加选项**
   - 加 `--meta-algo-mode` flag，默认 `decision`（=当前行为）
   - 加 `results/candidates/` 目录生成逻辑（即使 decision mode 也可选输出）
   - 验证 decision mode 行为与现状 bit-identical

2. **Phase 2: multi-method 并列**
   - `meta_algorithm_unified.py` 加 for 循环跑多 method
   - 每个 method 结果存到 `raw_scores/<method>.json`
   - decision mode 仍只采 `decision_method` 的结果落地

3. **Phase 3: 共识矩阵 + 报告**
   - 新写 `generate_candidate_report.py`
   - 聚合 raw_scores → markdown
   - 不改 decision mode 行为

4. **Phase 4: 切换默认**
   - `--meta-algo-mode` 默认从 `decision` 改为 `candidate`
   - 用户需显式传 `--apply-candidates` 才会落地
   - 文档更新

每个 Phase 独立可验证，不动现有 pipeline 的默认行为，避免 Wave 3 类事故（改动一上线就影响所有策略）。

## 和 Wave 3 已建基础设施的对接

本次 Wave 3 留下的两样基础设施在改造时直接复用：

### `env_extra` 机制
- 改造过程中的灰度测试可用 `META_ALGO_MODE=candidate` 环境变量触发
- 不污染 config，单次实验干净

### `compare_monthly_pnl.py`
- 改造后慢管线候选进入快管线 A/B 时，直接用它做裁决
- 不用再造轮子

## 不做什么

以下方向**明确不做**（有理由的放弃）：
- ❌ 加 Purged K-fold：详见 [02](02_meta_findings_on_meta_algo.md) §3
- ❌ 让慢管线自动改 `is_good` label 语义：详见 [02](02_meta_findings_on_meta_algo.md) §6
- ❌ 给慢管线加更多 scoring method（除非新方法对应明确的策略语义维度）
- ❌ 月度触发慢管线：每 3 月足够，月度增量价值小而噪声大

## 验收标准

改造完成的标志：

1. 默认 slow 模式运行后，`config/strategies/<stg>/` **无文件变更**
2. `results/candidates/<stg>_<ts>/` 产出完整 5 个文件（4 yaml + 1 md + raw_scores/）
3. 候选报告里共识矩阵、退役候选、人审建议三节齐全
4. `--apply-candidates` flag 开启时，行为与现在 Wave 2-E 一致（bit-identical）
5. 单元测试 coverage ≥ 80%
