# ML Trading Bot TODO

---

## 📋 Evidence V2 架构升级 TODOLIST

> **核心目标**：Gate（硬） + Evidence（软评分） + Execution（分档）
>
> **参考文档**：
> - `docs/architecture/EVIDENCE_ARCHITECTURE_V2.md`
> - `docs/architecture/FAILURE_TO_RETURN_PIPELINE.md`
>
> **配置目录**：`config/nnmultihead/archetypes/<archetype>/`

### 阶段零：基础设施（前置依赖） ✅

| 序号 | 任务 | 输入 | 输出 | 状态 |
|------|------|------|------|------|
| **0.1** | 创建 archetypes 目录结构 | 无 | `config/nnmultihead/archetypes/{bpc,htf,...}/` | ✅ |
| **0.2** | `export-rules` 增加 `--generate-risk-gate` | `bpc_tree_rules.md` | `risk_gate_draft.yaml`（含语义注释） | ✅ |
| **0.3** | 输出路径统一到 `results/train_final_<时间戳>/` | 无 | HTML + 规则 + 模型在同一目录 | ✅ |

### 阶段一：研究轨（树模型）— 规则发现与导出

| 序号 | 任务 | 输入 | 输出 | 验证标准 |
|------|------|------|------|----------|
| **1.1** | 完成 Failure Audit 封口 | 当前模型 + failure labels | lift 分析报告 | 确认 failure 无治理空间 |
| **1.2** | 构造 GOOD 样本空间 | trades + failure_any | GOOD 数据集 | `~failure_any` 样本占比 |
| **1.3** | 训练 Return Tree | GOOD 样本 + realized_rr | RR 关键条件 | 高频分裂特征列表 |
| **1.4** | 导出 Evidence 轴候选 | Return Tree 分裂点 | 各轴 quantile_mapping 建议 | 每轴 3-5 个断点 |

### 阶段二：分层模型 — Evidence 软化

| 序号 | 任务 | 输入 | 输出 | 验证标准 |
|------|------|------|------|----------|
| **2.1** | 设计 Evidence Axis 配置格式 | 架构文档 | `evidence_axes.yaml` schema | 支持 piecewise/quantile mapping |
| **2.2** | 实现 Evidence 评分函数 | axis 配置 | `src/.../evidence_scorer.py` | 单元测试通过 |
| **2.3** | 实现 Evidence 聚合器 | 各轴 score + 权重 | `overall_score ∈ [0, 1]` | 加权求和正确 |
| **2.4** | 为 BPC 创建 evidence_axes.yaml | 1.4 的输出 | `config/nnmultihead/archetypes/bpc/evidence_axes.yaml` | 配置可解析 |

### 阶段三：分层模型 — Execution 分档

| 序号 | 任务 | 输入 | 输出 | 验证标准 |
|------|------|------|------|----------|
| **3.1** | 设计 Execution Tier 配置格式 | 架构文档 | `execution_tiers.yaml` schema | 支持 3-5 档 |
| **3.2** | 实现 Tier 选择器 | evidence_score + tier 配置 | `src/.../tier_selector.py` | 根据 score 返回正确档位 |
| **3.3** | 为 BPC 创建 execution_tiers.yaml | 初始参数估计 | `config/nnmultihead/archetypes/bpc/execution_tiers.yaml` | 配置可解析 |

### 阶段四：参数优化 — 平坦高原搜索

| 序号 | 任务 | 输入 | 输出 | 验证标准 |
|------|------|------|------|----------|
| **4.1** | 新写 `optimize_evidence_plateau.py` | 无 | `scripts/optimize_evidence_plateau.py` | 脚本可运行 |
| **4.2** | Evidence 参数搜索 | GOOD 样本 + axis 参数空间 | 最优轴权重 | Sharpe 提升 |
| **4.3** | Execution 参数搜索 | GOOD 样本 + tier 参数空间 | 最优档位参数 | 高原区域稳定 |
| **4.4** | 联合微调（可选） | 4.2 + 4.3 附近 | 最终参数 | 参数变化 ±10% 结果稳定 |

### 阶段五：集成与迁移

| 序号 | 任务 | 输入 | 输出 | 验证标准 |
|------|------|------|------|----------|
| **5.1** | 集成 Evidence + Execution 到回测 | 评分器 + 选择器 | 可回测的完整流程 | 回测结果合理 |
| **5.2** | 修改 `tree_gate.py` 加载新目录 | 新目录结构 | 更新后的加载逻辑 | 能加载 archetypes 子目录 |
| **5.3** | 更新 `execution_archetypes.yaml` 引用 | BPC 配置路径 | 添加 `config_path` 字段 | 配置可解析 |
| **5.4** | 扩展到其他 Archetype | BPC 模板 | HTF/ME/FBF/LSR/AER 配置 | 各 archetype 独立可运行 |

### 依赖关系

```
0.1 → 0.2 → 0.3  (前置基础设施)
         ↓
1.1 → 1.2 → 1.3 → 1.4  (树模型规则发现)
                    ↓
              2.1 → 2.2 → 2.3 → 2.4  (Evidence 软化)
                              ↓
                        3.1 → 3.2 → 3.3  (Execution 分档)
                                    ↓
                        4.1 → 4.2 → 4.3 → 4.4  (平坦高原搜索)
                                          ↓
                              5.1 → 5.2 → 5.3 → 5.4  (集成迁移)
```

### 当前起点

**阶段零已完成** ✅，从 **1.1 Failure Audit 封口** 开始。

> 注：阶段一需要训练数据，请先确保 feature_store 已准备好。

---

## 历史 TODO（已完成）

### 6 个 Archetype 对齐策略创建 ✅

| 策略 | Label 文件 | 配置目录 | 状态 |
|------|-----------|---------|------|
| ME | 复用 compression_breakout | config/strategies/me/ | ✅ |
| AER | exhaustion_reversal_label.py | config/strategies/aer/ | ✅ |
| BPC | bpc_label.py | config/strategies/bpc/ | ✅ |
| HTF | htf_label.py | config/strategies/htf/ | ✅ |
| FBF | fbf_label.py | config/strategies/fbf/ | ✅ |
| LSR | lsr_label.py | config/strategies/lsr/ | ✅ |

### Outcome-Based Tree Labeling 方案 ✅

- Phase 1: 基础设施 ✅
- Phase 2: 负规则导出 ✅
- Phase 3: BPC Archetype 审计配置 ✅

### 待执行任务（需要数据）

- S3.2 训练 6 个策略模型
- 3.2-3.4 BPC 审计执行
- Phase 4 Gate 写回
- Phase 5 其他 Archetype 审计
