# 树模型 vs 规则类：特征与过拟合

本轮两轨验证的一个副产品，是把「树比规则难」具体化成了可操作的检查清单。

## 为什么树往往比规则难

| 维度 | 规则类（BPC/TPC gate 等） | 树 ranker（fast_scalp） |
|------|---------------------------|-------------------------|
| 假设空间 | 人工限定（几条语义规则） | 特征 × 分裂 × 深度，极大 |
| 标签对齐 | 规则直接对应执行语义 | 120T label 与 1min event 常有 gap |
| 验证路径 | 规则 plateau 多在 holdout IC / 单调性 | 容易在 **vector τ-scan** 上「看起来还行」，event 却亏 |
| 过拟合形态 | 参数边界、样本少 | **样本内四段全正、唯一 OOS 段转负**（G7 双 head 即此） |

规则难在「找对语义」；树难在「找对特征 + 控制容量 + 用对 OOS 判据」。

## 本轮实证（20260602）

| 现象 | 解读 |
|------|------|
| G7 样本内 +12% / +22% / +19%，**OOS recent_6m −3.75%** | 典型过拟合：多 head、同特征、训练窗内好看 |
| G3 H=3 short **OOS +12.02%** | 更简单假设 + 同一 IC 池，OOS 反而最好 |
| g5-label vector τ-scan 全负，event 四段全负 | 不是「特征不够」，是 **label 与 ranker 无 edge** |
| gate IC-prune 8 特征，`adverse_avoided=0.145` | **第二层 gate 可以学**，但救不了无 edge 的 ranker（G16 OOS +3.61% < G3 +12%） |

**结论：** 先证明 ranker 在 **唯一干净 OOS**（`recent_6m_oos`）有正 edge，再叠 gate / 双 head；否则是在给噪声加复杂度。

## 防过拟合实践（本仓库约定）

### 1. 特征选择：只信 holdout IC prune

- 脚本：`PYTHONPATH=.:src python scripts/research/ic_prune.py`
- 参数：`--holdout-start` / `--holdout-end` 与训练 holdout **一致**
- 上限：`--top-n-columns 20`（或更少）；**禁止**把 train 段 IC 写回 deploy
- 泄漏：label、forward_rr、realized_r_* 不得进 `model_features.yaml`（ic_screen 应已挡）

### 2. 模型容量：浅树 + 少特征

- `model.yaml` 里 depth/leaves 保持保守；特征越多、树越深，越易吃掉样本内噪声
- 双 head = 两个分类器共享特征 → **有效容量翻倍**，本轮已证伪，除非有强正则或更少特征

### 3. 判据：event segment > vector τ-scan

- τ-scan 在 holdout 上全负（g5-label）时，**不要**硬找 plateau promote
- **Promote 门禁**：以 `recent_6m_oos` 为主；bear/bull/range 仅作稳健性，且须标明是否在 gate/head 训练窗内

### 4. Gate 与 entry 特征分离

- Gate：**宽候选池** → IC + 真实 MAE lift → 浅树（见 `train_tree_adverse_gate.py`）
- Entry ranker 与 gate **不要共用同一套 IC 写回**；gate 特征应偏波动/尾部（evt、vol_accel），与 score 正交
- Export 注入 parquet 必须含 **IC 选中的 gate 列**（见 `TRAINING.md` §3.5 `--save-predictions` + 重建 extra_cols）

### 5. 全历史 predictions（gate / dual head 训练）

- `train_*/predictions.parquet` **仅 holdout**；gate 与 dual head 必须用 `--save-predictions` 全历史表
- Gate 的 `--long-entry/--short-entry` 必须与 ranker τ-scan **一致**

## 建议的下一步（在 G3 正 edge 上）

1. **G3 + adverse gate（OOS）**：用 H=3 全历史 score + 同一 gate 管线，看 recent_6m 是否 **> +12.02%**（净增益）
2. **特征 ablation**：固定 20 列 IC 池，每次减 5 列或按 |IC| 递减，看 OOS 是否单调变差（防「多特征 = 更好」错觉）
3. **不急着加 head**：除非有明确结构假设（如 long/short 标签噪声不同），否则优先单 signed 回归 + 执行对齐 label 对照

规则类策略的「难」在语义；树模型的「难」在 **用 holdout 约束搜索空间，并用 event OOS 一票否决样本内幻觉**。

## 树在 ABC 中的职责（doctrine，2026-06）

**树不做第四条平行策略线**；嵌进 B/C 流水线，只做 **排序** 与 **否决**：

| 用树 | 用规则 |
|------|--------|
| entry score（G3）、adverse gate（G18） | entry 语义（TPC E2）、prefilter/regime |
| 在「已放行」bar 上 top-q | 能不能做、做哪一侧、防追高 |

口诀：**规则定语义，树定先后/否决。** 详见 [`docs/strategy/短期树独立策略_设计与落地_CN.md`](../../docs/strategy/短期树独立策略_设计与落地_CN.md) §1.4。
