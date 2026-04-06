## NN 多资产合约系统：宪法驱动架构与落地设计（v1）

本文把你现有系统（FeatureStore + NN path primitives + Router(3-action) + E2E/Shadow/Counterfactual）升级成一个**宪法驱动（Constitutional）**的可上线架构：**模型只产出证据（evidence）**，所有“自由度”都被关在可解释、可开关、可回滚的模块中（Router/Gate/Execution/Slot/kill-switch）。

> 关联背景与路线图：`docs/archive/guides/LIVE_TRADING_ROADMAP_MULTI_ASSET_CN.md`  
> 宪法与自由度理论：`docs/architecture/自由度限制.md`、`docs/architecture/自由度限制-归因-仓位和加仓.md`

---

## 0. 目标锁定（Training/Eval 的“不可变前提”）

### 0.1 交易目标（工程可验收版）
- **目标**：在固定 universe（TopN）与固定 bar（4H）下，形成一个可上线的“策略机器”：能跑、能监控、能回退，且 OOS 指标稳定。
- **非目标**：无法工程保证“1w→10w”。我们只能用宪法 + Gate + 运维把“归零概率”压低，并用证据链逐步放大风险预算。

### 0.2 冻结 6 个关键轴（任何实验必须声明）
每一次实验/部署都必须固定并记录（写入 artifacts meta）：
- **Universe**：symbols 列表 + 分组（HighCap/Alt/Meme 等）
- **时间窗**：train start/end；OOS start/end
- **时间粒度**：timeframe（例如 240T）
- **执行假设**：returns_source / cost / slippage / entry_delay
- **动作空间**：Router action space（v1 固定 3-action）
- **宪法版本**：position/execution/gate/detector 的 config + hash

---

## 1. Task 格式（TaskSpec / 可复盘产物）

### 1.1 TaskSpec（建议的最小字段）
每次 run 都绑定一个 `task_id`，并把 TaskSpec 写入 `{out}/meta.json`：
- **task_id**：稳定 ID（由 config + timeframe + horizon + 版本号 hash）
- **data**：symbols、timeframe、train/oos window、data_path、feature_store_root/layer
- **features**：feature contract（可选 blocks / missingness policy）
- **labels**：path primitives label config（horizon_bars、entry_offset、atr_col 等）
- **model**：MLP 超参（hidden/depth/dropout/lr/batch/epochs）
- **router**：rule thresholds（可解释旋钮）
- **gate**：gate policy（规则或树导出的规则）+ enable flags
- **execution/position constitution**：slot/replace/add/kill-switch/leverage 上限

> v1 最重要原则：**任何新增自由度必须：可开关、可统计贡献、可回滚**（见自由度宪法）。

---

## 2. 特征选择（Feature Contract + 可选 Block）

### 2.1 约束：宁愿少、也要稳定
- v1 推荐：**语义清晰、尺度稳定、跨币可比**的特征优先。
- 不建议把“复杂特征”（订单流/WPT/Hilbert/Mamba 表征）直接变成核心依赖；先作为 optional blocks（可一键关闭）。

### 2.2 Feature Contract（机制）
- **required**：模型必须依赖的最小集合（例如基础趋势/波动/结构）
- **optional_blocks**：可选组（例如 orderflow_*、hilbert_*、wpt_*）
- **missingness_policy**：缺块时如何处理（mask、block dropout 等）

---

## 3. NN 模型：Path Primitives Multi-Head（执行证据层）

### 3.1 模型输出（你现有设计）
多头输出用于“参与后怎么管仓位”（而不是“是否参与市场”）：
- `pred_dir_prob`：方向置信
- `pred_mfe_atr`：潜在利润空间（TP/上限）
- `pred_mae_atr`：潜在不利波动（SL/风险）
- `pred_t_to_mfe`：时间维度（time stop）

### 3.2 训练目标与数据构造
- **标签**：按每个 symbol 独立计算（避免跨币 horizon leakage）
- **训练目标**：不是直接最大化 PnL，而是拟合“路径形状”指标（更可迁移）
- **评估**：report 输出 head 的 AUC/Rank-IC/ICIR + 分层（例如 trend_high slice）

### 3.3 可选：Regime Heads（v2/v3）
Regime heads 属于“是否值得参与市场”的上游信息，但会改变标签构造成本。建议：
- v1：先用 Router + Gate（规则/树）完成参与控制
- v2：在 shadow 模式训练 regime heads，对比其是否减少阈值调参成本

### 3.4 如何解读“核心指标看起来不好”（非常重要）

很多人（包括你刚才的直觉）会把 `dir_auc / spearman / mae` 这类“全样本回归指标”当作模型是否有效的唯一判断。对 **Path Primitives + Router（阈值决策）** 这种系统来说，这是不完整的：

- **回归头不需要“数值拟合很准”也可能有用**：Router 只关心“有没有跨过阈值、排序是否把好样本推到上面”，而不是全样本的拟合误差。
- **正确的评估口径应该与 Router 对齐**：
  - **阈值一致的二分类评估**：例如用 `pred_mfe_atr` 排序去判断 `true_mfe_atr > mfe_min`，输出 AUC/PR（按 symbol + 汇总）。
  - **条件切片评估**：只在 Router 会交易的子集（MEAN/TREND）里看 IC/滚动漂移，而不是被大量 NO_TRADE 样本稀释。
- **`mask_rate` 的含义**：在当前实现里，`mask_rate` 是 `mfe_valid>0.5` 的比例，表示“有效样本占比”（不是“被 mask 的比例”）。

> 对应落地：我们已把上述两类评估接入到 `report.html`（见 “Threshold-consistent evaluation (Router-aligned)” 与 rolling preview 里的 `trade` 切片）。

---

## 4. Router：Rule Router（3-action，宪法下的“可控旋钮”）

### 4.1 定义
Router 将 primitives 映射到 `{NO_TRADE, MEAN, TREND}`，原则是：
- action space 固定
- 阈值少且可解释（可调但必须防过拟合）

### 4.2 阈值调参的“可泛化”验收
调到正 Sharpe ≠ 可预测。必须做：
- 多窗口 walk-forward
- 成本/滑点/延迟敏感性
- 单币贡献分解（避免单币撑起组合）
- 限制阈值自由度（防“炼丹”）

---

## 5. Gate：宪法驱动的“参与裁决层”（你提的 Tree Gate 可做）

Gate 的职责是：**在不引入不可控自由度的前提下**，减少错误参与（尤其尾部亏损/高换手/坏流动性）。

### 5.1 Gate 的输入与输出（统一接口）
- **输入**（只允许 t 时刻可得信息）：
  - features（结构/波动/订单流等）
  - primitives（dir/mfe/mae/ttm）
  - router 结果（mode）与关键阈值距离（margin）
- **输出**：
  - `allow: bool`
  - `reason: str`（必须可归因）
  - `severity: {info, warn, hard}`（便于降级策略）

### 5.2 用树模型训练 Gate，并“导出规则”（你的需求：可行）
可以做，推荐 v1 按以下路径落地：

#### 方案 A（推荐）：Tree 训练 + Surrogate Rule Distillation（低维护）
1) **定义 gate label**（只服务于“过滤坏交易”，不是做 alpha）：
   - 例：`bad_trade=1` 若在未来 H 根内出现 `DD > x` 或 `MAE > y` 或 `hit_stop_fast` 或 `cost_spike`  
   - 或：`improve_sharpe_proxy`/`reduce_tail` 的分段标签
2) 训练 LightGBM/XGBoost 得到 `p_bad` 或 `score_bad`
3) 用一个**小深度决策树**拟合 `p_bad`（surrogate tree），并限制深度/叶子数
4) 从 surrogate tree 导出 **YAML 规则**（if-else 阈值树），上线时只跑规则（稳定、可解释）
5) 离线做 ablation：`router_only` vs `router+gate_rules`（看 Sharpe/DD/turnover 的边际贡献）

#### 方案 B：直接训练可导出树（超低复杂度）
直接训练 `sklearn.DecisionTreeClassifier(max_depth<=3~5)`，得到天然可导出的规则树；缺点是表达力弱，但维护最低。

#### 关键宪法约束（必须遵守）
- Gate 必须 **可开关**、可回测“有/无”的差异
- Gate 默认 **shadow-only** 上线一段时间（先观测，不裁决）
- Gate 不允许“连续可调自由度爆炸”：规则数量、深度、输入特征必须受限

### 5.3 “树模型输入特征 + 路径原语”到底训练什么？（推荐：Gate/ExecutionRules，不训练 Router）

你问的核心是：树模型要不要加入？训练后要不要导出规则？导出后还要不要调阈值、找平坦高原？

这里给一个 **工程可落地且低维护** 的推荐结论：

#### 结论（v1 推荐）

- **Router**：继续用 rule thresholds（少量、可解释、可控）。  
  - Router 是“状态划分器”：把市场划成 `NO_TRADE/MEAN/TREND`，属于系统的 **可控旋钮**。
- **Tree 模型**：用于 **Gate/ExecutionRules**（刹车/开关/分档），而不是直接替代 Router。  
  - Tree 是“参与裁决器 / 质量评分器”：只在 Router 给出候选参与时，决定 `VETO / THROTTLE / ALLOW`，或对执行采取离散分档（如 size 档位、止盈止损档位）。

为什么不推荐“Tree 训练 Router”？
- Router 的职责是“可控地减少自由度”（动作空间固定、阈值少、可解释）。  
  让 Tree 直接决定 Router 等于把自由度从 7 个阈值升级成“树结构 + 叶子阈值 + 特征组合”，会更难归因、也更难做“平坦高原”的稳健验收。
- Tree 更适合做 **detector / veto**：尤其是尾部风险、坏流动性、假突破、拥挤交易等“坏样本过滤”。

#### Tree 的输入建议（统一数据合同，避免泄露）

Tree 输入建议用三类（全部是 t 时刻可得）：
- **features**：结构/波动/订单流/语义特征（可含 optional blocks）
- **path primitives**：`pred_dir_prob/pred_mfe_atr/pred_mae_atr/pred_t_to_mfe`（以及派生：`eff = mfe/(mae+eps)`、`dir_conf`）
- **router margin**：各阈值的 margin（距离阈值多远），用于让 gate 学“接近阈值的脆弱区”

Tree 的输出建议保持离散化（对应 `GateDecision` 宪法精神）：
- `VETO`（硬拒绝）
- `THROTTLE_25/50`（降低风险预算/减少交易频率）
- `ALLOW`（放行）

#### 是否要“导出规则”？（推荐：是，尤其上线）

- 训练阶段：可以用 LightGBM/XGBoost（表达强，便于快速迭代）
- 上线阶段：推荐导出/蒸馏成 **小深度规则树**（或显式 YAML 规则），原因：
  - **可审计**：线上到底依据什么拒绝/放行，一眼可看
  - **可回放一致**：离线回测与实盘逻辑一致（避免“训练模型版本”漂移）
  - **可控降级**：规则可一键关、或切换到更保守版本

> 你现在 repo 已经有 `TreeGate` 与规则导出能力（`src/time_series_model/gating/tree_gate.py`），属于正确方向：训练可以复杂，但线上必须简单。

#### 导出规则后还要不要调阈值？要，但目标变成“找平坦高原”

导出规则并不等于“从此不用调参”，而是把调参从“无限自由度”收敛到少数几个可控旋钮：
- **gate 的 operating point**：例如 `p_bad` 的 veto 阈值 / throttle 档位分界
- **Router 阈值**：仍然存在，但更稳定（因为 Gate 在过滤坏样本）
- **Execution 档位阈值**：例如 SL/TP/时间止损的离散档位边界

关键是：这些阈值都应该按 “平坦高原（plateau）” 思路选，而不是追尖峰：
- 多窗口一致（walk-forward）
- bootstrap 稳健性
- 局部敏感性（阈值微调不应导致性能剧烈变化）
- 约束优先（先满足宪法/成本/回撤门槛，再看收益）

#### 与 `docs/architecture/6种对称策略的启发式规则.md` 的关系（推荐落地顺序）

v1 的落地顺序建议：
1) **Execution archetype 先用启发式规则**（6 种对称 archetype 的必要条件/证据链先固化）  
2) Tree 先训练 **Gate**（过滤坏样本）  
3) 再逐步把“启发式条件中最脆弱的部分”交给树模型做 **规则补丁**（仍需导出规则 + plateau tuning 验收）

---

## 6. Execution：宪法驱动的执行层（风险→名义仓位→订单）

### 6.1 目标
把 `slot_risk` 变成可执行的合约张数/名义仓位，并确保：
- 最坏亏损不超过 `risk_per_slot`
- 杠杆上限（notional leverage）受控
- 止损结构简单（v1 推荐 shared stop）

### 6.2 “硬止损 + 软监控”
参考你文档的建议：
- **硬止损**：交易所挂单略宽于系统 stop（+ε）
- **软监控**：临近 stop 时提前市价退出，避免最差点成交

---

## 7. Position Constitution：Slot / Replace / Add（最重要的宪法层）

### 7.1 Slot（坑位）宪法（v1 建议）
- `max_slots = 2`
- `risk_per_slot = 0.015`（可在 1%~2% 内选；你文档推荐 1.5%）
- `max_total_risk = max_slots * risk_per_slot`

### 7.2 Replacement Judge（替换三关 + 单维保守优势）
- 替换必须“有罪”（必须输出失败原因）
- 单维保守比较：
  - `E[R]_new > Remaining_R_old * beta_advantage`
  - `beta_advantage >= 1.2`（强保守）

### 7.3 Add Position（加仓宪法：Trend-only，风险不增加）
- `allowed_path_types: [trend]`
- `max_adds_per_slot: 1`（v1）
- `require_locked_R: 0.5`（燃料来自已锁定浮盈）
- `shared_stop: true`
- 约束：`worst_loss(existing + add) <= risk_per_slot`

### 7.4 Symbol-level Max Cap（解决 ETH drag 的宪法工具）
不看模型、不看分组：
- `BTC max = 30%`
- `ETH max = 20%`
- `SOL max = 20%`
- `others total <= 30%`

---

## 8. 运维与治理：监控、报告、回退降级

### 8.1 报告（每次 OOS run 必须产出）
- **数据覆盖**：bars/month、缺失率
- **primitives report**：AUC/IC/ICIR + 分层
- **router report**：mode 分布、交易次数、阈值边际
- **gate report**：allow/deny 分布、原因 top-k、贡献对比（ablation）
- **execution control**：turnover/cost/DD/NaN 异常 + kill-switch

### 8.4 产物补全（从 `model.pt + FeatureStore` 重新生成 report/metrics）

有时训练过程会出现这种情况：**`model.pt` 已经落盘，但 `meta.json / metrics.json / report.html` 没写出来**（例如训练结束后在 evaluation/report 阶段被中断、或者你更新了 report 渲染逻辑想重渲染）。

这时可以用脚本 `scripts/eval_path_primitives_from_model.py` 从 **`model.pt + FeatureStore`** 重新评估并写出标准产物（`meta.json / metrics.json / metrics_summary.md / report.html / pred_sample.csv / model_path.txt`）。

> 说明：目前 `mlbot` 里还没有对应子命令（未来可以封装成 `mlbot nnmultihead eval-from-model`），所以这里先用脚本。

示例命令（可直接复制；`CUDA_VISIBLE_DEVICES=""` 强制走 CPU，避免评估阶段受 GPU/显存波动影响）：

```bash
CUDA_VISIBLE_DEVICES="" /usr/bin/python3 scripts/eval_path_primitives_from_model.py \
  --model results/exp006_group_ablation/highcap9_only_rerun3/path_primitives_4h_80h_min__prune_try__keep_semantic__volume_profile__poolb__volatility_reversal_score_f__poolb__compression_duration_f__rm_poolb__volume_profile_volatility_features_f_multi_240T/model.pt \
  --config results/nn_feature_group_search/pipeline_poolb_dir_auc_highcap6_2023_2024_e6_20260104_rerun1/tmp_configs/path_primitives_4h_80h_min__prune_try__keep_semantic__volume_profile__poolb__volatility_reversal_score_f__poolb__compression_duration_f__rm_poolb__volume_profile_volatility_features_f \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,DOTUSDT \
  --timeframe 240T --start-date 2023-01-01 --end-date 2025-04-30 \
  --features-store-root feature_store --features-store-layer features_83f12ecc5e \
  --out-dir results/exp006_group_ablation/highcap9_only_rerun3/path_primitives_4h_80h_min__prune_try__keep_semantic__volume_profile__poolb__volatility_reversal_score_f__poolb__compression_duration_f__rm_poolb__volume_profile_volatility_features_f_multi_240T \
  --max-rows-per-symbol 6000
```

**什么时候用**：
- 训练目录里只有 `model.pt`，缺 `meta/metrics/report`，需要补齐产物以便复盘与对比。
- 你更新了 `render_html_dashboard` / 指标计算逻辑，想对旧模型**重渲染报告**（不重训）。

### 8.2 降级顺序（明确、可自动化）
当漂移/异常触发：
1) **禁用 detector/gate**（回到 router-only）
2) **提高 NO_TRADE**（更保守）
3) **降低风险预算**（risk_per_slot 下调，只降不升）
4) **全局 NO_TRADE（停机）**

### 8.3 回滚
任何线上产物必须可以回滚到“上一个通过 gate 的版本”：
- model.pt + feature_contract + router thresholds + constitution yaml（全套 hash）

---

## 9. v1 推荐目录结构（产物与配置）
- `config/nnmultihead/...`：features/labels/model
- `config/constitution/...`：position/execution/gate/detector（版本化）
- `results/<exp_id>/...`：
  - `model/`、`preds/`、`mode/`、`logs_3action.parquet`
  - `router_report/`、`gate_report/`、`exec_control/`
  - `meta.json`（记录全部 config + hash）

---

## 10. 你要我检查的重点清单（review checklist）
- **自由度是否被显式限制**：max_slots、替换判定、加仓次数、symbol cap、kill-switch
- **每个自由度是否可开关**：gate/detector/add/replacement/soft_stop
- **是否能归因**：每个 deny/replace/add/kill 都有 reason
- **是否能回滚**：artifact 完整 + hash 绑定
- **Tree Gate 是否“导出规则”上线**：训练可以复杂，但线上必须简单、稳定

