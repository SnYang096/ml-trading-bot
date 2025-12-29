# nnmultihead：Feature Contract 的 block gating（可选特征块）落地说明

本文档解释 **optional blocks** 在 `nnmultihead`（path-primitives MLP）里的工程实现：  
如何做到 **“一个模型 + 可选特征块（有就用，没有就不用）”**，并且保证训练/推理口径一致。

## 核心结论（先说清楚）

- **线上不要“删列/不传列”**：模型输入维度必须固定。  
  正确做法是：缺失的 block 对应列保持为缺失（NaN）→ 进入 torch 前会被填成 0，并且同时提供 **block availability mask**。
- **`append_block_mask` 不是 drop 特征**：它是 **给模型额外拼一个 block 是否可用的 0/1 标记**（输入维度增加 `n_blocks`）。
- **`block_dropout_p` 是训练时的“整块随机 drop”**：按样本、按 block 随机把该 block 的所有特征置 0，并把对应 mask 置 0（只在训练阶段生效）。
- 当前实现是 **block 级别**，不是“block 内随机 mask 某些特征”。（后续如需 per-feature masking，可以再扩展。）

---

## 配置（Feature Contract）

位置：`config/nnmultihead/<task>/feature_contract.yaml`

示例（节选）：

```yaml
feature_contract:
  minimal_required_cols:
    - open
    - high
    - low
    - close
    - volume
    - atr

  # 新格式：block_name -> patterns（fnmatch）
  optional_blocks:
    ticks_orderflow_semantic_blocks:
      - "*vpin*"
    trade_cluster_semantic_blocks:
      - "*trade_cluster*"
      - "*TradeCluster*"

  missingness_policy:
    # 可选：线上缺失时如何处理（当前实现以 NaN->0 + mask 为主）
    optional_blocks_on_missing: skip
    # 开启：在输入尾部拼上每个 block 的可用性 mask
    append_block_mask: true
    # 训练时：整块随机 drop 的概率（建议从 0.01~0.10 起步）
    block_dropout_p: 0.05
```

---

## 算法与数据流（训练 / 推理）

### 1) 解析 block → 列集合（pattern match）

代码：`src/time_series_model/models/nn/path_primitives_dataset.py`

- `resolve_block_cols_by_name(feature_cols, optional_blocks=...)`
  - 输入：训练用 `feature_cols`（固定列集合）+ 配置里的 patterns
  - 输出：`block_cols_by_name: Dict[str, List[str]]`

### 2) 计算每行的 block availability mask（0/1）

同文件：

- `_compute_block_availability_mask(...)`
  - 规则：某一行只要 block 内 **任意列**是 finite（notna 且不是 inf），该 block mask=1，否则 0。

### 3) 构建 torch 输入矩阵（NaN→0，并拼 mask）

同文件：

- `build_feature_matrix(..., append_block_mask=True, block_cols_by_name=...)`
  - 先把所有 feature cols 转数值、inf→NaN、NaN→0
  - 再把 `block_mask` 拼到输入尾部

结果：模型实际输入维度变成：

\[
d_{in} = |feature\_cols| + |blocks|
\]

### 4) 训练时 block-dropout（随机整块 drop）

代码：`src/time_series_model/models/nn/path_primitives_trainer.py`

在每个训练 batch 上：
- 对每个样本、每个 block 生成随机数 `rand ~ U(0,1)`
- 若该 block 本来可用（mask=1）且 `rand < block_dropout_p`
  - 将该 block 对应的所有 feature dims 置 0
  - 将该 block 的 mask dim 置 0

这相当于对“线上 block 开/关、缺失、成本 gating”的情形做 Monte-Carlo augmentation。

### 5) 推理阶段（不做 dropout，但保留 mask）

推理/评估代码：`src/time_series_model/models/nn/path_primitives_reporting.py`

- `predict_path_primitives(..., append_block_mask=True, block_cols_by_name=...)`
  - 不会做 dropout
  - 仍然会构建相同维度的输入（NaN→0 + mask）

---

## 用现有命令验证

训练命令（例）：

```bash
python scripts/train_path_primitives_mlp.py \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --horizon-hours 80 \
  --output-dir results/nn_path_primitives
```

训练产物中 `meta.json` / `report.html` 的 Raw meta 会包含：
- `feature_contract`
- `block_cols_by_name`
- `append_block_mask`
- `block_dropout_p`
- `block_mask_names`

---

## 常见误解澄清

- **Q：append_block_mask 是“删除整个 block 特征”吗？**  
  不是。它是“增加一个 mask 输入”，让模型识别 block 是否可用。

- **Q：block_dropout_p 是“mask block 里某些特征”吗？**  
  不是。当前实现是“整块 drop”（block 内全部特征一起 drop）。

- **Q：随机 drop 足够覆盖各种线上情况吗？**  
  它只能覆盖“缺失/关闭”的分布，但不能替代分组 ablation 的“准入证据”。  
  推荐：`block_dropout_p` 小比例做鲁棒性正则；是否把某块默认 on/off 仍要靠 add-back + slice 评估决定。


