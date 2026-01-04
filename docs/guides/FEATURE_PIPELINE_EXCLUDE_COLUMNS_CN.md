## feature_pipeline.exclude_columns（CN）

这份文档解释一个常见但容易踩坑的问题：**某些列（如 `atr`）需要存在于特征 DataFrame 中用于标签/回测**，但**不应该作为模型输入特征喂给模型**（尤其在 multi-symbol 训练时，`atr` 的尺度可能成为 shortcut）。

本文档描述 tree 策略模型与 nnmultihead（路径原语）两条体系的统一做法：**数据中保留、输入中排除**。

---

### 背景：为什么需要“保留但不喂给模型”

很多标签/回测逻辑会使用 ATR 作为尺度（例如 “\( \Delta price / atr \)” 或 “signal_threshold_atr * atr”）：
- **标签需要**：标签生成器要用 `atr` 来归一化收益/阈值，使不同波动率阶段可比。
- **回测需要**：执行/风险控制可能也需要尺度列。
- **模型不一定需要**：对 tree/线性/NN 来说，直接输入 `atr` 可能让模型学到“价格/尺度捷径”（尤其跨币种时）。

因此我们希望：
- DataFrame 里 **仍然有 `atr`**（供 label/backtest 使用）
- 但训练时选出的 `feature_cols` **不包含 `atr`**

---

### Tree（策略模型）：features_base.yaml vs features.yaml 的职责

#### 1) `features_base.yaml`（只用于搜索起点 / label&backtest 必需节点）

`features_base.yaml` 是一个 **YAML list**，仅用于：
- 给 `feature-group-search` 提供 **base_features 起点**
- 确保 baseline/candidate 运行时 label/backtest 所需的节点会被计算

它**不是**训练管线的配置载体，也不支持写 `feature_pipeline` 的结构字段。

#### 2) `features.yaml`（训练/回测“管线配置载体”）

`features.yaml` 是 strategy 的 “pipeline 配置载体”，除了 `requested_features` 之外还承载：
- `ensure_signal_column`
- `invert_features`
- 以及本文新增的 `exclude_columns`

`feature-group-search` 的实现会：
- copy 整个 strategy 目录到临时目录
- **只覆写 `features.yaml` 里的 `feature_pipeline.requested_features`**
- 其它字段（包括 `exclude_columns`）会自动继承

所以 `exclude_columns` 应该写在 `features.yaml`。

---

### 统一规则：在 YAML 中声明 exclude_columns

在 strategy 的 `features.yaml` 中写：

```yaml
feature_pipeline:
  exclude_columns: [atr]
  requested_features:
    - ...
```

含义：
- 仍然会计算/保留 `atr`（因为 requested_features/base_features 可能包含 `atr_f`）
- 但最终用于训练的 `feature_cols` 会从候选列中移除 `atr`

---

### nnmultihead（路径原语）：同样的“保留但不喂给模型”

nnmultihead 侧同样存在尺度捷径风险：
- primitives labels/contract 常常要求 `atr` 存在
- 但 MLP 输入默认应排除 `atr`

当前推荐原则：**`atr` 留在 DataFrame（labels/contract），从模型输入列里排除**。

（如果你希望 nn 侧也完全 YAML 化，可在 nn 配置的 `features.yaml` 增加同名字段，并让脚本优先读取它；CLI 参数仅做 override。）

---

### 常见误解澄清

- **Q: “feature-group-search 不应该用 features.yaml，因为它是最终训练特征文件？”**  
  A: search 并不会“沿用你原来的 requested_features”，它会在临时策略目录里把 `requested_features` 覆写成 baseline/candidate 的组合。它使用 `features.yaml` 只是为了继承其它 pipeline 配置（如 ensure_signal / invert / exclude）。

- **Q: “exclude_columns 应该写到 features_base.yaml 吗？”**  
  A: 不应该。`features_base.yaml` 是 list，不是 pipeline 配置载体；真正决定“喂给模型哪些列”的是在训练时的 `feature_cols` 选择逻辑，所以要在 `features.yaml` 的 `feature_pipeline` 下声明。


