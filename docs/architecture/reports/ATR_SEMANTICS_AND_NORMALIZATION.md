### ATR 语义结论（必须统一，否则会导致计算错误）

本仓库里有两种“ATR”概念，容易混淆：

- **`atr`（价格单位 ATR）**：用于“尺度”而不是用于“归一化特征”  
  - 用途 1：**路径原语 label 归一化**  
    - `mfe_atr = (future_high - entry) / atr(t)`  
    - `mae_atr = (entry - future_low) / atr(t)`  
    - 这要求 `atr` 必须是**价格单位**（和 `close/high/low` 同一单位），否则 label 会被错误放大或缩小。
  - 用途 2：**把 ATR 倍数的结构量（例如 SR 距离）“反归一化”回价格单位**  
    - 例如 `sr_price = close + dist_atr * atr`

- **“归一化 ATR”（跨币可比）**：用于“特征输入/状态指标”  
  - 常见形式：`atr_ratio = atr / close`（无量纲），或 `natr_14`，或 `atr_percentile`

---

### 当前仓库的硬约束（重要）

- `src/time_series_model/models/nn/path_primitives_labels.py` 使用 `atr_col="atr"` 作为分母做 label 归一化。  
  因此 `atr` 必须是**价格单位**。

- 多个 SR/结构特征（例如 SQS / SR strength）会把 “(level-close)/ATR” 的归一化量反推回 SR 价格：
  - `level_raw = level_norm * atr + close`
  因此 `atr` 同样必须是**价格单位**。

---

### 结论（回答你的问题）

1) `atr_f` 输出的 `atr` 必须是**价格单位 ATR**，否则依赖 `atr` 的特征与 label 会计算错误。  
2) “其他列依赖 atr” 时，它们拿到的就是同一列 `atr`：因此统一语义很关键。  
3) 如果你需要“归一化 ATR”，应该使用单独列（例如 `atr_ratio` / `natr_14` / `atr_percentile`），而不是把 `atr` 本身改成 `atr/close`。

---

### 已执行的修复

- `compute_atr_from_series`（也就是 `atr_f`）现在输出 **价格单位 ATR** 到列 `atr`。  
- `atr/close` 这种跨币可比形式，请用 `atr_ratio`（或 `natr_14` / `atr_percentile`）获取。

---

### 快速审计：还有没有别的“像 atr_f 一样会把依赖方算坏”的风险？

结论（当前仓库状态）：**高风险的“被依赖尺度列”主要就是 `atr`**，已经修正并加了 `normalized:false` 标注。

- `config/feature_dependencies.yaml` 里明确把 `atr` 作为 `required_columns` 的特征主要集中在 SR/结构类：
  - `sqs_hal_high_f` / `sqs_hal_low_f`
  - `sr_strength_max_f` / `sr_strength_max_close_f`
  这些都要求 `atr` 为**价格单位**，否则内部的 “level_raw = level_norm * atr + close” 会错误。

- 其它看起来“像 ATR 但已归一化”的列（例如 `atr_7/atr_14/atr_21`，描述为 normalized by close）：
  - 目前 **没有被其它特征作为依赖输入使用**（它们更多是终端特征列），因此不会出现“下游把它当价格单位尺度”这类连锁错误。

因此你担心的那类问题，最应该盯住的是：

> **“输出列名是尺度列/会被反归一化使用”，但值被改成了无量纲**。

目前我们已把 `atr` 这条链路纠正并文档化；如果未来引入新的“尺度列”（例如把 `close` 替换为某种归一化 price），也应按同样方式做语义固定与审计。


