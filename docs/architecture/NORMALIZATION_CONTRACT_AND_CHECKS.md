### Normalization Contract：它是什么、为什么要做推断、以及怎么用“硬检查”避免 silent bug

这份文档用于回答两类经常出现的困惑：

1) **“normalization contract 的推断代码有什么用？”**  
2) **“我怎么自动发现：某些被依赖的列必须是 raw（价格单位），而不能被偷偷改成归一化？”**

---

## 1) 什么是 normalization contract（代码级契约）

在本仓库里，`config/feature_dependencies.yaml` 不只是“特征列表”，它还是一个**可执行的契约**：

- 每一个 feature node 的每一个 `output_columns` 都应该有明确的归一化语义（method），例如：
  - `unitless` / `bounded_0_1` / `bounded_-1_1`
  - `zscore_rolling` / `rank_rolling` / `log1p_robust_rolling`
  - `usd`（美元尺度）/ `price_unit`（价格单位）/ `raw`
- 契约的目的不是“写文档”，而是：
  - **让 CI/测试可以 fail-fast**
  - **让语义变更可回归**
  - **让多人协作不靠记忆**

对应代码入口：
- 推断与收集：`src/features/normalization/feature_contract.py`
- CI gate（现有脚本）：`scripts/check_normalization_contract_ci.py`

---

## 2) 为什么需要“推断代码”（而不是要求所有节点都手写完整 map）

现实约束：

- feature node 数量很多（上百/上千），短期内不可能把所有节点都手工补齐 `output_normalization_map`。
- 但我们又希望 contract **马上能运行**，否则永远只能“靠人看”。

所以推断器的作用是：

- 从 `compute_params.normalize_mode` / `compute_params.output_normalization` / `output_normalization_map` 等显式字段提取 method
- 在缺失时做保守推断（例如识别 0..1 / -1..1 / unitless 的常见描述）
- 如果仍然无法判断，就标记为 `raw`（这不是“通过”，而是告诉你：该列还没有明确归一化语义）

这一步的价值在于：**你可以把“文档式声明”变成“可执行检查”。**

---

## 3) 为什么 “raw / price_unit / usd” 这种标注不是多余的？

关键点：**raw 并不是“不能用”，而是“语义必须明确，否则依赖链会 silent wrong”。**

### 3.0 raw vs price_unit vs usd：区别是什么？

- **`raw`**：表示“还没有明确归一化语义/尺度语义”（未知/未治理）。  
  - 这不是一种“合法的已定义尺度”，而是一个 **治理待办**：表示需要你补齐归一化或补齐明确的尺度标注。

- **`price_unit`**：表示“价格单位尺度”（和 `open/high/low/close` 同单位）。  
  - 这类列通常被用于 **尺度化/反归一化**（例如 `x / atr`、`level_norm * atr + close`），因此不应被悄悄改成无量纲还沿用原名。
  - 这种列通常 **不可直接跨资产比较**（cross_asset_comparable=false），但完全可以作为中间尺度用于构造无量纲特征。

- **`usd`**：表示“美元尺度”（例如 `market_cap_usd`）。  
  - 同样不可直接跨资产比较，但可以作为中间尺度构造 unitless 的 ratio（例如 `dollar_volume_over_mcap`）。

一句话：`raw` 是“未知/未治理”，`price_unit/usd` 是“已明确但非归一化的物理尺度”。

### 3.1 典型事故：ATR 语义被悄悄替换

`atr` 在本系统里扮演“尺度列”的角色：

- 路径原语 labels：`mfe_atr = price_diff / atr(t)`  
- SR 反归一化：`level_raw = level_norm * atr + close`

因此 `atr` 必须是 **价格单位**，不能被改成 `atr/close`（无量纲）并继续叫 `atr`。

这类问题的可怕之处在于：

- 代码仍然能跑（不会报错）
- 但数学意义错了（指标被放大/缩小）
- 如果没有 contract + 检查，你只能靠经验/肉眼发现

我们现在通过 `output_normalization_map` 显式标注：

- `atr_f` 输出列 `atr` 的 method 为 `price_unit`

并用检查命令/测试把它固化（改变语义会 fail）。

### 3.2 混合输出：market_cap 节点为什么也要标注

`market_cap_normalized_orderflow_f` 同时输出：

- `market_cap_usd`（USD 尺度，不可直接跨资产比较）
- `*_over_mcap`（无量纲，可跨资产比较）

如果不标注，就会把这类节点误当成“全部 unitless”，从而在 CI/审计时看不出风险。

因此我们用 `output_normalization_map` 把混合输出分清楚，并用测试固化。

---

## 4) “raw 源列”与“自动化找 bug”的关系（回答你的追问）

你说得对：**下游拿 raw 输入也能算出归一化输出**，这本身没问题。

真正要抓的是：

> **“一个下游在数学上需要 raw/price_unit，但上游把同名列变成 unitless 并继续叫原名”**

为了让机器能抓这种问题，它必须知道两件事：

- 上游输出列的真实语义（raw / price_unit / usd / unitless）
- 下游对输入列的“预期语义”（至少要能区分：这是尺度列/反归一化用，还是普通 unitless）

这就是为什么需要在关键尺度列上做明确标注，并把检查写进 CI/命令里。

---

## 5) 我应该怎么跑这些检查（推荐命令）

新增/统一后的命令入口（推荐用于本地+CI）：

```bash
mlbot diagnose feature-contract --no-docker
```

它会执行：

- normalization contract 完整性检查（无缺失 method）
- 混合输出检查（例如 USD vs unitless）
- 尺度列保护检查（例如 `atr` 必须是 `price_unit`）
- 订单流 raw 源列相关的“泄漏风险提示”（见输出报告）

输出建议写到 `results/` 或 CI artifacts 目录，便于长期追踪。


