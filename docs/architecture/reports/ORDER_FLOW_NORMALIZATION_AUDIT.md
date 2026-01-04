### 订单流（Order Flow）特征：哪些应该是 raw（normalized:false），哪些应该提供归一化版本？

这份审计的目标是避免出现类似 `atr_f` 的错误：**把一个“会被下游当作尺度/基准”的列改成归一化值，导致依赖方计算错误**。

---

## 1) 结论（先给最重要的）

- **订单流这条链路里，最容易出错的不是 VPIN / trade_cluster（它们大多是无量纲/有界/统计量），而是“原始成交量/数量/累计量”这些源列。**
- 这些源列很多来自原始数据（不是由某个 feature node 计算出来），因此它们不会出现在 `feature_dependencies.yaml` 的 `output_columns` 里，但依然需要你在设计上当成 **raw 源** 对待：不要“原地归一化并保留同名”。

---

## 2) 建议的“raw 源列”清单（应视为 normalized:false）

这些列通常在工程里扮演 **原始输入/中间变量** 的角色：允许被用来构造其它“无量纲/可比”的特征，但不建议直接喂给 NN（除非你明确做了归一化/标准化）。

- **`buy_qty` / `sell_qty`**：原始买/卖数量（base units）
- **`delta`**：净主动买卖量（base units）
- **`cvd`**：累计净量（cumulative），强依赖合约单位/交易所统计口径
- **`cvd_change_1` / `cvd_change_5` / `cvd_change_20`**：CVD 的变化量（仍是 raw units）

> 上面这些“raw 源列”的归一化版本应该以 **新列名** 提供，而不是覆盖原列名。

---

## 3) 推荐的“归一化/可比版本”（建议优先喂给模型/Router）

这些列通常是无量纲/有界，更适合跨币种和跨阶段稳定使用：

- **`taker_buy_ratio`**：0..1（方向强度/主动买比例）
- **`order_flow_delta`**：建议用 `delta/volume` 或 `cvd_change_1/volume` 之类构造（无量纲）
- **`cvd_normalized`**：若存在，优先使用（仓库里已有使用逻辑：`baseline_features.py` 会优先用它）
- **market-cap normalized flow**（多币训练强烈推荐）：
  - `dollar_volume_over_mcap`
  - `net_buy_usd_over_mcap`
  - `abs_net_buy_usd_over_mcap`

---

## 4) 当前 repo 的审计发现（是否存在“像 atr_f 一样会把依赖方算坏”的问题？）

### 4.1 VPIN / TradeCluster

- `vpin_base_aligned_features_f` 输出为统计量（unitless）
- `trade_cluster_base_aligned_features_f` 输出显式有界（`bounded_0_1` / `bounded_-1_1`）

这两块 **不存在“需要 raw 尺度列才能反归一化”的依赖链**，因此不太会出现 `atr_f` 那种“语义错了导致下游计算错误”的连锁问题。

### 4.2 CVD 派生语义特征

例如 `cvd_divergence_f` 只把 `cvd` 当作序列做 rolling min/max，并最终输出 0/1 + 0..1 的强度：
它对 `cvd` 的绝对量级不敏感（通过窗口内 range 归一化），因此不会要求 `cvd` 必须是某种特定尺度。

但依然建议：**把 `cvd` 保留为 raw 源列**，归一化版本另起列名（`cvd_normalized`）。

---

## 5) 实操建议（你下一步怎么做）

1) 把上面列出的 raw 源列当作“数据层字段”，不要在任何地方“原地归一化并保留同名”。  
2) 在策略/NN 输入里，优先选择：
   - `taker_buy_ratio`
   - `cvd_normalized`（如果有）
   - `delta/volume`（如果你要自己做，输出新列名）
   - market-cap normalized flows（多币训练时性价比很高）
3) 只有当你明确知道某个下游需要 raw（例如用于算美元值：qty*close），才使用 raw 源列。


