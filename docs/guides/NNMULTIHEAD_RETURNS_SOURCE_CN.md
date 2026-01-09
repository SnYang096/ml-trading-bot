## nnmultihead / 3-action：`--returns-source` 详解（rr_execution vs momentum_proxy）

这份文档解释 `mlbot rl build-logs-3action --returns-source ...` 的含义：它决定 `logs_3action.parquet` 里 **`ret_mean` / `ret_trend`** 两列怎么计算，也就是你当前实验的“execution 假设”来源。

> TL;DR：  
> - **`momentum_proxy`**：更像“execution 解耦对照”（先评估 Router 切分本身）  
> - **`rr_execution`**：更像“主链路落地评估”（Router + 执行假设一起评估）  
> 两者不要混着直接比 Sharpe，回答的问题不同。

---

## 1) 这些名字是类吗？

不是。它们是一个 **字符串枚举**（CLI choices），由代码分支选择不同的 returns 计算实现。

代码位置：
- `scripts/rl_build_logs_3action.py`：`--returns-source` 的 choices 列表
- `src/time_series_model/rl/build_logs_3action.py`：`BuildLogs3ActionConfig.returns_source` + `build_logs_3action()` 的分支逻辑

---

## 2) `momentum_proxy`：做“execution 解耦对照”

### 它干啥？
只用价格的过去动量方向构造一个很简化的执行回报：
- `ret_trend = sign(momentum) * next_return`
- `ret_mean  = -sign(momentum) * next_return`

这里的 `momentum` 只用历史价格（lookback 默认 5），`next_return` 是下一根 close-to-close return。

代码入口：
- `src/time_series_model/rl/build_logs_3action.py::_compute_mode_returns`

### 你什么时候用它？
- 你想先回答：**Router 的 action 切分本身有没有信息/有没有 edge**  
  （尽量不引入复杂 execution）
- 做 sanity check / baseline 对照口径

### 怎么解读？
- 如果 `momentum_proxy` 下都不稳定/很差：优先怀疑 **Router 切分/阈值/样本切片**，而不是 execution。

---

## 3) `rr_execution`：做“更贴近落地”的执行假设

### 它干啥？
用 RR/ATR 执行模拟器生成 `ret_mean/ret_trend`，更接近“primitives → 执行控制”的主路线。

代码入口：
- `src/time_series_model/rl/build_logs_3action.py` 分支 `compute_rr_execution_mode_returns(...)`

### 你什么时候用它？
- 你想评估：**Router + 当前执行假设**作为整体的落地表现（主链路）

### 怎么解读？
- `rr_execution` 下的 Sharpe/DD **强依赖 execution 假设**：`entry_delay / cost / slippage / rr 参数`。  
  因此必须做稳健性扫描（网格/敏感性），否则只是在理想假设下好看。

---

## 4) 一句话建议（实操顺序）

1. 先用 **`momentum_proxy`**：验证 Router 切分本身是否站得住（解耦对照）
2. 再用 **`rr_execution`**：看主链路落地表现（强依赖 execution 假设）
3. 最后对 `rr_execution` 做 **cost/slippage/entry_delay** 稳健性扫描

