## Execution Rules（导出规则：用于替代/覆盖启发式 required_conditions）

定位：
- 训练阶段：可以用树模型（LightGBM/XGB）训练 execution quality / failure detector
- 上线阶段：把结果导出成 **简单可审计的规则**（YAML），由 live 直接加载执行

约束：
- 规则必须 **fail-closed**：缺输入/解析失败 => NO_TRADE
- 规则必须 **离散化**：只允许 veto / allow（或后续扩展 throttle 档位），禁止连续 score 直接乘仓位

接入点（live）：
- `MetaRouterStrategy` 会优先读取环境变量：
  - `MLBOT_EXECUTION_RULES_YAML`（可选）
  - 若存在：执行规则将作为 `required_conditions` 的补丁/替代（v1：只做 veto）

