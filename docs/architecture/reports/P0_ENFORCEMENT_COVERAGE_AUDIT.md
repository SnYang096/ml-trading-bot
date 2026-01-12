## P0 Enforcement Coverage Audit（Live 下单前强制宪法的覆盖率清单）

目标：确保 **所有 live 下单路径** 都只能通过 `ExecutionManager.submit_order_guarded()` 进入，从而保证下单前必经 `enforce_before_order()`（宪法/白名单/slot 等强制点不可绕过）。

### 范围

- 目录：`src/time_series_model/live/*.py`
- 关键 API：
  - `ExecutionManager.submit_order_guarded()`（唯一允许的下单入口）
  - `enforce_before_order()`（必须在下单前调用）

### 当前覆盖结论（2026-01-12）

- **单入口已实现**：`src/time_series_model/live/execution_manager.py`
- **已迁移到单入口的策略**：
  - `src/time_series_model/live/meta_router_strategy.py`
  - `src/time_series_model/live/event_driven_strategy.py`
  - `src/time_series_model/live/nautilus_strategy_with_features.py`
  - `src/time_series_model/live/nautilus_strategy_enhanced.py`

### 回归测试（防新增绕过）

- `tests/unit/test_live_order_submission_must_be_guarded.py`
  - 规则：`src/time_series_model/live/*.py` 中禁止出现 `.submit_order(`（仅允许 `execution_manager.py` 内部使用）。

### 注意事项（后续可能要做的增强）

- **Exit 下单**：目前 `nautilus_strategy_enhanced.py` 的 exit 也走了 guarded submit（保守做法）。
  - 若未来发现 exit 被 whitelist 误伤，可新增 “exit 专用 enforcement hook”（仍保持单入口）。

