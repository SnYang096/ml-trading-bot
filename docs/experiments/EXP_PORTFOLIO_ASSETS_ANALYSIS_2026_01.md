# portfolio_assets.yaml 使用情况分析

**分析时间**: 2026-01-22  
**删除时间**: 2026-01-27  
**状态**: ❌ **已删除**

---

## 删除原因

`portfolio_assets.yaml` 及其相关功能已被删除，原因如下：

1. **系统设计简化**：当前系统只有 2 个 slot，不是大量开仓场景，不需要组合层的资金分配控制
2. **功能冗余**：Slot 机制已经足够控制仓位数量，portfolio assets 的组合层控制是多余的
3. **维护成本**：该功能依赖 router（已移除），且主要用于诊断，实际交易逻辑中未使用

---

## 历史信息（保留用于参考）

### 原始设计意图

`portfolio_assets.yaml` 原本设计用于：
- 定义 5 个抽象的"组合资产"（GLOBAL_TREND, GLOBAL_MEAN, GLOBAL_CASH, HIGH_BETA_OVERLAY, DEFENSIVE_MEAN）
- 从 router aggregate signals 映射到资产权重
- 在组合层控制整体资金分配

### 已删除的文件

- `config/portfolio_assets/portfolio_assets.yaml` - 配置文件
- `src/time_series_model/portfolio/portfolio_assets.py` - 实现代码
- `src/time_series_model/portfolio/portfolio_assets_artifacts.py` - 诊断 artifacts 生成
- `scripts/diagnose_portfolio_allocation_plateau.py` - 诊断脚本
- 相关测试文件

### 已清理的引用

- 从所有 `task_spec*.yaml` 中删除 `portfolio_assets_plan` 配置
- 从 `counterfactual_eval_3action.py` 中删除相关代码
- 从 CLI 中删除 `diagnose portfolio-allocation-plateau` 命令
- 从 `scripts/rl_counterfactual_eval_3action.py` 中删除相关参数

---

## 结论

**当前系统通过 Slot 机制控制仓位数量，不再需要组合层的资金分配控制。**
