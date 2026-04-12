# 工程指南（当前推荐）

**索引**：完整表格见 [上级 README](../README.md#工程指南guides-子目录)。

- **Plateau / 工作流**：`THRESHOLD_PLATEAU_TUNING_PROTOCOL_CN.md`、`PLATEAU_OPTIMIZATION_*.md`、`BASELINE_*`、`PRODUCTION_*`
- **Gate / BPC / 特征**：`GATE_*.md`、`HARD_GATE_SYSTEM.md`、`BPC_ADD_POSITION_*`、`FEATURE_*`
- **树模型（已归档）**：原 `tree/` 子目录已迁至 [docs/archive/guides/tree/](../../archive/guides/tree/)（说明见 [archive/guides/README.md](../../archive/guides/README.md)）

历史快照与截面 pipeline：[docs/archive/guides/](../../archive/guides/README.md)  
原 `docs/guides/` 占位说明：[docs/guides/README.md](../../guides/README.md)

---

## 与当前代码一致（维护说明）

- **Gate 高原 / 硬门控脚本**：若干文档仍写 `scripts/optimize_gate_plateau.py`、`optimize_gate_plateau_hard_gate.py` 等；这些文件**已不在仓库**。当前可直接运行的统一入口是 **`python scripts/optimize_gate_unified.py --help`**（策略侧 logs + `gate.yaml` 工作流）。`src/cli/main.py` 里的 `mlbot optimize gate-plateau` 仍指向缺失的 `optimize_gate_plateau.py`，若 CLI 报错需在代码侧修复后，文档中的 `mlbot optimize gate-plateau` 示例才能恢复有效。
- **归因分层**：`PRODUCTION_ATTRIBUTION_WORKFLOW.md` 中部分 `diagnose_*.py` 为历史示例名；请以 `scripts/` 下**实际存在的** `diagnose_*.py` 为准，或优先使用文内已对接 `mlbot diagnose …` 的命令。
- **长文链接**：`RD_TO_LIVE_TIERED_WORKFLOW_V1_CN.md` 等若引用 `docs/architecture/ARCH_UPGRADE_*.md`、`FINAL_SIMPLIFIED_ARCHITECTURE_*.md`、`树模型在多头…`，多数已迁至 **`docs/archive/`**（见各文件内已更正链接）。
