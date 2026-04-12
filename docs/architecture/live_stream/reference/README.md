# `reference/` 说明

本目录曾存放以 **QuestDB 为实时主存** 的补全/存储长文草案，与当前仓库主线（**Parquet + `StorageManager` / `GapFiller` + `run_live.py`**）不一致，已移除以免误导。

**请以以下为准**：

- 补全与模式：`../数据补全架构.md`、`src/live_data_stream/system_mode.py`、`src/live_data_stream/gap_filler.py`
- 存储布局：`src/live_data_stream/README_STORAGE.md`、`src/live_data_stream/feature_storage.py`
- 事件回测对齐：`docs/architecture/event_drive_backtest/`
