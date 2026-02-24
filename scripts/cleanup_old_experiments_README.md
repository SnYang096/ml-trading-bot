# 清理历史实验脚本

## 功能概述

`cleanup_old_experiments.py` 是一个用于批量清理历史实验的工具，旨在帮助管理 `results/research_history/` 目录下的实验数据，释放磁盘空间。

## 使用方法

### 1. 按时间戳删除

```bash
# 删除单个实验
python scripts/cleanup_old_experiments.py --strategy me --timestamp 20260223_121948

# 批量删除多个实验
python scripts/cleanup_old_experiments.py --strategy me --timestamp 20260223_114526 20260223_121948 20260223_121438
```

### 2. 按状态删除

```bash
# 删除错误状态的实验
python scripts/cleanup_old_experiments.py --strategy me --status error

# 删除采纳状态的实验
python scripts/cleanup_old_experiments.py --strategy me --status adopt
```

### 3. 按日期范围删除

```bash
# 删除指定日期范围内的实验
python scripts/cleanup_old_experiments.py --strategy me --date-range "2026-02-20" "2026-02-22"
```

### 4. 删除全部实验

```bash
# 删除指定策略的全部历史实验（谨慎使用）
python scripts/cleanup_old_experiments.py --strategy me --all
```

### 5. dry-run 模式

```bash
# 预览将要删除的内容，不实际删除
python scripts/cleanup_old_experiments.py --strategy me --timestamp 20260223_121948 --dry-run
```

## 注意事项

- 仅删除 `results/research_history/{strategy}/{timestamp}/` 目录
- 不影响 `config/strategies/` 和 `live/` 中的生产配置
- 建议先使用 `--dry-run` 确认将要删除的内容
- 删除后无法恢复，请谨慎操作
- 执行删除前会有二次确认