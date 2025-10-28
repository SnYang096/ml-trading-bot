# 📚 项目文档中心

统一管理所有项目文档，避免散落各处。

## 📁 目录结构

```
docs/
├── README.md                    # 本文档索引
├── reports/                     # 📊 报告文档
│   ├── FINAL_COMPREHENSIVE_REPORT.md
│   ├── MODEL_COMPARISON_SUMMARY.md
│   ├── DIAGNOSTIC_REPORT.md
│   └── TRAINING_RESULTS_2021_2023.md
│
├── guides/                      # 📖 使用指南
│   ├── QUICK_START.md
│   ├── QUICK_SUMMARY.md
│   ├── MULTI_YEAR_TRAINING_GUIDE.md
│   ├── USAGE_GUIDE.md
│   ├── DATA_DOWNLOAD_README.md
│   └── CONDA_ENV_README.md
│
├── analysis/                    # 🔬 分析文档
│   ├── check_missing_features.md
│   └── BATCH_TEST_EXAMPLES.md
│
└── archive/                     # 📦 归档文档
    ├── 项目完成总结.md
    ├── 最终完整总结报告.md
    ├── 快速参考指南.md
    └── 数据下载总结.md
```

## 📋 文档分类

### 📊 报告文档 (reports/)
**项目报告和结果总结**
- `FINAL_COMPREHENSIVE_REPORT.md` - 最终综合报告
- `MODEL_COMPARISON_SUMMARY.md` - 模型对比总结
- `DIAGNOSTIC_REPORT.md` - 诊断报告
- `TRAINING_RESULTS_2021_2023.md` - 训练结果

### 📖 使用指南 (guides/)
**用户指南和快速开始**
- `QUICK_START.md` - 快速开始
- `QUICK_SUMMARY.md` - 快速总结
- `MULTI_YEAR_TRAINING_GUIDE.md` - 多年训练指南
- `USAGE_GUIDE.md` - 使用指南
- `DATA_DOWNLOAD_README.md` - 数据下载说明
- `CONDA_ENV_README.md` - Conda环境说明

### 🔬 分析文档 (analysis/)
**技术分析和诊断**
- `check_missing_features.md` - 缺失特征检查
- `BATCH_TEST_EXAMPLES.md` - 批量测试示例

### 📦 归档文档 (archive/)
**历史文档和总结**
- `项目完成总结.md` - 项目完成总结
- `最终完整总结报告.md` - 最终完整报告
- `快速参考指南.md` - 快速参考
- `数据下载总结.md` - 数据下载总结

## 🎯 快速导航

| 需求 | 文档位置 |
|------|---------|
| **开始使用** | [guides/QUICK_START.md](guides/QUICK_START.md) |
| **查看结果** | [reports/FINAL_COMPREHENSIVE_REPORT.md](reports/FINAL_COMPREHENSIVE_REPORT.md) |
| **模型对比** | [reports/MODEL_COMPARISON_SUMMARY.md](reports/MODEL_COMPARISON_SUMMARY.md) |
| **数据下载** | [guides/DATA_DOWNLOAD_README.md](guides/DATA_DOWNLOAD_README.md) |
| **环境设置** | [guides/CONDA_ENV_README.md](guides/CONDA_ENV_README.md) |
| **项目总结** | [archive/项目完成总结.md](archive/项目完成总结.md) |

## 📝 文档维护原则

### 1. 统一管理
- ✅ 所有文档放在 `docs/` 目录
- ✅ 按功能分类到子目录
- ✅ 根目录不保留散落文档

### 2. 分类规则
- **reports/** - 项目报告、结果总结
- **guides/** - 用户指南、快速开始
- **analysis/** - 技术分析、诊断
- **archive/** - 历史文档、归档

### 3. 命名规范
- 重要文档：`UPPER_SNAKE_CASE.md`
- 一般文档：`lower_snake_case.md`
- 中文文档：`中文名称.md`

## 🔄 迁移计划

### 第一步：移动现有文档
```bash
# 移动报告文档
mv FINAL_COMPREHENSIVE_REPORT.md docs/reports/
mv MODEL_COMPARISON_SUMMARY.md docs/reports/
mv DIAGNOSTIC_REPORT.md docs/reports/
mv TRAINING_RESULTS_2021_2023.md docs/reports/

# 移动指南文档
mv QUICK_START.md docs/guides/
mv QUICK_SUMMARY.md docs/guides/
mv MULTI_YEAR_TRAINING_GUIDE.md docs/guides/
mv USAGE_GUIDE.md docs/guides/
mv DATA_DOWNLOAD_README.md docs/guides/
mv CONDA_ENV_README.md docs/guides/

# 移动分析文档
mv check_missing_features.md docs/analysis/
mv BATCH_TEST_EXAMPLES.md docs/analysis/

# 移动归档文档
mv "🎉_项目完成总结.md" docs/archive/项目完成总结.md
mv "🎊_最终完整总结报告.md" docs/archive/最终完整总结报告.md
mv "📌_快速参考指南.md" docs/archive/快速参考指南.md
mv "数据下载总结.md" docs/archive/
```

### 第二步：更新引用
- 更新所有文档中的相对链接
- 更新 README.md 中的文档链接
- 检查脚本中的文档引用

### 第三步：清理根目录
- 删除根目录的散落文档
- 保留必要的 README.md
- 保持项目结构清晰

## 📚 相关目录

- **scripts/docs/** - 脚本相关文档
- **src/** - 源代码
- **results/** - 结果文件
- **models/** - 模型文件

---

**文档版本**: 1.0.0  
**最后更新**: 2025-10-22  
**维护者**: AI Trading Team
