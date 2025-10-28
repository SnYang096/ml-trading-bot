# 📁 ML Project 文件整理计划

## 🎯 整理目标

将散落在 `ml_project` 根目录的文件按功能分类整理。

## 📊 当前散落文件分析

### Python 脚本 (11个)
- `analyze_enhanced_features.py` - 分析增强特征
- `analyze_feature_differences.py` - 分析特征差异
- `check_gpu.py` - 检查GPU
- `compare_models.py` - 对比模型
- `corrected_evaluation.py` - 修正评估
- `diagnostic_analysis.py` - 诊断分析
- `spectral_fix_proposal.py` - 频谱修复提案
- `test_enhanced_oos.py` - 测试增强OOS
- `test_oos_months.py` - 测试OOS月份
- `test_real_data.py` - 测试真实数据
- `setup.py` - 项目设置

### PowerShell 脚本 (10个)
- `batch_test.ps1` - 批量测试
- `download_data.ps1` - 下载数据
- `download_to_agg_data.ps1` - 下载聚合数据
- `quarterly_rolling.ps1` - 季度滚动
- `quick_gpu_train.ps1` - 快速GPU训练
- `run_gpu_training.ps1` - 运行GPU训练
- `START_HERE.ps1` - 开始使用
- `test_gpu.ps1` - 测试GPU
- `train_2021_2023_test_2024_2025.ps1` - 训练2021-2023测试2024-2025
- `train_and_test.ps1` - 训练和测试
- `train_jan_test_feb.ps1` - 训练一月测试二月

### 数据文件 (6个)
- `enhanced_feature_analysis_output.txt` - 增强特征分析输出
- `enhanced_feature_importance_by_category.csv` - 按类别分组的特征重要性
- `enhanced_feature_importance_full.csv` - 完整特征重要性
- `feature_importance_5T.csv` - 5分钟特征重要性
- `model_info_enhanced_may_2025.json` - 增强模型信息
- `model_info_wavelet_may_2025.json` - 小波模型信息
- `oos_test_results_enhanced.json` - 增强OOS测试结果
- `oos_test_results_with_timeseries_cv.json` - 时间序列CV测试结果

### 配置文件 (4个)
- `Makefile` - 构建配置
- `pyproject.toml` - Python项目配置
- `requirements.txt` - 依赖配置
- `generate_full_report.sh` - 生成完整报告

### 文档文件 (2个)
- `开始使用.txt` - 开始使用说明
- `快速下载说明.txt` - 快速下载说明

## 📁 新的目录结构

```
ml_project/
├── tools/                        # 🔧 工具脚本
│   ├── analysis/                 # 分析工具
│   │   ├── analyze_enhanced_features.py
│   │   ├── analyze_feature_differences.py
│   │   ├── compare_models.py
│   │   ├── corrected_evaluation.py
│   │   ├── diagnostic_analysis.py
│   │   └── spectral_fix_proposal.py
│   ├── testing/                  # 测试工具
│   │   ├── test_enhanced_oos.py
│   │   ├── test_oos_months.py
│   │   └── test_real_data.py
│   ├── gpu/                      # GPU工具
│   │   └── check_gpu.py
│   └── powershell/               # PowerShell脚本
│       ├── batch_test.ps1
│       ├── download_data.ps1
│       ├── download_to_agg_data.ps1
│       ├── quarterly_rolling.ps1
│       ├── quick_gpu_train.ps1
│       ├── run_gpu_training.ps1
│       ├── START_HERE.ps1
│       ├── test_gpu.ps1
│       ├── train_2021_2023_test_2024_2025.ps1
│       ├── train_and_test.ps1
│       └── train_jan_test_feb.ps1
│
├── data/                         # 📊 数据文件
│   ├── feature_importance/       # 特征重要性数据
│   │   ├── enhanced_feature_importance_by_category.csv
│   │   ├── enhanced_feature_importance_full.csv
│   │   └── feature_importance_5T.csv
│   ├── model_info/               # 模型信息
│   │   ├── model_info_enhanced_may_2025.json
│   │   └── model_info_wavelet_may_2025.json
│   ├── test_results/             # 测试结果
│   │   ├── oos_test_results_enhanced.json
│   │   └── oos_test_results_with_timeseries_cv.json
│   └── analysis_output/          # 分析输出
│       └── enhanced_feature_analysis_output.txt
│
├── config/                       # ⚙️ 配置文件
│   ├── Makefile
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── generate_full_report.sh
│
├── docs/                         # 📚 文档（已整理）
├── scripts/                      # 📜 脚本（已整理）
├── src/                          # 💻 源代码
├── results/                      # 📈 结果文件
└── README.md                     # 主文档
```

## 🎯 分类规则

### 1. tools/ - 工具脚本
- **analysis/** - 分析工具
- **testing/** - 测试工具  
- **gpu/** - GPU相关工具
- **powershell/** - PowerShell脚本

### 2. data/ - 数据文件
- **feature_importance/** - 特征重要性数据
- **model_info/** - 模型信息
- **test_results/** - 测试结果
- **analysis_output/** - 分析输出

### 3. config/ - 配置文件
- 构建配置
- 依赖配置
- 项目配置

## 🚀 执行计划

### 第一步：创建目录结构
```bash
mkdir tools\analysis
mkdir tools\testing  
mkdir tools\gpu
mkdir tools\powershell
mkdir data\feature_importance
mkdir data\model_info
mkdir data\test_results
mkdir data\analysis_output
```

### 第二步：移动文件
```bash
# 移动分析工具
move analyze_enhanced_features.py tools\analysis\
move analyze_feature_differences.py tools\analysis\
move compare_models.py tools\analysis\
move corrected_evaluation.py tools\analysis\
move diagnostic_analysis.py tools\analysis\
move spectral_fix_proposal.py tools\analysis\

# 移动测试工具
move test_enhanced_oos.py tools\testing\
move test_oos_months.py tools\testing\
move test_real_data.py tools\testing\

# 移动GPU工具
move check_gpu.py tools\gpu\

# 移动PowerShell脚本
move *.ps1 tools\powershell\

# 移动数据文件
move enhanced_feature_importance_*.csv data\feature_importance\
move feature_importance_5T.csv data\feature_importance\
move model_info_*.json data\model_info\
move oos_test_results_*.json data\test_results\
move enhanced_feature_analysis_output.txt data\analysis_output\

# 移动配置文件
move Makefile config\
move pyproject.toml config\
move requirements.txt config\
move generate_full_report.sh config\
```

### 第三步：更新引用
- 更新脚本中的相对路径
- 更新文档中的文件引用
- 检查所有链接的有效性

## 📈 预期效果

### 整理前 ❌
- 35个文件散落在根目录
- 难以找到需要的文件
- 项目结构混乱
- 维护困难

### 整理后 ✅
- 文件按功能分类
- 清晰的目录结构
- 易于查找和维护
- 专业的项目组织

## 🎯 分类详情

| 类别 | 文件数量 | 位置 | 说明 |
|------|---------|------|------|
| **分析工具** | 6个 | `tools/analysis/` | 特征分析、模型对比 |
| **测试工具** | 3个 | `tools/testing/` | OOS测试、真实数据测试 |
| **GPU工具** | 1个 | `tools/gpu/` | GPU检查 |
| **PowerShell** | 10个 | `tools/powershell/` | 批处理脚本 |
| **特征数据** | 3个 | `data/feature_importance/` | 特征重要性 |
| **模型信息** | 2个 | `data/model_info/` | 模型元数据 |
| **测试结果** | 2个 | `data/test_results/` | 测试结果 |
| **分析输出** | 1个 | `data/analysis_output/` | 分析输出 |
| **配置文件** | 4个 | `config/` | 项目配置 |

---

**整理目标**: 将35个散落文件分类整理  
**预期效果**: 清晰的项目结构，易于维护  
**执行时间**: 2025-10-22
