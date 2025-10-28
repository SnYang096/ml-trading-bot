# Feature Count Tool - 特征统计工具

## 概述

特征统计工具用于统计项目中所有特征工程模块生成的特征数量，并生成详细的分析报告。

## 功能特性

- ✅ 统计所有5个特征工程模块的特征数量
- ✅ 生成文本格式报告（TXT）
- ✅ 生成JSON格式数据（便于程序解析）
- ✅ 生成HTML可视化报告（带图表）
- ✅ 集成到Makefile（Linux/Mac）
- ✅ 支持Windows批处理和PowerShell脚本

## 统计结果总览

| 模块 | 文件 | 特征数 | 占比 |
|------|------|--------|------|
| 基础特征工程 | `feature_engineering.py` | 13 | 2.6% |
| 改进特征工程 | `feature_engineering_improved.py` | 25 | 5.0% |
| 增强特征工程 | `feature_engineering_enhanced.py` | 331 | 65.7% |
| 小波特征工程 | `feature_engineering_wavelet.py` | 71 | 14.1% |
| 深度学习特征 | `dl_sequence_features.py` | 64 | 12.7% |
| **总计** | | **504** | **100%** |

## 使用方法

### 方法 1: 直接运行Python脚本（推荐，所有平台）

```bash
cd ml_project
python scripts/analysis/count_features.py
python scripts/analysis/generate_html_report.py  # 可选：生成HTML报告
```

### 方法 2: 使用Makefile (Linux/Mac/WSL/Git Bash)

```bash
cd ml_project/config
make count-features
```

这个命令会：
1. 统计所有特征
2. 生成文本报告
3. 生成JSON数据
4. 生成HTML可视化报告

### 方法 3: Windows批处理文件

双击运行或在命令行执行：
```cmd
cd ml_project\scripts\analysis
count_features.bat
```

### 方法 4: PowerShell脚本 (Windows)

```powershell
cd ml_project\scripts\analysis
.\count_features.ps1
```

## 输出文件

所有报告保存在 `ml_project/reports/` 目录：

1. **feature_count_report.txt** - 详细的文本报告（中文）
   - 每个模块的特征分类
   - 特征列表（前5个示例）
   - 使用建议

2. **feature_count_data.json** - 结构化JSON数据
   - 完整的特征列表
   - 模块元数据
   - 时间戳
   - 便于程序解析和二次处理

3. **feature_count_report.html** - 可视化HTML报告
   - 交互式图表（使用Chart.js）
   - 美观的卡片式布局
   - 详细的模块信息
   - 可在浏览器中查看

## 特征详细分类

### 1. 基础特征工程 (13个)

```python
# feature_engineering.py
- RSI (1)
- MACD (3): macd, macd_signal, macd_histogram
- Bollinger Bands (3): bb_upper, bb_middle, bb_lower
- ATR (1)
- ZigZag (1)
- Price Features (2): price_change, volatility
- Volume Features (2): volume_sma, volume_ratio
```

### 2. 改进特征工程 (25个)

```python
# feature_engineering_improved.py
包含基础特征 + 以下新增：
- Normalized Features (4): bb_position, rsi_normalized, macd_normalized, atr_normalized
- Momentum Features (3): momentum_5, momentum_10, momentum_20
- Moving Averages (5): sma_5, sma_10, sma_20, sma_ratio_5_20, sma_ratio_10_20
```

### 3. 增强特征工程 (331个) ⭐ 最强大

```python
# feature_engineering_enhanced.py
- Basic Features (35+): 价格、波动率、动量、RSI、BB、MACD、成交量
- Hurst Features (30): 对5个信号源（close, open, volume, cvd, taker_buy_ratio）
  * 每个信号源6个特征：hurst, deviation, trend_signal, mean_revert_signal, change, acceleration
- Wavelet Packet Transform (180): 对5个信号源
  * 每个信号源36个特征：8个节点×4特征 + 4个全局特征
- Hilbert Transform (15): 对5个信号源，每个3个特征
- Spectral Analysis (15): 对5个信号源，每个3个特征
- Advanced Derived (26): 压缩、波动率、结构等高级特征
- Order Flow (36): 订单流不平衡、背离、流动性等
```

### 4. 小波特征工程 (71个)

```python
# feature_engineering_wavelet.py
包含改进特征 + 以下新增：
- Wavelet Features (Close) (20): db4小波分解，4层
- Wavelet Features (Volume) (20): volume的小波特征
- Hilbert Transform (3): 瞬时幅度、相位、频率
- Spectral Analysis (3): 频谱质心、带宽、滚降
```

### 5. 深度学习序列特征 (64个，可配置)

```python
# dl_sequence_features.py
- Deep Learning Sequence Features (默认64个)
  * 使用 Mamba/Transformer 提取序列模式
  * 可配置维度：32, 64, 128, 256
  * 支持 FP16 混合精度
  * 支持 Flash Attention 加速
```

## 建议使用场景

### 快速原型 → 基础版 (13个特征)
```python
from ml_trading.data_tools.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.engineer_features(multi_tf_data)
```
- ✅ 最快速度
- ✅ 最低计算成本
- ✅ 适合快速验证想法

### 标准训练 → 改进版 (25个特征)
```python
from ml_trading.data_tools.feature_engineering_improved import ImprovedFeatureEngineer

engineer = ImprovedFeatureEngineer(scaler_type='standard')
features = engineer.engineer_features(multi_tf_data)
```
- ✅ 快速计算
- ✅ 包含归一化
- ✅ 适合日常训练

### 高级研究 → 增强版 (331个特征) ⭐
```python
from ml_trading.data_tools.feature_engineering_enhanced import EnhancedFeatureEngineer

engineer = EnhancedFeatureEngineer(
    wavelet='db4',
    wpt_level=3,
    hurst_window=100
)
features = engineer.engineer_features(multi_tf_data)
```
- ⚠️ 计算密集（WPT + Hurst）
- ✅ 最全面的特征集
- ✅ 适合高级研究和竞赛
- 💡 建议使用特征选择（选top 100-200）

### 深度学习增强 → DL序列特征 (64个)
```python
from ml_trading.data_tools.dl_sequence_features import add_dl_sequence_features

df_with_dl = add_dl_sequence_features(
    df,
    backend='mamba',  # or 'flash_attention', 'transformer'
    seq_length=120,
    d_model=64,
    use_fp16=True
)
```
- ✅ 捕获长期依赖
- ✅ 自动特征学习
- 💡 需要GPU加速
- 💡 可与传统特征组合使用

## 特征选择建议

### 维度灾难问题

504个特征如果全部使用，可能导致：
- 🔴 过拟合
- 🔴 训练缓慢
- 🔴 模型复杂度过高

### 推荐策略

1. **特征重要性分析**
   ```python
   # 使用LightGBM内置的特征重要性
   importance = model.feature_importance(importance_type='gain')
   top_features = features[importance.argsort()[-100:]]  # Top 100
   ```

2. **相关性分析**
   ```python
   # 移除高度相关的特征
   correlation_matrix = df.corr()
   # 如果相关性 > 0.95，保留一个
   ```

3. **递归特征消除 (RFE)**
   ```python
   from sklearn.feature_selection import RFE
   selector = RFE(estimator, n_features_to_select=100)
   ```

4. **分阶段使用**
   - 第一阶段：基础版（13个） → 快速验证
   - 第二阶段：改进版（25个） → 标准训练
   - 第三阶段：增强版（选100个） → 精细优化
   - 第四阶段：添加DL特征（64个） → 最终提升

## 计算成本对比

| 模块 | 相对速度 | 内存占用 | GPU需求 |
|------|---------|---------|---------|
| 基础版 | 1x (最快) | 低 | ❌ |
| 改进版 | 1.2x | 低 | ❌ |
| 增强版 | 10x (WPT+Hurst) | 中 | ❌ |
| 小波版 | 5x | 中 | ❌ |
| 深度学习 | 3x | 中-高 | ✅ 推荐 |

## 集成到训练流程

```python
# 示例：使用增强版 + 特征选择
from ml_trading.data_tools.feature_engineering_enhanced import EnhancedFeatureEngineer
import lightgbm as lgb
import numpy as np

# 1. 生成所有特征
engineer = EnhancedFeatureEngineer()
features_dict = engineer.engineer_features(multi_tf_data)

# 2. 训练模型获取特征重要性
X_train = features_dict['5T'].drop(columns=['open', 'high', 'low', 'close', 'volume'])
y_train = generate_labels(features_dict['5T'])

model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# 3. 选择top特征
importance = model.feature_importance(importance_type='gain')
top_100_idx = np.argsort(importance)[-100:]
top_100_features = X_train.columns[top_100_idx]

print(f"Selected {len(top_100_features)} features")
print(top_100_features.tolist())

# 4. 保存特征列表以供推理使用
import json
with open('selected_features.json', 'w') as f:
    json.dump(top_100_features.tolist(), f)
```

## 文件结构

```
ml_project/
├── scripts/
│   └── analysis/
│       ├── count_features.py          # 主统计脚本
│       ├── generate_html_report.py    # HTML报告生成器
│       ├── count_features.bat         # Windows批处理
│       └── count_features.ps1         # PowerShell脚本
├── reports/
│   ├── feature_count_report.txt       # 文本报告
│   ├── feature_count_data.json        # JSON数据
│   ├── feature_count_report.html      # HTML可视化
│   └── README.md                      # 报告说明文档
├── config/
│   └── Makefile                       # Make命令（含count-features）
└── docs/
    └── feature_count_tool.md          # 本文档
```

## 扩展和自定义

### 添加新的特征模块

如果你添加了新的特征工程模块，可以轻松扩展统计工具：

1. 在 `count_features.py` 中添加新函数：
```python
def count_new_module_features() -> Dict[str, List[str]]:
    """统计新模块的特征."""
    features = {
        'Category 1': ['feat1', 'feat2'],
        'Category 2': ['feat3', 'feat4']
    }
    return features
```

2. 在 `generate_report()` 中添加统计：
```python
new_features = count_new_module_features()
all_features['new_module'] = new_features
```

## 常见问题 (FAQ)

**Q: 为什么特征数量这么多？**
A: 增强版使用了WPT、Hurst、Hilbert等高级信号处理方法，对多个信号源进行分解，因此特征数量较多。实际使用时建议通过特征选择筛选出top 100-200个。

**Q: 应该使用哪个模块？**
A: 
- 快速原型：基础版
- 日常训练：改进版
- 竞赛/研究：增强版（带特征选择）
- 深度学习：添加DL序列特征

**Q: 如何减少计算时间？**
A:
1. 使用更快的模块（基础版/改进版）
2. 减少WPT的层数（level=2 instead of 3）
3. 减少Hurst窗口大小（window=50 instead of 100）
4. 使用GPU加速（深度学习特征）

**Q: JSON数据如何使用？**
A:
```python
import json
with open('reports/feature_count_data.json', 'r') as f:
    data = json.load(f)

# 获取所有特征名称
for module in data['modules'].values():
    for category, features in module['features'].items():
        print(f"{category}: {features}")
```

## 参考资料

- [特征工程最佳实践](../docs/特征设计需要注意的地方.md)
- [归一化方法](../docs/归一化方法.md)
- [50个中低频features](../docs/50个中低频features和归一化方法.md)
- [小波变换分析](../docs/小波变换和传统特征提取优缺点分析.md)

---

**最后更新**: 2025-10-22
**作者**: ML Trading Project Team

