# 🚀 降维训练系统使用指南

## 📁 文件结构

```
ml_project/scripts/
├── dimensionality/                    # 降维训练主目录
│   ├── 01_feature_engineering.py    # 特征工程和IC/IR筛选
│   ├── 02_rolling_training.py        # 滚动训练和季度评估
│   ├── 03_production_training.py    # 生产级训练
│   ├── 04_integration.py            # 集成部署
│   ├── 05_report_generator.py       # 综合报告生成
│   └── 06_dimensionality_pipeline.py # 降维管道
├── common/                          # 通用工具
│   ├── autoencoder.py              # 统一Autoencoder类
│   ├── data_loader.py              # 数据加载工具
│   └── training_utils.py           # 训练工具
└── README_dimensionality.md         # 本文件
```

## 🎯 使用流程

### 阶段1: 特征工程 (01_feature_engineering.py)
```bash
cd ml_project/scripts/dimensionality
python 01_feature_engineering.py
```

**功能:**
- ✅ 生成1000+特征
- ✅ IC/IR筛选高质量特征
- ✅ 特征类型分析
- ✅ 降维方法对比 (Autoencoder vs PCA)

**输出:**
- `results/feature_engineering_YYYYMMDD_HHMMSS.json`

### 阶段2: 滚动训练 (02_rolling_training.py)
```bash
# 季度滚动训练
python 02_rolling_training.py --mode quarterly --symbol ETH-USD

# 漂移触发训练
python 02_rolling_training.py --mode drift-triggered --symbol ETH-USD
```

**功能:**
- ✅ 2024年数据训练，2025年数据测试
- ✅ 季度滚动评估
- ✅ 漂移检测和自动重训练
- ✅ 性能对比分析

**输出:**
- `results/rolling_dim_ETH_USD/final_results_*.json`
- `results/rolling_dim_ETH_USD/summary_report.json`

### 阶段3: 生产训练 (03_production_training.py)
```bash
python 03_production_training.py
```

**功能:**
- ✅ 生产级模型训练
- ✅ 真实数据加载
- ✅ 模型保存和部署准备
- ✅ 性能评估

**输出:**
- `results/production_dimensionality_*/production_results.json`
- `results/production_dimensionality_*/production_model.pkl`
- `results/production_dimensionality_*/production_autoencoder.pth`

### 阶段4: 集成部署 (04_integration.py)
```bash
python 04_integration.py
```

**功能:**
- ✅ 模型集成到实际系统
- ✅ 实时预测演示
- ✅ 性能监控

### 阶段5: 报告生成 (05_report_generator.py)
```bash
python 05_report_generator.py
```

**功能:**
- ✅ 综合性能报告
- ✅ 方法对比分析
- ✅ 可视化图表
- ✅ 推荐建议

## 🔧 参数配置

### 特征工程参数
```python
# IC/IR筛选阈值
ic_threshold = 0.05      # IC阈值
ir_threshold = 0.1      # IR阈值

# 降维参数
encoding_dim = 8         # 编码维度
architecture = 'deep'    # 网络架构
```

### 滚动训练参数
```python
# 模型参数
encoding_dim = 8         # 编码维度
drift_threshold = 0.3    # 漂移阈值
min_improvement = 0.005  # 最小改进阈值

# 数据参数
train_data_path = "data/train_2024.csv"
test_data_path = "data/test_2025.csv"
symbol = "ETH-USD"
```

## 📊 输出文件说明

### 特征工程输出
```json
{
  "timestamp": "20250124_143022",
  "total_features": 1200,
  "filtered_features": 450,
  "feature_type_analysis": {
    "hurst_features": 45,
    "wpt_features": 120,
    "hilbert_features": 80,
    "spectral_features": 60,
    "order_flow_features": 30
  },
  "dimensionality_results": {
    "autoencoder": {
      "accuracy": 0.7234,
      "compression_ratio": 56.25
    },
    "pca": {
      "accuracy": 0.6891,
      "compression_ratio": 56.25
    }
  }
}
```

### 滚动训练输出
```json
{
  "symbol": "ETH-USD",
  "summary_statistics": {
    "training_original_r2": 0.6543,
    "training_compressed_r2": 0.7123,
    "training_improvement": 0.058,
    "average_test_r2": 0.6891,
    "compression_ratio": 56.25
  },
  "recommendations": [
    "✅ Dimensionality reduction shows significant improvement",
    "✅ Model shows good generalization on test data"
  ]
}
```

## 🚀 快速开始

### 1. 完整流程
```bash
# 1. 特征工程
python 01_feature_engineering.py

# 2. 滚动训练
python 02_rolling_training.py --mode quarterly --symbol ETH-USD

# 3. 生产训练
python 03_production_training.py

# 4. 集成部署
python 04_integration.py

# 5. 生成报告
python 05_report_generator.py
```

### 2. 单独运行
```bash
# 只做特征工程
python 01_feature_engineering.py

# 只做滚动训练
python 02_rolling_training.py --mode quarterly

# 只做生产训练
python 03_production_training.py
```

## 🔍 故障排除

### 常见问题

1. **特征数量不足**
   - 检查 `ComprehensiveFeatureEngineer` 是否正常工作
   - 调整特征工程参数

2. **IC/IR筛选后特征太少**
   - 降低 `ic_threshold` 和 `ir_threshold`
   - 检查数据质量

3. **降维效果不佳**
   - 尝试不同的 `encoding_dim`
   - 调整网络架构
   - 增加训练轮数

4. **模型性能差**
   - 检查特征工程质量
   - 调整模型参数
   - 增加训练数据

### 调试模式
```bash
# 启用详细输出
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python -u 01_feature_engineering.py 2>&1 | tee feature_engineering.log
```

## 📈 性能监控

### 关键指标
- **特征数量**: 目标1000+特征
- **IC/IR筛选率**: 目标30-50%通过率
- **压缩比**: 目标50x以上
- **性能提升**: 目标5%以上R²提升
- **测试R²**: 目标0.6以上

### 监控脚本
```bash
# 检查结果文件
ls -la results/

# 查看最新结果
tail -f results/feature_engineering_*.json

# 性能对比
python -c "
import json
with open('results/feature_engineering_*.json') as f:
    data = json.load(f)
    print(f'Features: {data[\"total_features\"]} -> {data[\"filtered_features\"]}')
    print(f'Compression: {data[\"dimensionality_results\"][\"autoencoder\"][\"compression_ratio\"]:.1f}x')
"
```

## 🎯 最佳实践

1. **数据质量**: 确保输入数据质量，处理缺失值和异常值
2. **特征工程**: 使用多种特征类型，包括技术指标、统计特征、序列特征
3. **IC/IR筛选**: 设置合理的阈值，平衡特征数量和质量
4. **降维方法**: 对比不同方法，选择最适合的
5. **滚动训练**: 定期重训练，监控性能变化
6. **生产部署**: 使用生产级模型，确保稳定性

## 📞 支持

如有问题，请检查：
1. 日志文件
2. 结果文件
3. 错误信息
4. 数据路径

或联系开发团队获取支持。
