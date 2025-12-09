# 集成测试目录

这个目录包含需要完整数据环境的集成测试。

## 📁 目录结构

```
tests/integration/
├── __init__.py                    # 包初始化
├── conftest.py                    # Pytest fixtures 和配置
├── README.md                      # 本文件
├── test_example.py                # 示例集成测试（环境验证）
└── test_dimensionality_comparison_integration.py  # 维度比较集成测试
```

## 🎯 测试范围

集成测试验证以下功能：

1. **数据加载和特征工程**
   - 从 parquet 文件加载市场数据
   - 特征工程管道
   - 特征依赖解析

2. **策略配置**
   - 策略配置文件加载
   - 标签生成
   - 特征选择

3. **维度比较流程**
   - 三阶段特征选择（过滤 → IC 排名 → 相关性选择）
   - 模型训练和评估
   - 结果输出（top_factors.json, results.json）

4. **因子评估工具**
   - `factor_ts_eval.py` - 时间序列因子评估（`make ts-factor-eval`）
   - `cross_sectional_eval.py` - 横截面因子评估（`make cs-factor-eval`）
   - IC 计算、衰减分析、分位数分析等

## 🚀 运行集成测试

### 前提条件

1. **Python 环境**：
   ```bash
   # 确保所有依赖已安装
   pip install -r requirements.txt
   ```

2. **测试数据**：
   - 集成测试会自动生成测试数据
   - 如果需要使用真实数据，确保 `data/parquet_data/` 目录存在

### 运行所有集成测试

```bash
# 运行所有集成测试
pytest tests/integration/ -v

# 运行特定测试文件
pytest tests/integration/test_dimensionality_comparison_integration.py -v

# 运行特定测试类
pytest tests/integration/test_dimensionality_comparison_integration.py::TestDimensionalityComparisonIntegration -v

# 运行特定测试方法
pytest tests/integration/test_dimensionality_comparison_integration.py::TestDimensionalityComparisonIntegration::test_run_dim_compare_basic -v
```

### 在 Docker 中运行

```bash
# 使用 Makefile
make test-integration

# 或直接使用 Docker
docker run --rm -v $(pwd):/workspace -w /workspace \
  hansenlovefiona017/lightgbm-runtime:v0.0.5 \
  pytest tests/integration/ -v
```

### 运行慢速测试（使用真实配置）

```bash
# 运行标记为 'slow' 的测试
pytest tests/integration/ -v -m slow

# 跳过慢速测试
pytest tests/integration/ -v -m "not slow"
```

## 📊 Fixtures 说明

### `integration_test_dir`
临时测试目录，用于存储测试生成的文件。

### `integration_data_dir`
数据目录，包含生成的 parquet 市场数据。

### `integration_config_dir`
策略配置目录，包含测试用的策略配置文件。

### `generate_market_data`
数据生成函数，可以生成指定数量的市场数据样本。

### `setup_strategy_config`
设置测试策略配置（features.yaml, labels.yaml, model.yaml, evaluation.yaml）。

### `integration_env`
完整的集成测试环境，包含：
- 数据目录路径
- 配置目录路径
- 生成的测试数据文件路径
- 测试参数（symbol, timeframe）

## 🔧 测试环境设置

集成测试会自动：

1. **生成测试数据**：
   - 创建真实的 OHLCV 数据
   - 包含订单流数据（cvd, taker_buy_ratio）
   - 保存为 parquet 格式

2. **创建策略配置**：
   - 生成完整的策略配置文件
   - 包含特征、标签、模型、评估配置

3. **设置测试环境**：
   - 创建临时目录结构
   - 配置数据路径
   - 准备测试参数

## 📝 添加新的集成测试

1. 在 `tests/integration/` 目录下创建新的测试文件
2. 使用 `integration_env` fixture 获取测试环境
3. 编写测试用例，验证完整流程

示例：

```python
def test_my_integration(integration_env):
    """我的集成测试"""
    # 使用 integration_env 中的数据路径和配置路径
    data_dir = integration_env["data_dir"]
    config_dir = integration_env["config_dir"]
    
    # 运行测试逻辑
    result = my_function(data_dir, config_dir)
    
    # 验证结果
    assert result is not None
```

## ⚠️ 注意事项

1. **测试数据**：
   - 集成测试会生成临时测试数据
   - 测试结束后会自动清理
   - 如果需要保留数据用于调试，可以修改 fixture 的 scope

2. **运行时间**：
   - 集成测试需要完整的特征工程流程
   - 单个测试可能需要几分钟
   - 使用 `-m "not slow"` 跳过耗时的测试

3. **依赖**：
   - 确保所有特征依赖都已正确配置
   - 确保策略配置文件格式正确
   - 确保标签生成器可以正常导入

4. **Docker 环境**：
   - 推荐在 Docker 中运行集成测试
   - 确保 Docker 镜像包含所有必需的包
   - 确保数据目录可以正确挂载

## 🐛 故障排除

### 问题：导入错误

```bash
# 确保项目根目录在 Python 路径中
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/integration/ -v
```

### 问题：数据加载失败

检查：
- parquet 文件格式是否正确
- 数据目录路径是否正确
- 文件权限是否正确

### 问题：特征计算失败

检查：
- `config/feature_dependencies.yaml` 是否存在
- 特征函数是否可以正确导入
- 特征依赖是否完整

## 📚 相关文档

- [测试准备完成总结](../../docs/测试准备完成总结.md)
- [Docker 启动说明](../../docs/Docker启动说明.md)
- [时序模型重构总结](../../docs/时序模型/重构总结.md)

