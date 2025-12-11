# Optuna 测试运行说明

## 问题

在本地环境运行 Optuna 相关测试时，由于依赖问题，很多测试会被跳过：
```
SKIPPED [1] tests/test_ts_sr_reversal_optuna.py:30: Cannot import ts_sr_reversal_optuna due to dependencies
```

## 解决方案

**所有 Optuna 测试都应该在 Docker 中运行**，以确保完整的依赖支持。

## 使用方式

### 方式 1：使用 Makefile（推荐）

```bash
# 运行所有 Optuna 测试
make test-optuna-all

# 分别运行不同类型的测试
make test-optuna              # 阈值优化测试
make test-optuna-joint        # 联合优化测试
make test-optuna-imbalanced   # 不平衡数据处理测试
make test-optuna-integration  # 集成测试
```

### 方式 2：直接使用 Docker

```bash
# 运行所有 Optuna 测试
docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  --user $(id -u):$(id -g) \
  -e PYTHONPATH=/workspace/src \
  -e PYTHONUNBUFFERED=1 \
  -v $(pwd):/workspace \
  -v $(pwd)/data/parquet_data:/workspace/data/parquet_data \
  -w /workspace \
  --shm-size=8gb \
  hansenlovefiona017/lightgbm-runtime:v0.0.6 \
  pytest tests/test_ts_sr_reversal_optuna.py \
         tests/test_ts_sr_reversal_optuna_joint.py \
         tests/test_optuna_imbalanced_data.py \
         tests/integration/test_optimization_integration.py::TestTSRReversalOptuna \
         tests/integration/test_optimization_integration.py::test_optimization_scripts_importable \
         tests/integration/test_ts_sr_reversal_optuna_integration.py \
         -v
```

### 方式 3：使用 pytest（在 Docker 容器内）

```bash
# 进入 Docker 容器
make builder-shell

# 在容器内运行测试
pytest tests/test_ts_sr_reversal_optuna.py -v
pytest tests/test_ts_sr_reversal_optuna_joint.py -v
pytest tests/test_optuna_imbalanced_data.py -v
```

## 测试文件清单

### 单元测试
- `tests/test_ts_sr_reversal_optuna.py` - 阈值优化测试（9个测试用例）
- `tests/test_ts_sr_reversal_optuna_joint.py` - 联合优化测试（6个测试用例）
- `tests/test_optuna_imbalanced_data.py` - 不平衡数据处理测试（8个测试用例）

### 集成测试
- `tests/integration/test_optimization_integration.py` - 优化集成测试
- `tests/integration/test_ts_sr_reversal_optuna_integration.py` - Optuna 集成测试

## 预期结果

在 Docker 中运行后，所有测试应该能够：
- ✅ 成功导入所有模块
- ✅ 运行所有测试用例（不再跳过）
- ✅ 验证优化目标选择逻辑
- ✅ 验证不平衡数据约束
- ✅ 验证配置更新逻辑

## 为什么需要在 Docker 中运行？

### 依赖问题

Optuna 优化脚本依赖：
- `src.time_series_model.strategies.evaluation.strategy_feature_compare`
- `src.strategy_config.StrategyConfigLoader`
- `src.data_tools.data_utils.load_raw_data`
- 这些模块又依赖其他模块（如 `time_series_model.config.settings`）

在本地环境中，这些依赖可能：
- 路径配置不正确
- 缺少某些模块
- Python 路径设置不完整

### Docker 环境的优势

- ✅ 完整的依赖环境
- ✅ 正确的 Python 路径配置
- ✅ 所有必需的包都已安装
- ✅ 与生产环境一致

## 快速验证

```bash
# 检查 Docker 是否运行
docker ps

# 运行快速测试验证
make test-optuna

# 查看测试结果
# 应该看到所有测试运行，而不是跳过
```

## 注意事项

1. **首次运行**：如果 Docker 镜像不存在，需要先构建：
   ```bash
   make docker-build
   ```

2. **数据目录**：确保 `data/parquet_data` 目录存在（即使为空）

3. **权限问题**：Docker 容器以当前用户身份运行，确保有读写权限

4. **GPU 支持**：如果使用 GPU，确保 Docker 有 GPU 访问权限

## 相关命令

```bash
# 查看所有测试命令
make help | grep test-optuna

# 运行特定测试
make test-optuna

# 查看测试详细输出
make test-optuna-all -v
```

## 总结

✅ **推荐做法**：
- 始终在 Docker 中运行 Optuna 测试
- 使用 `make test-optuna-all` 运行所有测试
- 在 CI/CD 中也使用 Docker 运行测试

✅ **优势**：
- 所有测试都能运行（不再跳过）
- 环境一致性
- 与生产环境匹配

