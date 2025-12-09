# pytest.ini 配置文件说明

## 什么是 pytest.ini？

`pytest.ini` 是 **pytest 的配置文件**，用于设置 pytest 的默认行为和配置选项。

## 主要作用

### 1. **定义自定义标记（Markers）**

```ini
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

**用途**：
- 可以标记测试为 `slow`、`integration`、`unit` 等
- 运行测试时可以筛选：`pytest -m "not slow"` 跳过慢速测试
- 使用 `--strict-markers` 确保标记必须注册，避免拼写错误

**示例**：
```python
@pytest.mark.slow
def test_expensive_operation():
    # 这个测试会被标记为 slow
    pass
```

运行：
```bash
pytest -m "not slow"  # 跳过所有 slow 测试
pytest -m slow        # 只运行 slow 测试
```

### 2. **设置测试发现模式**

```ini
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

**用途**：
- 告诉 pytest 哪些文件、类、函数是测试
- `test_*.py`：所有以 `test_` 开头的文件
- `Test*`：所有以 `Test` 开头的类
- `test_*`：所有以 `test_` 开头的函数

### 3. **设置默认选项**

```ini
addopts = 
    -ra
    --strict-markers
    --tb=short
```

**含义**：
- `-ra`：显示所有测试摘要（passed, failed, skipped 等）
- `--strict-markers`：严格模式，未注册的标记会导致错误
- `--tb=short`：简短的错误追踪格式

### 4. **设置 Python 版本要求**

```ini
minversion = 6.0
```

要求 pytest 版本至少为 6.0。

## 为什么需要 pytest.ini？

### ✅ 好处

1. **统一配置**：所有测试使用相同的配置
2. **减少命令行参数**：不需要每次运行都加 `-v -ra --strict-markers`
3. **团队协作**：确保所有人使用相同的测试配置
4. **CI/CD 友好**：CI 环境自动使用这些配置

### 对比

**没有 pytest.ini**：
```bash
pytest tests/ -v -ra --strict-markers -m "not slow"
```

**有 pytest.ini**：
```bash
pytest tests/ -m "not slow"  # 其他选项已在配置文件中
```

## 当前项目的 pytest.ini

```ini
[pytest]
# 自定义标记
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests

# 测试发现模式
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# 默认选项
addopts = 
    -ra
    --strict-markers
    --tb=short

# 最低版本
minversion = 6.0
```

## 使用示例

### 运行所有测试
```bash
pytest
# 等同于：pytest -ra --strict-markers --tb=short
```

### 运行快速测试（跳过 slow）
```bash
pytest -m "not slow"
```

### 只运行集成测试
```bash
pytest -m integration
```

### 运行特定文件
```bash
pytest tests/integration/test_factor_eval_integration.py -v
```

## 相关文件

- `tests/conftest.py`：pytest fixtures 和共享配置
- `pytest.ini`：pytest 默认配置
- 测试文件：使用 `@pytest.mark.xxx` 标记测试

## 总结

`pytest.ini` 是 pytest 的**全局配置文件**，用于：
- ✅ 定义测试标记
- ✅ 设置测试发现规则
- ✅ 配置默认选项
- ✅ 统一团队测试环境

无需在每个测试文件或命令行中重复配置，**一次设置，全局生效**。

