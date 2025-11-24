# 如何选择 ts-dim-compare 结果用于 Rolling 训练和回测

## 问题回答

### 1. Rolling train 是否都可以加载 ts-dim-compare 输出的 top factors？

**答案：是的！** 所有 rolling train 命令都支持通过 `--use-top-factors` 参数加载 ts-dim-compare 输出的 `top_factors.json`。

支持的 rolling train 命令：
- `make rolling` - 单配置滚动训练
- `make rolling-multi` - 多配置滚动训练  
- `make auto-rolling-update` - 自动滚动更新（推荐用于生产）

### 2. 应该用哪个 ts-dim-compare 结果去训练生产模型进行回测？

## 选择原则

### 原则 1: 时间范围匹配

选择与你的**训练/回测时间范围最接近**的 ts-dim-compare 结果：

```bash
# 示例：如果你要回测 2024-01-01 到 2024-12-31
# 应该选择：
results/dim_compare/BTCUSDT-ETHUSDT_comprehensive_20240101_20241231_tf60T_h5/top_factors.json

# 而不是：
results/dim_compare/BTCUSDT-ETHUSDT_comprehensive_20200101_20201231_tf60T_h5/top_factors.json  # 太旧
```

**最佳实践**：选择**覆盖时间范围最长**或**最新**的 ts-dim-compare 结果，因为：
- 更长的历史数据能更好地识别稳定特征
- 最新的结果反映了当前市场环境

### 原则 2: 特征类型一致

确保 ts-dim-compare 使用的 `--feature-type` 与训练时一致：

```bash
# ts-dim-compare 使用 comprehensive
make ts-dim-compare DIM_COMPARE_FEATURE_TYPE=comprehensive ...

# 训练时也要使用 comprehensive
make rolling ROLLING_FEATURE_TYPE=comprehensive \
  ROLLING_USE_TOP_FACTORS=results/dim_compare/.../top_factors.json
```

**可用的特征类型**：
- `baseline` - 基础特征
- `comprehensive` - 综合特征（推荐）
- `enhanced` - 增强特征
- `baseline-default-alpha101-orderflow` - 包含 Alpha101 和订单流

### 原则 3: 时间框架和 Forward Bars 匹配

确保 ts-dim-compare 的 `TIMEFRAME` 和 `HORIZONS` 与训练配置一致：

```bash
# 示例：ts-dim-compare 使用 60T 时间框架，forward bars=5
# 目录名包含：tf60T_h5
results/dim_compare/..._tf60T_h5/top_factors.json

# 训练时也要使用相同配置
make rolling ROLLING_FREQ=60T ROLLING_FBS=5 \
  ROLLING_USE_TOP_FACTORS=results/dim_compare/..._tf60T_h5/top_factors.json
```

### 原则 4: 性能指标最优

查看 `production_results.json` 中的性能指标，选择表现最好的：

```bash
# 查看性能指标
cat results/dim_compare/.../production_results.json | jq '.performance.stage3_representatives'
```

关键指标：
- **R²** (r2) - 越高越好
- **MSE/MAE** - 越低越好
- **Financial metrics** - Sharpe ratio, win rate 等

## 推荐工作流程

### 方案 A: 使用最新最全面的 ts-dim-compare 结果（推荐）

```bash
# 1. 运行 ts-dim-compare（使用较长的时间窗口，如最近 2-3 年）
make ts-dim-compare SYMBOLS=BTCUSDT,ETHUSDT \
  START_DATE=2022-01-01 END_DATE=2024-12-31 \
  DIM_COMPARE_FEATURE_TYPE=comprehensive \
  TIMEFRAME=60T \
  HORIZONS=5

# 2. 设置 DIM_DIR 变量
DIM_DIR=results/dim_compare/BTCUSDT-ETHUSDT_comprehensive_20220101_20241231_tf60T_h5

# 3. 使用该结果进行 rolling 训练和回测
make rolling SYMBOLS=BTCUSDT,ETHUSDT \
  ROLLING_START=2024-01 ROLLING_END=2024-12 \
  ROLLING_FEATURE_TYPE=comprehensive \
  ROLLING_FREQ=60T \
  ROLLING_FBS=5 \
  ROLLING_USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json
```

### 方案 B: 使用多个时间窗口的 ts-dim-compare 结果（更稳健）

如果你有多个时间窗口的 ts-dim-compare 结果，可以：

1. **选择表现最好的**：比较不同时间窗口的性能指标
2. **选择最稳定的**：查看特征在不同时间窗口的一致性
3. **使用最新的**：如果市场环境变化较大，使用最新的结果

```bash
# 比较不同时间窗口的结果
# 2020-2021
DIM_DIR_2020=results/dim_compare/..._20200101_20211231_.../top_factors.json
# 2022-2023  
DIM_DIR_2022=results/dim_compare/..._20220101_20231231_.../top_factors.json
# 2024
DIM_DIR_2024=results/dim_compare/..._20240101_20241231_.../top_factors.json

# 选择最新的或表现最好的
DIM_DIR=$DIM_DIR_2024
```

## 实际使用示例

### 示例 1: 使用 2024 年 comprehensive 结果

```bash
# 假设你已经运行了：
# make ts-dim-compare SYMBOLS=BTCUSDT,ETHUSDT START_DATE=2024-01-01 END_DATE=2024-12-31 ...

DIM_DIR=results/dim_compare/BTCUSDT-ETHUSDT_comprehensive_20240101_20241231_tf60T_h5

# 用于 rolling 训练
make rolling SYMBOLS=BTCUSDT,ETHUSDT \
  ROLLING_START=2024-01 ROLLING_END=2024-12 \
  ROLLING_FEATURE_TYPE=comprehensive \
  ROLLING_FREQ=60T \
  ROLLING_FBS=5 \
  ROLLING_USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json
```

### 示例 2: 使用 auto-rolling-update（推荐用于生产）

```bash
DIM_DIR=results/dim_compare/BTCUSDT-ETHUSDT_comprehensive_20240101_20241231_tf60T_h5

make auto-rolling-update SYMBOL=BTCUSDT \
  INITIAL_TRAIN_MONTHS=6 \
  --use-top-factors $(DIM_DIR)/top_factors.json \
  --feature-type comprehensive \
  --freq 60T \
  --forward-bars 5
```

### 示例 3: 手动执行完整流程

```bash
# 步骤 1: 运行 ts-dim-compare
make ts-dim-compare SYMBOLS=BTCUSDT \
  DIM_COMPARE_START=2024-01-01 DIM_COMPARE_END=2024-12-31 \
  DIM_COMPARE_FEATURE_TYPE=comprehensive \
  DIM_COMPARE_FREQ=60T

# 步骤 2: 使用 ts-dim-compare 结果运行 rolling
make rolling SYMBOLS=BTCUSDT \
  ROLLING_USE_TOP_FACTORS=results/dim_compare/.../top_factors.json \
  ROLLING_FEATURE_TYPE=comprehensive \
  ROLLING_FREQ=60T \
  ROLLING_FBS=5 \
  ROLLING_START=2024-01 ROLLING_END=2024-12
```

**注意**: `auto-workflow` 已删除，因为整个研发上线流程（timeframe、feature binning、features挑选）都需要人工分析，无法完全自动化。

## 检查清单

在选择 ts-dim-compare 结果前，确认：

- [ ] **时间范围匹配**：ts-dim-compare 的时间范围覆盖或接近训练/回测时间
- [ ] **特征类型一致**：`DIM_COMPARE_FEATURE_TYPE` = `ROLLING_FEATURE_TYPE`
- [ ] **时间框架一致**：ts-dim-compare 的 `TIMEFRAME` = rolling 的 `ROLLING_FREQ`
- [ ] **Forward bars 一致**：ts-dim-compare 的 `HORIZONS` = rolling 的 `ROLLING_FBS`
- [ ] **性能指标良好**：查看 `production_results.json` 确认性能可接受
- [ ] **文件存在**：确认 `top_factors.json` 文件存在且可读

## 常见问题

### Q: 如果找不到完全匹配的 ts-dim-compare 结果怎么办？

A: 使用**最接近的**结果，优先考虑：
1. 时间范围最接近的
2. 特征类型相同的
3. 时间框架相同的

### Q: 可以使用不同时间框架的 top_factors 吗？

A: **不推荐**。不同时间框架的特征重要性可能不同，应该为每个时间框架运行独立的 ts-dim-compare。

### Q: 多久需要重新运行 ts-dim-compare？

A: 建议：
- **季度更新**：每季度运行一次 ts-dim-compare，使用最近 1-2 年的数据
- **市场变化时**：当市场环境发生重大变化时（如牛熊转换）
- **性能下降时**：当模型性能明显下降时

### Q: 多个 ts-dim-compare 结果可以混合使用吗？

A: **不推荐**。每个 `top_factors.json` 是基于特定时间范围和配置优化的，混合使用可能导致特征不一致。

## 总结

**最佳实践**：
1. ✅ 使用**最新、时间范围最长**的 ts-dim-compare 结果
2. ✅ 确保**所有配置参数一致**（特征类型、时间框架、forward bars）
3. ✅ 优先使用 `auto-rolling-update` 进行生产训练
4. ✅ 定期（季度）重新运行 ts-dim-compare 以保持特征集的时效性

**推荐命令**：
```bash
# 一次性设置
DIM_DIR=results/dim_compare/BTCUSDT-ETHUSDT_comprehensive_20240101_20241231_tf60T_h5

# 用于所有 rolling 训练
ROLLING_USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json
```

