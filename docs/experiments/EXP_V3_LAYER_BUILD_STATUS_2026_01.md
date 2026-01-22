# v3 layer构建状态检查报告

**检查时间**: 2026-01-22  
**目的**: 确认v3 layer是否真的在构建中

---

## 检查结果

### ❌ v3 layer目录不存在

- **路径**: `feature_store/nnmh_highcap6_240T_2024_202510_v3`
- **状态**: 目录不存在
- **结论**: v3 layer可能**没有开始构建**

### 进程检查

- **FeatureStore构建进程**: 未发现真正的构建进程
  - 发现的进程5466只是一个bash shell，不是构建进程
- **结论**: 没有正在运行的FeatureStore构建进程

### 现有layers

- `nnmh_highcap6_240T_2024_202510` - 存在
- `nnmh_highcap6_240T_2024_202510_ma_adx_cvd_vwap_v1` - 存在
- `nnmh_highcap6_240T_2024_202510_ma_adx_v1` - 存在
- `nnmh_highcap6_240T_2024_202510_v2` - 存在（最后修改: 2026-01-22 04:59）
- `nnmh_highcap6_240T_2024_202510_v3` - **不存在**

### TaskSpec配置检查

- **文件**: `config/tasks/task_spec_highcap6_2024_202510.yaml`
- **optional_blocks_enabled**: 空列表 `[]`
- **结论**: 没有明确配置`volume_profile_block`，依赖auto-detect功能

---

## 可能的原因

1. **构建命令没有执行**
   - v3 layer可能根本没有开始构建
   - 之前的检查可能误判了构建状态

2. **构建进程失败**
   - 如果曾经启动过构建，可能已经失败
   - 没有找到构建日志文件

3. **构建还在准备阶段**
   - 构建命令可能还在准备中，目录尚未创建

---

## 建议

### 选项1: 手动启动v3 layer构建（推荐）

如果需要v3 layer（包含volume_profile特征），需要手动执行构建命令：

```bash
mlbot nnmultihead build-feature-store \
  --task-spec config/tasks/task_spec_highcap6_2024_202510.yaml \
  --symbols BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --layer nnmh_highcap6_240T_2024_202510_v3
```

### 选项2: 使用v2 layer继续测试

- v2 layer有vpin特征（30个相关列）
- 但缺少volume_profile特征
- 可以先用v2 layer测试vpin特征的效果

### 选项3: 检查auto-detect功能

- 确认auto-detect功能是否正确识别了volume_profile需求
- 检查`execution_archetypes.yaml`中ET archetype的`has_volume_profile` evidence

---

## 下一步行动

1. **确认是否需要v3 layer**
   - 如果ET策略测试需要volume_profile特征，必须构建v3 layer
   - 如果可以先测试vpin特征，可以使用v2 layer

2. **如果需要v3 layer**
   - 手动执行构建命令
   - 监控构建进度
   - 构建完成后继续ET策略测试

3. **如果使用v2 layer**
   - 继续使用现有的v2 layer
   - 注意：缺少volume_profile特征，ET的`has_volume_profile` evidence会失败

---

## 相关文件

- `config/tasks/task_spec_highcap6_2024_202510.yaml` - TaskSpec配置
- `config/nnmultihead/execution_archetypes.yaml` - ET archetype配置
- `scripts/build_feature_store_nnmultihead.py` - 构建脚本
