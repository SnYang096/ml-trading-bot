# PCM 仲裁机制对比验证

## 概述
该脚本用于对比不同PCM（Portfolio Construction Model）仲裁方案的效果，验证仲裁机制的必要性和有效性。

## 设计背景
在多策略联合回测中，多个策略可能会在相同时间对相同资产发出交易信号，需要通过PCM进行仲裁。本工具提供三种不同的仲裁方案进行对比：

1. **方案A (优先级模式)**: 现有PCM仲裁机制，采用LV > FER > ME > BPC的优先级
2. **方案B (多Slot模式)**: 每个archetype拥有独立的slot，BPC/ME可加仓，FER不加仓

## 使用方法

### 基本用法
```bash
python pcm_arbitration_comparison.py \
  --strategy-paths \
    bpc:results/train_final_XXXXX_rr_extreme/bpc/predictions.parquet \
    me:results/train_final_XXXXX_rr_extreme/me/predictions.parquet \
    fer:results/train_final_XXXXX_rr_extreme/fer/predictions.parquet
```

### 参数说明
- `--strategy-paths`: 指定各策略的预测结果文件路径，格式为 `strategy:path` 对
- `--output`: 输出报告路径（默认: pcm_comparison_report.json）

## 评估指标

### 主要KPI
- **总交易数**: 对比不同方案的交易频率
- **夏普比率**: 风险调整后收益对比
- **平均R**: 单笔交易平均收益
- **胜率**: 盈利交易占比
- **总收益**: 累计收益对比

### 对比维度
1. **效率对比**: 不同仲裁方案的交易数量和频率
2. **收益对比**: 不同仲裁方案的风险调整后收益
3. **风险对比**: 不同仲裁方案的风险分散效果
4. **冲突解决能力**: 不同方案处理策略冲突的效果

## 输出结果

### 文件输出
1. **控制台结果**: 直接在命令行显示对比结果
2. **可视化图表**: `pcm_comparison_results.png` - KPI对比图表
3. **详细报告**: `pcm_comparison_report.json` - 完整分析报告

### 推荐逻辑
脚本会基于夏普比率等关键指标，推荐表现最优的PCM方案。

## 实现原理

### 方案A - 优先级模式
- 按照LV > FER > ME > BPC优先级处理信号
- 同一时间、同一资产只允许最高优先级策略的信号

### 方案B - 多Slot模式  
- 每个策略类型拥有独立的交易slot
- BPC和ME策略允许加仓（可多次交易）
- FER策略不允许加仓（同一时间只允许一个信号）

### 方案C - 两Slot模式
- 整体最多只允许两个并发交易信号
- 按优先级顺序分配slot，限制过度交易

## 应用场景

### 适用条件
- 多策略联合回测环境
- 存在策略间信号冲突的情况
- 需要优化PCM仲裁机制

### 验证目的
- 验证现有PCM仲裁机制的有效性
- 对比不同仲裁策略的优劣
- 为PCM机制优化提供数据支撑

## 注意事项
1. 预测结果文件必须包含timestamp、symbol、entry_direction、rr等字段
2. 所有策略的时间范围需要一致才能有效对比
3. 建议在相同市场环境下测试不同方案以保证公平性