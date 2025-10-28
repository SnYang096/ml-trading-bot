# DynamicSR策略快速参考

## 🚀 快速开始

```bash
# 运行回测
make backtest-dynamic-sr-btc

# 查看报告
nautilus_project/reports/dynamic_sr_report.html
```

## 📊 当前状态 (v2.1 - CVD Enhanced)

### 最新回测结果
- **总盈亏**: -529.09 USDT
- **交易数**: 34 (做多: 1, 做空: 33)
- **做多**: 100%胜率, +968.85 USDT
- **做空**: 0%胜率, -1497.94 USDT
- **改进**: 相比v2.0改善 +943 USDT (+64%)

### 核心问题
❌ **做空逻辑需要优化** - 33笔全亏

## ⚙️ 推荐配置

### 方案A: 只做多（稳健）
```yaml
# config.yaml
multi_timeframe_reverse:
  allow_reverse_on_smaller_tf: false  # 禁用做空

state_filter:
  enabled: true
  allowed_states: ["accumulation", "expansion"]
```
**预期**: +968.85 USDT, 100%胜率（基于当前数据）

### 方案B: 提高门槛（平衡）
```yaml
# config.yaml
min_confidence_to_fire: 0.5  # 原0.3

multi_timeframe_reverse:
  allow_reverse_on_smaller_tf: true
  reverse_size_limit_pct: 0.1  # 做空仅10%仓位

state_filter:
  enabled: true
  allowed_states: ["accumulation", "expansion", "compression"]
```
**预期**: 更少但更高质量的信号

### 方案C: 使用预设
```bash
# 复制预设到config.yaml
cat config_presets.yaml  # 查看所有预设
```

可用预设：
- `long_only` - 只做多
- `trend_following` - 趋势追踪
- `conservative` - 保守模式
- `balanced` - 平衡模式（当前）

## 📁 重要文件

### 配置
- `config.yaml` - 主配置文件
- `config_presets.yaml` - 预设配置集合

### 代码
- `strategy.py` - 策略主逻辑 (947行)
- `state_detector.py` - 市场状态检测 (212行)
- `confluence_layer.py` - 多周期融合
- `nautilus_backtest.py` - 回测和报告 (962行)

### 文档
- `UPDATE_SUMMARY.md` - 完整更新总结
- `CONFIG_GUIDE.md` - 详细配置指南
- `CVD_IMPROVEMENT.md` - CVD改进说明
- `CVD_IMPROVEMENT_RESULTS.md` - CVD效果报告

### 输出
- `reports/dynamic_sr_report.html` - 可视化报告
- `reports/dynamic_sr_trade_context.csv` - 信号上下文
- `reports/dynamic_sr_positions_report.csv` - 仓位明细

## 🎯 关键功能

### 1. CVD计算 (v2.1新增)
```python
# 三层优先级
1. buy_volume - sell_volume  # 最准确
2. aggressor_side × volume   # 较准确
3. price_direction × volume  # 兜底
```

### 2. 多空分离统计
- 独立显示多头/空头的胜率、盈亏
- 一目了然看出哪个方向更好

### 3. 跨周期控制
```yaml
# 大级别多头时，小级别能否做空？
allow_reverse_on_smaller_tf: true/false
reverse_size_limit_pct: 0.3  # 限制反向仓位
```

### 4. 状态过滤
```yaml
# 只在特定状态开仓
allowed_states:
  - accumulation  # 积累（支撑做多/阻力做空）
  - expansion     # 扩张（跟随趋势）
  - compression   # 压缩（突破前夕）
```

### 5. 双向Accumulation
- 支撑区 + ACCUMULATION → 做多
- 阻力区 + ACCUMULATION → 做空
- 通过SR位置 + CVD + 订单流综合判断

## 🔧 常用调整

### 提高信号质量
```yaml
min_confidence_to_fire: 0.5  # 从0.3提高到0.5
```

### 禁用做空
```yaml
multi_timeframe_reverse:
  allow_reverse_on_smaller_tf: false
```

### 放宽状态过滤
```yaml
state_filter:
  enabled: false  # 关闭所有过滤
```

### 调整CVD灵敏度
```python
# state_detector.py
cvd_abs < cvd.std() * 0.7  # 原0.5，提高阈值=更宽松
```

## 📈 优化建议

### 高优先级
1. ✅ 禁用做空（使用方案A）
2. ✅ 提高confidence阈值到0.5
3. ⚠️ 调整状态检测阈值（增加信号）

### 中优先级
4. 🔄 增加CVD可视化图表
5. 🔄 多周期CVD协同验证
6. 🔄 动态CVD阈值

### 低优先级
7. 📝 改进阻力位检测算法
8. 📝 增加订单流高级指标
9. 📝 引入机器学习模型

## ⚡ 快速命令

```bash
# 编译检查
python -m compileall nautilus_project/src/yin_bot/dynamic_sr/

# 查看最新报告
ls -lh nautilus_project/reports/dynamic_sr_report.html

# 分析交易结果
python -c "import pandas as pd; df=pd.read_csv('nautilus_project/reports/dynamic_sr_positions_report.csv'); print(df['entry'].value_counts())"

# 查看信号统计
python -c "import pandas as pd; df=pd.read_csv('nautilus_project/reports/dynamic_sr_trade_context.csv'); print(df['state'].value_counts())"
```

## 🐛 问题排查

### 报告生成失败
```bash
# 检查是否有错误
make backtest-dynamic-sr-btc 2>&1 | grep -i error

# 查看完整日志
make backtest-dynamic-sr-btc 2>&1 | tee backtest.log
```

### CVD数据缺失
```python
# 检查tick数据
grep "Processed.*ticks" backtest.log

# 检查buy_volume是否存在
python -c "import pandas as pd; bars=pd.read_csv('...'); print('buy_volume' in bars.columns)"
```

### 没有交易信号
```yaml
# 1. 降低confidence要求
min_confidence_to_fire: 0.1  # 测试用

# 2. 关闭状态过滤
state_filter:
  enabled: false

# 3. 检查数据量
# 需要至少50个bar才能生成信号
```

## 📞 帮助

- 详细配置说明: `CONFIG_GUIDE.md`
- 完整更新历史: `UPDATE_SUMMARY.md`
- CVD技术细节: `docs/CVD_IMPROVEMENT.md`
- 效果对比报告: `CVD_IMPROVEMENT_RESULTS.md`

---

**版本**: v2.1 (CVD Enhanced)
**最后更新**: 2025-10-19
**状态**: ✅ CVD改进完成，盈亏改善64%
**下一步**: 优化做空逻辑或暂时禁用

