# 🔍 三层架构快速可视化检查指南

## 功能说明

这是一个**不需要完整回测**的快速可视化工具，帮助你立即看到三层架构的决策过程。

### 输出内容

1. **第一层（战略层）**: 市场状态区域标记
   - COMPRESSION (灰色) - 压缩
   - ACCUMULATION (浅蓝) - 蓄势
   - EXPANSION (浅绿) - 扩张
   - EXHAUSTION (橙色) - 衰竭
   - VACUUM (红色) - 真空

2. **第二层（战术层）**: SR级别线条
   - 红色虚线 - 阻力位
   - 绿色虚线 - 支撑位
   - 线条粗细 - 强度

3. **第三层（执行层）**: 入场信号
   - 金色圆点 - 满足三层条件的信号
   - 悬停显示详细信息

4. **决策列表**: 表格显示每个信号的详细原因

---

## 使用方法

### 快速检查（半天数据）
```bash
make quick-visual
```

**输出**: `nautilus_project/quick_check_report.html`

### 自定义数据文件
```bash
cd nautilus_project
.venv/bin/python -m yin_bot.dynamic_sr.quick_visual_check \
    --data "data/agg_data/BTCUSDT-aggTrades-2025-05-02.csv" \
    --config "src/yin_bot/dynamic_sr/config.yaml" \
    --output "check_05_02.html"
```

---

## 首次运行结果分析

### 数据统计
```
时间范围: 2025-05-01 00:00 ~ 13:42 (半天)
4H bars: 4个 ⚠️ 不足
30m bars: 28个
5m bars: 165个
```

### 第一层：战略层
```
⚠️ 分析了 0 个4H bars
原因: 4H数据不足（只有4个bar，需要至少10个）
```

### 第二层：战术层
```
✅ 检测到 3 个SR级别:
   local_high @ 96281.90 (强度: 1.00)
   local_high @ 96321.10 (强度: 1.00)
   local_high @ 96586.40 (强度: 1.00)
```

### 第三层：执行层
```
⚠️ 没有产生入场信号
原因: 战略层失败（4H数据不足） → 三层AND逻辑失败
```

---

## 问题诊断

### 为什么没有信号？

**原因1**: 数据不足
- 半天数据只有4个4H bars
- 战略层需要至少10个bars进行状态检测
- → 战略层始终失败

**原因2**: 三层AND逻辑
```python
if strategic_conf < 0.4:
    continue  # ❌ 战略层失败，直接跳过
```

即使战术层和执行层满足条件，战略层失败就不会产生信号。

---

## 如何看到效果？

### 方案1: 使用更多数据（推荐）
```bash
# 使用完整一天数据
cd nautilus_project
.venv/bin/python -m yin_bot.dynamic_sr.quick_visual_check \
    --data "data/agg_data/BTCUSDT-aggTrades-2025-05-01.csv" \
    --config "src/yin_bot/dynamic_sr/config.yaml" \
    --output "full_day_check.html"
```

**说明**: 完整一天 = 6个4H bars，仍然不足...

### 方案2: 修改脚本，不截取数据
编辑 `quick_visual_check.py` L58:
```python
# 注释掉这行
# df_ticks = df_ticks.iloc[:len(df_ticks)//2]
```

然后运行：
```bash
make quick-visual
```

### 方案3: 使用两天数据
```bash
cd nautilus_project
cat data/agg_data/BTCUSDT-aggTrades-2025-05-01.csv \
    data/agg_data/BTCUSDT-aggTrades-2025-05-02.csv \
    > data/agg_data/BTCUSDT-aggTrades-2days.csv

.venv/bin/python -m yin_bot.dynamic_sr.quick_visual_check \
    --data "data/agg_data/BTCUSDT-aggTrades-2days.csv" \
    --output "2days_check.html"
```

### 方案4: 临时降低置信度阈值
编辑 `config.yaml`:
```yaml
three_tier:
  layer_roles:
    strategic:
      min_confidence: 0.15  # 从0.4降低，快速测试
    tactical:
      min_confidence: 0.10
    execution:
      min_confidence: 0.10
```

---

## 报告解读

### 图表部分

#### 主图（K线+市场状态+SR+信号）
- **背景色块**: 市场状态（4H级别）
- **水平虚线**: SR级别（30m级别）
- **金色圆点**: 入场信号（5m级别）

#### 悬停提示
鼠标悬停在信号点上显示：
- 时间、价格
- 战略层方向和置信度
- 战术层置信度
- 执行层置信度
- 最终决策原因

### 表格部分

列名说明：
- **时间**: 信号时间
- **价格**: 当前价格
- **战略方向**: long/short/neutral
- **战略置信度**: 0.0-1.0
- **战术置信度**: 0.0-1.0
- **执行置信度**: 0.0-1.0
- **最终置信度**: 融合后的置信度
- **决策原因**: 文字说明

---

## 常见问题

### Q1: 为什么没有市场状态区域？
**A**: 4H数据不足（<10个bars）。解决：使用更多天的数据。

### Q2: 为什么没有入场信号？
**A**: 可能原因：
1. 4H数据不足 → 战略层失败
2. 置信度阈值太高
3. SR级别不在当前价格附近
4. 确实没有满足条件的机会

### Q3: 如何快速看到效果？
**A**: 
1. 使用Nautilus已测试的数据（5月1日有26笔交易）
2. 临时降低置信度阈值
3. 使用完整日期数据而非半天

### Q4: 报告在哪里？
**A**: `nautilus_project/quick_check_report.html`

用浏览器打开即可。

---

## 对比：快速检查 vs 完整回测

| 特性 | 快速检查 | Nautilus回测 |
|------|---------|-------------|
| 耗时 | 10-30秒 | 15-25分钟 |
| 数据量 | 可调整（半天~全月） | 全月 |
| 输出 | 决策可视化 | 完整报告+交易统计 |
| 目的 | 检查三层逻辑 | 策略回测验证 |
| 仓位管理 | ❌ 无 | ✅ 完整 |
| 推荐用途 | 开发调试 | 最终验证 |

---

## 下一步

### 看到效果后
1. 分析哪些市场状态产生信号最多
2. 检查SR级别是否准确
3. 验证三层决策原因是否合理

### 调整参数
```bash
# 修改config.yaml
vim nautilus_project/src/yin_bot/dynamic_sr/config.yaml

# 快速测试
make quick-visual

# 满意后完整回测
make backtest-dynamic-sr-month
```

### 批量测试多天
```bash
# 测试5月1-7日
for day in 01 02 03 04 05 06 07; do
    cd nautilus_project
    .venv/bin/python -m yin_bot.dynamic_sr.quick_visual_check \
        --data "data/agg_data/BTCUSDT-aggTrades-2025-05-${day}.csv" \
        --output "check_05_${day}.html"
done
```

---

**总结**: 这是一个快速验证三层架构效果的工具。首次运行因数据不足未产生信号，使用更多数据或降低阈值即可看到效果。🚀

