# 回测脚本 (Backtesting)

回测和样本外测试工具

## 分类

### OOS (Out-of-Sample) 测试
- `oos_batch_test.py` - 批量OOS测试
- `oos_february.py` / `oos_february_simple.py` - 二月测试
- `oos_june.py` - 六月测试  
- `oos_months.py` - 多月测试
- `oos_test.py` - 一般OOS测试
- `test_2025_oos.py` - 2025 OOS测试

### 一般回测
- `backtest_btcusdt.py` / `backtest_btcusdt_fixed.py` - BTCUSDT回测
- `quick_test_quarterly_models.py` - 快速测试季度模型

### VectorBT 回测
- `vectorbot_backtest.py` - 基础版本
- `vectorbot_backtest_improved.py` - 改进版本
- `vectorbot_backtest_improved_features.py` - 改进特征
- `vectorbot_backtest_wavelet.py` - 小波特征
- `vectorbot_backtest_15min.py` - 15分钟时间框架

## 使用示例

```bash
# OOS测试
python scripts/backtesting/oos_batch_test.py

# VectorBT回测
python scripts/backtesting/vectorbot_backtest_improved.py
```

