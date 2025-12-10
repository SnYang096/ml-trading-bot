# 可视化工具 (Visualization)

交易信号和结果可视化

## 脚本

- `trading_map_visualization.py` - 交易地图可视化
- `simple_trading_map.py` - 简单交易地图
- `view_btcusdt_signals.py` / `view_btcusdt_signals_fixed.py` - BTCUSDT信号查看
- `feature_indicator_visualizer.py` - 特征指标可视化（生成HTML报告）

## 使用示例

```bash
python src/time_series_model/visualization/trading_map_visualization.py
python src/time_series_model/visualization/view_btcusdt_signals.py
```

## 使用说明

### feature_indicator_visualizer.py

生成特征指标的HTML可视化报告，显示数据集中可用的特征类型。

**配置文件**: `config/visualization/feature_indicators.yaml`

该脚本从配置文件加载要可视化的特征类型，而不是通过命令行参数指定。

**通过 Makefile 使用**:
```bash
# 基本使用
make feature-indicators SYMBOL=BTCUSDT TIMEFRAME=15T

# 指定日期范围
make feature-indicators SYMBOL=BTCUSDT TIMEFRAME=15T START_DATE=2024-01-01 END_DATE=2024-12-31
```

**注意**: 报告会自动生成带时间戳的文件名，格式为：
`{SYMBOL}_{TIMEFRAME}_{CONFIG_NAME}_{START_DATE}_{END_DATE}_{TIMESTAMP}.html`

例如：`BTCUSDT_15min_feature_indicators_from20240101_to20241231_20251210_143824.html`

**直接使用**:
```bash
python src/time_series_model/visualization/feature_indicator_visualizer.py \
    --data-path data/parquet_data \
    --symbol BTCUSDT \
    --timeframe 15T \
    --config config/visualization/feature_indicators.yaml \
    --output results/feature_indicators/BTCUSDT_15T.html
```

**自定义配置**:
编辑 `config/visualization/feature_indicators.yaml` 来：
- 启用/禁用特定的特征类型
- 修改特征列的匹配模式
- 调整显示设置

## 注意事项

- `trading_map_visualization.py` 和 `view_btcusdt_signals.py` 依赖特定的 CSV 文件，请确保数据文件存在
- 对于详细的特征可视化，建议使用配置驱动的特征导出工具（`make rolling`）

