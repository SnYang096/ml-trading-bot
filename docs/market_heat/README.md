# Market Heat（市场热度）说明

本模块实现「市场 → 板块 → 标的」三层热度，用于辅助判断数字货币（Binance 现货周线）环境下哪些板块更值得关注，并可与 Prometheus / Grafana 集成；策略侧可通过 Python API 做前置过滤。

## 热度定义（周线 EMA50）

- **HOT**：收盘价在周线 EMA50 之上，且 EMA50 近 4 周斜率为正（价在线上、均线抬头）。
- **WARM**：价在 EMA50 之上，但斜率非正（动能减弱）。
- **COLD**：价在 EMA50 之下。

连续分数 `score` 在 0～1 之间，板块与市场分数由成分标的聚合得到；市场整体默认用 **BTC 60% + ETH 40%** 加权（见 `config/market_heat/crypto_sectors.yaml` 中 `market_proxy`）。

## 目录与文件

| 路径 | 说明 |
|------|------|
| `config/market_heat/crypto_sectors.yaml` | 板块与标的列表、市场代理权重 |
| `src/market_heat/` | 拉数、算分、聚合、Prometheus 指标、策略 prefilter、CoinGecko 同步脚本 |
| `scripts/run_heat_update.py` | 定时任务入口：拉数、写指标；可 `--loop` 常驻 |
| `scripts/generate_heat_dashboard.py` | 生成 Grafana 面板 JSON |
| `deploy/monitoring/grafana-provisioning/dashboards/market_heat.json` | 由脚本生成的面板（勿手改，改脚本后重跑） |
| `data/market_heat/weekly_ohlcv.parquet` | 运行后自动生成的周线缓存（可删以强制全量重拉） |

## 命令行快速查看

在项目根目录：

```bash
# 终端表格（默认）
python -m src.market_heat.run

# JSON
python -m src.market_heat.run --format json

# 只看部分板块
python -m src.market_heat.run --sector L1 --sector Meme

# 忽略本地 parquet 缓存时间，尽量重新拉取（仍可能命中刚写入的缓存）
python -m src.market_heat.run --no-cache
```

## 定时更新与 Prometheus

`scripts/run_heat_update.py` 会：

1. 拉取配置中全部标的的周线 OHLCV（ccxt Binance，与仓库内其它 Binance 用法一致，可走 `USE_SOCKS5_PROXY` 等环境变量）。
2. 计算热度并调用 `export_heat_to_prometheus`，写入 `mlbot_heat_*` 系列 Gauge。
3. 在本进程内 `start_http_server` 暴露 `/metrics`（默认端口 **9091**，避免与实盘 `MLBOT_METRICS_PORT` 默认 9090 冲突）。

示例：

```bash
# 单次更新并打印表格
python scripts/run_heat_update.py --print

# 每 3600 秒循环，指标端口 9091
python scripts/run_heat_update.py --loop 3600 --metrics-port 9091
```

生产环境请用 **systemd timer** 或 **cron** 调用单次模式；若与现有 Prometheus 同机，在 `prometheus.yml` 中增加一个 job，抓取 `host:9091/metrics`（或与 Docker 内 `extra_hosts` 等网络方案对齐）。

指标名称（节选）：

- `mlbot_heat_score{symbol, sector}`
- `mlbot_heat_state{symbol, sector}`（0=COLD，1=WARM，2=HOT）
- `mlbot_heat_ema_slope` / `mlbot_heat_ema_distance`
- `mlbot_heat_sector_score{sector}`
- `mlbot_heat_market_score{market="crypto"}`

## Grafana 面板

修改板块或面板布局后：

```bash
python scripts/generate_heat_dashboard.py
```

将更新 `deploy/monitoring/grafana-provisioning/dashboards/market_heat.json`。Grafana 通过既有 provisioning 加载该目录即可。

## 板块数据与同步

默认板块与代币列表在 `crypto_sectors.yaml` 中手工维护。可选从 CoinGecko 拉分类做「建议增量」（不直接覆盖）：

```bash
python -m src.market_heat.sync_sectors
# 确认后再写入 YAML
python -m src.market_heat.sync_sectors --update
```

## 策略侧 Prefilter（Python）

与 `archetypes/prefilter.yaml` 的「单根 bar 特征」不同，热度是**周线层面**的外部信号，通过独立模块注入：

```python
from src.market_heat.prefilter import HeatPrefilter, HeatFilterConfig

heat_pf = HeatPrefilter(
    HeatFilterConfig(
        enabled=True,
        min_symbol_heat=0.3,
        min_sector_heat=0.2,
        min_market_heat=0.2,
        refresh_interval_hours=6,
    )
)

if not heat_pf.is_tradeable("BTCUSDT"):
    # 跳过该标的的下单/信号管线
    ...
```

无数据或刷新失败时 **`is_tradeable` 默认可通过（fail-open）**，避免误杀实盘；若需严格模式，可在业务层再加开关。

## 依赖

- `pandas`、`pyyaml`；拉数需要 `ccxt`；缓存 parquet 需要 `pyarrow` 或 `fastparquet`（与项目其余 parquet 一致）；Prometheus 需要 `prometheus_client`。

## 后续扩展（未实现）

多市场（A 股、美股、黄金、外汇等）可在同一套「注册表 + 拉数器 + 聚合」抽象上扩展；当前版本聚焦 **Binance USDT 现货周线** 与 `crypto_sectors.yaml` 中的板块划分。
