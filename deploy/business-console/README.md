# MLBot Business Console

Read-only FastAPI console for **Trade Map Live** and lightweight ops views over Feature Bus parquet and SQLite order stores.

## Quick start (local)

```bash
cd /path/to/ml_trading_bot
pip install -r deploy/business-console/requirements-dev.txt
# Optional e2e: playwright install chromium
chmod +x deploy/business-console/run_console.sh
./deploy/business-console/run_console.sh
```

- UI: http://127.0.0.1:8800/trade-map  
- With fake data date window: `?from=2024-01-01T00:00:00Z&to=2024-01-02T00:00:00Z`

## P2 features

- Multi-leg markers (`multi_leg_orders`, `multi_leg_execution_reports`)
- Pending order markers (hollow/circle)
- Marker detail drawer (`GET /api/trade-map/marker-detail`)
- Spot eligibility panel (`GET /api/spot/eligibility`)
- `weekly_ema_200_position` overlay from `features/<tf>` parquet
- Optional volume sub-chart

## P3 deployment

**Docker** (production paths under `/opt/quant-engine`):

```bash
cd /opt/quant-engine/business-console   # CI 解压路径
docker compose up -d --build
```

推送到 `main` 且变更 `deploy/business-console/**` 时，`.github/workflows/deploy.yml` 会自动打包并在服务器执行 `docker compose up -d --build`（监听 `127.0.0.1:8800`）。

**systemd**: `deploy/systemd/quant-business-console.service` (bind `127.0.0.1:8800`).

**Basic Auth** (recommended beyond SSH tunnel):

```bash
export MLBOT_CONSOLE_BASIC_AUTH_USER=admin
export MLBOT_CONSOLE_BASIC_AUTH_PASSWORD=change-me
```

**External links**: `MLBOT_CONSOLE_GRAFANA_URL`, `MLBOT_CONSOLE_ROLLING_BACKTEST_URL` → `GET /api/links`.

## Environment

| Variable | Default |
|----------|---------|
| `MLBOT_CONSOLE_FEATURE_BUS_ROOT` | `live/shared_feature_bus` |
| `MLBOT_CONSOLE_LIVE_DATA_ROOT` | `live/highcap/data` |
| `MLBOT_CONSOLE_ENGINE_DATA_ROOT` | `data` |
| `MLBOT_CONSOLE_MAX_OHLCV_DAYS` | `90` |
| `MLBOT_CONSOLE_MAP_POLL_SECONDS` | `10` |
| `MLBOT_CONSOLE_GRAFANA_URL` | `http://127.0.0.1:3000` |

## Tests

```bash
pip install -r deploy/business-console/requirements-dev.txt
pytest tests/business_console -q
# E2E (needs Node for core JS + Playwright chromium for browser):
pytest tests/business_console/test_web_e2e.py -m integration -v
node -e "$(cat tests/business_console/test_frontend_core.py)"  # see test_frontend_core.py
```

## Docs

- `docs/deployment/BUSINESS_CONSOLE_DESIGN_CN.md`
- `docs/deployment/MONITORING_VS_BUSINESS_CONSOLE_CN.md`
