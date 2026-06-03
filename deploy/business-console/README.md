# MLBot Business Console (deploy shell)

Read-only FastAPI console for **Trade Map Live** and lightweight ops views.

**Application code** lives under `src/mlbot_console/` (routers, services, static UI).  
This directory holds **Docker / compose / run script** only.

## Quick start (local)

```bash
cd /path/to/ml_trading_bot
pip install -r deploy/business-console/requirements-dev.txt
chmod +x deploy/business-console/run_console.sh
./deploy/business-console/run_console.sh
```

- UI: http://127.0.0.1:8800/trade-map

## Production (Docker)

```bash
cd /opt/quant-engine/deploy/business-console
docker compose up -d --build
```

Build context is the **repository root** (`../..` → `/opt/quant-engine`): image includes `src/mlbot_console`, `src/time_series_model`, `src/config` (archetype prefilter/gate regions), and `config/strategies`. Account overview uses `requests`, `ccxt`, and `python-dotenv` via `mlbot_console.services.spot_ccxt` (no `order_management` copy).

CI packs `deploy/business-console`, `src/{mlbot_console,live_data_stream,time_series_model,monitoring,config}`, `config/{strategies,monitoring}`, `live/highcap/universe.yaml`, and **`live/highcap/config/`** (constitution + live strategy YAML) under `/opt/quant-engine/`. Compose build context is that tree; image also embeds research `config/strategies` + `config/monitoring` for defaults. Runtime strategies come from the volume `/data/live_root/config/strategies`.

**Pre-deploy smoke** (no CI wait):

```bash
./scripts/smoke_console_live_strategies.sh
./scripts/smoke_console_live_strategies.sh --remote ubuntu@YOUR_HOST -i ~/.ssh/key.pem
```

**systemd**: `deploy/systemd/quant-business-console.service` — `PYTHONPATH=/opt/quant-engine/src`, `uvicorn mlbot_console.main:app`.

## Tests

```bash
pytest tests/business_console -q
```

## Docs

- `docs/deployment/BUSINESS_CONSOLE_DESIGN_CN.md`
