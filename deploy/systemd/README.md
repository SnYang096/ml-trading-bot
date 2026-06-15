# systemd services (live)

This folder contains sample unit files used on deploy hosts.

## Spot accumulator service

- Unit: `quant-spot-accum.service`
- Runner: `live/scripts/start_spot_live.sh` (inside `quant-engine:latest` container on deploy hosts)
- Metrics port: `9193` (Prometheus job `quant-spot-accum`)
- **Production:** `.github/workflows/deploy.yml` writes `live/binance_spot_mainnet.env` from
  `BINANCE_SPOT_API_KEY` / `BINANCE_SPOT_API_SECRET` and installs/enables the unit when both are set.

## Multi-leg account Telegram watch

- Units: `mlbot-monitor-account-watch.service` + `.timer` (poll every ~60s)
- Runner: `deploy/systemd/mlbot-account-watch-docker-run.sh`
- **Binance keys:** `--env-file /opt/quant-engine/live/binance_mainnet.env` (`MULTI_LEG_BINANCE_FUTURES_*`)
- **Telegram:** `/opt/quant-engine/monitoring/.env` (`GRAFANA_ALERT_TELEGRAM_BOT_TOKEN`, `GRAFANA_ALERT_TELEGRAM_CHAT_ID`) — same file as Grafana / rebalance TG; **not** in git
- **State:** `/opt/quant-engine/data/monitoring/multi_leg_account_tg_state.json`

```bash
sudo cp etc/systemd/mlbot-monitor-account-watch.{service,timer} /etc/systemd/system/
sudo cp deploy/systemd/mlbot-account-watch-docker-run.sh /opt/quant-engine/deploy/systemd/
sudo chmod +x /opt/quant-engine/deploy/systemd/mlbot-account-watch-docker-run.sh
sudo systemctl daemon-reload
sudo systemctl enable --now mlbot-monitor-account-watch.timer
systemctl list-timers 'mlbot-monitor-account-watch*'
```

Local/manual install (non-CI) still uses the sample host-native unit in this folder; production uses the Docker-based unit embedded in deploy.yml.

Restart after env/config change:

```bash
sudo systemctl restart quant-spot-accum
```
