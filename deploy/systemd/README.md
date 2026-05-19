# systemd services (live)

This folder contains sample unit files used on deploy hosts.

## Spot accumulator service

- Unit: `quant-spot-accum.service`
- Runner: `live/scripts/start_spot_live.sh` (inside `quant-engine:latest` container on deploy hosts)
- Metrics port: `9193` (Prometheus job `quant-spot-accum`)
- **Production:** `.github/workflows/deploy.yml` writes `live/binance_spot_mainnet.env` from
  `BINANCE_SPOT_API_KEY` / `BINANCE_SPOT_API_SECRET` and installs/enables the unit when both are set.

Local/manual install (non-CI) still uses the sample host-native unit in this folder; production uses the Docker-based unit embedded in deploy.yml.

Restart after env/config change:

```bash
sudo systemctl restart quant-spot-accum
```
