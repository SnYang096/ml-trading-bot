# systemd services (live)

This folder contains sample unit files used on deploy hosts.

## Spot accumulator service

- Unit: `quant-spot-accum.service`
- Runner: `live/scripts/start_spot_live.sh`
- Metrics port: `9193` (Prometheus job `quant-spot-accum`)

Install example:

```bash
sudo cp deploy/systemd/quant-spot-accum.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now quant-spot-accum
sudo systemctl status quant-spot-accum --no-pager
```

Restart after env/config change:

```bash
sudo systemctl restart quant-spot-accum
```
