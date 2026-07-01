#!/usr/bin/env bash
# Idempotent host bootstrap for business-console / monitoring compose.
set -euo pipefail

if docker compose version >/dev/null 2>&1; then
  exit 0
fi

echo "Installing docker compose plugin..."
export DEBIAN_FRONTEND=noninteractive
run_apt() {
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    "$@"
  fi
}
run_apt apt-get update -qq
for pkg in docker-compose-v2 docker-compose-plugin; do
  if apt-cache show "$pkg" >/dev/null 2>&1; then
    run_apt apt-get install -y "$pkg"
    break
  fi
done

if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: docker compose still unavailable after apt install" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: cannot talk to docker daemon (add deploy user to docker group?)" >&2
  exit 1
fi
