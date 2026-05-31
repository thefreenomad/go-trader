#!/usr/bin/env bash
# Biweekly rebalance of the cross-sectional momentum book (paper by default).
# Called by cron on the 1st & 15th. Switch to testnet/live by setting XS_LIVE=1
# (after putting HYPERLIQUID_TESTNET=1 + key in .env and recreating the container).
#
#   env knobs:  XS_AUM (default 100000)  XS_LIVE (0=paper record-only, 1=live)
#               XS_DRYRUN (1 = preview only, no execute)
set -euo pipefail

REPO="/Users/tony/Documents/go-trader"
DOCKER="${DOCKER_BIN:-/usr/local/bin/docker}"
CONTAINER="go-trader-paper"
AUM="${XS_AUM:-10000}"     # $10k AUM -> ~$20k gross (2x, market-neutral); grow over time

cd "$REPO"
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# build flags
flags=(--aum "$AUM")
[ "${XS_LIVE:-0}" = "1" ] || flags+=(--paper)        # paper unless XS_LIVE=1
[ "${XS_DRYRUN:-0}" = "1" ] || flags+=(--execute)    # execute unless XS_DRYRUN=1

# guard: container must be running
if ! "$DOCKER" ps --filter "name=$CONTAINER" --filter "status=running" --format '{{.Names}}' | grep -q "$CONTAINER"; then
  echo "[$(ts)] ERROR: container $CONTAINER not running — skipping rebalance" >&2
  exit 1
fi

echo "[$(ts)] rebalance start — flags: ${flags[*]}"
"$DOCKER" exec "$CONTAINER" .venv/bin/python research/orchestrator.py "${flags[@]}"
echo "[$(ts)] rebalance done (exit 0)"
