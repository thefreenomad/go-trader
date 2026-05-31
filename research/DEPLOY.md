# Paper deployment — operating runbook

The cross-sectional momentum strategy runs as a **paper** instance in Docker on the
Mac Mini. Container = stable running version; working tree = development.

## What's running
- `go-trader-paper` container: the go-trader scheduler (paper), 38 `type:manual` HL
  slots, marking positions to live Hyperliquid prices every `interval_seconds`.
- The **selection engine** (`research/select_engine.py`) pulls 4h candles from HL,
  ranks the crypto universe (4w momentum, inverse-vol, vol-target, per-name cap,
  weak-bear gate), and the **orchestrator** drives the manual paper positions.
- State persists on `./docker-data` (state.db) and `research/results/xs_positions.json`.

## Bring up / down
```bash
docker compose up -d            # build (first time) + start
docker compose restart          # after a config edit (config is mounted)
docker compose down             # stop + remove (state.db persists on volume)
docker compose up -d --build    # rebuild image (after Go/Python code changes)
```

## Monitor (status server is loopback-bound inside the container)
```bash
docker exec go-trader-paper curl -fsS http://localhost:8099/health
docker logs go-trader-paper --tail 20                  # scheduler cycles
docker exec go-trader-paper sh -c '.venv/bin/python - <<PY
import sqlite3; c=sqlite3.connect("data/state.db")
r=c.execute("select owner_strategy_id,side,quantity,avg_cost from positions").fetchall()
print(len(r),"positions"); [print(" ",x[0],x[1],round(x[2],3),"@",round(x[3],4)) for x in r]
PY'
```
For the browser dashboard, add a proxy sidecar (socat 0.0.0.0:8099 -> 127.0.0.1:8099)
or use Tailscale Serve; go-trader stays loopback by design.

## Rebalance (biweekly) — the core loop
```bash
docker exec go-trader-paper .venv/bin/python research/orchestrator.py \
    --aum 100000 --paper --execute
```
Diffs the live target vs current book, closes/opens the delta, persists state.
Schedule it every 14 days (host cron / launchd):
```
# crontab -e   (runs 00:00 on the 1st and 15th)
0 0 1,15 * * cd /Users/tony/Documents/go-trader && /usr/local/bin/docker exec go-trader-paper .venv/bin/python research/orchestrator.py --aum 100000 --paper --execute >> docker-data/rebalance.log 2>&1
```
Dry-run first (omit `--execute`) to preview the plan.

## Reset
```bash
docker compose stop
rm -f docker-data/state.db* research/results/xs_positions.json
docker compose up -d
```

## Going LIVE (later — real capital)
1. Put `HYPERLIQUID_SECRET_KEY` in `.env` (never in git/chat); compose passes it through.
2. Drop `--paper` from the orchestrator (real `manual-open --notional`, on-chain fills).
3. Decide on stops (manual record-only arms none; live arms ATR stops — may conflict
   with biweekly rebalancing of a market-neutral book — review before enabling).
4. Reconcile paper realized vs the `paper_shadow` expectation (~1.1 Sharpe) for weeks first.
