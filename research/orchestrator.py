"""Orchestrator (Step 4, piece 2) — the biweekly cron that turns the selection
engine's target into live go-trader positions.

Flow each rebalance:
  1. compute target portfolio (select_engine)
  2. read current positions (state file = what WE opened in the hl-xs-* slots)
  3. diff -> CLOSE (gone / side flipped / drifted), OPEN (new / flipped), HOLD (within tol)
  4. dry-run print (default) OR --execute (drives ./go-trader manual-open/close)
  5. persist new state
Weak-bear gate -> target is FLAT -> close everything.

Also: --emit-config writes the type:"manual" slot fragment for scheduler/config.json.

Safe by default: prints the plan; only touches go-trader with --execute.
"""
import sys, os, json, argparse, subprocess, time, fcntl
sys.path.insert(0, "research")
from select_engine import compute_target

STATE = "research/results/xs_positions.json"     # what we currently hold (per-coin side+notional)
RESIZE_TOL = 0.25                                  # re-open only if notional drifts > 25%
ATTEMPTS = "research/results/xs_attempts.json"   # per-coin last on-chain action ts (idempotency)
SETTLE_GRACE_S = 600                              # suppress re-acting on a coin for 10 min (settlement lag)
LOCK = "/tmp/xs_orchestrator.lock"               # process lock: never two rebalances at once


def load_state(path):
    if os.path.exists(path):
        return json.load(open(path))
    return {}


def load_attempts(path):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            return {}
    return {}


def save_attempts(path, d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(d, open(path, "w"), indent=2)


def filter_settling(closes, opens, attempts, now, grace):
    """Idempotency guard: drop every action on a coin acted on within ``grace``.

    HL's clearinghouseState lags freshly-placed fills, so a rebalance that
    re-runs inside the settlement window reads stale account state and would
    re-open coins that already filled — doubling the position. We suppress all
    actions (close+open, so a resize isn't half-applied) on any coin touched in
    the last ``grace`` seconds; genuinely-unfilled coins retry once the window
    passes. Returns (closes, opens, suppressed_coins)."""
    suppress = {c for c, ts in attempts.items() if (now - ts) < grace}
    if not suppress:
        return closes, opens, []
    acted = {c[0] for c in closes} | {o[0] for o in opens}
    fc = [c for c in closes if c[0] not in suppress]
    fo = [o for o in opens if o[0] not in suppress]
    return fc, fo, sorted(suppress & acted)


def acquire_lock(path):
    """Non-blocking exclusive lock so two rebalances never run concurrently (a
    manual run colliding with the cron, or a double-fire). Returns the held file
    handle (keep it alive for the process lifetime) or None if already locked."""
    f = open(path, "w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def live_positions():
    """Current book from the ACTUAL HL account (live/testnet ground truth), so the
    diff reconciles against reality instead of assuming every open filled."""
    sys.path.insert(0, "platforms/hyperliquid")
    from adapter import HyperliquidExchangeAdapter
    a = HyperliquidExchangeAdapter()
    st = a._info.user_state(os.environ["HYPERLIQUID_ACCOUNT_ADDRESS"])
    cur = {}
    for p in st.get("assetPositions", []):
        pp = p["position"]; szi = float(pp.get("szi", 0) or 0)
        if szi == 0:
            continue
        cur[pp["coin"]] = {"side": "long" if szi > 0 else "short",
                           "notional": abs(szi) * float(pp.get("entryPx") or 0)}
    return cur


def diff(current, target, gated):
    tgt = {t["coin"]: t for t in target}
    closes, opens, holds = [], [], []
    for coin, cur in current.items():
        t = tgt.get(coin)
        if gated or t is None or t["side"] != cur["side"]:
            closes.append((coin, cur["side"], "gate-flat" if gated else
                           ("dropped" if t is None else "side-flip")))
    for coin, t in ({} if gated else tgt).items():
        cur = current.get(coin)
        if cur is None or cur["side"] != t["side"]:
            opens.append((coin, t["side"], t["notional"], t.get("price", 0), "new" if cur is None else "flip"))
        else:
            drift = abs(t["notional"] - cur["notional"]) / max(cur["notional"], 1)
            if drift > RESIZE_TOL:
                closes.append((coin, cur["side"], f"resize {drift*100:.0f}%"))
                opens.append((coin, t["side"], t["notional"], t.get("price", 0), "resize"))
            else:
                holds.append((coin, t["side"], t["notional"]))
    return closes, opens, holds


def run_cmd(args, execute, gt):
    cmd = [gt] + args
    print("   $ " + " ".join(cmd))
    if execute:
        r = subprocess.run(cmd, capture_output=True, text=True)
        print("     " + (r.stdout or r.stderr).strip()[:200])
        return r.returncode == 0
    return True


def emit_config(target_universe_hint=40):
    """Emit the type:manual slot fragment for the universe (one-time config setup)."""
    r = compute_target(100_000)
    coins = sorted(r["universe_coins"]) or ["BTC", "ETH"]   # full universe (any coin we may trade)
    slots = [{
        "id": f"hl-xs-{c.lower()}", "type": "manual", "platform": "hyperliquid",
        "script": "shared_scripts/check_hyperliquid.py", "args": ["hold", c, "4h"],
        "capital": 0, "leverage": 1, "margin_mode": "cross",
        "allowed_regimes": [],            # LOOSE: gating is portfolio-level in our engine
    } for c in coins]
    out = "research/results/xs_manual_slots.json"
    json.dump({"strategies": slots}, open(out, "w"), indent=2)
    print(f"wrote {len(slots)} manual slots -> {out}")
    print("  Declare one per coin you may trade (the full universe). Merge into scheduler/config.json once.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aum", type=float, default=100_000)
    ap.add_argument("--lev", type=float, default=1.0)
    ap.add_argument("--execute", action="store_true", help="actually drive go-trader (default: dry-run)")
    ap.add_argument("--paper", action="store_true", help="paper: record-only opens at current price")
    ap.add_argument("--gotrader", default="./go-trader")
    ap.add_argument("--emit-config", action="store_true")
    a = ap.parse_args()
    if a.emit_config:
        emit_config(); return

    lock = None
    if a.execute:                       # hold an exclusive lock for the whole live run
        lock = acquire_lock(LOCK)
        if lock is None:
            print("another rebalance is in progress (lock held) — exiting"); return

    r = compute_target(a.aum, a.lev)
    if a.paper:
        current = load_state(STATE)
    else:                       # live/testnet: reconcile against the real account
        try:
            current = live_positions()
        except Exception as e:
            print(f"warn: live_positions failed ({e}); falling back to state file")
            current = load_state(STATE)
    closes, opens, holds = diff(current, r["target"], r["gated"])

    # Idempotency: suppress actions on coins touched within the settlement window
    # so a too-soon re-run can't double a position the account hasn't surfaced yet.
    suppressed = []
    if a.execute and not a.paper:
        closes, opens, suppressed = filter_settling(
            closes, opens, load_attempts(ATTEMPTS), time.time(), SETTLE_GRACE_S)

    print(f"=== ORCHESTRATOR  (as of {r['asof']}, AUM ${a.aum:,.0f}, "
          f"{'EXECUTE' if a.execute else 'DRY-RUN'}) ===")
    print(f"BTC regime {r['regime']}"
          f"{'  -> WEAK BEAR: FLATTEN BOOK' if r['gated'] else ''}")
    print(f"current {len(current)} positions | target {len(r['target'])} | "
          f"close {len(closes)} / open {len(opens)} / hold {len(holds)}\n")
    if suppressed:
        print(f"settling (skipped, acted <{SETTLE_GRACE_S // 60}m ago): "
              f"{', '.join(suppressed)}\n")

    if not closes and not opens:
        print("No changes needed (book already matches target).")
    print("CLOSES:")
    for coin, side, why in closes:
        print(f"  hl-xs-{coin.lower():6s} ({side:5s})  [{why}]")
        run_cmd(["manual-close", f"hl-xs-{coin.lower()}"], a.execute, a.gotrader)
    print("OPENS:")
    for coin, side, notion, price, why in opens:
        sid = f"hl-xs-{coin.lower()}"
        size = notion / price if price else 0.0
        if a.paper:
            cmd = ["manual-open", sid, "--side", side, "--size", f"{size:.6f}",
                   "--record-only", "--fill-price", f"{price}"]
        else:   # live/testnet: size from engine price (Go's --notional mark fetch is unreliable)
            cmd = ["manual-open", sid, "--side", side, "--size", f"{size:.6f}"]
        print(f"  {sid:14s} ({side:5s}) ${notion:>10,.0f}  [{why}]")
        run_cmd(cmd, a.execute, a.gotrader)
    if holds:
        print(f"HOLDS (within {RESIZE_TOL*100:.0f}% tol): " + ", ".join(f"{c}/{s}" for c, s, _ in holds))

    if a.execute:
        new_state = {t["coin"]: {"side": t["side"], "notional": t["notional"]} for t in r["target"]}
        os.makedirs(os.path.dirname(STATE), exist_ok=True)
        json.dump(new_state, open(STATE, "w"), indent=2)
        print(f"\nstate persisted -> {STATE}")
        # stamp coins we just acted on so the next run inside the window skips them
        now = time.time()
        attempts = {c: ts for c, ts in load_attempts(ATTEMPTS).items() if (now - ts) < SETTLE_GRACE_S}
        for coin in ({c[0] for c in closes} | {o[0] for o in opens}):
            attempts[coin] = now
        save_attempts(ATTEMPTS, attempts)
        # log a rebalance event (dashboard marks these on the equity curve)
        import csv as _csv
        from datetime import datetime as _dt, timezone as _tz
        EV = "research/results/rebalance_events.csv"
        _new = not os.path.exists(EV)
        with open(EV, "a") as _f:
            if _new:
                _f.write("time,n_positions,gross,note\n")
            _g = sum(t["notional"] for t in r["target"])
            _f.write(f"{_dt.now(_tz.utc).isoformat()},{len(r['target'])},{_g:.0f},"
                     f"opens={len(opens)} closes={len(closes)} holds={len(holds)}\n")
    else:
        print("\n(dry-run: no go-trader commands executed, state unchanged. Add --execute to apply.)")


if __name__ == "__main__":
    main()
