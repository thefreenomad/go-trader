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
import sys, os, json, argparse, subprocess
sys.path.insert(0, "research")
from select_engine import compute_target

STATE = "research/results/xs_positions.json"     # what we currently hold (per-coin side+notional)
RESIZE_TOL = 0.25                                  # re-open only if notional drifts > 25%


def load_state(path):
    if os.path.exists(path):
        return json.load(open(path))
    return {}


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
            opens.append((coin, t["side"], t["notional"], "new" if cur is None else "flip"))
        else:
            drift = abs(t["notional"] - cur["notional"]) / max(cur["notional"], 1)
            if drift > RESIZE_TOL:
                closes.append((coin, cur["side"], f"resize {drift*100:.0f}%"))
                opens.append((coin, t["side"], t["notional"], "resize"))
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
    ap.add_argument("--gotrader", default="./go-trader")
    ap.add_argument("--emit-config", action="store_true")
    a = ap.parse_args()
    if a.emit_config:
        emit_config(); return

    r = compute_target(a.aum, a.lev)
    current = load_state(STATE)
    closes, opens, holds = diff(current, r["target"], r["gated"])

    print(f"=== ORCHESTRATOR  (as of {r['asof']}, AUM ${a.aum:,.0f}, "
          f"{'EXECUTE' if a.execute else 'DRY-RUN'}) ===")
    print(f"BTC regime {r['regime']}"
          f"{'  -> WEAK BEAR: FLATTEN BOOK' if r['gated'] else ''}")
    print(f"current {len(current)} positions | target {len(r['target'])} | "
          f"close {len(closes)} / open {len(opens)} / hold {len(holds)}\n")

    if not closes and not opens:
        print("No changes needed (book already matches target).")
    print("CLOSES:")
    for coin, side, why in closes:
        print(f"  hl-xs-{coin.lower():6s} ({side:5s})  [{why}]")
        run_cmd(["manual-close", f"hl-xs-{coin.lower()}"], a.execute, a.gotrader)
    print("OPENS:")
    for coin, side, notion, why in opens:
        print(f"  hl-xs-{coin.lower():6s} ({side:5s}) ${notion:>10,.0f}  [{why}]")
        run_cmd(["manual-open", f"hl-xs-{coin.lower()}", "--side", side, "--notional", f"{notion:.0f}"],
                a.execute, a.gotrader)
    if holds:
        print(f"HOLDS (within {RESIZE_TOL*100:.0f}% tol): " + ", ".join(f"{c}/{s}" for c, s, _ in holds))

    if a.execute:
        new_state = {t["coin"]: {"side": t["side"], "notional": t["notional"]} for t in r["target"]}
        os.makedirs(os.path.dirname(STATE), exist_ok=True)
        json.dump(new_state, open(STATE, "w"), indent=2)
        print(f"\nstate persisted -> {STATE}")
    else:
        print("\n(dry-run: no go-trader commands executed, state unchanged. Add --execute to apply.)")


if __name__ == "__main__":
    main()
