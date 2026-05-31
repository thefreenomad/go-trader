"""Selection engine (Step 4, piece 1) — the deployable cross-sectional brain.

Reads live HL universe + prices, computes the TARGET PORTFOLIO for this rebalance
encoding the locked, validated config, and emits the go-trader manual-position
commands to realize it. Connector-agnostic; the manual-open/close list is the
bridge to go-trader (which tracks P&L / stops / kill-switch).

Locked config (from research):
  universe = top-N HL-liquid perps     lookback = 4 weeks (momentum)
  legs     = top/bottom 20%            weighting = inverse-vol, per-name cap 25%
  sizing   = vol-target 15% (live: from running track record; cold start lev=1)
  gate     = weak-bear: if BTC composite == trending_down_choppy -> FLAT the book
  cadence  = biweekly
NOTE: signals here use cached OHLCV; a live run refreshes 4h candles to 'now'.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd, ccxt
from data_fetcher import load_cached_data
from regime import compute_regime_composite

TF = "4h"; BARS_DAY = 6; LB = 4 * 7 * BARS_DAY
QFRAC = 0.20; NLIQ = 40; NAME_CAP = 0.25; REGIME_PERIOD = 90
AUM = float(sys.argv[1]) if len(sys.argv) > 1 else 100_000.0
LEV = 1.0   # cold start; live engine sets from trailing realized vol -> 15% target

hl = ccxt.hyperliquid({"enableRateLimit": True}); hl.load_markets(); htk = hl.fetch_tickers()
def hl_vol(c): return (htk.get(f"{c}/USDC:USDC") or {}).get("quoteVolume")

syms = json.load(open("research/universe.json"))
universe = []
for sym in syms:
    c = sym.split("/")[0]
    if hl_vol(c) is None:
        continue
    universe.append(c)
universe = sorted(universe, key=lambda c: -(hl_vol(c) or 0))[:NLIQ]

# prices (cached; live -> refresh to now). Build close matrix.
closes = {}
for c in universe:
    d = load_cached_data(f"{c}/USDT", TF)
    closes[c] = d["close"]
P = pd.DataFrame(closes).sort_index()
asof = P.index[-1]
px = P.iloc[-1]
mom = P.iloc[-1] / P.iloc[-1 - LB] - 1
vol = P.pct_change().iloc[-LB:].std()
elig = [c for c in universe if not (pd.isna(mom[c]) or pd.isna(vol[c]) or vol[c] <= 0)]

# weak-bear gate
btc = load_cached_data("BTC/USDT", TF)
regime = compute_regime_composite(btc, period=REGIME_PERIOD)["regime"].iloc[-1]
gated = regime == "trending_down_choppy"

print(f"=== SELECTION ENGINE  (as of {asof}, AUM ${AUM:,.0f}) ===")
print(f"universe: {len(universe)} HL-liquid coins | eligible: {len(elig)} | "
      f"BTC regime: {regime}{'  -> WEAK BEAR: FLAT THE BOOK' if gated else ''}\n")

target = []   # (coin, side, weight, notional)
if not gated and len(elig) >= 8:
    ranked = sorted(elig, key=lambda c: mom[c])
    k = max(1, int(len(elig) * QFRAC))
    losers, winners = ranked[:k], ranked[-k:]
    for leg, side in [(winners, "long"), (losers, "short")]:
        iv = np.array([1.0 / vol[c] for c in leg]); iv = iv / iv.sum()
        iv = np.minimum(iv, NAME_CAP); iv = iv / iv.sum()
        for c, w in zip(leg, iv):
            notion = AUM * LEV * w
            target.append((c, side, w, notion))

if not target:
    print("TARGET PORTFOLIO: FLAT (no positions)")
else:
    print(f"TARGET PORTFOLIO ({len([t for t in target if t[1]=='long'])} long / "
          f"{len([t for t in target if t[1]=='short'])} short, gross ${sum(t[3] for t in target):,.0f}):")
    print(f"  {'coin':6s} {'side':6s} {'mom4w':>8s} {'weight':>7s} {'notional':>12s}")
    for c, side, w, notion in sorted(target, key=lambda t: (t[1], -t[2])):
        print(f"  {c:6s} {side:6s} {mom[c]*100:>7.1f}% {w*100:>6.1f}% ${notion:>11,.0f}")

print("\n=== go-trader connector commands (manual-position) ===")
print("# 1) close positions no longer selected:  ./go-trader manual-close hl-xs-<coin>")
print("# 2) (re)open the target set:")
for c, side, w, notion in target:
    print(f"  ./go-trader manual-open hl-xs-{c.lower()} --side {side} --notional {notion:.0f}")
print("\n# config: declare one `type:\"manual\"` HL strategy per slot (id hl-xs-<coin>);")
print("# the cron runs this engine biweekly, diffs vs current positions, and drives open/close.")
