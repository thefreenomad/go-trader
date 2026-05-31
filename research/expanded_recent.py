"""Pre-deploy SMOKE TEST: deployable engine on a broader, NATIVE Hyperliquid
universe over the PAST MONTH, marked per-4h-bar (not just at rebalances).

NOTE: 1 month = ~2 biweekly selections -> NOT statistical validation (wide error
bars). It's a sanity check: sensible positions on the real HL universe + sane
recent P&L. Real validation was the multi-year work.
"""
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd, ccxt
from regime import compute_regime_composite

TF = "4h"; BARS_DAY = 6; LB = 4 * 7 * BARS_DAY   # 4w momentum
STEP = 2 * 7 * BARS_DAY                            # biweekly reselect
QFRAC = 0.20; NAME_CAP = 0.25; NUNIV = 60; MIN_VOL = 3e6
TARGET_VOL = 0.15; WEEKS_YR = 365 / 7
NBARS = 420                                       # ~70 days of 4h (1mo test + warmup)

hl = ccxt.hyperliquid({"enableRateLimit": True}); hl.load_markets()
tk = hl.fetch_tickers()
try: hfr = hl.fetch_funding_rates()
except Exception: hfr = {}
# broader native HL universe: top liquid perps by 24h notional
cand = []
for sym, t in tk.items():
    if not sym.endswith("/USDC:USDC"): continue
    base = sym.split("/")[0]
    qv = t.get("quoteVolume") or 0
    if qv >= MIN_VOL and not base.startswith(("k", "@")):
        cand.append((base, sym, qv))
cand.sort(key=lambda x: -x[2]); cand = cand[:NUNIV]
print(f"Native HL universe: {len(cand)} perps with >= ${MIN_VOL/1e6:.0f}M/day "
      f"(${cand[-1][2]/1e6:.0f}M .. ${cand[0][2]/1e6:.0f}M)\n")

closes = {}
for base, sym, qv in cand:
    try:
        o = hl.fetch_ohlcv(sym, TF, limit=NBARS)
        if len(o) >= LB + 30:
            closes[base] = pd.Series([c[4] for c in o], index=pd.to_datetime([c[0] for c in o], unit="ms"))
    except Exception:
        continue
    time.sleep(0.05)
P = pd.DataFrame(closes).sort_index().ffill()
coins = list(P.columns); Pv = P.values; Rv = P.pct_change().values; N = len(P)
F = np.array([float(np.clip((hfr.get(f"{c}/USDC:USDC") or {}).get("fundingRate", 0) * 24 * 365, -0.5, 1.0)) for c in coins])
print(f"Fetched {len(coins)} coins x {N} 4h bars ({P.index[0].date()} -> {P.index[-1].date()})")

# weak-bear gate from HL BTC composite
btc_ohlc = hl.fetch_ohlcv("BTC/USDC:USDC", TF, limit=NBARS)
btc_df = pd.DataFrame(btc_ohlc, columns=["t", "open", "high", "low", "close", "v"])
btc_lab = compute_regime_composite(btc_df, period=90)["regime"].values

def cap(iv):
    w = iv / iv.sum()
    for _ in range(20):
        over = w > NAME_CAP + 1e-9
        if not over.any(): break
        exc = (w[over] - NAME_CAP).sum(); w[over] = NAME_CAP; un = ~over
        if not un.any(): break
        w[un] += exc * (w[un] / w[un].sum())
    return w

# per-4h-bar marking; reselect every STEP bars
w = np.zeros(len(coins)); eq = [1.0]; gate_bars = 0; last_book = None
for b in range(LB, N - 1):
    if (b - LB) % STEP == 0:                      # reselect
        w = np.zeros(len(coins))
        if btc_lab[b] == "trending_down_choppy":
            gate_bars += 1
        else:
            mom = Pv[b] / Pv[b - LB] - 1
            v = np.nanstd(Rv[b - LB:b, :], axis=0)
            elig = np.where(~np.isnan(mom) & ~np.isnan(v) & (v > 0))[0]
            if elig.size >= 8:
                order = elig[np.argsort(mom[elig])]; k = max(1, int(elig.size * QFRAC))
                for ix, sg in [(order[-k:], 1), (order[:k], -1)]:
                    w[ix] = sg * cap(1.0 / v[ix])
                last_book = (b, order[-k:], order[:k], w.copy())
    bar_ret = float(np.nansum(w * np.nan_to_num(Rv[b + 1])))   # mark to next bar
    eq.append(eq[-1] * (1 + bar_ret))
E = pd.Series(eq, index=P.index[LB:N])

# last 30 days
recent = E[E.index >= E.index[-1] - pd.Timedelta(days=30)]
r = recent.pct_change().dropna()
shp = (r.mean() / r.std()) * np.sqrt(365 * BARS_DAY) if r.std() > 0 else 0
print(f"\n=== PAST-MONTH SMOKE TEST (per-4h-bar marked, ~{len(r)} obs) ===")
print(f"  return: {(recent.iloc[-1]/recent.iloc[0]-1)*100:+.1f}%   ann.Sharpe: {shp:.2f}   "
      f"maxDD: {(recent/recent.cummax()-1).min()*100:.1f}%")
print(f"  weak-bear gate fired on {gate_bars} reselect(s)")

if last_book:
    b, win, los, wv = last_book
    print(f"\n=== CURRENT TARGET BOOK (as of {P.index[b].date()}) ===")
    print(f"  LONG : " + ", ".join(f"{coins[i]}({wv[i]*100:.0f}%)" for i in sorted(win, key=lambda i:-wv[i])))
    print(f"  SHORT: " + ", ".join(f"{coins[i]}({-wv[i]*100:.0f}%)" for i in sorted(los, key=lambda i:wv[i])))
    names = set([coins[i] for i in win] + [coins[i] for i in los])
    print(f"  ({len(names)} names; broader universe -> less single-name concentration)")
