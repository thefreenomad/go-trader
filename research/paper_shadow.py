"""Step 5: paper-shadow — the honest pre-deployment expectation + live ledger.

Replays the DEPLOYABLE config's exact logic (expanded HL universe, inverse-vol +
per-name cap, vol-target, biweekly, weak-bear gate) net of real HL costs+funding,
focused on the RECENT window (deployment-relevant), and writes a per-rebalance
ledger (target + realized) -- the same format the live paper shadow continues.

Surfaces the recent-vs-full gap (the trade data showed the edge faded recently),
so we deploy with realistic expectations, not bull-loaded full-sample numbers.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd, ccxt
from data_fetcher import load_cached_data
from regime import compute_regime_composite

START = "2023-01-01"; TF = "4h"; BARS_DAY = 6
WEEK = 7 * BARS_DAY; STEP = 2 * WEEK; LB = 4 * WEEK; AUM = 1e6
QFRAC = 0.20; NLIQ = 40; NAME_CAP = 0.25
TARGET_VOL = 0.15; LEV_CAP = (0.25, 3.0); VT_WARMUP = 8; WEEKS_YR = 365 / 7
IMPACT_FULL = 0.005; HALF_K = 8.0; RECENT = pd.Timestamp("2025-09-01")

hl = ccxt.hyperliquid({"enableRateLimit": True}); hl.load_markets(); htk = hl.fetch_tickers()
try: hfr = hl.fetch_funding_rates()
except Exception: hfr = {}
def hl_vol(c): return (htk.get(f"{c}/USDC:USDC") or {}).get("quoteVolume")
def hl_fund(c):
    fr = (hfr.get(f"{c}/USDC:USDC") or {}).get("fundingRate")
    return 0.11 if fr is None else float(np.clip(fr * 24 * 365, -0.5, 1.0))

syms = json.load(open("research/universe.json"))
data = {}; master = None
for sym in syms:
    c = sym.split("/")[0]
    if hl_vol(c) is None: continue
    d = load_cached_data(sym, TF, start_date=START); d = d[d.index >= pd.Timestamp(START)]
    if len(d) < 200: continue
    data[c] = d
    master = d.index if master is None else master.union(d.index)
master = master.sort_values()
Pall = pd.DataFrame({c: data[c]["close"] for c in data}).reindex(master)
keep = [c for c in Pall.columns if (lambda s: s.first_valid_index() is not None and
        s.loc[s.first_valid_index():s.last_valid_index()].notna().mean() >= 0.97)(Pall[c])]
liq = sorted(keep, key=lambda c: -(hl_vol(c) or 0))[:NLIQ]
Pv = Pall[liq].values; Rv = Pall[liq].pct_change().values; N = len(master)
V = np.array([max(hl_vol(c) or 1e5, 1e5) for c in liq]); F = np.array([hl_fund(c) for c in liq])
hspread = np.clip(HALF_K / np.sqrt(V / 1e6), 1, 25) / 1e4
btc_lab = compute_regime_composite(data["BTC"], period=90)["regime"].reindex(master)


def cap(iv):
    w = iv / iv.sum()
    for _ in range(20):
        over = w > NAME_CAP + 1e-9
        if not over.any(): break
        exc = (w[over] - NAME_CAP).sum(); w[over] = NAME_CAP
        un = ~over
        if not un.any(): break
        w[un] += exc * (w[un] / w[un].sum())
    return w


ledger = []
eq = 1.0; wp = np.zeros(len(liq)); hist = []
for b in range(LB, N - STEP, STEP):
    date = master[b]; w = np.zeros(len(liq)); raw = 0.0; fund = 0.0; lev = 1.0; nl = ns = 0
    gate = btc_lab.iloc[b] == "trending_down_choppy"
    if not gate:
        mom = Pv[b] / Pv[b - LB] - 1
        seg = Pv[b:b + STEP + 1, :]; fwd = seg[-1] / seg[0] - 1 if seg.shape[0] >= 2 else np.full(len(liq), np.nan)
        v = np.nanstd(Rv[b - LB:b, :], axis=0)
        elig = np.where(~np.isnan(mom) & ~np.isnan(fwd) & ~np.isnan(v) & (v > 0))[0]
        if elig.size >= 8:
            order = elig[np.argsort(mom[elig])]; k = max(1, int(elig.size * QFRAC))
            losers, winners = order[:k], order[-k:]; nl = ns = k
            if len(hist) >= VT_WARMUP:
                rv = np.std(hist[-VT_WARMUP:], ddof=1)
                if rv > 0: lev = float(np.clip((TARGET_VOL / np.sqrt(WEEKS_YR)) / rv, *LEV_CAP))
            for ix, sg in [(winners, 1), (losers, -1)]:
                w[ix] = sg * lev * cap(1.0 / v[ix])
            raw = float(np.sum(w * np.nan_to_num(fwd)))
            fund = (STEP / BARS_DAY / 365) * float(np.sum(np.where(w > 0, w, 0) * F) - np.sum(np.where(w < 0, -w, 0) * F))
    hist.append(raw)
    turn = np.abs(w - wp); cost = float(np.sum(turn * (hspread + IMPACT_FULL * np.sqrt(np.maximum(turn * AUM, 0) / V))))
    r = raw - cost - fund; eq *= (1 + r); wp = w
    ledger.append(dict(date=date, gated=gate, n_long=nl, n_short=ns, lev=round(lev, 2),
                       gross=round(np.sum(np.abs(w)), 2), ret=round(r, 4), equity=round(eq, 4)))

L = pd.DataFrame(ledger)
L.to_csv("research/results/paper_ledger.csv", index=False)


def stats(s):
    if len(s) < 3 or s.std() == 0: return 0, 0, 0
    e = (1 + s).cumprod()
    return (e.iloc[-1] - 1) * 100, (s.mean() / s.std()) * np.sqrt(WEEKS_YR), (e / e.cummax() - 1).min() * 100


full = pd.Series(L.ret.values, index=pd.DatetimeIndex(L.date))
recent = full[full.index >= RECENT]
print(f"Paper-shadow ledger: {len(L)} rebalances -> research/results/paper_ledger.csv")
print(f"Deployable config: expanded HL universe ({len(liq)} coins), caps, weak-bear gate, net HL costs+funding\n")
print(f"  {'window':22s} {'rebal':>6s} {'total%':>8s} {'Sharpe':>7s} {'maxDD%':>7s}")
for lab, s in [("FULL (2023->now)", full), (f"RECENT (since {RECENT.date()})", recent)]:
    t, sh, m = stats(s)
    print(f"  {lab:22s} {len(s):>6d} {t:>8.1f} {sh:>7.2f} {m:>7.1f}")

print("\n=== last 8 rebalances (the live-shadow ledger tail) ===")
print(L.tail(8).to_string(index=False))

print("""
=== DEPLOYMENT RUNBOOK (go-trader paper mode) ===
 1. Build:    cd scheduler && go build -o ../go-trader .
 2. Config:   merge research/results/xs_manual_slots.json strategies into scheduler/config.json
              (type:"manual" HL slots, --mode=paper); set discord hyperliquid-paper channel.
 3. Cron:     biweekly -> PYTHONPATH=. python research/orchestrator.py --aum <AUM> --execute
              (drives ./go-trader manual-open/close on the paper book)
 4. Monitor:  curl localhost:8099/status ; dashboard /dashboard ; Discord paper channel.
 5. Reconcile: weekly, append realized fills to this ledger; compare realized vs this
              shadow's predicted Sharpe. Go live only after weeks of matching paper track.
""")
