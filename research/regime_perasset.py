"""Should each asset's OWN regime gate whether we trade it? Test per-asset
regime gating vs the market/breadth gate for the cross-sectional book.

  none        — rank all (base)
  directional — long only coins NOT in a downtrend; short only coins NOT in uptrend
  strict      — long only UP-trend coins; short only DOWN-trend coins (step 2b form)
  cleanonly   — only rank coins in CLEAN trend states (drop choppy from universe)
  breadth     — MARKET gate: trade only when universe breadth is high (the winner)
  breadth+strict — do they stack?

Per-coin composite labels via the repo classifier. 4w & 8w, top-25 liquid.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd, ccxt
from data_fetcher import load_cached_data
from regime import compute_regime_composite

START = "2023-01-01"; TF = "4h"; BARS_DAY = 6
WEEK = 7 * BARS_DAY; STEP = 2 * WEEK
QFRAC = 0.20; COST = 0.00085
TARGET_VOL = 0.15; LEV_CAP = (0.25, 3.0); VT_WARMUP = 8; WEEKS_YR = 365 / 7
REGIME_PERIOD = 90; TOPLIQ = 25; SPLIT = pd.Timestamp("2025-01-18")
UP = {"trending_up_clean", "trending_up_choppy"}
DOWN = {"trending_down_clean", "trending_down_choppy"}
CLEAN = {"trending_up_clean", "trending_down_clean"}

syms = json.load(open("research/universe.json"))
data = {}; master = None
for sym in syms:
    d = load_cached_data(sym, TF, start_date=START); d = d[d.index >= pd.Timestamp(START)]
    if len(d) < 200:
        continue
    data[sym.split("/")[0]] = d
    master = d.index if master is None else master.union(d.index)
master = master.sort_values()
P = pd.DataFrame({c: data[c]["close"] for c in data}).reindex(master)
keep = [c for c in P.columns if (lambda s: s.first_valid_index() is not None and
        s.loc[s.first_valid_index():s.last_valid_index()].notna().mean() >= 0.97)(P[c])]
ex = ccxt.binanceus({"enableRateLimit": True}); tk = ex.fetch_tickers()
vol = {c: (tk.get(f"{c}/USDT") or {}).get("quoteVolume", 0) or 0 for c in keep}
liq = sorted(keep, key=lambda c: -vol[c])[:TOPLIQ]
Pl = P[liq]; Pv = Pl.values; Rv = Pl.pct_change().values; N = len(master)
LAB = pd.DataFrame({c: compute_regime_composite(data[c], period=REGIME_PERIOD)["regime"].reindex(master)
                    for c in liq})[liq].values
mom4 = (Pl / Pl.shift(4 * WEEK)) - 1
breadth = ((mom4 > 0).sum(axis=1) / mom4.notna().sum(axis=1)).values
br_hi = np.nanquantile(breadth, 2 / 3)


def run(lb, gate="none"):
    rebals = list(range(lb, N - STEP, STEP))
    eq = 1.0; wr = []; idx = []; pl = set(); ps = set(); hist = []
    for b in rebals:
        if gate in ("breadth", "breadth+strict") and not (breadth[b] > br_hi):
            wr.append(0.0); idx.append(master[b]); hist.append(0.0); continue
        mom = Pv[b] / Pv[b - lb] - 1
        seg = Pv[b:b + STEP + 1, :]; fwd = seg[-1] / seg[0] - 1 if seg.shape[0] >= 2 else np.full(len(liq), np.nan)
        v = np.nanstd(Rv[b - lb:b, :], axis=0)
        ok = ~np.isnan(mom) & ~np.isnan(fwd) & ~np.isnan(v) & (v > 0)
        lab_b = LAB[b]
        if gate == "cleanonly":
            ok = ok & np.array([s in CLEAN for s in lab_b])
        elig = np.where(ok)[0]
        if elig.size < 8:
            wr.append(0.0); idx.append(master[b]); hist.append(0.0); continue
        order = elig[np.argsort(mom[elig])]; k = max(1, int(elig.size * QFRAC))
        losers, winners = order[:k], order[-k:]
        if gate in ("directional",):
            winners = np.array([i for i in winners if lab_b[i] not in DOWN], int)
            losers = np.array([i for i in losers if lab_b[i] not in UP], int)
        if gate in ("strict", "breadth+strict"):
            winners = np.array([i for i in winners if lab_b[i] in UP], int)
            losers = np.array([i for i in losers if lab_b[i] in DOWN], int)

        def leg(ix):
            if ix.size == 0:
                return 0.0
            w = 1.0 / v[ix]; w /= w.sum(); return float(np.sum(w * fwd[ix]))
        raw = leg(winners) - leg(losers)
        lev = 1.0
        if len(hist) >= VT_WARMUP:
            rv = np.std(hist[-VT_WARMUP:], ddof=1)
            if rv > 0:
                lev = float(np.clip((TARGET_VOL / np.sqrt(WEEKS_YR)) / rv, *LEV_CAP))
        nl, ns = set(winners.tolist()), set(losers.tolist())
        turn = (len(nl ^ pl) + len(ns ^ ps)) / (2 * k * 2)
        r = lev * raw - turn * 2 * COST; pl, ps = nl, ns
        eq *= (1 + r); wr.append(r); idx.append(master[b]); hist.append(lev * raw)
    return pd.Series(wr, index=pd.DatetimeIndex(idx))


def shp(w):
    return (w.mean() / w.std()) * np.sqrt(WEEKS_YR) if len(w) > 2 and w.std() > 0 else 0.0


print("Per-asset regime gating vs market/breadth gate (top-25 liquid)\n")
print(f"  {'gate':16s} | {'4w: full/bull/bear':>26s} | {'8w: full/bull/bear':>26s}")
for gate in ["none", "directional", "strict", "cleanonly", "breadth", "breadth+strict"]:
    cells = []
    for lb in [4 * WEEK, 8 * WEEK]:
        w = run(lb, gate)
        traded = (w != 0).sum()
        cells.append(f"{shp(w):>5.2f}/{shp(w[w.index<SPLIT]):>5.2f}/{shp(w[w.index>=SPLIT]):>5.2f} ({traded})")
    print(f"  {gate:16s} | {cells[0]:>26s} | {cells[1]:>26s}")
print("\n  (full/bull/bear Sharpe; (n)=rebalances actually traded; breadth gate uses in-sample tercile)")
