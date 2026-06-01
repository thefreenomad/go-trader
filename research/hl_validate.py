"""Re-validate the cadence x lookback decision on the ACTUAL deployed universe:
HL mainnet top-40 crypto (36 coins, from fetch_hl_data.py), with REAL per-coin
HL funding when the cache exists (else flat 20% for the universe-only check).

Answers loose-ends #1 (universe) and #2 (funding): is weekly/1w still the winner
on the universe we trade, and what's the honest Sharpe once real funding — which
momentum systematically pays (it longs high-funding winners) — is charged?

  hl_validate.py            # flat funding (isolates the universe effect, #1)
  hl_validate.py real       # real per-coin HL funding (#2)

Bare book (monitor unscheduled), same sizing/gate/cost as the BinanceUS sweep.
Point-in-time: coins enter when their HL history begins (NaN-gated).
"""
import sys, os, json, glob, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd
from regime import compute_regime_composite

CACHE = "research/data/hl"; BARS_DAY = 6; WEEK = 7 * BARS_DAY
QFRAC = 0.20; COST = 0.00085; TARGET_VOL = 0.15; LEV_CAP = (0.25, 3.0); VT_WARMUP = 8
FUND_FLAT = 0.20; REGIME_PERIOD = 90; DOWN = "trending_down_choppy"
SPLIT = pd.Timestamp("2025-06-01")   # ~midpoint of the HL window (2024-02..2026-06)
REAL_FUND = len(sys.argv) > 1 and sys.argv[1] == "real"

# ---- load cached HL OHLCV ----
ohlcv = {}
for f in glob.glob(f"{CACHE}/*_4h.json"):
    c = os.path.basename(f)[:-8]
    o = json.load(open(f))
    df = pd.DataFrame(o, columns=["t", "o", "h", "l", "c", "v"])
    df.index = pd.to_datetime(df["t"], unit="ms")
    ohlcv[c] = df[~df.index.duplicated()]
master = None
for df in ohlcv.values():
    master = df.index if master is None else master.union(df.index)
master = master.sort_values()
coins = sorted(ohlcv)
P = pd.DataFrame({c: ohlcv[c]["c"] for c in coins}).reindex(master).ffill()
Pv = P.values; Rv = P.pct_change().values; N = len(master)
btc = ohlcv["BTC"].reindex(master).ffill()
reg = compute_regime_composite(btc.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}),
                               period=REGIME_PERIOD)["regime"].reindex(master).values

# ---- real per-coin funding: cumulative funding fraction aligned to the 4h grid ----
cumF = None
if REAL_FUND:
    cumF = np.zeros((N, len(coins)))
    pos = {c: i for i, c in enumerate(coins)}
    t0 = master.asi8 // 10**6                           # 4h-bar start times, ms (ndarray)
    for c in coins:
        fp = f"{CACHE}/{c}_fund.json"
        if not os.path.exists(fp):
            continue
        pts = json.load(open(fp))                       # [(ts_ms, hourly_rate)]
        if not pts:
            continue
        per4h = np.zeros(N)
        ts = np.array([p[0] for p in pts]); rt = np.array([p[1] for p in pts], float)
        bins = np.searchsorted(t0, ts, side="right") - 1   # which 4h bar each hourly pt falls in
        for bi, r in zip(bins, rt):
            if 0 <= bi < N:
                per4h[bi] += r
        cumF[:, pos[c]] = np.cumsum(per4h)


def book(b, LB):
    mom = Pv[b] / Pv[b - LB] - 1
    seg = Pv[b:b + STEP_G + 1, :]; fwd = seg[-1] / seg[0] - 1
    v = np.nanstd(Rv[b - LB:b, :], axis=0)
    elig = np.where(~np.isnan(mom) & ~np.isnan(fwd) & ~np.isnan(v) & (v > 0))[0]
    if elig.size < 8:
        return None
    order = elig[np.argsort(mom[elig])]; k = max(1, int(elig.size * QFRAC))
    return order[:k], order[-k:], v, fwd, k


def simulate(LB, STEP):
    global STEP_G; STEP_G = STEP
    step_days = STEP / BARS_DAY; pp = 365.0 / step_days
    rebals = list(range(LB, N - STEP, STEP))
    ret, idx, hist = [], [], []; pl, ps = set(), set()
    for b in rebals:
        bk = book(b, LB)
        if bk is None:
            ret.append(0.0); idx.append(master[b]); hist.append(0.0); continue
        L, W, v, fwd, k = bk
        wl = 1.0 / v[W]; wl /= wl.sum(); ws = 1.0 / v[L]; ws /= ws.sum()
        raw = float(np.sum(wl * fwd[W]) - np.sum(ws * fwd[L]))
        lev = 1.0
        if len(hist) >= VT_WARMUP:
            rv = np.std(hist[-VT_WARMUP:], ddof=1)
            if rv > 0:
                lev = float(np.clip((TARGET_VOL / np.sqrt(pp)) / rv, *LEV_CAP))
        hist.append(lev * raw)
        if reg[b] == DOWN:
            ret.append(0.0); idx.append(master[b]); pl, ps = set(), set(); continue
        nlw = set(W.tolist()); nls = set(L.tolist())
        turn = (len(nlw ^ pl) + len(nls ^ ps)) / (2 * k * 2)
        if REAL_FUND:                                   # signed per-coin: long pays F, short receives F
            F = cumF[b + STEP] - cumF[b]
            fund = lev * (float(np.sum(wl * F[W])) - float(np.sum(ws * F[L])))
        else:
            fund = FUND_FLAT * (step_days / 365) * lev   # flat long-leg haircut
        ret.append(lev * raw - turn * 2 * COST - fund)
        idx.append(master[b]); pl, ps = nlw, nls
    return pd.Series(ret, index=pd.DatetimeIndex(idx)), pp


def stats(w, pp):
    if len(w) < 3 or w.std() == 0:
        return 0, 0, 0
    eq = (1 + w).cumprod()
    return (eq.iloc[-1] - 1) * 100, (w.mean() / w.std()) * np.sqrt(pp), (eq / eq.cummax() - 1).min() * 100


LOOK = [("1w", WEEK), ("2w", 2 * WEEK), ("3w", 3 * WEEK), ("4w", 4 * WEEK), ("6w", 6 * WEEK)]
CAD = [("weekly", WEEK), ("biweekly", 2 * WEEK)]
fmode = "REAL per-coin HL funding" if REAL_FUND else "flat 20%/yr funding"
print(f"HL universe ({len(coins)} crypto): {', '.join(coins)}")
print(f"window {str(master[0].date())}..{str(master[-1].date())} ({(master[-1]-master[0]).days/365:.1f}y); {fmode}; weak-bear gated\n")
res = {}
for cn, st in CAD:
    for ln, lb in LOOK:
        res[(cn, ln)] = simulate(lb, st)
print("=== SHARPE grid (cadence x lookback) — HL universe ===")
print(f"  {'cadence':10s}" + "".join(f"{l:>9s}" for l, _ in LOOK))
for cn, _ in CAD:
    print(f"  {cn:10s}" + "".join(f"{stats(*res[(cn, ln)])[1]:>9.2f}" for ln, _ in LOOK))
print(f"\n=== TOTAL RETURN % over {(master[-1]-master[0]).days/365:.1f}y — HL universe ===")
print(f"  {'cadence':10s}" + "".join(f"{l:>9s}" for l, _ in LOOK))
for cn, _ in CAD:
    print(f"  {cn:10s}" + "".join(f"{stats(*res[(cn, ln)])[0]:>9.0f}" for ln, _ in LOOK))

print("\n=== focus: deployed config + neighbours (bull/bear split) ===")
print(f"  {'config':16s} {'tot%':>7s} {'Sharpe':>7s} {'bull':>6s} {'bear':>6s} {'maxDD':>6s}")
for cn, ln in [("weekly", "1w"), ("weekly", "2w"), ("biweekly", "3w"), ("biweekly", "4w")]:
    w, pp = res[(cn, ln)]; t, s, m = stats(w, pp)
    bull = stats(w[w.index < SPLIT], pp)[1]; bear = stats(w[w.index >= SPLIT], pp)[1]
    print(f"  {cn+'/'+ln:16s} {t:>7.0f} {s:>7.2f} {bull:>6.2f} {bear:>6.2f} {m:>6.0f}")
print(f"\n  (compare to BinanceUS top-25: weekly/1w ~1.4 bare. Lower here = honest universe/funding effect.)")
