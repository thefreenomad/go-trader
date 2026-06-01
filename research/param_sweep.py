"""Step 6: sweep the high-level XS-momentum knobs — momentum LOOKBACK x rebalance
CADENCE — to see if 4w/biweekly is actually the best operating point or just a
reasonable prior we validated once.

Same book as xsmom_validate (top-25 liquid, inverse-vol, vol-target 15%,
weak-bear regime gate, net cost+funding). Sharpe is annualized CORRECTLY per
cadence (sqrt(365/step_days)) so weekly and biweekly are comparable — so the
absolute numbers differ slightly from the biweekly-only validation, but the
cross-config ranking is apples-to-apples. We look for a ROBUST REGION (a plateau
of good configs), not the single max-Sharpe cell, and confirm the pick beats a
random-timing null.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd, ccxt
from data_fetcher import load_cached_data
from regime import compute_regime_composite

START = "2023-01-01"; TF = "4h"; BARS_DAY = 6; WEEK = 7 * BARS_DAY
QFRAC = 0.20; COST = 0.00085; TARGET_VOL = 0.15; LEV_CAP = (0.25, 3.0); VT_WARMUP = 8
TOPLIQ = 25; SPLIT = pd.Timestamp("2025-01-18"); FUND_BASE = 0.20; REGIME_PERIOD = 90
DOWN = "trending_down_choppy"

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
reg = compute_regime_composite(data["BTC"], period=REGIME_PERIOD)["regime"].reindex(master).values


def run(LB, STEP, trade=True, funding=FUND_BASE, gate_weakbear=True, cost=COST):
    """Biweekly/weekly XS-momentum return series for a given lookback+cadence."""
    step_days = STEP / BARS_DAY
    pp = 365.0 / step_days                              # rebalances per year (for VT + Sharpe)
    rebals = list(range(LB, N - STEP, STEP))
    ret, idx, hist = [], [], []
    pl, ps = set(), set()
    for b in rebals:
        mom = Pv[b] / Pv[b - LB] - 1
        seg = Pv[b:b + STEP + 1, :]; fwd = seg[-1] / seg[0] - 1
        v = np.nanstd(Rv[b - LB:b, :], axis=0)
        elig = np.where(~np.isnan(mom) & ~np.isnan(fwd) & ~np.isnan(v) & (v > 0))[0]
        if elig.size < 8:
            ret.append(0.0); idx.append(master[b]); hist.append(0.0); continue
        order = elig[np.argsort(mom[elig])]; k = max(1, int(elig.size * QFRAC))
        losers, winners = order[:k], order[-k:]
        def leg(ix):
            w = 1.0 / v[ix]; w /= w.sum(); return float(np.sum(w * fwd[ix]))
        raw = leg(winners) - leg(losers)
        lev = 1.0
        if len(hist) >= VT_WARMUP:
            rv = np.std(hist[-VT_WARMUP:], ddof=1)
            if rv > 0:
                lev = float(np.clip((TARGET_VOL / np.sqrt(pp)) / rv, *LEV_CAP))
        hist.append(lev * raw)
        gated = gate_weakbear and reg[b] == DOWN
        if gated or (trade is not True and not trade(b, rebals)):
            ret.append(0.0); idx.append(master[b]); pl, ps = set(), set(); continue
        nl, ns = set(winners.tolist()), set(losers.tolist())
        turn = (len(nl ^ pl) + len(ns ^ ps)) / (2 * k * 2)
        r = lev * raw - turn * 2 * cost - funding * (step_days / 365) * lev
        pl, ps = nl, ns
        ret.append(r); idx.append(master[b])
    return pd.Series(ret, index=pd.DatetimeIndex(idx)), pp


def stats(w, pp):
    if len(w) < 3 or w.std() == 0:
        return 0, 0, 0
    eq = (1 + w).cumprod()
    return (eq.iloc[-1] - 1) * 100, (w.mean() / w.std()) * np.sqrt(pp), (eq / eq.cummax() - 1).min() * 100


LOOKBACKS = [("1w", WEEK), ("2w", 2 * WEEK), ("3w", 3 * WEEK),
             ("4w", 4 * WEEK), ("6w", 6 * WEEK), ("8w", 8 * WEEK)]
CADENCES = [("weekly", WEEK), ("biweekly", 2 * WEEK)]

print(f"Top-{TOPLIQ} liquid; weak-bear gated; net cost+funding 20%/yr; "
      f"{START}..{str(master[-1].date())}\n")
print("=== SHARPE grid (rebalance cadence x momentum lookback) ===")
print(f"  {'cadence':10s}" + "".join(f"{lb:>9s}" for lb, _ in LOOKBACKS))
results = {}
for cname, step in CADENCES:
    cells = []
    for lbn, lb in LOOKBACKS:
        w, pp = run(lb, step)
        t, s, m = stats(w, pp)
        results[(cname, lbn)] = (t, s, m, w, pp)
        cells.append(f"{s:>9.2f}")
    print(f"  {cname:10s}" + "".join(cells))

print("\n=== same grid: TOTAL RETURN % ===")
print(f"  {'cadence':10s}" + "".join(f"{lb:>9s}" for lb, _ in LOOKBACKS))
for cname, _ in CADENCES:
    print(f"  {cname:10s}" + "".join(f"{results[(cname, lb)][0]:>9.0f}" for lb, _ in LOOKBACKS))

print("\n=== same grid: MAX DRAWDOWN % ===")
print(f"  {'cadence':10s}" + "".join(f"{lb:>9s}" for lb, _ in LOOKBACKS))
for cname, _ in CADENCES:
    print(f"  {cname:10s}" + "".join(f"{results[(cname, lb)][2]:>9.0f}" for lb, _ in LOOKBACKS))

# bull/bear split + random-null for the top configs (robustness, not just max in-sample)
ranked = sorted(results.items(), key=lambda kv: -kv[1][1])[:4]
print("\n=== top-4 configs: bull/bear split + random-timing null (30 seeds) ===")
print(f"  {'config':22s} {'Sharpe':>7s} {'bull':>6s} {'bear':>6s} {'maxDD':>6s} {'vs-random':>10s}")
for (cname, lbn), (t, s, m, w, pp) in ranked:
    step = dict(CADENCES)[cname]; lb = dict(LOOKBACKS)[lbn]
    bull = stats(w[w.index < SPLIT], pp)[1]; bear = stats(w[w.index >= SPLIT], pp)[1]
    n_trade = int((w != 0).sum()); rebals = list(range(lb, N - step, step))
    rnd = []
    for sd in range(30):
        rng = np.random.default_rng(sd + 1)
        pick = set(rng.choice(len(rebals), size=min(n_trade, len(rebals)), replace=False).tolist())
        wr, ppr = run(lb, step, trade=lambda b, rb, pick=pick: rb.index(b) in pick, gate_weakbear=False)
        rnd.append(stats(wr, ppr)[1])
    sd_above = (s - np.mean(rnd)) / (np.std(rnd) or 1)
    print(f"  {cname+'/'+lbn:22s} {s:>7.2f} {bull:>6.2f} {bear:>6.2f} {m:>6.0f}  {sd_above:>+8.1f}sd")

print(f"\n  (current live config = biweekly/4w. Look for a robust REGION, not the single peak.)")

# COST-SENSITIVITY — the decider for high-turnover configs. 8.5bp = base model;
# 17/25bp simulate worse real taker+slippage. Weekly/1w turns over ~2x biweekly,
# so if its edge is fee-driven it collapses here while the plateau holds.
print("\n=== COST sensitivity (Sharpe @ one-way cost) — high-turnover configs decay fastest ===")
probe = [("weekly", "1w"), ("weekly", "2w"), ("biweekly", "3w"), ("biweekly", "4w")]
print(f"  {'config':14s} {'8.5bp':>7s} {'17bp':>7s} {'25bp':>7s} {'turnover/yr':>12s}")
for cname, lbn in probe:
    step = dict(CADENCES)[cname]; lb = dict(LOOKBACKS)[lbn]
    rebals_per_yr = 365.0 / (step / BARS_DAY)
    row = []
    for c in (0.00085, 0.0017, 0.0025):
        w, pp = run(lb, step, cost=c)
        row.append(stats(w, pp)[1])
    print(f"  {cname+'/'+lbn:14s} {row[0]:>7.2f} {row[1]:>7.2f} {row[2]:>7.2f} {rebals_per_yr:>10.0f}x")

