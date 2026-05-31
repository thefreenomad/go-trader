"""Cross-sectional momentum v2 — breadth fix + risk layer (bearing in mind XS-mom
is the edge we found).

Upgrades over xsmom.py:
  1. POINT-IN-TIME eligibility — a coin trades only when it has full trailing data
     and is still listed. Keeps clean late-listers AND mid-period delistings
     (reduces survivorship bias). Drops only interior-gap series (e.g. TRX).
  2. RISK LAYER — inverse-vol weighting within each leg + portfolio vol-targeting
     to a target annualized vol (tames the -57% DD).
  3. BTC correlation of the strategy's weekly returns (diversification value).

Long winners / short losers, dollar-neutral, weekly. Fee-aware, no funding.
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd
from data_fetcher import load_cached_data

START = "2023-01-01"; TF = "4h"; BARS_DAY = 6
STEP = 7 * BARS_DAY; COST = 0.00085; WEEKS_YR = 365 / 7
QFRAC = 0.20
TARGET_VOL = 0.15                      # annualized portfolio vol target
LEV_CAP = (0.25, 3.0); VT_WARMUP = 8   # weeks before vol-targeting kicks in
INTERIOR_COV_MIN = 0.97                # drop only interior-gap series
LOOKBACKS = {"4w": 180, "8w": 360, "12w": 540}

syms = json.load(open("research/universe.json"))
closes = {}; master = None
for sym in syms:
    d = load_cached_data(sym, TF, start_date=START)
    d = d[d.index >= pd.Timestamp(START)]
    if len(d) < 200:
        continue
    closes[sym.split("/")[0]] = d["close"]
    master = d.index if master is None else master.union(d.index)
P = pd.DataFrame(closes).reindex(master).sort_index()

# Clean filter: coverage WITHIN each coin's active [first_valid, last_valid] range.
keep = []
for c in P.columns:
    s = P[c]; fv, lv = s.first_valid_index(), s.last_valid_index()
    if fv is None:
        continue
    interior = s.loc[fv:lv]
    if interior.notna().mean() >= INTERIOR_COV_MIN:
        keep.append(c)
dropped = [c for c in P.columns if c not in keep]
P = P[keep]
coins = list(P.columns)
Pv = P.values; Rv = P.pct_change().values
N = len(master)
print(f"Universe: {len(coins)} coins (point-in-time); dropped interior-gap: {dropped}")
# avg eligible coins per week (rough)
print(f"Grid: {N} bars {master[0].date()} -> {master[-1].date()}\n")

rebals = list(range(max(LOOKBACKS.values()), N - STEP, STEP))


def fwd_block(b):
    seg = Pv[b:b + STEP + 1, :]
    return seg[-1] / seg[0] - 1 if seg.shape[0] >= 2 else np.full(len(coins), np.nan)


def run(lb, weighting="equal", voltarget=False, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    eq = 1.0; curve = []; prev_l = set(); prev_s = set()
    hist = []                              # weekly strategy returns for vol-targeting
    n_elig = []
    for b in rebals:
        mom = Pv[b] / Pv[b - lb] - 1
        fwd = fwd_block(b)
        vol = np.nanstd(Rv[b - lb:b, :], axis=0)
        elig = np.where(~np.isnan(mom) & ~np.isnan(fwd) & ~np.isnan(vol) & (vol > 0))[0]
        n_elig.append(elig.size)
        if elig.size < 8:
            curve.append(eq); hist.append(0.0); continue
        k = max(1, int(elig.size * QFRAC))
        order = elig[np.argsort(mom[elig])]
        if rng is not None:
            perm = rng.permutation(elig); sh, lo = perm[:k], perm[k:2 * k]
        else:
            sh, lo = order[:k], order[-k:]

        def legret(idx):
            if weighting == "invvol":
                w = 1.0 / vol[idx]; w = w / w.sum()
            else:
                w = np.full(len(idx), 1.0 / len(idx))
            return float(np.sum(w * fwd[idx]))
        raw = legret(lo) - legret(sh)
        lev = 1.0
        if voltarget and len(hist) >= VT_WARMUP:
            rv = np.std(hist[-VT_WARMUP:], ddof=1)
            if rv > 0:
                lev = np.clip((TARGET_VOL / np.sqrt(WEEKS_YR)) / rv, *LEV_CAP)
        nl, ns = set(lo.tolist()), set(sh.tolist())
        turn = (len(nl ^ prev_l) + len(ns ^ prev_s)) / (2 * k * 2)
        wk = lev * raw - turn * 2 * COST
        prev_l, prev_s = nl, ns
        eq *= (1 + wk); curve.append(eq); hist.append(lev * raw)
    return pd.Series(curve, index=[master[b] for b in rebals]), np.mean(n_elig)


def stats(c):
    wr = c.pct_change().dropna()
    tot = (c.iloc[-1] - 1) * 100
    shp = (wr.mean() / wr.std()) * np.sqrt(WEEKS_YR) if wr.std() > 0 else 0.0
    mdd = (c / c.cummax() - 1).min() * 100
    return tot, shp, mdd, wr


span0, span1 = master[rebals[0]], master[rebals[-1] + STEP]
btc = P["BTC"].loc[span0:span1]; btc_tot = (btc.dropna().iloc[-1] / btc.dropna().iloc[0] - 1) * 100
btc_wk = P["BTC"].reindex([master[b] for b in rebals]).pct_change()
ewp = (P.loc[span0:span1] / P.loc[span0:span1].bfill().iloc[0]).mean(axis=1)
print(f"=== BENCHMARKS (span {span0.date()} -> {span1.date()}, {len(rebals)} weeks) ===")
print(f"  BTC buy&hold {btc_tot:+.1f}% | equal-weight universe {(ewp.dropna().iloc[-1]-1)*100:+.1f}%")
_, avg_elig = run(540)
print(f"  avg eligible coins / week: {avg_elig:.0f}\n")

print("=== RISK-MANAGED CROSS-SECTIONAL MOMENTUM ===")
print(f"  {'lookback':9s} {'variant':22s} {'total%':>8s} {'Sharpe':>7s} {'maxDD%':>7s} {'BTCcorr':>8s}")
for lb_name, lb in LOOKBACKS.items():
    for variant, kw in [("equal-weight", dict(weighting="equal")),
                        ("inverse-vol", dict(weighting="invvol")),
                        ("invvol+voltarget", dict(weighting="invvol", voltarget=True))]:
        c, _ = run(lb, **kw)
        tot, shp, mdd, wr = stats(c)
        corr = wr.corr(btc_wk.reindex(wr.index))
        print(f"  {lb_name:9s} {variant:22s} {tot:>7.0f}% {shp:>7.2f} {mdd:>7.0f} {corr:>8.2f}")
    print()

print("=== RANDOM CONTROL (invvol+voltarget, avg 20 seeds) ===")
for lb_name, lb in LOOKBACKS.items():
    rr = [stats(run(lb, weighting="invvol", voltarget=True, seed=s)[0])[:3] for s in range(20)]
    print(f"  {lb_name:9s} random total {np.mean([x[0] for x in rr]):+6.1f}%  Sharpe {np.mean([x[1] for x in rr]):+.2f}")
