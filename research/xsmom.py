"""Cross-sectional momentum / reversal on the ~58-coin universe — "pick the best
coins to trade." Each week, rank all eligible coins by trailing return, go long a
top quantile / short a bottom quantile, hold 7d, rebalance.

Tests the matrix of LOOKBACK x SIGNAL:
  momentum  — long top / short bottom (winners keep winning)
  reversal  — long bottom / short top (losers bounce)
  long-only — long top quantile only (no shorting)
vs RANDOM long/short selection (control) and BTC buy&hold.

Point-in-time eligibility (only coins listed with full trailing data). Fee-aware,
1x, no funding. Survivorship-biased universe (today's survivors).
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd
from data_fetcher import load_cached_data

START = "2023-01-01"; TF = "4h"; BARS_DAY = 6
STEP = 7 * BARS_DAY
COST = 0.00085; WEEKS_YR = 365 / 7
QFRAC = 0.20            # top/bottom 20%
COVERAGE_MIN = 0.85    # drop gappy series (e.g. TRX)
LOOKBACKS = {"1w": 42, "2w": 84, "4w": 180, "8w": 360, "12w": 540}

syms = json.load(open("research/universe.json"))
# Load closes, build master grid
closes = {}; master = None
for sym in syms:
    d = load_cached_data(sym, TF, start_date=START)
    d = d[d.index >= pd.Timestamp(START)]
    if len(d) < 200:
        continue
    closes[sym.split("/")[0]] = d["close"]
    master = d.index if master is None else master.union(d.index)
P = pd.DataFrame(closes).reindex(master).sort_index()
# Coverage filter (drop gappy)
cov = P.notna().mean()
keep = cov[cov >= COVERAGE_MIN].index.tolist()
dropped = cov[cov < COVERAGE_MIN].index.tolist()
P = P[keep]
coins = list(P.columns)
ret = P.pct_change()                      # per-bar simple returns
Pv = P.values; Rv = ret.values
N = len(master)
print(f"Universe: {len(coins)} coins (dropped {len(dropped)} gappy: {dropped})")
print(f"Grid: {N} bars {master[0].date()} -> {master[-1].date()}; weekly rebalance\n")

rebals = list(range(max(LOOKBACKS.values()), N - STEP, STEP))


def fwd_block(b):
    """next-STEP compounded return per coin (NaN if not fully present)."""
    seg = Pv[b:b + STEP + 1, :]
    if seg.shape[0] < 2:
        return np.full(len(coins), np.nan)
    return seg[-1] / seg[0] - 1


def run(lb_bars, signal, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    eq = 1.0; curve = []; prev_l = set(); prev_s = set()
    for b in rebals:
        mom = Pv[b] / Pv[b - lb_bars] - 1            # trailing momentum
        fwd = fwd_block(b)
        elig = np.where(~np.isnan(mom) & ~np.isnan(fwd))[0]
        if elig.size < 6:
            curve.append(eq); continue
        k = max(1, int(len(elig) * QFRAC))
        order = elig[np.argsort(mom[elig])]          # ascending: losers..winners
        if rng is not None:
            perm = rng.permutation(elig); lo, sh = perm[:k], perm[k:2 * k]
        elif signal == "momentum":
            sh, lo = order[:k], order[-k:]
        elif signal == "reversal":
            lo, sh = order[:k], order[-k:]
        else:  # long-only momentum
            lo, sh = order[-k:], np.array([], int)
        rl = np.nanmean(fwd[lo]) if lo.size else 0.0
        rs = np.nanmean(fwd[sh]) if sh.size else 0.0
        wk = rl - rs if sh.size else rl
        # turnover cost (both legs)
        nl, ns = set(lo.tolist()), set(sh.tolist())
        turn = (len(nl ^ prev_l) + len(ns ^ prev_s)) / max(2 * k * (2 if sh.size else 1), 1)
        wk -= turn * 2 * COST; prev_l, prev_s = nl, ns
        eq *= (1 + wk); curve.append(eq)
    return pd.Series(curve, index=[master[b] for b in rebals])


def stats(c):
    wr = c.pct_change().dropna()
    tot = (c.iloc[-1] - 1) * 100
    shp = (wr.mean() / wr.std()) * np.sqrt(WEEKS_YR) if wr.std() > 0 else 0.0
    mdd = (c / c.cummax() - 1).min() * 100
    return tot, shp, mdd


# Benchmarks
span0, span1 = master[rebals[0]], master[rebals[-1] + STEP]
btc = P["BTC"].loc[span0:span1]; btc_tot = (btc.dropna().iloc[-1] / btc.dropna().iloc[0] - 1) * 100
ewp = (P.loc[span0:span1] / P.loc[span0:span1].bfill().iloc[0]).mean(axis=1)
ew_tot = (ewp.dropna().iloc[-1] - 1) * 100
print(f"=== BENCHMARKS (span {span0.date()} -> {span1.date()}, {len(rebals)} weeks) ===")
print(f"  BTC buy&hold          {btc_tot:+8.1f}%")
print(f"  Equal-weight universe {ew_tot:+8.1f}%\n")

print("=== CROSS-SECTIONAL: total% / weekly-Sharpe / maxDD%, by lookback x signal ===")
print(f"  {'lookback':9s} | {'MOMENTUM (L win/S lose)':>26s} | {'REVERSAL (L lose/S win)':>26s} | {'LONG-ONLY mom':>20s}")
for lb, bars in LOOKBACKS.items():
    cols = []
    for sig in ["momentum", "reversal", "longonly"]:
        t, s, m = stats(run(bars, sig))
        cols.append(f"{t:+7.0f}% {s:+5.2f} {m:5.0f}")
    print(f"  {lb:9s} | {cols[0]:>26s} | {cols[1]:>26s} | {cols[2]:>20s}")

print("\n=== RANDOM CONTROL (long/short random, avg 20 seeds) per lookback ===")
for lb, bars in LOOKBACKS.items():
    rr = [stats(run(bars, "momentum", seed=s)) for s in range(20)]
    print(f"  {lb:9s}  random total {np.mean([x[0] for x in rr]):+7.1f}%  Sharpe {np.mean([x[1] for x in rr]):+.2f}")
