"""Robustness lock for the cross-sectional momentum edge.

Takes the best config (8w lookback, inverse-vol + vol-target) and stresses it:
  A. sub-period stability — bull (2023-24) vs bear (2025-26)  [the acid test]
  B. transaction-cost sensitivity
  C. quantile breadth (decile..30%)
  D. skip-recent (classic 12-1 style) momentum
  E. multi-horizon blend vs single lookback
  F. rebalance frequency (weekly vs biweekly)
  G. lookback fine grid (is it a plateau or a knife-edge?)

If the edge holds in BOTH market halves and survives realistic costs, it's real.
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd
from data_fetcher import load_cached_data

START = "2023-01-01"; TF = "4h"; BARS_DAY = 6
WEEK = 7 * BARS_DAY; COST0 = 0.00085; WEEKS_YR = 365 / 7
TARGET_VOL = 0.15; LEV_CAP = (0.25, 3.0); VT_WARMUP = 8
SPLIT = pd.Timestamp("2025-01-18")    # bull/bear boundary (cycle top)

syms = json.load(open("research/universe.json"))
closes = {}; master = None
for sym in syms:
    d = load_cached_data(sym, TF, start_date=START); d = d[d.index >= pd.Timestamp(START)]
    if len(d) < 200:
        continue
    closes[sym.split("/")[0]] = d["close"]
    master = d.index if master is None else master.union(d.index)
P = pd.DataFrame(closes).reindex(master).sort_index()
keep = [c for c in P.columns if (lambda s: s.first_valid_index() is not None and
        s.loc[s.first_valid_index():s.last_valid_index()].notna().mean() >= 0.97)(P[c])]
P = P[keep]; Pv = P.values; Rv = P.pct_change().values; N = len(master)
print(f"Universe: {len(keep)} coins, {N} bars {master[0].date()}->{master[-1].date()}\n")


def momentum(b, lb, skip):
    return Pv[b - skip] / Pv[b - lb] - 1


def blended_rank(b, lbs, skip):
    zs = []
    for lb in lbs:
        m = Pv[b - skip] / Pv[b - lb] - 1
        zs.append(m)
    M = np.vstack(zs)
    # z-score each row across coins, then average
    mu = np.nanmean(M, axis=1, keepdims=True); sd = np.nanstd(M, axis=1, keepdims=True)
    Z = (M - mu) / np.where(sd > 0, sd, np.nan)
    return np.nanmean(Z, axis=0)


def run(lb=360, skip=0, qfrac=0.2, voltarget=True, cost=COST0, step=WEEK, blend=None):
    rebals = list(range((max(blend) if blend else lb) + skip, N - step, step))
    eq = 1.0; wk_rets = []; idx = []; prev_l = set(); prev_s = set(); hist = []
    for b in rebals:
        score = blended_rank(b, blend, skip) if blend else momentum(b, lb, skip)
        seg = Pv[b:b + step + 1, :]; fwd = seg[-1] / seg[0] - 1 if seg.shape[0] >= 2 else np.full(len(keep), np.nan)
        vol = np.nanstd(Rv[b - lb:b, :], axis=0)
        elig = np.where(~np.isnan(score) & ~np.isnan(fwd) & ~np.isnan(vol) & (vol > 0))[0]
        if elig.size < 8:
            wk_rets.append(0.0); idx.append(master[b]); hist.append(0.0); continue
        k = max(1, int(elig.size * qfrac)); order = elig[np.argsort(score[elig])]
        sh, lo = order[:k], order[-k:]
        def leg(ix):
            w = 1.0 / vol[ix]; w /= w.sum(); return float(np.sum(w * fwd[ix]))
        raw = leg(lo) - leg(sh)
        lev = 1.0
        if voltarget and len(hist) >= VT_WARMUP:
            rv = np.std(hist[-VT_WARMUP:], ddof=1)
            if rv > 0:
                lev = float(np.clip((TARGET_VOL / np.sqrt(WEEKS_YR)) / rv, *LEV_CAP))
        nl, ns = set(lo.tolist()), set(sh.tolist())
        turn = (len(nl ^ prev_l) + len(ns ^ prev_s)) / (2 * k * 2)
        r = lev * raw - turn * 2 * cost
        prev_l, prev_s = nl, ns
        eq *= (1 + r); wk_rets.append(r); idx.append(master[b]); hist.append(lev * raw)
    return pd.Series(wk_rets, index=pd.DatetimeIndex(idx))


def stats(wr):
    if len(wr) < 3:
        return 0, 0, 0
    eq = (1 + wr).cumprod(); tot = (eq.iloc[-1] - 1) * 100
    shp = (wr.mean() / wr.std()) * np.sqrt(WEEKS_YR) if wr.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min() * 100
    return tot, shp, mdd


base = run()  # 8w, invvol, voltarget
btc_wk = P["BTC"].reindex(base.index).pct_change()

print("=== A. SUB-PERIOD STABILITY (base: 8w, invvol+voltarget) — THE ACID TEST ===")
print(f"  {'period':18s} {'weeks':>5s} {'total%':>8s} {'Sharpe':>7s} {'maxDD%':>7s} {'BTCcorr':>8s}")
for label, mask in [("full", base.index == base.index),
                    ("bull (->2025-01)", base.index < SPLIT),
                    ("bear (2025-01->)", base.index >= SPLIT)]:
    wr = base[mask]; t, s, m = stats(wr)
    corr = wr.corr(btc_wk[mask])
    print(f"  {label:18s} {len(wr):>5d} {t:>7.0f}% {s:>7.2f} {m:>7.0f} {corr:>8.2f}")

print("\n=== B. TRANSACTION-COST SENSITIVITY (bps/side) ===")
for bps in [5, 8.5, 15, 25, 40]:
    t, s, m = stats(run(cost=bps / 1e4))
    print(f"  {bps:>5.1f} bps  total {t:>6.0f}%  Sharpe {s:>5.2f}  maxDD {m:>5.0f}")

print("\n=== C. QUANTILE BREADTH ===")
for q in [0.10, 0.15, 0.20, 0.30]:
    t, s, m = stats(run(qfrac=q))
    print(f"  top/bottom {q*100:>3.0f}%   total {t:>6.0f}%  Sharpe {s:>5.2f}  maxDD {m:>5.0f}")

print("\n=== D. SKIP-RECENT (classic 12-1 style) ===")
for sk, name in [(0, "skip 0"), (WEEK, "skip 1w"), (2 * WEEK, "skip 2w")]:
    t, s, m = stats(run(skip=sk))
    print(f"  {name:8s}  total {t:>6.0f}%  Sharpe {s:>5.2f}  maxDD {m:>5.0f}")

print("\n=== E. MULTI-HORIZON BLEND (4/8/12w z-score) vs single 8w ===")
t, s, m = stats(run()); print(f"  single 8w        total {t:>6.0f}%  Sharpe {s:>5.2f}  maxDD {m:>5.0f}")
t, s, m = stats(run(blend=[180, 360, 540])); print(f"  blend 4/8/12w    total {t:>6.0f}%  Sharpe {s:>5.2f}  maxDD {m:>5.0f}")

print("\n=== F. REBALANCE FREQUENCY ===")
for st, name in [(WEEK, "weekly"), (2 * WEEK, "biweekly")]:
    t, s, m = stats(run(step=st))
    print(f"  {name:9s}  total {t:>6.0f}%  Sharpe {s:>5.2f}  maxDD {m:>5.0f}")

print("\n=== G. LOOKBACK FINE GRID (plateau check) ===")
for wks in [2, 4, 6, 8, 10, 12, 16, 20]:
    t, s, m = stats(run(lb=wks * WEEK))
    print(f"  {wks:>2d}w  total {t:>6.0f}%  Sharpe {s:>5.2f}  maxDD {m:>5.0f}")
