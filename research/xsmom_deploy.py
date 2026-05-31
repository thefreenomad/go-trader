"""Deployability tests for the locked cross-sectional momentum config
(8w / inverse-vol / vol-target / biweekly / q20 / no-skip).

  1. LIQUIDITY restriction — does the edge survive on only the top-N liquid coins
     (dropping the illiquid small-caps where slippage/borrow is worst)?
  2. DIRECTION mode — long/short (base) vs long-only vs long-tilt (1.0L / 0.5S).
     Long-only = far more deployable (no borrow, no short funding, spot-capable).
  3. FUNDING sensitivity — momentum LONGS recent winners (high positive funding ->
     you pay). Apply an annualized funding drag on the long leg; find breakeven.

All with the locked config. Sub-period (bull/bear) reported throughout, since
that's the bar a config must clear.
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd, ccxt
from data_fetcher import load_cached_data

START = "2023-01-01"; TF = "4h"; BARS_DAY = 6
WEEK = 7 * BARS_DAY; STEP = 2 * WEEK          # biweekly (locked)
LB = 360; QFRAC = 0.20; COST = 0.00085
TARGET_VOL = 0.15; LEV_CAP = (0.25, 3.0); VT_WARMUP = 8; WEEKS_YR = 365 / 7
STEP_DAYS = 14; SPLIT = pd.Timestamp("2025-01-18")

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
P = P[keep]

# liquidity rank (current 24h quote volume as a proxy)
ex = ccxt.binanceus({"enableRateLimit": True}); tk = ex.fetch_tickers()
vol = {c: (tk.get(f"{c}/USDT") or {}).get("quoteVolume", 0) or 0 for c in P.columns}
by_liq = sorted(P.columns, key=lambda c: -vol[c])
N = len(master)


def run(cols, lb=LB, step=STEP, mode="ls", funding=0.0):
    sub = P[cols]; Pv = sub.values; Rv = sub.pct_change().values
    rebals = list(range(lb, N - step, step))
    eq = 1.0; wr = []; idx = []; pl = set(); ps = set(); hist = []
    long_w = 1.0; short_w = {"ls": 1.0, "longtilt": 0.5, "longonly": 0.0}[mode]
    for b in rebals:
        mom = Pv[b] / Pv[b - lb] - 1
        seg = Pv[b:b + step + 1, :]; fwd = seg[-1] / seg[0] - 1 if seg.shape[0] >= 2 else np.full(len(cols), np.nan)
        v = np.nanstd(Rv[b - lb:b, :], axis=0)
        elig = np.where(~np.isnan(mom) & ~np.isnan(fwd) & ~np.isnan(v) & (v > 0))[0]
        if elig.size < 8:
            wr.append(0.0); idx.append(master[b]); hist.append(0.0); continue
        k = max(1, int(elig.size * QFRAC)); order = elig[np.argsort(mom[elig])]
        sh, lo = order[:k], order[-k:]
        def leg(ix):
            w = 1.0 / v[ix]; w /= w.sum(); return float(np.sum(w * fwd[ix]))
        raw = long_w * leg(lo) - short_w * leg(sh)
        lev = 1.0
        if len(hist) >= VT_WARMUP:
            rv = np.std(hist[-VT_WARMUP:], ddof=1)
            if rv > 0:
                lev = float(np.clip((TARGET_VOL / np.sqrt(WEEKS_YR)) / rv, *LEV_CAP))
        nl, ns = set(lo.tolist()), set(sh.tolist())
        turn = (len(nl ^ pl) + (len(ns ^ ps) if short_w else 0)) / (2 * k * (2 if short_w else 1))
        gross = lev * (long_w + short_w)
        fund_drag = funding * (STEP_DAYS / 365) * lev * long_w     # longs pay funding
        r = lev * raw - turn * 2 * COST - fund_drag
        pl, ps = nl, ns
        eq *= (1 + r); wr.append(r); idx.append(master[b]); hist.append(lev * raw)
    return pd.Series(wr, index=pd.DatetimeIndex(idx))


def stats(wr):
    if len(wr) < 3:
        return 0, 0, 0
    eq = (1 + wr).cumprod()
    return (eq.iloc[-1] - 1) * 100, (wr.mean() / wr.std()) * np.sqrt(WEEKS_YR) if wr.std() > 0 else 0, (eq / eq.cummax() - 1).min() * 100


def sub(wr):
    f = stats(wr); b = stats(wr[wr.index < SPLIT]); e = stats(wr[wr.index >= SPLIT])
    return f, b, e


print(f"Universe {len(keep)} coins; liquidity order (top 12): {by_liq[:12]}\n")

print("=== 1. LIQUIDITY RESTRICTION (locked L/S config) — full / bull / bear Sharpe ===")
print(f"  {'universe':14s} {'full':>14s} {'bull':>8s} {'bear':>8s} {'maxDD':>7s}")
for n in [15, 25, 40, len(keep)]:
    f, b, e = sub(run(by_liq[:n]))
    print(f"  top-{n:<3d} liquid {f[1]:>6.2f} ({f[0]:>4.0f}%) {b[1]:>7.2f} {e[1]:>7.2f} {f[2]:>7.0f}")

print("\n=== 2. DIRECTION MODE (top-25 liquid) ===")
print(f"  {'mode':12s} {'full Shp':>9s} {'total%':>8s} {'bull':>7s} {'bear':>7s} {'maxDD':>7s}")
for mode in ["ls", "longtilt", "longonly"]:
    f, b, e = sub(run(by_liq[:25], mode=mode))
    print(f"  {mode:12s} {f[1]:>9.2f} {f[0]:>7.0f}% {b[1]:>7.2f} {e[1]:>7.2f} {f[2]:>7.0f}")

print("\n=== 3. FUNDING DRAG SENSITIVITY (annualized on long leg; top-25 liquid) ===")
print(f"  {'funding/yr':11s} {'L/S Sharpe':>11s} {'L/S total':>10s} | {'long-only Shp':>13s} {'LO total':>9s}")
for fund in [0.0, 0.10, 0.20, 0.30, 0.50]:
    ls = stats(run(by_liq[:25], mode="ls", funding=fund))
    lo = stats(run(by_liq[:25], mode="longonly", funding=fund))
    print(f"  {fund*100:>8.0f}%   {ls[1]:>11.2f} {ls[0]:>9.0f}% | {lo[1]:>13.2f} {lo[0]:>8.0f}%")

print("\n=== BTC buy&hold reference (same span) ===")
btc = P["BTC"].iloc[LB:]; print(f"  BTC {(btc.iloc[-1]/btc.iloc[0]-1)*100:+.0f}%")
