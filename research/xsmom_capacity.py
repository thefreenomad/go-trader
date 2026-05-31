"""Final cost gate: realistic LIQUIDITY-SCALED slippage + market impact -> true
deployable Sharpe and CAPACITY (how much AUM the edge absorbs).

Validated config: 4w lookback, inverse-vol, vol-target 15%, biweekly, top-25
liquid, L/S, + weak-bear gate (sit out BTC trending_down_choppy), net of 20%/yr
funding on the long leg.

Cost model (per coin, per side), transparent + adjustable:
  half-spread  = clip(25 / sqrt(daily_$vol_M), 3, 50) bps     (liquidity-tiered)
  impact       = IMPACT_FULL * sqrt(traded_notional / daily_$vol)   (square-root law)
                 IMPACT_FULL = 100 bps  (cost at 100% of ADV) -> ~10bps at 1% ADV
Trade notional scales with AUM, so impact grows with size -> capacity curve.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd, ccxt
from data_fetcher import load_cached_data
from regime import compute_regime_composite

START = "2023-01-01"; TF = "4h"; BARS_DAY = 6
WEEK = 7 * BARS_DAY; STEP = 2 * WEEK; LB = 4 * WEEK
QFRAC = 0.20; STEP_DAYS = 14
TARGET_VOL = 0.15; LEV_CAP = (0.25, 3.0); VT_WARMUP = 8; WEEKS_YR = 365 / 7
TOPLIQ = 25; SPLIT = pd.Timestamp("2025-01-18")
FUNDING = 0.20; IMPACT_FULL = 0.01     # 100 bps at 100% of ADV

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
V = np.array([max(vol[c], 1e5) for c in liq])                     # daily $ volume per coin
half_spread = np.clip(25.0 / np.sqrt(V / 1e6), 3, 50) / 1e4       # fraction, per side
btc_lab = compute_regime_composite(data["BTC"], period=90)["regime"].reindex(master)
mom4 = (Pl / Pl.shift(LB)) - 1
breadth = ((mom4 > 0).sum(axis=1) / mom4.notna().sum(axis=1)).values
rebals = list(range(LB, N - STEP, STEP))

print(f"Top-{TOPLIQ} liquid. daily $vol range: ${V.min()/1e6:.1f}M (smallest) .. ${V.max()/1e6:.0f}M (BTC)")
print(f"half-spread/side: {half_spread.min()*1e4:.0f}-{half_spread.max()*1e4:.0f} bps; "
      f"impact = {IMPACT_FULL*1e4:.0f}bps*sqrt(traded/ADV)\n")


def run(AUM, gate=True, flat_cost=None):
    eq = 1.0; ret = []; idx = []; w_prev = np.zeros(len(liq)); hist = []
    for b in rebals:
        if gate and btc_lab.iloc[b] == "trending_down_choppy":
            ret.append(0.0); idx.append(master[b]); w_prev = np.zeros(len(liq)) if False else w_prev; hist.append(hist[-1] if hist else 0.0)
            # sit out: hold cash, unwind book -> turnover cost to flatten
            turn = np.abs(0 - w_prev)
            if flat_cost is None:
                notional = turn * AUM
                cost = np.sum(turn * (half_spread + IMPACT_FULL * np.sqrt(np.maximum(notional, 0) / V)))
            else:
                cost = np.sum(turn) * flat_cost
            eq *= (1 - cost); ret[-1] = -cost; w_prev = np.zeros(len(liq))
            continue
        mom = Pv[b] / Pv[b - LB] - 1
        seg = Pv[b:b + STEP + 1, :]; fwd = seg[-1] / seg[0] - 1 if seg.shape[0] >= 2 else np.full(len(liq), np.nan)
        v = np.nanstd(Rv[b - LB:b, :], axis=0)
        elig = np.where(~np.isnan(mom) & ~np.isnan(fwd) & ~np.isnan(v) & (v > 0))[0]
        if elig.size < 8:
            ret.append(0.0); idx.append(master[b]); hist.append(0.0); continue
        order = elig[np.argsort(mom[elig])]; k = max(1, int(elig.size * QFRAC))
        losers, winners = order[:k], order[-k:]
        lev = 1.0
        if len(hist) >= VT_WARMUP:
            rv = np.std(hist[-VT_WARMUP:], ddof=1)
            if rv > 0:
                lev = float(np.clip((TARGET_VOL / np.sqrt(WEEKS_YR)) / rv, *LEV_CAP))
        w = np.zeros(len(liq))
        for leg_ix, sgn in [(winners, +1), (losers, -1)]:
            iv = 1.0 / v[leg_ix]; iv = iv / iv.sum()
            w[leg_ix] = sgn * lev * iv
        raw = float(np.sum(w * np.nan_to_num(fwd)))
        hist.append(raw)
        turn = np.abs(w - w_prev)
        if flat_cost is None:
            notional = turn * AUM
            cost = np.sum(turn * (half_spread + IMPACT_FULL * np.sqrt(np.maximum(notional, 0) / V)))
        else:
            cost = np.sum(turn) * flat_cost
        fund = FUNDING * (STEP_DAYS / 365) * lev      # long-leg funding
        r = raw - cost - fund
        eq *= (1 + r); ret.append(r); idx.append(master[b]); w_prev = w
    return pd.Series(ret, index=pd.DatetimeIndex(idx))


def stats(w):
    if len(w) < 3 or w.std() == 0:
        return 0, 0, 0
    eq = (1 + w).cumprod()
    return (eq.iloc[-1] - 1) * 100, (w.mean() / w.std()) * np.sqrt(WEEKS_YR), (eq / eq.cummax() - 1).min() * 100


print("=== reference: flat 8.5bps/side (what prior steps assumed) ===")
t, s, m = stats(run(0, flat_cost=0.00085)); print(f"  Sharpe {s:.2f}  total {t:.0f}%  maxDD {m:.0f}\n")

print("=== CAPACITY: realistic liquidity-scaled slippage + impact, by AUM ===")
print(f"  {'AUM':>8s} {'Sharpe':>7s} {'total%':>8s} {'maxDD%':>7s} {'avg cost drag/yr':>17s}")
for AUM in [50e3, 250e3, 1e6, 5e6, 20e6, 50e6]:
    w = run(AUM)
    t, s, m = stats(w)
    # estimate annual cost drag vs no-cost
    wn = run(AUM, flat_cost=0.0); drag = (stats(wn)[1] - s)  # sharpe drag (rough)
    label = f"${AUM/1e6:.2f}M" if AUM >= 1e6 else f"${AUM/1e3:.0f}k"
    print(f"  {label:>8s} {s:>7.2f} {t:>8.0f} {m:>7.0f} {('~%.2f Sharpe' % drag):>17s}")

print("\n=== sub-period (bull/bear) at $1M AUM, realistic costs ===")
w = run(1e6)
for lab, mask in [("full", w.index == w.index), ("bull", w.index < SPLIT), ("bear", w.index >= SPLIT)]:
    _, s, m = stats(w[mask]); print(f"  {lab:5s} Sharpe {s:>5.2f}  maxDD {m:>5.0f}")
