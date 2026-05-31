"""Hyperliquid-native capacity + cost model for the validated XS-momentum config.

Venue = Hyperliquid only. Real HL data:
  * universe = our coins that trade on HL, ranked by HL perp $volume
  * slippage = reasonable HL-CLOB: tight half-spread (liquidity-tiered) + sqrt impact
  * funding  = each coin's actual current HL funding (long pays its funding, short
               receives its funding) -> net momentum-book funding from real rates
Signals from BinanceUS OHLCV (prices ~identical across venues for these coins).
Config: 4w / inverse-vol / vol-target 15% / biweekly / L/S / weak-bear gate.
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
NLIQ = 25; SPLIT = pd.Timestamp("2025-01-18")
HALF_SPREAD_K = 8.0; SPREAD_FLOOR, SPREAD_CAP = 1.0, 25.0   # bps, HL CLOB (tight)
IMPACT_FULL = 0.005          # 50 bps at 100% of ADV (HL majors deep)

# --- HL data ---
hl = ccxt.hyperliquid({"enableRateLimit": True}); hl.load_markets()
htk = hl.fetch_tickers()
try:
    hfr = hl.fetch_funding_rates()
except Exception:
    hfr = {}
def hl_vol(c):
    t = htk.get(f"{c}/USDC:USDC") or {}
    return t.get("quoteVolume")
def hl_fund(c):  # annualized, clipped to sane range
    fr = (hfr.get(f"{c}/USDC:USDC") or {}).get("fundingRate")
    if fr is None:
        return 0.11
    return float(np.clip(fr * 24 * 365, -0.5, 1.0))

syms = json.load(open("research/universe.json"))
data = {}; master = None
for sym in syms:
    c = sym.split("/")[0]
    if hl_vol(c) is None:
        continue                           # HL-only: skip non-HL coins
    d = load_cached_data(sym, TF, start_date=START); d = d[d.index >= pd.Timestamp(START)]
    if len(d) < 200:
        continue
    data[c] = d
    master = d.index if master is None else master.union(d.index)
master = master.sort_values()
P = pd.DataFrame({c: data[c]["close"] for c in data}).reindex(master)
keep = [c for c in P.columns if (lambda s: s.first_valid_index() is not None and
        s.loc[s.first_valid_index():s.last_valid_index()].notna().mean() >= 0.97)(P[c])]
liq = sorted(keep, key=lambda c: -(hl_vol(c) or 0))[:NLIQ]
Pl = P[liq]; Pv = Pl.values; Rv = Pl.pct_change().values; N = len(master)
V = np.array([max(hl_vol(c) or 1e5, 1e5) for c in liq])
F = np.array([hl_fund(c) for c in liq])                       # annualized funding per coin
half_spread = np.clip(HALF_SPREAD_K / np.sqrt(V / 1e6), SPREAD_FLOOR, SPREAD_CAP) / 1e4
btc_lab = compute_regime_composite(data["BTC"], period=90)["regime"].reindex(master)
rebals = list(range(LB, N - STEP, STEP))

print(f"HL universe: {len(liq)} coins. HL $vol ${V.min()/1e6:.0f}M..${V.max()/1e6:.0f}M; "
      f"funding {F.min()*100:.0f}%..{F.max()*100:.0f}% (mean {F.mean()*100:.0f}%)")
print(f"slippage/side {half_spread.min()*1e4:.1f}-{half_spread.max()*1e4:.1f}bps + {IMPACT_FULL*1e4:.0f}bps*sqrt(traded/ADV)\n")


def run(AUM, flat_cost=None, funding_on=True):
    eq = 1.0; ret = []; idx = []; w_prev = np.zeros(len(liq)); hist = []
    for b in rebals:
        gate_off = btc_lab.iloc[b] == "trending_down_choppy"
        mom = Pv[b] / Pv[b - LB] - 1
        seg = Pv[b:b + STEP + 1, :]; fwd = seg[-1] / seg[0] - 1 if seg.shape[0] >= 2 else np.full(len(liq), np.nan)
        v = np.nanstd(Rv[b - LB:b, :], axis=0)
        elig = np.where(~np.isnan(mom) & ~np.isnan(fwd) & ~np.isnan(v) & (v > 0))[0]
        w = np.zeros(len(liq)); raw = 0.0; fund = 0.0; lev = 1.0
        if not gate_off and elig.size >= 8:
            order = elig[np.argsort(mom[elig])]; k = max(1, int(elig.size * QFRAC))
            losers, winners = order[:k], order[-k:]
            if len(hist) >= VT_WARMUP:
                rv = np.std(hist[-VT_WARMUP:], ddof=1)
                if rv > 0:
                    lev = float(np.clip((TARGET_VOL / np.sqrt(WEEKS_YR)) / rv, *LEV_CAP))
            for leg_ix, sgn in [(winners, +1), (losers, -1)]:
                iv = 1.0 / v[leg_ix]; iv = iv / iv.sum(); w[leg_ix] = sgn * lev * iv
            raw = float(np.sum(w * np.nan_to_num(fwd)))
            if funding_on:
                # long pays its funding, short receives its funding
                fund = (STEP_DAYS / 365) * float(np.sum(np.where(w > 0, w, 0) * F) - np.sum(np.where(w < 0, -w, 0) * F))
        hist.append(raw)
        turn = np.abs(w - w_prev)
        if flat_cost is None:
            notional = turn * AUM
            cost = float(np.sum(turn * (half_spread + IMPACT_FULL * np.sqrt(np.maximum(notional, 0) / V))))
        else:
            cost = float(np.sum(turn)) * flat_cost
        r = raw - cost - fund
        eq *= (1 + r); ret.append(r); idx.append(master[b]); w_prev = w
    return pd.Series(ret, index=pd.DatetimeIndex(idx))


def stats(w):
    if len(w) < 3 or w.std() == 0:
        return 0, 0, 0
    eq = (1 + w).cumprod()
    return (eq.iloc[-1] - 1) * 100, (w.mean() / w.std()) * np.sqrt(WEEKS_YR), (eq / eq.cummax() - 1).min() * 100


print("=== reference: no costs / no funding (pure signal) ===")
t, s, m = stats(run(0, flat_cost=0.0, funding_on=False)); print(f"  Sharpe {s:.2f}  total {t:.0f}%\n")

print("=== HYPERLIQUID CAPACITY: real HL slippage+impact + HL funding, by AUM ===")
print(f"  {'AUM':>8s} {'Sharpe':>7s} {'total%':>8s} {'maxDD%':>7s}")
for AUM in [50e3, 250e3, 1e6, 5e6, 20e6, 50e6]:
    t, s, m = stats(run(AUM))
    label = f"${AUM/1e6:.1f}M" if AUM >= 1e6 else f"${AUM/1e3:.0f}k"
    print(f"  {label:>8s} {s:>7.2f} {t:>8.0f} {m:>7.0f}")

print("\n=== decompose at $1M AUM ===")
base = stats(run(1e6, flat_cost=0.0, funding_on=False))[1]
nofund = stats(run(1e6, funding_on=False))[1]
full = stats(run(1e6))[1]
print(f"  pure signal {base:.2f}  -> +slippage {nofund:.2f}  -> +HL funding {full:.2f}")
w = run(1e6)
for lab, mask in [("full", w.index == w.index), ("bull", w.index < SPLIT), ("bear", w.index >= SPLIT)]:
    _, s, m = stats(w[mask]); print(f"  {lab:5s} Sharpe {s:>5.2f}  maxDD {m:>5.0f}")
