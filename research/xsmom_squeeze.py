"""Test the squeeze refinements from the trade-data analysis, under discipline:
each net of HL costs, with concentration re-checked, combined vs random.

  base          — current config (top-25 HL, short bottom 20%, no cap, 2w hold)
  skip-oversold — short the band 10-30% (skip the bouncing absolute bottom)
  name-cap      — cap each coin at 25% of its leg (reduce single-name dominance)
  hold-3w       — 3-week holding (signal is slow; cut turnover)
  expand-40     — top-40 HL coins (more breadth -> less FET dependence)
  COMBINED      — expand + skip-oversold + name-cap

Reports Sharpe (full/bull/bear), top-5 & FET P&L share, net at $1M on Hyperliquid.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd, ccxt
from data_fetcher import load_cached_data
from regime import compute_regime_composite

START = "2023-01-01"; TF = "4h"; BARS_DAY = 6
WEEK = 7 * BARS_DAY; LB = 4 * WEEK; AUM = 1e6
QFRAC = 0.20; STEP_DAYS_PER_BAR = 1 / BARS_DAY
TARGET_VOL = 0.15; LEV_CAP = (0.25, 3.0); VT_WARMUP = 8; WEEKS_YR = 365 / 7
SPLIT = pd.Timestamp("2025-01-18")
IMPACT_FULL = 0.005; HALF_K = 8.0

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
ranked = sorted(keep, key=lambda c: -(hl_vol(c) or 0))
btc_lab = compute_regime_composite(data["BTC"], period=90)["regime"].reindex(master)
N = len(master)
print(f"HL-listed coins available: {len(ranked)}\n")


def run(nliq=25, short_skip=0.0, name_cap=1.0, hold_w=2, seed=None):
    liq = ranked[:nliq]
    Pv = Pall[liq].values; Rv = Pall[liq].pct_change().values
    V = np.array([max(hl_vol(c) or 1e5, 1e5) for c in liq])
    F = np.array([hl_fund(c) for c in liq])
    hspread = np.clip(HALF_K / np.sqrt(V / 1e6), 1, 25) / 1e4
    STEP = hold_w * WEEK
    rebals = list(range(LB, N - STEP, STEP))
    rng = np.random.default_rng(seed) if seed is not None else None
    eq = 1.0; ret = []; idx = []; wp = np.zeros(len(liq)); hist = []; pnl_by_coin = {}
    for b in rebals:
        w = np.zeros(len(liq)); raw = 0.0; fund = 0.0; lev = 1.0
        if btc_lab.iloc[b] != "trending_down_choppy":
            mom = Pv[b] / Pv[b - LB] - 1
            seg = Pv[b:b + STEP + 1, :]; fwd = seg[-1] / seg[0] - 1 if seg.shape[0] >= 2 else np.full(len(liq), np.nan)
            v = np.nanstd(Rv[b - LB:b, :], axis=0)
            elig = np.where(~np.isnan(mom) & ~np.isnan(fwd) & ~np.isnan(v) & (v > 0))[0]
            if elig.size >= 8:
                order = elig[np.argsort(mom[elig])]; k = max(1, int(elig.size * QFRAC))
                skip = int(elig.size * short_skip)
                if rng is not None:
                    perm = rng.permutation(elig); losers, winners = perm[:k], perm[k:2 * k]
                else:
                    winners = order[-k:]; losers = order[skip:skip + k]
                if len(hist) >= VT_WARMUP:
                    rv = np.std(hist[-VT_WARMUP:], ddof=1)
                    if rv > 0: lev = float(np.clip((TARGET_VOL / np.sqrt(WEEKS_YR)) / rv, *LEV_CAP))
                for leg_ix, sgn in [(winners, +1), (losers, -1)]:
                    iv = 1.0 / v[leg_ix]; iv = iv / iv.sum()
                    iv = np.minimum(iv, name_cap); iv = iv / iv.sum()      # per-name cap
                    w[leg_ix] = sgn * lev * iv
                raw = float(np.sum(w * np.nan_to_num(fwd)))
                fund = (STEP * STEP_DAYS_PER_BAR / 365) * float(np.sum(np.where(w > 0, w, 0) * F) - np.sum(np.where(w < 0, -w, 0) * F))
                for i in np.where(w != 0)[0]:
                    pnl_by_coin[liq[i]] = pnl_by_coin.get(liq[i], 0) + w[i] * np.nan_to_num(fwd)[i]
        turn = np.abs(w - wp); notional = turn * AUM
        cost = float(np.sum(turn * (hspread + IMPACT_FULL * np.sqrt(np.maximum(notional, 0) / V))))
        r = raw - cost - fund
        eq *= (1 + r); ret.append(r); idx.append(master[b]); wp = w
    s = pd.Series(ret, index=pd.DatetimeIndex(idx))
    return s, pnl_by_coin


def shp(s): return (s.mean() / s.std()) * np.sqrt(WEEKS_YR) if len(s) > 2 and s.std() > 0 else 0.0
def conc(pnl):
    cc = pd.Series(pnl); tot = cc.sum()
    top5 = cc.sort_values(ascending=False).head(5).sum() / tot * 100 if tot else 0
    fet = cc.get("FET", 0) / tot * 100 if tot else 0
    return top5, fet


print(f"  {'variant':16s} {'full':>5s} {'bull':>5s} {'bear':>5s} {'top5%':>6s} {'FET%':>6s}")
configs = [
    ("base", dict()),
    ("skip-oversold", dict(short_skip=0.10)),
    ("name-cap .25", dict(name_cap=0.25)),
    ("hold-3w", dict(hold_w=3)),
    ("expand-40", dict(nliq=40)),
    ("COMBINED", dict(nliq=40, short_skip=0.10, name_cap=0.25)),
]
combined_s = None
for name, kw in configs:
    s, pnl = run(**kw)
    if name == "COMBINED": combined_s = s
    t5, fet = conc(pnl)
    print(f"  {name:16s} {shp(s):>5.2f} {shp(s[s.index<SPLIT]):>5.2f} {shp(s[s.index>=SPLIT]):>5.2f} {t5:>6.0f} {fet:>6.0f}")

# random null for COMBINED
rs = [shp(run(nliq=40, short_skip=0.10, name_cap=0.25, seed=i)[0]) for i in range(20)]
print(f"\n  COMBINED Sharpe {shp(combined_s):.2f}  vs random {np.mean(rs):.2f} +/- {np.std(rs):.2f}  "
      f"-> {((shp(combined_s)-np.mean(rs))/(np.std(rs) or 1)):+.1f} sd")
