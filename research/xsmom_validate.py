"""Step 3: rigorously validate the breadth-gated cross-sectional momentum strategy.

Config (locked, deployable): 4w lookback, inverse-vol, vol-target 15%, biweekly,
top-25 liquid, L/S. Gate: universe BREADTH (% of coins with +4w momentum).

Validation discipline:
  * CAUSAL breadth threshold — rolling quantile of TRAILING breadth (no lookahead).
  * vs RANDOM-GATE null — sit out the same fraction of periods, at random times.
    If breadth-gating doesn't beat random-gating, the timing is noise.
  * Rolling-window Sharpe distribution (per ~6mo block) — consistency, not 2 halves.
  * Net of FUNDING (the deferred cost) — long leg pays; sensitivity shown.
  * ONE gate only.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd, ccxt
from data_fetcher import load_cached_data

START = "2023-01-01"; TF = "4h"; BARS_DAY = 6
WEEK = 7 * BARS_DAY; STEP = 2 * WEEK; LB = 4 * WEEK
QFRAC = 0.20; COST = 0.00085; STEP_DAYS = 14
TARGET_VOL = 0.15; LEV_CAP = (0.25, 3.0); VT_WARMUP = 8; WEEKS_YR = 365 / 7
TOPLIQ = 25; SPLIT = pd.Timestamp("2025-01-18")
BR_WIN_BARS = 360          # trailing window for causal breadth quantile (~60d)
BR_Q = 2 / 3               # trade when breadth in top third of its recent range
FUND_BASE = 0.20           # base annualized funding on the long leg

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
mom4 = (Pl / Pl.shift(LB)) - 1
breadth = ((mom4 > 0).sum(axis=1) / mom4.notna().sum(axis=1)).values

rebals = list(range(max(LB, BR_WIN_BARS), N - STEP, STEP))


def causal_breadth_trade(b):
    """Causal: trade if current breadth exceeds the BR_Q quantile of trailing breadth."""
    hist = breadth[b - BR_WIN_BARS:b]
    hist = hist[~np.isnan(hist)]
    if hist.size < 30 or np.isnan(breadth[b]):
        return True
    return breadth[b] > np.quantile(hist, BR_Q)


def book_return(b):
    """Gross (lev-pre) book return + the vol-target leverage given history -- helpers."""
    mom = Pv[b] / Pv[b - LB] - 1
    seg = Pv[b:b + STEP + 1, :]; fwd = seg[-1] / seg[0] - 1 if seg.shape[0] >= 2 else np.full(len(liq), np.nan)
    v = np.nanstd(Rv[b - LB:b, :], axis=0)
    elig = np.where(~np.isnan(mom) & ~np.isnan(fwd) & ~np.isnan(v) & (v > 0))[0]
    if elig.size < 8:
        return None
    order = elig[np.argsort(mom[elig])]; k = max(1, int(elig.size * QFRAC))
    losers, winners = order[:k], order[-k:]
    def leg(ix):
        w = 1.0 / v[ix]; w /= w.sum(); return float(np.sum(w * fwd[ix]))
    return leg(winners) - leg(losers), set(winners.tolist()), set(losers.tolist()), k


def run(trade_fn, funding=FUND_BASE):
    eq = 1.0; ret = []; idx = []; pl = set(); ps = set(); hist = []
    for b in rebals:
        bk = book_return(b)
        if bk is None:
            ret.append(0.0); idx.append(master[b]); hist.append(0.0); continue
        raw, nl, ns, k = bk
        lev = 1.0
        if len(hist) >= VT_WARMUP:
            rv = np.std(hist[-VT_WARMUP:], ddof=1)
            if rv > 0:
                lev = float(np.clip((TARGET_VOL / np.sqrt(WEEKS_YR)) / rv, *LEV_CAP))
        hist.append(lev * raw)
        if not trade_fn(b):
            ret.append(0.0); idx.append(master[b]); continue   # sit out (flat)
        turn = (len(nl ^ pl) + len(ns ^ ps)) / (2 * k * 2)
        fund = funding * (STEP_DAYS / 365) * lev
        r = lev * raw - turn * 2 * COST - fund; pl, ps = nl, ns
        eq *= (1 + r); ret.append(r); idx.append(master[b])
    return pd.Series(ret, index=pd.DatetimeIndex(idx))


def stats(w):
    if len(w) < 3 or w.std() == 0:
        return 0, 0, 0
    eq = (1 + w).cumprod()
    return (eq.iloc[-1] - 1) * 100, (w.mean() / w.std()) * np.sqrt(WEEKS_YR), (eq / eq.cummax() - 1).min() * 100


def sub(w):
    return stats(w)[1], stats(w[w.index < SPLIT])[1], stats(w[w.index >= SPLIT])[1]


# traded mask from the causal breadth gate
trade_flags = [causal_breadth_trade(b) for b in rebals]
n_trade = sum(trade_flags); frac = n_trade / len(rebals)
print(f"Top-{TOPLIQ} liquid; {len(rebals)} biweekly rebalances; causal breadth gate trades {n_trade} ({frac*100:.0f}%)\n")

always = run(lambda b: True)
gated = run(causal_breadth_trade)
print("=== Always-on vs CAUSAL breadth gate (net of cost + funding 20%/yr) ===")
print(f"  {'config':22s} {'total%':>8s} {'full Shp':>9s} {'bull':>6s} {'bear':>6s} {'maxDD%':>7s}")
for name, w in [("always-on", always), ("breadth-gated (causal)", gated)]:
    t, s, m = stats(w); _, bu, be = sub(w)
    print(f"  {name:22s} {t:>8.0f} {s:>9.2f} {bu:>6.2f} {be:>6.2f} {m:>7.0f}")

print("\n=== vs RANDOM-GATE null (sit out same fraction, random times; 30 seeds) ===")
rng_master = np.random.default_rng(0)
rs = []
for sd in range(30):
    rng = np.random.default_rng(sd + 1)
    pick = set(rng.choice(len(rebals), size=n_trade, replace=False).tolist())
    w = run(lambda b, pick=pick: rebals.index(b) in pick)
    rs.append(stats(w)[1])
print(f"  breadth-gated Sharpe {stats(gated)[1]:.2f}  vs  random-gate Sharpe {np.mean(rs):.2f} "
      f"+/- {np.std(rs):.2f}  -> {((stats(gated)[1]-np.mean(rs))/ (np.std(rs) or 1)):+.1f} sd")

print("\n=== ROLLING consistency: Sharpe per ~6-month block (gated, realized incl. sit-outs) ===")
g = gated.copy(); g.index = pd.DatetimeIndex(g.index)
blocks = g.groupby(pd.Grouper(freq="2QS"))
pos = 0; tot = 0
for ts, w in blocks:
    if len(w) < 3:
        continue
    s = stats(w)[1]; tot += 1; pos += s > 0
    print(f"  {str(ts.date()):>11s}  n={len(w):>2d}  Sharpe {s:>5.2f}")
print(f"  -> {pos}/{tot} blocks positive")

print("\n=== FUNDING sensitivity (causal breadth gate) ===")
for f in [0.0, 0.10, 0.20, 0.30, 0.50]:
    t, s, m = stats(run(causal_breadth_trade, funding=f))
    print(f"  funding {f*100:>3.0f}%/yr  total {t:>6.0f}%  Sharpe {s:>5.2f}  maxDD {m:>5.0f}")

# ---- also re-test the step-2 composite weak-bear gate vs random null ----
from regime import compute_regime_composite
btc_lab = compute_regime_composite(data["BTC"], period=90)["regime"].reindex(master)
def weakbear_trade(b):
    return btc_lab.iloc[b] != "trending_down_choppy"
wb = run(weakbear_trade)
n_wb = sum(weakbear_trade(b) for b in rebals)
rs2 = []
for sd in range(30):
    rng = np.random.default_rng(sd + 100)
    pick = set(rng.choice(len(rebals), size=n_wb, replace=False).tolist())
    rs2.append(stats(run(lambda b, pick=pick: rebals.index(b) in pick))[1])
print("\n=== Step-2 composite WEAK-BEAR gate vs random null (causal labels; net funding 20%) ===")
print(f"  weak-bear-gated Sharpe {stats(wb)[1]:.2f} (trades {n_wb}/{len(rebals)})  vs  "
      f"random-gate {np.mean(rs2):.2f} +/- {np.std(rs2):.2f}  -> {((stats(wb)[1]-np.mean(rs2))/(np.std(rs2) or 1)):+.1f} sd")
