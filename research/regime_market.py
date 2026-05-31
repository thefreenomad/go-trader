"""Which MARKET-regime proxy best gates the alt cross-sectional book?
  BTC composite   — what we used (BTC-only)
  UNIVERSE index  — equal-weight rebased index of the traded universe (composite)
  BREADTH         — fraction of coins with positive trailing-4w momentum (terciles)

The strategy already trades all top-25 liquid coins; this only changes the GATE.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd, ccxt
from data_fetcher import load_cached_data
from regime import compute_regime_composite

START = "2023-01-01"; TF = "4h"; BARS_DAY = 6
WEEK = 7 * BARS_DAY; STEP = 2 * WEEK
QFRAC = 0.20; COST = 0.00085
TARGET_VOL = 0.15; LEV_CAP = (0.25, 3.0); VT_WARMUP = 8; WEEKS_YR = 365 / 7
REGIME_PERIOD = 90; TOPLIQ = 25; SPLIT = pd.Timestamp("2025-01-18")
STATES = ["trending_up_clean", "trending_up_choppy", "trending_down_clean",
          "trending_down_choppy", "ranging_directional", "ranging_volatile", "ranging_quiet"]

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

# --- market-regime proxies ---
btc_lab = compute_regime_composite(data["BTC"], period=REGIME_PERIOD)["regime"].reindex(master)

# equal-weight universe index OHLC (rebased per coin to its first valid bar)
def ew_index():
    cols = {}
    for f in ["open", "high", "low", "close"]:
        norm = []
        for c in liq:
            s = data[c][f].reindex(master)
            base = data[c]["close"].reindex(master).dropna()
            if base.empty:
                continue
            norm.append(s / base.iloc[0])
        cols[f] = pd.concat(norm, axis=1).mean(axis=1)
    return pd.DataFrame(cols).dropna()
uni = ew_index()
uni_lab = compute_regime_composite(uni, period=REGIME_PERIOD)["regime"].reindex(master)

# breadth = fraction of coins with positive trailing-4w return
lb4 = 4 * WEEK
mom4 = (Pl / Pl.shift(lb4)) - 1
breadth = (mom4 > 0).sum(axis=1) / mom4.notna().sum(axis=1)


def run(lb):
    rebals = list(range(lb, N - STEP, STEP))
    eq = 1.0; wr = []; idx = []; pl = set(); ps = set(); hist = []
    for b in rebals:
        mom = Pv[b] / Pv[b - lb] - 1
        seg = Pv[b:b + STEP + 1, :]; fwd = seg[-1] / seg[0] - 1 if seg.shape[0] >= 2 else np.full(len(liq), np.nan)
        v = np.nanstd(Rv[b - lb:b, :], axis=0)
        elig = np.where(~np.isnan(mom) & ~np.isnan(fwd) & ~np.isnan(v) & (v > 0))[0]
        if elig.size < 8:
            wr.append(0.0); idx.append(master[b]); hist.append(0.0); continue
        order = elig[np.argsort(mom[elig])]; k = max(1, int(elig.size * QFRAC))
        losers, winners = order[:k], order[-k:]
        def leg(ix):
            w = 1.0 / v[ix]; w /= w.sum(); return float(np.sum(w * fwd[ix]))
        raw = leg(winners) - leg(losers)
        lev = 1.0
        if len(hist) >= VT_WARMUP:
            rv = np.std(hist[-VT_WARMUP:], ddof=1)
            if rv > 0:
                lev = float(np.clip((TARGET_VOL / np.sqrt(WEEKS_YR)) / rv, *LEV_CAP))
        nl, ns = set(winners.tolist()), set(losers.tolist())
        turn = (len(nl ^ pl) + len(ns ^ ps)) / (2 * k * 2)
        r = lev * raw - turn * 2 * COST; pl, ps = nl, ns
        eq *= (1 + r); wr.append(r); idx.append(master[b]); hist.append(lev * raw)
    return pd.Series(wr, index=pd.DatetimeIndex(idx))


def shp(w):
    return (w.mean() / w.std()) * np.sqrt(WEEKS_YR) if len(w) > 2 and w.std() > 0 else 0.0


def sub(w):
    return shp(w), shp(w[w.index < SPLIT]), shp(w[w.index >= SPLIT])


print("Strategy trades all top-25 liquid coins; comparing GATE signals.\n")
print("=== UNIVERSE-index composite label distribution ===")
d = uni_lab.value_counts(normalize=True)
print("  " + "  ".join(f"{s.replace('trending_','t').replace('ranging_','r').replace('_','')}:{d.get(s,0)*100:.0f}%" for s in STATES))

for lb, name in [(4 * WEEK, "4w"), (8 * WEEK, "8w")]:
    wr = run(lb)
    print(f"\n========== {name} book (overall Sharpe {shp(wr):.2f}) ==========")
    for gate, lab in [("BTC", btc_lab), ("UNIVERSE", uni_lab)]:
        st = lab.reindex(wr.index)
        worst = min(((s, shp(wr[st == s])) for s in STATES if (st == s).sum() >= 5), key=lambda x: x[1])
        gated = wr[st != worst[0]]
        f, b, e = sub(gated)
        print(f"  gate={gate:9s} worst state={worst[0]:22s}(Shp {worst[1]:+.2f}) "
              f"-> sit-out: full {f:.2f} bull {b:.2f} bear {e:.2f} ({len(gated)}/{len(wr)})")
    # breadth tercile gate
    br = breadth.reindex(wr.index); lo, hi = br.quantile(1/3), br.quantile(2/3)
    for blab, m in [("low breadth", br <= lo), ("mid breadth", (br > lo) & (br <= hi)), ("high breadth", br > hi)]:
        w = wr[m]
        print(f"    breadth {blab:12s} n={len(w):>3d}  Sharpe {shp(w):+.2f}  mean {w.mean()*100:+.2f}%/2w")