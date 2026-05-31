"""Steps 1+2: marry our cross-sectional momentum with the repo's composite
7-state regime classifier (shared_tools/regime.py).

  1. Label market (BTC) + per-coin bars with compute_regime_composite (the repo's
     live classifier). Sanity-check label distribution across periods.
  2a. MARKET-level: bucket the deployable XS-momentum book's forward returns by
      the market's composite state -> which states to trade vs sit out.
  2b. PER-ASSET: condition the long leg on up-trend coins / short leg on down-trend
      coins (each coin's own regime) -> does it beat the raw ranking?

Deployable config: top-25 liquid, L/S, inverse-vol, vol-target, biweekly. 4w & 8w.
Causal: composite label at bar b uses data <= b, gates forward b->b+step.
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
UP = {"trending_up_clean", "trending_up_choppy"}
DOWN = {"trending_down_clean", "trending_down_choppy"}

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
print(f"Deployable universe: top-{TOPLIQ} liquid; market regime = BTC composite\n")

# ---------- STEP 1: label distributions ----------
print("=== STEP 1: BTC composite label distribution (sanity) ===")
btc = data["BTC"]
for per in [42, 90, 180]:
    lab = compute_regime_composite(btc, period=per)["regime"].reindex(master)
    dist = lab.value_counts(normalize=True)
    short = lambda s: s.replace("trending_", "t").replace("ranging_", "r").replace("_", "")
    print(f"  period {per:>3d}: " + "  ".join(f"{short(s)}:{dist.get(s,0)*100:>3.0f}%" for s in STATES))
btc_lab = compute_regime_composite(btc, period=REGIME_PERIOD)["regime"].reindex(master)
print(f"\n  (using period={REGIME_PERIOD} for the measurement below)\n")

# per-coin labels (deployable universe) for 2b
coin_lab = {}
for c in liq:
    coin_lab[c] = compute_regime_composite(data[c], period=REGIME_PERIOD)["regime"].reindex(master)
LAB = pd.DataFrame(coin_lab)[liq]
Pl = P[liq]; Pv = Pl.values; Rv = Pl.pct_change().values; N = len(master)
LABv = LAB.values


def run(lb, conditioned=False):
    rebals = list(range(lb, N - STEP, STEP))
    eq = 1.0; wr = []; idx = []; pl = set(); ps = set(); hist = []
    for b in rebals:
        mom = Pv[b] / Pv[b - lb] - 1
        seg = Pv[b:b + STEP + 1, :]; fwd = seg[-1] / seg[0] - 1 if seg.shape[0] >= 2 else np.full(len(liq), np.nan)
        v = np.nanstd(Rv[b - lb:b, :], axis=0)
        ok = ~np.isnan(mom) & ~np.isnan(fwd) & ~np.isnan(v) & (v > 0)
        elig = np.where(ok)[0]
        if elig.size < 8:
            wr.append(0.0); idx.append(master[b]); hist.append(0.0); continue
        order = elig[np.argsort(mom[elig])]
        k = max(1, int(elig.size * QFRAC))
        losers, winners = order[:k], order[-k:]
        if conditioned:
            lab_b = LABv[b]
            winners = np.array([i for i in winners if lab_b[i] in UP], int)
            losers = np.array([i for i in losers if lab_b[i] in DOWN], int)
            if winners.size == 0 and losers.size == 0:
                wr.append(0.0); idx.append(master[b]); hist.append(0.0); continue
        def leg(ix):
            if ix.size == 0:
                return 0.0
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


def shp(wr):
    return (wr.mean() / wr.std()) * np.sqrt(WEEKS_YR) if len(wr) > 2 and wr.std() > 0 else 0.0


# ---------- STEP 2a: market-state conditional ----------
print("=== STEP 2a: XS-momentum forward returns bucketed by MARKET (BTC) regime ===")
for lb, name in [(4 * WEEK, "4w"), (8 * WEEK, "8w")]:
    wr = run(lb)
    states_at = btc_lab.reindex(wr.index)
    print(f"\n  --- {name} L/S book; overall Sharpe {shp(wr):.2f} ({len(wr)} rebalances) ---")
    print(f"    {'state':22s} {'n':>4s} {'mean%/2w':>9s} {'Sharpe':>7s} {'hit%':>6s}")
    rows = []
    for s in STATES:
        m = states_at == s; w = wr[m]
        if len(w) == 0:
            continue
        rows.append((s, len(w), w.mean() * 100, shp(w), (w > 0).mean() * 100))
    for s, n, mn, sh, hit in sorted(rows, key=lambda x: -x[3]):
        print(f"    {s:22s} {n:>4d} {mn:>9.2f} {sh:>7.2f} {hit:>6.0f}")
    # illustration: sit out ranging_quiet (flagged in-sample)
    gated = wr[states_at != "ranging_quiet"]
    print(f"    -> sit-out ranging_quiet: Sharpe {shp(gated):.2f} ({len(gated)} traded, {len(wr)-len(gated)} skipped) [in-sample illustration]")

# ---------- STEP 2b: per-asset regime conditioning of legs ----------
print("\n=== STEP 2b: condition legs on each coin's OWN regime (long up-trend / short down-trend) ===")
print(f"  {'config':16s} {'full Shp':>9s} {'bull':>7s} {'bear':>7s}")
for lb, name in [(4 * WEEK, "4w"), (8 * WEEK, "8w")]:
    for cond, tag in [(False, "raw rank"), (True, "regime-cond")]:
        wr = run(lb, conditioned=cond)
        b = shp(wr[wr.index < SPLIT]); e = shp(wr[wr.index >= SPLIT])
        print(f"  {name+' '+tag:16s} {shp(wr):>9.2f} {b:>7.2f} {e:>7.2f}")
