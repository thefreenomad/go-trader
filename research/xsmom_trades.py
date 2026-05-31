"""Deep trade-data analysis of the validated HL config — find where the edge
lives and where there's more to squeeze.

Logs every position (and every eligible coin) per rebalance, then dissects:
  A. Long leg vs short leg — does the short earn its risk?
  B. Decile monotonicity — forward return by momentum rank (where is the signal?)
  C. Coin concentration — is P&L broad or a few names? (robustness)
  D. Time concentration — a few great periods or consistent?
  E. Holding-period decay — does the long-short spread persist past 2 weeks?
Config: 4w / inverse-vol / vol-target / biweekly / weak-bear gate, top-25 HL-liquid.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd, ccxt
from data_fetcher import load_cached_data
from regime import compute_regime_composite

START = "2023-01-01"; TF = "4h"; BARS_DAY = 6
WEEK = 7 * BARS_DAY; STEP = 2 * WEEK; LB = 4 * WEEK
QFRAC = 0.20; NLIQ = 25; WEEKS_YR = 365 / 7

hl = ccxt.hyperliquid({"enableRateLimit": True}); hl.load_markets(); htk = hl.fetch_tickers()
def hl_vol(c):
    return (htk.get(f"{c}/USDC:USDC") or {}).get("quoteVolume")

syms = json.load(open("research/universe.json"))
data = {}; master = None
for sym in syms:
    c = sym.split("/")[0]
    if hl_vol(c) is None:
        continue
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
btc_lab = compute_regime_composite(data["BTC"], period=90)["regime"].reindex(master)
rebals = list(range(LB, N - STEP, STEP))

elig_rows = []   # every eligible coin: mom_pctile, fwd2w
pos_rows = []    # every selected position
for b in rebals:
    if btc_lab.iloc[b] == "trending_down_choppy":
        continue
    mom = Pv[b] / Pv[b - LB] - 1
    fwd2 = (Pv[min(b + STEP, N - 1)] / Pv[b]) - 1
    fwd1 = (Pv[min(b + WEEK, N - 1)] / Pv[b]) - 1
    fwd3 = (Pv[min(b + 3 * WEEK, N - 1)] / Pv[b]) - 1
    fwd4 = (Pv[min(b + 4 * WEEK, N - 1)] / Pv[b]) - 1
    v = np.nanstd(Rv[b - LB:b, :], axis=0)
    elig = np.where(~np.isnan(mom) & ~np.isnan(fwd2) & ~np.isnan(v) & (v > 0))[0]
    if elig.size < 8:
        continue
    ranks = pd.Series(mom[elig]).rank(pct=True).values
    for j, i in enumerate(elig):
        elig_rows.append(dict(date=master[b], coin=liq[i], pctile=ranks[j], fwd2=fwd2[i]))
    order = elig[np.argsort(mom[elig])]; k = max(1, int(elig.size * QFRAC))
    losers, winners = order[:k], order[-k:]
    for ix, side in [(winners, +1), (losers, -1)]:
        iv = 1.0 / v[ix]; iv = iv / iv.sum()
        for jj, i in enumerate(ix):
            pos_rows.append(dict(date=master[b], coin=liq[i], side=side, w=iv[jj],
                                 fwd1=fwd1[i], fwd2=fwd2[i], fwd3=fwd3[i], fwd4=fwd4[i],
                                 pnl=side * iv[jj] * fwd2[i]))

E = pd.DataFrame(elig_rows); T = pd.DataFrame(pos_rows)
print(f"{len(rebals)} rebalances; {len(T)} positions; {T.coin.nunique()} coins traded\n")

print("=== A. LONG vs SHORT leg ===")
for side, name in [(1, "LONG (winners)"), (-1, "SHORT (losers)")]:
    s = T[T.side == side]; r = s.side * s.fwd2          # directional return per position
    print(f"  {name:16s} positions {len(s):>4d}  mean ret/2w {r.mean()*100:+.2f}%  "
          f"win% {(r>0).mean()*100:.0f}  total pnl(contrib) {s.pnl.sum():+.2f}")

print("\n=== B. DECILE monotonicity: mean 2w forward return by momentum decile ===")
E["dec"] = (E.pctile * 10).clip(0, 9.999).astype(int)
dec = E.groupby("dec")["fwd2"].agg(["mean", "count"])
for d, row in dec.iterrows():
    bar = "#" * int(abs(row["mean"]) * 500)
    print(f"  decile {d} (mom {d*10:>2d}-{d*10+10}%)  mean fwd2w {row['mean']*100:+6.2f}%  {bar}")
print(f"  top-decile minus bottom-decile spread: {(dec.loc[9,'mean']-dec.loc[0,'mean'])*100:+.2f}%/2w")

print("\n=== C. COIN concentration (P&L contribution share) ===")
cc = T.groupby("coin")["pnl"].sum().sort_values(ascending=False)
tot = cc.sum()
print("  top contributors: " + ", ".join(f"{c} {v/tot*100:+.0f}%" for c, v in cc.head(6).items()))
print("  worst:            " + ", ".join(f"{c} {v/tot*100:+.0f}%" for c, v in cc.tail(4).items()))
print(f"  top-5 coins = {cc.head(5).sum()/tot*100:.0f}% of P&L  ({T.coin.nunique()} coins total)")

print("\n=== D. TIME concentration: P&L by 6-month block ===")
T["blk"] = pd.PeriodIndex(T.date, freq="Q")
half = T.groupby(pd.Grouper(key="date", freq="2QS"))["pnl"].sum()
for ts, v in half.items():
    print(f"  {str(ts.date()):>11s}  pnl {v:+.2f}  {'#'*int(abs(v)*20)}")

print("\n=== E. HOLDING-PERIOD decay: long-short spread at 1/2/3/4 weeks ===")
for h, col in [("1w", "fwd1"), ("2w", "fwd2"), ("3w", "fwd3"), ("4w", "fwd4")]:
    sprd = (T[T.side == 1][col].mean() - T[T.side == -1][col].mean()) * 100
    print(f"  {h}: long-short spread {sprd:+.2f}%")
