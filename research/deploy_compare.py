"""Step 7 (decider): compare the ACTUAL deployed system — book + 4h monitor
(regime flatten + exit-decay) — across cadence x lookback, so we choose between
"biweekly/3w + monitor" and "weekly/1w" on what we'd really run, not the bare book.

Answers: does the monitor's intra-cycle responsiveness make a faster rebalance
redundant? And is weekly/1w's extra turnover worth the extra return once the
monitor is on top and costs are realistic?

Path-aware (4h marks within each cycle). Same sizing/gate/cost/funding as the
prior backtests. EXIT_PCT=0.45 (the deployed hysteresis). Sharpe annualized per
cadence. Turnover reported as rebalances/yr + monitor-closes/yr; Sharpe shown at
8.5bp (base) and 25bp (conservative real taker+slippage).
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd, ccxt
from data_fetcher import load_cached_data
from regime import compute_regime_composite

START = "2023-01-01"; TF = "4h"; BARS_DAY = 6; WEEK = 7 * BARS_DAY
QFRAC = 0.20; TARGET_VOL = 0.15; LEV_CAP = (0.25, 3.0); VT_WARMUP = 8
TOPLIQ = 25; SPLIT = pd.Timestamp("2025-01-18"); FUND_BASE = 0.20; REGIME_PERIOD = 90
DOWN = "trending_down_choppy"; EXIT_PCT = 0.45

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
reg = compute_regime_composite(data["BTC"], period=REGIME_PERIOD)["regime"].reindex(master).values
YEARS = (master[-1] - master[0]).days / 365.0


def build_book(b, LB):
    mom = Pv[b] / Pv[b - LB] - 1
    v = np.nanstd(Rv[b - LB:b, :], axis=0)
    elig = np.where(~np.isnan(mom) & ~np.isnan(v) & (v > 0))[0]
    if elig.size < 8:
        return None
    order = elig[np.argsort(mom[elig])]; k = max(1, int(elig.size * QFRAC))
    L, W = order[:k], order[-k:]
    wl = 1.0 / v[W]; wl /= wl.sum(); ws = 1.0 / v[L]; ws /= ws.sum()
    return dict(W=W, L=L, wl=wl, ws=ws, k=k)


def simulate(LB, STEP, monitor, cost):
    step_days = STEP / BARS_DAY; pp = 365.0 / step_days
    rebals = list(range(LB, N - STEP, STEP))
    # vol-target ladder from raw book returns (gate-independent)
    lev_at = {}; hist = []
    for b in rebals:
        bk = build_book(b, LB)
        raw = 0.0
        if bk:
            seg = Pv[b:b + STEP + 1, :]; fwd = seg[-1] / seg[0] - 1
            raw = float(np.sum(bk["wl"] * fwd[bk["W"]]) - np.sum(bk["ws"] * fwd[bk["L"]]))
        lev = 1.0
        if len(hist) >= VT_WARMUP:
            rv = np.std(hist[-VT_WARMUP:], ddof=1)
            if rv > 0:
                lev = float(np.clip((TARGET_VOL / np.sqrt(pp)) / rv, *LEV_CAP))
        lev_at[b] = lev; hist.append(lev * raw)

    ret, idx = [], []; pl, ps = set(), set(); ncloses = 0
    for b in rebals:
        bk = build_book(b, LB)
        if bk is None:
            ret.append(0.0); idx.append(master[b]); continue
        lev, k = lev_at[b], bk["k"]
        Wn = [liq[i] for i in bk["W"]]; Ln = [liq[i] for i in bk["L"]]
        if reg[b] == DOWN:                              # weak-bear gate at entry
            t = (len(pl) + len(ps)) / (2 * k * 2) if (pl or ps) else 0.0
            ret.append(-t * cost); idx.append(master[b]); pl, ps = set(), set(); continue
        entry_turn = (len(set(Wn) ^ pl) + len(set(Ln) ^ ps)) / (2 * k * 2)
        Pw = Pv[b:b + STEP + 1, bk["W"]]; Pll = Pv[b:b + STEP + 1, bk["L"]]
        retw = Pw / Pw[0] - 1.0; retl = Pll / Pll[0] - 1.0
        wl, ws = bk["wl"], bk["ws"]

        if not monitor:
            pnl = lev * (np.sum(wl * retw[-1]) - np.sum(ws * retl[-1]))
            hold, cclose, eL, eS = 1.0, 0.0, set(Wn), set(Ln)
        else:                                           # combo: regime flatten + exit-decay
            held_w = np.ones(len(bk["W"]), bool); held_s = np.ones(len(bk["L"]), bool)
            locked = 0.0; cc = 0.0; hsum = 0.0; npos = len(bk["W"]) + len(bk["L"])
            for t in range(1, STEP + 1):
                if reg[b + t] == DOWN:
                    for li in np.where(held_w)[0]:
                        locked += lev * wl[li] * retw[t, li]; hsum += t / STEP
                    for li in np.where(held_s)[0]:
                        locked += -lev * ws[li] * retl[t, li]; hsum += t / STEP
                    ncloses += int(held_w.sum() + held_s.sum()); cc += cost
                    held_w[:] = False; held_s[:] = False; break
                momt = Pv[b + t] / Pv[b + t - LB] - 1.0
                vals = momt[~np.isnan(momt)]
                lo = np.quantile(vals, EXIT_PCT); hi = np.quantile(vals, 1 - EXIT_PCT)
                for li, gi in enumerate(bk["W"]):
                    if held_w[li] and not np.isnan(momt[gi]) and momt[gi] < lo:
                        locked += lev * wl[li] * retw[t, li]; held_w[li] = False
                        cc += cost / (2 * k); hsum += t / STEP; ncloses += 1
                for li, gi in enumerate(bk["L"]):
                    if held_s[li] and not np.isnan(momt[gi]) and momt[gi] > hi:
                        locked += -lev * ws[li] * retl[t, li]; held_s[li] = False
                        cc += cost / (2 * k); hsum += t / STEP; ncloses += 1
            live = lev * (np.sum(wl[held_w] * retw[-1, held_w]) - np.sum(ws[held_s] * retl[-1, held_s]))
            pnl = locked + live
            hsum += held_w.sum() + held_s.sum(); hold = hsum / max(npos, 1); cclose = cc
            eL = {liq[gi] for li, gi in enumerate(bk["W"]) if held_w[li]}
            eS = {liq[gi] for li, gi in enumerate(bk["L"]) if held_s[li]}
        fund = FUND_BASE * (step_days / 365) * lev * hold
        r = pnl - entry_turn * cost - cclose - fund
        ret.append(r); idx.append(master[b]); pl, ps = eL, eS
    return pd.Series(ret, index=pd.DatetimeIndex(idx)), pp, 365.0 / step_days, ncloses / YEARS


def stats(w, pp):
    if len(w) < 3 or w.std() == 0:
        return 0, 0, 0
    eq = (1 + w).cumprod()
    return (eq.iloc[-1] - 1) * 100, (w.mean() / w.std()) * np.sqrt(pp), (eq / eq.cummax() - 1).min() * 100


def row(name, LB, STEP, monitor):
    w, pp, rebyr, clyr = simulate(LB, STEP, monitor, 0.00085)
    t, s, m = stats(w, pp)
    w2, _, _, _ = simulate(LB, STEP, monitor, 0.0025)
    s25 = stats(w2, pp)[1]
    bull = stats(w[w.index < SPLIT], pp)[1]; bear = stats(w[w.index >= SPLIT], pp)[1]
    print(f"  {name:22s} {t:>7.0f} {s:>6.2f} {s25:>7.2f} {bull:>6.2f} {bear:>6.2f} {m:>6.0f} "
          f"{rebyr:>6.0f} {clyr:>8.0f}")
    return s, s25, t


print(f"Top-{TOPLIQ} liquid; book + 4h monitor (regime+exit-decay, pct={EXIT_PCT}); "
      f"{START}..{str(master[-1].date())}; {YEARS:.1f}y\n")
hdr = f"  {'config':22s} {'tot%':>7s} {'Shp':>6s} {'Shp25bp':>7s} {'bull':>6s} {'bear':>6s} {'mDD':>6s} {'reb/y':>6s} {'mClos/y':>8s}"

print("=== A) bare book (NO monitor) ===")
print(hdr)
row("biweekly/4w  (current)", 4 * WEEK, 2 * WEEK, False)
row("biweekly/3w", 3 * WEEK, 2 * WEEK, False)
row("weekly/1w", WEEK, WEEK, False)
row("weekly/2w", 2 * WEEK, WEEK, False)

print("\n=== B) book + MONITOR (the deployed system) ===")
print(hdr)
configs = [("biweekly/4w +mon", 4 * WEEK, 2 * WEEK), ("biweekly/3w +mon", 3 * WEEK, 2 * WEEK),
           ("weekly/2w +mon", 2 * WEEK, WEEK), ("weekly/1w +mon", WEEK, WEEK)]
B = {name: row(name, lb, st, True) for name, lb, st in configs}

print("\n=== C) fine SHORT-HORIZON sweep, with monitor (lookback in days; cadence) ===")
print(hdr)
D3, D5, D10 = 3 * BARS_DAY, 5 * BARS_DAY, 10 * BARS_DAY; HALFWK = WEEK // 2
row("weekly/3d +mon", D3, WEEK, True)
row("weekly/5d +mon", D5, WEEK, True)
row("twice-wk/1w +mon", WEEK, HALFWK, True)
row("twice-wk/5d +mon", D5, HALFWK, True)
row("weekly/10d +mon", D10, WEEK, True)

print("\n  Decision lens: weekly/1w must beat biweekly/3w+mon by enough to justify "
      "~2x rebalances/yr AND the extra monitor closes — at the conservative 25bp Sharpe.")
