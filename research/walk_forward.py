"""Step 8: walk-forward / out-of-sample stress on the weekly/1w deploy choice.

weekly/1w was the best cell in an in-sample grid -> selection-bias risk. Two
honest tests:

  TEST 1  Sub-period stability — split the history into 4 sequential periods and
          show each config's Sharpe per period. Is weekly/1w consistently strong,
          or lucky in one stretch?

  TEST 2  Walk-forward SELECTION — expanding train, 3mo test. At each fold pick
          the config with the best TRAILING Sharpe (no hindsight), then measure
          that pick OOS. Compare the realistically-selected ("WF-adaptive")
          strategy vs always-weekly/1w vs always-biweekly/4w vs the per-fold
          oracle. If WF-adaptive ~ always-weekly/1w and mostly PICKS it, the
          edge is robustly selectable; if WF flip-flops and lags, it was luck.

Bare book (monitor is unscheduled in deploy), same sizing/gate/cost/funding.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd, ccxt
from data_fetcher import load_cached_data
from regime import compute_regime_composite

START = "2023-01-01"; TF = "4h"; BARS_DAY = 6; WEEK = 7 * BARS_DAY
QFRAC = 0.20; COST = 0.00085; TARGET_VOL = 0.15; LEV_CAP = (0.25, 3.0); VT_WARMUP = 8
TOPLIQ = 25; FUND_BASE = 0.20; REGIME_PERIOD = 90; DOWN = "trending_down_choppy"

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


def simulate(LB, STEP):
    step_days = STEP / BARS_DAY; pp = 365.0 / step_days
    rebals = list(range(LB, N - STEP, STEP))
    ret, idx, hist = [], [], []
    pl, ps = set(), set()
    for b in rebals:
        mom = Pv[b] / Pv[b - LB] - 1
        seg = Pv[b:b + STEP + 1, :]; fwd = seg[-1] / seg[0] - 1
        v = np.nanstd(Rv[b - LB:b, :], axis=0)
        elig = np.where(~np.isnan(mom) & ~np.isnan(fwd) & ~np.isnan(v) & (v > 0))[0]
        if elig.size < 8:
            ret.append(0.0); idx.append(master[b]); hist.append(0.0); continue
        order = elig[np.argsort(mom[elig])]; k = max(1, int(elig.size * QFRAC))
        losers, winners = order[:k], order[-k:]
        def leg(ix):
            w = 1.0 / v[ix]; w /= w.sum(); return float(np.sum(w * fwd[ix]))
        raw = leg(winners) - leg(losers)
        lev = 1.0
        if len(hist) >= VT_WARMUP:
            rv = np.std(hist[-VT_WARMUP:], ddof=1)
            if rv > 0:
                lev = float(np.clip((TARGET_VOL / np.sqrt(pp)) / rv, *LEV_CAP))
        hist.append(lev * raw)
        if reg[b] == DOWN:
            ret.append(0.0); idx.append(master[b]); pl, ps = set(), set(); continue
        nl, ns = set(winners.tolist()), set(losers.tolist())
        turn = (len(nl ^ pl) + len(ns ^ ps)) / (2 * k * 2)
        ret.append(lev * raw - turn * 2 * COST - FUND_BASE * (step_days / 365) * lev)
        idx.append(master[b]); pl, ps = nl, ns
    return pd.Series(ret, index=pd.DatetimeIndex(idx)), pp


def sharpe(w, pp):
    return (w.mean() / w.std()) * np.sqrt(pp) if len(w) >= 3 and w.std() > 0 else 0.0


def total(w):
    return ((1 + w).prod() - 1) * 100 if len(w) else 0.0


CONFIGS = {f"{cn}/{ln}": (lb, st)
           for cn, st in [("wk", WEEK), ("bi", 2 * WEEK)]
           for ln, lb in [("1w", WEEK), ("2w", 2 * WEEK), ("3w", 3 * WEEK),
                          ("4w", 4 * WEEK), ("6w", 6 * WEEK)]}
SERIES = {name: simulate(lb, st) for name, (lb, st) in CONFIGS.items()}

# ── TEST 1: sub-period stability ───────────────────────────────────────────
print(f"Top-{TOPLIQ} liquid, bare book; {START}..{str(master[-1].date())}\n")
edges = pd.date_range(master[0], master[-1], periods=5)
print("=== TEST 1: Sharpe per sequential quarter-ish period (stability, not luck) ===")
hdr = "  " + f"{'config':10s}" + "".join(f"  P{i+1}:{str(edges[i].date())[2:]}" for i in range(4))
print(hdr)
focus = ["wk/1w", "bi/4w", "bi/3w", "wk/2w", "bi/6w"]
for name in focus:
    w, pp = SERIES[name]; cells = []
    for i in range(4):
        a, b = edges[i], edges[i + 1]
        cells.append(f"{sharpe(w[(w.index >= a) & (w.index < b)], pp):>11.2f}")
    print(f"  {name:10s}" + "".join(cells))

# ── TEST 2: walk-forward selection ─────────────────────────────────────────
print("\n=== TEST 2: walk-forward selection (expanding train, 3mo OOS test) ===")
test_edges = pd.date_range(master[0] + pd.DateOffset(months=12), master[-1], freq="3MS")
picks = []; wf_folds = []; w1_folds = []; b4_folds = []; oracle_folds = []
for j in range(len(test_edges) - 1):
    a, b = test_edges[j], test_edges[j + 1]
    train_sh = {n: sharpe(SERIES[n][0][SERIES[n][0].index < a], SERIES[n][1]) for n in CONFIGS}
    pick = max(train_sh, key=train_sh.get)
    picks.append(pick)
    def oos(n):
        w, pp = SERIES[n]; seg = w[(w.index >= a) & (w.index < b)]
        return sharpe(seg, pp), total(seg)
    wf_folds.append(oos(pick)); w1_folds.append(oos("wk/1w")); b4_folds.append(oos("bi/4w"))
    # per-fold oracle = best OOS Sharpe in this test window (hindsight ceiling)
    oracle_folds.append(max(oos(n)[0] for n in CONFIGS))
    print(f"  {str(a.date())}..{str(b.date())}  picked {pick:7s}  "
          f"OOS Sharpe: pick {oos(pick)[0]:>5.2f} | wk/1w {oos('wk/1w')[0]:>5.2f} | "
          f"bi/4w {oos('bi/4w')[0]:>5.2f} | oracle {oracle_folds[-1]:>5.2f}")


def agg(folds):
    sh = [f[0] for f in folds]; tot = [f[1] for f in folds]
    return np.mean(sh), sum(1 for s in sh if s > 0), len(sh), float(np.prod([1 + t / 100 for t in tot]) - 1) * 100


print("\n=== walk-forward summary (mean OOS fold Sharpe | folds positive | compounded OOS total%) ===")
for label, folds in [("WF-adaptive (realistic)", wf_folds), ("always wk/1w", w1_folds),
                     ("always bi/4w (old)", b4_folds)]:
    m, pos, n, tot = agg(folds)
    print(f"  {label:26s} meanSharpe {m:>5.2f} | {pos}/{n} folds + | OOS total {tot:>6.0f}%")
print(f"  {'per-fold oracle (ceiling)':26s} meanSharpe {np.mean(oracle_folds):>5.2f}")
from collections import Counter
print(f"\n  WF pick frequency: {dict(Counter(picks))}")
print(f"  -> wk/1w chosen {picks.count('wk/1w')}/{len(picks)} folds")
