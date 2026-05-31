"""Does a DEPLOYABLE (liquid) cross-sectional edge exist in the BEAR?

Decides the regime architecture: if yes -> regime-SWITCHED (bull-config | bear-config);
if no -> regime-GATED (bull-config | cash).

Economically-motivated bear hypotheses only (one bear period -> avoid blind search):
  shortonly  — short the biggest losers (they keep falling in a bear)
  shorttilt  — 0.5L / 1.0S (short-heavy)
  ls         — balanced long/short (reference)
  longtilt   — 1.0L / 0.5S (the deployable bull config)
  longonly   — long winners only
  reversal   — long losers / short winners (does the bear mean-revert?)

Deployable universe (top-25 liquid). Locked mechanics (8w, inverse-vol,
vol-target, biweekly). Reports full / bull / bear Sharpe for each.
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd, ccxt
from data_fetcher import load_cached_data

START = "2023-01-01"; TF = "4h"; BARS_DAY = 6
WEEK = 7 * BARS_DAY; STEP = 2 * WEEK
QFRAC = 0.20; COST = 0.00085
TARGET_VOL = 0.15; LEV_CAP = (0.25, 3.0); VT_WARMUP = 8; WEEKS_YR = 365 / 7
SPLIT = pd.Timestamp("2025-01-18"); TOPLIQ = 25
MODES = {  # (long_weight, short_weight, reversal?)
    "shortonly": (0.0, 1.0, False), "shorttilt": (0.5, 1.0, False),
    "ls": (1.0, 1.0, False), "longtilt": (1.0, 0.5, False),
    "longonly": (1.0, 0.0, False), "reversal": (1.0, 1.0, True),
}

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
ex = ccxt.binanceus({"enableRateLimit": True}); tk = ex.fetch_tickers()
vol = {c: (tk.get(f"{c}/USDT") or {}).get("quoteVolume", 0) or 0 for c in P.columns}
cols = sorted(P.columns, key=lambda c: -vol[c])[:TOPLIQ]
P = P[cols]; Pv = P.values; Rv = P.pct_change().values; N = len(master)
print(f"Deployable universe: top-{TOPLIQ} liquid {cols[:10]}...\n")


def run(lb, mode):
    lw, sw, rev = MODES[mode]
    rebals = list(range(lb, N - STEP, STEP))
    eq = 1.0; wr = []; idx = []; pl = set(); psr = set(); hist = []
    for b in rebals:
        mom = Pv[b] / Pv[b - lb] - 1
        seg = Pv[b:b + STEP + 1, :]; fwd = seg[-1] / seg[0] - 1 if seg.shape[0] >= 2 else np.full(len(cols), np.nan)
        v = np.nanstd(Rv[b - lb:b, :], axis=0)
        elig = np.where(~np.isnan(mom) & ~np.isnan(fwd) & ~np.isnan(v) & (v > 0))[0]
        if elig.size < 8:
            wr.append(0.0); idx.append(master[b]); hist.append(0.0); continue
        k = max(1, int(elig.size * QFRAC)); order = elig[np.argsort(mom[elig])]
        losers, winners = order[:k], order[-k:]
        lo, sh = (winners, losers) if not rev else (losers, winners)
        def leg(ix):
            w = 1.0 / v[ix]; w /= w.sum(); return float(np.sum(w * fwd[ix]))
        raw = lw * leg(lo) - sw * leg(sh)
        lev = 1.0
        if len(hist) >= VT_WARMUP:
            rv = np.std(hist[-VT_WARMUP:], ddof=1)
            if rv > 0:
                lev = float(np.clip((TARGET_VOL / np.sqrt(WEEKS_YR)) / rv, *LEV_CAP))
        nl, ns = set(lo.tolist()), set(sh.tolist())
        turn = ((len(nl ^ pl) if lw else 0) + (len(ns ^ psr) if sw else 0)) / (2 * k * (1 if (lw == 0 or sw == 0) else 2))
        r = lev * raw - turn * 2 * COST
        pl, psr = nl, ns
        eq *= (1 + r); wr.append(r); idx.append(master[b]); hist.append(lev * raw)
    return pd.Series(wr, index=pd.DatetimeIndex(idx))


def stats(wr):
    if len(wr) < 3 or wr.std() == 0:
        return 0.0, 0.0
    eq = (1 + wr).cumprod()
    return (eq.iloc[-1] - 1) * 100, (wr.mean() / wr.std()) * np.sqrt(WEEKS_YR)


print("=== DEPLOYABLE CROSS-SECTIONAL by MODE x LOOKBACK: bull Sharpe / BEAR Sharpe ===")
print(f"  {'mode':10s} | " + " | ".join(f"{w:>13s}" for w in ["4w", "8w", "12w"]))
for mode in MODES:
    cells = []
    for lb in [180, 360, 540]:
        wr = run(lb, mode)
        _, sb = stats(wr[wr.index < SPLIT]); _, se = stats(wr[wr.index >= SPLIT])
        cells.append(f"{sb:>5.2f} / {se:>5.2f}")
    print(f"  {mode:10s} | " + " | ".join(f"{c:>13s}" for c in cells))

print("\n  (each cell = bull Sharpe / BEAR Sharpe; deployable top-25 liquid, net of trading cost)")
print("\n=== Verdict aid: best BEAR Sharpe per mode (any lookback) ===")
for mode in MODES:
    bears = []
    for lb in [180, 360, 540]:
        wr = run(lb, mode)
        bears.append(stats(wr[wr.index >= SPLIT])[1])
    print(f"  {mode:10s} best bear Sharpe {max(bears):+.2f}")
