"""Step 5: pick the intra-cycle MONITOR by data.

The book enters biweekly (validated cross-sectional momentum: 4w lookback,
inverse-vol, vol-target 15%, top/bottom 20%, weak-bear regime gate at entry).
Between rebalances it currently sits unmanaged for two weeks. This backtest is
PATH-AWARE at the 4h candle: it marks the book on every bar and tests adding a
4h "monitor pass" that can only de-risk (close/trim), never open. Variants:

  base        biweekly only (weak-bear gate at entry, no intra-cycle action)
  regime4h    flatten the whole book the bar BTC regime -> trending_down_choppy
  ddstop_X    flatten if book equity draws down > X% from its running peak
  rehedge_X   trim the heavier leg back toward neutral when |net|/gross > X
  exitdecay   close a name whose 4w-momentum rank breaks (out of its leg's half)

Same universe / sizing / cost / funding as xsmom_validate.py, so 'base' here
equals the validated strategy. Cost = COST one-way per full-book turn; a
mid-cycle close costs its name's share now and re-entry next rebalance (round
trip). Funding prorated by holding time. Sharpe on biweekly returns, annualized
like the validation, plus a bull/bear split at SPLIT.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd, ccxt
from data_fetcher import load_cached_data
from regime import compute_regime_composite

START = "2023-01-01"; TF = "4h"; BARS_DAY = 6
WEEK = 7 * BARS_DAY; STEP = 2 * WEEK; LB = 4 * WEEK
QFRAC = 0.20; COST = 0.00085; STEP_DAYS = 14
TARGET_VOL = 0.15; LEV_CAP = (0.25, 3.0); VT_WARMUP = 8; WEEKS_YR = 365 / 7
TOPLIQ = 25; SPLIT = pd.Timestamp("2025-01-18")
FUND_BASE = 0.20; REGIME_PERIOD = 90
DOWN = "trending_down_choppy"

# ---- data (identical to the validation harness) ----
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

rebals = list(range(LB, N - STEP, STEP))


def build_book(b):
    """Winner/loser column indices + inverse-vol leg weights + endpoint raw return."""
    mom = Pv[b] / Pv[b - LB] - 1
    v = np.nanstd(Rv[b - LB:b, :], axis=0)
    seg = Pv[b:b + STEP + 1, :]; fwd = seg[-1] / seg[0] - 1
    elig = np.where(~np.isnan(mom) & ~np.isnan(fwd) & ~np.isnan(v) & (v > 0))[0]
    if elig.size < 8:
        return None
    order = elig[np.argsort(mom[elig])]; k = max(1, int(elig.size * QFRAC))
    L, W = order[:k], order[-k:]
    wl = (1.0 / v[W]); wl /= wl.sum()
    ws = (1.0 / v[L]); ws /= ws.sum()
    raw = float(np.sum(wl * fwd[W]) - np.sum(ws * fwd[L]))
    return dict(W=W, L=L, wl=wl, ws=ws, k=k, raw=raw)


# vol-target leverage ladder (gate-independent, from raw book returns) — precomputed once
books = {b: build_book(b) for b in rebals}
lev_at = {}; _hist = []
for b in rebals:
    bk = books[b]
    lev = 1.0
    if bk is not None and len(_hist) >= VT_WARMUP:
        rv = np.std(_hist[-VT_WARMUP:], ddof=1)
        if rv > 0:
            lev = float(np.clip((TARGET_VOL / np.sqrt(WEEKS_YR)) / rv, *LEV_CAP))
    lev_at[b] = lev
    _hist.append(lev * (bk["raw"] if bk else 0.0))


def cycle_paths(b, bk):
    """Per-bar cumulative returns of each held name over the cycle [0..STEP]."""
    W, L = bk["W"], bk["L"]
    Pw = Pv[b:b + STEP + 1, W]; Pll = Pv[b:b + STEP + 1, L]
    retw = Pw / Pw[0] - 1.0          # [STEP+1, nW]
    retl = Pll / Pll[0] - 1.0        # [STEP+1, nL]
    return retw, retl


DIAG = []   # (closes, npos, hold_frac) per cycle for the last run, for turnover reporting


def run(variant, **kw):
    """Return a biweekly return series for the named monitor variant."""
    ret, idx = [], []
    pl, ps = set(), set()             # names held into this cycle (for entry turnover)
    eq = 1.0; peak = 1.0
    DIAG.clear()
    rng = np.random.default_rng(kw.get("seed", 0))
    for b in rebals:
        bk = books[b]
        if bk is None:
            ret.append(0.0); idx.append(master[b]); continue
        lev, k = lev_at[b], bk["k"]
        Wn = [liq[i] for i in bk["W"]]; Ln = [liq[i] for i in bk["L"]]
        gate_flat = reg[b] == DOWN    # weak-bear gate at ENTRY (live design, all variants)

        # entry turnover vs prior held book; charge one-way COST on the traded fraction
        if gate_flat:
            entry_turn = (len(pl) + len(ps)) / (2 * k * 2) if (pl or ps) else 0.0
            ret.append(-entry_turn * COST); idx.append(master[b]); pl, ps = set(), set()
            eq *= (1 + ret[-1]); peak = max(peak, eq); continue
        entry_turn = (len(set(Wn) ^ pl) + len(set(Ln) ^ ps)) / (2 * k * 2)

        retw, retl = cycle_paths(b, bk)
        wl, ws = bk["wl"], bk["ws"]

        # --- per-variant intra-cycle simulation: returns (cycle_pnl, hold_frac, close_cost, end_long, end_short) ---
        if variant == "base":
            pnl = lev * (np.sum(wl * retw[-1]) - np.sum(ws * retl[-1]))
            hold, cclose, eL, eS = 1.0, 0.0, set(Wn), set(Ln)

        elif variant == "regime4h":
            j = next((t for t in range(1, STEP + 1) if reg[b + t] == DOWN), None)
            if j is None:
                pnl = lev * (np.sum(wl * retw[-1]) - np.sum(ws * retl[-1]))
                hold, cclose, eL, eS = 1.0, 0.0, set(Wn), set(Ln)
            else:
                pnl = lev * (np.sum(wl * retw[j]) - np.sum(ws * retl[j]))
                hold, cclose, eL, eS = j / STEP, COST, set(), set()   # flatten all

        elif variant == "ddstop":
            X = kw["X"]; jstop = None
            for t in range(1, STEP + 1):
                bpnl = lev * (np.sum(wl * retw[t]) - np.sum(ws * retl[t]))
                if (eq * (1 + bpnl)) / peak - 1 < -X:
                    jstop = t; break
            if jstop is None:
                pnl = lev * (np.sum(wl * retw[-1]) - np.sum(ws * retl[-1]))
                hold, cclose, eL, eS = 1.0, 0.0, set(Wn), set(Ln)
            else:
                pnl = lev * (np.sum(wl * retw[jstop]) - np.sum(ws * retl[jstop]))
                hold, cclose, eL, eS = jstop / STEP, COST, set(), set()

        elif variant == "rehedge":
            X = kw["X"]; reh = 0
            for t in range(1, STEP + 1):
                legL = float(np.sum(wl * retw[t])); legS = float(np.sum(ws * retl[t]))
                gross = 2 + legL + legS
                if gross > 0 and abs(legL - legS) / gross > X:
                    reh += 1
            pnl = lev * (np.sum(wl * retw[-1]) - np.sum(ws * retl[-1]))
            hold, cclose, eL, eS = 1.0, reh * (X) * COST, set(Wn), set(Ln)  # cost per rehedge ~ imbalance*COST

        elif variant in ("exitdecay", "combo"):
            regime_flat = variant == "combo"
            held_w = np.ones(len(bk["W"]), bool); held_s = np.ones(len(bk["L"]), bool)
            locked = 0.0; closed_cost = 0.0; hold_sum = 0.0; npos = len(bk["W"]) + len(bk["L"])
            ncloses = 0; flat_break = False
            for t in range(1, STEP + 1):
                if regime_flat and reg[b + t] == DOWN:   # macro risk-off: flatten everything left
                    for li in np.where(held_w)[0]:
                        locked += lev * wl[li] * retw[t, li]; hold_sum += t / STEP
                    for li in np.where(held_s)[0]:
                        locked += -lev * ws[li] * retl[t, li]; hold_sum += t / STEP
                    ncloses += held_w.sum() + held_s.sum(); closed_cost += COST
                    held_w[:] = False; held_s[:] = False; flat_break = True; break
                momt = Pv[b + t] / Pv[b + t - LB] - 1.0     # 4w momentum rank at this bar
                vals = momt[~np.isnan(momt)]
                pct = kw.get("pct", 0.50)                    # hysteresis: pct<0.5 -> dead band
                lo = np.quantile(vals, pct); hi = np.quantile(vals, 1 - pct)
                for li, gi in enumerate(bk["W"]):
                    if held_w[li] and not np.isnan(momt[gi]) and momt[gi] < lo:     # winner clearly broke
                        locked += lev * wl[li] * retw[t, li]; held_w[li] = False
                        closed_cost += COST / (2 * k); hold_sum += t / STEP; ncloses += 1
                for li, gi in enumerate(bk["L"]):
                    if held_s[li] and not np.isnan(momt[gi]) and momt[gi] > hi:     # loser clearly broke
                        locked += -lev * ws[li] * retl[t, li]; held_s[li] = False
                        closed_cost += COST / (2 * k); hold_sum += t / STEP; ncloses += 1
            live = lev * (np.sum(wl[held_w] * retw[-1, held_w]) - np.sum(ws[held_s] * retl[-1, held_s]))
            pnl = locked + live
            hold_sum += held_w.sum() + held_s.sum()         # survivors held full cycle (frac 1 each)
            hold = hold_sum / max(npos, 1)
            cclose = closed_cost
            eL = {liq[gi] for li, gi in enumerate(bk["W"]) if held_w[li]}
            eS = {liq[gi] for li, gi in enumerate(bk["L"]) if held_s[li]}
            DIAG.append((ncloses, npos, hold))

        elif variant == "randexit":
            counts = kw["counts"]; ci = len(ret)             # match exitdecay's close-count this cycle
            nclose = counts[ci] if ci < len(counts) else 0
            npos = len(bk["W"]) + len(bk["L"]); locked = 0.0; hold_sum = 0.0
            held_w = np.ones(len(bk["W"]), bool); held_s = np.ones(len(bk["L"]), bool)
            slots = [("w", i) for i in range(len(bk["W"]))] + [("s", i) for i in range(len(bk["L"]))]
            pick = rng.choice(len(slots), size=min(nclose, len(slots)), replace=False) if nclose else []
            for sidx in pick:
                side, li = slots[sidx]; t = int(rng.integers(1, STEP + 1))
                if side == "w":
                    locked += lev * wl[li] * retw[t, li]; held_w[li] = False
                else:
                    locked += -lev * ws[li] * retl[t, li]; held_s[li] = False
                hold_sum += t / STEP
            live = lev * (np.sum(wl[held_w] * retw[-1, held_w]) - np.sum(ws[held_s] * retl[-1, held_s]))
            pnl = locked + live
            hold_sum += held_w.sum() + held_s.sum()
            hold = hold_sum / max(npos, 1); cclose = len(pick) * COST / (2 * k)
            eL = {liq[gi] for li, gi in enumerate(bk["W"]) if held_w[li]}
            eS = {liq[gi] for li, gi in enumerate(bk["L"]) if held_s[li]}

        fund = FUND_BASE * (STEP_DAYS / 365) * lev * hold
        r = pnl - entry_turn * COST - cclose - fund
        ret.append(r); idx.append(master[b]); pl, ps = eL, eS
        eq *= (1 + r); peak = max(peak, eq)
    return pd.Series(ret, index=pd.DatetimeIndex(idx))


def stats(w):
    if len(w) < 3 or w.std() == 0:
        return 0, 0, 0
    eq = (1 + w).cumprod()
    return (eq.iloc[-1] - 1) * 100, (w.mean() / w.std()) * np.sqrt(WEEKS_YR), (eq / eq.cummax() - 1).min() * 100


def sub(w):
    return stats(w[w.index < SPLIT])[1], stats(w[w.index >= SPLIT])[1]


variants = [
    ("base", dict()),
    ("regime4h", dict()),
    ("ddstop", dict(X=0.10)), ("ddstop", dict(X=0.15)), ("ddstop", dict(X=0.20)),
    ("rehedge", dict(X=0.15)), ("rehedge", dict(X=0.25)),
    ("exitdecay", dict()),
    ("combo", dict()),
    ("combo", dict(pct=0.45)), ("combo", dict(pct=0.40)), ("combo", dict(pct=0.35)),
]

print(f"Top-{TOPLIQ} liquid; {len(rebals)} biweekly cycles; 4h path-aware "
      f"({START}..{str(master[-1].date())}); net cost+funding 20%/yr\n")
print(f"  {'variant':16s} {'total%':>8s} {'Sharpe':>7s} {'bull':>6s} {'bear':>6s} {'maxDD%':>7s}")
series = {}
for name, kw in variants:
    w = run(name, **kw)
    label = name + (f"_{int(kw['X']*100)}" if "X" in kw else "") + (f"_p{int(kw['pct']*100)}" if "pct" in kw else "")
    series[label] = w
    t, s, m = stats(w); bu, be = sub(w)
    print(f"  {label:16s} {t:>8.0f} {s:>7.2f} {bu:>6.2f} {be:>6.2f} {m:>7.0f}")

# exitdecay turnover + the matched RANDOM-EXIT null (signal vs just de-grossing)
run("exitdecay"); ed_counts = [d[0] for d in DIAG]
avg_cl = np.mean(ed_counts); avg_npos = np.mean([d[1] for d in DIAG]); avg_hold = np.mean([d[2] for d in DIAG])
rnd = [stats(run("randexit", counts=ed_counts, seed=sd))[1] for sd in range(30)]
ed_s = stats(series["exitdecay"])[1]
print(f"\n=== exitdecay turnover & RANDOM-EXIT null ===")
print(f"  exitdecay closes {avg_cl:.1f}/{avg_npos:.0f} names per cycle, avg holding {avg_hold*100:.0f}% of cycle")
print(f"  exitdecay Sharpe {ed_s:.2f}  vs  random-exit (same counts) {np.mean(rnd):.2f} "
      f"+/- {np.std(rnd):.2f}  -> {((ed_s-np.mean(rnd))/(np.std(rnd) or 1)):+.1f} sd")

print(f"\n=== ROLLING consistency: Sharpe per ~6-month block ===")
top = ["base", "regime4h", "exitdecay", "combo"]
hdr = "  " + f"{'block':>11s}" + "".join(f"{n:>11s}" for n in top)
print(hdr)
blocks = sorted(set(series["base"].groupby(pd.Grouper(freq="2QS")).groups.keys()))
posct = {n: 0 for n in top}; totct = 0
for ts in blocks:
    cells = []
    ok = False
    for n in top:
        w = series[n]; wb = w[(w.index >= ts) & (w.index < ts + pd.offsets.QuarterBegin(2))]
        if len(wb) < 3:
            cells.append(f"{'-':>11s}"); continue
        ok = True; s = stats(wb)[1]; posct[n] += s > 0; cells.append(f"{s:>11.2f}")
    if ok:
        totct += 1; print(f"  {str(ts.date()):>11s}" + "".join(cells))
print("  " + f"{'positive':>11s}" + "".join(f"{posct[n]:>8d}/{totct:>2d}" for n in top))
