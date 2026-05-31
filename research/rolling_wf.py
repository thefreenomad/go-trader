"""Rolling walk-forward — models the live operating loop:
  every 7 days, look back 30 days, rank the universe, deploy the top-K configs
  for the next 7 days, realize, roll forward. Re-decided weekly, like live.

Universe: 8 assets x 28 strategies x {long, short}, direction-isolated, 4h.
Per-config returns: vectorized, fee-aware, 1-bar fill lag, 1x (matches backtester
sizing). No funding. Weekly cross-config rebalance cost on turnover.

The decision function (which asset/strategy/direction to run next week) is LEARNED
weekly from trailing performance — regime/vol/direction selection emerge implicitly.

Anti-fooling controls:
  * RANDOM-K weekly selection (avg of many seeds) — does skill beat luck?
  * BTC buy&hold and equal-weight-all-long-hold — does it beat just holding?
"""
import sys, warnings, collections
warnings.filterwarnings("ignore")
sys.path.insert(0, "backtest"); sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd
from data_fetcher import load_cached_data
from run_backtest import load_registry

START = "2023-01-01"; TF = "4h"; BARS_DAY = 6
TRAIN = 30 * BARS_DAY; STEP = 7 * BARS_DAY
COST = 0.00085; REBAL_RT = 2 * COST
PPY = 365 * BARS_DAY; WEEKS_YR = 365 / 7
KS = [1, 3, 5, 10]; MIN_ACTIVE = 8; N_SEEDS = 40
ASSETS = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","DOGE/USDT","ADA/USDT","AVAX/USDT","LINK/USDT"]
DIRECTIONS = ["long","short"]
reg = load_registry("spot")
STRATS = [s for s in reg.list_strategies() if s != "hold"]


def config_returns(close, sig, direction):
    s = sig.fillna(0).astype(int)
    want = 1 if direction == "long" else -1
    held = (s.replace(0, np.nan).ffill().fillna(0).astype(int) == want)
    pos = (held.astype(float) * want).shift(1).fillna(0.0)
    bar = close.pct_change().fillna(0.0)
    turn = pos.diff().abs().fillna(abs(pos.iloc[0]))
    return pos * bar - turn * COST, pos.abs()


master = None; streams = {}; activ = {}; close_by_asset = {}
for sym in ASSETS:
    a = sym.split("/")[0]
    df = load_cached_data(sym, TF, start_date=START)
    df = df[df.index >= pd.Timestamp(START)]
    if len(df) < TRAIN + STEP:
        continue
    close_by_asset[a] = df["close"]
    master = df.index if master is None else master.union(df.index)
    for name in STRATS:
        try:
            sig = reg.apply_strategy(name, df, reg.STRATEGY_REGISTRY[name]["default_params"])["signal"]
        except Exception:
            continue
        for d in DIRECTIONS:
            net, ac = config_returns(df["close"], sig, d)
            streams[(a, name, d)] = net; activ[(a, name, d)] = ac

R = pd.DataFrame(streams).reindex(master)
A = pd.DataFrame(activ).reindex(master)
cfg = list(R.columns)
Rmat = R.values; Amat = A.values
C = len(cfg)
rebals = list(range(TRAIN, len(master) - STEP, STEP))
NR = len(rebals)

# Precompute matrices: (NR x C)
sharpe = np.full((NR, C), np.nan); retsc = np.full((NR, C), np.nan)
fwd = np.full((NR, C), np.nan); elig = np.zeros((NR, C), bool)
for i, b in enumerate(rebals):
    tw = Rmat[b - TRAIN:b, :]
    valid = ~np.isnan(tw).any(axis=0)
    act = np.nansum(Amat[b - TRAIN:b, :], axis=0)
    elig[i] = valid & (act >= MIN_ACTIVE)
    mean = np.nanmean(tw, axis=0); sd = np.nanstd(tw, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sharpe[i] = np.where(sd > 0, mean / sd, 0.0) * np.sqrt(PPY)
    retsc[i] = np.nanprod(1 + np.nan_to_num(tw), axis=0) - 1
    fw = Rmat[b:b + STEP, :]
    fwd[i] = np.nanprod(1 + np.nan_to_num(fw), axis=0) - 1

print(f"Universe: {C} configs ({len(close_by_asset)} assets x {len(STRATS)} strat x 2 dir)")
print(f"Master grid: {len(master)} bars; {NR} weekly rebalances; train {TRAIN}b/step {STEP}b\n")


def roll(K, metric="sharpe", seed=None):
    eq = 1.0; curve = []; prev = set(); picks = []
    rng = np.random.default_rng(seed) if seed is not None else None
    score = {"sharpe": sharpe, "ret": retsc}.get(metric)
    for i in range(NR):
        e = np.where(elig[i])[0]
        if e.size == 0:
            curve.append(eq); continue
        if rng is not None:
            sel = rng.choice(e, size=min(K, e.size), replace=False)
        else:
            sc = score[i][e]
            sel = e[np.argsort(sc)[::-1][:K]]
        wk = float(np.nanmean(fwd[i][sel]))
        ns = set(sel.tolist()); turn = len(ns ^ prev) / max(2 * K, 1)
        wk -= turn * REBAL_RT; prev = ns
        eq *= (1 + wk); curve.append(eq); picks.append([cfg[j] for j in sel])
    return pd.Series(curve, index=[master[b] for b in rebals]), picks


def stats(curve):
    wr = curve.pct_change().dropna()
    tot = (curve.iloc[-1] - 1) * 100
    shp = (wr.mean() / wr.std()) * np.sqrt(WEEKS_YR) if wr.std() > 0 else 0.0
    mdd = (curve / curve.cummax() - 1).min() * 100
    yrs = (curve.index[-1] - curve.index[0]).days / 365
    cagr = (curve.iloc[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 and curve.iloc[-1] > 0 else -100
    return tot, cagr, shp, mdd


span0, span1 = master[rebals[0]], master[rebals[-1] + STEP - 1]
btc = close_by_asset["BTC"].reindex(master).loc[span0:span1]
btc_tot = (btc.iloc[-1] / btc.iloc[0] - 1) * 100
ewc = pd.concat([(close_by_asset[a].reindex(master).loc[span0:span1] /
                  close_by_asset[a].reindex(master).loc[span0:span1].iloc[0])
                 for a in close_by_asset], axis=1).mean(axis=1)
ew_tot = (ewc.iloc[-1] - 1) * 100

print("=== BENCHMARKS (hold, same span) ===")
print(f"  BTC buy&hold          total {btc_tot:+8.1f}%")
print(f"  Equal-weight 8 (long) total {ew_tot:+8.1f}%   span {span0.date()} -> {span1.date()}\n")

print("=== ROLLING 'DEPLOY THE WINNERS' vs RANDOM weekly selection ===")
print(f"  {'K':>3s} {'metric':7s} | {'total%':>9s} {'CAGR%':>7s} {'Sharpe':>7s} {'maxDD%':>7s} | "
      f"{'RANDOM tot%':>11s} {'rand Shp':>8s} {'beats rand?':>11s}")
rand_cache = {}
for K in KS:
    rc = [stats(roll(K, seed=1000 + s)[0]) for s in range(N_SEEDS)]
    rand_cache[K] = (np.mean([x[0] for x in rc]), np.mean([x[2] for x in rc]),
                     np.std([x[0] for x in rc]))
for K in KS:
    rt, rs, rstd = rand_cache[K]
    for metric in ["sharpe", "ret"]:
        tot, cagr, shp, mdd = stats(roll(K, metric)[0])
        edge = (tot - rt) / rstd if rstd > 0 else 0          # z-score vs random
        verdict = f"+{edge:.1f}sd" if edge > 0 else f"{edge:.1f}sd"
        print(f"  {K:>3d} {metric:7s} | {tot:>8.1f}% {cagr:>6.1f}% {shp:>7.2f} {mdd:>7.1f} | "
              f"{rt:>10.1f}% {rs:>8.2f} {verdict:>11s}")

_, picks = roll(5, "sharpe")
flat = [p for wk in picks for p in wk]; n = len(flat)
dc = collections.Counter(p[2] for p in flat); ac = collections.Counter(p[0] for p in flat)
scn = collections.Counter(p[1] for p in flat)
print(f"\n=== LEARNED POLICY (K=5, trailing-Sharpe), {n} deployments ===")
print("  direction:  " + ", ".join(f"{k} {100*v/n:.0f}%" for k, v in dc.most_common()))
print("  top assets: " + ", ".join(f"{k} {100*v/n:.0f}%" for k, v in ac.most_common(5)))
print("  top strats: " + ", ".join(f"{k} {100*v/n:.0f}%" for k, v in scn.most_common(6)))
