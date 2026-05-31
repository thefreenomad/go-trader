"""Decision matrix: f(regime, volatility_tier, asset, direction) -> strategy.

Long and short treated as SEPARATE books (an asset can be long-good/short-bad).
Full cycle 2023-01 -> now (bull+bear). 4h. 1x unlevered, no funding (caveats stand).

Anti-overfit discipline:
  * pool by volatility TIER (not per-asset) so a cell rule must hold across assets
  * TRAIN on first 60% of time, build the matrix; TEST picks on unseen last 40%
  * report the train->test Sharpe gap (the overfit tax)

Regime gates (single-regime entry): trending_up / trending_down / ranging.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "backtest"); sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd
from data_fetcher import load_cached_data
from run_backtest import load_registry
from backtester import Backtester

START = "2023-01-01"; TF = "4h"; CAPITAL = 10000.0; PLATFORM = "hyperliquid"
TRAIN_FRAC = 0.6; MIN_TRADES = 8
ASSETS = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","DOGE/USDT","ADA/USDT","AVAX/USDT","LINK/USDT"]
REGIMES = ["trending_up","trending_down","ranging"]
DIRECTIONS = ["long","short"]
reg = load_registry("spot")
STRATS = [s for s in reg.list_strategies() if s != "hold"]


def ann(n=365*6):
    return np.sqrt(n)


def vol_tier_map():
    """Annualized realized vol per asset from 1d returns over the window."""
    vols = {}
    for sym in ASSETS:
        d = load_cached_data(sym, "1d", start_date=START)
        d = d[d.index >= pd.Timestamp(START)]
        r = d["close"].pct_change().dropna()
        vols[sym.split("/")[0]] = r.std() * np.sqrt(365) * 100
    s = pd.Series(vols).sort_values()
    # terciles -> low / mid / high
    tiers = {}
    q1, q2 = s.quantile(1/3), s.quantile(2/3)
    for a, v in s.items():
        tiers[a] = "low" if v <= q1 else ("mid" if v <= q2 else "high")
    return s, tiers


def dir_columns(sigdf, direction):
    """Direction-isolated state machine: only hold `direction`; opposite signal -> flat."""
    sig = sigdf["signal"].fillna(0).astype(int)
    want = 1 if direction == "long" else -1
    in_pos = (sig.replace(0, np.nan).ffill().fillna(0).astype(int) == want)
    prev = in_pos.shift(1).fillna(False)
    oa = pd.Series("none", index=sigdf.index, dtype=object)
    oa[in_pos & ~prev] = direction
    cf = pd.Series(0.0, index=sigdf.index)
    cf[~in_pos & prev] = 1.0
    out = sigdf.copy(); out["open_action"] = oa; out["close_fraction"] = cf
    return out


def run_one(coldf, name, regime, direction):
    bt = Backtester(initial_capital=CAPITAL, platform=PLATFORM,
                    open_strategy={"name": name, "params": {}}, strategy_type="perps",
                    regime_enabled=True, allowed_regimes=[regime])
    res = bt.run(coldf.copy(), strategy_name=name, symbol="x", timeframe=TF)
    return res["sharpe_ratio"], res["total_return_pct"], res["total_trades"]


volser, TIER = vol_tier_map()
rows = []
for sym in ASSETS:
    asset = sym.split("/")[0]
    df0 = load_cached_data(sym, TF, start_date=START)
    df0 = df0[df0.index >= pd.Timestamp(START)]
    if len(df0) < 500:
        continue
    cut = int(len(df0) * TRAIN_FRAC)
    splits = {"train": df0.iloc[:cut], "test": df0.iloc[cut:]}
    for name in STRATS:
        try:
            sig_full = reg.apply_strategy(name, df0, reg.STRATEGY_REGISTRY[name]["default_params"])
        except Exception:
            continue
        for direction in DIRECTIONS:
            col_full = dir_columns(sig_full, direction)
            for split, sdf in splits.items():
                cdf = col_full.loc[sdf.index]
                for regime in REGIMES:
                    try:
                        shp, ret, trd = run_one(cdf, name, regime, direction)
                    except Exception:
                        continue
                    rows.append(dict(asset=asset, tier=TIER[asset], strategy=name,
                                     direction=direction, regime=regime, split=split,
                                     sharpe=shp, ret=ret, trades=trd))

df = pd.DataFrame(rows)
df.to_csv("research/results/decision_matrix_results.csv", index=False)

print("\n=== VOLATILITY PROFILE (annualized, from daily returns 2023->now) ===")
for a, v in volser.items():
    print(f"  {a:5s} {v:6.0f}%   tier={TIER[a]}")

# Build matrix on TRAIN: per (direction, regime, tier) pick strategy w/ best mean Sharpe across tier assets
tr = df[(df.split == "train") & (df.trades >= MIN_TRADES)]
te = df[df.split == "test"]


def cell_pick(direction, regime, tier):
    sub = tr[(tr.direction == direction) & (tr.regime == regime) & (tr.tier == tier)]
    if sub.empty:
        return None
    g = sub.groupby("strategy").agg(shp=("sharpe", "mean"), trd=("trades", "mean"),
                                    n=("asset", "nunique"))
    g = g[g.n >= 2]  # rule must appear on >=2 assets in the tier
    if g.empty:
        return None
    return g.sort_values("shp", ascending=False).iloc[0:1].assign(strategy=g.sort_values("shp", ascending=False).index[0]).iloc[0]


def test_score(direction, regime, tier, strat):
    sub = te[(te.direction == direction) & (te.regime == regime) & (te.tier == tier)
             & (te.strategy == strat)]
    if sub.empty:
        return np.nan, np.nan
    return sub["sharpe"].mean(), sub["ret"].mean()


for direction in DIRECTIONS:
    print(f"\n{'='*78}\n  {direction.upper()} BOOK — matrix f(regime, vol_tier) -> strategy "
          f"[train Sharpe -> TEST Sharpe]\n{'='*78}")
    print(f"  {'regime':16s} {'tier':5s} {'strategy':18s} {'trainShp':>8s} {'testShp':>8s} {'testRet%':>8s}")
    gaps = []
    for regime in REGIMES:
        for tier in ["low", "mid", "high"]:
            pick = cell_pick(direction, regime, tier)
            if pick is None:
                print(f"  {regime:16s} {tier:5s} {'(no qualifying)':18s}")
                continue
            strat = pick["strategy"]
            tshp, tret = test_score(direction, regime, tier, strat)
            gaps.append((pick["shp"], tshp))
            print(f"  {regime:16s} {tier:5s} {strat:18s} {pick['shp']:>8.2f} "
                  f"{tshp:>8.2f} {tret:>8.1f}")
    g = [(a, b) for a, b in gaps if not np.isnan(b)]
    if g:
        tr_m = np.mean([a for a, _ in g]); te_m = np.mean([b for _, b in g])
        print(f"  --> avg train Sharpe {tr_m:.2f}  vs  avg TEST Sharpe {te_m:.2f}  "
              f"(overfit tax {tr_m-te_m:+.2f}); cells holding up (testShp>0): "
              f"{sum(b>0 for _,b in g)}/{len(g)}")

# Per-asset long vs short bias (avg sharpe across regimes, TRAIN, qualifying)
print(f"\n{'='*60}\n  PER-ASSET LONG vs SHORT BIAS (avg best-strategy Sharpe, train)\n{'='*60}")
print(f"  {'asset':5s} {'tier':5s} {'long_shp':>8s} {'short_shp':>9s}   profile")
for sym in ASSETS:
    a = sym.split("/")[0]
    def side_best(direction):
        sub = tr[(tr.asset == a) & (tr.direction == direction)]
        if sub.empty: return np.nan
        return sub.groupby("regime")["sharpe"].max().mean()
    ls, ss = side_best("long"), side_best("short")
    prof = "long-biased" if (ls or -9) > (ss or -9) + 0.15 else \
           ("short-biased" if (ss or -9) > (ls or -9) + 0.15 else "symmetric")
    print(f"  {a:5s} {TIER[a]:5s} {ls:>8.2f} {ss:>9.2f}   {prof}")
