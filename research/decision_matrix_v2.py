"""Decision matrix v2 — both confounds removed.

  1. BETA-RELATIVE scoring: score = strategy_Sharpe - regime_beta_Sharpe, where
     regime-beta = hold `direction` whenever `regime` is active (fee-aware).
     Cancels the bull/bear drift that fooled v1.
  2. CROSS-ASSET holdout (leave-one-asset-out): pick best strategy per
     (regime,direction) on 7 assets, score its alpha on the held-out 8th, over the
     FULL 2023->now period. Tests transfer across assets, not bull-vs-bear time.

1x unlevered, no funding (caveats stand). 4h. HL fees.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "backtest"); sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd
from data_fetcher import load_cached_data
from run_backtest import load_registry
from backtester import Backtester
from regime import ensure_regime_columns

START = "2023-01-01"; TF = "4h"; CAPITAL = 10000.0; PLATFORM = "hyperliquid"
COST = 0.00085          # HL taker 0.035% + slippage 0.05%, per side
PPY = 365 * 6
MIN_TRADES = 15         # over the full ~3yr window
ASSETS = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","DOGE/USDT","ADA/USDT","AVAX/USDT","LINK/USDT"]
REGIMES = ["trending_up","trending_down","ranging"]
DIRECTIONS = ["long","short"]
TIER = {"BTC":"low","ETH":"low","XRP":"low","LINK":"mid","DOGE":"mid","SOL":"high","AVAX":"high","ADA":"high"}
reg = load_registry("spot")
STRATS = [s for s in reg.list_strategies() if s != "hold"]


def regime_beta(close, labels, regime, direction):
    """Fee-aware Sharpe & total-return of mechanically holding `direction`
    on every bar `regime` is active (decision lagged 1 bar)."""
    sgn = 1.0 if direction == "long" else -1.0
    pos = (labels.shift(1) == regime).astype(float) * sgn
    bar = close.pct_change().fillna(0.0)
    turns = pos.diff().abs().fillna(abs(pos.iloc[0]))
    net = pos * bar - turns * COST
    sd = net.std()
    shp = (net.mean() / sd) * np.sqrt(PPY) if sd > 0 else 0.0
    tot = ((1 + net).prod() - 1) * 100
    return shp, tot, int((pos.diff().abs() > 0).sum())


def dir_columns(sigdf, direction):
    sig = sigdf["signal"].fillna(0).astype(int)
    want = 1 if direction == "long" else -1
    in_pos = (sig.replace(0, np.nan).ffill().fillna(0).astype(int) == want)
    prev = in_pos.shift(1).fillna(False)
    oa = pd.Series("none", index=sigdf.index, dtype=object)
    oa[in_pos & ~prev] = direction
    cf = pd.Series(0.0, index=sigdf.index); cf[~in_pos & prev] = 1.0
    out = sigdf.copy(); out["open_action"] = oa; out["close_fraction"] = cf
    return out


rows = []
for sym in ASSETS:
    asset = sym.split("/")[0]
    df0 = load_cached_data(sym, TF, start_date=START)
    df0 = df0[df0.index >= pd.Timestamp(START)].copy()
    if len(df0) < 500:
        continue
    labeled = ensure_regime_columns(df0.copy(), period=14, adx_threshold=20.0)
    labels = labeled["regime"]
    close = df0["close"]
    beta = {(rg, d): regime_beta(close, labels, rg, d) for rg in REGIMES for d in DIRECTIONS}
    for name in STRATS:
        try:
            sig = reg.apply_strategy(name, df0, reg.STRATEGY_REGISTRY[name]["default_params"])
        except Exception:
            continue
        for direction in DIRECTIONS:
            cdf = dir_columns(sig, direction)
            for rg in REGIMES:
                try:
                    bt = Backtester(initial_capital=CAPITAL, platform=PLATFORM,
                                    open_strategy={"name": name, "params": {}},
                                    strategy_type="perps", regime_enabled=True,
                                    allowed_regimes=[rg])
                    r = bt.run(cdf.copy(), strategy_name=name, symbol="x", timeframe=TF)
                except Exception:
                    continue
                bshp, btot, _ = beta[(rg, direction)]
                rows.append(dict(asset=asset, tier=TIER[asset], strategy=name,
                                 direction=direction, regime=rg,
                                 strat_shp=r["sharpe_ratio"], strat_ret=r["total_return_pct"],
                                 trades=r["total_trades"], beta_shp=bshp, beta_ret=btot,
                                 alpha_shp=r["sharpe_ratio"] - bshp,
                                 alpha_ret=r["total_return_pct"] - btot))

df = pd.DataFrame(rows)
df.to_csv("research/results/decision_matrix_v2_results.csv", index=False)

print("\n=== REGIME-BETA BENCHMARKS (Sharpe of mechanically holding direction in-regime, fee-aware) ===")
b = (df.groupby(["regime","direction"]).agg(beta_shp=("beta_shp","mean"),
                                            beta_ret=("beta_ret","mean")))
with pd.option_context("display.float_format", lambda x: f"{x:.2f}"):
    print(b.to_string())

elig = df[df.trades >= MIN_TRADES]
print(f"\n(eligible strategy-cells with >= {MIN_TRADES} trades: {len(elig)} / {len(df)})")

print("\n" + "="*86)
print("  CROSS-ASSET VALIDATION — leave-one-asset-out, full period (bull+bear)")
print("  pick best-alpha strategy on 7 assets -> measure its ALPHA on the held-out 8th")
print("="*86)
print(f"  {'direction':9s} {'regime':14s} {'most-picked strat':20s} {'mean held-out alphaShp':>22s} {'#pos':>6s}")
summary = []
for direction in DIRECTIONS:
    for rg in REGIMES:
        cell = elig[(elig.direction == direction) & (elig.regime == rg)]
        if cell.empty:
            continue
        picks, oos = [], []
        for held in cell.asset.unique():
            others = cell[cell.asset != held]
            g = others.groupby("strategy").agg(a=("alpha_shp","mean"), n=("asset","nunique"))
            g = g[g.n >= 3]
            if g.empty:
                continue
            best = g["a"].idxmax()
            ho = cell[(cell.asset == held) & (cell.strategy == best)]
            if not ho.empty:
                picks.append(best); oos.append(ho["alpha_shp"].iloc[0])
        if not oos:
            continue
        mp = pd.Series(picks).value_counts().index[0]
        npos = sum(x > 0 for x in oos)
        summary.append((direction, rg, mp, np.mean(oos), npos, len(oos)))
        print(f"  {direction:9s} {rg:14s} {mp:20s} {np.mean(oos):>22.2f} {npos:>3d}/{len(oos)}")

print("\n=== VERDICT ===")
pos_cells = [s for s in summary if s[3] > 0]
print(f"  (regime,direction) cells with POSITIVE mean held-out alpha over regime-beta: "
      f"{len(pos_cells)} / {len(summary)}")
for d, rg, mp, a, npos, n in sorted(summary, key=lambda x: -x[3]):
    tag = "  <-- transfers" if a > 0 and npos >= max(3, n*0.6) else ""
    print(f"    {d:6s} {rg:14s} {mp:18s} alpha {a:+.2f}  ({npos}/{n} assets positive){tag}")

# Does conditioning on vol tier help? mean alpha by tier for the cross-asset picks
print("\n=== Does VOL TIER matter? mean held-out-style alpha by (direction, tier), best strat per cell ===")
for direction in DIRECTIONS:
    line = []
    for tier in ["low","mid","high"]:
        sub = elig[(elig.direction == direction) & (elig.tier == tier)]
        if sub.empty:
            line.append(f"{tier}: n/a"); continue
        best = sub.groupby("strategy")["alpha_shp"].mean().max()
        line.append(f"{tier}: {best:+.2f}")
    print(f"  {direction:6s}  " + "   ".join(line))
